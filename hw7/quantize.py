"""AWQ, блочный GPTQ и статический SmoothQuant без внешнего компрессора."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from experiment import BASE, SEED, Example, dataset, evaluate, load_model, save_json


def calibration_ids(args: Any, tokenizer: Any, train: Sequence[Example]) -> torch.Tensor:
    if args.samples < 1:
        raise ValueError("параметр `samples` должен быть положительным")
    if args.cal_length < 2:
        raise ValueError("параметр `cal-length` должен быть не меньше 2")
    if args.source == "massive":
        texts = [row.text for row in train]
    else:
        table = dataset(args.archive, "wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet")
        texts = [line.strip() for line in table["text"].to_pylist() if line.strip()]
    random.Random(SEED).shuffle(texts)
    size = args.samples * args.cal_length
    flat: list[int] = []
    for offset in range(0, len(texts), 128):
        chunk = tokenizer(texts[offset:offset + 128], add_special_tokens=False).input_ids
        for ids in chunk:
            flat.extend(ids)
            flat.append(tokenizer.eos_token_id)
            if len(flat) >= size:
                return torch.tensor(flat[:size], dtype=torch.long).view(args.samples, args.cal_length)
    raise ValueError("для параметра `samples` недостаточно калибровочных токенов")


class ActivationStats:
    def __init__(self, method: str, group_size: int) -> None:
        self.method = method
        self.group_size = group_size
        self.count = 0
        self.value: torch.Tensor | None = None
        self.examples: list[torch.Tensor] = []

    def update(self, _module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        x = args[0].detach().flatten(0, -2)
        self.count += len(x)
        if self.method == "awq":
            if len(self.examples) < 16:
                stride = max(1, len(x) // 8)
                self.examples.append(x[::stride][:8].contiguous())
            value = x.float().abs().sum(dim=0)
        elif self.method == "gptq":
            groups = x.float().reshape(len(x), -1, self.group_size).transpose(0, 1).contiguous()
            value = torch.bmm(groups.transpose(1, 2), groups)
        else:
            value = x.abs().amax(dim=0).float()
            self.value = value if self.value is None else torch.maximum(self.value, value)
            return
        if self.value is None:
            self.value = value
        else:
            self.value.add_(value)


def group_quant(weight: torch.Tensor, bits: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    groups = weight.reshape(weight.shape[0], -1, size)
    top = 2 ** (bits - 1) - 1
    scale = groups.abs().amax(dim=-1).clamp_min(1e-8) / top
    q = (groups / scale.unsqueeze(-1)).round_().clamp_(-top, top).to(torch.int8)
    return q.view_as(weight), scale


def restore_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.reshape(q.shape[0], scale.shape[1], -1).float() * scale.unsqueeze(-1)).reshape_as(q)


def awq(
    weight: torch.Tensor, stats: ActivationStats, size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    assert stats.value is not None
    x = torch.cat(stats.examples)
    reference = x @ weight.T.to(x.dtype)
    mean = (stats.value / stats.count).clamp_min(1e-6)
    best_error = float("inf")
    best = None
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = mean.pow(ratio)
        s /= (s.min() * s.max()).sqrt()
        q, scale = group_quant(weight * s, 4, size)
        restored = (restore_weight(q, scale) / s).to(x.dtype)
        error = ((x @ restored.T).float() - reference.float()).square().mean().item()
        if error < best_error:
            best_error = error
            best = q, scale, s, ratio
    if best is None:
        raise RuntimeError("не удалось подобрать масштаб AWQ")
    return best


def gptq(
    weight: torch.Tensor, stats: ActivationStats, size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert stats.value is not None
    hessian = stats.value / stats.count
    diagonal = hessian.diagonal(dim1=-2, dim2=-1)
    diagonal.add_(diagonal.mean(dim=-1, keepdim=True).clamp_min(1e-6) * 0.01)
    factor = torch.linalg.cholesky(hessian)
    upper = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)

    groups = weight.reshape(weight.shape[0], -1, size).transpose(0, 1).contiguous()
    scale = groups.abs().amax(dim=-1).clamp_min(1e-8) / 7
    codes = torch.empty_like(groups, dtype=torch.int8)
    # внедиагональные блоки не хранятся, чтобы размер статистики рос линейно.
    for column in range(size):
        v = groups[..., column]
        q = (v / scale).round_().clamp_(-7, 7)
        codes[..., column] = q.to(torch.int8)
        if column + 1 < size:
            error = (v - q * scale) / upper[:, column, column, None]
            groups[..., column + 1:] -= error.unsqueeze(-1) * upper[:, None, column, column + 1:]
    return codes.transpose(0, 1).reshape_as(weight), scale.T.contiguous()


def smooth_quant(
    weight: torch.Tensor, stats: ActivationStats, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert stats.value is not None
    weight_max = weight.abs().amax(dim=0).clamp_min(1e-6)
    act_max = stats.value.clamp_min(1e-6)
    s = act_max.pow(alpha) / weight_max.pow(1 - alpha)
    q, scale = group_quant(weight * s, 8, weight.shape[-1])
    return q, scale, s, (act_max / s).max() / 127


def activate_smooth(layer: nn.Linear, s: torch.Tensor, act_scale: torch.Tensor) -> None:
    def quantize_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x = args[0]
        qx = (x.float() / s / act_scale).round_().clamp_(-127, 127) * act_scale
        return (qx.to(x.dtype), *args[1:])

    layer.register_forward_pre_hook(quantize_input)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    unsigned = (q + 8).to(torch.uint8)
    return unsigned[..., ::2] | (unsigned[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).to(torch.int8) - 8


@torch.inference_mode()
def load_quantized(path: Path, device: str = "cuda") -> tuple[Any, Any]:
    spec = json.loads((path / "compression.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    with safe_open(path / "dense.safetensors", framework="pt") as src:
        model.load_state_dict({key: src.get_tensor(key) for key in src.keys()}, strict=False, assign=True)
    with safe_open(path / "quantized.safetensors", framework="pt") as src:
        for name in spec["layers"]:
            layer = model.get_submodule(name)
            q = src.get_tensor(name + ".q").to(device)
            if spec["bits"] == 4:
                q = unpack_int4(q)
            weight = restore_weight(q, src.get_tensor(name + ".scale").to(device))
            if spec["method"] == "awq":
                weight /= src.get_tensor(name + ".input_scale").to(device)
            if spec["method"] == "smoothquant":
                activate_smooth(layer, src.get_tensor(name + ".input_scale").to(device),
                                src.get_tensor(name + ".act_scale").to(device))
            layer.weight = nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)
    model.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def quant_run(
    args: Any, folder: Path, train: Sequence[Example], dev: Sequence[Example],
    intents: list[str], types: list[str], system: str,
) -> None:
    if args.method is None:
        raise ValueError("для квантизации нужен параметр `method`")
    if args.group_size < 2 or args.group_size % 2:
        raise ValueError("параметр `group-size` должен быть чётным и не меньше 2")
    if not 0 <= args.alpha <= 1:
        raise ValueError("параметр `alpha` должен быть в [0, 1]")
    model, tokenizer = load_model(args.models / BASE)
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, nn.Linear) and name.startswith("model.layers.")}
    if args.method != "smoothquant" and any(layer.in_features % args.group_size for layer in layers.values()):
        raise ValueError("параметр `group-size` должен делить входную размерность всех целевых слоёв")
    stats = {name: ActivationStats(args.method, args.group_size) for name in layers}
    hooks = [layer.register_forward_pre_hook(stats[name].update) for name, layer in layers.items()]
    calibration = calibration_ids(args, tokenizer, train).to(model.device)
    print(f"калибровка: {tuple(calibration.shape)}, линейных слоёв: {len(layers)}", flush=True)
    for batch in calibration.split(8):
        model.model(input_ids=batch, use_cache=False)
    for hook in hooks:
        hook.remove()

    checkpoint = folder / "checkpoints" / "quantized"
    checkpoint.mkdir()
    bits = 8 if args.method == "smoothquant" else 4
    extra: dict[str, float] = {}
    stored: dict[str, torch.Tensor] = {}
    for index, (name, layer) in enumerate(layers.items(), 1):
        weight = layer.weight.detach().float()
        current = stats.pop(name)
        if args.method == "awq":
            q, scale, s, ratio = awq(weight, current, args.group_size)
            extra[name] = ratio
            stored[name + ".input_scale"] = s.cpu()
            restored = restore_weight(q, scale) / s
        elif args.method == "gptq":
            q, scale = gptq(weight, current, args.group_size)
            restored = restore_weight(q, scale)
        else:
            q, scale, s, act_scale = smooth_quant(weight, current, args.alpha)
            stored[name + ".input_scale"] = s.cpu()
            stored[name + ".act_scale"] = act_scale.cpu()
            activate_smooth(layer, s, act_scale)
            restored = restore_weight(q, scale)
        layer.weight.copy_(restored)
        stored[name + ".q"] = (pack_int4(q) if bits == 4 else q).cpu().contiguous()
        stored[name + ".scale"] = scale.cpu().contiguous()
        if index % 36 == 0 or index == len(layers):
            print(f"квантизация: {index}/{len(layers)}", flush=True)

    save_file(stored, checkpoint / "quantized.safetensors")
    stored.clear()
    quant_names = {name + ".weight" for name in layers}
    save_file(
        {key: value.cpu().contiguous() for key, value in model.state_dict().items() if key not in quant_names},
        checkpoint / "dense.safetensors",
    )
    model.config.save_pretrained(checkpoint)
    model.generation_config.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    save_json(checkpoint / "compression.json", {
        "method": args.method, "bits": bits, "group_size": None if bits == 8 else args.group_size,
        "calibration": args.source, "samples": args.samples, "tokens_per_sample": args.cal_length,
        "alpha": args.alpha if bits == 8 else None, "layers": list(layers), "awq_ratios": extra,
    })
    print(f"чекпоинт сохранён: {checkpoint}", flush=True)
    evaluate(model, tokenizer, system, dev, args.batch, folder / "metrics", intents, types)
