"""Hard-label SFT ученика Qwen3 с LoRA и маской токенов промпта."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from experiment import BASE, SEED, Example, evaluate, load_model, read_jsonl, save_json


def load_teacher_rows(path: Path, samples: int) -> tuple[list[tuple[list[int], list[int]]], int]:
    if samples < 1:
        raise ValueError("параметр `samples` должен быть положительным")
    rows = []
    count = 0
    for count, row in enumerate(read_jsonl(path), 1):
        if row["valid"] and row["answer_ids"]:
            prompt, answer = row["prompt_ids"], row["answer_ids"]
            rows.append((prompt + answer, [-100] * len(prompt) + answer))
        if count == samples:
            break
    if not rows:
        raise ValueError("в answers.jsonl нет валидных ответов учителя")
    return rows, count


def collate(
    rows: Sequence[tuple[list[int], list[int]]], pad: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids, _ in rows)
    ids, masks, labels = [], [], []
    for tokens, target in rows:
        diff = width - len(tokens)
        ids.append(tokens + [pad] * diff)
        masks.append([True] * len(tokens) + [False] * diff)
        labels.append(target + [-100] * diff)
    return tuple(torch.tensor(rows, device=device) for rows in (ids, masks, labels))


def student_run(
    args: Any, folder: Path, dev: Sequence[Example], intents: list[str], types: list[str], system: str,
) -> None:
    for name in ("epochs", "lr", "rank", "micro_batch", "accumulation"):
        if getattr(args, name) <= 0:
            raise ValueError(f"параметр `{name.replace('_', '-')}` должен быть положительным")
    if args.epochs % 1:
        raise ValueError("параметр `epochs` должен быть целым")
    rows, read_count = load_teacher_rows(
        args.artifacts / args.teacher_run / "checkpoints" / "teacher_answers" / "answers.jsonl",
        args.samples,
    )
    model, tokenizer = load_model(args.models / BASE)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.train()
    causal_lm = model.get_base_model()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, fused=True)
    effective_batch = args.micro_batch * args.accumulation
    steps = math.ceil(len(rows) / effective_batch) * int(args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(steps * 0.03), steps)
    print(f"ответы учителя: {len(rows)}/{read_count} валидны, шагов: {steps}", flush=True)
    start = time.monotonic()
    rng = random.Random(SEED)
    trace = []
    step = 0
    for epoch in range(int(args.epochs)):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for offset in range(0, len(order), effective_batch):
            update = order[offset:offset + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            summed_loss = 0.0
            num_tokens = sum(len(rows[i][1]) - rows[i][1].count(-100) for i in update)
            for start_batch in range(0, len(update), args.micro_batch):
                batch = update[start_batch:start_batch + args.micro_batch]
                ids, mask, labels = collate([rows[i] for i in batch], tokenizer.pad_token_id, model.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    states = causal_lm.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                    targets = labels[:, 1:]
                    predicted = targets != -100
                    # считаем logits только для ответа: это та же CE с ignore_index, но без logits промпта.
                    logits = causal_lm.lm_head(states[:, :-1][predicted])
                    loss = F.cross_entropy(logits.float(), targets[predicted], reduction="sum") / num_tokens
                loss.backward()
                summed_loss += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            trace.append({"step": step, "epoch": epoch + 1, "loss": summed_loss,
                          "lr": scheduler.get_last_lr()[0]})
            if step == 1 or step % 10 == 0 or step == steps:
                print(f"обучение: {step}/{steps}, loss={summed_loss:.4f}, "
                      f"{time.monotonic()-start:.0f} с", flush=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, parameters
    save_json(folder / "metrics" / "training.json", {"teacher_rows": read_count, "valid_rows": len(rows),
                                                     "steps": trace})

    checkpoint = folder / "checkpoints" / "student_sft"
    checkpoint.mkdir()
    student = model.merge_and_unload().eval()
    student.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    print(f"чекпоинт сохранён: {checkpoint}", flush=True)
    evaluate(student, tokenizer, system, dev, args.batch, folder / "metrics", intents, types)
