import json
from collections import Counter
from pathlib import Path
from typing import Any

from prepare import canonical


def score_calls(
    calls: list[dict[str, Any]], target: list[dict[str, Any]], mode: str
) -> float:
    exact = Counter(map(canonical, calls)) == Counter(map(canonical, target))
    if mode == "exact":
        return float(exact)
    if mode != "shaped":
        raise ValueError("`mode` должен принимать значение 'exact' или 'shaped'")
    if not target:
        return float(not calls)
    if not calls:
        return 0.0
    predicted = Counter(call["name"] for call in calls)
    expected = Counter(call["name"] for call in target)
    name_f1 = 2 * sum((predicted & expected).values()) / (len(calls) + len(target))
    full_matches = sum(
        (Counter(map(canonical, calls)) & Counter(map(canonical, target))).values()
    )
    call_f1 = 2 * full_matches / (len(calls) + len(target))
    return 0.6 * exact + 0.2 * name_f1 + 0.2 * call_f1


def make_reward(
    tokenizer: Any,
    mode: str,
    max_completion_length: int,
    sample_path: Path | None = None,
) -> Any:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    parser = Qwen3CoderToolParser(tokenizer)
    last_sampled_step = None

    def tool_reward(
        completions: list[Any],
        completion_ids: list[list[int]],
        tools_json: list[str],
        target_json: list[str],
        **kwargs: Any,
    ) -> list[float]:
        nonlocal last_sampled_step
        # Сохраняем специальные токены, иначе разборщик потеряет разметку вызовов.
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
        rewards, valid_flags, exact_flags, samples = [], [], [], []
        for text, ids, tools_text, target_text in zip(
            texts, completion_ids, tools_json, target_json, strict=True
        ):
            tools, target = json.loads(tools_text), json.loads(target_text)
            payload = {
                "model": "Qwen/Qwen3.5-2B",
                "messages": [{"role": "user", "content": "score"}],
            }
            if tools:
                payload["tools"] = tools
            request = ChatCompletionRequest.model_validate(payload)
            parser.tools = request.tools or []
            malformed = any(
                text.count(left) != text.count(right)
                for left, right in [
                    ("<tool_call>", "</tool_call>"),
                    ("<function=", "</function>"),
                    ("<parameter=", "</parameter>"),
                ]
            )
            content = text
            for token in tokenizer.all_special_tokens:
                content = content.replace(token, "")
            truncated = (
                len(ids) >= max_completion_length and ids[-1] != tokenizer.eos_token_id
            )
            try:
                parsed = parser.extract_tool_calls(text, request)
                calls = [
                    {
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments),
                    }
                    for call in parsed.tool_calls
                ]
                invalid_markup = text.count("<tool_call>") != len(calls) or (
                    not calls
                    and any(tag in text for tag in ("<tool", "<function", "<parameter"))
                )
                valid = bool(content.strip()) and not (
                    malformed or truncated or invalid_markup
                )
            except (ValueError, TypeError, KeyError):
                calls, valid = [], False
            rewards.append(score_calls(calls, target, mode) if valid else 0.0)
            valid_flags.append(float(valid))
            exact_flags.append(score_calls(calls, target, "exact") if valid else 0.0)
            samples.append(
                {
                    "response": text,
                    "calls": calls,
                    "target": target,
                    "valid": valid,
                    "exact_reward": exact_flags[-1],
                    "shaped_reward": score_calls(calls, target, "shaped")
                    if valid
                    else 0.0,
                }
            )
        step = getattr(kwargs.get("trainer_state"), "global_step", 0)
        if sample_path is not None and step % 50 == 0 and step != last_sampled_step:
            with sample_path.open("a") as output:
                for index, sample in enumerate(samples[:4]):
                    output.write(
                        json.dumps(
                            {
                                "step": step,
                                "mode": mode,
                                "id": kwargs["id"][index],
                                "prompt": kwargs["prompts"][index],
                                **sample,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            last_sampled_step = step
        if "log_metric" in kwargs:
            kwargs["log_metric"](
                "tool_format_valid", sum(valid_flags) / len(valid_flags)
            )
            kwargs["log_metric"]("tool_exact", sum(exact_flags) / len(exact_flags))
        return rewards

    return tool_reward
