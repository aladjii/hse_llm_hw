import argparse
import asyncio
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, quantiles
from time import perf_counter

import httpx


@dataclass
class Request:
    index: int
    schedule_delay_s: float
    latency_s: float | None = None
    ttft_s: float | None = None
    output_tokens: int | None = None
    error: str = ""


async def measure(
    client: httpx.AsyncClient, payload: bytes, index: int, scheduled: float
) -> Request:
    start = perf_counter()
    result = Request(index, max(0.0, start - scheduled))
    done = False
    try:
        async with asyncio.timeout(120), client.stream(
            "POST", "/v1/chat/completions", content=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    result.error = "stream_error"
                    return result
                if chunk.get("usage"):
                    result.output_tokens = chunk["usage"]["completion_tokens"]
                for choice in chunk.get("choices", []):
                    if choice.get("delta", {}).get("content"):
                        latency = perf_counter() - start
                        if result.ttft_s is None:
                            result.ttft_s = latency
                        result.latency_s = latency
    except httpx.HTTPStatusError as exc:
        result.error = f"http_{exc.response.status_code}"
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        result.error = type(exc).__name__
    if not result.error and (not done or result.latency_s is None):
        result.error = "incomplete_stream"
    return result


def p90(values: list[float]) -> float | None:
    if len(values) < 2:
        return values[0] if values else None
    return quantiles(values, n=10, method="inclusive")[8]


async def benchmark(rpm: int, duration: int | None, output: Path) -> None:
    payloads = Path("data/requests.jsonl").read_bytes().splitlines()
    planned = len(payloads) if duration is None else (duration * rpm + 59) // 60
    port = int(os.environ.get("PORT", "8000"))
    output.mkdir(parents=True, exist_ok=False)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        headers={"Content-Type": "application/json"},
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
        timeout=120,
    ) as client:
        gate = asyncio.Semaphore(8)

        async def warmup(index: int) -> Request:
            async with gate:
                return await measure(client, payloads[index], index, perf_counter())

        warm = await asyncio.gather(*(warmup(i) for i in range(40)))
        errors = sorted({req.error for req in warm if req.error})
        if errors:
            raise RuntimeError(f"прогрев завершился с ошибками: {', '.join(errors)}")

        # открытая нагрузка: старт нового запроса не зависит от предыдущего ответа.
        start = perf_counter()
        pending = []
        for i in range(planned):
            when = start + i * 60 / rpm
            await asyncio.sleep(max(0.0, when - perf_counter()))
            index = i % len(payloads)
            pending.append(asyncio.create_task(measure(client, payloads[index], index, when)))
        window = planned * 60 / rpm if duration is None else duration
        await asyncio.sleep(max(0.0, start + window - perf_counter()))
        records = await asyncio.gather(*pending)
        elapsed = perf_counter() - start

    with (output / "requests.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, asdict(records[0]))
        writer.writeheader()
        writer.writerows(asdict(req) for req in records)
    ok = [req for req in records if not req.error]
    latency = [req.latency_s for req in ok if req.latency_s is not None]
    ttft = [req.ttft_s for req in ok if req.ttft_s is not None]
    tokens = [req.output_tokens for req in ok if req.output_tokens is not None]
    summary = {
        "rpm": rpm,
        "window_s": window,
        "elapsed_s": elapsed,
        "sent": len(records),
        "ok": len(ok),
        "errors": len(records) - len(ok),
        "mean_s": mean(latency) if latency else None,
        "p90_s": p90(latency),
        "p90_ttft_s": p90(ttft),
        "mean_output_tokens": mean(tokens) if tokens else None,
        "p90_schedule_delay_s": p90([req.schedule_delay_s for req in records]),
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpm", type=int, required=True)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rpm < 1:
        parser.error("`--rpm` должен быть положительным")
    if args.duration is not None and args.duration < 1:
        parser.error("`--duration` должен быть положительным")
    asyncio.run(benchmark(args.rpm, args.duration, args.output))


if __name__ == "__main__":
    main()
