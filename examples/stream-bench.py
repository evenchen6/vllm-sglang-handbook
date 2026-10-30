#!/usr/bin/env python3
"""Small closed-loop OpenAI-compatible SSE experiment; Python standard library only.

Network chunks are NOT tokens. Record text-chunk gaps, and estimate TPOT only
when the server supplies completion_tokens. Prefix-cache hits per request come
from usage.prompt_tokens_details.cached_tokens when the server reports them.
See docs/t4-qwen35-lab.md.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def sse_data(response):
    """Yield complete SSE data fields, supporting CRLF and multi-line events."""
    parts = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(" ") else value)
    if parts:
        yield "\n".join(parts)


def run_request(base_url, model, messages, max_tokens=256, thinking=False,
                timeout=180, api_key="EMPTY", show_output=False):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 "Authorization": "Bearer " + api_key},
    )
    started = time.perf_counter()
    text_times, content_parts, reasoning_parts = [], [], []
    first_answer = None
    usage, finish_reason, done = {}, None, False
    try:
        with urlopen(request, timeout=timeout) as response:
            for data in sse_data(response):
                if data.strip() == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError("server error: " + str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue  # usage-only events are not generated text
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not isinstance(content, str) or not isinstance(reasoning, str):
                    raise ValueError("expected string text deltas")
                if not content and not reasoning:
                    continue  # role-only / empty deltas do not define TTFT
                now = time.perf_counter()
                text_times.append(now)
                if content and first_answer is None:
                    first_answer = now
                content_parts.append(content)
                reasoning_parts.append(reasoning)
                if show_output:
                    print(reasoning + content, end="", flush=True)
        if not done:
            raise ValueError("stream ended without [DONE]; treat as incomplete")
        if not text_times:
            raise ValueError("stream completed without any text delta")
        tokens = usage.get("completion_tokens")
        tokens = tokens if isinstance(tokens, int) and tokens >= 0 else None
        tpot = ((text_times[-1] - text_times[0]) * 1000 / (tokens - 1)
                if tokens is not None and tokens > 1 else None)
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        return {
            "ok": True,
            "ttft_text_ms": (text_times[0] - started) * 1000,
            "first_answer_ms": ((first_answer - started) * 1000
                                if first_answer is not None else None),
            "e2e_ms": (time.perf_counter() - started) * 1000,
            "estimated_tpot_ms": tpot,
            "text_chunk_gaps_ms": [(b - a) * 1000 for a, b in zip(text_times, text_times[1:])],
            "text_chunks": len(text_times),
            "completion_tokens": tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_tokens_details": details,
            "cached_tokens": cached if isinstance(cached, int) and cached >= 0 else None,
            "finish_reason": finish_reason,
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
        }
    except (HTTPException, OSError, ValueError, KeyError, TypeError) as error:
        detail = str(error)
        if isinstance(error, HTTPError):
            detail += ": " + error.read(2048).decode("utf-8", errors="replace")
        return {"ok": False, "error": detail, "e2e_ms": (time.perf_counter() - started) * 1000}


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180, help="socket I/O timeout in seconds, not a total request deadline")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--prefix-lines", type=int, default=32)
    parser.add_argument("--prompt", default="请用中文解释 KV cache，并举一个具体例子。")
    parser.add_argument("--show-output", action="store_true")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) < 1 or args.warmup < 0:
        parser.error("requests, concurrency and max-tokens must be positive; warmup >= 0")
    if args.timeout <= 0 or args.prefix_lines < 1:
        parser.error("timeout and prefix-lines must be positive")
    if args.show_output and (args.requests != 1 or args.concurrency != 1):
        parser.error("--show-output requires --requests 1 --concurrency 1")
    system = "你是严谨的中文技术助手。请直接回答，说明结论的适用条件。"
    if args.shared_prefix:
        system += "\n以下是所有请求共享的背景资料，请据此回答：\n" + "\n".join(
            f"资料{i:03d}：推理服务需要同时观察排队、首包延迟、输出速度和显存；前缀缓存复用已有计算。"
            for i in range(args.prefix_lines))

    def one(index):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"请求编号{index}。{args.prompt}"}]
        return run_request(args.base_url, args.model, messages, args.max_tokens,
                           args.thinking, args.timeout,
                           os.environ.get("OPENAI_API_KEY", "EMPTY"), args.show_output and index >= 0)

    for index in range(args.warmup):
        result = one(-index - 1)
        if not result["ok"]:
            print("Warmup failed: " + result["error"], file=sys.stderr)
            return 1
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(one, range(args.requests)))
    elapsed = time.perf_counter() - started
    good = [r for r in results if r["ok"]]
    all_tokens_known = bool(good) and all(r["completion_tokens"] is not None for r in good)
    total_tokens = sum(r["completion_tokens"] for r in good) if all_tokens_known else None
    # Servers report this only when asked (vLLM --enable-prompt-tokens-details, SGLang --enable-cache-report).
    cached = [r["cached_tokens"] for r in good if r["cached_tokens"] is not None]
    summary = {
        "requests": len(results), "successful": len(good), "failed": len(results) - len(good),
        "finish_length": sum(r.get("finish_reason") == "length" for r in good),
        "wall_seconds": elapsed, "successful_req_per_s": len(good) / elapsed,
        "output_tok_per_s": total_tokens / elapsed if total_tokens is not None else None,
        "ttft_text_ms": distribution([r["ttft_text_ms"] for r in good]),
        "first_answer_ms": distribution([r["first_answer_ms"] for r in good]),
        "e2e_ms": distribution([r["e2e_ms"] for r in good]),
        "estimated_tpot_ms": distribution([r["estimated_tpot_ms"] for r in good]),
        "text_chunk_gap_ms": distribution([v for r in good for v in r["text_chunk_gaps_ms"]]),
        "cached_tokens": distribution(cached),
        "cache_hit_requests": sum(v > 0 for v in cached) if cached else None,
    }
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "measurement": "closed-loop; worker-start timing; network text chunks != tokens; TPOT estimate from server usage",
        "summary": summary, "results": results,
    }
    if args.show_output:
        print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for result in results:
        if not result["ok"]:
            print(result["error"], file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Saved: " + str(args.output))
    return 0 if len(good) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
