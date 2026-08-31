#!/usr/bin/env python3
"""Measure work-conserving generation concurrency on a live vLLM server."""

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[int(round((len(values) - 1) * q))])


def parse_metric(text, name):
    values = []
    prefix = "vllm:%s" % name
    for line in text.splitlines():
        if line.startswith(prefix) and not line.startswith("#"):
            try:
                values.append(float(line.rsplit(" ", 1)[-1]))
            except ValueError:
                pass
    return sum(values)


def fetch_metrics(url):
    try:
        text = urllib.request.urlopen(url, timeout=2).read().decode()
    except Exception:
        return {}
    return {
        "running": parse_metric(text, "num_requests_running"),
        "waiting": parse_metric(text, "num_requests_waiting"),
        "generation_tokens": parse_metric(text, "generation_tokens_total"),
        "prompt_tokens": parse_metric(text, "prompt_tokens_total"),
        "success": parse_metric(text, "request_success_total"),
    }


def gpu_sample():
    try:
        text = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=3)
        rows = []
        for line in text.splitlines():
            util, memory = [float(value.strip()) for value in line.split(",")]
            rows.append((util, memory))
        return rows
    except Exception:
        return []


def tokenizer_count(tokenizer, messages, template_kwargs):
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        **template_kwargs)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "numel"):
        return int(encoded.numel())
    if encoded and isinstance(encoded[0], (list, tuple)):
        return len(encoded[0])
    return len(encoded)


async def run_load(args, client, requests, concurrency, duration, warmup=False):
    start_metrics = await asyncio.to_thread(fetch_metrics, args.metrics_url)
    begin = time.monotonic()
    deadline = begin + duration
    submitted = 0
    completed = []
    errors = []
    active = set()
    samples = []

    async def one(request):
        started = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=request["messages"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=request["max_tokens"],
                extra_body=request["extra_body"] or None,
            )
            ended = time.monotonic()
            choice = response.choices[0]
            usage = response.usage
            return {
                "problem_id": request["problem_id"],
                "latency_s": ended - started,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "finish_reason": getattr(choice, "finish_reason", None),
            }
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {"error": "%s: %s" % (
                type(error).__name__, str(error)[:300])}

    def submit():
        nonlocal submitted
        request = requests[submitted % len(requests)]
        active.add(asyncio.create_task(one(request)))
        submitted += 1

    for _ in range(concurrency):
        submit()

    next_sample = begin
    while active and time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_sample:
            metric = await asyncio.to_thread(fetch_metrics, args.metrics_url)
            gpu = await asyncio.to_thread(gpu_sample)
            samples.append({"metrics": metric, "gpu": gpu})
            next_sample = now + 1.0
        timeout = max(0.0, min(0.25, deadline - time.monotonic()))
        done, _ = await asyncio.wait(
            active, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            active.remove(task)
            row = task.result()
            if row.get("error"):
                errors.append(row["error"])
            else:
                completed.append(row)
            if time.monotonic() < deadline:
                submit()

    canceled = len(active)
    for task in active:
        task.cancel()
    await asyncio.gather(*active, return_exceptions=True)
    await asyncio.sleep(2)
    end_metrics = await asyncio.to_thread(fetch_metrics, args.metrics_url)
    elapsed = time.monotonic() - begin

    generated = max(
        0.0,
        end_metrics.get("generation_tokens", 0)
        - start_metrics.get("generation_tokens", 0))
    prompt = max(
        0.0,
        end_metrics.get("prompt_tokens", 0)
        - start_metrics.get("prompt_tokens", 0))
    latencies = [row["latency_s"] for row in completed]
    completion_tokens = [row["completion_tokens"] for row in completed]
    completed_by_problem = {}
    for row in completed:
        key = str(row["problem_id"])
        completed_by_problem[key] = completed_by_problem.get(key, 0) + 1
    waiting = [row["metrics"].get("waiting", 0) for row in samples]
    running = [row["metrics"].get("running", 0) for row in samples]
    gpu_utils = [
        util for row in samples for util, _ in row["gpu"]
    ]
    gpu_memory = [
        memory for row in samples for _, memory in row["gpu"]
    ]
    result = {
        "warmup": warmup,
        "concurrency": concurrency,
        "duration_target_s": duration,
        "elapsed_s": elapsed,
        "submitted": submitted,
        "completed": len(completed),
        "tasks_with_completion": len(completed_by_problem),
        "completed_by_problem": completed_by_problem,
        "canceled": canceled,
        "errors": len(errors),
        "error_examples": errors[:3],
        "generation_tokens": generated,
        "generation_tokens_per_s": generated / max(elapsed, 1e-9),
        "prompt_tokens": prompt,
        "completed_per_s": len(completed) / max(elapsed, 1e-9),
        "latency_median_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_s": percentile(latencies, 0.95),
        "completion_tokens_median": (
            statistics.median(completion_tokens) if completion_tokens else 0.0),
        "waiting_mean": statistics.mean(waiting) if waiting else 0.0,
        "waiting_max": max(waiting) if waiting else 0.0,
        "running_mean": statistics.mean(running) if running else 0.0,
        "running_max": max(running) if running else 0.0,
        "gpu_util_mean": statistics.mean(gpu_utils) if gpu_utils else 0.0,
        "gpu_util_p10": percentile(gpu_utils, 0.10),
        "gpu_memory_max_mib": max(gpu_memory) if gpu_memory else 0.0,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


async def main_async(args):
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer
    from kernelbench.dataset import construct_kernelbench_dataset
    from kernelbench.prompt_constructor_toml import get_custom_prompt
    from repair_loop import chat_extra_body

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)
    template_kwargs = {}
    thinking = os.environ.get("ENABLE_THINKING", "").strip().lower()
    if thinking in ("0", "1", "true", "false", "yes", "no"):
        template_kwargs["enable_thinking"] = thinking in ("1", "true", "yes")
    strength = os.environ.get("REASONING_STRENGTH", "").strip()
    if strength:
        template_kwargs["reasoning_strength"] = strength

    dataset = construct_kernelbench_dataset(args.level)
    problem_ids = list(dataset.get_problem_ids())[:args.tasks]
    extra_body = chat_extra_body(args.top_k)
    requests = []
    for pid in problem_ids:
        problem = dataset.get_problem_by_id(pid)
        prompt = get_custom_prompt(
            args.prompt_tier, ref_arch_src=problem.code,
            backend="cutile", option="one_shot", precision="fp32")
        messages = [{"role": "user", "content": prompt}]
        input_tokens = tokenizer_count(tokenizer, messages, template_kwargs)
        maximum = min(
            args.max_tokens,
            args.native_context - input_tokens - args.safety_margin)
        requests.append({
            "problem_id": int(pid),
            "messages": messages,
            "input_tokens": input_tokens,
            "max_tokens": maximum,
            "extra_body": extra_body,
        })

    client = AsyncOpenAI(
        api_key="local-no-auth", base_url=args.base_url,
        timeout=args.duration + 120, max_retries=0)
    output = {
        "tag": args.tag,
        "level": args.level,
        "tasks": problem_ids,
        "duration_s": args.duration,
        "native_context": args.native_context,
        "max_tokens": args.max_tokens,
        "results": [],
    }
    if args.warmup > 0:
        output["warmup"] = await run_load(
            args, client, requests, min(32, args.concurrencies[0]),
            args.warmup, warmup=True)
        await asyncio.sleep(5)
    for concurrency in args.concurrencies:
        output["results"].append(await run_load(
            args, client, requests, concurrency, args.duration))
        await asyncio.sleep(5)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--native-context", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=131072)
    ap.add_argument("--safety-margin", type=int, default=1024)
    ap.add_argument("--level", type=int, default=60)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--prompt-tier", default="cutile_concepts")
    ap.add_argument("--concurrencies", default="32,64,96,128")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--model", default="Qwen3-Coder-Next")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.concurrencies = [
        int(value) for value in args.concurrencies.split(",")]
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
