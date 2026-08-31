#!/usr/bin/env python3
"""Asynchronous, verifier-guided Hybrid search with 60 seconds per problem.

The process runs in the evaluation container on the verifier GPUs and talks to
the TP2 vLLM server over HTTP.  Each problem owns an independent deadline.
Completed problems are durable; interrupted attempts are retained for audit and
the problem receives a fresh full budget after restart.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "verify"))

from repair_loop import chat_extra_body, choice_text  # noqa: E402
from sequential_prompts import build_adaptive_prompt  # noqa: E402
from sequential_search import atomic_json, select_problem_ids  # noqa: E402
from worker import INCONCLUSIVE_STAGES, VerifierPool  # noqa: E402


TERMINAL_STATUSES = frozenset(("solved", "timeout", "stagnated"))
PROFILE_CONFIGS = {
    "60s": {
        "GLE": {
            "global_slots": 256, "active_problems": 32,
            "per_task_slots": 8, "initial_candidates": 8,
            "min_early_exit_s": 0, "min_evidence": 2,
            "stagnant_waves": 1, "no_code_evidence": 2,
            "launch_guard_s": 0,
        },
        "GL": {
            "global_slots": 96, "active_problems": 8,
            "per_task_slots": 12, "initial_candidates": 8,
            "min_early_exit_s": 0, "min_evidence": 2,
            "stagnant_waves": 1, "no_code_evidence": 2,
            "launch_guard_s": 0,
        },
        "Q": {
            "global_slots": 256, "active_problems": 32,
            "per_task_slots": 8, "initial_candidates": 8,
            "min_early_exit_s": 0, "min_evidence": 2,
            "stagnant_waves": 1, "no_code_evidence": 2,
            "launch_guard_s": 0,
        },
        "base": {
            "global_slots": 256, "active_problems": 32,
            "per_task_slots": 8, "initial_candidates": 8,
            "min_early_exit_s": 0, "min_evidence": 2,
            "stagnant_waves": 1, "no_code_evidence": 2,
            "launch_guard_s": 0,
        },
        "G4t": {
            "global_slots": 64, "active_problems": 8,
            "per_task_slots": 8, "initial_candidates": 8,
            "min_early_exit_s": 0, "min_evidence": 2,
            "stagnant_waves": 1, "no_code_evidence": 2,
            "launch_guard_s": 0,
        },
        "Q38": {
            "global_slots": 32, "active_problems": 4,
            "per_task_slots": 8, "initial_candidates": 8,
            "min_early_exit_s": 0, "min_evidence": 2,
            "stagnant_waves": 1, "no_code_evidence": 2,
            "launch_guard_s": 0,
        },
    },
    "600s": {
        "GLE": {
            "global_slots": 256, "active_problems": 128,
            "per_task_slots": 2, "initial_candidates": 2,
            "min_early_exit_s": 120, "min_evidence": 6,
            "stagnant_waves": 2, "no_code_evidence": 4,
            "launch_guard_s": 90,
        },
        "GL": {
            "global_slots": 96, "active_problems": 48,
            "per_task_slots": 2, "initial_candidates": 2,
            "min_early_exit_s": 120, "min_evidence": 6,
            "stagnant_waves": 2, "no_code_evidence": 4,
            "launch_guard_s": 120,
        },
        "Q": {
            "global_slots": 256, "active_problems": 128,
            "per_task_slots": 2, "initial_candidates": 2,
            "min_early_exit_s": 120, "min_evidence": 6,
            "stagnant_waves": 2, "no_code_evidence": 4,
            "launch_guard_s": 30,
        },
        "base": {
            "global_slots": 256, "active_problems": 128,
            "per_task_slots": 2, "initial_candidates": 2,
            "min_early_exit_s": 120, "min_evidence": 6,
            "stagnant_waves": 2, "no_code_evidence": 4,
            "launch_guard_s": 45,
        },
        "G4t": {
            "global_slots": 64, "active_problems": 32,
            "per_task_slots": 2, "initial_candidates": 2,
            "min_early_exit_s": 120, "min_evidence": 6,
            "stagnant_waves": 2, "no_code_evidence": 4,
            "launch_guard_s": 120,
        },
        "Q38": {
            "global_slots": 32, "active_problems": 16,
            "per_task_slots": 2, "initial_candidates": 2,
            "min_early_exit_s": 120, "min_evidence": 6,
            "stagnant_waves": 2, "no_code_evidence": 4,
            "launch_guard_s": 180,
        },
    },
}


def profile_for(tag, profile="60s"):
    if profile not in PROFILE_CONFIGS:
        raise ValueError("unknown Hybrid search profile: %s" % profile)
    if tag not in PROFILE_CONFIGS[profile]:
        raise ValueError("unknown Hybrid model tag: %s" % tag)
    return dict(PROFILE_CONFIGS[profile][tag])


def wall_timestamp(value=None):
    value = time.time() if value is None else value
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def code_digest(code):
    normalized = "\n".join(line.rstrip() for line in (code or "").strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def result_rank(result):
    """Frozen frontier ordering for the 60-second search protocol."""
    result = result or {}
    if result.get("passed"):
        return 5
    stage = result.get("stage") or "no_code"
    if stage == "purity":
        return 4 if result.get("impure_correct") else 2
    if stage == "exec" and result.get("rel_diff") is not None:
        return 3
    if stage in ("exec", "timeout", "compile"):
        return 1
    return 0


def result_label(result):
    rank = result_rank(result)
    return {
        0: "no_code",
        1: "compile/exec",
        2: "purity_wrong",
        3: "numeric_mismatch",
        4: "purity_correct",
        5: "pass",
    }[rank]


def frontier_improved(previous, current):
    """Whether current earns a fresh pair of repair opportunities."""
    old = (previous or {}).get("result") or {}
    new = (current or {}).get("result") or {}
    old_rank = result_rank(old)
    new_rank = result_rank(new)
    if new_rank != old_rank:
        return new_rank > old_rank
    if new_rank == 3:
        old_diff = old.get("rel_diff")
        new_diff = new.get("rel_diff")
        return (
            old_diff is not None and new_diff is not None
            and float(new_diff) <= 0.8 * float(old_diff)
        )
    if new_rank in (2, 4):
        old_ops = old.get("torch_ops_left")
        new_ops = new.get("torch_ops_left")
        return (
            old_ops is not None and new_ops is not None
            and int(new_ops) < int(old_ops)
        )
    return False


def frontier_sort_key(candidate):
    result = (candidate or {}).get("result") or {}
    rank = result_rank(result)
    if rank == 3:
        metric = -float(result.get("rel_diff", float("inf")))
    elif rank in (2, 4):
        metric = -float(result.get("torch_ops_left", float("inf")))
    else:
        metric = 0.0
    return rank, metric, int((candidate or {}).get("seq", -1))


def deadline_accepts(started, finished, deadline_s):
    return (
        started is not None
        and finished is not None
        and float(finished) - float(started) <= float(deadline_s)
    )


class AsyncDynamicChat:
    """Async OpenAI client with per-prompt native-context output budgeting."""

    def __init__(self, base_url, model, tokenizer_path, temperature, top_p,
                 top_k, desired_max_tokens, native_context, safety_margin,
                 request_timeout):
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer

        self.client = AsyncOpenAI(
            api_key="local-no-auth", base_url=base_url,
            timeout=request_timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.desired_max_tokens = int(desired_max_tokens)
        self.native_context = int(native_context)
        self.safety_margin = int(safety_margin)
        self.extra_body = chat_extra_body(top_k)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True)

    def estimate_input_tokens(self, messages):
        template_kwargs = {}
        thinking = os.environ.get("ENABLE_THINKING", "").strip().lower()
        if thinking in ("0", "1", "true", "false", "yes", "no"):
            template_kwargs["enable_thinking"] = thinking in (
                "1", "true", "yes")
        strength = os.environ.get("REASONING_STRENGTH", "").strip()
        if strength:
            template_kwargs["reasoning_strength"] = strength
        try:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                **template_kwargs)
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict):
                ids = ids.get("input_ids", ids)
            if hasattr(ids, "numel"):
                return int(ids.numel())
            if (isinstance(ids, (list, tuple)) and ids
                    and isinstance(ids[0], (list, tuple))):
                return len(ids[0])
            return len(ids)
        except Exception:
            text = "\n".join(str(row.get("content") or "") for row in messages)
            return len(self.tokenizer.encode(text)) + 64

    async def _request(self, messages, input_tokens, max_tokens):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return {
            "text": choice_text(choice),
            "input_tokens_estimate": input_tokens,
            "max_tokens_sent": max_tokens,
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "desired_max_tokens": self.desired_max_tokens,
            "native_context": self.native_context,
            "safety_margin": self.safety_margin,
            "hit_token_limit": getattr(choice, "finish_reason", None) == "length",
            "error": None,
        }

    async def complete(self, messages):
        input_tokens = self.estimate_input_tokens(messages)
        available = self.native_context - input_tokens - self.safety_margin
        if available <= 0:
            return {
                "text": "",
                "input_tokens_estimate": input_tokens,
                "max_tokens_sent": 0,
                "finish_reason": "context_exhausted",
                "prompt_tokens": None,
                "completion_tokens": None,
                "desired_max_tokens": self.desired_max_tokens,
                "native_context": self.native_context,
                "safety_margin": self.safety_margin,
                "hit_token_limit": True,
                "error": "prompt leaves no output tokens in native context",
            }
        max_tokens = min(self.desired_max_tokens, available)
        try:
            return await self._request(messages, input_tokens, max_tokens)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            match = re.search(
                r"prompt contains at least (\d+) input tokens", str(error))
            if match is None:
                match = re.search(
                    r"parameter=input_tokens,\s*value=(\d+)", str(error))
            if match:
                actual = int(match.group(1))
                retry_max = min(
                    self.desired_max_tokens,
                    self.native_context - actual - self.safety_margin)
                if 0 < retry_max < max_tokens:
                    try:
                        row = await self._request(
                            messages, actual, retry_max)
                        row["input_tokens_estimate"] = input_tokens
                        row["prompt_tokens_from_error"] = actual
                        return row
                    except asyncio.CancelledError:
                        raise
                    except Exception as retry_error:
                        error = retry_error
            return {
                "text": "",
                "input_tokens_estimate": input_tokens,
                "max_tokens_sent": max_tokens,
                "finish_reason": "error",
                "prompt_tokens": None,
                "completion_tokens": None,
                "desired_max_tokens": self.desired_max_tokens,
                "native_context": self.native_context,
                "safety_margin": self.safety_margin,
                "hit_token_limit": False,
                "error": "%s: %s" % (
                    type(error).__name__, str(error)[:1000]),
            }


class VerifierDispatcher:
    """Feed small batches to one persistent VerifierPool and stream results."""

    def __init__(self, event_queue, workers=4, gpus=2, timeout_s=45,
                 batch_size=4, batch_wait_s=0.02):
        self.event_queue = event_queue
        self.workers = workers
        self.gpus = gpus
        self.timeout_s = timeout_s
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_s
        self.queue = asyncio.Queue()
        self.pool = None
        self.runner = None
        self.loop = None
        self.stopping = False
        self.canceled = set()

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.pool = VerifierPool(
            workers=self.workers, gpus=self.gpus,
            num_correct_trials=2, timeout_s=self.timeout_s)
        ready = await self.loop.run_in_executor(
            None, self.pool.wait_ready, self.workers, 120.0)
        if not ready:
            self.pool.close()
            raise RuntimeError(
                "only %d/%d verifier workers became ready"
                % (len(self.pool._ready_pids), self.workers))
        self.runner = asyncio.create_task(self._run())

    async def submit(self, candidate_id, code, ref_src):
        await self.queue.put((candidate_id, code, ref_src))

    def cancel(self, candidate_ids):
        # CUDA work already dispatched cannot be interrupted safely, but queued
        # candidates from solved/expired tasks must not consume later windows.
        self.canceled.update(candidate_ids)

    def _verify_batch(self, batch):
        emitted = set()

        def emit(rec):
            key = rec.get("key")
            if key in emitted or rec.get("stage") in INCONCLUSIVE_STAGES:
                return
            emitted.add(key)
            event = {
                "type": "verified",
                "candidate_id": key,
                "result": dict(rec),
                "finished_mono": time.monotonic(),
                "finished_wall": time.time(),
            }
            self.loop.call_soon_threadsafe(
                self.event_queue.put_nowait, event)

        results = self.pool.verify_batch(batch, on_result=emit)
        for key, rec in results.items():
            if key in emitted:
                continue
            emitted.add(key)
            event = {
                "type": "verified",
                "candidate_id": key,
                "result": dict(rec),
                "finished_mono": time.monotonic(),
                "finished_wall": time.time(),
            }
            self.loop.call_soon_threadsafe(
                self.event_queue.put_nowait, event)

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            if first is None:
                break
            if first[0] in self.canceled:
                continue
            batch = [first]
            stop_after = False
            limit = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                remaining = limit - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    stop_after = True
                    break
                if item[0] in self.canceled:
                    continue
                batch.append(item)
            await loop.run_in_executor(None, self._verify_batch, batch)
            if stop_after:
                break

    async def close(self):
        if self.stopping:
            return
        self.stopping = True
        await self.queue.put(None)
        if self.runner is not None:
            await self.runner
        if self.pool is not None:
            self.pool.close()


class HybridScheduler:
    def __init__(self, args, dataset, problem_ids, chat=None,
                 verifier=None, monotonic=None, wall_clock=None):
        self.args = args
        self.dataset = dataset
        self.problem_ids = list(problem_ids)
        self.run_dir = Path(args.run_dir)
        self.state_path = self.run_dir / "state.json"
        self.frozen_dir = self.run_dir / "frozen"
        self.event_queue = asyncio.Queue()
        self.monotonic = monotonic or time.monotonic
        self.wall_clock = wall_clock or time.time
        self.chat = chat
        self.verifier = verifier
        self.state = None
        self.pending = deque()
        self.active = {}
        self.runtimes = {}
        self.candidates = {}
        self.generation_tasks = {}
        self.active_order = deque()
        self.backpressured = False
        self._extract_best_code = None

    def _new_state(self):
        tasks = {}
        for pid in self.problem_ids:
            problem = self.dataset.get_problem_by_id(pid)
            tasks[str(pid)] = {
                "problem_id": int(pid),
                "problem": problem.name,
                "status": "pending",
                "attempts": 0,
                "interrupted_attempts": 0,
                "terminal_reason": None,
                "elapsed_s": None,
                "solution": None,
            }
        return {
            "version": 1,
            "protocol": "hybrid_search_%s" % self.args.profile,
            "tag": self.args.tag,
            "level": self.args.level,
            "created_at": wall_timestamp(),
            "updated_at": wall_timestamp(),
            "config": {
                "profile": self.args.profile,
                "deadline_s": self.args.deadline,
                "global_slots": self.args.global_slots,
                "active_problems": self.args.active_problems,
                "per_task_slots": self.args.per_task_slots,
                "initial_candidates": self.args.initial_candidates,
                "min_early_exit_s": self.args.min_early_exit_s,
                "min_evidence": self.args.min_evidence,
                "stagnant_waves": self.args.stagnant_waves,
                "no_code_evidence": self.args.no_code_evidence,
                "launch_guard_s": self.args.launch_guard_s,
                "backpressure_high": self.args.backpressure_high,
                "backpressure_low": self.args.backpressure_low,
                "verify_workers": self.args.verify_workers,
                "verify_gpus": self.args.verify_gpus,
                "verify_timeout": self.args.verify_timeout,
                "native_context": self.args.native_context,
                "max_tokens": self.args.max_tokens,
            },
            "tasks": tasks,
        }

    def load_or_initialize(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frozen_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text())
            if (self.state.get("tag") != self.args.tag
                    or int(self.state.get("level")) != self.args.level):
                raise SystemExit("existing Hybrid state belongs to another run")
            expected_config = {
                "profile": self.args.profile,
                "deadline_s": self.args.deadline,
                "global_slots": self.args.global_slots,
                "active_problems": self.args.active_problems,
                "per_task_slots": self.args.per_task_slots,
                "initial_candidates": self.args.initial_candidates,
                "min_early_exit_s": self.args.min_early_exit_s,
                "min_evidence": self.args.min_evidence,
                "stagnant_waves": self.args.stagnant_waves,
                "no_code_evidence": self.args.no_code_evidence,
                "launch_guard_s": self.args.launch_guard_s,
                "backpressure_high": self.args.backpressure_high,
                "backpressure_low": self.args.backpressure_low,
                "verify_workers": self.args.verify_workers,
                "verify_gpus": self.args.verify_gpus,
                "verify_timeout": self.args.verify_timeout,
                "native_context": self.args.native_context,
                "max_tokens": self.args.max_tokens,
            }
            existing_config = self.state.get("config") or {}
            changed = sorted(
                key for key, value in expected_config.items()
                if existing_config.get(key) != value)
            if changed:
                raise SystemExit(
                    "existing Hybrid protocol differs for: %s"
                    % ", ".join(changed))
            existing = set(int(pid) for pid in self.state["tasks"])
            if existing != set(self.problem_ids):
                raise SystemExit(
                    "existing Hybrid state problem set differs from request")
            for record in self.state["tasks"].values():
                if record.get("status") == "active":
                    record["status"] = "pending"
                    record["interrupted_attempts"] = (
                        int(record.get("interrupted_attempts", 0)) + 1)
                    record["terminal_reason"] = "interrupted_restart"
            self._save_state()
        else:
            self.state = self._new_state()
            self._save_state()
        self.pending = deque(
            int(pid) for pid in self.problem_ids
            if self.state["tasks"][str(pid)].get("status")
            not in TERMINAL_STATUSES)

    def _save_state(self):
        self.state["updated_at"] = wall_timestamp()
        counts = Counter(
            row.get("status", "pending")
            for row in self.state.get("tasks", {}).values())
        self.state["counts"] = dict(counts)
        atomic_json(self.state_path, self.state)

    def _problem_dir(self, pid):
        return self.run_dir / "problems" / ("problem_%04d" % pid)

    def _attempt_dir(self, pid, attempt):
        return self._problem_dir(pid) / ("attempt_%03d" % attempt)

    def _task_snapshot(self, runtime):
        now = self.monotonic()
        started = runtime.get("started_mono")
        return {
            "problem_id": runtime["pid"],
            "problem": runtime["problem"],
            "attempt": runtime["attempt"],
            "status": runtime["status"],
            "started_wall": runtime.get("started_wall"),
            "deadline_wall": runtime.get("deadline_wall"),
            "elapsed_s": (
                round(now - started, 6) if started is not None else None),
            "initial_done": sum(
                int(row["done"]) for row in runtime["initial_slots"].values()),
            "generation_inflight": sorted(runtime["gen_inflight"]),
            "verification_pending": sorted(runtime["verify_pending"]),
            "frontier_id": runtime.get("frontier_id"),
            "frontier_stage": (
                result_label(self.candidates[runtime["frontier_id"]]["result"])
                if runtime.get("frontier_id") in self.candidates else None),
            "repair_cycle": runtime.get("repair_cycle"),
            "effective_conclusions": runtime.get("effective_conclusions", 0),
            "no_code_conclusions": runtime.get("no_code_conclusions", 0),
            "stagnant_waves": runtime.get("stagnant_waves", 0),
            "terminal_reason": runtime.get("terminal_reason"),
        }

    def _save_task(self, runtime):
        atomic_json(runtime["dir"] / "task.json", self._task_snapshot(runtime))

    def _candidate_prefix(self, candidate):
        return candidate["dir"] / ("%06d" % candidate["seq"])

    def _save_candidate(self, candidate):
        serial = dict(candidate)
        serial["dir"] = str(serial["dir"])
        serial.pop("prompt", None)
        serial.pop("response", None)
        atomic_json(
            Path(str(self._candidate_prefix(candidate)) + "_candidate.json"),
            serial)

    def _write_text(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "")

    def _activate_one(self, pid):
        from kernelbench.prompt_constructor_toml import get_custom_prompt

        record = self.state["tasks"][str(pid)]
        record["attempts"] = int(record.get("attempts", 0)) + 1
        attempt = record["attempts"]
        problem = self.dataset.get_problem_by_id(pid)
        base_prompt = get_custom_prompt(
            self.args.prompt_tier, ref_arch_src=problem.code,
            backend="cutile", option="one_shot", precision="fp32")
        directory = self._attempt_dir(pid, attempt)
        (directory / "candidates").mkdir(parents=True, exist_ok=True)
        runtime = {
            "pid": pid,
            "problem": problem.name,
            "ref_src": problem.code,
            "base_prompt": base_prompt,
            "attempt": attempt,
            "dir": directory,
            "status": "active",
            "started_mono": None,
            "started_wall": None,
            "deadline_mono": None,
            "deadline_wall": None,
            "seq": 0,
            "gen_inflight": set(),
            "verify_pending": set(),
            "seen_code": {},
            "initial_slots": {
                index: {"done": False, "candidate_id": None}
                for index in range(self.args.initial_candidates)
            },
            "frontier_id": None,
            "repair_cycle": None,
            "cycle_seq": 0,
            "effective_conclusions": 0,
            "no_code_conclusions": 0,
            "stagnant_waves": 0,
            "terminal_reason": None,
            "next_branch": "repair",
        }
        record["status"] = "active"
        record["terminal_reason"] = None
        self.active[pid] = runtime
        self.runtimes[pid] = runtime
        self.active_order.append(pid)
        self._save_task(runtime)
        self._save_state()

    def _activate_available(self):
        while (self.pending
               and len(self.active) < self.args.active_problems):
            self._activate_one(self.pending.popleft())

    def _generation_count(self):
        return len(self.generation_tasks)

    def _verification_count(self):
        # Include work belonging to a task that just timed out.  The persistent
        # verifier cannot cancel an individual CUDA call, so excluding those
        # items would release backpressure while the GPUs are still occupied.
        return sum(
            candidate.get("status") == "verifying"
            for candidate in self.candidates.values())

    def _update_backpressure(self):
        count = self._verification_count()
        if not self.backpressured and count >= self.args.backpressure_high:
            self.backpressured = True
        elif self.backpressured and count <= self.args.backpressure_low:
            self.backpressured = False

    def _initial_slot_to_submit(self, runtime):
        for index in range(self.args.initial_candidates):
            row = runtime["initial_slots"][index]
            if not row["done"] and row["candidate_id"] is None:
                return index
        return None

    def _initial_gate_open(self, runtime):
        return all(row["done"] for row in runtime["initial_slots"].values())

    def _candidate_outstanding(self, candidate_id):
        candidate = self.candidates.get(candidate_id) or {}
        return candidate.get("status") in ("generating", "verifying")

    def _cycle_outstanding(self, runtime):
        cycle = runtime.get("repair_cycle")
        if not cycle:
            return 0
        return sum(
            self._candidate_outstanding(candidate_id)
            for candidate_id in cycle["submitted"])

    def _start_cycle(self, runtime):
        runtime["cycle_seq"] += 1
        kind = "repair" if runtime.get("frontier_id") else "fresh_retry"
        runtime["next_branch"] = "repair"
        runtime["repair_cycle"] = {
            "id": runtime["cycle_seq"],
            "kind": kind,
            "target_id": runtime.get("frontier_id"),
            "submitted": [],
            "effective": [],
        }
        self._save_task(runtime)

    def _next_spec(self, runtime):
        initial_slot = self._initial_slot_to_submit(runtime)
        if initial_slot is not None:
            return {
                "branch": "initial",
                "initial_slot": initial_slot,
                "parent_id": None,
                "depth": 0,
                "cycle_id": None,
            }
        if runtime.get("frontier_id") is None and not self._initial_gate_open(runtime):
            return {
                "branch": "fresh",
                "initial_slot": None,
                "parent_id": None,
                "depth": 0,
                "cycle_id": None,
            }
        if runtime.get("repair_cycle") is None:
            self._start_cycle(runtime)
        cycle = runtime["repair_cycle"]
        needed = 2 - len(cycle["effective"]) - self._cycle_outstanding(runtime)
        if needed > 0:
            if (cycle["kind"] == "repair"
                    and runtime.get("next_branch") == "fresh"
                    and self._cycle_outstanding(runtime) > 0):
                runtime["next_branch"] = "repair"
                return {
                    "branch": "fresh",
                    "initial_slot": None,
                    "parent_id": None,
                    "depth": 0,
                    "cycle_id": None,
                }
            runtime["next_branch"] = "fresh"
            return {
                "branch": cycle["kind"],
                "initial_slot": None,
                "parent_id": cycle.get("target_id"),
                "depth": cycle["id"],
                "cycle_id": cycle["id"],
            }
        return {
            "branch": "fresh",
            "initial_slot": None,
            "parent_id": None,
            "depth": 0,
            "cycle_id": None,
        }

    def _build_prompt(self, runtime, spec):
        if spec["branch"] != "repair":
            return runtime["base_prompt"]
        parent = self.candidates.get(spec["parent_id"])
        if not parent or not parent.get("code"):
            return runtime["base_prompt"]
        history = [{
            "round": max(0, int(spec["depth"]) - 1),
            "code": parent["code"],
            "result": parent.get("result") or {},
        }]
        role_round = ((max(1, int(spec["depth"])) - 1) % 3) + 1
        return build_adaptive_prompt(
            runtime["base_prompt"], history, role_round)

    def _submit(self, runtime, spec):
        runtime["seq"] += 1
        seq = runtime["seq"]
        candidate_id = "%d:%d:%d" % (
            runtime["pid"], runtime["attempt"], seq)
        prompt = self._build_prompt(runtime, spec)
        now_mono = self.monotonic()
        now_wall = self.wall_clock()
        candidate = {
            "candidate_id": candidate_id,
            "problem_id": runtime["pid"],
            "attempt": runtime["attempt"],
            "seq": seq,
            "branch": spec["branch"],
            "initial_slot": spec.get("initial_slot"),
            "parent_id": spec.get("parent_id"),
            "depth": spec.get("depth", 0),
            "cycle_id": spec.get("cycle_id"),
            "status": "generating",
            "submitted_mono": now_mono,
            "submitted_wall": now_wall,
            "submitted_at": wall_timestamp(now_wall),
            "response_finished_mono": None,
            "verify_finished_mono": None,
            "code_hash": None,
            "code": None,
            "result": None,
            "request": None,
            "dir": runtime["dir"] / "candidates",
            "prompt": prompt,
        }
        self.candidates[candidate_id] = candidate
        runtime["gen_inflight"].add(candidate_id)
        if spec.get("initial_slot") is not None:
            runtime["initial_slots"][spec["initial_slot"]][
                "candidate_id"] = candidate_id
        if spec.get("cycle_id") is not None:
            runtime["repair_cycle"]["submitted"].append(candidate_id)
        self._write_text(
            Path(str(self._candidate_prefix(candidate)) + "_prompt.txt"),
            prompt)
        self._save_candidate(candidate)
        task = asyncio.create_task(self._generate(candidate))
        self.generation_tasks[candidate_id] = task

        if (runtime["started_mono"] is None
                and all(row["candidate_id"] is not None
                        for row in runtime["initial_slots"].values())):
            runtime["started_mono"] = now_mono
            runtime["started_wall"] = now_wall
            runtime["deadline_mono"] = now_mono + self.args.deadline
            runtime["deadline_wall"] = now_wall + self.args.deadline
            record = self.state["tasks"][str(runtime["pid"])]
            record["started_at"] = wall_timestamp(now_wall)
            record["deadline_at"] = wall_timestamp(
                runtime["deadline_wall"])
            self._save_state()
        self._save_task(runtime)

    async def _generate(self, candidate):
        try:
            row = await self.chat.complete([{
                "role": "user", "content": candidate["prompt"]
            }])
            event = {
                "type": "generated",
                "candidate_id": candidate["candidate_id"],
                "response": row,
                "finished_mono": self.monotonic(),
                "finished_wall": self.wall_clock(),
            }
        except asyncio.CancelledError:
            event = {
                "type": "generation_canceled",
                "candidate_id": candidate["candidate_id"],
                "finished_mono": self.monotonic(),
                "finished_wall": self.wall_clock(),
            }
        except Exception as error:
            event = {
                "type": "generated",
                "candidate_id": candidate["candidate_id"],
                "response": {
                    "text": "",
                    "error": "%s: %s" % (
                        type(error).__name__, str(error)[:1000]),
                    "finish_reason": "error",
                },
                "finished_mono": self.monotonic(),
                "finished_wall": self.wall_clock(),
            }
        self.event_queue.put_nowait(event)

    def _mark_initial(self, runtime, candidate, concluded):
        slot = candidate.get("initial_slot")
        if slot is None:
            return
        row = runtime["initial_slots"][int(slot)]
        if concluded:
            row["done"] = True
        else:
            row["candidate_id"] = None

    def _cycle_effective(self, runtime, candidate):
        cycle = runtime.get("repair_cycle")
        if (not cycle or candidate.get("cycle_id") != cycle.get("id")
                or candidate["candidate_id"] in cycle["effective"]):
            return
        cycle["effective"].append(candidate["candidate_id"])
        if len(cycle["effective"]) >= 2:
            self._finish_cycle(runtime)

    def _note_evidence(self, runtime, candidate, no_code=False):
        if candidate.get("evidence_counted"):
            return
        candidate["evidence_counted"] = True
        runtime["effective_conclusions"] += 1
        if no_code:
            runtime["no_code_conclusions"] += 1

    def _early_exit_ready(self, runtime):
        elapsed = self._elapsed(runtime)
        return (
            self._initial_gate_open(runtime)
            and elapsed is not None
            and elapsed >= self.args.min_early_exit_s
            and runtime["effective_conclusions"] >= self.args.min_evidence
            and runtime["stagnant_waves"] >= self.args.stagnant_waves
        )

    def _no_code_exit_ready(self, runtime):
        elapsed = self._elapsed(runtime)
        return (
            self._initial_gate_open(runtime)
            and elapsed is not None
            and elapsed >= self.args.min_early_exit_s
            and runtime["no_code_conclusions"] >= self.args.no_code_evidence
        )

    def _maybe_frontier(self, runtime, candidate):
        if not candidate.get("code") or not candidate.get("result"):
            return
        if candidate["result"].get("passed"):
            return
        current_id = runtime.get("frontier_id")
        current = self.candidates.get(current_id)
        if current is None or frontier_sort_key(candidate) > frontier_sort_key(current):
            runtime["frontier_id"] = candidate["candidate_id"]

    def _finish_cycle(self, runtime):
        cycle = runtime.get("repair_cycle")
        if not cycle or len(cycle["effective"]) < 2:
            return
        candidates = [
            self.candidates[candidate_id]
            for candidate_id in cycle["effective"][:2]
        ]
        if cycle["kind"] == "fresh_retry":
            usable = [
                candidate for candidate in candidates
                if candidate.get("code") and candidate.get("result")
            ]
            if usable or runtime.get("frontier_id"):
                runtime["repair_cycle"] = None
                self._save_task(runtime)
                return
            if self._no_code_exit_ready(runtime):
                self._terminalize(
                    runtime, "stagnated", "no_code_after_fresh_retry")
            else:
                runtime["repair_cycle"] = None
                self._save_task(runtime)
            return

        target = self.candidates.get(cycle.get("target_id"))
        improved = any(
            frontier_improved(target, candidate) for candidate in candidates)
        current = self.candidates.get(runtime.get("frontier_id"))
        if target is not None and current is not None:
            improved = improved or frontier_improved(target, current)
        if improved:
            runtime["stagnant_waves"] = 0
            runtime["repair_cycle"] = None
            self._save_task(runtime)
            return
        runtime["stagnant_waves"] += 1
        if self._early_exit_ready(runtime):
            self._terminalize(
                runtime, "stagnated", "frontier_no_improvement")
        else:
            runtime["repair_cycle"] = None
            self._save_task(runtime)

    def _elapsed(self, runtime, finished_mono=None):
        if runtime.get("started_mono") is None:
            return None
        finished_mono = (
            self.monotonic() if finished_mono is None else finished_mono)
        return float(finished_mono) - float(runtime["started_mono"])

    def _within_deadline(self, runtime, finished_mono):
        elapsed = self._elapsed(runtime, finished_mono)
        return elapsed is not None and elapsed <= self.args.deadline

    def _terminalize(self, runtime, status, reason, candidate=None,
                     finished_mono=None, finished_wall=None):
        if runtime["status"] != "active":
            return
        finished_mono = (
            self.monotonic() if finished_mono is None else finished_mono)
        finished_wall = (
            self.wall_clock() if finished_wall is None else finished_wall)
        runtime["status"] = status
        runtime["terminal_reason"] = reason
        elapsed = self._elapsed(runtime, finished_mono)
        record = self.state["tasks"][str(runtime["pid"])]
        record["status"] = status
        record["terminal_reason"] = reason
        record["elapsed_s"] = round(elapsed, 6) if elapsed is not None else None
        record["finished_at"] = wall_timestamp(finished_wall)
        record["attempt"] = runtime["attempt"]
        record["frontier_id"] = runtime.get("frontier_id")
        if candidate is not None:
            frozen = self.frozen_dir / (
                "level_%d_problem_%d_sample_0_kernel.py"
                % (self.args.level, runtime["pid"]))
            self._write_text(frozen, candidate["code"])
            meta = {
                "candidate_id": candidate["candidate_id"],
                "problem_id": runtime["pid"],
                "attempt": runtime["attempt"],
                "elapsed_to_solution_s": round(elapsed, 6),
                "verified_at": wall_timestamp(finished_wall),
                "kernel_file": str(frozen),
                "online_result": candidate.get("result"),
            }
            atomic_json(
                self.frozen_dir / (
                    "level_%d_problem_%d_sample_0_meta.json"
                    % (self.args.level, runtime["pid"])),
                meta)
            record["solution"] = meta
        for candidate_id in list(runtime["gen_inflight"]):
            task = self.generation_tasks.get(candidate_id)
            if task is not None and not task.done():
                task.cancel()
        canceled_verify = list(runtime["verify_pending"])
        if canceled_verify and hasattr(self.verifier, "cancel"):
            self.verifier.cancel(canceled_verify)
        for candidate_id in canceled_verify:
            candidate = self.candidates.get(candidate_id)
            if candidate is not None and candidate.get("status") == "verifying":
                candidate["status"] = "verification_canceled"
                self._save_candidate(candidate)
        runtime["verify_pending"].clear()
        self.active.pop(runtime["pid"], None)
        try:
            self.active_order.remove(runtime["pid"])
        except ValueError:
            pass
        self._save_task(runtime)
        self._save_state()

    async def _handle_generated(self, event):
        from kernelbench.utils import extract_best_code

        candidate_id = event["candidate_id"]
        candidate = self.candidates[candidate_id]
        runtime = self.runtimes[candidate["problem_id"]]
        runtime["gen_inflight"].discard(candidate_id)
        self.generation_tasks.pop(candidate_id, None)
        candidate["response_finished_mono"] = event["finished_mono"]
        candidate["response_finished_wall"] = event["finished_wall"]
        if event["type"] == "generation_canceled":
            candidate["status"] = "canceled"
            self._save_candidate(candidate)
            self._save_task(runtime)
            return

        response = event["response"]
        text = response.get("text") or ""
        candidate["request"] = {
            key: value for key, value in response.items() if key != "text"
        }
        self._write_text(
            Path(str(self._candidate_prefix(candidate)) + "_response.txt"),
            text)
        if runtime["status"] != "active" or not self._within_deadline(
                runtime, event["finished_mono"]):
            candidate["status"] = "late_response"
            self._save_candidate(candidate)
            return
        if response.get("error"):
            candidate["status"] = "api_error"
            self._mark_initial(runtime, candidate, concluded=False)
            self._save_candidate(candidate)
            self._save_task(runtime)
            return

        code = extract_best_code(text, ["python", "cpp"]) if text else None
        if code is None:
            candidate["status"] = "no_code"
            candidate["result"] = {
                "key": candidate_id,
                "passed": False,
                "stage": "no_code",
                "error": "no complete code block in response",
                "seconds": 0.0,
            }
            self._mark_initial(runtime, candidate, concluded=True)
            if candidate["branch"] != "repair":
                self._note_evidence(runtime, candidate, no_code=True)
            if candidate["branch"] == "fresh_retry":
                self._cycle_effective(runtime, candidate)
            self._save_candidate(candidate)
            self._save_task(runtime)
            return

        digest = code_digest(code)
        candidate["code_hash"] = digest
        candidate["code"] = code
        self._write_text(
            Path(str(self._candidate_prefix(candidate)) + "_kernel.py"),
            code)
        if digest in runtime["seen_code"]:
            candidate["status"] = "duplicate"
            candidate["duplicate_of"] = runtime["seen_code"][digest]
            self._mark_initial(runtime, candidate, concluded=True)
            if candidate["branch"] == "fresh_retry":
                self._cycle_effective(runtime, candidate)
            self._save_candidate(candidate)
            self._save_task(runtime)
            return
        runtime["seen_code"][digest] = candidate_id
        candidate["status"] = "verifying"
        runtime["verify_pending"].add(candidate_id)
        self._save_candidate(candidate)
        self._save_task(runtime)
        await self.verifier.submit(candidate_id, code, runtime["ref_src"])

    def _handle_verified(self, event):
        candidate_id = event["candidate_id"]
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            return
        runtime = self.runtimes[candidate["problem_id"]]
        runtime["verify_pending"].discard(candidate_id)
        result = event["result"]
        candidate["result"] = result
        candidate["verify_finished_mono"] = event["finished_mono"]
        candidate["verify_finished_wall"] = event["finished_wall"]
        if runtime["status"] != "active" or not self._within_deadline(
                runtime, event["finished_mono"]):
            candidate["status"] = "verified_late"
            self._save_candidate(candidate)
            self._save_task(runtime)
            return
        if result.get("stage") in INCONCLUSIVE_STAGES:
            candidate["status"] = "inconclusive"
            if candidate.get("code_hash"):
                runtime["seen_code"].pop(candidate["code_hash"], None)
            self._mark_initial(runtime, candidate, concluded=False)
            self._save_candidate(candidate)
            self._save_task(runtime)
            return

        candidate["status"] = "verified"
        self._mark_initial(runtime, candidate, concluded=True)
        if result.get("passed"):
            self._save_candidate(candidate)
            self._terminalize(
                runtime, "solved", "first_valid", candidate=candidate,
                finished_mono=event["finished_mono"],
                finished_wall=event["finished_wall"])
            return
        self._note_evidence(runtime, candidate)
        self._maybe_frontier(runtime, candidate)
        if candidate["branch"] in ("repair", "fresh_retry"):
            self._cycle_effective(runtime, candidate)
        self._save_candidate(candidate)
        self._save_task(runtime)

    async def _handle_event(self, event):
        if event["type"] in ("generated", "generation_canceled"):
            await self._handle_generated(event)
        elif event["type"] == "verified":
            self._handle_verified(event)
        self._update_backpressure()

    def _expire_deadlines(self):
        now = self.monotonic()
        for runtime in list(self.active.values()):
            deadline = runtime.get("deadline_mono")
            if deadline is not None and now >= deadline:
                self._terminalize(
                    runtime, "timeout", "deadline",
                    finished_mono=deadline,
                    finished_wall=runtime["deadline_wall"])

    def _launch_guard_reached(self, runtime):
        if not self.args.launch_guard_s or runtime.get("deadline_mono") is None:
            return False
        return (
            runtime["deadline_mono"] - self.monotonic()
            <= self.args.launch_guard_s
        )

    def _apply_launch_guards(self):
        for runtime in list(self.active.values()):
            if (self._launch_guard_reached(runtime)
                    and not runtime["gen_inflight"]
                    and not runtime["verify_pending"]):
                self._terminalize(
                    runtime, "stagnated",
                    "insufficient_time_for_new_attempt")

    def _fill_generation(self):
        self._apply_launch_guards()
        self._update_backpressure()
        if self.backpressured or not self.active_order:
            return
        idle_rounds = 0
        while (self._generation_count() < self.args.global_slots
               and self.active_order and not self.backpressured):
            pid = self.active_order[0]
            self.active_order.rotate(-1)
            runtime = self.active.get(pid)
            if runtime is None or runtime["status"] != "active":
                idle_rounds += 1
            elif self._launch_guard_reached(runtime):
                idle_rounds += 1
            elif len(runtime["gen_inflight"]) >= self.args.per_task_slots:
                idle_rounds += 1
            else:
                self._submit(runtime, self._next_spec(runtime))
                idle_rounds = 0
            if idle_rounds >= len(self.active_order):
                break

    def _next_timeout(self):
        now = self.monotonic()
        deadlines = [
            runtime["deadline_mono"] for runtime in self.active.values()
            if runtime.get("deadline_mono") is not None
        ]
        if not deadlines:
            return 0.5
        return max(0.0, min(0.5, min(deadlines) - now))

    async def _shutdown(self):
        for task in list(self.generation_tasks.values()):
            if not task.done():
                task.cancel()
        if self.generation_tasks:
            await asyncio.gather(
                *list(self.generation_tasks.values()),
                return_exceptions=True)
        await self.verifier.close()
        while not self.event_queue.empty():
            await self._handle_event(self.event_queue.get_nowait())

    def _write_trajectories(self):
        path = self.run_dir / "trajectories.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as out:
            for pid in self.problem_ids:
                record = self.state["tasks"][str(pid)]
                candidates = []
                problem_dir = self._problem_dir(pid)
                if problem_dir.is_dir():
                    for candidate_path in sorted(
                            problem_dir.glob(
                                "attempt_*/candidates/*_candidate.json")):
                        candidates.append(json.loads(candidate_path.read_text()))
                out.write(json.dumps({
                    "tag": self.args.tag,
                    "level": self.args.level,
                    "problem_id": pid,
                    "problem": record["problem"],
                    "status": record["status"],
                    "terminal_reason": record.get("terminal_reason"),
                    "elapsed_s": record.get("elapsed_s"),
                    "solution": record.get("solution"),
                    "candidates": candidates,
                }) + "\n")
        os.replace(str(tmp), str(path))

    async def run(self):
        self.load_or_initialize()
        if not self.pending:
            self._write_trajectories()
            print("Hybrid run already complete: %d tasks" % len(self.problem_ids))
            return
        if self.chat is None:
            self.chat = AsyncDynamicChat(
                self.args.base_url, self.args.model, self.args.tokenizer,
                self.args.temperature, self.args.top_p, self.args.top_k,
                self.args.max_tokens, self.args.native_context,
                self.args.safety_margin, self.args.request_timeout)
        if self.verifier is None:
            self.verifier = VerifierDispatcher(
                self.event_queue, workers=self.args.verify_workers,
                gpus=self.args.verify_gpus,
                timeout_s=self.args.verify_timeout,
                batch_size=self.args.verify_batch)
        await self.verifier.start()
        try:
            while self.pending or self.active:
                self._expire_deadlines()
                self._activate_available()
                self._fill_generation()
                if not self.active:
                    continue
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=self._next_timeout())
                except asyncio.TimeoutError:
                    continue
                await self._handle_event(event)
        finally:
            await self._shutdown()
            self._save_state()
            self._write_trajectories()
        counts = self.state.get("counts") or {}
        print("Hybrid complete: solved=%d timeout=%d stagnated=%d total=%d"
              % (counts.get("solved", 0), counts.get("timeout", 0),
                 counts.get("stagnated", 0), len(self.problem_ids)))


def resolve_profile_args(args):
    profile = profile_for(args.tag, args.profile)
    for name in (
            "global_slots", "active_problems", "per_task_slots",
            "initial_candidates", "min_early_exit_s", "min_evidence",
            "stagnant_waves", "no_code_evidence", "launch_guard_s"):
        if getattr(args, name) is None:
            setattr(args, name, profile[name])
    if args.request_timeout is None:
        args.request_timeout = args.deadline + 60.0
    if args.backpressure_low >= args.backpressure_high:
        raise SystemExit("backpressure-low must be below backpressure-high")
    if args.initial_candidates > args.per_task_slots:
        raise SystemExit("initial candidates exceed per-task slots")


def command_run(args):
    from kernelbench.dataset import construct_kernelbench_dataset

    resolve_profile_args(args)
    dataset = construct_kernelbench_dataset(args.level)
    problem_ids = select_problem_ids(dataset, args.ids, args.limit)
    if not problem_ids:
        raise SystemExit("no problems selected")
    scheduler = HybridScheduler(args, dataset, problem_ids)
    asyncio.run(scheduler.run())


def command_status(args):
    path = Path(args.run_dir) / "state.json"
    if not path.is_file():
        raise SystemExit("missing Hybrid state: %s" % path)
    state = json.loads(path.read_text())
    counts = Counter(
        row.get("status", "pending") for row in state["tasks"].values())
    print("%s level %s: %s" % (
        state["tag"], state["level"],
        " ".join("%s=%d" % pair for pair in sorted(counts.items()))))


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run")
    run.add_argument("--run-dir", required=True)
    run.add_argument(
        "--tag", required=True, choices=sorted(PROFILE_CONFIGS["60s"]))
    run.add_argument(
        "--profile", choices=sorted(PROFILE_CONFIGS), default="60s")
    run.add_argument("--level", type=int, required=True)
    run.add_argument("--ids", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--prompt-tier", default="cutile_concepts")
    run.add_argument("--base-url", default="http://localhost:8000/v1")
    run.add_argument("--model", default="Qwen3-Coder-Next")
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--native-context", type=int, required=True)
    run.add_argument("--safety-margin", type=int, default=1024)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--top-k", type=int, default=40)
    run.add_argument("--max-tokens", type=int, default=131072)
    run.add_argument("--request-timeout", type=float, default=None)
    run.add_argument("--deadline", type=float, default=60.0)
    run.add_argument("--global-slots", type=int, default=None)
    run.add_argument("--active-problems", type=int, default=None)
    run.add_argument("--per-task-slots", type=int, default=None)
    run.add_argument("--initial-candidates", type=int, default=None)
    run.add_argument("--min-early-exit-s", type=float, default=None)
    run.add_argument("--min-evidence", type=int, default=None)
    run.add_argument("--stagnant-waves", type=int, default=None)
    run.add_argument("--no-code-evidence", type=int, default=None)
    run.add_argument("--launch-guard-s", type=float, default=None)
    run.add_argument("--verify-workers", type=int, default=4)
    run.add_argument("--verify-gpus", type=int, default=2)
    run.add_argument("--verify-timeout", type=float, default=45.0)
    run.add_argument("--verify-batch", type=int, default=4)
    run.add_argument("--backpressure-high", type=int, default=64)
    run.add_argument("--backpressure-low", type=int, default=32)
    run.set_defaults(func=command_run)

    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True)
    status.set_defaults(func=command_status)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.error("choose run or status")
    args.func(args)


if __name__ == "__main__":
    main()
