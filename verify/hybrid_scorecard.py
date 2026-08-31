#!/usr/bin/env python3
"""Score formal validation after a fixed 60-second Hybrid search."""

import argparse
import collections
import json
import os
import re
import statistics
from pathlib import Path


KERNEL_DEF_RE = re.compile(r"@\s*ct\.kernel\b")
LAUNCH_RE = re.compile(r"\bct\.launch\s*\(")
TERMINAL = frozenset(("solved", "timeout", "stagnated"))
INCONCLUSIVE = frozenset(("oom", "cuda_poison", "worker_crash"))


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    return ordered[int(round((len(ordered) - 1) * quantile))]


def numeric_summary(values):
    values = [float(value) for value in values if value is not None]
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def frontier_rank(result):
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


def is_frontier_improvement(previous, current):
    old = (previous or {}).get("result") or {}
    new = (current or {}).get("result") or {}
    old_rank = frontier_rank(old)
    new_rank = frontier_rank(new)
    if old_rank != new_rank:
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


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(row["problem_id"]): row for row in data["problems"]}


def load_jsonl(path):
    if not path or not os.path.isfile(path):
        return []
    with open(path) as source:
        return [
            json.loads(line) for line in source if line.strip()
        ]


def load_candidate(path):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    row["_path"] = str(path)
    return row


def final_attempt_candidates(run_dir, state):
    by_problem = {}
    for pid_text, task in state["tasks"].items():
        pid = int(pid_text)
        attempt = int(task.get("attempt") or task.get("attempts") or 0)
        pattern = (
            Path(run_dir) / "problems" / ("problem_%04d" % pid)
            / ("attempt_%03d" % attempt) / "candidates"
            / "*_candidate.json"
        )
        by_problem[pid] = [
            load_candidate(path) for path in sorted(pattern.parent.glob(pattern.name))
        ] if attempt else []
    return by_problem


def all_attempt_candidates(run_dir, state):
    by_problem = {}
    for pid_text in state["tasks"]:
        pid = int(pid_text)
        root = Path(run_dir) / "problems" / ("problem_%04d" % pid)
        by_problem[pid] = [
            load_candidate(path) for path in sorted(
                root.glob("attempt_*/candidates/*_candidate.json"))
        ]
    return by_problem


def ids_by_group(manifest, selected_ids):
    selected = set(selected_ids)
    groups = {"all": sorted(selected)}
    for role in ("latency", "throughput"):
        ids = sorted(
            pid for pid in selected
            if manifest.get(pid, {}).get("role") == role)
        if ids:
            groups[role] = ids
    categories = sorted(set(
        manifest.get(pid, {}).get("category") for pid in selected
        if manifest.get(pid, {}).get("category")))
    for category in categories:
        groups[category] = sorted(
            pid for pid in selected
            if manifest.get(pid, {}).get("category") == category)
    return groups


def search_terminal_stats(state, ids):
    tasks = state["tasks"]
    rows = [tasks[str(pid)] for pid in ids]
    return {
        "n": len(rows),
        "candidate_selected": sum(
            row.get("status") == "solved" for row in rows),
        "timeout": sum(row.get("status") == "timeout" for row in rows),
        "stagnated": sum(row.get("status") == "stagnated" for row in rows),
        "nonterminal": sum(row.get("status") not in TERMINAL for row in rows),
    }


def official_result_stats(post, ids):
    checked = [pid for pid in ids if pid in post]
    passed = [pid for pid in checked if post[pid].get("passed")]
    return {
        "n": len(ids),
        "submitted": len(checked),
        "passed": len(passed),
        "pass_rate": 100.0 * len(passed) / len(ids) if ids else 0.0,
        "validation_failed": len(checked) - len(passed),
    }


def candidate_task_summary(candidates):
    submitted = len(candidates)
    completed = sum(
        row.get("response_finished_mono") is not None
        and row.get("status") != "canceled"
        for row in candidates)
    verified = sum(row.get("result") is not None for row in candidates)
    repairs = sum(row.get("branch") == "repair" for row in candidates)
    fresh = sum(
        row.get("branch") in ("initial", "fresh", "fresh_retry")
        for row in candidates)
    depths = [
        int(row.get("depth") or 0)
        for row in candidates if row.get("branch") == "repair"
    ]
    return {
        "submitted": submitted,
        "completed": completed,
        "verified": verified,
        "fresh": fresh,
        "repair": repairs,
        "max_depth": max(depths) if depths else 0,
        "duplicates": sum(row.get("status") == "duplicate" for row in candidates),
        "api_errors": sum(row.get("status") == "api_error" for row in candidates),
        "no_code": sum(row.get("status") == "no_code" for row in candidates),
        "inconclusive": sum(
            row.get("status") == "inconclusive" for row in candidates),
        "verification_canceled": sum(
            row.get("status") == "verification_canceled"
            for row in candidates),
        "late": sum(
            row.get("status") in ("late_response", "verified_late")
            for row in candidates),
    }


def aggregate_candidates(by_problem, ids):
    rows = {
        pid: candidate_task_summary(by_problem.get(pid, [])) for pid in ids
    }
    keys = (
        "submitted", "completed", "verified", "fresh", "repair",
        "max_depth", "duplicates", "api_errors", "no_code",
        "inconclusive", "verification_canceled", "late",
    )
    return {
        "tasks": len(rows),
        "per_task": {
            key: numeric_summary([row[key] for row in rows.values()])
            for key in keys
        },
        "totals": {
            key: sum(row[key] for row in rows.values()) for key in keys
        },
    }


def request_summary(by_problem, ids):
    requests = [
        row.get("request") or {}
        for pid in ids for row in by_problem.get(pid, [])
        if row.get("request")
    ]

    def numbers(key):
        return [
            row[key] for row in requests if row.get(key) is not None
        ]

    inputs = [
        row.get("prompt_tokens")
        if row.get("prompt_tokens") is not None
        else row.get("input_tokens_estimate")
        for row in requests
        if (row.get("prompt_tokens") is not None
            or row.get("input_tokens_estimate") is not None)
    ]
    return {
        "n": len(requests),
        "input_tokens": numeric_summary(inputs),
        "max_tokens_sent": numeric_summary(numbers("max_tokens_sent")),
        "completion_tokens": numeric_summary(numbers("completion_tokens")),
        "hit_token_limit": sum(
            bool(row.get("hit_token_limit")) for row in requests),
        "request_errors": sum(bool(row.get("error")) for row in requests),
        "context_exhausted": sum(
            row.get("finish_reason") == "context_exhausted"
            for row in requests),
    }


def failure_summary(state, by_problem, ids):
    terminal_reasons = collections.Counter(
        state["tasks"][str(pid)].get("terminal_reason") or "none"
        for pid in ids
    )
    stages = collections.Counter()
    statuses = collections.Counter()
    for pid in ids:
        for candidate in by_problem.get(pid, []):
            statuses[candidate.get("status") or "unknown"] += 1
            result = candidate.get("result")
            if result:
                if result.get("passed"):
                    stages["pass"] += 1
                else:
                    stages[result.get("stage") or "unknown"] += 1
    return {
        "terminal_reasons": dict(terminal_reasons),
        "candidate_statuses": dict(statuses),
        "verifier_stages": dict(stages),
    }


def early_exit_summary(run_dir, state, ids):
    elapsed = []
    reasons = collections.Counter()
    invalid = []
    config = state.get("config") or {}
    deadline = float(config.get("deadline_s", 60.0))
    initial_needed = int(config.get("initial_candidates", 8))
    min_early = float(config.get("min_early_exit_s", 0.0))
    min_evidence = int(config.get("min_evidence", 2))
    required_waves = int(config.get("stagnant_waves", 1))
    no_code_needed = int(config.get("no_code_evidence", 2))
    launch_guard = float(config.get("launch_guard_s", 0.0))
    legacy_policy = "min_evidence" not in config
    for pid in ids:
        task = state["tasks"][str(pid)]
        if task.get("status") != "stagnated":
            continue
        elapsed.append(task.get("elapsed_s") or 0.0)
        reasons[task.get("terminal_reason") or "unknown"] += 1
        attempt = int(task.get("attempt") or task.get("attempts") or 0)
        snapshot = (
            Path(run_dir) / "problems" / ("problem_%04d" % pid)
            / ("attempt_%03d" % attempt) / "task.json"
        )
        if not snapshot.is_file():
            invalid.append({"problem_id": pid, "reason": "missing_task_snapshot"})
            continue
        detail = json.loads(snapshot.read_text(encoding="utf-8"))
        cycle = detail.get("repair_cycle") or {}
        if detail.get("initial_done") != initial_needed:
            invalid.append({
                "problem_id": pid,
                "reason": "initial_candidates_not_all_concluded",
            })
        reason = task.get("terminal_reason")
        if reason == "insufficient_time_for_new_attempt":
            if ((task.get("elapsed_s") or 0.0)
                    < deadline - launch_guard - 1.0):
                invalid.append({
                    "problem_id": pid,
                    "reason": "launch_guard_exit_too_early",
                })
            if (detail.get("generation_inflight")
                    or detail.get("verification_pending")):
                invalid.append({
                    "problem_id": pid,
                    "reason": "launch_guard_exit_with_pending_work",
                })
            continue
        if len(cycle.get("effective") or []) < 2:
            invalid.append({
                "problem_id": pid,
                "reason": "fewer_than_two_effective_repairs",
            })
            continue
        candidates = {
            row.get("candidate_id"): row for row in (
                load_candidate(path) for path in sorted(
                    snapshot.parent.glob("candidates/*_candidate.json")))
        }
        effective_ids = list(cycle.get("effective") or [])[:2]
        effective = [candidates.get(candidate_id)
                     for candidate_id in effective_ids]
        if any(row is None for row in effective):
            invalid.append({
                "problem_id": pid,
                "reason": "missing_effective_candidate_record",
            })
            continue
        if any(
                ((row.get("result") or {}).get("stage") in INCONCLUSIVE)
                for row in effective):
            invalid.append({
                "problem_id": pid,
                "reason": "infrastructure_failure_counted_as_repair",
            })
        hashes = [
            row.get("code_hash") for row in effective
            if row.get("code_hash")
        ]
        if len(hashes) != len(set(hashes)):
            invalid.append({
                "problem_id": pid,
                "reason": "duplicate_code_counted_as_repair",
            })
        if reason == "frontier_no_improvement":
            if (task.get("elapsed_s") or 0.0) < min_early:
                invalid.append({
                    "problem_id": pid,
                    "reason": "frontier_exit_before_minimum_time",
                })
            if (not legacy_policy
                    and detail.get("effective_conclusions", 0) < min_evidence):
                invalid.append({
                    "problem_id": pid,
                    "reason": "frontier_exit_without_minimum_evidence",
                })
            if (not legacy_policy
                    and detail.get("stagnant_waves", 0) < required_waves):
                invalid.append({
                    "problem_id": pid,
                    "reason": "frontier_exit_without_required_waves",
                })
            target = candidates.get(cycle.get("target_id"))
            if target is None:
                invalid.append({
                    "problem_id": pid,
                    "reason": "missing_frontier_target",
                })
            elif any(is_frontier_improvement(target, row)
                     for row in effective):
                invalid.append({
                    "problem_id": pid,
                    "reason": "improving_repair_was_early_exited",
                })
        elif reason == "no_code_after_fresh_retry":
            if (task.get("elapsed_s") or 0.0) < min_early:
                invalid.append({
                    "problem_id": pid,
                    "reason": "no_code_exit_before_minimum_time",
                })
            if (not legacy_policy
                    and detail.get("no_code_conclusions", 0) < no_code_needed):
                invalid.append({
                    "problem_id": pid,
                    "reason": "no_code_exit_without_minimum_evidence",
                })
            if any(
                    row.get("code") and row.get("result")
                    and frontier_rank(row.get("result")) > 0
                    for row in effective):
                invalid.append({
                    "problem_id": pid,
                    "reason": "usable_fresh_retry_was_early_exited",
                })
    return {
        "count": len(elapsed),
        "elapsed_s": numeric_summary(elapsed),
        "saved_problem_seconds": sum(max(0.0, deadline - value)
                                     for value in elapsed),
        "reasons": dict(reasons),
        "audit_passed": not invalid,
        "audit_failures": invalid,
    }


def load_post_verified(path):
    out = {}
    for row in load_jsonl(path):
        key = str(row.get("key", ""))
        try:
            pid = int(key.split(":")[0])
        except (ValueError, IndexError):
            continue
        out[pid] = row
    return out


def official_evaluation_summary(post, state, ids):
    online = [
        pid for pid in ids
        if state["tasks"][str(pid)].get("status") == "solved"
    ]
    checked = [pid for pid in online if pid in post]
    passed = [pid for pid in checked if post[pid].get("passed")]
    timed = [
        pid for pid in passed if post[pid].get("kernel_ms") is not None
    ]
    kernel_ms = [post[pid]["kernel_ms"] for pid in timed]
    speedups = [
        post[pid]["speedup"] for pid in timed
        if post[pid].get("speedup") is not None
    ]
    return {
        "selected_candidates": len(online),
        "checked": len(checked),
        "validated": len(passed),
        "validation_failed": len(checked) - len(passed),
        "timed": len(timed),
        "kernel_ms": numeric_summary(kernel_ms),
        "speedup": numeric_summary(speedups),
        "faster_than_reference": sum(value > 1.0 for value in speedups),
        "speedup_ge_1_05": sum(value >= 1.05 for value in speedups),
        "speedup_le_0_95": sum(value <= 0.95 for value in speedups),
    }


def selected_code_summary(run_dir, level, state, ids):
    definitions = []
    launches = []
    for pid in ids:
        if state["tasks"][str(pid)].get("status") != "solved":
            continue
        path = (
            Path(run_dir) / "frozen"
            / ("level_%d_problem_%d_sample_0_kernel.py" % (level, pid))
        )
        code = path.read_text(encoding="utf-8") if path.is_file() else ""
        definitions.append(len(KERNEL_DEF_RE.findall(code)))
        launches.append(len(LAUNCH_RE.findall(code)))
    return {
        "n": len(definitions),
        "kernel_defs": numeric_summary(definitions),
        "launch_sites": numeric_summary(launches),
    }


def build_scorecard(tag, level, run_dir, manifest_path, post_path=None):
    state = json.loads(
        (Path(run_dir) / "state.json").read_text(encoding="utf-8"))
    if state.get("tag") != tag or int(state.get("level")) != int(level):
        raise ValueError("scorecard arguments do not match Hybrid state")
    manifest = load_manifest(manifest_path)
    ids = sorted(int(pid) for pid in state["tasks"])
    missing = sorted(set(ids) - set(manifest))
    if missing:
        raise ValueError("manifest missing problem ids: %s" % missing[:8])
    by_problem = final_attempt_candidates(run_dir, state)
    all_attempts = all_attempt_candidates(run_dir, state)
    groups = ids_by_group(manifest, ids)
    post = load_post_verified(
        post_path or os.path.join(run_dir, "post_verified.jsonl"))
    official_results = {
        name: official_result_stats(post, group_ids)
        for name, group_ids in groups.items()
    }
    time_to_selection = numeric_summary([
        state["tasks"][str(pid)].get("elapsed_s")
        for pid in ids
        if state["tasks"][str(pid)].get("status") == "solved"
    ])
    return {
        "tag": tag,
        "level": level,
        "protocol": state.get("protocol"),
        "config": state.get("config"),
        "official_results": official_results,
        "search_telemetry": {
            "terminal": search_terminal_stats(state, ids),
            "time_to_candidate_selection_s": time_to_selection,
        },
        "candidates": aggregate_candidates(by_problem, ids),
        "all_attempts": aggregate_candidates(all_attempts, ids),
        "requests": request_summary(by_problem, ids),
        "failures": failure_summary(state, by_problem, ids),
        "early_exit": early_exit_summary(run_dir, state, ids),
        "official_evaluation": official_evaluation_summary(post, state, ids),
        "selected_code": selected_code_summary(
            run_dir, level, state, ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--post-verified", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    score = build_scorecard(
        args.tag, args.level, args.run_dir, args.manifest,
        args.post_verified)
    all_stats = score["official_results"]["all"]
    search = score["search_telemetry"]["terminal"]
    early = score["early_exit"]
    official = score["official_evaluation"]
    print("%s Hybrid %.0f-second search formal results"
          % (args.tag, float(score["config"]["deadline_s"])))
    print("  formal pass %d/%d (%.1f%%), validation failures %d"
          % (all_stats["passed"], all_stats["n"],
             all_stats["pass_rate"], all_stats["validation_failed"]))
    print("  search terminal: selected %d, timeout %d, stagnated %d"
          % (search["candidate_selected"], search["timeout"],
             search["stagnated"]))
    print("  early exits %d, saved %.1f problem-seconds, audit %s"
          % (early["count"], early["saved_problem_seconds"],
             "PASS" if early["audit_passed"] else "FAIL"))
    print("  official validation passed %d/%d, timed %d"
          % (official["validated"], official["selected_candidates"],
             official["timed"]))
    with open(args.out, "w") as out:
        json.dump(score, out, indent=2, sort_keys=True)
        out.write("\n")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
