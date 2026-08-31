#!/usr/bin/env python3
"""Equal-budget scorecard: independent k=4 versus sequential 1+1+1+1."""

import argparse
import collections
import json
import os
import re
import statistics


KERNEL_DEF_RE = re.compile(r"@\s*ct\.kernel\b")
LAUNCH_RE = re.compile(r"\bct\.launch\s*\(")


def load_manifest(path):
    data = json.load(open(path))
    return {int(row["problem_id"]): row for row in data["problems"]}


def load_jsonl(path):
    by = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid, sid = map(int, str(row["key"]).split(":"))
            by[pid].append((sid, row))
    return {pid: [row for _, row in sorted(rows)]
            for pid, rows in by.items()}


def load_adaptive(path):
    state = json.load(open(os.path.join(path, "state.json")))
    by = {}
    for pid, task in state["tasks"].items():
        by[int(pid)] = sorted(
            task.get("history", []),
            key=lambda row: int(row["round"]))
    return by


def passed(row):
    return bool((row.get("result") or row).get("passed"))


def result_of(row):
    return row.get("result") or row


def stats(by, ids, adaptive=False):
    rows = [row for pid in ids for row in by.get(pid, [])]
    n_pass = sum(passed(row) for row in rows)
    solved = sum(any(passed(row) for row in by.get(pid, [])) for pid in ids)
    round0 = [
        row for pid in ids for row in by.get(pid, [])
        if (int(row.get("round", -1)) == 0 if adaptive else
            int(str(result_of(row)["key"]).split(":")[1]) == 0)
    ]
    return {
        "n": len(ids),
        "solved": solved,
        "p4": 100.0 * solved / len(ids) if ids else 0.0,
        "candidate_pass_rate": (
            100.0 * n_pass / len(rows) if rows else 0.0),
        "round0_pass_rate": (
            100.0 * sum(passed(row) for row in round0) / len(round0)
            if round0 else 0.0),
    }


def fastest(by, ids):
    out = {}
    for pid in ids:
        candidates = [
            row for row in by.get(pid, [])
            if passed(row) and result_of(row).get("kernel_ms")
        ]
        if candidates:
            out[pid] = min(
                candidates,
                key=lambda row: float(result_of(row)["kernel_ms"]))
    return out


def pairwise(control, adaptive):
    common = sorted(set(control) & set(adaptive))
    ratios = [
        float(result_of(control[pid])["kernel_ms"])
        / float(result_of(adaptive[pid])["kernel_ms"])
        for pid in common
    ]
    return {
        "n": len(ratios),
        "median": statistics.median(ratios) if ratios else 0.0,
        "faster": sum(value >= 1.05 for value in ratios),
        "slower": sum(value <= 0.95 for value in ratios),
    }


def read_control_code(kernel_dir, level, pid, row):
    sid = int(str(row["key"]).split(":")[1])
    path = os.path.join(
        kernel_dir,
        "level_%d_problem_%d_sample_%d_kernel.py" % (level, pid, sid))
    return open(path).read() if os.path.isfile(path) else ""


def code_counts(code):
    return {
        "kernel_defs": len(KERNEL_DEF_RE.findall(code or "")),
        "launch_sites": len(LAUNCH_RE.findall(code or "")),
    }


def selected_code_summary(rows, adaptive, control_kernel_dir, level):
    defs = []
    launches = []
    for pid, row in rows.items():
        if adaptive:
            code = row.get("code") or ""
        else:
            code = read_control_code(
                control_kernel_dir, level, pid, result_of(row))
        counts = code_counts(code)
        defs.append(counts["kernel_defs"])
        launches.append(counts["launch_sites"])
    return {
        "n": len(defs),
        "median_kernel_defs": statistics.median(defs) if defs else 0.0,
        "median_launch_sites": statistics.median(launches) if launches else 0.0,
        "mean_kernel_defs": sum(defs) / len(defs) if defs else 0.0,
        "mean_launch_sites": (
            sum(launches) / len(launches) if launches else 0.0),
    }


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def summarize_requests(rows):
    rows = [row for row in rows if row]

    def numbers(key):
        return [float(row[key]) for row in rows if row.get(key) is not None]

    input_tokens = [
        float(row.get("prompt_tokens")
              if row.get("prompt_tokens") is not None
              else row.get("input_tokens_estimate"))
        for row in rows
        if (row.get("prompt_tokens") is not None
            or row.get("input_tokens_estimate") is not None)
    ]
    max_tokens = numbers("max_tokens_sent")
    completion = numbers("completion_tokens")
    return {
        "n": len(rows),
        "input_tokens_median": (
            statistics.median(input_tokens) if input_tokens else 0.0),
        "input_tokens_p90": percentile(input_tokens, 0.90),
        "max_tokens_median": (
            statistics.median(max_tokens) if max_tokens else 0.0),
        "max_tokens_p90": percentile(max_tokens, 0.90),
        "max_tokens_min": min(max_tokens) if max_tokens else 0.0,
        "completion_tokens_median": (
            statistics.median(completion) if completion else 0.0),
        "completion_tokens_p90": percentile(completion, 0.90),
        "hit_token_limit": sum(bool(row.get("hit_token_limit")) for row in rows),
        "hit_token_limit_rate": (
            100.0 * sum(bool(row.get("hit_token_limit")) for row in rows)
            / len(rows) if rows else 0.0),
        "request_errors": sum(bool(row.get("error")) for row in rows),
        "context_exhausted": sum(
            row.get("finish_reason") == "context_exhausted" for row in rows),
    }


def load_control_request_metadata(kernel_dir):
    rows = []
    for name in os.listdir(kernel_dir):
        if not name.endswith("_meta.json"):
            continue
        rows.append(json.load(open(os.path.join(kernel_dir, name))))
    return rows


def adaptive_request_metadata(by):
    by_round = collections.defaultdict(list)
    for rows in by.values():
        for row in rows:
            by_round[int(row.get("round", -1))].append(row.get("request") or {})
    return {str(round_index): summarize_requests(by_round.get(round_index, []))
            for round_index in range(4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--control", required=True,
                    help="Verified jsonl for independent k=4.")
    ap.add_argument("--control-kernels", required=True)
    ap.add_argument("--adaptive", required=True,
                    help="Sequential run directory containing state.json.")
    ap.add_argument("--min-extra-solved", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest_by = load_manifest(args.manifest)
    control = load_jsonl(args.control)
    adaptive = load_adaptive(args.adaptive)
    ids_by = {
        "all": sorted(manifest_by),
        "matmul": sorted(pid for pid, meta in manifest_by.items()
                         if meta.get("category") == "matmul"),
        "conv": sorted(pid for pid, meta in manifest_by.items()
                       if meta.get("category") == "conv"),
    }
    for role in ("latency", "throughput"):
        role_ids = sorted(
            pid for pid, meta in manifest_by.items()
            if meta.get("role") == role)
        if role_ids:
            ids_by[role] = role_ids
    control_stats = {
        name: stats(control, ids, adaptive=False)
        for name, ids in ids_by.items()
    }
    adaptive_stats = {
        name: stats(adaptive, ids, adaptive=True)
        for name, ids in ids_by.items()
    }
    control_fast = {
        name: fastest(control, ids) for name, ids in ids_by.items()
    }
    adaptive_fast = {
        name: fastest(adaptive, ids) for name, ids in ids_by.items()
    }
    pairs = {
        name: pairwise(control_fast[name], adaptive_fast[name])
        for name in ids_by
    }

    print("%s sequential 1+1+1+1 vs independent k=4" % args.tag)
    display_names = [
        name for name in ("all", "latency", "throughput", "matmul", "conv")
        if name in ids_by
    ]
    for name in display_names:
        c = control_stats[name]
        a = adaptive_stats[name]
        print("  %-6s control %d/%d p@4 %.1f%% pass/cand %.1f%% | "
              "adaptive %d/%d p@4 %.1f%% round0 %.1f%% pass/cand %.1f%%"
              % (name, c["solved"], c["n"], c["p4"],
                 c["candidate_pass_rate"], a["solved"], a["n"], a["p4"],
                 a["round0_pass_rate"], a["candidate_pass_rate"]))
        p = pairs[name]
        print("           best kernel_ms adaptive/control %.3fx n=%d "
              ">=1.05 %d <=0.95 %d"
              % (p["median"], p["n"], p["faster"], p["slower"]))

    rounds = {}
    solved_before = set()
    for round_index in range(4):
        rows = [
            row for task_rows in adaptive.values() for row in task_rows
            if int(row.get("round", -1)) == round_index
        ]
        solved_now = {
            pid for pid, task_rows in adaptive.items()
            if any(
                int(row.get("round", -1)) <= round_index and passed(row)
                for row in task_rows)
        }
        rounds[str(round_index)] = {
            "passed": sum(passed(row) for row in rows),
            "attempted": len(rows),
            "newly_solved": len(solved_now - solved_before),
            "cumulative_solved": len(solved_now),
        }
        solved_before = solved_now
    control_code = selected_code_summary(
        control_fast["all"], False, args.control_kernels, args.level)
    adaptive_code = selected_code_summary(
        adaptive_fast["all"], True, args.control_kernels, args.level)
    print("  selected code control defs %.1f launches %.1f | "
          "adaptive defs %.1f launches %.1f"
          % (control_code["median_kernel_defs"],
             control_code["median_launch_sites"],
             adaptive_code["median_kernel_defs"],
             adaptive_code["median_launch_sites"]))

    control_tokens = summarize_requests(
        load_control_request_metadata(args.control_kernels))
    adaptive_tokens = adaptive_request_metadata(adaptive)
    if control_tokens["n"]:
        print("  token budget control max median/p90 %.0f/%.0f "
              "input median/p90 %.0f/%.0f hit-limit %.1f%%"
              % (control_tokens["max_tokens_median"],
                 control_tokens["max_tokens_p90"],
                 control_tokens["input_tokens_median"],
                 control_tokens["input_tokens_p90"],
                 control_tokens["hit_token_limit_rate"]))
        for round_index in range(4):
            row = adaptive_tokens[str(round_index)]
            print("  token budget round %d max median/p90 %.0f/%.0f "
                  "input median/p90 %.0f/%.0f hit-limit %.1f%%"
                  % (round_index, row["max_tokens_median"],
                     row["max_tokens_p90"], row["input_tokens_median"],
                     row["input_tokens_p90"], row["hit_token_limit_rate"]))

    extra = (adaptive_stats["all"]["solved"]
             - control_stats["all"]["solved"])
    no_coverage_regression = extra >= 0
    material_gain = (
        extra >= args.min_extra_solved
        or pairs["all"]["median"] >= 1.05)
    gate = {
        "passed": no_coverage_regression and material_gain,
        "coverage_not_lower": no_coverage_regression,
        "extra_solved": extra,
        "min_extra_solved": args.min_extra_solved,
        "kernel_ms_ge_1_05": pairs["all"]["median"] >= 1.05,
    }
    print("GATE %s coverage=%s extra_solved=%d speed=%s"
          % ("PASS" if gate["passed"] else "FAIL",
             "Y" if no_coverage_regression else "N", extra,
             "Y" if gate["kernel_ms_ge_1_05"] else "N"))

    output = {
        "tag": args.tag,
        "control": control_stats,
        "adaptive": adaptive_stats,
        "kernel_ms": pairs,
        "rounds": rounds,
        "selected_code": {
            "control": control_code,
            "adaptive": adaptive_code,
        },
        "token_budget": {
            "control": control_tokens,
            "adaptive": adaptive_tokens,
        },
        "gate": gate,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
