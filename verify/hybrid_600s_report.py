#!/usr/bin/env python3
"""Compare Hybrid-600s formal results with prior 60s and k=4 protocols."""

import argparse
import json
import statistics
from pathlib import Path


ORDER = ("GLE", "GL", "Q", "base", "G4t", "Q38")
DISPLAY = {
    "GLE": "GL-E",
    "GL": "GL",
    "Q": "Next-Q",
    "base": "Next",
    "G4t": "G4t",
    "Q38": "Q38",
}


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]


def control_passes(path):
    passed = set()
    for row in jsonl(path):
        if row.get("passed"):
            passed.add(int(str(row["key"]).split(":")[0]))
    return passed


def adaptive_passes_and_ms(path):
    state = json_file(Path(path) / "state.json")
    passed = set()
    fastest = {}
    for pid_text, task in state["tasks"].items():
        pid = int(pid_text)
        for entry in task.get("history", []):
            result = entry.get("result") or {}
            if not result.get("passed"):
                continue
            passed.add(pid)
            if result.get("kernel_ms") is not None:
                value = float(result["kernel_ms"])
                fastest[pid] = min(value, fastest.get(pid, value))
    return passed, fastest


def formal_passes_and_ms(run_dir):
    passed = set()
    timing = {}
    for row in jsonl(Path(run_dir) / "post_verified.jsonl"):
        if not row.get("passed"):
            continue
        pid = int(str(row["key"]).split(":")[0])
        passed.add(pid)
        if row.get("kernel_ms") is not None:
            timing[pid] = float(row["kernel_ms"])
    return passed, timing


def rate(count, total=909):
    return 100.0 * count / total


def pairwise(old_ms, new_ms):
    common = sorted(set(old_ms) & set(new_ms))
    ratios = [old_ms[pid] / new_ms[pid] for pid in common]
    return {
        "n": len(common),
        "median": statistics.median(ratios) if ratios else 0.0,
        "new_faster": sum(value >= 1.05 for value in ratios),
        "new_slower": sum(value <= 0.95 for value in ratios),
    }


def collect(runs_root):
    root = Path(runs_root)
    out = {}
    for tag in ORDER:
        control = control_passes(
            root / ("seq128ctrl_%s_l60_verified.jsonl" % tag))
        adaptive, adaptive_ms = adaptive_passes_and_ms(
            root / ("seq128_%s_l60" % tag))
        short, short_ms = formal_passes_and_ms(
            root / ("hybrid60_%s_l60" % tag))
        long_run = root / ("hybrid600_%s_l60" % tag)
        long, long_ms = formal_passes_and_ms(long_run)
        score = json_file(
            root / ("hybrid600_%s_l60_scorecard.json" % tag))
        out[tag] = {
            "control": control,
            "adaptive": adaptive,
            "short": short,
            "long": long,
            "score": score,
            "pairwise_adaptive": pairwise(adaptive_ms, long_ms),
            "long_ms": long_ms,
            "short_ms": short_ms,
        }
    return out


def render(data):
    lines = [
        "# 六模型 Hybrid 600 秒搜索正式结果",
        "",
        "每题生成与搜索预算为 600 秒。在线 verifier 仅用于反馈与候选选择；",
        "以下通过数、通过率和 `kernel_ms` 均来自搜索结束后的正式校验与打分。",
        "旧 control/adaptive 每题完整执行 4 次生成，没有 600 秒预算，因而只作",
        "协议间对照，不是等预算排名。",
        "",
        "## 正式通过率对比",
        "",
        "| 模型 | 旧 control p@4 | 旧 adaptive p@4 | 60秒搜索 | "
        "600秒搜索 | 600秒相对 adaptive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag in ORDER:
        row = data[tag]
        delta = rate(len(row["long"])) - rate(len(row["adaptive"]))
        lines.append(
            "| %s | %d/909 (%.1f%%) | %d/909 (%.1f%%) | "
            "%d/909 (%.1f%%) | %d/909 (%.1f%%) | %+.1f pp |"
            % (
                DISPLAY[tag],
                len(row["control"]), rate(len(row["control"])),
                len(row["adaptive"]), rate(len(row["adaptive"])),
                len(row["short"]), rate(len(row["short"])),
                len(row["long"]), rate(len(row["long"])),
                delta,
            ))
    lines.extend([
        "",
        "## 与旧 adaptive 的覆盖互补",
        "",
        "| 模型 | 共同通过 | 600秒新增 | adaptive 独有 | 合集 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for tag in ORDER:
        old = data[tag]["adaptive"]
        new = data[tag]["long"]
        lines.append(
            "| %s | %d | %d | %d | %d |"
            % (
                DISPLAY[tag], len(old & new), len(new - old),
                len(old - new), len(old | new),
            ))
    lines.extend([
        "",
        "## 正式性能对比",
        "",
        "`旧/新 kernel_ms` 大于 1 表示 600 秒方案更快；只比较两种协议都正式",
        "通过且完成计时的同题。",
        "",
        "| 模型 | 共同题 | 旧/新中位 | 新方案 ≥1.05 | 新方案 ≤0.95 | "
        "600秒正式 speedup 中位 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for tag in ORDER:
        row = data[tag]
        pair = row["pairwise_adaptive"]
        official = row["score"]["official_evaluation"]
        lines.append(
            "| %s | %d | %.3fx | %d | %d | %.3fx |"
            % (
                DISPLAY[tag], pair["n"], pair["median"],
                pair["new_faster"], pair["new_slower"],
                official["speedup"]["median"],
            ))
    lines.extend([
        "",
        "## 600 秒搜索行为",
        "",
        "| 模型 | 槽 / 活跃题 / 每题在途 | 提交 / 完成 / 在线验证 | "
        "fresh / repair | 提前退出 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for tag in ORDER:
        score = data[tag]["score"]
        config = score["config"]
        totals = score["candidates"]["totals"]
        lines.append(
            "| %s | %d / %d / %d | %d / %d / %d | %d / %d | %d |"
            % (
                DISPLAY[tag], config["global_slots"],
                config["active_problems"], config["per_task_slots"],
                totals["submitted"], totals["completed"], totals["verified"],
                totals["fresh"], totals["repair"],
                score["early_exit"]["count"],
            ))
    best = max(ORDER, key=lambda tag: len(data[tag]["long"]))
    total_selected = sum(
        row["score"]["official_evaluation"]["selected_candidates"]
        for row in data.values())
    total_validated = sum(
        row["score"]["official_evaluation"]["validated"]
        for row in data.values())
    lines.extend([
        "",
        "## 结论",
        "",
        "- 600 秒搜索正式通过率最高的是 %s：%d/909（%.1f%%）。"
        % (DISPLAY[best], len(data[best]["long"]), rate(len(data[best]["long"]))),
        "- 正式校验通过 %d/%d 个搜索后提交候选。"
        % (total_validated, total_selected),
        "- 提前退出、跨题 batch 和反馈深度的详细数据见各模型 scorecard。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    text = render(collect(args.runs_root))
    Path(args.out).write_text(text, encoding="utf-8")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
