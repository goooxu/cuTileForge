#!/usr/bin/env python3
"""Render six Hybrid scorecards into the final Markdown report."""

import argparse
import json
from pathlib import Path


DISPLAY = {
    "GLE": "GL-E",
    "GL": "GL",
    "Q": "Next-Q",
    "base": "Next",
    "G4t": "G4t",
    "Q38": "Q38",
}
ORDER = ("GLE", "GL", "Q", "base", "G4t", "Q38")


def parse_scorecards(values):
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--scorecard must be TAG=PATH")
        tag, path = value.split("=", 1)
        score = json.loads(Path(path).read_text(encoding="utf-8"))
        if score.get("tag") != tag:
            raise ValueError("%s contains tag %s" % (path, score.get("tag")))
        out[tag] = score
    missing = [tag for tag in ORDER if tag not in out]
    if missing:
        raise ValueError("missing scorecards: %s" % ", ".join(missing))
    return out


def formal_cell(row):
    return "%d/%d (%.1f%%)" % (
        row["passed"], row["n"], row["pass_rate"])


def official_result(score):
    if "official_evaluation" in score:
        return score["official_evaluation"]
    # Compatibility for scorecards produced before the terminology fix.
    old = score["post_facto"]
    return {
        "selected_candidates": old["online_solved"],
        "checked": old["rechecked"],
        "validated": old["revalidated"],
        "validation_failed": old["revalidation_failed"],
        "timed": old["timed"],
        "kernel_ms": old["kernel_ms"],
        "speedup": old["speedup"],
        "faster_than_reference": old.get("faster_than_reference", 0),
        "speedup_ge_1_05": old.get("speedup_ge_1_05", 0),
        "speedup_le_0_95": old.get("speedup_le_0_95", 0),
    }


def render(scores):
    lines = [
        "# 六模型 909 题 Hybrid 60 秒搜索正式结果",
        "",
        "固定 4 张 GPU，GPU 0–1 运行 TP2 生成，GPU 2–3 运行 4 个在线",
        "verifier workers。每题生成与搜索预算为 60 秒；在线 verifier 只属于搜索",
        "反馈与候选选择，不记录为测试成绩。搜索结束后的干净 4-GPU 阶段是唯一的",
        "正式校验与打分，以下通过数和通过率均来自该正式阶段。",
        "",
        "## 主结果",
        "",
        "| 模型 | 全部正式通过 | latency | throughput | 提前退出 | 正式校验 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag in ORDER:
        score = scores[tag]
        formal = score["official_results"]
        early = score["early_exit"]
        official = official_result(score)
        lines.append(
            "| %s | %s | %s | %s | %d | %d/%d |"
            % (
                DISPLAY[tag],
                formal_cell(formal["all"]),
                formal_cell(formal.get("latency", {
                    "passed": 0, "n": 0, "pass_rate": 0.0})),
                formal_cell(formal.get("throughput", {
                    "passed": 0, "n": 0, "pass_rate": 0.0})),
                early["count"],
                official["validated"], official["selected_candidates"],
            ))
    categories = sorted(set(
        name for score in scores.values() for name in score["official_results"]
        if name not in ("all", "latency", "throughput")))
    lines.extend([
        "",
        "## 各算子族正式通过率",
        "",
        "| 算子族 | %s |" % " | ".join(DISPLAY[tag] for tag in ORDER),
        "| --- | %s |" % " | ".join("---:" for _ in ORDER),
    ])
    for category in categories:
        cells = []
        for tag in ORDER:
            row = scores[tag]["official_results"].get(category)
            cells.append(formal_cell(row) if row else "0/0 (0.0%)")
        lines.append("| %s | %s |" % (category, " | ".join(cells)))
    lines.extend([
        "",
        "## 正式校验与打分",
        "",
        "| 模型 | 正式校验 | `kernel_ms` 中位 / p95 | speedup 中位 / p95 | "
        "speedup ≥1.05 / ≤0.95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for tag in ORDER:
        official = official_result(scores[tag])
        lines.append(
            "| %s | %d/%d | %.5f / %.5f | %.3fx / %.3fx | %d / %d |"
            % (
                DISPLAY[tag], official["validated"],
                official["selected_candidates"],
                official["kernel_ms"]["median"],
                official["kernel_ms"]["p95"],
                official["speedup"]["median"],
                official["speedup"]["p95"],
                official["speedup_ge_1_05"],
                official["speedup_le_0_95"],
            ))
    lines.extend([
        "",
        "## 搜索行为",
        "",
        "| 模型 | 槽 / 活跃题 | 候选提交 / 完成 / 验证 | fresh / repair | "
        "最大反馈深度 | 节省题秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for tag in ORDER:
        score = scores[tag]
        config = score["config"]
        totals = score["candidates"]["totals"]
        early = score["early_exit"]
        lines.append(
            "| %s | %d / %d | %d / %d / %d | %d / %d | %d | %.1f |"
            % (
                DISPLAY[tag], config["global_slots"],
                config["active_problems"], totals["submitted"],
                totals["completed"], totals["verified"], totals["fresh"],
                totals["repair"],
                score["candidates"]["per_task"]["max_depth"]["max"],
                early["saved_problem_seconds"],
            ))
    best_tag = max(
        ORDER, key=lambda value:
        scores[value]["official_results"]["all"]["pass_rate"])
    total_selected = sum(
        official_result(scores[tag])["selected_candidates"] for tag in ORDER)
    total_validated = sum(
        official_result(scores[tag])["validated"] for tag in ORDER)
    total_exits = sum(scores[tag]["early_exit"]["count"] for tag in ORDER)
    total_saved = sum(
        scores[tag]["early_exit"]["saved_problem_seconds"] for tag in ORDER)
    q38_candidates = scores["Q38"]["candidates"]["totals"]
    lines.extend([
        "",
        "## 结论",
        "",
        "- %s 的正式通过率最高，为 %.1f%%；第二名是 GL-E 的 %.1f%%。"
        % (
            DISPLAY[best_tag],
            scores[best_tag]["official_results"]["all"]["pass_rate"],
            scores["GLE"]["official_results"]["all"]["pass_rate"],
        ),
        "- Q38 只完成 %d/%d 个生成请求，正式通过 %d 题；60 秒主要受长 thinking / "
        "decode 限制；增加并发不会解决单请求来不及结束的问题。"
        % (
            q38_candidates["completed"], q38_candidates["submitted"],
            scores["Q38"]["official_results"]["all"]["passed"],
        ),
        "- GL-E、GL、G4t、Q38 没有进入 repair；它们通常无法在 deadline 前让每题 "
        "8 个初始候选全部得到结论。反馈分支只在 Next-Q / Next 实际发生。",
        "- 激进提前退出共触发 %d 题，节省 %.1f 题秒；大部分失败题仍由 60 秒 "
        "deadline 终止。" % (total_exits, total_saved),
        "- 正式校验共通过 %d/%d 个搜索后提交的候选，没有校验失败。"
        % (total_validated, total_selected),
    ])
    lines.extend([
        "",
        "## 审计",
        "",
    ])
    for tag in ORDER:
        score = scores[tag]
        early = score["early_exit"]
        official = official_result(score)
        lines.append(
            "- %s：提前退出审计 %s；提交正式校验 %d，正式通过 %d，完成 "
            "`kernel_ms` %d。"
            % (
                DISPLAY[tag],
                "通过" if early["audit_passed"] else "失败",
                official["selected_candidates"], official["validated"],
                official["timed"],
            ))
    lines.extend([
        "",
        "候选只在搜索结束后的正式阶段产生测试成绩。本协议单列，不替换 p@4",
        "表 A/B 或 128K control/adaptive 表。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scores = parse_scorecards(args.scorecard)
    text = render(scores)
    Path(args.out).write_text(text, encoding="utf-8")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
