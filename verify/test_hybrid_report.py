import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_report import ORDER, render  # noqa: E402


class HybridReportTest(unittest.TestCase):
    def test_render_six_models(self):
        scores = {}
        for tag in ORDER:
            scores[tag] = {
                "official_results": {
                    "all": {"passed": 1, "n": 2, "pass_rate": 50.0},
                    "latency": {
                        "passed": 1, "n": 1, "pass_rate": 100.0},
                    "throughput": {
                        "passed": 0, "n": 1, "pass_rate": 0.0},
                },
                "early_exit": {
                    "count": 1,
                    "saved_problem_seconds": 20.0,
                    "audit_passed": True,
                },
                "official_evaluation": {
                    "selected_candidates": 1,
                    "checked": 1,
                    "validated": 1,
                    "validation_failed": 0,
                    "timed": 1,
                    "kernel_ms": {"median": 0.1, "p95": 0.2},
                    "speedup": {"median": 1.1, "p95": 1.2},
                    "faster_than_reference": 1,
                    "speedup_ge_1_05": 1,
                    "speedup_le_0_95": 0,
                },
                "config": {"global_slots": 64, "active_problems": 8},
                "candidates": {
                    "totals": {
                        "submitted": 10,
                        "completed": 8,
                        "verified": 6,
                        "fresh": 8,
                        "repair": 2,
                    },
                    "per_task": {"max_depth": {"max": 2.0}},
                },
            }
        text = render(scores)
        self.assertIn("# 六模型 909 题 Hybrid 60 秒搜索正式结果", text)
        self.assertIn("| GL-E | 1/2 (50.0%)", text)
        self.assertIn("| Q38 | 1/2 (50.0%)", text)
        self.assertIn("提前退出审计 通过", text)


if __name__ == "__main__":
    unittest.main()
