import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_600s_report import ORDER, pairwise, render  # noqa: E402


class Hybrid600sReportTest(unittest.TestCase):
    def test_pairwise_and_render(self):
        self.assertEqual(
            pairwise({1: 2.0, 2: 1.0}, {1: 1.0, 2: 2.0}),
            {
                "n": 2,
                "median": 1.25,
                "new_faster": 1,
                "new_slower": 1,
            })
        data = {}
        for tag in ORDER:
            data[tag] = {
                "control": {1},
                "adaptive": {1, 2},
                "short": {1},
                "long": {1, 3},
                "pairwise_adaptive": {
                    "n": 1,
                    "median": 1.1,
                    "new_faster": 1,
                    "new_slower": 0,
                },
                "score": {
                    "config": {
                        "global_slots": 32,
                        "active_problems": 16,
                        "per_task_slots": 2,
                    },
                    "candidates": {
                        "totals": {
                            "submitted": 10,
                            "completed": 8,
                            "verified": 6,
                            "fresh": 6,
                            "repair": 4,
                        }
                    },
                    "early_exit": {"count": 1},
                    "official_evaluation": {
                        "selected_candidates": 2,
                        "validated": 2,
                        "speedup": {"median": 1.2},
                    },
                },
            }
        text = render(data)
        self.assertIn("Hybrid 600 秒搜索正式结果", text)
        self.assertIn("600秒新增", text)
        self.assertIn("正式校验通过 12/12", text)


if __name__ == "__main__":
    unittest.main()
