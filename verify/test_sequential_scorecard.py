import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequential_scorecard import (  # noqa: E402
    code_counts,
    fastest,
    pairwise,
    summarize_requests,
)


class SequentialScorecardTest(unittest.TestCase):
    def test_fastest_uses_kernel_ms_not_speedup(self):
        rows = {
            1: [
                {"key": "1:0", "passed": True,
                 "kernel_ms": 2.0, "speedup": 10.0},
                {"key": "1:1", "passed": True,
                 "kernel_ms": 1.0, "speedup": 2.0},
            ]
        }
        self.assertEqual(fastest(rows, [1])[1]["kernel_ms"], 1.0)

    def test_adaptive_result_shape_is_supported(self):
        rows = {
            1: [
                {"round": 0, "code": "x", "result": {
                    "passed": True, "kernel_ms": 3.0}},
                {"round": 1, "code": "y", "result": {
                    "passed": True, "kernel_ms": 1.5}},
            ]
        }
        self.assertEqual(
            fastest(rows, [1])[1]["result"]["kernel_ms"], 1.5)

    def test_pairwise_direction(self):
        control = {1: {"passed": True, "kernel_ms": 2.0}}
        adaptive = {1: {"passed": True, "kernel_ms": 1.0}}
        self.assertEqual(pairwise(control, adaptive)["median"], 2.0)

    def test_counts_static_kernel_and_launch_sites(self):
        code = """
@ct.kernel
def a(): pass
@ct.kernel()
def b(): pass
ct.launch(stream, grid, a, ())
ct.launch (stream, grid, b, ())
"""
        self.assertEqual(code_counts(code), {
            "kernel_defs": 2,
            "launch_sites": 2,
        })

    def test_token_budget_summary(self):
        summary = summarize_requests([
            {"prompt_tokens": 100, "max_tokens_sent": 1000,
             "completion_tokens": 10, "hit_token_limit": False},
            {"prompt_tokens": 200, "max_tokens_sent": 900,
             "completion_tokens": 20, "hit_token_limit": True},
        ])
        self.assertEqual(summary["input_tokens_median"], 150)
        self.assertEqual(summary["max_tokens_min"], 900)
        self.assertEqual(summary["hit_token_limit_rate"], 50)


if __name__ == "__main__":
    unittest.main()
