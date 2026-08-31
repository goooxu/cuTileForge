import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_scorecard import build_scorecard, numeric_summary  # noqa: E402


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class HybridScorecardTest(unittest.TestCase):
    def make_run(self, root):
        run_dir = Path(root) / "run"
        state = {
            "protocol": "hybrid_search_60s",
            "tag": "GLE",
            "level": 60,
            "config": {"deadline_s": 60.0},
            "tasks": {
                "1": {
                    "problem": "one",
                    "status": "solved",
                    "attempt": 1,
                    "attempts": 1,
                    "elapsed_s": 10.0,
                    "terminal_reason": "first_valid",
                },
                "2": {
                    "problem": "two",
                    "status": "timeout",
                    "attempt": 1,
                    "attempts": 1,
                    "elapsed_s": 60.0,
                    "terminal_reason": "deadline",
                },
                "3": {
                    "problem": "three",
                    "status": "stagnated",
                    "attempt": 1,
                    "attempts": 1,
                    "elapsed_s": 20.0,
                    "terminal_reason": "frontier_no_improvement",
                },
            },
        }
        write_json(run_dir / "state.json", state)
        manifest = {
            "problems": [
                {"problem_id": 1, "role": "latency", "category": "matmul"},
                {"problem_id": 2, "role": "latency", "category": "conv"},
                {"problem_id": 3, "role": "throughput", "category": "matmul"},
            ]
        }
        manifest_path = Path(root) / "manifest.json"
        write_json(manifest_path, manifest)

        candidates = {
            1: {
                "candidate_id": "1:1:1",
                "problem_id": 1,
                "attempt": 1,
                "seq": 1,
                "branch": "initial",
                "status": "verified",
                "response_finished_mono": 5.0,
                "result": {"stage": "pass", "passed": True},
                "request": {
                    "input_tokens_estimate": 2000,
                    "max_tokens_sent": 120000,
                    "completion_tokens": 1000,
                },
            },
            2: {
                "candidate_id": "2:1:1",
                "problem_id": 2,
                "attempt": 1,
                "seq": 1,
                "branch": "repair",
                "depth": 2,
                "status": "verified",
                "response_finished_mono": 50.0,
                "result": {
                    "stage": "exec",
                    "passed": False,
                    "rel_diff": 0.3,
                },
            },
            3: {
                "candidate_id": "3:1:1",
                "problem_id": 3,
                "attempt": 1,
                "seq": 1,
                "branch": "initial",
                "depth": 0,
                "status": "verified",
                "response_finished_mono": 12.0,
                "result": {"stage": "purity", "passed": False},
            },
        }
        for pid, candidate in candidates.items():
            path = (
                run_dir / "problems" / ("problem_%04d" % pid)
                / "attempt_001" / "candidates" / "000001_candidate.json"
            )
            write_json(path, candidate)
        for seq in (2, 3):
            write_json(
                run_dir / "problems/problem_0003/attempt_001/candidates"
                / ("%06d_candidate.json" % seq),
                {
                    "candidate_id": "3:1:%d" % seq,
                    "problem_id": 3,
                    "attempt": 1,
                    "seq": seq,
                    "branch": "repair",
                    "depth": 1,
                    "status": "verified",
                    "response_finished_mono": 15.0 + seq,
                    "code_hash": "hash-%d" % seq,
                    "code": "class ModelNew%d: pass" % seq,
                    "result": {"stage": "purity", "passed": False},
                })
        write_json(
            run_dir / "problems/problem_0003/attempt_001/task.json",
            {
                "initial_done": 8,
                "repair_cycle": {
                    "target_id": "3:1:1",
                    "effective": ["3:1:2", "3:1:3"],
                },
            })
        frozen = run_dir / "frozen/level_60_problem_1_sample_0_kernel.py"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_text(
            "@ct.kernel\n"
            "def kernel():\n"
            "    pass\n"
            "ct.launch(pool, kernel, (1,), ())\n")
        post = run_dir / "post_verified.jsonl"
        post.write_text(json.dumps({
            "key": "1:0",
            "passed": True,
            "stage": "pass",
            "kernel_ms": 0.5,
            "speedup": 1.2,
        }) + "\n")
        return run_dir, manifest_path

    def test_scorecard_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self.make_run(tmp)
            score = build_scorecard(
                "GLE", 60, str(run_dir), str(manifest))
            self.assertEqual(
                score["official_results"]["all"]["passed"], 1)
            self.assertEqual(
                score["official_results"]["latency"]["n"], 2)
            self.assertEqual(
                score["official_results"]["matmul"]["n"], 2)
            self.assertEqual(
                score["search_telemetry"][
                    "time_to_candidate_selection_s"]["median"],
                10.0)
            self.assertEqual(
                score["candidates"]["totals"]["submitted"], 5)
            self.assertEqual(
                score["candidates"]["per_task"]["max_depth"]["max"], 2.0)
            self.assertEqual(score["early_exit"]["count"], 1)
            self.assertTrue(score["early_exit"]["audit_passed"])
            self.assertEqual(
                score["early_exit"]["saved_problem_seconds"], 40.0)
            self.assertEqual(
                score["official_evaluation"]["validated"], 1)
            self.assertEqual(
                score["official_evaluation"]["timed"], 1)
            self.assertEqual(score["selected_code"]["kernel_defs"]["max"], 1)
            self.assertEqual(score["selected_code"]["launch_sites"]["max"], 1)

    def test_early_exit_audit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self.make_run(tmp)
            write_json(
                run_dir / "problems/problem_0003/attempt_001/task.json",
                {"initial_done": 7, "repair_cycle": {"effective": ["one"]}})
            score = build_scorecard(
                "GLE", 60, str(run_dir), str(manifest))
            self.assertFalse(score["early_exit"]["audit_passed"])
            self.assertEqual(
                len(score["early_exit"]["audit_failures"]), 2)

    def test_600s_early_exit_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest = self.make_run(tmp)
            state_path = run_dir / "state.json"
            state = json.loads(state_path.read_text())
            state["config"] = {
                "deadline_s": 600.0,
                "initial_candidates": 2,
                "min_early_exit_s": 120.0,
                "min_evidence": 6,
                "stagnant_waves": 2,
                "no_code_evidence": 4,
                "launch_guard_s": 90.0,
            }
            state["tasks"]["3"]["elapsed_s"] = 130.0
            write_json(state_path, state)
            write_json(
                run_dir / "problems/problem_0003/attempt_001/task.json",
                {
                    "initial_done": 2,
                    "effective_conclusions": 6,
                    "no_code_conclusions": 0,
                    "stagnant_waves": 2,
                    "generation_inflight": [],
                    "verification_pending": [],
                    "repair_cycle": {
                        "target_id": "3:1:1",
                        "effective": ["3:1:2", "3:1:3"],
                    },
                })
            score = build_scorecard(
                "GLE", 60, str(run_dir), str(manifest))
            self.assertTrue(score["early_exit"]["audit_passed"])

    def test_numeric_summary_empty(self):
        self.assertEqual(
            numeric_summary([]),
            {"n": 0, "mean": 0.0, "median": 0.0,
             "p95": 0.0, "max": 0.0})


if __name__ == "__main__":
    unittest.main()
