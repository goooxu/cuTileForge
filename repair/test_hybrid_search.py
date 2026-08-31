import argparse
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_search import (
    HybridScheduler,
    PROFILE_CONFIGS,
    code_digest,
    deadline_accepts,
    frontier_improved,
    profile_for,
    resolve_profile_args,
    result_label,
    result_rank,
)


class FakeProblem:
    def __init__(self, pid):
        self.name = "problem-%d" % pid
        self.code = "class Model: pass  # %d" % pid


class FakeDataset:
    def __init__(self, count):
        self.problems = {
            pid: FakeProblem(pid) for pid in range(1, count + 1)}

    def get_problem_ids(self):
        return list(self.problems)

    def get_problem_by_id(self, pid):
        return self.problems[pid]


class BlockingChat:
    async def complete(self, messages):
        await asyncio.Event().wait()


class FakeVerifier:
    async def start(self):
        return None

    async def submit(self, candidate_id, code, ref_src):
        return None

    async def close(self):
        return None


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds
        return self.value


class ScriptedChat:
    def __init__(self, mode="code"):
        self.mode = mode
        self.counter = 0

    async def complete(self, messages):
        self.counter += 1
        if self.mode == "no_code":
            text = "No implementation was produced."
        else:
            is_repair = "Previous implementation:" in messages[0]["content"]
            text = (
                "```python\n"
                "# %s %d\n"
                "class ModelNew:\n"
                "    pass\n"
                "```"
                % ("repair" if is_repair else "fresh", self.counter)
            )
        return {
            "text": text,
            "error": None,
            "finish_reason": "stop",
            "input_tokens_estimate": 10,
            "max_tokens_sent": 100,
            "completion_tokens": 20,
        }


class ScriptedVerifier:
    def __init__(self, event_queue, clock, mode="fail"):
        self.event_queue = event_queue
        self.clock = clock
        self.mode = mode
        self.submitted = 0

    async def start(self):
        return None

    async def submit(self, candidate_id, code, ref_src):
        self.submitted += 1
        if self.mode == "pass":
            result = {"key": candidate_id, "passed": True, "stage": "pass"}
        elif self.mode == "late_pass":
            result = {"key": candidate_id, "passed": True, "stage": "pass"}
            self.clock.value = 61.0
        elif self.mode == "one_inconclusive" and self.submitted == 9:
            result = {
                "key": candidate_id,
                "passed": False,
                "stage": "worker_crash",
            }
        else:
            result = {
                "key": candidate_id,
                "passed": False,
                "stage": "exec",
                "error": "compile failed",
            }
        self.event_queue.put_nowait({
            "type": "verified",
            "candidate_id": candidate_id,
            "result": result,
            "finished_mono": self.clock.advance(0.01),
            "finished_wall": self.clock.value,
        })

    async def close(self):
        return None


def make_args(run_dir, tag, **overrides):
    profile_name = overrides.get("profile", "60s")
    profile = profile_for(tag, profile_name)
    values = {
        "run_dir": run_dir,
        "tag": tag,
        "profile": profile_name,
        "level": 60,
        "prompt_tier": "cutile_concepts",
        "deadline": 600.0 if profile_name == "600s" else 60.0,
        "global_slots": profile["global_slots"],
        "active_problems": profile["active_problems"],
        "per_task_slots": profile["per_task_slots"],
        "initial_candidates": profile["initial_candidates"],
        "min_early_exit_s": profile["min_early_exit_s"],
        "min_evidence": profile["min_evidence"],
        "stagnant_waves": profile["stagnant_waves"],
        "no_code_evidence": profile["no_code_evidence"],
        "launch_guard_s": profile["launch_guard_s"],
        "backpressure_high": 64,
        "backpressure_low": 32,
        "verify_workers": 4,
        "verify_gpus": 2,
        "verify_timeout": 45.0,
        "native_context": 131072 if tag in ("GLE", "GL") else 262144,
        "max_tokens": 131072,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class HybridPolicyTest(unittest.TestCase):
    def test_frozen_profiles(self):
        self.assertEqual(
            profile_for("GLE", "600s"),
            {
                "global_slots": 256,
                "active_problems": 128,
                "per_task_slots": 2,
                "initial_candidates": 2,
                "min_early_exit_s": 120,
                "min_evidence": 6,
                "stagnant_waves": 2,
                "no_code_evidence": 4,
                "launch_guard_s": 90,
            })
        self.assertEqual(profile_for("GL")["per_task_slots"], 12)
        self.assertEqual(profile_for("Q38", "600s")["active_problems"], 16)

    def test_rank_and_labels(self):
        rows = [
            ({"stage": "no_code"}, 0, "no_code"),
            ({"stage": "exec"}, 1, "compile/exec"),
            ({"stage": "purity"}, 2, "purity_wrong"),
            ({"stage": "exec", "rel_diff": 0.4}, 3, "numeric_mismatch"),
            (
                {"stage": "purity", "impure_correct": True},
                4,
                "purity_correct",
            ),
            ({"stage": "pass", "passed": True}, 5, "pass"),
        ]
        for result, rank, label in rows:
            self.assertEqual(result_rank(result), rank)
            self.assertEqual(result_label(result), label)

    def test_quantitative_frontier_improvement(self):
        numeric = {"result": {"stage": "exec", "rel_diff": 1.0}}
        self.assertTrue(frontier_improved(
            numeric, {"result": {"stage": "exec", "rel_diff": 0.8}}))
        self.assertFalse(frontier_improved(
            numeric, {"result": {"stage": "exec", "rel_diff": 0.81}}))
        purity = {
            "result": {
                "stage": "purity",
                "impure_correct": True,
                "torch_ops_left": 2,
            }
        }
        self.assertTrue(frontier_improved(
            purity,
            {
                "result": {
                    "stage": "purity",
                    "impure_correct": True,
                    "torch_ops_left": 1,
                }
            }))
        self.assertFalse(frontier_improved(purity, purity))

    def test_deadline_and_dedup(self):
        self.assertTrue(deadline_accepts(10.0, 70.0, 60.0))
        self.assertFalse(deadline_accepts(10.0, 70.001, 60.0))
        self.assertEqual(code_digest("x = 1  \n"), code_digest("x = 1\n"))

    def test_invalid_backpressure_and_initial_configuration(self):
        args = argparse.Namespace(
            tag="GLE", profile="60s",
            global_slots=None, active_problems=None,
            per_task_slots=None, backpressure_low=64,
            backpressure_high=64, initial_candidates=8,
            min_early_exit_s=None, min_evidence=None,
            stagnant_waves=None, no_code_evidence=None,
            launch_guard_s=None, request_timeout=None,
            deadline=60.0)
        with self.assertRaises(SystemExit):
            resolve_profile_args(args)
        args.backpressure_low = 32
        args.initial_candidates = 9
        args.per_task_slots = 8
        with self.assertRaises(SystemExit):
            resolve_profile_args(args)


class HybridSchedulerStateTest(unittest.TestCase):
    def run_scripted(self, tmp, chat_mode, verify_mode):
        args = make_args(
            tmp, "Q38", global_slots=8, active_problems=1,
            per_task_slots=8, backpressure_high=8,
            backpressure_low=4)
        dataset = FakeDataset(1)
        clock = FakeClock(0.0)
        chat = ScriptedChat(chat_mode)
        scheduler = HybridScheduler(
            args, dataset, [1], chat=chat, verifier=None,
            monotonic=clock, wall_clock=clock)
        verifier = ScriptedVerifier(
            scheduler.event_queue, clock, verify_mode)
        scheduler.verifier = verifier
        run(scheduler.run())
        return scheduler, verifier

    def test_interrupted_task_restarts_from_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q38")
            dataset = FakeDataset(4)
            first = HybridScheduler(
                args, dataset, dataset.get_problem_ids(),
                chat=BlockingChat(), verifier=FakeVerifier())
            first.load_or_initialize()
            first.state["tasks"]["1"]["status"] = "active"
            first._save_state()

            second = HybridScheduler(
                args, dataset, dataset.get_problem_ids(),
                chat=BlockingChat(), verifier=FakeVerifier())
            second.load_or_initialize()
            self.assertEqual(
                second.state["tasks"]["1"]["status"], "pending")
            self.assertEqual(
                second.state["tasks"]["1"]["interrupted_attempts"], 1)
            self.assertEqual(list(second.pending), [1, 2, 3, 4])

    def test_protocol_change_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q38")
            dataset = FakeDataset(4)
            first = HybridScheduler(
                args, dataset, dataset.get_problem_ids(),
                chat=BlockingChat(), verifier=FakeVerifier())
            first.load_or_initialize()
            changed = make_args(tmp, "Q38", deadline=15.0)
            second = HybridScheduler(
                changed, dataset, dataset.get_problem_ids(),
                chat=BlockingChat(), verifier=FakeVerifier())
            with self.assertRaises(SystemExit):
                second.load_or_initialize()

    def test_fake_clock_boundary(self):
        clock = FakeClock(100.0)
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q38")
            dataset = FakeDataset(4)
            scheduler = HybridScheduler(
                args, dataset, dataset.get_problem_ids(),
                chat=BlockingChat(), verifier=FakeVerifier(),
                monotonic=clock, wall_clock=clock)
            runtime = {"started_mono": 40.0}
            self.assertTrue(scheduler._within_deadline(runtime, 100.0))
            self.assertFalse(scheduler._within_deadline(runtime, 100.001))

    def test_600s_early_exit_requires_time_evidence_and_two_waves(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q", profile="600s")
            dataset = FakeDataset(1)
            clock = FakeClock(119.0)
            scheduler = HybridScheduler(
                args, dataset, [1], chat=BlockingChat(),
                verifier=FakeVerifier(), monotonic=clock, wall_clock=clock)
            runtime = {
                "started_mono": 0.0,
                "initial_slots": {
                    0: {"done": True}, 1: {"done": True}},
                "effective_conclusions": 6,
                "stagnant_waves": 2,
                "no_code_conclusions": 4,
            }
            self.assertFalse(scheduler._early_exit_ready(runtime))
            clock.value = 120.0
            self.assertTrue(scheduler._early_exit_ready(runtime))
            runtime["stagnant_waves"] = 1
            self.assertFalse(scheduler._early_exit_ready(runtime))
            runtime["stagnant_waves"] = 2
            runtime["effective_conclusions"] = 5
            self.assertFalse(scheduler._early_exit_ready(runtime))

    def test_600s_repair_starts_before_second_initial_concludes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q", profile="600s")
            dataset = FakeDataset(1)
            scheduler = HybridScheduler(
                args, dataset, [1], chat=BlockingChat(),
                verifier=FakeVerifier())
            scheduler._save_task = lambda runtime: None
            scheduler.candidates["frontier"] = {
                "candidate_id": "frontier",
                "seq": 1,
                "code": "class ModelNew: pass",
                "result": {"stage": "exec", "passed": False},
            }
            runtime = {
                "pid": 1,
                "dir": Path(tmp),
                "initial_slots": {
                    0: {"done": True, "candidate_id": "frontier"},
                    1: {"done": False, "candidate_id": "pending"},
                },
                "frontier_id": "frontier",
                "repair_cycle": None,
                "cycle_seq": 0,
                "next_branch": "repair",
            }
            spec = scheduler._next_spec(runtime)
            self.assertEqual(spec["branch"], "repair")
            self.assertEqual(spec["parent_id"], "frontier")

    def test_600s_launch_guard_is_model_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q38", profile="600s")
            dataset = FakeDataset(1)
            clock = FakeClock(420.0)
            scheduler = HybridScheduler(
                args, dataset, [1], chat=BlockingChat(),
                verifier=FakeVerifier(), monotonic=clock)
            runtime = {"deadline_mono": 600.0}
            self.assertTrue(scheduler._launch_guard_reached(runtime))
            clock.value = 419.9
            self.assertFalse(scheduler._launch_guard_reached(runtime))

    def test_backpressure_counts_late_verifier_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, "Q38")
            dataset = FakeDataset(4)
            scheduler = HybridScheduler(
                args, dataset, dataset.get_problem_ids(),
                chat=BlockingChat(), verifier=FakeVerifier())
            for index in range(64):
                scheduler.candidates[str(index)] = {"status": "verifying"}
            scheduler._update_backpressure()
            self.assertTrue(scheduler.backpressured)
            for index in range(32, 64):
                scheduler.candidates[str(index)]["status"] = "verified_late"
            scheduler._update_backpressure()
            self.assertFalse(scheduler.backpressured)

    def test_each_profile_fills_model_specific_slots(self):
        async def exercise(tag):
            profile = profile_for(tag)
            with tempfile.TemporaryDirectory() as tmp:
                args = make_args(tmp, tag)
                dataset = FakeDataset(profile["active_problems"])
                scheduler = HybridScheduler(
                    args, dataset, dataset.get_problem_ids(),
                    chat=BlockingChat(), verifier=FakeVerifier())
                scheduler.load_or_initialize()
                scheduler._activate_available()
                scheduler._fill_generation()
                self.assertEqual(
                    len(scheduler.active), profile["active_problems"])
                self.assertEqual(
                    scheduler._generation_count(), profile["global_slots"])
                for runtime in scheduler.active.values():
                    self.assertEqual(
                        len(runtime["gen_inflight"]),
                        profile["per_task_slots"])
                    self.assertTrue(all(
                        row["candidate_id"] is not None
                        for row in runtime["initial_slots"].values()))
                    self.assertIsNotNone(runtime["started_mono"])
                await scheduler._shutdown()

        for tag in sorted(PROFILE_CONFIGS["60s"]):
            run(exercise(tag))

    def test_first_valid_freezes_and_cancels_other_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            scheduler, verifier = self.run_scripted(
                tmp, "code", "pass")
            task = scheduler.state["tasks"]["1"]
            self.assertEqual(task["status"], "solved")
            self.assertLessEqual(task["elapsed_s"], 60.0)
            self.assertTrue(Path(
                tmp, "frozen", "level_60_problem_1_sample_0_kernel.py"
            ).is_file())

    def test_two_non_improving_repairs_exit_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            scheduler, verifier = self.run_scripted(
                tmp, "code", "fail")
            task = scheduler.state["tasks"]["1"]
            self.assertEqual(task["status"], "stagnated")
            self.assertEqual(
                task["terminal_reason"], "frontier_no_improvement")
            snapshot = json.loads(Path(
                tmp, "problems/problem_0001/attempt_001/task.json"
            ).read_text())
            self.assertEqual(snapshot["initial_done"], 8)
            self.assertEqual(
                len(snapshot["repair_cycle"]["effective"]), 2)

    def test_no_code_gets_two_fresh_retries_then_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            scheduler, verifier = self.run_scripted(
                tmp, "no_code", "fail")
            task = scheduler.state["tasks"]["1"]
            self.assertEqual(task["status"], "stagnated")
            self.assertEqual(
                task["terminal_reason"], "no_code_after_fresh_retry")
            snapshot = json.loads(Path(
                tmp, "problems/problem_0001/attempt_001/task.json"
            ).read_text())
            self.assertEqual(
                snapshot["repair_cycle"]["kind"], "fresh_retry")
            self.assertEqual(
                len(snapshot["repair_cycle"]["effective"]), 2)

    def test_late_pass_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            scheduler, verifier = self.run_scripted(
                tmp, "code", "late_pass")
            task = scheduler.state["tasks"]["1"]
            self.assertEqual(task["status"], "timeout")
            self.assertIsNone(task.get("solution"))


if __name__ == "__main__":
    unittest.main()
