import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequential_prompts import (  # noqa: E402
    build_adaptive_prompt,
    choose_seed,
    failure_priority,
)
from sequential_search import DynamicChat  # noqa: E402


class SequentialPromptTest(unittest.TestCase):
    def test_round_zero_is_exact_baseline(self):
        self.assertEqual(
            build_adaptive_prompt("BASE PROMPT", [], 0),
            "BASE PROMPT",
        )

    def test_fastest_passing_seed_wins(self):
        history = [
            {"round": 0, "code": "slow", "result": {
                "passed": True, "kernel_ms": 3.0}},
            {"round": 1, "code": "fast", "result": {
                "passed": True, "kernel_ms": 1.0}},
            {"round": 2, "code": "failed", "result": {
                "passed": False, "stage": "exec"}},
        ]
        self.assertEqual(choose_seed(history)["code"], "fast")
        prompt = build_adaptive_prompt("BASE", history, 3)
        self.assertIn("fast", prompt)
        self.assertNotIn("slow", prompt)
        self.assertIn("kernel_ms=1.0", prompt)

    def test_numerically_correct_purity_is_best_failure(self):
        close = {"round": 0, "code": "close", "result": {
            "passed": False, "stage": "purity",
            "impure_correct": True, "error": "torch op left"}}
        compile_error = {"round": 1, "code": "compile", "result": {
            "passed": False, "stage": "exec", "error": "bad shape"}}
        self.assertLess(
            failure_priority(close), failure_priority(compile_error))
        self.assertEqual(
            choose_seed([close, compile_error])["code"], "close")
        prompt = build_adaptive_prompt(
            "BASE", [close, compile_error], 2)
        self.assertIn("close", prompt)
        self.assertIn("correct numbers", prompt)

    def test_latest_failure_breaks_priority_tie(self):
        history = [
            {"round": 0, "code": "old", "result": {
                "passed": False, "stage": "exec"}},
            {"round": 1, "code": "new", "result": {
                "passed": False, "stage": "exec"}},
        ]
        self.assertEqual(choose_seed(history)["code"], "new")

    def test_no_code_starts_over(self):
        history = [{"round": 0, "code": None, "result": {
            "passed": False, "stage": "no_code"}}]
        prompt = build_adaptive_prompt("BASE", history, 1)
        self.assertIn("Start over", prompt)
        self.assertIn("fusion-first", prompt)


class DynamicBudgetTest(unittest.TestCase):
    @staticmethod
    def fake_chat(native, desired, input_tokens):
        chat = DynamicChat.__new__(DynamicChat)
        chat.native_context = native
        chat.desired_max_tokens = desired
        chat.safety_margin = 256
        chat.estimate_input_tokens = lambda messages: input_tokens
        chat._request = lambda messages, estimated, maximum: {
            "text": "ok",
            "input_tokens_estimate": estimated,
            "max_tokens_sent": maximum,
            "finish_reason": "stop",
            "prompt_tokens": estimated,
            "completion_tokens": 1,
            "desired_max_tokens": desired,
            "native_context": native,
            "safety_margin": 256,
            "hit_token_limit": False,
            "error": None,
        }
        return chat

    def test_gl_native_context_clips_128k_output(self):
        chat = self.fake_chat(131072, 131072, 3000)
        row = chat.complete([{"role": "user", "content": "x"}])
        self.assertEqual(row["max_tokens_sent"], 127816)

    def test_256k_model_keeps_full_output_cap(self):
        chat = self.fake_chat(262144, 131072, 3000)
        row = chat.complete([{"role": "user", "content": "x"}])
        self.assertEqual(row["max_tokens_sent"], 131072)

    def test_context_exhaustion_is_recorded(self):
        chat = self.fake_chat(1000, 1000, 900)
        row = chat.complete([{"role": "user", "content": "x"}])
        self.assertEqual(row["finish_reason"], "context_exhausted")
        self.assertEqual(row["max_tokens_sent"], 0)
        self.assertTrue(row["hit_token_limit"])

    def test_chat_template_batch_encoding_counts_input_ids(self):
        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1] * 4]}

        chat = DynamicChat.__new__(DynamicChat)
        chat.tokenizer = FakeTokenizer()
        self.assertEqual(
            chat.estimate_input_tokens(
                [{"role": "user", "content": "x"}]),
            4,
        )

    def test_chat_template_object_input_ids(self):
        class Encoded:
            input_ids = [[1, 2, 3]]

        class FakeTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return Encoded()

        chat = DynamicChat.__new__(DynamicChat)
        chat.tokenizer = FakeTokenizer()
        self.assertEqual(
            chat.estimate_input_tokens(
                [{"role": "user", "content": "x"}]),
            3,
        )


if __name__ == "__main__":
    unittest.main()
