"""Tokenise a GRPO rollout the same way SFT tokenised a completion.

Kept off torch so the drop-on-overflow contract can be tested on the
workspace host. Must stay in lockstep with CompletionOnlyDataset._encode.
"""


def _as_ids(x):
    """Normalise apply_chat_template output to a flat list of token ids.

    Copied from train_lora.CompletionOnlyDataset: transformers 5.x returns a
    BatchEncoding here rather than a list, and may nest the ids one level.
    """
    if hasattr(x, "input_ids"):
        x = x.input_ids
    elif isinstance(x, dict):
        x = x["input_ids"]
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


def build_sequence(tok, prompt: str, completion: str, max_len: int,
                   chat_kwargs=None, turn_end=None):
    """Tokenise a rollout the way training saw it, and mark the completion.

    `completion` is the sampled assistant text, not the extracted kernel.
    Training on extract_code() used to drop the reasoning tokens the policy
    actually emitted, so the gradient was applied to a string it never
    generated. Must match CompletionOnlyDataset: the same chat template, the
    same turn terminator, and drop rather than front-truncate when the
    sequence does not fit -- truncating would strip the task description.
    """
    prompt_ids = _as_ids(tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True,
        **dict(chat_kwargs or {})))
    end = turn_end if turn_end is not None else tok.eos_token
    if end and not completion.endswith(end):
        completion = completion + end
    completion_ids = tok(completion, add_special_tokens=False)["input_ids"]

    ids = prompt_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    if len(ids) > max_len:
        return None, None
    return ids, mask
