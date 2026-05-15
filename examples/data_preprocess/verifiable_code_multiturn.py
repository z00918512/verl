# SPDX-License-Identifier: Apache-2.0
"""Preprocess open-r1/verifiable-coding-problems-python for multi-turn GRPO.

Produces the same stdin/stdout split as verifiable_code_dataset.py but adds
the fields required by the tool_agent loop:
  - agent_name: "tool_agent"
  - extra_info.need_tools_kwargs: True
  - extra_info.tools_kwargs.code_interpreter.create_kwargs.test_cases
      → first SAMPLE_CASES test cases, passed to LocalSandboxTool.create()
      → the tool uses these for interactive per-turn feedback
  - reward_model.ground_truth → ALL test cases (JSON), used by the final
      reward function (local_sandbox_reward.py) after the episode ends

Usage:
    python examples/data_preprocess/verifiable_code_multiturn.py \
        --local_save_dir ~/data/code_multiturn \
        --train_size 7500 \
        --test_size 1000
"""

import argparse
import json
import os

import datasets

DATA_SOURCE = "open-r1/verifiable-coding-problems-python-multiturn"
_RAW_SOURCE = "open-r1/verifiable-coding-problems-python"

SAMPLE_CASES = 2  # test cases surfaced to the model during interactive execution

SYSTEM_PROMPT = (
    "You are a competitive programmer. You will be given a programming problem "
    "that reads from standard input and writes to standard output.\n\n"
    "You have access to a code execution tool. Use it to test your solution "
    "iteratively:\n"
    "1. Write a Python solution.\n"
    "2. Call code_interpreter to test it with sample inputs.\n"
    "3. Fix any issues based on the PASS/FAIL feedback.\n"
    "4. Repeat until all sample tests pass.\n\n"
    "When you are confident your solution is correct, output it wrapped in a "
    "```python ... ``` code block as your final answer."
)


def convert_test_cases(verification_info: dict) -> dict | None:
    """Return {'inputs': [...], 'outputs': [...]} or None for call-based problems."""
    tcs = verification_info.get("test_cases", [])
    if not tcs or tcs[0].get("fn_name"):
        return None
    inputs = [tc["input"] for tc in tcs]
    outputs = [tc["output"] for tc in tcs]
    if not inputs:
        return None
    return {"inputs": inputs, "outputs": outputs}


def make_map_fn(split: str):
    def process_fn(example, idx):
        test_cases = convert_test_cases(example["verification_info"])
        if test_cases is None:
            return None

        problem = example["problem_statement"].strip()

        sample_cases = {
            "inputs": test_cases["inputs"][:SAMPLE_CASES],
            "outputs": test_cases["outputs"][:SAMPLE_CASES],
        }

        return {
            "data_source": DATA_SOURCE,
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                # ALL test cases for final reward scoring
                "ground_truth": json.dumps(test_cases),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "source": example.get("source", ""),
                "problem_id": example.get("problem_id", ""),
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "code_interpreter": {
                        # Passed to LocalSandboxTool.create() so the tool
                        # can show sample feedback during rollout.
                        "create_kwargs": {
                            "test_cases": sample_cases,
                            "sample_size": SAMPLE_CASES,
                        },
                    },
                },
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/code_multiturn")
    parser.add_argument("--train_size", type=int, default=7500)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {_RAW_SOURCE} ...", flush=True)
    ds = datasets.load_dataset(_RAW_SOURCE, split="train")
    print(f"Raw size: {len(ds)}")

    ds = ds.shuffle(seed=args.seed)
    ds_mapped = ds.map(
        make_map_fn("all"),
        with_indices=True,
        remove_columns=ds.column_names,
    )
    ds_mapped = ds_mapped.filter(lambda x: x is not None and x.get("prompt") is not None)
    print(f"After filter (stdin/stdout only): {len(ds_mapped)}")

    train_size = min(args.train_size, len(ds_mapped) - args.test_size)
    test_size = args.test_size

    train_ds = ds_mapped.select(range(train_size))
    test_ds = ds_mapped.select(range(train_size, train_size + test_size))

    train_ds = train_ds.map(
        lambda x, i: {**x, "extra_info": {**x["extra_info"], "split": "train", "index": i}},
        with_indices=True,
    )
    test_ds = test_ds.map(
        lambda x, i: {**x, "extra_info": {**x["extra_info"], "split": "test", "index": i}},
        with_indices=True,
    )

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(save_dir, "test.parquet"))

    example = train_ds[0]
    with open(os.path.join(save_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    print(f"Saved {len(train_ds)} train + {len(test_ds)} test to {save_dir}")
    print(f"System prompt (first 120 chars): {SYSTEM_PROMPT[:120]}")
    print(f"Train example problem (first 200 chars): {example['prompt'][1]['content'][:200]}")
