# SPDX-License-Identifier: Apache-2.0
"""Preprocess open-r1/verifiable-coding-problems-python for GRPO training.

Converts the dataset to the standard verl parquet format with prime_code
test cases as ground truth.  Only stdin/stdout problems are kept (call-based
problems with fn_name are excluded for simplicity).

Usage:
    python examples/data_preprocess/verifiable_code_dataset.py \
        --local_save_dir ~/data/code \
        --train_size 7500 \
        --test_size 1000
"""

import argparse
import json
import os
import random

import datasets

DATA_SOURCE = "open-r1/verifiable-coding-problems-python"

INSTRUCTION = (
    "Write a Python program that reads from standard input and writes to standard output. "
    "Wrap your final solution in a ```python ... ``` code block."
)


def convert_test_cases(verification_info: dict) -> str | None:
    """Convert open-r1 verification_info to prime_code format.

    prime_code expects: {"inputs": [...], "outputs": [...]}
    open-r1 provides:  {"language": "python", "test_cases": [{"fn_name": ..., "input": ..., "output": ...}]}
    """
    tcs = verification_info.get("test_cases", [])
    if not tcs:
        return None
    # Skip call-based problems
    if tcs[0].get("fn_name"):
        return None
    inputs = [tc["input"] for tc in tcs]
    outputs = [tc["output"] for tc in tcs]
    if not inputs:
        return None
    return json.dumps({"inputs": inputs, "outputs": outputs})


def make_map_fn(split: str):
    def process_fn(example, idx):
        ground_truth = convert_test_cases(example["verification_info"])
        if ground_truth is None:
            return None

        problem = example["problem_statement"].strip()
        prompt = problem + "\n\n" + INSTRUCTION

        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "source": example.get("source", ""),
                "problem_id": example.get("problem_id", ""),
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="~/data/code")
    parser.add_argument("--train_size", type=int, default=7500)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {DATA_SOURCE} ...", flush=True)
    ds = datasets.load_dataset(DATA_SOURCE, split="train")
    print(f"Raw size: {len(ds)}")

    # Shuffle with fixed seed for reproducibility
    ds = ds.shuffle(seed=args.seed)

    # Map and filter (returns None for call-based / invalid problems)
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

    # Re-tag splits
    train_ds = train_ds.map(lambda x, i: {**x, "extra_info": {**x["extra_info"], "split": "train", "index": i}}, with_indices=True)
    test_ds = test_ds.map(lambda x, i: {**x, "extra_info": {**x["extra_info"], "split": "test", "index": i}}, with_indices=True)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_ds.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(save_dir, "test.parquet"))

    example = train_ds[0]
    with open(os.path.join(save_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)

    print(f"Saved {len(train_ds)} train + {len(test_ds)} test examples to {save_dir}")
    print(f"Train example prompt[:200]: {example['prompt'][0]['content'][:200]}")
