# SPDX-License-Identifier: Apache-2.0
"""Final-episode reward for multi-turn coding RL with local sandbox execution.

Extracts the last ```python ... ``` code block from solution_str (which may be
the full multi-turn conversation or just the last assistant message) and runs it
against ALL test cases from ground_truth using subprocess.

Reward = fraction of test cases passed (0.0–1.0), same metric as the single-turn
prime_code reward so results are directly comparable.

Usage (via verl custom_reward_function config):
    custom_reward_function.path=scripts/local_sandbox_reward.py
    custom_reward_function.name=compute_score
"""
import json
import logging
import os
import re
import subprocess
import tempfile

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_TIMEOUT = 10  # seconds per test case
_MAX_TEST_CASES = 10  # cap to keep reward latency bounded
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_last_code_block(text: str) -> str | None:
    """Return the last ```python ... ``` block in text, or None."""
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _run_code(code: str, stdin: str, timeout: int) -> tuple[str, str, bool]:
    """Run code with stdin, return (stdout, stderr, timed_out)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            ["python3", tmp],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired:
        return "", "", True
    except Exception as e:
        return "", str(e), False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def compute_score(
    data_source,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
) -> float:
    """Score the last code block in solution_str against all test cases.

    Args:
        data_source: Unused; present for verl reward function interface compat.
        solution_str: The model's full response (single-turn) or full
            conversation (multi-turn). The last ```python...``` block is used.
        ground_truth: JSON string {"inputs": [...], "outputs": [...]}.
        extra_info: Passed by verl; unused here.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        score ∈ [0, 1], fraction of test cases passed.
    """
    # 1. Extract code
    code = _extract_last_code_block(solution_str)
    if code is None:
        return 0.0

    # 2. Parse test cases
    if isinstance(ground_truth, dict):
        test_cases = ground_truth
    else:
        try:
            test_cases = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return 0.0

    inputs = test_cases.get("inputs", [])[:_MAX_TEST_CASES]
    outputs = test_cases.get("outputs", [])[:len(inputs)]
    if not inputs:
        return 0.0

    # 3. Run against each test case
    passed = 0
    metadata = []
    for i, (inp, exp) in enumerate(zip(inputs, outputs)):
        stdout, stderr, timed_out = _run_code(code, str(inp), _TIMEOUT)
        if timed_out:
            metadata.append({"test": i, "status": "timeout"})
            continue
        ok = stdout.strip() == str(exp).strip()
        if ok:
            passed += 1
        metadata.append({
            "test": i,
            "status": "pass" if ok else "fail",
            "stderr": stderr[:200] if stderr else None,
        })

    score = passed / len(inputs)
    logger.info("local_sandbox_reward: %d/%d tests passed (score=%.3f)", passed, len(inputs), score)
    return float(score)
