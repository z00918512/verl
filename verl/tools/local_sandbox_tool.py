# SPDX-License-Identifier: Apache-2.0
"""Local subprocess sandbox tool for multi-turn coding RL.

Wraps Python execution in a subprocess with timeout so the actor can
interactively test code during rollout.  The tool runs the model's code
against a small sample of test cases (stored per-instance at create time)
and returns pass/fail + output for each case.  The *final* episode reward
is computed separately by the reward function (local_sandbox_reward.py),
which runs the last submitted code against all test cases.
"""
import asyncio
import json
import os
import subprocess
import tempfile
import logging
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class LocalSandboxTool(BaseTool):
    """Execute Python code locally via subprocess for multi-turn rollout feedback.

    Per-instance state (test cases) is stored in self._instances keyed by
    instance_id.  The tool is stateless across episodes; create() allocates
    a slot and release() frees it.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._timeout: int = int(config.get("timeout", 10))
        self._sample_size: int = int(config.get("sample_size", 2))
        self._python_bin: str = config.get("python_bin", "python3")
        self._max_output_chars: int = int(config.get("max_output_chars", 500))
        self._instances: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id, resp = await super().create(instance_id)
        test_cases = kwargs.get("test_cases")
        sample_size = int(kwargs.get("sample_size", self._sample_size))
        # test_cases may arrive as a JSON string or a dict
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError:
                test_cases = None
        self._instances[instance_id] = {
            "test_cases": test_cases,
            "sample_size": sample_size,
        }
        return instance_id, resp

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        inst = self._instances.get(instance_id, {})
        test_cases = inst.get("test_cases")
        sample_size = inst.get("sample_size", self._sample_size)

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None, self._run, code, test_cases, sample_size
        )
        return ToolResponse(text=text), None, None

    # ------------------------------------------------------------------
    # Sync execution helpers (run in executor)
    # ------------------------------------------------------------------

    def _run(self, code: str, test_cases: dict | None, sample_size: int) -> str:
        if not test_cases or not test_cases.get("inputs"):
            return self._run_bare(code)

        inputs = test_cases["inputs"][:sample_size]
        outputs = test_cases["outputs"][:len(inputs)]
        results = []
        for i, (inp, exp) in enumerate(zip(inputs, outputs)):
            stdout, stderr, timed_out = self._run_with_stdin(code, str(inp))
            if timed_out:
                results.append(f"Test {i + 1}: TIMEOUT (>{self._timeout}s)")
                continue
            got = stdout.strip()
            expected = str(exp).strip()
            ok = got == expected
            tag = "PASS" if ok else "FAIL"
            entry = f"Test {i + 1}: {tag}"
            if not ok:
                entry += f"\n  Input:    {str(inp)[:120]}"
                entry += f"\n  Expected: {expected[:120]}"
                entry += f"\n  Got:      {got[:120]}"
            if stderr:
                entry += f"\n  Stderr:   {stderr[:200]}"
            results.append(entry)
        return "\n".join(results)

    def _run_bare(self, code: str) -> str:
        """Run without stdin when no test cases are available."""
        stdout, stderr, timed_out = self._run_with_stdin(code, "")
        if timed_out:
            return f"TIMEOUT (>{self._timeout}s)"
        out = stdout[:self._max_output_chars]
        err = stderr[:self._max_output_chars]
        if out and err:
            return f"{out}\n{err}"
        return out or err or "(no output)"

    def _run_with_stdin(self, code: str, stdin: str) -> tuple[str, str, bool]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            proc = subprocess.run(
                [self._python_bin, tmp],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self._timeout,
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
