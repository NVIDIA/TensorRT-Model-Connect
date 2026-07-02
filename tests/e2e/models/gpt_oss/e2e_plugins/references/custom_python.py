# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom Python reference backend — execute a user-provided Python script.

The script path is specified in case metadata as ``custom_python_script``.
Flexible for non-standard models that don't fit HF Transformers/Diffusers.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


class CustomPythonReference:
    """Execute a custom Python script as reference backend."""

    @property
    def backend_name(self) -> str:
        return "custom_python"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        script_path = case.metadata.get("custom_python_script")
        if not script_path:
            raise ValueError(
                f"Case {case.name} uses custom_python reference but "
                f"metadata.custom_python_script is not set"
            )

        # Resolve script path relative to project root
        if not os.path.isabs(script_path):
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            script_path = os.path.join(project_root, script_path)

        python = ctx.reference_python_path() or "python3"

        cmd = [
            python, script_path,
            "--model", case.hf_id,
            "--stage", stage.name,
        ]

        # Pass prompt if available
        prompt = case.inputs.get("prompt", "")
        if prompt:
            cmd.extend(["--prompt", prompt])

        # Pass image if available
        image = case.inputs.get("image")
        if image:
            cmd.extend(["--image", image])

        # Pass audio if available
        audio = case.inputs.get("audio")
        if audio:
            cmd.extend(["--audio", audio])

        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

        # Pass trust_remote_code if needed
        if case.metadata.get("trust_remote_code"):
            cmd.append("--trust-remote-code")

        # Pass any extra script args from metadata
        extra_args = case.metadata.get("custom_python_args", [])
        cmd.extend(extra_args)

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running custom Python reference: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=1800,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "custom_python", case.name)
            msg = (f"Custom Python reference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        # Parse output: expect JSON on stdout
        data = _parse_output(result.stdout.strip())

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data.get("text"),
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "script_path": script_path,
            },
        )


def _parse_output(stdout: str) -> dict:
    """Parse reference output from custom Python script stdout."""
    # Try JSON (preferred format)
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Try parsing last line as JSON (script may print progress before JSON)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            continue

    # Fall back to raw text output
    return {"text": stdout, "raw_output": stdout}


plugin = CustomPythonReference()
