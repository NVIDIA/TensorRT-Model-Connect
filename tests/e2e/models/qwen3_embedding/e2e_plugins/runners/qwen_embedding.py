# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-family ``trtmc embed`` E2E runner."""

from __future__ import annotations

import json
import os
import subprocess
import time

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


def parse_embedding(stdout: str) -> list[float]:
    try:
        value = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen embedding runtime did not emit JSON") from exc
    if isinstance(value, dict):
        value = value.get("embedding")
    if not isinstance(value, list) or not value:
        raise ValueError("Qwen embedding runtime emitted no vector")
    return [float(item) for item in value]


class EmbeddingRunner:
    @property
    def strategy_name(self) -> str:
        return "embedding"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        bundle = os.path.join(ctx.engine_dir, case.bundle)
        command = [ctx.binary_path, "embed", bundle, "--prompt", case.inputs["prompt"]]
        if ctx.runtime_cli_hf_python():
            command.extend(["--hf-python", ctx.runtime_cli_hf_python()])
        environment = dict(os.environ)
        if ctx.ld_library_path:
            environment["LD_LIBRARY_PATH"] = ctx.ld_library_path

        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=600,
        )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise RuntimeError(
                f"Qwen embedding inference failed (rc={result.returncode}): {result.stderr[-4000:]}"
            )
        return StageOutput(
            stage_name=stage.name,
            data={"embedding": parse_embedding(result.stdout)},
            timing_s=elapsed,
            metadata={
                "command": command,
                "returncode": result.returncode,
                "stderr": result.stderr,
            },
        )


plugin = EmbeddingRunner()
