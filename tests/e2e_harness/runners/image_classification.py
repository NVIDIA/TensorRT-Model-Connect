"""Image-classification strategy runner."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[3]


class ImageClassificationRunner:
    @property
    def strategy_name(self) -> str:
        return "image_classification"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        image_path = self._resolve_image_path(case, ctx)
        cmd = [
            ctx.binary_path,
            "classify",
            bundle_path,
            "--image",
            image_path or "",
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running image classification: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600)
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "image_classification", case.name)
            msg = (
                f"Image classification failed (rc={result.returncode}): "
                f"{truncated}"
            )
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        try:
            data = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            data = {"raw_output": result.stdout.strip()}

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            },
        )

    def _resolve_image_path(self, case: E2ECase, ctx: RunContext) -> str | None:
        image = (
            case.inputs.get("image") or case.inputs.get("test_image")
            or case.inputs.get("image_path")
        )
        if not image:
            return None
        path = Path(image)
        if path.is_absolute():
            return str(path)
        for base in (ctx.engine_dir, str(PROJECT_DIR), str(PROJECT_DIR / "tests" / "e2e")):
            candidate = Path(base) / image
            if candidate.is_file():
                return str(candidate)
        return str(path)


plugin = ImageClassificationRunner()
