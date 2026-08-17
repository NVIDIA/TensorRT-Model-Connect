# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the native DINOv3 image-feature extraction CLI."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .. import case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

PROJECT_DIR = Path(__file__).resolve().parents[6]


def _resolve_image_path(case: E2ECase, ctx: RunContext) -> str:
    image = (
        case.inputs.get("image") or case.inputs.get("test_image") or case.inputs.get("image_path")
    )
    if not image:
        raise ValueError("DINOv3 E2E requires an image input")
    path = Path(str(image))
    if path.is_absolute():
        return str(path)
    for base in (Path(ctx.engine_dir), PROJECT_DIR, PROJECT_DIR / "tests" / "e2e"):
        candidate = base / path
        if candidate.is_file():
            return str(candidate)
    return str(path)


class ImageFeatureExtractionRunner:
    @property
    def strategy_name(self) -> str:
        return "image_feature_extraction"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported DINOv3 TRT stage: {stage.name!r}")
        if not ctx.binary_path:
            raise RuntimeError("DINOv3 E2E requires the trtmc binary")

        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        artifact_dir = Path(case_artifact_dir(artifact_root, case.name))
        output_path = artifact_dir / "trt_image_features.json"
        command = [
            ctx.binary_path,
            "extract-features",
            str(Path(ctx.engine_dir) / case.bundle),
            "--image",
            _resolve_image_path(case, ctx),
            "--output-json",
            str(output_path),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        start = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        elapsed = time.monotonic() - start
        stderr, stderr_path = save_full_stderr(
            completed.stderr or "",
            ctx.artifacts_dir or "",
            "image_feature_extraction",
            case.name,
        )
        if completed.returncode != 0:
            detail = f"DINOv3 extract-features failed (rc={completed.returncode}): {stderr}"
            if stderr_path:
                detail += f" (full stderr: {stderr_path})"
            raise RuntimeError(detail)
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 CLI did not create {output_path}")

        with open(output_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["num_register_tokens"] = int(case.metadata.get("num_register_tokens", 0))
        data["features_json_path"] = str(output_path)
        metadata = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": stderr,
        }
        if stderr_path:
            metadata["stderr_log"] = stderr_path
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )


plugin = ImageFeatureExtractionRunner()
