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

from tests.e2e_harness.contracts import (
    E2ECase,
    ReproCommandProvider,
    RunContext,
    StageOutput,
    StageSpec,
)

from . import case_artifact_dir, image_input, resolve_image_path, save_full_stderr

PROJECT_DIR = Path(__file__).resolve().parents[6]


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

        artifact_dir = Path(
            case_artifact_dir(ctx.artifacts_dir or tempfile.gettempdir(), case.name)
        )
        output_path = artifact_dir / "trt_image_features.json"
        command = [
            ctx.binary_path,
            "extract-features",
            str(Path(ctx.engine_dir) / case.bundle),
            "--image",
            resolve_image_path(
                case,
                (Path(ctx.engine_dir), PROJECT_DIR, PROJECT_DIR / "tests" / "e2e"),
                "DINOv3 E2E requires an image input",
            ),
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
        if completed.returncode:
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


class Dinov3ReproCommandProvider:
    @property
    def family_name(self) -> str:
        return "dinov3"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        image = image_input(case)
        if case.task_strategy != "image_feature_extraction" or not image:
            return None
        output = Path(tempfile.gettempdir()) / f"{case.name}-image-features.json"
        command = [
            ctx.binary_path,
            "extract-features",
            bundle_path,
            "--image",
            image,
            "--output-json",
            str(output),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        return command


runner = ImageFeatureExtractionRunner()
repro_provider: ReproCommandProvider = Dinov3ReproCommandProvider()
