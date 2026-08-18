# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native five-frame HOI video-tracking runner."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .. import case_artifact_dir
from .._schema import (
    FRAME_COUNT,
    load_npz_arrays,
    normalize_runtime_json,
    resolve_project_path,
    structured_summary,
    validate_dimensions,
)
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class HoiVideoTrackingRunner:
    """Run the family-owned ``track-hoi`` CLI and normalize its output."""

    @property
    def strategy_name(self) -> str:
        return "hoi_video_tracking"

    @staticmethod
    def _bundle_path(case: E2ECase, ctx: RunContext) -> Path:
        bundle = Path(case.bundle or f"{case.name}.bundle")
        return bundle if bundle.is_absolute() else Path(ctx.engine_dir) / bundle

    @staticmethod
    def _frames_dir(case: E2ECase) -> Path:
        value = case.inputs.get("frames_dir")
        if not isinstance(value, str) or not value:
            raise ValueError("SAM2 HOI E2E input frames_dir is required")
        frames_dir = resolve_project_path(value)
        expected = [frames_dir / f"{index:06d}.jpg" for index in range(FRAME_COUNT)]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"SAM2 HOI requires exactly the archive's five JPEG inputs; missing={missing}"
            )
        extras = sorted(path.name for path in frames_dir.glob("*.jpg") if path not in expected)
        if extras:
            raise ValueError(f"SAM2 HOI frames_dir contains unexpected JPEGs: {extras}")
        return frames_dir

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str]:
        frames_dir = self._frames_dir(case)
        artifact_dir = case_artifact_dir(ctx.artifacts_dir, case.name)
        return [
            ctx.binary_path,
            "track-hoi",
            bundle_path,
            "--frames-dir",
            str(frames_dir),
            "--output-json",
            str(artifact_dir / "trt_tracking.json"),
            "--output-masks-dir",
            str(artifact_dir / "trt_masks"),
        ]

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_tracking":
            raise ValueError(f"Unsupported SAM2 HOI E2E stage: {stage.name}")
        binary = Path(ctx.binary_path)
        if not ctx.binary_path or not binary.is_file():
            raise FileNotFoundError(f"TensorRT-Model-Connect binary not found: {binary}")
        bundle = self._bundle_path(case, ctx)
        if not bundle.is_file():
            raise FileNotFoundError(f"SAM2 HOI bundle not found: {bundle}")

        command = self.build_trt_inference_command(case, ctx, str(bundle))
        environment = dict(os.environ)
        if ctx.ld_library_path:
            environment["LD_LIBRARY_PATH"] = ctx.ld_library_path
        if ctx.model_plugin_dir:
            environment["TRTMC_MODEL_PLUGIN_DIR"] = ctx.model_plugin_dir
            environment["TRTMC_MODEL_PLUGIN_STRICT"] = "1"

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("SAM2 HOI track-hoi command timed out") from error
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise RuntimeError(
                "SAM2 HOI requires the native track-hoi CLI; "
                f"command failed with rc={completed.returncode}: {detail}"
            )

        output_json = Path(command[command.index("--output-json") + 1])
        output_npz = output_json.with_name("trt_tracking.npz")
        normalize_runtime_json(output_json, output_npz)
        arrays = load_npz_arrays(output_npz)
        validate_dimensions(
            arrays,
            height=int(case.inputs.get("expected_height", 1280)),
            width=int(case.inputs.get("expected_width", 1088)),
        )
        return StageOutput(
            stage_name=stage.name,
            data={
                "schema_version": 1,
                "frame_count": FRAME_COUNT,
                "frames": structured_summary(arrays),
                "output_path": str(output_npz),
                "output_npz": str(output_npz),
                "returncode": completed.returncode,
            },
            timing_s=elapsed,
            metadata={
                "command": command,
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[-2000:],
                "stderr": (completed.stderr or "")[-2000:],
            },
        )


plugin = HoiVideoTrackingRunner()
