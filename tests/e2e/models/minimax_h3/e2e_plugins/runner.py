# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-device native MiniMax-H3 E2E runner."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from . import (
    MODEL_DIR,
    PROJECT_DIR,
    artifact_dir,
    bundle_path,
    model_plugin_dir,
    resolve_owned_file,
    source_revision,
    subprocess_env,
    validate_fixed_profile,
)
from .contracts import E2ECase, RunContext, StageOutput, StageSpec


_GENERATION_STAGES = {"end_to_end", "end_to_end_video", "generate", "frame_quality"}


def build_native_command(
    case: E2ECase,
    ctx: RunContext,
    output_dir: Path,
    *,
    resolved_bundle: Path | None = None,
) -> list[str]:
    validate_fixed_profile(case)
    python = ctx.runtime_python_path() or sys.executable
    return [
        python,
        str(MODEL_DIR / "native_reference.py"),
        "--bundle",
        str(resolved_bundle or bundle_path(case, ctx)),
        "--prompt-file",
        str(resolve_owned_file(str(case.inputs["prompt_file"]))),
        "--trtf",
        ctx.binary_path,
        "--plugin-dir",
        str(model_plugin_dir(ctx)),
        "--output-dir",
        str(output_dir),
        "--source-revision",
        source_revision(case, ctx),
    ]


class MiniMaxH3NativeRunner:
    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        resolved_bundle: str,
    ) -> list[str] | None:
        if case.task_strategy != self.strategy_name:
            return None
        return build_native_command(
            case,
            ctx,
            Path("/tmp/trtmc_minimax_h3_native"),
            resolved_bundle=Path(resolved_bundle),
        )

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name not in _GENERATION_STAGES:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported MiniMax-H3 stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )

        output_dir = artifact_dir(ctx, case, "trt_native")
        command = build_native_command(case, ctx, output_dir)
        timeout_s = int(case.metadata.get("runtime_timeout_s", 7200))
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env={
                **subprocess_env(ctx),
                "TRTMC_MODEL_PLUGIN_DIR": str(model_plugin_dir(ctx)),
                "TRTMC_MODEL_PLUGIN_STRICT": "1",
            },
        )
        elapsed = time.monotonic() - started
        receipt_path = output_dir / "trt_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        frames_path = output_dir / "trt_frames.npy"
        frames_dir = output_dir / "frames"
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        data = {
            "returncode": result.returncode,
            "num_frames": len(frame_paths),
            "frames_dir": str(frames_dir),
            "frame_paths": [str(path) for path in frame_paths],
            "frames_path": str(frames_path) if frames_path.is_file() else "",
            "receipt_path": str(receipt_path) if receipt_path.is_file() else "",
            "receipt": receipt,
            "source_revision": source_revision(case, ctx),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


runner = MiniMaxH3NativeRunner()
