# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face MiniMax-H3 E2E reference backend."""

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
    resolve_owned_file,
    source_revision,
    subprocess_env,
    validate_fixed_profile,
)
from .contracts import E2ECase, RunContext, StageOutput, StageSpec


_GENERATION_STAGES = {"end_to_end", "end_to_end_video", "generate", "frame_quality"}


def _model_snapshot(case: E2ECase) -> Path:
    local = Path(case.hf_id)
    if local.is_dir():
        return local.resolve()
    if not case.hf_revision:
        raise ValueError("MiniMax-H3 E2E requires a pinned hf_revision")

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(case.hf_id, revision=case.hf_revision)).resolve()


class MiniMaxH3HfReference:
    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name not in _GENERATION_STAGES:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported MiniMax-H3 reference stage: {stage.name}"},
            )

        validate_fixed_profile(case)
        output_dir = artifact_dir(ctx, case, "hf_reference")
        python = ctx.reference_python_path() or sys.executable
        revision = source_revision(case, ctx)
        command = [
            python,
            str(MODEL_DIR / "hf_reference.py"),
            "--model-path",
            str(_model_snapshot(case)),
            "--prompt-file",
            str(resolve_owned_file(str(case.inputs["prompt_file"]))),
            "--output-dir",
            str(output_dir),
            "--source-revision",
            revision,
            "--warmup",
            "0",
            "--measure",
            "1",
            "--steps",
            str(case.inputs["num_inference_steps"]),
            "--output-type",
            "np",
        ]
        timeout_s = int(case.metadata.get("reference_timeout_s", 7200))
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env=subprocess_env(ctx),
        )
        elapsed = time.monotonic() - started
        receipt_path = output_dir / "hf_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        frames_path = output_dir / "hf_frames.npy"
        return StageOutput(
            stage_name=stage.name,
            data={
                "returncode": result.returncode,
                "frames_path": str(frames_path) if frames_path.is_file() else "",
                "receipt_path": str(receipt_path) if receipt_path.is_file() else "",
                "receipt": receipt,
                "source_revision": revision,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            text=result.stdout,
            timing_s=elapsed,
            metadata={
                "backend": "hf_diffusers",
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


reference = MiniMaxH3HfReference()
