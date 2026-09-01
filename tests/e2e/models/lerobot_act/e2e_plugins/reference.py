# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned exact-source LeRobot ACT PyTorch reference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tensorrt_model_connect.families.lerobot_act.plugin import LEROBOT_REVISION
from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec

_SOURCE_ENV = "TRTMC_LEROBOT_SOURCE_DIR"


def _snapshot(case: E2ECase, *, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    if not case.hf_revision:
        raise ValueError("LeRobot ACT reference requires an immutable hf_revision")
    return Path(
        snapshot_download(
            case.hf_id,
            revision=case.hf_revision,
            allow_patterns=["config.json", "model.safetensors"],
            local_files_only=local_files_only,
        )
    )


def _prepare(case: E2ECase, ctx: RunContext, directory: Path) -> tuple[Path, Path, dict]:
    script = Path(__file__).resolve().parents[1] / "prepare_recorded_observation.py"
    command = [
        ctx.reference_python_path() or sys.executable,
        str(script),
        "--output",
        str(directory),
        "--episode-index",
        str(case.inputs["episode_index"]),
        "--frame-index",
        str(case.inputs["frame_index"]),
    ]
    if ctx.local_files_only:
        command.append("--local-files-only")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if completed.returncode != 0:
        raise RuntimeError(f"recorded observation preparation failed: {completed.stderr}")
    metadata = json.loads((directory / "recorded_observation.json").read_text(encoding="utf-8"))
    return (
        directory / "observation.images.top.png",
        directory / "observation.state.f32",
        metadata,
    )


class LeRobotActTorchReference:
    @property
    def backend_name(self) -> str:
        return "lerobot_act_torch"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "recorded_action_chunk":
            raise ValueError(f"Unsupported LeRobot ACT reference stage: {stage.name!r}")
        source_root = Path(os.environ.get(_SOURCE_ENV, ""))
        if not (source_root / "lerobot/common/policies/act/modeling_act.py").is_file():
            raise FileNotFoundError(f"{_SOURCE_ENV} must point at the pinned LeRobot source")
        checkpoint_dir = _snapshot(case, local_files_only=ctx.local_files_only)
        artifact_dir = Path(ctx.artifacts_dir or "/tmp") / case.name
        replay_dir = artifact_dir / "recorded_observation"
        image, state, replay_metadata = _prepare(case, ctx, replay_dir)
        output = artifact_dir / "lerobot_torch_actions.npz"
        output.unlink(missing_ok=True)
        script = Path(__file__).resolve().parents[1] / "official_reference.py"
        command = [
            ctx.reference_python_path() or sys.executable,
            str(script),
            "--source-root",
            str(source_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--image",
            str(image),
            "--state",
            str(state),
            "--output",
            str(output),
        ]
        environment = dict(os.environ)
        if ctx.ld_library_path:
            environment["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800,
            env=environment,
        )
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "recorded_observation": replay_metadata,
        }
        if completed.returncode == 0 and output.is_file():
            with np.load(output, allow_pickle=False) as payload:
                data.update(
                    actions=np.array(payload["actions"], copy=True),
                    actions_path=str(output),
                )
        elif completed.returncode == 0:
            data["output_error"] = f"reference exited 0 without creating {output}"
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "command": command,
                "returncode": completed.returncode,
                "hf_revision": case.hf_revision,
                "source_revision": LEROBOT_REVISION,
            },
        )


reference = LeRobotActTorchReference()
