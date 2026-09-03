# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native recorded-observation runner for LeRobot ACT action chunks."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def _bundle_path(case: E2ECase, context: RunContext) -> str:
    bundle = case.bundle or f"{case.name}.bundle"
    return bundle if os.path.isabs(bundle) else os.path.join(context.engine_dir, bundle)


def _prepare(case: E2ECase, ctx: RunContext, directory: Path) -> tuple[Path, Path, dict]:
    script = Path(__file__).resolve().parents[1] / "prepare_recorded_observation.py"
    command = [
        ctx.runtime_python_path(),
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
    return directory / "observation.images.top.png", directory / "observation.state.f32", metadata


class RobotActionChunkRunner:
    @property
    def strategy_name(self) -> str:
        return "robot_action_chunk"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "recorded_action_chunk":
            raise ValueError(f"Unsupported LeRobot ACT runtime stage: {stage.name!r}")
        artifact_dir = Path(ctx.artifacts_dir or "/tmp") / case.name
        replay_dir = artifact_dir / "recorded_observation"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        image, state, replay_metadata = _prepare(case, ctx, replay_dir)
        actions_path = artifact_dir / "trt_actions.f32"
        actions_path.unlink(missing_ok=True)
        command = [
            ctx.binary_path,
            "act",
            _bundle_path(case, ctx),
            "--image",
            str(image),
            "--state",
            str(state),
            "--output",
            str(actions_path),
            "--benchmark",
            "10",
            "--warmup",
            "2",
            "--control-hz",
            str(case.inputs["control_frequency_hz"]),
        ]
        if ctx.model_plugin_dir:
            command.extend(("--model-plugin-dir", ctx.model_plugin_dir))
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
        if completed.returncode == 0:
            try:
                summary = json.loads(completed.stdout)
                actions = np.fromfile(actions_path, dtype="<f4").reshape(100, 14)
                data.update(summary=summary, actions=actions, actions_path=str(actions_path))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                data["output_error"] = str(error)
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": command, "returncode": completed.returncode},
        )


runner = RobotActionChunkRunner()
