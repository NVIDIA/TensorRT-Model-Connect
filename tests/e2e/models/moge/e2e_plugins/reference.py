# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official MoGe-2 PyTorch reference."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec

_SOURCE_ENV = "TRTMC_MOGE_SOURCE_DIR"


def _checkpoint_snapshot(case: E2ECase, *, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    if not case.hf_revision:
        raise ValueError("MoGe reference requires an immutable hf_revision")
    return Path(
        snapshot_download(
            case.hf_id,
            revision=case.hf_revision,
            allow_patterns=["model.pt"],
            local_files_only=local_files_only,
        )
    )


class MoGeTorchReference:
    @property
    def backend_name(self) -> str:
        return "moge_torch"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported MoGe reference stage: {stage.name!r}")
        num_tokens = int(case.inputs.get("num_tokens", 0))
        if num_tokens != 1800:
            raise ValueError(f"MoGe reference requires num_tokens=1800, got {num_tokens}")
        image = Path(str(case.inputs.get("image", "")))
        if not image.is_file():
            raise FileNotFoundError(f"MoGe reference input image is missing: {image}")
        source_root = Path(os.environ.get(_SOURCE_ENV, ""))
        if not (source_root / "moge/model/v2.py").is_file():
            raise FileNotFoundError(
                f"{_SOURCE_ENV} must point at the pinned Microsoft MoGe source"
            )
        checkpoint = _checkpoint_snapshot(case, local_files_only=ctx.local_files_only)

        artifact_dir = Path(ctx.artifacts_dir or "/tmp") / case.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_dir / "moge_torch_geometry.npz"
        output_path.unlink(missing_ok=True)
        script = Path(__file__).resolve().parents[1] / "official_reference.py"
        command = [
            ctx.reference_python_path() or sys.executable,
            str(script),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint / "model.pt"),
            "--image",
            str(image),
            "--output",
            str(output_path),
            "--num-tokens",
            str(num_tokens),
        ]
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        elapsed = time.monotonic() - started
        data: dict = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "num_tokens": num_tokens,
        }
        if completed.returncode == 0 and output_path.is_file():
            with np.load(output_path, allow_pickle=False) as payload:
                data.update(
                    points=np.array(payload["points"], copy=True),
                    depth=np.array(payload["depth"], copy=True),
                    mask=np.array(payload["mask"], copy=True),
                    intrinsics=np.array(payload["intrinsics"], copy=True),
                    height=int(payload["height"]),
                    width=int(payload["width"]),
                    output_path=str(output_path),
                )
        elif completed.returncode == 0:
            data["output_error"] = f"MoGe reference exited 0 but did not create {output_path}"
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "backend": self.backend_name,
                "command": command,
                "returncode": completed.returncode,
                "hf_id": case.hf_id,
                "hf_revision": case.hf_revision,
                "source_revision": "74fbce054ebed49800de42d0ad0e83495065719a",
            },
        )


reference = MoGeTorchReference()
