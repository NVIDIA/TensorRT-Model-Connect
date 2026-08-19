# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face MiniMax-H3 E2E reference backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tensorrt_model_connect.models.minimax_h3.provenance import (
    validate_source_revision,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

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
_DIFFUSERS_REPO_ENV = "TRTMC_MINIMAX_H3_DIFFUSERS_REPO"
_FAMILY_MANIFEST = (
    PROJECT_DIR / "python" / "tensorrt_model_connect" / "models" / "minimax_h3" / "MODEL.toml"
)


def _reference_source_revision(case: E2ECase, ctx: RunContext) -> str:
    if case.metadata.get("validation_sample_id"):
        return validate_source_revision(
            str(case.metadata.get("reference_source_revision", ""))
        )
    return source_revision(case, ctx)


def _reference_environment(ctx: RunContext) -> dict[str, str]:
    environment = subprocess_env(ctx)
    source_value = environment.get(_DIFFUSERS_REPO_ENV, "").strip()
    if not source_value:
        raise ValueError(
            f"MiniMax-H3 HF reference requires {_DIFFUSERS_REPO_ENV} to select "
            "the pinned Diffusers source"
        )
    source = Path(source_value)
    if source.is_symlink():
        raise ValueError("MiniMax-H3 pinned Diffusers source must not be a symlink")
    source = source.resolve(strict=True)
    entrypoint = source / "src" / "diffusers" / "__init__.py"
    if not source.is_dir() or entrypoint.is_symlink() or not entrypoint.is_file():
        raise ValueError("MiniMax-H3 pinned Diffusers source is incomplete")
    existing = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source / "src"), existing) if value
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _reference_evidence_path(ctx: RunContext) -> Path | None:
    if not ctx.artifacts_dir:
        return None
    artifacts_root = Path(ctx.artifacts_dir)
    candidates = (
        artifacts_root / "model-reference-cache.json",
        artifacts_root.parent / "model-reference-cache.json",
    )
    # Preserve the lexical path so the provenance validator can reject evidence
    # supplied through a symlink instead of silently accepting its target.
    matches = [candidate.absolute() for candidate in candidates if candidate.is_file()]
    if len(matches) > 1 and matches[0] != matches[1]:
        raise ValueError("MiniMax-H3 found ambiguous model reference cache evidence")
    return matches[0] if matches else None


def _reference_allow_patterns() -> tuple[str, ...]:
    manifest = tomllib.loads(_FAMILY_MANIFEST.read_text(encoding="utf-8"))
    patterns = manifest.get("hf_allow_patterns")
    if (
        not isinstance(patterns, list)
        or not patterns
        or any(not isinstance(pattern, str) or not pattern for pattern in patterns)
        or len(set(patterns)) != len(patterns)
    ):
        raise ValueError("MiniMax-H3 MODEL.toml requires unique non-empty hf_allow_patterns")
    return tuple(patterns)


def _model_snapshot(case: E2ECase) -> Path:
    local = Path(case.hf_id)
    if local.is_dir():
        return local.resolve()
    if not case.hf_revision:
        raise ValueError("MiniMax-H3 E2E requires a pinned hf_revision")

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            case.hf_id,
            revision=case.hf_revision,
            allow_patterns=_reference_allow_patterns(),
            local_files_only=True,
        )
    ).resolve()


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
        revision = _reference_source_revision(case, ctx)
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
        evidence_path = _reference_evidence_path(ctx)
        if evidence_path is not None:
            command.extend(("--diffusers-evidence", str(evidence_path)))
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
            env=_reference_environment(ctx),
        )
        elapsed = time.monotonic() - started
        receipt_path = output_dir / "hf_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        frames_path = output_dir / "hf_frames.npy"
        frames_dir = output_dir / "frames"
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        data = {
            "returncode": result.returncode,
            "frames_path": str(frames_path) if frames_path.is_file() else "",
            "receipt_path": str(receipt_path) if receipt_path.is_file() else "",
            "receipt": receipt,
            "source_revision": revision,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if frame_paths:
            data.update(
                {
                    "num_frames": len(frame_paths),
                    "frames_dir": str(frames_dir),
                    "frame_paths": [str(path) for path in frame_paths],
                }
            )
        return StageOutput(
            stage_name=stage.name,
            data=data,
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
