# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MiniMax-H3 model-owned E2E plugins."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct

from tests.e2e_harness.contracts import E2ECase, RunContext


MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = MODEL_DIR.parents[3]
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLAN_FILENAMES = {
    "text_encoder.plan",
    "adaln_precompute.plan",
    "denoiser_cp.plan",
    "vae_tile_decoder.plan",
}


def artifact_dir(ctx: RunContext, case: E2ECase, name: str) -> Path:
    root = Path(ctx.artifacts_dir) if ctx.artifacts_dir else Path("/tmp/e2e_artifacts")
    output = root / case.name / name
    output.mkdir(parents=True, exist_ok=True)
    return output


def resolve_owned_file(value: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [PROJECT_DIR / path, MODEL_DIR / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"MiniMax-H3 E2E input file does not exist: {value}")


def bundle_path(case: E2ECase, ctx: RunContext) -> Path:
    path = Path(case.bundle)
    return path if path.is_absolute() else Path(ctx.engine_dir) / path


def validate_fixed_profile(case: E2ECase) -> None:
    expected = {
        "video_num_frames": 124,
        "video_height": 768,
        "video_width": 1344,
        "num_inference_steps": 50,
        "fps": 24,
    }
    mismatches = {
        name: (case.inputs.get(name), value)
        for name, value in expected.items()
        if case.inputs.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 fixed E2E profile: {mismatches}")


def _bundle_config(path: Path) -> dict:
    with path.open("rb") as bundle:
        if bundle.read(8) != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid TRTMC bundle: {path}")
        header_length = struct.unpack("<Q", bundle.read(8))[0]
        header = json.loads(bundle.read(header_length).decode("utf-8"))
        section = header.get("sections", {}).get("config.json")
        if not isinstance(section, dict):
            return {}
        bundle.seek(16 + header_length + int(section["offset"]))
        return json.loads(bundle.read(int(section["size"])).decode("utf-8"))


def source_revision(case: E2ECase, ctx: RunContext) -> str:
    """Resolve the exact source revision recorded by both E2E backends."""

    path = bundle_path(case, ctx)
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 E2E bundle does not exist: {path}")
    config = _bundle_config(path)
    revision = str(config.get("source_revision", "")).strip().lower()
    if _SOURCE_REVISION.fullmatch(revision) is None:
        raise ValueError("MiniMax-H3 bundle has no valid source_revision")
    if _SHA256.fullmatch(str(config.get("builder_source_sha256", ""))) is None:
        raise ValueError("MiniMax-H3 bundle has no valid builder_source_sha256")
    if _SHA256.fullmatch(str(config.get("checkpoint_inventory_sha256", ""))) is None:
        raise ValueError("MiniMax-H3 bundle has no valid checkpoint_inventory_sha256")
    plan_sha = config.get("plan_sha256")
    if not isinstance(plan_sha, dict) or set(plan_sha) != _PLAN_FILENAMES:
        raise ValueError("MiniMax-H3 bundle does not identify exactly all four native plans")
    if any(_SHA256.fullmatch(str(value)) is None for value in plan_sha.values()):
        raise ValueError("MiniMax-H3 bundle contains an invalid native plan SHA256")

    explicit_revision = os.environ.get("TRTMC_MINIMAX_H3_SOURCE_REVISION", "").strip().lower()
    if explicit_revision:
        if _SOURCE_REVISION.fullmatch(explicit_revision) is None:
            raise ValueError("TRTMC_MINIMAX_H3_SOURCE_REVISION is not an exact Git SHA")
        if explicit_revision != revision:
            raise ValueError(
                "MiniMax-H3 bundle source_revision does not match TRTMC_MINIMAX_H3_SOURCE_REVISION"
            )
    return revision


def model_plugin_dir(ctx: RunContext) -> Path:
    candidates = []
    if ctx.model_plugin_dir:
        candidates.append(Path(ctx.model_plugin_dir))
    candidates.append(PROJECT_DIR / "build" / "models" / "minimax_h3")
    for candidate in candidates:
        if (candidate / "libtrtmc_model_minimax_h3.so").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MiniMax-H3 E2E requires libtrtmc_model_minimax_h3.so via "
        "--model-plugin-dir or build/models/minimax_h3"
    )


def subprocess_env(ctx: RunContext) -> dict[str, str]:
    env = os.environ.copy()
    python_path = str(PROJECT_DIR / "python")
    if env.get("PYTHONPATH"):
        python_path = f"{python_path}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = python_path
    if ctx.ld_library_path:
        env["LD_LIBRARY_PATH"] = ctx.ld_library_path
    return env
