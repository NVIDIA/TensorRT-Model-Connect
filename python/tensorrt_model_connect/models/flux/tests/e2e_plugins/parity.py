# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared deterministic inputs for Flux HF-to-TRTMC parity."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import E2ECase, RunContext


@dataclass(frozen=True)
class InitialLatents:
    path: Path
    sha256: str
    shape: tuple[int, int, int, int]
    is_flux2: bool


def _parity_root(ctx: RunContext) -> Path:
    if not ctx.artifacts_dir:
        return Path(tempfile.gettempdir()) / "trtmc_flux_parity"
    artifacts_dir = Path(ctx.artifacts_dir)
    if artifacts_dir.name in {"hf_artifacts", "bundle_artifacts"}:
        return artifacts_dir.parent / "shared_initial_latents"
    return artifacts_dir / "shared_initial_latents"


def ensure_initial_latents(case: E2ECase, ctx: RunContext) -> InitialLatents:
    height = int(case.inputs.get("image_height", 1024))
    width = int(case.inputs.get("image_width", height))
    if height <= 0 or width <= 0 or height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"Flux parity dimensions must be positive and divisible by 16: {height}x{width}"
        )

    model_type = str(case.metadata.get("model_type", "")).lower()
    is_flux2 = model_type in {"flux.2", "flux2"}
    shape = (
        (1, 128, height // 16, width // 16)
        if is_flux2
        else (1, 16, height // 8, width // 8)
    )
    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    latents = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)
    output_dir = _parity_root(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = "flux2" if is_flux2 else "flux1"
    output_path = output_dir / f"{case.name}.{variant}.seed-{seed}.{height}x{width}.f32"
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    latents.tofile(temporary_path)
    os.replace(temporary_path, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return InitialLatents(
        path=output_path,
        sha256=digest,
        shape=shape,
        is_flux2=is_flux2,
    )
