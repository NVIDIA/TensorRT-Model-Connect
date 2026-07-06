# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs for deterministic PixArt HF-to-TRTMC parity runs."""

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


def uses_shared_initial_latents(case: E2ECase) -> bool:
    """Return whether this case explicitly requests HF-to-TRT latent parity."""
    return case.inputs.get("use_shared_initial_latents") is True


def _parity_root(ctx: RunContext) -> Path:
    if not ctx.artifacts_dir:
        return Path(tempfile.gettempdir()) / "trtmc_pixart_parity"
    artifacts_dir = Path(ctx.artifacts_dir)
    if artifacts_dir.name in {"hf_artifacts", "trtfb_artifacts"}:
        return artifacts_dir.parent / "shared_initial_latents"
    return artifacts_dir / "shared_initial_latents"


def ensure_initial_latents(case: E2ECase, ctx: RunContext) -> InitialLatents:
    """Materialize the exact float32 latent consumed by both implementations."""
    height = int(case.inputs.get("image_height", 1024))
    width = int(case.inputs.get("image_width", height))
    if height <= 0 or width <= 0 or height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f"PixArt parity image shape must be positive and divisible by 8: "
            f"{height}x{width}"
        )

    shape = (1, 4, height // 8, width // 8)
    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal(shape, dtype=np.float32)

    output_dir = _parity_root(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.name}.seed-{seed}.{height}x{width}.f32"
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    latents.tofile(temporary_path)
    os.replace(temporary_path, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return InitialLatents(path=output_path, sha256=digest, shape=shape)
