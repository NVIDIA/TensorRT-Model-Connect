# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local E2E contract aliases.

Contracts remain the stable harness API; concrete runners/references/comparators
are owned by the model package.
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import *  # noqa: F401,F403
from tests.e2e_harness.contracts import E2ECase, RunContext


@dataclass(frozen=True)
class InitialLatents:
    path: Path
    sha256: str
    shape: tuple[int, int, int]


def uses_shared_initial_latents(case: E2ECase) -> bool:
    return case.inputs.get("use_shared_initial_latents") is True


def _parity_root(ctx: RunContext) -> Path:
    if not ctx.artifacts_dir:
        return Path(tempfile.gettempdir()) / "trtmc_ltx_video_parity"
    artifacts_dir = Path(ctx.artifacts_dir)
    if artifacts_dir.name in {"hf_artifacts", "trtfb_artifacts"}:
        return artifacts_dir.parent / "shared_initial_latents"
    return artifacts_dir / "shared_initial_latents"


def ensure_initial_latents(case: E2ECase, ctx: RunContext) -> InitialLatents:
    """Materialize the exact packed float32 latent consumed by both backends."""
    frames = int(case.inputs.get("video_num_frames", 9))
    height = int(case.inputs.get("video_height", 256))
    width = int(case.inputs.get("video_width", 256))
    z_dim = int(case.inputs.get("z_dim", 128))
    scale_t = int(case.inputs.get("scale_factor_temporal", 8))
    scale_s = int(case.inputs.get("scale_factor_spatial", 32))
    pt, ph, pw = (int(value) for value in case.inputs.get("patch_size", [1, 1, 1]))
    if min(frames, height, width, z_dim, scale_t, scale_s, pt, ph, pw) <= 0:
        raise ValueError("LTX shared latent dimensions and scale factors must be positive")
    if height % scale_s or width % scale_s:
        raise ValueError(
            f"LTX parity dimensions must be divisible by {scale_s}: {height}x{width}"
        )

    t_lat = (frames - 1) // scale_t + 1
    h_lat = height // scale_s
    w_lat = width // scale_s
    if t_lat % pt or h_lat % ph or w_lat % pw:
        raise ValueError(
            f"LTX latent shape {t_lat}x{h_lat}x{w_lat} is not divisible by "
            f"patch {pt}x{ph}x{pw}"
        )

    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    unpacked = np.random.default_rng(seed).standard_normal(
        (1, z_dim, t_lat, h_lat, w_lat), dtype=np.float32
    )
    packed = (
        unpacked.reshape(
            1,
            z_dim,
            t_lat // pt,
            pt,
            h_lat // ph,
            ph,
            w_lat // pw,
            pw,
        )
        .transpose(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(1, (t_lat // pt) * (h_lat // ph) * (w_lat // pw), -1)
    )

    output_dir = _parity_root(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.name}.seed-{seed}.{frames}x{height}x{width}.f32"
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    packed.astype("<f4", copy=False).tofile(temporary_path)
    os.replace(temporary_path, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return InitialLatents(path=output_path, sha256=digest, shape=packed.shape)
