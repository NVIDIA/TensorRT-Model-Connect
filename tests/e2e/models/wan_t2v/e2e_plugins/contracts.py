# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local E2E contract aliases.

Contracts remain the stable harness API; concrete runners/references/comparators
are owned by the model package.
"""

import hashlib
import html
import os
import re
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
    shape: tuple[int, int, int, int, int]


def normalize_wan_prompt(text: str) -> str:
    """Match Diffusers Wan ``prompt_clean`` before either tokenizer runs."""
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def uses_shared_initial_latents(case: E2ECase) -> bool:
    return case.inputs.get("use_shared_initial_latents") is True


def _parity_root(ctx: RunContext) -> Path:
    if not ctx.artifacts_dir:
        return Path(tempfile.gettempdir()) / "trtmc_wan_t2v_parity"
    artifacts_dir = Path(ctx.artifacts_dir)
    if artifacts_dir.name in {"hf_artifacts", "trtfb_artifacts"}:
        return artifacts_dir.parent / "shared_initial_latents"
    return artifacts_dir / "shared_initial_latents"


def ensure_initial_latents(case: E2ECase, ctx: RunContext) -> InitialLatents:
    """Materialize the exact CTHW float32 latent consumed by both backends."""
    frames = int(case.inputs.get("video_num_frames", 17))
    height = int(case.inputs.get("video_height", 480))
    width = int(case.inputs.get("video_width", 832))
    z_dim = int(case.inputs.get("z_dim", 16))
    scale_t = int(case.inputs.get("scale_factor_temporal", 4))
    scale_s = int(case.inputs.get("scale_factor_spatial", 8))
    if min(frames, height, width, z_dim, scale_t, scale_s) <= 0:
        raise ValueError("Wan shared latent dimensions and scale factors must be positive")
    if height % scale_s or width % scale_s:
        raise ValueError(
            f"Wan parity dimensions must be divisible by {scale_s}: {height}x{width}"
        )

    shape = (
        1,
        z_dim,
        (frames - 1) // scale_t + 1,
        height // scale_s,
        width // scale_s,
    )
    seed = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
    latents = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)

    output_dir = _parity_root(ctx)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.name}.seed-{seed}.{frames}x{height}x{width}.f32"
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    latents.astype("<f4", copy=False).tofile(temporary_path)
    os.replace(temporary_path, output_path)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return InitialLatents(path=output_path, sha256=digest, shape=shape)
