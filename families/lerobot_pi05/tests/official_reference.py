# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned LeRobot ``pi05`` policy on one recorded observation with a fixed noise.

This is the oracle for the family: LeRobot's own preprocessing, model and
postprocessing, with the flow noise injected so the comparison is deterministic.
It needs ``lerobot[pi]`` (see ``fixture-source.json``) and a CUDA device for bfloat16.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHUNK_SIZE = 50
MAX_ACTION_DIM = 32
NOISE_SEED = 0


def fixed_noise(seed: int = NOISE_SEED) -> np.ndarray:
    """The reference's ``torch.randn`` draw for ``[chunk, max_action_dim]``, as float32."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, CHUNK_SIZE, MAX_ACTION_DIM), generator=generator, dtype=torch.float32)[0].numpy()


def load_frame(frame_dir: Path) -> dict:
    """Read one recorded observation directory into LeRobot's raw observation layout."""
    import torch
    from PIL import Image

    meta = json.loads((frame_dir / "observation.json").read_text(encoding="utf-8"))

    def image(name: str):
        pixels = np.asarray(Image.open(frame_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy())

    return {
        "observation.images.image": image("image.png"),
        "observation.images.image2": image("wrist_image.png"),
        "observation.state": torch.tensor(meta["state"], dtype=torch.float32),
        "task": meta["task"],
    }


def run_reference(
    model_dir: Path,
    tokenizer_dir: Path,
    frame_dir: Path,
    *,
    dtype: str,
    device: str,
) -> np.ndarray:
    """Return the ``[50, action_dim]`` action chunk in robot units for one frame."""
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    torch.set_grad_enabled(False)
    config = PI05Config.from_pretrained(str(model_dir))
    config.dtype = dtype
    config.device = device
    # The oracle is eager PyTorch. Some trained checkpoints ask for torch.compile with autotuned
    # kernels, whose numerics differ from eager at the 1e-3 level and are not a stable reference.
    config.compile_model = False
    policy = PI05Policy.from_pretrained(str(model_dir), config=config).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_dir),
        # The tokenizer is only ever read from a local directory, so a gated repo is not needed.
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": device},
        },
    )
    batch = pre(load_frame(frame_dir))
    noise = torch.from_numpy(fixed_noise()).unsqueeze(0).to(device)
    chunk = policy.predict_action_chunk(batch, noise=noise)
    return post(chunk.float()).cpu().numpy()[0]
