# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned LeRobot ``groot`` (GR00T N1.7) policy on one recorded observation with a fixed noise.

This is the oracle for the family: LeRobot's own preprocessing, model and postprocessing.
The action head draws its flow noise with ``torch.randn``; that one draw is served the fixed
noise so the comparison is deterministic. It needs ``lerobot[groot]`` (see
``fixture-source.json``), the pinned base model the checkpoint names, and a CUDA device.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np

ACTION_HORIZON = 40
MAX_ACTION_DIM = 132
NOISE_SEED = 0
CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def fixed_noise(seed: int = NOISE_SEED) -> np.ndarray:
    """The reference's ``torch.randn`` draw for ``[action_horizon, max_action_dim]``, as float32."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, ACTION_HORIZON, MAX_ACTION_DIM), generator=generator, dtype=torch.float32)[0].numpy()


def load_frame(frame_dir: Path, task: str | None = None) -> dict:
    """One recorded observation in LeRobot's raw layout: float images in [0, 1], CHW."""
    import torch
    from PIL import Image

    meta = json.loads((frame_dir / "observation.json").read_text(encoding="utf-8"))

    def image(name: str):
        pixels = np.asarray(Image.open(frame_dir / name).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy())

    return {
        CAMERAS[0]: image("image.png"),
        CAMERAS[1]: image("wrist_image.png"),
        "observation.state": torch.tensor(meta["state"], dtype=torch.float32),
        "task": meta["task"] if task is None else task,
    }


@contextlib.contextmanager
def _served_noise(noise):
    """Serve ``noise`` to the action head's single ``torch.randn`` of that shape."""
    import torch

    real = torch.randn
    calls = []

    def randn(*args, **kwargs):
        size = kwargs.get("size", args[0] if args else None)
        if size is not None and tuple(size) == tuple(noise.shape):
            calls.append(size)
            return noise.to(device=kwargs.get("device"), dtype=kwargs.get("dtype") or noise.dtype)
        return real(*args, **kwargs)

    torch.randn = randn
    try:
        yield calls
    finally:
        torch.randn = real


def load_policy(model_dir: Path, base_dir: Path, tokenizer_dir: Path, *, precision: str, device: str):
    """The policy and its processors. ``base_dir`` is the base model snapshot the checkpoint's
    ``base_model_path`` names on its training machine; ``tokenizer_dir`` holds the backbone's
    tokenizer and processor files (``nvidia/Cosmos-Reason2-2B`` or its identical public copy)."""
    import torch
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained

    torch.set_grad_enabled(False)
    if precision == "fp32":
        # The oracle: full float32 matmuls and convolutions, no TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    config = GrootConfig.from_pretrained(str(model_dir))
    config.base_model_path = str(base_dir)
    config.device = device
    # bf16 is LeRobot's default: float32 weights under bfloat16 autocast.
    config.use_bf16 = precision == "bf16"
    policy = GrootPolicy.from_pretrained(str(model_dir), config=config).to(device).eval()
    pre, post = make_groot_pre_post_processors_from_pretrained(
        policy.config,
        str(model_dir),
        preprocessor_overrides={
            "groot_n1_7_vlm_encode_v1": {"model_name": str(tokenizer_dir)},
            "device_processor": {"device": device},
        },
    )
    return policy, pre, post


def run_reference(policy, pre, post, frame_dir: Path, *, task: str | None = None) -> np.ndarray:
    """The ``[chunk, action_dim]`` action chunk in robot units for one frame (and instruction)."""
    import torch

    batch = pre(load_frame(frame_dir, task))
    noise = torch.from_numpy(fixed_noise()).unsqueeze(0)
    with _served_noise(noise) as calls:
        chunk = policy.predict_action_chunk(batch)
    if len(calls) != 1:
        raise AssertionError(f"expected the action head to draw its noise once, saw {len(calls)}")
    return post(chunk.float()).cpu().numpy()[0]
