# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the BiRefNet geometry the builders need.

The checkpoint's ``config.json`` carries no architecture fields at all - only an
``auto_map`` into its own Python. Everything below is therefore fixed by the
published Swin-T variant and checked against the weights at build time rather
than read from a config that does not describe the model.
"""

from __future__ import annotations

from pathlib import Path

_SWIN_T = {
    "patch_size": 4,
    "window_size": 7,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
}


def resolve(model_dir: str | Path, *, image_size: int) -> dict:
    if image_size % 32:
        raise ValueError("birefnet image size must be a multiple of 32")
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing birefnet config: {config_path}")
    return {
        "image_size": int(image_size),
        # ImageNet statistics, from the checkpoint's own handler.
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "mask_threshold": 0.5,
        **_SWIN_T,
    }


def check_weights(weights: dict) -> None:
    """Fail loudly if the checkpoint is not the Swin-T variant this builds."""
    expected = 96
    actual = int(weights["bb.patch_embed.proj.weight"].shape[0])
    if actual != expected:
        raise NotImplementedError(
            f"birefnet supports the Swin-T variant (embed {expected}), found {actual}")
    for name in ("decoder.ipt_blk5.conv1.weight", "decoder.gdt_convs_attn_4.0.weight",
                 "squeeze_module.0.dec_att.aspp1.atrous_conv.offset_conv.weight"):
        if name not in weights:
            raise NotImplementedError(
                f"birefnet expects {name}; this checkpoint disables a path the builder needs")
