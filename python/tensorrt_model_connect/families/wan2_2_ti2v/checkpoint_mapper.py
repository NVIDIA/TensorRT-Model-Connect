# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Wan2.2 TI2V checkpoint loading and canonical key conversion.

The canonical names are the public Diffusers ``WanTransformer3DModel`` and
``AutoencoderKLWan`` names.  The conversion is intentionally implemented in
this family because the native Wan2.2 checkpoint layout is its public input
contract and differs from earlier generations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRANSFORMER_KEY_RENAMES: tuple[tuple[str, str], ...] = (
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    # The upstream implementation executes norm1, norm3, norm2.  Canonical
    # Diffusers ordering is norm1, norm2, norm3, hence the placeholder swap.
    ("norm2", "norm__placeholder"),
    ("norm3", "norm2"),
    ("norm__placeholder", "norm3"),
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("self_attn.norm_q", "attn1.norm_q"),
    ("self_attn.norm_k", "attn1.norm_k"),
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("cross_attn.norm_q", "attn2.norm_q"),
    ("cross_attn.norm_k", "attn2.norm_k"),
)


VAE22_CONFIG: dict[str, Any] = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 16,
    "patch_size": 2,
    "latents_mean": [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    "latents_std": [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
}


def load_native_transformer_state_dict(model_dir: str | Path) -> dict[str, Any]:
    """Load every native sharded safetensor without materializing duplicates."""

    from safetensors.torch import load_file

    root = Path(model_dir)
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        filenames = sorted(set(index["weight_map"].values()))
        paths = [root / name for name in filenames]
    else:
        paths = sorted(root.glob("diffusion_pytorch_model*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No Wan2.2 transformer safetensors in {root}")

    state: dict[str, Any] = {}
    for path in paths:
        for key, value in load_file(path, device="cpu").items():
            if key in state:
                raise ValueError(f"Duplicate transformer tensor {key!r}")
            state[key] = value
    return state


def canonical_transformer_key(native_key: str) -> str:
    key = native_key
    for old, new in TRANSFORMER_KEY_RENAMES:
        key = key.replace(old, new)
    return key


def convert_transformer_state_dict(
    state: dict[str, Any],
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for native_key, value in state.items():
        key = canonical_transformer_key(native_key)
        if key in converted:
            raise ValueError(f"Wan2.2 transformer mapping collision: {native_key!r} -> {key!r}")
        converted[key] = value
    return converted


_VAE_MIDDLE_KEYS = {
    "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
    "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
    "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
    "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
    "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
    "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
    "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
    "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
    "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
    "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
    "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
    "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
    "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
    "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
    "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
    "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
    "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
    "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
    "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
    "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
    "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
    "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
    "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
    "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
}

_VAE_ATTENTION_KEYS = {
    f"{side}.middle.1.{suffix}": f"{side}.mid_block.attentions.0.{suffix2}"
    for side in ("encoder", "decoder")
    for suffix, suffix2 in (
        ("norm.gamma", "norm.gamma"),
        ("to_qkv.weight", "to_qkv.weight"),
        ("to_qkv.bias", "to_qkv.bias"),
        ("proj.weight", "proj.weight"),
        ("proj.bias", "proj.bias"),
    )
}

_VAE_HEAD_KEYS = {
    "encoder.head.0.gamma": "encoder.norm_out.gamma",
    "encoder.head.2.bias": "encoder.conv_out.bias",
    "encoder.head.2.weight": "encoder.conv_out.weight",
    "decoder.head.0.gamma": "decoder.norm_out.gamma",
    "decoder.head.2.bias": "decoder.conv_out.bias",
    "decoder.head.2.weight": "decoder.conv_out.weight",
}

_VAE_QUANT_KEYS = {
    "conv1.weight": "quant_conv.weight",
    "conv1.bias": "quant_conv.bias",
    "conv2.weight": "post_quant_conv.weight",
    "conv2.bias": "post_quant_conv.bias",
}


def _convert_residual_suffix(key: str) -> str:
    replacements = (
        (".residual.0.gamma", ".norm1.gamma"),
        (".residual.2.weight", ".conv1.weight"),
        (".residual.2.bias", ".conv1.bias"),
        (".residual.3.gamma", ".norm2.gamma"),
        (".residual.6.weight", ".conv2.weight"),
        (".residual.6.bias", ".conv2.bias"),
        (".shortcut.weight", ".conv_shortcut.weight"),
        (".shortcut.bias", ".conv_shortcut.bias"),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def canonical_vae_key(native_key: str) -> str:
    if native_key in _VAE_MIDDLE_KEYS:
        return _VAE_MIDDLE_KEYS[native_key]
    if native_key in _VAE_ATTENTION_KEYS:
        return _VAE_ATTENTION_KEYS[native_key]
    if native_key in _VAE_HEAD_KEYS:
        return _VAE_HEAD_KEYS[native_key]
    if native_key in _VAE_QUANT_KEYS:
        return _VAE_QUANT_KEYS[native_key]
    if native_key == "encoder.conv1.weight":
        return "encoder.conv_in.weight"
    if native_key == "encoder.conv1.bias":
        return "encoder.conv_in.bias"
    if native_key == "decoder.conv1.weight":
        return "decoder.conv_in.weight"
    if native_key == "decoder.conv1.bias":
        return "decoder.conv_in.bias"

    if native_key.startswith("encoder.downsamples."):
        key = native_key.replace("encoder.downsamples.", "encoder.down_blocks.", 1)
        if "residual" in key or "shortcut" in key:
            return _convert_residual_suffix(key.replace(".downsamples.", ".resnets."))
        if "resample" in key or "time_conv" in key:
            parts = key.split(".")
            if len(parts) >= 6 and parts[3] == "downsamples":
                return ".".join(parts[:3] + ["downsampler"] + parts[5:])
        return key

    if native_key.startswith("decoder.upsamples."):
        key = native_key.replace("decoder.upsamples.", "decoder.up_blocks.", 1)
        if "residual" in key or "shortcut" in key:
            return _convert_residual_suffix(key.replace(".upsamples.", ".resnets."))
        if "resample" in key or "time_conv" in key:
            parts = key.split(".")
            if len(parts) >= 6 and parts[3] == "upsamples":
                return ".".join(parts[:3] + ["upsampler"] + parts[5:])
        return key
    return native_key


def load_native_vae_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    """Load the native VAE from its model directory or resolved checkpoint path."""

    import torch

    path = Path(checkpoint)
    if path.is_dir():
        path = path / "Wan2.2_VAE.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state)!r}")
    return state


def convert_vae_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for native_key, value in state.items():
        key = canonical_vae_key(native_key)
        if key in converted:
            raise ValueError(f"Wan2.2 VAE mapping collision at {key!r}")
        converted[key] = value
    return converted
