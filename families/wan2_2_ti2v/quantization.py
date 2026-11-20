# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V NVFP4 recipe: which DiT projections are NVFP4, their scales and weight packing.

No ModelOpt NVFP4 checkpoint of Wan2.2 TI2V-5B is published, so ``quantization="nvfp4"``
self-quantizes the original checkpoint at build time. It changes the same 90 DiT projections
as the FP8 recipe (``ffn.net.0.proj``, ``ffn.net.2`` and ``attn2.to_q`` in each of the 30
blocks) from FP8 W8A8 to NVFP4 W4A4; everything else is built exactly as in the FP8 build.

Weights: ``weight_scale_2 = amax / (6 * 448)`` per tensor, one e4m3 block scale per 16 inputs
(``block_amax / 6 / weight_scale_2``, clamped to 448), values rounded to the E2M1 grid and
packed two per byte (first value in the low nibble). Activations: ``add_dynamic_quantize`` in
blocks of 16 with a static per-tensor global scale read from
``data/wan22-ti2v-5b-921dbaf3-nvfp4-scales.json``. Each value there is the FP8 activation scale
of the same projection divided by 6 (``amax / (6 * 448)``), so both recipes use one calibration.
Like FP8, NVFP4 is qualified only for the 1280x704, 121-frame profile.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .model_config import WAN22_TI2V_5B, Wan22TI2VConfig, select_generation_profile


FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = E2M1_MAX * FP8_E4M3_MAX

# Projection names (inside each DiT block) that run as NVFP4 W4A4 GEMMs.
NVFP4_PROJECTIONS = ("ffn.net.0.proj", "ffn.net.2", "attn2.to_q")

_SCALE_PATH = Path(__file__).with_name("data") / "wan22-ti2v-5b-921dbaf3-nvfp4-scales.json"

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between neighbouring E2M1 values. A value exactly on a midpoint goes to the
# neighbour with an even code: the first, third, fifth and seventh midpoint round down.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MIDPOINT_ROUNDS_UP = (False, True, False, True, False, True, False)


def nvfp4_layer_names(profile: Wan22TI2VConfig) -> tuple[str, ...]:
    """The DiT projections that are NVFP4 under the recipe, in block order."""
    return tuple(
        f"blocks.{index}.{projection}"
        for index in range(profile.num_layers)
        for projection in NVFP4_PROJECTIONS
    )


def validated_nvfp4_scales(scales, profile: Wan22TI2VConfig) -> dict[str, float]:
    """Return ``{layer: input_scale}`` for exactly the recipe layers, or raise."""
    if not isinstance(scales, dict):
        raise TypeError("Wan2.2 NVFP4 scales must be a dictionary")
    expected = set(nvfp4_layer_names(profile))
    provided = set(scales)
    if provided != expected:
        raise ValueError(
            "Wan2.2 NVFP4 scales must cover exactly ffn.net.0.proj, ffn.net.2 and "
            "attn2.to_q in every block; "
            f"missing={sorted(expected - provided)}, unexpected={sorted(provided - expected)}"
        )
    result = {}
    for name in nvfp4_layer_names(profile):
        entry = scales[name]
        if not isinstance(entry, dict) or set(entry) != {"input_scale"}:
            raise ValueError(f"Wan2.2 NVFP4 scale entry is invalid: {name}")
        value = entry["input_scale"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Wan2.2 NVFP4 input_scale for {name} must be positive and finite")
        result[name] = float(value)
    return result


def load_precomputed_nvfp4_scales(config) -> dict[str, dict[str, float]]:
    """Load the family-owned NVFP4 scale map (``{layer: {"input_scale": ...}}``) for the
    qualified full profile."""
    if select_generation_profile(config.raw) != WAN22_TI2V_5B:
        raise ValueError("Wan2.2 NVFP4 supports only the 1280x704, 121-frame profile")
    scales = json.loads(_SCALE_PATH.read_text(encoding="utf-8"))
    validated_nvfp4_scales(scales, WAN22_TI2V_5B)
    return scales


def _e2m1_codes(scaled):
    """Sign-magnitude E2M1 codes (0..15) of already block-scaled values."""
    import torch

    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.uint8, device=scaled.device)
    for midpoint, up in zip(_MIDPOINTS, _MIDPOINT_ROUNDS_UP, strict=True):
        code += (magnitude >= midpoint if up else magnitude > midpoint).to(torch.uint8)
    return code | ((scaled < 0).to(torch.uint8) << 3)


def quantize_nvfp4_weight(weight) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to NVFP4.

    Returns the packed bytes ``[out, in / 2]``, the raw e4m3 block-scale bytes
    ``[out, in / 16]`` and ``weight_scale_2``. An all-zero block gets block scale 1.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.as_tensor(np.asarray(weight, dtype=np.float32)).to(device)
    if tensor.ndim != 2:
        raise ValueError(f"expected a [out, in] weight, got shape {tuple(tensor.shape)}")
    out_features, in_features = tensor.shape
    if in_features % NVFP4_BLOCK_SIZE:
        raise ValueError(f"in_features={in_features} is not a multiple of {NVFP4_BLOCK_SIZE}")
    amax = float(tensor.abs().max())
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError("NVFP4 weight must have a positive finite absolute maximum")
    weight_scale_2 = float(np.float32(amax) * np.float32(1.0 / NVFP4_MAX))
    blocks = tensor.reshape(out_features, in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_scale = (blocks.abs().amax(-1) / E2M1_MAX / weight_scale_2).clamp_(max=FP8_E4M3_MAX)
    scale_float = block_scale.to(torch.float8_e4m3fn).float()
    scale_float = torch.where(scale_float == 0, torch.ones_like(scale_float), scale_float)
    scaled = blocks / (scale_float.unsqueeze(-1) * weight_scale_2)
    code = _e2m1_codes(scaled).reshape(out_features, in_features)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().cpu().numpy()
    scales = scale_float.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales), weight_scale_2


def dequantize_nvfp4_weight(packed, block_scale, weight_scale_2: float) -> np.ndarray:
    """The float32 ``[out, in]`` values the NVFP4 bytes stand for."""
    import torch

    packed_t = torch.from_numpy(np.ascontiguousarray(packed))
    codes = torch.stack([packed_t & 0xF, packed_t >> 4], dim=-1).reshape(packed_t.shape[0], -1)
    values = torch.tensor(_E2M1_GRID)[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    scale = torch.from_numpy(np.ascontiguousarray(block_scale)).view(torch.float8_e4m3fn).float()
    result = values * scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1) * weight_scale_2
    return result.numpy()
