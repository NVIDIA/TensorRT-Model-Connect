# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V FP8 projection set contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from families.wan2_2_ti2v import fp8_profile, trt_builder
from families.wan2_2_ti2v.model_config import WAN22_TI2V_5B


def test_precomputed_scales_exclude_cross_attention_output_projection() -> None:
    # TensorRT 11.4 cannot compile an FP8 Quantize after the fused cross-attention.
    scales = fp8_profile.load_precomputed_fp8_scales(SimpleNamespace(raw={}))

    assert len(scales) == 3 * WAN22_TI2V_5B.num_layers
    assert not any("attn2.to_out" in name for name in scales)


def test_qualified_fp8_scale_names_exclude_cross_attention_output_projection() -> None:
    ffn, cross_q = trt_builder._qualified_fp8_scale_names(WAN22_TI2V_5B)

    assert len(ffn) == 2 * WAN22_TI2V_5B.num_layers
    assert len(cross_q) == WAN22_TI2V_5B.num_layers
    assert all(name.endswith("attn2.to_q") for name in cross_q)


def test_split_rejects_scale_map_with_cross_attention_output_projection() -> None:
    scales = fp8_profile.load_precomputed_fp8_scales(SimpleNamespace(raw={}))
    scales["blocks.0.attn2.to_out.0"] = {"input_scale": 1.0, "weight_scale": 1.0}

    with pytest.raises(ValueError, match="unexpected="):
        trt_builder._split_qualified_fp8_scales(scales, WAN22_TI2V_5B)
