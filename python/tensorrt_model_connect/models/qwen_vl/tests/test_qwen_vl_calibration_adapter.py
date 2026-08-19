# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL-owned calibration adapter contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="Qwen-VL builder tests require TensorRT")

from tensorrt_model_connect.models.qwen_vl.model import (
    QwenVLCalibrationAdapter,
    quant_adapter,
)


def test_qwen_vl_owns_its_calibration_adapter() -> None:
    adapter = quant_adapter("int8")

    assert isinstance(adapter, QwenVLCalibrationAdapter)
    assert (
        adapter.map_layer_name("model.language_model.layers.3.self_attn.q_proj")
        == "layer.3.w_q"
    )
    assert adapter.map_layer_name("model.visual.blocks.0.attn.qkv") is None
