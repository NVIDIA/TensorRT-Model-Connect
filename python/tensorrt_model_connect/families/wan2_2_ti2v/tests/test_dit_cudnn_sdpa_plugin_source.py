# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source contracts for the production Wan2.2 cuDNN SDPA plugin."""

from __future__ import annotations

from pathlib import Path


def _source_dir() -> Path:
    return Path(__file__).parents[1] / "dit_cuda_plugins"


def test_sdpa_plugin_is_native_and_selects_a_target_local_plan() -> None:
    source = (_source_dir() / "wan22_cudnn_sdpa_plugin.cpp").read_text()
    lowered = source.lower()

    assert "cudnn_frontend.h" in source
    assert "CUDNN_FRONTEND_MAJOR_VERSION == 1" in source
    assert "CUDNN_FRONTEND_MINOR_VERSION == 22" in source
    assert "CUDNN_FRONTEND_PATCH_VERSION == 1" in source
    assert "create_execution_plans({fe::HeurMode_t::A})" in source
    assert "check_support(handle_)" in source
    assert "build_plans(handle_)" in source
    assert "create_execution_plan(" not in source
    assert "kernel_cfg" not in lowered
    assert "engine_id" not in lowered
    assert "torch/" not in lowered
    assert "aten/" not in lowered
    assert "libtorch" not in lowered
    assert "libpython" not in lowered
    assert "import torch" not in lowered


def test_sdpa_plugin_preserves_the_exact_bf16_bshd_contract() -> None:
    source = (_source_dir() / "wan22_cudnn_sdpa_plugin.cpp").read_text()
    header = (_source_dir() / "wan22_cudnn_sdpa_plugin.h").read_text()

    assert "set_io_data_type(fe::DataType_t::BFLOAT16)" in source
    assert "set_intermediate_data_type(fe::DataType_t::FLOAT)" in source
    assert "set_compute_data_type(fe::DataType_t::FLOAT)" in source
    assert "nvinfer1::DataType::kBF16" in source
    assert "q_strides{h * q_sequence * d, d, h * d, 1}" in source
    assert "kv_strides{h * kv_sequence * d, d, h * d, 1}" in source
    assert "kProductionQSequence = 27'280" in header
    assert "kSelfKvSequence = 27'280" in header
    assert "kCrossKvSequence = 512" in header
    assert "kProductionHeads = 24" in header
    assert "kProductionHeadDimension = 128" in header


def test_sdpa_plugin_fields_cannot_confuse_self_and_cross_attention() -> None:
    source = (_source_dir() / "wan22_cudnn_sdpa_plugin.cpp").read_text()
    header = (_source_dir() / "wan22_cudnn_sdpa_plugin.h").read_text()

    for field in (
        "attention_kind",
        "batch",
        "heads",
        "q_sequence",
        "kv_sequence",
        "head_dimension",
    ):
        assert f'"{field}"' in header
    assert "present != kAllRequiredFields" in source
    assert "AttentionKind::kSelf" in header
    assert "AttentionKind::kCross" in header
    assert "config.kv_sequence == kSelfKvSequence" in header
    assert "config.kv_sequence == kCrossKvSequence" in header


def test_sdpa_serialization_contains_only_the_shape_contract() -> None:
    source = (_source_dir() / "wan22_cudnn_sdpa_plugin.cpp").read_text()
    header = (_source_dir() / "wan22_cudnn_sdpa_plugin.h").read_text()

    assert "kSerializationMagic" in header
    assert "kSerializationVersion" in header
    assert "byte_size" in header
    assert "is_valid_serialized_config" in source
    assert "length != sizeof(SerializedConfig)" in source
    assert "make_serialized_config(config_)" in source
    assert "HeurMode A" in header
    assert "engine/config identifier is persisted" in header
