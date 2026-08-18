# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tensorrt_model_connect.families.sam2_hoi import native_plugin_builder


def _plugin_sources() -> Path:
    return Path(native_plugin_builder.__file__).with_name("native_plugins")


def test_msda_grouped_bf16_path_preserves_fixed_geometry_and_operation_order() -> None:
    source = (_plugin_sources() / "msda_plugin.cu").read_text(encoding="utf-8")

    for contract in (
        "kNumLevels = 3",
        "kNumPoints = 4",
        "kNumHeads = 8",
        "kChannels = 32",
        "kEncoderQueries = 21504",
        "kDecoderQueries = 1500",
        "grouped_bf16_ms_deform_attn_kernel<channels_per_thread>",
        "constexpr int32_t channels_per_thread = 8",
        "is_aligned(value_input, alignof(uint4))",
        "is_aligned(output, alignof(uint4))",
        "load_bf16_vector<kChannelsPerThread>",
        "store_bf16_vector<kChannelsPerThread>",
        "result = add(result, multiply(metadata.w3, v3[slot]))",
        "sampled[slot] = add(result, multiply(metadata.w4, v4[slot]))",
        "add(accumulators[slot], multiply(sampled[slot], attention))",
    ):
        assert contract in source
    assert source.count("constexpr int32_t channels_per_thread = 8") == 2

    grouped = source.split("grouped_bf16_ms_deform_attn_kernel", maxsplit=1)[1].split(
        "bool is_aligned", maxsplit=1
    )[0]
    assert grouped.index("for (int32_t level") < grouped.index("for (int32_t point")
    assert "if (to_float(h_im) > -1.0F" in grouped
    assert "__float2bfloat16_rn" in source
    assert "--use_fast_math" not in source


def test_msda_keeps_generic_fp32_and_unaligned_bf16_fallbacks() -> None:
    source = (_plugin_sources() / "msda_plugin.cu").read_text(encoding="utf-8")
    enqueue = source.split("MsDeformAttnPlugin::enqueue", maxsplit=1)[1]

    assert enqueue.count("ms_deform_attn_kernel<<<blocks, kThreads") == 2
    assert "input_descriptors[0].type == nvinfer1::DataType::kFLOAT" in enqueue
    assert "fixed_shape && num_queries == kEncoderQueries" in enqueue
    assert "fixed_shape && num_queries == kDecoderQueries" in enqueue
    assert "inputs[0] == nullptr" in enqueue
    assert "outputs[0] == nullptr" in enqueue


def test_msda_plugin_abi_and_serialization_remain_empty() -> None:
    sources = _plugin_sources()
    header = (sources / "msda_plugin.h").read_text(encoding="utf-8")
    creator = (sources / "msda_creator.cpp").read_text(encoding="utf-8")
    implementation = (sources / "msda_plugin.cu").read_text(encoding="utf-8")

    assert 'kPLUGIN_NAME = "Sam2HoiMsDeformAttn"' in header
    assert 'kPLUGIN_VERSION = "1"' in header
    assert "PluginRegistrar<trtmc::sam2_hoi::MsDeformAttnCreator>" in creator
    assert "MsDeformAttnPlugin::getSerializationSize() const noexcept" in implementation
    assert "MsDeformAttnPlugin::serialize(void* buffer) const noexcept" in implementation
