# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the SANA-WM TensorRT builders."""

from pathlib import Path

import pytest


FAMILY_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "sana_wm"
)


@pytest.mark.parametrize(
    "relative_path",
    [
        "stage1_dit_builder.py",
        "refiner_dit_builder.py",
        "refiner_text_connector_builder.py",
    ],
)
def test_production_builders_do_not_suppress_tensorrt_search(relative_path: str) -> None:
    source = (FAMILY_DIR / relative_path).read_text(encoding="utf-8")

    forbidden_assignments = (
        "builder_optimization_level = 0",
        "max_num_tactics = 1",
        "tiling_optimization_level = trt.TilingOptimizationLevel.NONE",
    )
    for assignment in forbidden_assignments:
        assert assignment not in source


def test_gdn_enqueue_paths_reuse_cublas_handles() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_gdn_plugin.cu").read_text(encoding="utf-8")

    assert "thread_local ThreadCublasHandles" in source
    for function_name in (
        "launch_camera_combined_cublas",
        "launch_main_combined_cublas",
        "launch_main_raw_combined_cublas",
    ):
        body = source.split(f"int32_t {function_name}", maxsplit=1)[1].split(
            "\nint32_t ", maxsplit=1
        )[0]
        assert "cublasCreate(" not in body
        assert "cublasDestroy(" not in body


def test_aten_plugins_do_not_synchronize_the_tensorrt_stream() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_torch_conv2d_plugin.cpp").read_text(
        encoding="utf-8"
    )

    assert "cudaStreamSynchronize(stream)" not in source


def test_raw_gdn_bz_reduction_parallelizes_each_output() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_gdn_plugin.cu").read_text(encoding="utf-8")

    assert "constexpr int32_t kRawBzThreadsPerOutput = 8;" in source
    assert "threadIdx.x / kRawBzThreadsPerOutput" in source
    assert "vector_elems_per_frame * kRawBzThreadsPerOutput" in source


def test_short_conv_uses_native_cuda_without_layout_round_trips() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_torch_conv2d_plugin.cpp").read_text(
        encoding="utf-8"
    )
    body = source.split("int32_t SanaWmShortConvPlugin::enqueue", maxsplit=1)[1].split(
        "\nSanaWmGateProjPlugin::", maxsplit=1
    )[0]

    assert "launch_sana_wm_short_conv(" in body
    assert "at::conv1d(" not in body
    assert ".permute(" not in body
    assert ".flip(" not in body


def test_glumbconv_fuses_bias_activation_and_gate_elementwise_work() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_torch_conv2d_plugin.cpp").read_text(
        encoding="utf-8"
    )
    body = source.split("int32_t SanaWmGlumbconvTempPlugin::enqueue", maxsplit=1)[1].split(
        "\nSanaWmTimestepEmbedPlugin::", maxsplit=1
    )[0]

    assert "launch_sana_wm_bias_silu(" in body
    assert "launch_sana_wm_gated_silu(" in body
    assert "at::conv2d(x, inverted_weight, std::nullopt" in body
    assert "at::conv2d(inverted, depth_weight, std::nullopt" in body
    assert "at::silu(chunks[1])" not in body


def test_camera_softmax_avoids_redundant_attention_layout_round_trips() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_torch_conv2d_plugin.cpp").read_text(
        encoding="utf-8"
    )
    body = source.split("int32_t SanaWmSoftmaxAttentionPlugin::enqueue", maxsplit=1)[1].split(
        "\nSanaWmTorchCamPrepPlugin::", maxsplit=1
    )[0]

    assert "auto q_bhnd =" in body
    assert "auto k_bhnd =" in body
    assert "auto v_bhnd =" in body
    assert ".permute({0, 2, 3, 1})" not in body
    assert ".transpose(-1, -2).contiguous()" not in body
    assert "at::from_blob(outputs[0], {batch, tokens, heads_, head_dim_}" in body


def test_t2i_modulation_fuses_bf16_elementwise_work_into_the_output() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_torch_conv2d_plugin.cpp").read_text(
        encoding="utf-8"
    )
    body = source.split("int32_t SanaWmT2IModulatePlugin::enqueue", maxsplit=1)[1].split(
        "\nSanaWmCaptionEmbedPlugin::", maxsplit=1
    )[0]

    assert "launch_sana_wm_t2i_modulate(" in body
    assert "x * (scale + 1) + shift" not in body
    assert "output.copy_(result)" not in body


def test_camera_phase_c_output_uses_a_coalesced_tiled_transpose() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_gdn_plugin.cu").read_text(encoding="utf-8")

    assert "__shared__ uint16_t tile[32][33];" in source
    assert "dim3 phase_c_copy_threads(32, 8);" in source
    assert "dim3 phase_c_copy_blocks(" in source


def test_camera_prep_coalesces_qkv_outputs_without_changing_inflation_reduction() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_gdn_plugin.cu").read_text(encoding="utf-8")

    assert "__global__ void cam_prep_output_tiled_kernel" in source
    assert "__global__ void cam_prep_inflation_kernel" in source
    assert "dim3 output_threads(32, 8);" in source
    assert "cam_prep_kernel<" not in source


def test_raw_phase_c_output_vectorizes_contiguous_head_channels() -> None:
    source = (FAMILY_DIR / "native_plugins" / "sana_wm_gdn_plugin.cu").read_text(encoding="utf-8")

    assert "__global__ void phase_c_raw_cublas_output_vectorized_kernel" in source
    assert "const float4 raw = *reinterpret_cast<const float4*>" in source
    assert "if (shape.head_dim % 4 == 0)" in source
