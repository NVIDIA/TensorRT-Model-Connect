# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source and linkage contracts for the bounded Wan2.2 VAE Conv3d plugin."""

from __future__ import annotations

import inspect
from pathlib import Path

from tensorrt_model_connect.families.wan2_2_ti2v import vae_builder


def _source_dir() -> Path:
    return Path(__file__).parents[1] / "vae_cuda_plugins"


def test_vae_conv3d_is_native_cudnn_with_bounded_target_local_selection() -> None:
    source = (_source_dir() / "wan22_vae_conv3d_plugin.cu").read_text()
    lowered = source.lower()

    assert "cudnnGetConvolutionForwardAlgorithm_v7" in source
    assert "cudnnGetConvolutionForwardWorkspaceSize" in source
    assert "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION" in source
    assert "kMAX_WORKSPACE_BYTES = size_t{512} << 20" in source
    assert "candidate.memory > kMAX_WORKSPACE_BYTES" in source
    assert "addChannelBias<<<" in source
    assert source.index("cudnnConvolutionForward(") < source.index("addChannelBias<<<")
    assert (
        "algorithm_"
        not in source[source.index("void serialize(") : source.index("void setPluginNamespace")]
    )
    assert "torch/" not in lowered
    assert "aten/" not in lowered
    assert "libtorch" not in lowered
    assert "libpython" not in lowered


def test_vae_conv3d_static_contract_is_deliberately_narrow() -> None:
    source = (_source_dir() / "wan22_vae_conv3d_plugin.cu").read_text()
    builder = inspect.getsource(vae_builder)

    assert "config.input_channels == 256 || config.input_channels == 512" in source
    assert "config.output_channels == 256" in source
    assert "config.input_depth == 3 || config.input_depth == 6" in source
    assert "config.input_height == 18 && config.input_width == 18" in source
    assert "config.input_height == 194 && config.input_width == 338" in source
    assert "config.input_height == 354 && config.input_width == 642" in source
    assert "for resnet in range(3)" in builder
    assert "for conv in (1, 2)" in builder
    assert "profile.latent_frames * len(_VAE_NATIVE_CONV_SCOPES)" in builder
    assert "_load_up_block3_conv_initializers" in builder
    assert "network.add_constant" in builder


def test_vae_plugin_links_only_runtime_cuda_and_cudnn_dependencies() -> None:
    cmake = (_source_dir() / "CMakeLists.txt").read_text().lower()

    assert "wan22_vae_conv3d_plugin.cu" in cmake
    assert "wan22_vae_rms_norm_plugin.cu" in cmake
    assert "wan22_vae_fp32_barrier_plugin.cu" in cmake
    assert "libcudnn.so" in cmake
    assert "cuda::cudart" in cmake
    assert "find_package(torch" not in cmake
    assert 'cuda_architectures "103;110"' in cmake
