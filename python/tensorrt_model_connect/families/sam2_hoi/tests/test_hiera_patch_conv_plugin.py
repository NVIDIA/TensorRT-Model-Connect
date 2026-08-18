# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tensorrt_model_connect.families.sam2_hoi import native_plugin_builder


def _plugin_sources() -> Path:
    return Path(native_plugin_builder.__file__).with_name("native_plugins")


def test_hiera_patch_conv_plugin_is_fixed_site_and_preserves_source_rounding_order():
    source = (_plugin_sources() / "hiera_patch_conv_plugin.cu").read_text(encoding="utf-8")

    for contract in (
        "kINPUT_CHANNELS = 3",
        "kINPUT_HEIGHT = 1024",
        "kINPUT_WIDTH = 1024",
        "kOUTPUT_CHANNELS = 96",
        "kOUTPUT_HEIGHT = 256",
        "kOUTPUT_WIDTH = 256",
        "kKERNEL_SIZE = 7",
        "kSTRIDE = 4",
        "kPADDING = 3",
        "#pragma unroll 1",
        "accumulator = fmaf",
        "const __nv_bfloat16 unbiased = __float2bfloat16_rn(accumulator)",
        "__bfloat162float(unbiased)",
        "__bfloat162float(bias[output_channel])",
    ):
        assert contract in source
    lowered = source.lower()
    assert "#include <cudnn" not in lowered
    assert "cudnnbackend" not in lowered
    assert "#include <cublas" not in lowered
    assert "#include <torch" not in lowered
    assert "onnx" not in lowered


def test_hiera_patch_conv_creator_is_registered_in_existing_family_dso():
    source_dir = _plugin_sources()
    creator = (source_dir / "hiera_patch_conv_creator.cpp").read_text(encoding="utf-8")
    header = (source_dir / "hiera_patch_conv_plugin.h").read_text(encoding="utf-8")
    cmake = (source_dir / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'kPLUGIN_NAME = "Sam2HoiHieraPatchConv"' in header
    assert 'kPLUGIN_VERSION = "1"' in header
    assert "PluginRegistrar<trtmc::sam2_hoi::HieraPatchConvCreator>" in creator
    assert "hiera_patch_conv_plugin.cu" in cmake
    assert "hiera_patch_conv_creator.cpp" in cmake
    assert cmake.count("add_library(trtmc_sam2_hoi_native_plugin SHARED") == 1
