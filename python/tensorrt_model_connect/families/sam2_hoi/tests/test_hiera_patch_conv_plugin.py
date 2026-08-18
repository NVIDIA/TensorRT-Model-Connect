# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import functools
import math
from pathlib import Path
import struct

from tensorrt_model_connect.families.sam2_hoi import native_plugin_builder

_INTERIOR_OUTPUTS_PER_THREAD = 24


def _plugin_sources() -> Path:
    return Path(native_plugin_builder.__file__).with_name("native_plugins")


@functools.lru_cache(maxsize=1)
def _host_fmaf():
    library_name = ctypes.util.find_library("m")
    if library_name is None:
        raise RuntimeError("the host C math library is required for the patch-conv proof")
    function = ctypes.CDLL(library_name).fmaf
    function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    function.restype = ctypes.c_float
    return function


def _bfloat16_to_float(bits: int) -> float:
    return struct.unpack("=f", struct.pack("=I", bits << 16))[0]


def _float_to_bfloat16_bits(value: float) -> int:
    bits = struct.unpack("=I", struct.pack("=f", value))[0]
    if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _final_bfloat16_bits(accumulator: float, bias_bits: int) -> int:
    unbiased = _bfloat16_to_float(_float_to_bfloat16_bits(accumulator))
    biased = ctypes.c_float(unbiased + _bfloat16_to_float(bias_bits)).value
    return _float_to_bfloat16_bits(biased)


def _flat_patch_output_bits(
    output_channel: int,
    output_y: int,
    output_x: int,
    input_bits_at,
    weight_bits_at,
    bias_bits_at,
) -> int:
    accumulator = 0.0
    for reduction in range(3 * 7 * 7):
        input_channel = reduction // (7 * 7)
        kernel_offset = reduction - input_channel * 7 * 7
        kernel_y = kernel_offset // 7
        kernel_x = kernel_offset - kernel_y * 7
        input_y = output_y * 4 + kernel_y - 3
        input_x = output_x * 4 + kernel_x - 3
        input_value = 0.0
        if 0 <= input_y < 1024 and 0 <= input_x < 1024:
            input_value = _bfloat16_to_float(input_bits_at(input_channel, input_y, input_x))
        weight_value = _bfloat16_to_float(weight_bits_at(output_channel, reduction))
        accumulator = _host_fmaf()(input_value, weight_value, accumulator)
    return _final_bfloat16_bits(accumulator, bias_bits_at(output_channel))


def _grouped_patch_output_bits(
    output_channel_base: int,
    output_y: int,
    output_x: int,
    input_bits_at,
    weight_bits_at,
    bias_bits_at,
) -> tuple[int, ...]:
    accumulators = [0.0] * _INTERIOR_OUTPUTS_PER_THREAD
    for input_channel in range(3):
        for kernel_y in range(7):
            input_y = output_y * 4 + kernel_y - 3
            for kernel_x in range(7):
                input_x = output_x * 4 + kernel_x - 3
                input_value = 0.0
                if 0 <= input_y < 1024 and 0 <= input_x < 1024:
                    input_value = _bfloat16_to_float(input_bits_at(input_channel, input_y, input_x))
                reduction = (input_channel * 7 + kernel_y) * 7 + kernel_x
                for output_slot in range(_INTERIOR_OUTPUTS_PER_THREAD):
                    weight_value = _bfloat16_to_float(
                        weight_bits_at(output_channel_base + output_slot, reduction)
                    )
                    accumulators[output_slot] = _host_fmaf()(
                        input_value, weight_value, accumulators[output_slot]
                    )
    return tuple(
        _final_bfloat16_bits(accumulator, bias_bits_at(output_channel_base + output_slot))
        for output_slot, accumulator in enumerate(accumulators)
    )


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
        f"kINTERIOR_OUTPUTS_PER_THREAD = {_INTERIOR_OUTPUTS_PER_THREAD}",
        "#pragma unroll 1",
        "for (int32_t input_channel = 0; input_channel < kINPUT_CHANNELS; ++input_channel)",
        "for (int32_t kernel_y = 0; kernel_y < kKERNEL_SIZE; ++kernel_y)",
        "for (int32_t kernel_x = 0; kernel_x < kKERNEL_SIZE; ++kernel_x)",
        "accumulators[output_slot] =",
        "fmaf(input_value",
        "*accumulator =",
        "const __nv_bfloat16 unbiased = __float2bfloat16_rn(accumulator)",
        "__bfloat162float(unbiased)",
        "__bfloat162float(bias[output_channel])",
        "hiera_patch_conv_interior_kernel<<<interior_blocks",
        "hiera_patch_conv_boundary_kernel<<<boundary_blocks",
        "const dim3 boundary_blocks(kOUTPUT_CHANNELS, 2)",
    ):
        assert contract in source
    interior = source.split("accumulate_interior_patch", maxsplit=1)[1].split(
        "accumulate_boundary_patch", maxsplit=1
    )[0]
    assert "input_y >= 0" not in interior
    assert "input_x >= 0" not in interior
    assert "reduction /" not in source
    assert "reduction %" not in source
    lowered = source.lower()
    assert "#include <cudnn" not in lowered
    assert "cudnnbackend" not in lowered
    assert "#include <cublas" not in lowered
    assert "#include <torch" not in lowered
    assert "onnx" not in lowered


def test_grouped_host_reference_is_bitwise_equal_on_boundaries_and_adversarial_values():
    finite_values = (
        0x0000,
        0x8000,
        0x0001,
        0x8001,
        0x0080,
        0x8080,
        0x3E80,
        0xBE80,
        0x3F7F,
        0xBF7F,
        0x3F80,
        0xBF80,
        0x3F81,
        0xBF81,
        0x4000,
        0xC000,
    )

    def input_bits_at(input_channel: int, input_y: int, input_x: int) -> int:
        return finite_values[(input_channel * 17 + input_y * 13 + input_x * 7) % 16]

    def weight_bits_at(output_channel: int, reduction: int) -> int:
        return finite_values[(output_channel * 11 + reduction * 5 + 3) % 16]

    def bias_bits_at(output_channel: int) -> int:
        return finite_values[(output_channel * 3 + 1) % 16]

    positions = ((0, 0), (0, 1), (0, 255), (1, 0), (255, 0), (1, 1), (255, 255))
    for output_channel_base in (0, 48, 72):
        for output_y, output_x in positions:
            expected = tuple(
                _flat_patch_output_bits(
                    output_channel_base + output_slot,
                    output_y,
                    output_x,
                    input_bits_at,
                    weight_bits_at,
                    bias_bits_at,
                )
                for output_slot in range(_INTERIOR_OUTPUTS_PER_THREAD)
            )
            actual = _grouped_patch_output_bits(
                output_channel_base,
                output_y,
                output_x,
                input_bits_at,
                weight_bits_at,
                bias_bits_at,
            )
            assert actual == expected

    def padding_infinity_weight(_output_channel: int, reduction: int) -> int:
        return 0x7F80 if reduction == 0 else 0x0000

    padded = _grouped_patch_output_bits(
        0,
        0,
        0,
        input_bits_at,
        padding_infinity_weight,
        bias_bits_at,
    )
    assert all(math.isnan(_bfloat16_to_float(bits)) for bits in padded)


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
