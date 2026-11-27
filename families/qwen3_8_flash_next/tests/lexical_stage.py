# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component test, not a substitute for checkpoint or multimodal qualification."""

import math
import subprocess

import numpy as np


def _reference(byte):
    # Independent E4M3FN equation, including both signed zeros and subnormals.
    exponent, fraction = (byte >> 3) & 15, byte & 7
    sign = -1.0 if byte & 128 else 1.0
    if exponent == 0:
        return sign * math.ldexp(fraction, -9)
    if exponent == 15 and fraction == 7:
        return math.nan
    return sign * math.ldexp(1 + fraction / 8, exponent - 7)


def check_packed_rows(directory, consumer):
    import tensorrt as trt
    from families.qwen3_8_flash_next.ple import dequantize_packed_ngram_rows

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    words = network.add_input("words", trt.int32, (-1, 640))
    scale = 0.125
    values = dequantize_packed_ngram_rows(network, words, scale)
    values.name = "values"
    network.mark_output(values)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("words", (1, 640), (2, 640), (3, 640))
    assert [tuple(v) for v in profile.get_shape("words")] == [(1, 640), (2, 640), (3, 640)]
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    plan_path = directory / "rows.plan"
    plan_path.write_bytes(bytes(plan))
    for n in (1, 3):
        # Odd stride permutes every byte value; neighboring bytes differ in each transport word.
        raw = ((np.arange(n * 2560, dtype=np.uint32) * 59 + 3) % 256).astype(np.uint8)
        source, output = directory / "packed.bin", directory / "values.f16"
        raw.tofile(source)
        subprocess.run([consumer, plan_path, source, output], check=True)
        expected = np.array([_reference(int(b)) * scale for b in raw], dtype="<f2")
        actual = np.fromfile(output, dtype="<f2")
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = np.isfinite(expected)
        # Compare bits: +0 and -0 must not be confused.
        np.testing.assert_array_equal(actual[finite].view("<u2"), expected[finite].view("<u2"))
