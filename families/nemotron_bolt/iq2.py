# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Bolt IQ2_XXS expert weights. Only routed experts become dense tensors."""

from __future__ import annotations

import numpy as np


def pack_experts(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change layout without changing any codes, scales, or the 66-byte block size."""
    if blocks.dtype != np.uint8 or blocks.ndim != 4 or blocks.shape[-1] != 66:
        raise ValueError("Expected uint8 [experts, rows, input_width / 256, 66]")
    shape = blocks.shape[:3]
    base = np.ascontiguousarray(blocks[..., :2]).view("<f2").reshape(shape)
    if not np.isfinite(base).all():
        raise ValueError("Nonfinite IQ2_XXS block scale")
    groups = blocks[..., 2:].reshape(*shape, 8, 8)
    # TensorRT treats INT8 constants as quantized tensors and rejects them
    # upstream of a gather. Two ordinary INT32 words store the same 64 bits.
    grid = np.ascontiguousarray(groups[..., :4]).view("<i4")
    control = np.ascontiguousarray(groups[..., 4:]).view("<i4")
    return base, grid, control


def codebooks() -> tuple[np.ndarray, np.ndarray]:
    """Read the public GGML format's 256 magnitude vectors, not checkpoint weights."""
    import gguf

    probe = np.zeros((256, 66), dtype=np.uint8)
    probe[:, :2] = np.array([8.0], dtype="<f2").view(np.uint8)
    probe[:, 2] = np.arange(256, dtype=np.uint8)
    magnitudes = gguf.dequantize(probe, gguf.GGMLQuantizationType.IQ2_XXS)[:, :8]
    codes = [value | ((value.bit_count() & 1) << 7) for value in range(128)]
    sign_bits = (np.array(codes)[:, None] >> np.arange(8)) & 1
    return np.ascontiguousarray(magnitudes), (1 - 2 * sign_bits).astype(np.float32)


def selected_weights(network, blocks: np.ndarray, expert_ids):
    """Return BF16 [selected, output_width, input_width]; do not expand inactive experts."""
    import tensorrt as trt

    if len(expert_ids.shape) != 1 or expert_ids.dtype != trt.int32:
        raise ValueError("Expected rank-one int32 expert indices")
    _, rows, kb, _ = blocks.shape
    op = trt.ElementWiseOperation

    def constant(array):
        array = np.ascontiguousarray(array)
        return network.add_constant(array.shape, array).get_output(0)

    def gather(table, indices):
        return network.add_gather(table, indices, 0).get_output(0)

    def reshape(tensor, shape):
        shuffle = network.add_shuffle(tensor)
        shuffle.reshape_dims = shape
        return shuffle.get_output(0)

    def cast(tensor, dtype):
        return network.add_cast(tensor, dtype).get_output(0)

    def binary(left, right, operation):
        return network.add_elementwise(left, right, operation).get_output(0)

    def scalar(value, dtype, rank):
        return constant(np.full((1,) * rank, value, dtype))

    base, grid, control = (gather(constant(a), expert_ids) for a in pack_experts(blocks))
    rank = len(control.shape)

    def fields(word, shifts, limit):
        divisor = constant(
            np.array([1 << s for s in shifts], dtype=np.int32).reshape((1,) * (rank - 1) + (-1,))
        )
        shifted = binary(word, divisor, op.FLOOR_DIV)
        modulus = scalar(limit, np.int32, rank)
        return binary(
            shifted, binary(binary(shifted, modulus, op.FLOOR_DIV), modulus, op.PROD), op.SUB
        )

    grid = fields(grid, (0, 8, 16, 24), 256)
    sign = fields(control, (0, 7, 14, 21), 128)
    nibble = fields(control, (28,), 16)
    magnitude_lut, sign_lut = codebooks()
    values = binary(
        gather(constant(magnitude_lut), grid), gather(constant(sign_lut), sign), op.PROD
    )
    scale = binary(cast(nibble, trt.float32), scalar(0.5, np.float32, rank), op.SUM)
    scale = binary(scale, scalar(0.25, np.float32, rank), op.PROD)
    base = reshape(cast(base, trt.float32), (0, rows, kb, 1, 1))
    scaled = binary(values, reshape(binary(scale, base, op.PROD), (0, rows, kb, 8, 1, 1)), op.PROD)
    return reshape(cast(scaled, trt.bfloat16), (0, rows, kb * 256))
