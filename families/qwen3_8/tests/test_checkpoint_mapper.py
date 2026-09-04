# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 checkpoint quantization contracts."""

from __future__ import annotations

import numpy as np
import pytest

from families.qwen3_8 import checkpoint_mapper as mapper


class _Tensor:
    def __init__(self, values: np.ndarray, dtype: str = "") -> None:
        self.values = np.asarray(values)
        self.dtype = dtype or self.values.dtype

    def numpy(self) -> np.ndarray:
        return self.values

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def _reader(monkeypatch, tensors: dict[str, object]) -> None:
    monkeypatch.setattr(mapper, "_has_tensor", lambda _readers, name: name in tensors)
    monkeypatch.setattr(mapper, "_get_raw_tensor", lambda _readers, name: tensors[name])


def test_fp8_block_scales_are_applied(monkeypatch) -> None:
    tensors = {
        "weight": _Tensor(
            np.ones((4, 4), dtype=np.float32),
            "torch.float8_e4m3fn",
        ),
        "weight_scale_inv": np.asarray(
            [[2.0, 3.0], [4.0, 5.0]],
            dtype=np.float32,
        ),
    }
    _reader(monkeypatch, tensors)

    actual = mapper._load_tensor(["reader"], "weight")

    expected = np.asarray(
        [
            [2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
            [4.0, 4.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)


def test_fp8_without_scales_is_rejected(monkeypatch) -> None:
    tensors = {
        "weight": _Tensor(
            np.ones((2, 2), dtype=np.float32),
            "torch.float8_e4m3fn",
        )
    }
    _reader(monkeypatch, tensors)

    with pytest.raises(KeyError, match="no companion"):
        mapper._load_tensor(["reader"], "weight")


def test_non_quantized_weights_are_unchanged(monkeypatch) -> None:
    weight = np.arange(16, dtype=np.float32).reshape(4, 4)
    _reader(monkeypatch, {"weight": weight})

    np.testing.assert_allclose(mapper._load_tensor(["reader"], "weight"), weight)


def test_block_scales_clip_the_trailing_partial_block() -> None:
    actual = mapper._apply_block_scales(
        np.ones((3, 3), dtype=np.float32),
        np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
    )
    expected = np.asarray(
        [
            [2.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [4.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)


def _nvfp4_tensors(
    name: str,
    packed: np.ndarray,
    group_scale: np.ndarray,
    global_scale: float,
) -> dict[str, object]:
    return {
        name: _Tensor(packed),
        mapper._scale_key(name, ".weight_scale"): group_scale,
        mapper._scale_key(name, ".weight_scale_2"): np.asarray(
            [global_scale],
            dtype=np.float32,
        ),
    }


def test_nvfp4_is_unpacked_and_double_scaled(monkeypatch) -> None:
    name = "projection.weight"
    packed = np.asarray([[0x10, 0x32], [0x9E, 0x00]], dtype=np.uint8)
    group_scale = np.asarray([[2.0], [10.0]], dtype=np.float32)
    _reader(monkeypatch, _nvfp4_tensors(name, packed, group_scale, 3.0))

    actual = mapper._load_tensor(["reader"], name)

    expected = (
        np.asarray(
            [
                [0.0, 0.5, 1.0, 1.5],
                [-4.0, -0.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        * group_scale
        * 3.0
    )
    np.testing.assert_allclose(actual, expected)


def test_nvfp4_without_group_scale_is_rejected(monkeypatch) -> None:
    name = "projection.weight"
    tensors = {
        name: _Tensor(np.zeros((2, 2), dtype=np.uint8)),
        mapper._scale_key(name, ".weight_scale_2"): np.asarray(
            [1.0],
            dtype=np.float32,
        ),
    }
    _reader(monkeypatch, tensors)

    with pytest.raises(KeyError, match="no .*weight_scale"):
        mapper._load_tensor(["reader"], name)


def test_modelopt_per_tensor_fp8_scale_is_applied(monkeypatch) -> None:
    name = "projection.weight"
    tensors = {
        name: _Tensor(
            np.full((2, 2), 3.0, dtype=np.float32),
            "torch.float8_e4m3fn",
        ),
        mapper._scale_key(name, ".weight_scale"): np.asarray(
            [0.5],
            dtype=np.float32,
        ),
    }
    _reader(monkeypatch, tensors)

    np.testing.assert_allclose(
        mapper._load_tensor(["reader"], name),
        np.full((2, 2), 1.5, dtype=np.float32),
    )


def test_e2m1_magnitudes_match_the_format() -> None:
    decoded = mapper._decode_e2m1(np.arange(16, dtype=np.uint8))
    expected = np.asarray(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(decoded[:8], expected)
    np.testing.assert_allclose(decoded[8:], -expected)
