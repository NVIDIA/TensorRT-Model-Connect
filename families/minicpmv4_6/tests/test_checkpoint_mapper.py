# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint weights are checked against the header and for finite values before graph construction."""

import pytest

np = pytest.importorskip("numpy")
ml_dtypes = pytest.importorskip("ml_dtypes")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from families.minicpmv4_6.checkpoint_mapper import (  # noqa: E402
    _load_tensor,
    _open_safetensors,
    decay_rates,
    narrowed,
)


def _readers(tmp_path, **tensors):
    safetensors_numpy.save_file(tensors, str(tmp_path / "model.safetensors"))
    return _open_safetensors(tmp_path)


@pytest.mark.parametrize("dtype", [ml_dtypes.bfloat16, np.float16, np.float32])
def test_a_floating_point_weight_is_read_as_fp32(tmp_path, dtype):
    readers = _readers(tmp_path, w=np.arange(6, dtype=np.float32).reshape(2, 3).astype(dtype))
    value = _load_tensor(readers, "w", (2, 3))
    assert value.dtype == np.float32
    assert value.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.mark.parametrize("dtype", [np.int32, np.int8, np.uint8, np.float64])
def test_a_weight_of_another_dtype_is_rejected(tmp_path, dtype):
    readers = _readers(tmp_path, w=np.ones((2, 3), dtype=dtype))
    with pytest.raises(ValueError, match="BF16, FP16 or FP32"):
        _load_tensor(readers, "w", (2, 3))


@pytest.mark.parametrize("shape", [(3, 2), (6,), (2, 3, 1)])
def test_a_weight_of_another_shape_is_rejected(tmp_path, shape):
    readers = _readers(tmp_path, w=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        _load_tensor(readers, "w", shape)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_nonfinite_weight_is_rejected(tmp_path, bad):
    value = np.ones((2, 3), dtype=np.float32)
    value[1, 2] = bad
    readers = _readers(tmp_path, w=value.astype(ml_dtypes.bfloat16))
    with pytest.raises(ValueError, match="NaN or infinite"):
        _load_tensor(readers, "w", (2, 3))


def test_a_missing_weight_is_reported_by_name(tmp_path):
    readers = _readers(tmp_path, w=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(KeyError, match="other"):
        _load_tensor(readers, "other", (2, 3))


def _shard(tmp_path, index_text):
    safetensors_numpy.save_file({"w": np.ones((2, 3), dtype=np.float32)}, str(tmp_path / "shard-1.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(index_text, encoding="utf-8")
    return tmp_path


def test_a_sharded_checkpoint_maps_each_tensor_to_its_shard(tmp_path):
    readers = _open_safetensors(_shard(tmp_path, '{"weight_map": {"w": "shard-1.safetensors"}}'))
    assert _load_tensor(readers, "w", (2, 3)).shape == (2, 3)


@pytest.mark.parametrize("text", [
    '{"weight_map": {"w": "shard-1.safetensors", "w": "shard-1.safetensors"}}',
    '{"weight_map": {"w": 1}}',
    '{"weight_map": {"w": ""}}',
    '{"weight_map": {"w": "shard-1.safetensors"}, "weight_map": {"w": "shard-1.safetensors"}}',
])
def test_a_shard_index_must_map_names_to_files_once(tmp_path, text):
    with pytest.raises(ValueError):
        _open_safetensors(_shard(tmp_path, text))


def test_decay_rates_are_minus_exp_of_a_log_in_fp32():
    rates = decay_rates(np.array([-2.0, 0.0, 88.0], dtype=ml_dtypes.bfloat16), "A_log")
    assert rates.dtype == np.float32
    np.testing.assert_allclose(rates, -np.exp(np.array([-2.0, 0.0, 88.0], dtype=np.float64)), rtol=1e-6)


@pytest.mark.parametrize("a_log", [89.0, 100.0])
def test_an_a_log_whose_exponential_overflows_fp32_is_rejected(a_log):
    # exp(89) is above FP32's maximum: the decay becomes -inf and the recurrence NaN.
    with pytest.raises(ValueError, match="overflows exp"):
        decay_rates(np.array([0.0, a_log], dtype=np.float32), "model.layers.0.linear_attn.A_log")


def test_weights_within_fp16_become_fp16_constants():
    value = narrowed(np.array([[-65504.0, 1.5], [0.0, 65504.0]], dtype=np.float32), np.float16)
    assert value.dtype == np.float16 and value.flags["C_CONTIGUOUS"]
    assert value.tolist() == [[-65504.0, 1.5], [0.0, 65504.0]]


@pytest.mark.parametrize("big", [65536.0, -1e5, 3e38])
def test_a_finite_weight_beyond_fp16_is_rejected_not_made_infinite(big):
    with pytest.raises(ValueError, match="leave the float16 range"):
        narrowed(np.array([1.0, big], dtype=np.float32), np.float16, "w_q")


def test_fp32_and_integer_constants_keep_their_values():
    assert narrowed(np.array([3e38], dtype=np.float64), np.float32).tolist() == [np.float32(3e38)]
    assert narrowed(np.array([1, 2], dtype=np.int64), np.int64).tolist() == [1, 2]
