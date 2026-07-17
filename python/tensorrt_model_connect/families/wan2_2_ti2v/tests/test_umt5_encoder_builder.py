# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the Wan2.2-owned pure TensorRT UMT5 builder."""

from __future__ import annotations

import inspect

import ml_dtypes
import numpy as np
import pytest

from tensorrt_model_connect.families.wan2_2_ti2v import umt5_encoder_builder as umt5


def _tiny_config() -> umt5.Umt5EncoderConfig:
    return umt5.Umt5EncoderConfig(
        vocab_size=7,
        hidden_size=4,
        attention_size=4,
        ffn_size=6,
        num_heads=2,
        num_layers=2,
        num_buckets=8,
        relative_attention_max_distance=16,
        sequence_length=8,
    )


def _tiny_native_state(
    model: umt5.Umt5EncoderConfig,
) -> dict[str, np.ndarray]:
    state = {}
    for index, (name, shape) in enumerate(umt5.expected_native_umt5_shapes(model).items()):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values = (values * np.float32(0.03125) + np.float32(index)).astype(ml_dtypes.bfloat16)
        state[name] = values
    return state


def test_official_umt5_xxl_contract_is_fixed() -> None:
    model = umt5.WAN22_UMT5_XXL
    assert model == umt5.Umt5EncoderConfig(
        vocab_size=256384,
        hidden_size=4096,
        attention_size=4096,
        ffn_size=10240,
        num_heads=64,
        num_layers=24,
        num_buckets=32,
        relative_attention_max_distance=128,
        sequence_length=512,
        epsilon=1e-6,
    )
    assert model.head_size == 64
    assert len(umt5.expected_native_umt5_shapes()) == 242


def test_native_mapping_is_complete_and_preserves_bf16_bits() -> None:
    model = _tiny_config()
    state = _tiny_native_state(model)
    mapped = umt5.convert_native_umt5_state_dict(state, model=model)
    key_map = umt5.native_to_canonical_umt5_keys(model)

    assert set(mapped) == set(key_map.values())
    for native_name, canonical_name in key_map.items():
        assert mapped[canonical_name].dtype == ml_dtypes.bfloat16
        np.testing.assert_array_equal(
            mapped[canonical_name].view(np.uint16),
            state[native_name].view(np.uint16),
        )

    assert mapped["layers.0.attention.q.weight"].shape == (4, 4)
    assert mapped["layers.0.ffn.fc1.weight"].shape == (6, 4)
    assert mapped["layers.0.ffn.fc2.weight"].shape == (4, 6)
    assert mapped["layers.0.relative_attention_bias.weight"].shape == (8, 2)


def test_native_mapping_rejects_key_shape_and_dtype_drift() -> None:
    model = _tiny_config()
    state = _tiny_native_state(model)

    missing = dict(state)
    missing.pop("blocks.1.attn.q.weight")
    with pytest.raises(ValueError, match="missing=.*blocks.1.attn.q.weight"):
        umt5.convert_native_umt5_state_dict(missing, model=model)

    unexpected = dict(state)
    unexpected["decoder.weight"] = np.zeros((1,), dtype=ml_dtypes.bfloat16)
    with pytest.raises(ValueError, match="unexpected=.*decoder.weight"):
        umt5.convert_native_umt5_state_dict(unexpected, model=model)

    wrong_shape = dict(state)
    wrong_shape["blocks.0.ffn.fc2.weight"] = np.zeros((4, 5), dtype=ml_dtypes.bfloat16)
    with pytest.raises(ValueError, match="blocks.0.ffn.fc2.weight.*shape"):
        umt5.convert_native_umt5_state_dict(wrong_shape, model=model)

    wrong_dtype = dict(state)
    wrong_dtype["norm.weight"] = wrong_dtype["norm.weight"].astype(np.float32)
    with pytest.raises(TypeError, match="norm.weight.*must be BF16"):
        umt5.convert_native_umt5_state_dict(wrong_dtype, model=model)


def test_relative_position_buckets_match_upstream_bidirectional_layout() -> None:
    actual = umt5.relative_position_buckets(8, 8, num_buckets=8, max_distance=16)
    expected = np.array(
        [
            [0, 5, 6, 6, 6, 6, 7, 7],
            [1, 0, 5, 6, 6, 6, 6, 7],
            [2, 1, 0, 5, 6, 6, 6, 6],
            [2, 2, 1, 0, 5, 6, 6, 6],
            [2, 2, 2, 1, 0, 5, 6, 6],
            [2, 2, 2, 2, 1, 0, 5, 6],
            [3, 2, 2, 2, 2, 1, 0, 5],
            [3, 3, 2, 2, 2, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_builder_stays_family_owned_and_has_no_aten_path() -> None:
    source = inspect.getsource(umt5)
    assert "families.wan_t2v" not in source
    assert "from ..wan_t2v" not in source
    assert "torch.ops" not in source
    assert "torch_tensorrt" not in source
    assert "Wan22Umt5SourceGelu" in source
    assert "Wan22Umt5SourceSoftmax" in source
    assert "Wan22Umt5SourceRmsNorm" in source
    assert "Wan22Umt5Bf16Barrier" in source
    assert (
        inspect.signature(umt5.build_umt5_encoder_engine).parameters["source_softmax"].default
        is False
    )
    assert (
        inspect.signature(umt5.build_umt5_encoder_engine).parameters["source_rmsnorm"].default
        is False
    )

    # Torch is restricted to deserializing the official build-time .pth file;
    # graph construction itself uses TensorRT layers and Wan2.2-owned pure-CUDA
    # plugins, never ATen/libtorch.
    build_source = inspect.getsource(umt5.build_umt5_encoder_engine)
    assert "torch" not in build_source.lower()
    assert '"input_ids", trt.int32, (1, model.sequence_length)' in build_source
    assert '"attention_mask", trt.int32, (1, model.sequence_length)' in build_source


def test_invalid_encoder_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        umt5.Umt5EncoderConfig(attention_size=10, num_heads=3)
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        umt5.Umt5EncoderConfig(sequence_length=0)
    with pytest.raises(ValueError, match="positive and even"):
        umt5.relative_position_buckets(2, 2, num_buckets=7)


def test_debug_layer_outputs_are_validated_before_build() -> None:
    model = _tiny_config()
    weights = umt5.convert_native_umt5_state_dict(_tiny_native_state(model), model=model)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        umt5.build_umt5_encoder_engine(weights, model=model, debug_layer_outputs=(0, 0))
    with pytest.raises(ValueError, match="valid encoder layer indices"):
        umt5.build_umt5_encoder_engine(weights, model=model, debug_layer_outputs=(2,))
    with pytest.raises(ValueError, match="debug_attention_outputs"):
        umt5.build_umt5_encoder_engine(weights, model=model, debug_attention_outputs=(-1,))


def test_source_softmax_is_opt_in_and_requires_the_cuda_plugin() -> None:
    model = _tiny_config()
    weights = umt5.convert_native_umt5_state_dict(_tiny_native_state(model), model=model)

    with pytest.raises(ValueError, match="source_softmax requires source_gelu_plugin"):
        umt5.build_umt5_encoder_engine(
            weights,
            model=model,
            source_softmax=True,
        )


def test_source_rmsnorm_is_opt_in_and_requires_the_cuda_plugin() -> None:
    model = _tiny_config()
    weights = umt5.convert_native_umt5_state_dict(_tiny_native_state(model), model=model)

    with pytest.raises(ValueError, match="source_rmsnorm requires source_gelu_plugin"):
        umt5.build_umt5_encoder_engine(
            weights,
            model=model,
            source_rmsnorm=True,
        )


@pytest.mark.parametrize(
    ("option", "message"),
    (
        ("source_softmax", "requires sequence_length=512 and num_heads=64"),
        ("source_rmsnorm", r"requires \[512,4096\].*epsilon=1e-6"),
    ),
)
def test_source_plugins_hard_fail_outside_the_qualified_fixed_profile(
    option: str,
    message: str,
) -> None:
    model = _tiny_config()
    weights = umt5.convert_native_umt5_state_dict(_tiny_native_state(model), model=model)

    with pytest.raises(ValueError, match=message):
        umt5.build_umt5_encoder_engine(
            weights,
            model=model,
            source_gelu_plugin="not-loaded-before-profile-validation.so",
            **{option: True},
        )
