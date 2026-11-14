# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 checkpoint quantization contracts."""

from __future__ import annotations

import numpy as np
import pytest

from families.qwen3_8 import checkpoint_mapper as mapper


@pytest.mark.parametrize("format_name", ["nvfp4", "fp8", "int4_awq"])
def test_prepared_q_gate_split_retains_exact_source_bytes(monkeypatch, format_name) -> None:
    import torch
    from families.qwen3_8 import quantization as quant

    name = "model.language_model.layers.0.self_attn.q_proj"
    width = 128 if format_name == "int4_awq" else 16
    if format_name == "int4_awq":
        codes = np.arange(8 * width, dtype=np.uint16).reshape(8, width).astype(np.uint8) & 15
        # Vary output rows too, so mixing adjacent rows cannot accidentally pass.
        codes = (codes + np.arange(8, dtype=np.uint8)[:, None]) & 15
        values = torch.from_numpy(codes[0::2] | (codes[1::2] << 4))
        expected_packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1) / 16
    elif format_name == "nvfp4":
        values = torch.arange(64, dtype=torch.uint8).reshape(8, 8)
        scales = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1).to(torch.float8_e4m3fn)
    else:
        values = torch.arange(128, dtype=torch.float32).reshape(8, 16).to(torch.float8_e4m3fn)
        scales = torch.tensor([0.25])
    tensors = {name + ".weight": values, name + ".weight_scale": scales}
    if format_name == "nvfp4":
        tensors[name + ".weight_scale_2"] = torch.tensor([0.125])
    if format_name == "int4_awq":
        tensors[name + ".pre_quant_scale"] = torch.linspace(0.5, 1.5, width)
    else:
        expected_packed = values.view(torch.uint8).numpy()
    monkeypatch.setattr(quant, "_has_tensor", lambda _, key: key in tensors)
    monkeypatch.setattr(quant, "_get_raw_tensor", lambda _, key: tensors[key])
    query, gate = quant._read_prepared_weight_split_q([], name, 2, 2, awq=format_name == "int4_awq")
    for result, rows in ((query, [0, 1, 4, 5]), (gate, [2, 3, 6, 7])):
        assert result.out_features == 4 and result.in_features == width
        np.testing.assert_array_equal(result.packed, expected_packed[rows])
        if format_name == "nvfp4":
            assert isinstance(result, quant._NVFP4Weight)
            assert result.global_scale == 0.125
            np.testing.assert_array_equal(result.block_scale_f8, scales.view(torch.uint8).numpy()[rows])
        elif format_name == "int4_awq":
            assert isinstance(result, quant._AWQWeight)
            np.testing.assert_array_equal(result.block_scale, scales.numpy()[rows])
            np.testing.assert_array_equal(result.pre_quant_scale, tensors[name + ".pre_quant_scale"].numpy())
        else:
            assert isinstance(result, quant._FP8Weight)
            assert result.weight_scale == 0.25
    if format_name == "nvfp4":
        # Missing FP4 metadata must not reinterpret the nibbles as FP8 bytes.
        del tensors[name + ".weight_scale_2"]
        with pytest.raises(ValueError, match="expected FP8"):
            quant._read_prepared_weight([], name)

    if format_name == "int4_awq":
        tensors[name + ".weight_scale"] = torch.ones(8, 2)
        with pytest.raises(ValueError, match="group-128"):
            quant._read_awq_weight([], name)
        tensors[name + ".weight_scale"] = scales
        tensors[name + ".pre_quant_scale"][0] = float("nan")
        with pytest.raises(ValueError, match="smoothing scales"):
            quant._read_awq_weight([], name)
        del tensors[name + ".pre_quant_scale"]
        assert quant._read_awq_weight([], name).pre_quant_scale is None


@pytest.mark.parametrize("format_name", ["nvfp4", "fp8", "int4_awq"])
def test_embedded_draft_keeps_its_own_quantization(monkeypatch, tmp_path, format_name) -> None:
    from types import SimpleNamespace
    import torch
    from families.qwen3_8 import engine_builder, quantization as quant

    hidden = 128 if format_name == "int4_awq" else 16
    head_dim = hidden // 2
    shapes = {
        "mtp.fc": (hidden, 2 * hidden),
        "mtp.layers.0.self_attn.q_proj": (2 * hidden, hidden),
        "mtp.layers.0.self_attn.k_proj": (head_dim, hidden),
        "mtp.layers.0.self_attn.v_proj": (head_dim, hidden),
        "mtp.layers.0.self_attn.o_proj": (hidden, hidden),
        **{f"mtp.layers.0.mlp.{name}_proj": (hidden, hidden) for name in ("gate", "up", "down")},
    }
    tensors = {}
    for prefix, (rows, columns) in shapes.items():
        if format_name == "int4_awq":
            tensors[prefix + ".weight"] = torch.full((rows // 2, columns), 0xF2, dtype=torch.uint8)
            tensors[prefix + ".weight_scale"] = torch.full((rows, columns // 128), 0.125)
            tensors[prefix + ".pre_quant_scale"] = torch.full((columns,), 0.75)
            continue
        tensors[prefix + ".input_scale"] = torch.tensor([0.25])
        if format_name == "nvfp4":
            tensors[prefix + ".weight"] = torch.full((rows, columns // 2), 0x32, dtype=torch.uint8)
            tensors[prefix + ".weight_scale"] = torch.ones(rows, columns // 16).to(torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale_2"] = torch.tensor([0.125])
        else:
            tensors[prefix + ".weight"] = torch.ones(rows, columns).to(torch.float8_e4m3fn)
            tensors[prefix + ".weight_scale"] = torch.tensor([0.125])
    monkeypatch.setattr(quant, "_has_tensor", lambda _, key: key in tensors)
    monkeypatch.setattr(quant, "_get_raw_tensor", lambda _, key: tensors[key])
    config = SimpleNamespace(num_hidden_layers=0, hidden_size=hidden, intermediate_size=hidden,
                             vocab_size=hidden, num_attention_heads=2, head_dim=head_dim)
    loader = quant.load_qwen3_8_awq if format_name == "int4_awq" else quant.calibrate_qwen3_8_nvfp4
    context = loader(tmp_path, config, None, readers=[])
    expected = {"fc", "w_q", "w_gate_attn", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down"}
    assert set(context.profile.scales) == {"mtp_layer." + name for name in expected}
    for item in context.profile.scales.values():
        if format_name == "int4_awq":
            assert item.input_scale is None  # W4A16 does not quantize activations.
            assert isinstance(item.weight, quant._AWQWeight)
            np.testing.assert_array_equal(item.weight.pre_quant_scale, 0.75)
        else:
            assert item.input_scale == 0.25
            assert isinstance(item.weight, quant._NVFP4Weight if format_name == "nvfp4" else quant._FP8Weight)

    def load_norm(_readers, name):
        assert "norm" in name, f"Packed projection must not be silently decoded: {name}"
        return np.zeros(head_dim if ".self_attn." in name else hidden, dtype=np.float32)

    monkeypatch.setattr(engine_builder, "_load_tensor", load_norm)
    monkeypatch.setattr(engine_builder, "_has_tensor",
                        lambda _, name: name in tensors or "norm" in name)
    weights = {}
    engine_builder.Qwen38Model()._load_mtp_weights(
        [], weights, hidden, hidden, 2, 1, head_dim, precision="fp16", quant_ctx=context)
    assert all("mtp_layer." + name not in weights for name in expected)
    np.testing.assert_array_equal(weights["mtp_layer.q_norm"], np.ones(hidden))
    np.testing.assert_array_equal(weights["mtp_layer.k_norm"], np.ones(head_dim))
    np.testing.assert_array_equal(weights["mtp_final_norm"], np.ones(hidden))

    for norm in ("q_norm", "k_norm"):
        missing = f"mtp.layers.0.self_attn.{norm}.weight"
        monkeypatch.setattr(engine_builder, "_has_tensor",
                            lambda _, name, missing=missing:
                            name != missing and (name in tensors or "norm" in name))
        with pytest.raises(KeyError, match="Missing required MTP norm"):
            engine_builder.Qwen38Model()._load_mtp_weights(
                [], {}, hidden, hidden, 2, 1, head_dim, precision="fp16", quant_ctx=context)


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
