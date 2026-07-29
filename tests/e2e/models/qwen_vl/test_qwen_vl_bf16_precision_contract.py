# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast precision-contract regressions for Qwen-VL BF16 accuracy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "tensorrt_model_connect",
    reason="Qwen-VL precision contracts require TensorRT",
)

from tensorrt_model_connect.families.qwen_vl import (  # noqa: E402
    graph_blocks,
    graph_ops,
)
from tensorrt_model_connect.families.qwen_vl.config import (  # noqa: E402
    get_rope_scaling,
)
from tensorrt_model_connect.families.qwen_vl.qwen_vl_vision_builder import (  # noqa: E402
    _interpolate_qwen3_position_bf16,
)


@dataclass
class _Tensor:
    name: str
    dtype: object
    shape: tuple[int, ...] = (1, 4)


class _Layer:
    def __init__(self, output: _Tensor) -> None:
        self._output = output

    def get_output(self, index: int) -> _Tensor:
        assert index == 0
        return self._output


class _Network:
    def __init__(self) -> None:
        self.operations: list[tuple] = []

    def add_cast(self, tensor: _Tensor, dtype: object) -> _Layer:
        self.operations.append(("cast", tensor.name, tensor.dtype, dtype))
        return _Layer(_Tensor(f"{tensor.name}_as_{dtype}", dtype, tensor.shape))

    def add_activation(self, tensor: _Tensor, operation: object) -> _Layer:
        self.operations.append(
            ("activation", tensor.name, tensor.dtype, operation))
        return _Layer(_Tensor("activation", tensor.dtype, tensor.shape))

    def add_elementwise(
        self, lhs: _Tensor, rhs: _Tensor, operation: object,
    ) -> _Layer:
        self.operations.append(
            ("elementwise", lhs.name, lhs.dtype, rhs.name, rhs.dtype, operation))
        assert lhs.dtype == rhs.dtype
        return _Layer(_Tensor("elementwise", lhs.dtype, lhs.shape))

    def add_shuffle(self, tensor: _Tensor) -> _Layer:
        self.operations.append(("shuffle", tensor.name, tensor.dtype))
        return _Layer(_Tensor("shuffle", tensor.dtype, tensor.shape))

    def add_rotary_embedding(
        self,
        tensor: _Tensor,
        cos: _Tensor,
        sin: _Tensor,
        interleaved: bool,
        rotary_embedding_dim: int,
    ) -> _Layer:
        self.operations.append(
            (
                "rotary",
                tensor.dtype,
                cos.dtype,
                sin.dtype,
                interleaved,
                rotary_embedding_dim,
            ))
        assert tensor.dtype == cos.dtype == sin.dtype
        return _Layer(_Tensor("rotary", tensor.dtype, tensor.shape))


def test_rope_scaling_reads_multimodal_text_config() -> None:
    nested = {
        "text_config": {
            "rope_scaling": {
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            },
        },
    }

    assert get_rope_scaling(nested) == nested["text_config"]["rope_scaling"]
    assert get_rope_scaling({"rope_scaling": {"rope_type": "default"}}) == {
        "rope_type": "default",
    }


def test_qwen3_position_interpolation_matches_bf16_operation_order() -> None:
    rng = np.random.default_rng(20260729)
    embeddings = rng.standard_normal((8, 16)).astype(np.float32)
    positions = (1, 3, 4, 7)
    weights = (0.1234, 0.2345, 0.3456, 0.2965)

    actual = _interpolate_qwen3_position_bf16(
        embeddings, positions, weights)

    embeddings_t = torch.from_numpy(embeddings).to(torch.bfloat16)
    weights_t = torch.tensor(weights, dtype=torch.bfloat16)
    expected = embeddings_t[positions[0]] * weights_t[0]
    for position, weight in zip(positions[1:], weights_t[1:]):
        expected = expected + embeddings_t[position] * weight
    expected = expected.float().numpy()

    np.testing.assert_array_equal(
        actual.view(np.uint32), expected.view(np.uint32))


def test_bf16_swiglu_uses_fp32_silu_then_bf16_up_product(
    monkeypatch,
) -> None:
    network = _Network()
    trt = graph_blocks.trt
    matmul_inputs: list[tuple[str, object]] = []

    def fake_make_matmul_fn(*_args, **_kwargs):
        def matmul(lhs, _lhs_w, _rhs_w, _weights, weight_name):
            matmul_inputs.append((weight_name, lhs.dtype))
            if weight_name.endswith("w_gate"):
                name = "gate"
            elif weight_name.endswith("w_up"):
                name = "up"
            else:
                name = "down"
            return _Tensor(name, lhs.dtype)

        return matmul

    monkeypatch.setattr(
        graph_blocks, "_make_matmul_fn", fake_make_matmul_fn)
    result = graph_blocks.add_swiglu_mlp(
        network,
        _Tensor("input", trt.bfloat16),
        weights={
            "mlp.w_gate": object(),
            "mlp.w_up": object(),
            "mlp.w_down": object(),
        },
        prefix="mlp",
        hidden_size=4,
        mlp_size=4,
        dtype=np.float16,
    )

    assert result.dtype == trt.bfloat16
    assert matmul_inputs == [
        ("mlp.w_gate", trt.bfloat16),
        ("mlp.w_up", trt.bfloat16),
        ("mlp.w_down", trt.bfloat16),
    ]
    assert network.operations[0] == (
        "cast", "gate", trt.bfloat16, trt.float32)
    assert network.operations[1] == (
        "activation",
        f"gate_as_{trt.float32}",
        trt.float32,
        trt.ActivationType.SIGMOID,
    )
    assert network.operations[-2] == (
        "cast", "elementwise", trt.float32, trt.bfloat16)
    assert network.operations[-1][0:5] == (
        "elementwise",
        f"elementwise_as_{trt.bfloat16}",
        trt.bfloat16,
        "up",
        trt.bfloat16,
    )


def test_bf16_silu_uses_fp32_internal_compute() -> None:
    network = _Network()
    trt = graph_ops.trt

    result = graph_ops.add_silu(
        network, _Tensor("input", trt.bfloat16))

    assert result.dtype == trt.bfloat16
    assert network.operations == [
        ("cast", "input", trt.bfloat16, trt.float32),
        (
            "activation",
            f"input_as_{trt.float32}",
            trt.float32,
            trt.ActivationType.SIGMOID,
        ),
        (
            "elementwise",
            f"input_as_{trt.float32}",
            trt.float32,
            "activation",
            trt.float32,
            trt.ElementWiseOperation.PROD,
        ),
        ("cast", "elementwise", trt.float32, trt.bfloat16),
    ]


def test_bf16_gelu_uses_fp32_internal_compute(monkeypatch) -> None:
    network = _Network()
    trt = graph_ops.trt

    def fake_constant(_network, shape, _values, dtype=np.float32):
        constant_dtype = trt.float32 if dtype == np.float32 else trt.float16
        return _Tensor("constant", constant_dtype, tuple(shape))

    monkeypatch.setattr(graph_ops, "add_constant", fake_constant)
    result = graph_ops.add_gelu_new(
        network, _Tensor("input", trt.bfloat16), dtype=np.float16)

    assert result.dtype == trt.bfloat16
    assert network.operations[0] == (
        "cast", "input", trt.bfloat16, trt.float32)
    assert network.operations[-1] == (
        "cast", "elementwise", trt.float32, trt.bfloat16)
    assert all(
        operation[2] == operation[4] == trt.float32
        for operation in network.operations
        if operation[0] == "elementwise"
    )


def test_bf16_vision_rope_computes_in_fp32_then_publishes_bf16() -> None:
    network = _Network()
    trt = graph_ops.trt
    output = graph_ops.add_apply_rope_native_sequence(
        network,
        _Tensor("q", trt.bfloat16, (4, 8)),
        num_heads=2,
        head_dim=4,
        cos_cache_3d=_Tensor("cos", trt.bfloat16, (1, 4, 2)),
        sin_cache_3d=_Tensor("sin", trt.bfloat16, (1, 4, 2)),
        rotary_embedding_dim=4,
        sequence_length=4,
    )

    assert output.dtype == trt.bfloat16
    assert network.operations[:3] == [
        ("cast", "q", trt.bfloat16, trt.float32),
        ("cast", "cos", trt.bfloat16, trt.float32),
        ("cast", "sin", trt.bfloat16, trt.float32),
    ]
    rotary = next(
        operation for operation in network.operations
        if operation[0] == "rotary")
    assert rotary == (
        "rotary",
        trt.float32,
        trt.float32,
        trt.float32,
        False,
        4,
    )
    assert network.operations[-1] == (
        "cast", "shuffle", trt.float32, trt.bfloat16)
