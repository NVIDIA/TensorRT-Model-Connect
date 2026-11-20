# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from families.sam2.image_builder import populate_image_network  # noqa: E402
from families.sam2.model import _Sam2Model  # noqa: E402
from families.sam2.support import describe  # noqa: E402
from families.sam2.tracker_builder import populate_tracker_network  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


class _Weights:
    """Deterministic stand-in for the checkpoint: graph structure only."""

    def tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name.endswith("running_var"):
            return np.ones(shape, dtype=np.float32)
        seed = sum(name.encode()) % 997
        return np.random.default_rng(seed).normal(0.0, 0.02, shape).astype(np.float32)


def _network(populate, half):
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(0)
    owner = populate(trt, network, _Weights(), half)
    return builder, network, owner


def _io(network) -> list[tuple[str, object, tuple[int, ...]]]:
    tensors = [network.get_input(i) for i in range(network.num_inputs)]
    tensors += [network.get_output(i) for i in range(network.num_outputs)]
    return [(tensor.name, tensor.dtype, tuple(tensor.shape)) for tensor in tensors]


def _matmul_input_types(network) -> set:
    types = set()
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type == trt.LayerType.MATRIX_MULTIPLY:
            types.update(layer.get_input(i).dtype for i in range(layer.num_inputs))
    return types


def test_default_precision_is_fp16() -> None:
    assert _Sam2Model.default_build_precision == "fp16"
    support = describe(ModelMetadata({"model_type": "sam2"}, {}))
    assert support is not None and support.default_precision == "fp16"


@pytest.mark.parametrize(
    ("precision", "half"),
    [("fp16", "float16"), ("mixed_bf16_fp32", "bfloat16")],
)
def test_precision_selects_the_compute_type(precision: str, half: str) -> None:
    assert _Sam2Model._require_precision(precision) == getattr(trt, half)


@pytest.mark.parametrize("precision", ["bf16", "fp32", "mixed_fp16_fp32"])
def test_unsupported_precision_is_rejected(precision: str) -> None:
    with pytest.raises(ValueError, match="SAM2 supports only"):
        _Sam2Model._require_precision(precision)


@pytest.mark.parametrize(
    "populate",
    [
        populate_image_network,
        lambda t, n, c, half: populate_tracker_network(t, n, c, 0, half),
        lambda t, n, c, half: populate_tracker_network(t, n, c, 4, half),
    ],
    ids=["image", "prompt", "recurrent4"],
)
def test_fp16_keeps_the_runtime_contract(populate) -> None:
    _, bf16, bf16_owner = _network(populate, trt.bfloat16)
    _, fp16, fp16_owner = _network(populate, trt.float16)
    assert _io(fp16) == _io(bf16)
    assert trt.bfloat16 in {dtype for _, dtype, _ in _io(fp16)}
    assert _matmul_input_types(bf16) <= {trt.bfloat16, trt.float32}
    assert trt.float16 in _matmul_input_types(fp16)
    assert trt.bfloat16 not in _matmul_input_types(fp16)
    del bf16_owner, fp16_owner
