# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for communication-free Cosmos3 replicated-row sharding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.cosmos3.trt_ops import select_replicated_rows


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Layer:
    def __init__(self, output: _Tensor) -> None:
        self._output = output

    def get_output(self, index: int) -> _Tensor:
        assert index == 0
        return self._output


class _Shuffle(_Layer):
    def __init__(self, source: _Tensor) -> None:
        super().__init__(_Tensor(tuple(source.shape)))

    @property
    def reshape_dims(self) -> tuple[int, ...]:
        return self._output.shape

    @reshape_dims.setter
    def reshape_dims(self, shape: tuple[int, ...]) -> None:
        self._output.shape = tuple(shape)


class _Network:
    def __init__(self) -> None:
        self.shuffles: list[_Shuffle] = []
        self.gathers: list[tuple[tuple[int, ...], _Tensor, int]] = []

    def add_shuffle(self, tensor: _Tensor) -> _Shuffle:
        layer = _Shuffle(tensor)
        self.shuffles.append(layer)
        return layer

    def add_gather(self, data: _Tensor, indices: _Tensor, axis: int) -> _Layer:
        self.gathers.append((tuple(data.shape), indices, axis))
        assert axis == 0
        return _Layer(_Tensor((indices.shape[0], *data.shape[1:])))


def test_select_replicated_rows_builds_static_cp8_rank_layout() -> None:
    network = _Network()
    rank = _Tensor((1,))

    output = select_replicated_rows(
        network,
        _Tensor((44_160, 4_096)),
        rank,
        8,
    )

    assert network.shuffles[0].reshape_dims == (8, 5_520, 4_096)
    assert network.gathers == [((8, 5_520, 4_096), rank, 0)]
    assert network.shuffles[1].reshape_dims == (5_520, 4_096)
    assert output.shape == (5_520, 4_096)


def test_rank_major_gather_matches_contiguous_reduce_scatter_layout() -> None:
    full = np.arange(32, dtype=np.int32).reshape(16, 2)
    rank_major = full.reshape(8, 2, 2)

    for rank in range(8):
        np.testing.assert_array_equal(rank_major[rank], full[rank * 2 : (rank + 1) * 2])


@pytest.mark.parametrize("shape", [(), (15, 2), (0, 2)])
def test_select_replicated_rows_rejects_invalid_leading_dimension(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="leading dimension"):
        select_replicated_rows(_Network(), _Tensor(shape), _Tensor((1,)), 8)


def test_select_replicated_rows_requires_single_rank_value() -> None:
    with pytest.raises(ValueError, match="rank input"):
        select_replicated_rows(_Network(), _Tensor((16, 2)), _Tensor((2,)), 8)


def test_cosmos3_runtime_binds_declared_context_parallel_rank_input() -> None:
    root = Path(__file__).resolve().parents[4]
    source = (root / "src/runtime/models/cosmos3/pipeline.cpp").read_text(encoding="utf-8")

    assert 'constexpr char kContextParallelRankInput[] = "context_parallel_rank";' in source
    assert "if (denoiser.has_input(kContextParallelRankInput))" in source
    assert (
        "!has_input_contract(module, kContextParallelRankInput, {1}, DType::kInt32)"
        in source
    )
    # The rank now lives on the pipeline because forward_async only guarantees
    # the H2D copy has completed after sync; a stack-local value would not have
    # a sufficient lifetime for the CFG-parallel enqueue path.
    assert "int32_t context_parallel_rank_{0};" in (
        root / "src/runtime/models/cosmos3/pipeline.h"
    ).read_text(encoding="utf-8")
    assert (
        "Tensor{const_cast<int32_t*>(&context_parallel_rank_), {1}, DType::kInt32}"
        in source
    )
