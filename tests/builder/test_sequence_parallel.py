"""Tests for sequence-parallel primitives and ParallelConfig SP fields.

The collective primitives are validated against stub network/tensor/layer
objects that mirror the TRT 11 ``add_dist_collective`` contract. The stubs
record the arguments TRT would receive and let us assert the declared output
shape without spinning up a real builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    add_all_gather,
    add_all_reduce_sum,
    add_all_to_all,
    add_reduce_scatter_sum,
)


# ---------------------------------------------------------------------------
# Lightweight stubs that imitate the TRT 11 collective layer surface
# ---------------------------------------------------------------------------


@dataclass
class _FakeTensor:
    shape: tuple[int, ...]


@dataclass
class _FakeCollectiveLayer:
    inp: _FakeTensor
    op: Any
    reduce_op: Any
    axis: Any
    sizes: list[int]
    num_ranks: int = 0
    output_shape: tuple[int, ...] = field(default_factory=tuple)

    def get_output(self, index: int) -> _FakeTensor:
        assert index == 0
        return _FakeTensor(shape=self.output_shape)


class _FakeCollectiveOperation:
    ALL_REDUCE = "ALL_REDUCE"
    ALL_GATHER = "ALL_GATHER"
    ALL_TO_ALL = "ALL_TO_ALL"
    REDUCE_SCATTER = "REDUCE_SCATTER"


class _FakeReduceOperation:
    SUM = "SUM"
    NONE = "NONE"


class _FakeTrtModule:
    CollectiveOperation = _FakeCollectiveOperation
    ReduceOperation = _FakeReduceOperation
    __version__ = "11.0.0.114"


@dataclass
class _FakeNetwork:
    """Records collective calls and emits stub layers with computed shapes."""

    calls: list[dict] = field(default_factory=list)
    reject_list_axis: bool = False

    def add_dist_collective(self, tensor, op, reduce_op, axis, sizes):
        if self.reject_list_axis and isinstance(axis, list):
            raise TypeError("test stub rejects list-axis form")
        call = {
            "tensor": tensor,
            "op": op,
            "reduce_op": reduce_op,
            "axis": axis,
            "sizes": list(sizes),
        }
        self.calls.append(call)
        layer = _FakeCollectiveLayer(
            inp=tensor,
            op=op,
            reduce_op=reduce_op,
            axis=axis,
            sizes=list(sizes),
            output_shape=tuple(tensor.shape),
        )
        # The helpers set num_ranks *after* construction; wrap so the setter
        # recomputes the declared output shape.
        return _RecordingLayer(layer, op_kind=op, axis=axis, sizes=sizes)


class _RecordingLayer:
    """Wrap _FakeCollectiveLayer so num_ranks assignment recomputes output."""

    def __init__(self, layer: _FakeCollectiveLayer, op_kind, axis, sizes):
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_op_kind", op_kind)
        object.__setattr__(self, "_axis", axis)
        object.__setattr__(self, "_sizes", list(sizes))

    @property
    def num_ranks(self) -> int:
        return self._layer.num_ranks

    @num_ranks.setter
    def num_ranks(self, value: int) -> None:
        self._layer.num_ranks = int(value)
        self._recompute_shape()

    def _recompute_shape(self) -> None:
        shape = list(self._layer.inp.shape)
        n = self._layer.num_ranks
        kind = self._op_kind
        axis = self._axis
        if kind == _FakeCollectiveOperation.ALL_REDUCE:
            pass  # shape preserved
        elif kind == _FakeCollectiveOperation.ALL_GATHER:
            ax = axis if axis >= 0 else len(shape) + axis
            shape[ax] *= n
        elif kind == _FakeCollectiveOperation.REDUCE_SCATTER:
            ax = axis if axis >= 0 else len(shape) + axis
            shape[ax] //= n
        elif kind == _FakeCollectiveOperation.ALL_TO_ALL:
            if isinstance(axis, (list, tuple)):
                scatter_ax, gather_ax = int(axis[0]), int(axis[1])
            else:
                scatter_ax = int(axis)
                gather_ax = int(self._sizes[0])
            scatter_ax = scatter_ax if scatter_ax >= 0 else len(shape) + scatter_ax
            gather_ax = gather_ax if gather_ax >= 0 else len(shape) + gather_ax
            shape[scatter_ax] *= n
            shape[gather_ax] //= n
        self._layer.output_shape = tuple(shape)

    def get_output(self, index: int) -> _FakeTensor:
        return self._layer.get_output(index)


@pytest.fixture
def fake_trt(monkeypatch):
    """Patch trt_compat.get_trt + is_available so helpers see our fake module."""
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "get_trt", lambda: _FakeTrtModule())
    monkeypatch.setattr(trt_compat, "is_available", lambda module_name=None: True)
    monkeypatch.setattr(trt_compat, "load_module", lambda: _FakeTrtModule())
    return _FakeTrtModule()


# ---------------------------------------------------------------------------
# ParallelConfig sequence-parallel validation
# ---------------------------------------------------------------------------


def test_parallel_config_defaults_to_cp_size_one() -> None:
    cfg = ParallelConfig()
    cfg.validate()
    assert cfg.cp_size == 1
    assert cfg.world_size == 1
    assert cfg.enabled is False


def test_parallel_config_accepts_sp_ulysses_with_cp_size() -> None:
    cfg = ParallelConfig(mode="sp_ulysses", cp_size=4, rank=2)
    cfg.validate()
    assert cfg.enabled is True
    assert cfg.world_size == 4


def test_parallel_config_accepts_sp_ring_and_allgather_kv() -> None:
    for mode in ("sp_ring", "sp_allgather_kv"):
        cfg = ParallelConfig(mode=mode, cp_size=2, rank=0)
        cfg.validate()
        assert cfg.enabled is True


def test_parallel_config_rejects_cp_size_with_tensor_parallel_mode() -> None:
    with pytest.raises(ValueError, match="cp_size"):
        ParallelConfig(mode="tensor_parallel", tp_size=2, cp_size=2).validate()


def test_parallel_config_rejects_cp_size_with_single_mode() -> None:
    with pytest.raises(ValueError, match="cp_size"):
        ParallelConfig(mode="single", cp_size=2).validate()


def test_parallel_config_rejects_sp_mode_without_cp_size() -> None:
    with pytest.raises(ValueError, match="cp_size"):
        ParallelConfig(mode="sp_ulysses", cp_size=1).validate()


def test_parallel_config_rejects_non_power_of_two_cp_size() -> None:
    with pytest.raises(ValueError, match="cp_size"):
        ParallelConfig(mode="sp_ring", cp_size=3).validate()


def test_parallel_config_rejects_cp_size_too_large() -> None:
    with pytest.raises(ValueError, match="cp_size"):
        ParallelConfig(mode="sp_ring", cp_size=16).validate()


def test_parallel_config_rejects_zero_cp_size() -> None:
    with pytest.raises(ValueError, match="cp_size"):
        ParallelConfig(mode="sp_ring", cp_size=0).validate()


def test_parallel_config_allows_hybrid_tp_and_cp() -> None:
    # Hybrid TP + SP is allowed for future work, even though no caller
    # consumes it yet. The dataclass must not reject simultaneous tp_size > 1
    # and cp_size > 1.
    cfg = ParallelConfig(mode="sp_ulysses", tp_size=2, cp_size=2, rank=3)
    cfg.validate()
    assert cfg.enabled is True
    assert cfg.world_size == 4


def test_parallel_config_world_size_hybrid_product() -> None:
    cfg = ParallelConfig(mode="sp_ring", tp_size=4, cp_size=2, rank=0)
    cfg.validate()
    assert cfg.world_size == 8


def test_parallel_config_to_config_dict_includes_cp_size() -> None:
    cfg = ParallelConfig(mode="sp_ulysses", cp_size=2, rank=0)
    data = cfg.to_config_dict()
    assert data["cp_size"] == 2
    assert data["mode"] == "sp_ulysses"


# ---------------------------------------------------------------------------
# Collective primitive shape declarations
# ---------------------------------------------------------------------------


def test_add_all_gather_grows_axis_by_cp_size(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    out = add_all_gather(net, inp, cp_size=2, gather_axis=-1)

    assert isinstance(out, _FakeTensor)
    assert out.shape == (1, 4, 32)
    assert len(net.calls) == 1
    call = net.calls[0]
    assert call["op"] == _FakeCollectiveOperation.ALL_GATHER
    assert call["reduce_op"] == _FakeReduceOperation.NONE
    assert call["axis"] == -1
    assert call["sizes"] == []


def test_add_all_gather_pass_through_when_cp_size_one(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    out = add_all_gather(net, inp, cp_size=1, gather_axis=-1)

    assert out is inp
    assert net.calls == []


def test_add_all_to_all_redistributes_axes(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    out = add_all_to_all(net, inp, cp_size=2, scatter_axis=1, gather_axis=2)

    assert out.shape == (1, 8, 8)
    call = net.calls[0]
    assert call["op"] == _FakeCollectiveOperation.ALL_TO_ALL
    assert call["axis"] == [1, 2]


def test_add_all_to_all_falls_back_to_scalar_axis(fake_trt) -> None:
    net = _FakeNetwork(reject_list_axis=True)
    inp = _FakeTensor(shape=(1, 4, 16))

    out = add_all_to_all(net, inp, cp_size=2, scatter_axis=1, gather_axis=2)

    assert out.shape == (1, 8, 8)
    # First call attempted the list form and the stub raised TypeError.
    # Only the successful scalar call is recorded.
    assert len(net.calls) == 1
    call = net.calls[0]
    assert call["axis"] == 1
    assert call["sizes"] == [2]


def test_add_all_to_all_rejects_identical_axes(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    with pytest.raises(RuntimeError, match="must differ"):
        add_all_to_all(net, inp, cp_size=2, scatter_axis=1, gather_axis=1)


def test_add_reduce_scatter_sum_shrinks_axis(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    out = add_reduce_scatter_sum(net, inp, cp_size=2, scatter_axis=-1)

    assert out.shape == (1, 4, 8)
    call = net.calls[0]
    assert call["op"] == _FakeCollectiveOperation.REDUCE_SCATTER
    assert call["reduce_op"] == _FakeReduceOperation.SUM


def test_add_all_reduce_sum_preserves_shape(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    out = add_all_reduce_sum(net, inp, tp_size=2)

    assert out.shape == (1, 4, 16)
    call = net.calls[0]
    assert call["op"] == _FakeCollectiveOperation.ALL_REDUCE
    assert call["reduce_op"] == _FakeReduceOperation.SUM
    assert call["axis"] == -1


# ---------------------------------------------------------------------------
# Defensive error paths
# ---------------------------------------------------------------------------


def test_primitives_raise_when_add_dist_collective_missing(fake_trt) -> None:
    class _NoCollectiveNet:
        pass

    inp = _FakeTensor(shape=(1, 4, 16))

    with pytest.raises(RuntimeError, match="TRT 11"):
        add_all_gather(_NoCollectiveNet(), inp, cp_size=2)
    with pytest.raises(RuntimeError, match="TRT 11"):
        add_all_to_all(
            _NoCollectiveNet(), inp, cp_size=2, scatter_axis=1, gather_axis=2)
    with pytest.raises(RuntimeError, match="TRT 11"):
        add_reduce_scatter_sum(_NoCollectiveNet(), inp, cp_size=2)


def test_primitives_reject_negative_parallel_size(fake_trt) -> None:
    net = _FakeNetwork()
    inp = _FakeTensor(shape=(1, 4, 16))

    with pytest.raises(RuntimeError, match="parallel_size"):
        add_all_gather(net, inp, cp_size=-1)
    with pytest.raises(RuntimeError, match="parallel_size"):
        add_all_to_all(net, inp, cp_size=-2, scatter_axis=1, gather_axis=2)
    with pytest.raises(RuntimeError, match="parallel_size"):
        add_reduce_scatter_sum(net, inp, cp_size=-3)
