"""Tests for tensor-parallel build config and sharding helpers."""

from __future__ import annotations

import pytest

from tensorrt_model_connect.distributed_plan import DISTRIBUTED_PLAN_SECTION
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
    parallel_config_from_bundle,
    require_tensorrt_11_for_tensor_parallel,
)
from tensorrt_model_connect.runtime_config import clear_for_testing, resolve_cli_config
from tensorrt_model_connect.runtime_config.schemas import load_all


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_for_testing()
    yield
    clear_for_testing()


def test_parallel_config_resolves_from_cli_sets() -> None:
    load_all()
    bundle = resolve_cli_config(
        config_path=None,
        set_tokens=[
            "parallel.mode=tensor_parallel",
            "parallel.tp_size=4",
            "parallel.rank=2",
        ],
    )

    cfg = parallel_config_from_bundle(bundle)

    assert cfg.enabled
    assert cfg.tp_size == 4
    assert cfg.rank == 2


def test_parallel_config_rejects_unsupported_tp_size() -> None:
    load_all()

    with pytest.raises(ValueError, match="Validator rejected"):
        resolve_cli_config(
            config_path=None,
            set_tokens=[
                "parallel.mode=tensor_parallel",
                "parallel.tp_size=3",
            ],
        )


def test_parallel_config_points_to_distributed_plan_without_legacy_runtime_fields() -> None:
    fields = ParallelConfig(mode="tensor_parallel", tp_size=2).to_bundle_config_fields()

    assert fields["parallelism"]["mode"] == "tensor_parallel"
    assert fields["parallelism"]["tp_size"] == 2
    assert fields["distributed_plan_section"] == DISTRIBUTED_PLAN_SECTION
    assert "tensor_parallel_mode" not in fields
    assert "tensor_parallel_size" not in fields
    assert "tensor_parallel_require_mpirun" not in fields


def test_tensor_parallel_requires_trt11(monkeypatch) -> None:
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "10.13.3")

    with pytest.raises(RuntimeError, match="TensorRT 11\\.0\\+"):
        require_tensorrt_11_for_tensor_parallel(
            ParallelConfig(mode="tensor_parallel", tp_size=2))


def test_tensor_parallel_trt11_guard_ignores_single_device(monkeypatch) -> None:
    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "10.13.3")

    require_tensorrt_11_for_tensor_parallel(ParallelConfig())
