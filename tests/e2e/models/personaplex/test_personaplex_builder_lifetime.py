# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builder lifetime contracts owned by PersonaPlex."""

from __future__ import annotations

import gc
import importlib
import inspect
import weakref
from types import SimpleNamespace

import pytest


pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")
pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires TensorRT",
)


personaplex_utils = importlib.import_module(
    "tensorrt_model_connect.families.personaplex.utils")


@pytest.fixture(autouse=True)
def _reset_process_logger(monkeypatch) -> None:
    monkeypatch.setattr(personaplex_utils, "_PROCESS_LOGGER", None)


class _TrackedResource:
    def __init__(self, name: str, released: list[str]) -> None:
        self._name = name
        self._released = released

    def __del__(self) -> None:
        self._released.append(self._name)


def test_builder_context_closes_once_in_tensor_rt_lifetime_order() -> None:
    released: list[str] = []
    context = personaplex_utils.BuilderContext(
        logger=_TrackedResource("logger", released),
        builder=_TrackedResource("builder", released),
        network=_TrackedResource("network", released),
        config=_TrackedResource("config", released),
    )

    context.close()
    context.close()

    assert released == ["config", "network", "builder", "logger"]


@pytest.mark.parametrize(
    ("failure_stage", "expected_released"),
    (
        ("builder", []),
        ("network", ["builder"]),
        ("config", ["network", "builder"]),
        ("workspace", ["config", "network", "builder"]),
    ),
)
def test_create_builder_context_cleans_up_partial_construction(
    monkeypatch,
    failure_stage: str,
    expected_released: list[str],
) -> None:
    released: list[str] = []

    class FakeLogger(_TrackedResource):
        VERBOSE = 1
        WARNING = 2

        def __init__(self, _level: int) -> None:
            super().__init__("logger", released)

    class FakeConfig(_TrackedResource):
        def __init__(self) -> None:
            super().__init__("config", released)
            self.set_memory_pool_limit = fail_workspace

    class FakeBuilder(_TrackedResource):
        def __init__(self) -> None:
            super().__init__("builder", released)
            self.create_network = create_network
            self.create_builder_config = create_config

    def fail(message: str):
        raise RuntimeError(message)

    def make_builder(_logger):
        if failure_stage == "builder":
            fail("builder")
        return FakeBuilder()

    def create_network(_flags):
        if failure_stage == "network":
            fail("network")
        return _TrackedResource("network", released)

    def create_config():
        if failure_stage == "config":
            fail("config")
        return FakeConfig()

    def fail_workspace(_pool_type, _workspace_bytes):
        if failure_stage == "workspace":
            fail("workspace")

    fake_trt = SimpleNamespace(
        Logger=FakeLogger,
        Builder=make_builder,
        MemoryPoolType=SimpleNamespace(WORKSPACE=object()),
    )
    monkeypatch.setattr(personaplex_utils, "trt", fake_trt)
    monkeypatch.setattr(
        personaplex_utils.trt_compat,
        "network_creation_flags",
        lambda **_kwargs: 0,
    )

    with pytest.raises(RuntimeError, match=failure_stage):
        personaplex_utils.create_builder_context(
            verbose=False,
            workspace_bytes=1234,
        )
    gc.collect()

    assert released == expected_released


def test_create_builder_context_reuses_process_lifetime_logger(monkeypatch) -> None:
    builder_loggers: list[weakref.ReferenceType] = []

    class FakeLogger:
        VERBOSE = 1
        WARNING = 2

        def __init__(self, _level: int) -> None:
            pass

    class FakeConfig:
        def set_memory_pool_limit(self, _pool_type, _workspace_bytes) -> None:
            pass

    class FakeBuilder:
        def __init__(self, logger) -> None:
            builder_loggers.append(weakref.ref(logger))

        def create_network(self, _flags):
            return object()

        def create_builder_config(self):
            return FakeConfig()

    fake_trt = SimpleNamespace(
        Logger=FakeLogger,
        Builder=FakeBuilder,
        MemoryPoolType=SimpleNamespace(WORKSPACE=object()),
    )
    monkeypatch.setattr(personaplex_utils, "trt", fake_trt)
    monkeypatch.setattr(
        personaplex_utils.trt_compat,
        "network_creation_flags",
        lambda **_kwargs: 0,
    )

    first = personaplex_utils.create_builder_context(
        verbose=False,
        workspace_bytes=1234,
    )
    logger_ref = weakref.ref(first.logger)
    first.close()
    gc.collect()

    second = personaplex_utils.create_builder_context(
        verbose=True,
        workspace_bytes=1234,
    )

    assert logger_ref() is not None
    assert [logger() for logger in builder_loggers] == [logger_ref(), logger_ref()]
    assert second.logger is logger_ref()
    second.close()
    gc.collect()
    assert logger_ref() is not None


def test_builder_context_wrapper_closes_after_an_exception(monkeypatch) -> None:
    released: list[str] = []

    class FakeContext:
        close_count = 0

        def close(self) -> None:
            self.close_count += 1
            released.append("context")

    context = FakeContext()
    create_calls: list[dict] = []

    def fake_create_builder_context(**kwargs):
        create_calls.append(kwargs)
        return context

    monkeypatch.setattr(
        personaplex_utils, "create_builder_context", fake_create_builder_context)

    @personaplex_utils.with_builder_context(
        workspace_bytes=1234,
        explicit_batch=True,
    )
    def failing_builder(*, verbose=False, _builder_context_factory):
        assert _builder_context_factory() is context
        frame_resource = _TrackedResource("frame", released)
        plan = _TrackedResource("plan", released)
        closure_resource = _TrackedResource("closure", released)

        def closure():
            return closure_resource

        assert frame_resource is not None
        assert plan is not None
        assert closure() is closure_resource
        raise RuntimeError("synthetic TensorRT build failure")

    with pytest.raises(RuntimeError, match="synthetic TensorRT build failure"):
        failing_builder(verbose=True)

    assert create_calls == [{
        "verbose": True,
        "workspace_bytes": 1234,
        "strongly_typed": True,
        "explicit_batch": True,
        "disable_tf32": False,
    }]
    assert context.close_count == 1
    assert released[-1] == "context"
    assert sorted(released[:-1]) == ["closure", "frame", "plan"]


def test_builder_context_wrapper_unwinds_builder_frame_before_close(
    monkeypatch,
) -> None:
    released: list[str] = []

    class FakeContext:
        def close(self) -> None:
            released.append("context")

    monkeypatch.setattr(
        personaplex_utils,
        "create_builder_context",
        lambda **_kwargs: FakeContext(),
    )

    @personaplex_utils.with_builder_context(workspace_bytes=1)
    def successful_builder(*, _builder_context_factory):
        _builder_context_factory()
        frame_resource = _TrackedResource("frame", released)
        plan = _TrackedResource("plan", released)
        closure_resource = _TrackedResource("closure", released)

        def closure():
            return closure_resource

        assert frame_resource is not None
        assert plan is not None
        assert closure() is closure_resource
        return b"serialized plan"

    assert successful_builder() == b"serialized plan"
    assert released[-1] == "context"
    assert sorted(released[:-1]) == ["closure", "frame", "plan"]


def test_builder_context_wrapper_is_lazy(monkeypatch) -> None:
    def unexpected_create(**_kwargs):
        pytest.fail("dispatch-only paths must not create a TensorRT builder")

    monkeypatch.setattr(
        personaplex_utils, "create_builder_context", unexpected_create)

    @personaplex_utils.with_builder_context(workspace_bytes=1)
    def dispatch_only(*, _builder_context_factory):
        return b"dispatched"

    assert dispatch_only() == b"dispatched"


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    (
        ("default_decoder", "build_standard_decoder_engine"),
        ("default_dual_profile_decoder", "build_dual_profile_decoder_engine"),
        ("default_dual_profile_decoder_tp", "build_dual_profile_tp_decoder_engine"),
        ("decoder_tp_builder", "build_personaplex_tp_decoder_engine"),
        ("plugin", "_build_mimi_encoder_engine"),
        ("plugin", "_build_mimi_decoder_engine"),
    ),
)
def test_personaplex_builders_use_ordered_context_cleanup(
    module_name: str,
    function_name: str,
) -> None:
    module = importlib.import_module(
        f"tensorrt_model_connect.families.personaplex.{module_name}")
    builder = getattr(module, function_name)

    assert builder._trtmc_ordered_builder_context is True
    assert "_builder_context_factory" not in inspect.signature(builder).parameters
