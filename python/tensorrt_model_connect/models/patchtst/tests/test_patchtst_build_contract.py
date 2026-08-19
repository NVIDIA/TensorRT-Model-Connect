# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned CI build contracts for PatchTST."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.models.patchtst import time_series_trt


_MANIFEST_DIR = Path(__file__).parent / "manifests"


@pytest.mark.parametrize(
    "manifest_name",
    (
        "patchtst-etth1-regression-distribution.json",
        "patchtst-granite-official.json",
    ),
)
def test_single_gpu_validation_reserves_an_exclusive_gpu(manifest_name: str) -> None:
    manifest = json.loads((_MANIFEST_DIR / manifest_name).read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


def test_tensor_rt_logger_is_reused_and_retained(monkeypatch) -> None:
    builder_loggers: list[weakref.ReferenceType] = []

    class FakeLogger:
        VERBOSE = 1
        WARNING = 2

        def __init__(self, _level: int) -> None:
            pass

    class FakeBuilder:
        def __init__(self, logger) -> None:
            builder_loggers.append(weakref.ref(logger))

        def create_network(self, _flags):
            return object()

    fake_trt = SimpleNamespace(Logger=FakeLogger, Builder=FakeBuilder)
    monkeypatch.setattr(time_series_trt, "trt", fake_trt)
    monkeypatch.setattr(time_series_trt, "_PROCESS_LOGGER", None)
    monkeypatch.setattr(
        time_series_trt.trt_compat,
        "network_creation_flags",
        lambda **_kwargs: 0,
    )

    first_builder, _first_network = time_series_trt.create_network(verbose=False)
    del first_builder
    gc.collect()
    retained_logger = builder_loggers[0]()

    second_builder, _second_network = time_series_trt.create_network(verbose=True)
    del second_builder
    gc.collect()

    assert retained_logger is not None
    assert [logger() for logger in builder_loggers] == [
        retained_logger,
        retained_logger,
    ]
