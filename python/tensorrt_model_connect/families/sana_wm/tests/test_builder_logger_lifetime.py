# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for SANA-WM's process-registered TensorRT logger."""

from __future__ import annotations

import gc
from pathlib import Path
import weakref

from tensorrt_model_connect.families.sana_wm import builder_lifetime


_BUILDER_MODULES = (
    "stage1_dit_builder.py",
    "refiner_text_connector_builder.py",
    "refiner_dit_builder.py",
    "components/gemma/utils.py",
    "components/gemma/standard_decoder_builder.py",
    "components/gemma/dual_profile_decoder_tp_builder.py",
    "components/gemma/dual_profile_decoder_builder.py",
    "components/ltx_video/utils.py",
    "components/ltx_video/t5_encoder_builder.py",
    "components/ltx_video/ltx_vae_builder.py",
    "components/ltx_video/ltx_dit_builder.py",
)


def test_process_logger_is_reused_and_retained() -> None:
    created: list[object] = []

    class FakeLogger:
        VERBOSE = 1
        WARNING = 2

        def __init__(self, severity: int) -> None:
            self.severity = severity
            created.append(self)

    class FakeTrt:
        Logger = FakeLogger

    builder_lifetime._PROCESS_LOGGERS.pop(FakeTrt, None)
    first = builder_lifetime.get_process_trt_logger(FakeTrt, verbose=False)
    logger_ref = weakref.ref(first)
    del first
    gc.collect()

    second = builder_lifetime.get_process_trt_logger(FakeTrt, verbose=True)

    assert logger_ref() is second
    assert created == [second]
    assert second.severity == FakeLogger.WARNING
    builder_lifetime._PROCESS_LOGGERS.pop(FakeTrt, None)


def test_all_sana_builders_use_the_process_logger() -> None:
    family_dir = Path(__file__).parents[1]

    for relative_path in _BUILDER_MODULES:
        source = (family_dir / relative_path).read_text(encoding="utf-8")
        assert ".Logger(" not in source, relative_path
        assert "get_process_trt_logger(" in source, relative_path
