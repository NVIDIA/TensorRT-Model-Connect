# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for explicit structured build timing output."""

from __future__ import annotations

import json

from tensorrt_model_connect.build_timing import add_build_timing, new_build_timing, write_build_timing


def test_write_build_timing_uses_explicit_output_path(tmp_path) -> None:
    output_path = tmp_path / "timing.json"
    timing = new_build_timing(output_path)

    add_build_timing(timing, "weights_loading_s", 1.25)
    write_build_timing(timing)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "phases": {"weights_loading_s": 1.25},
    }


def test_write_build_timing_accepts_call_site_output_path(tmp_path) -> None:
    output_path = tmp_path / "timing.json"
    timing = new_build_timing()

    add_build_timing(timing, "trt_compile_s", 2.5)
    write_build_timing(timing, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["phases"] == {"trt_compile_s": 2.5}


def test_write_build_timing_without_output_path_is_noop(tmp_path) -> None:
    timing = new_build_timing()
    add_build_timing(timing, "bundle_write_s", 0.5)

    write_build_timing(timing)

    assert list(tmp_path.iterdir()) == []
