# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the e2e_harness ``MODEL.toml`` loaders (standards-compliant parsing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import (
    _read_model_index,
    _read_runtime_model_manifest,
    tomllib,
)


def test_read_model_index_parses_edge_cases(tmp_path: Path):
    """quoted #, comments, and multiline lists parse correctly (the old flat
    extractor split on the first ``#`` and could corrupt such values)."""
    index = tmp_path / "MODEL.toml"
    index.write_text(
        "# leading comment\n"
        'note = "path with a # hash"        # trailing comment\n'
        "test_manifests = [\n"
        '    "manifests/a.toml",            # inline comment\n'
        '    "manifests/b.toml",\n'
        "]\n",
        encoding="utf-8",
    )

    data = _read_model_index(index)

    assert data["test_manifests"] == ["manifests/a.toml", "manifests/b.toml"]
    assert data["note"] == "path with a # hash"


def test_read_model_index_malformed_fails_closed(tmp_path: Path):
    """A malformed index raises rather than being silently misread."""
    index = tmp_path / "MODEL.toml"
    index.write_text('test_manifests = ["unterminated\n', encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        _read_model_index(index)


def test_read_runtime_model_manifest_list_and_scalar(tmp_path: Path):
    """Both the ``runtime_strategies`` list and the scalar ``runtime_strategy`` are read."""
    multi = tmp_path / "multi.toml"
    multi.write_text('runtime_strategies = ["torch_trt", "trt_llm"]\n', encoding="utf-8")
    assert _read_runtime_model_manifest(multi)["runtime_strategies"] == ["torch_trt", "trt_llm"]

    scalar = tmp_path / "scalar.toml"
    scalar.write_text('runtime_strategy = "solo"\n', encoding="utf-8")
    assert _read_runtime_model_manifest(scalar)["runtime_strategy"] == "solo"


def test_read_runtime_model_manifest_malformed_fails_closed(tmp_path: Path):
    """A malformed manifest raises rather than silently yielding a partial parse."""
    bad = tmp_path / "MODEL.toml"
    bad.write_text('runtime_strategies = ["unterminated\n', encoding="utf-8")

    with pytest.raises(tomllib.TOMLDecodeError):
        _read_runtime_model_manifest(bad)


def test_known_runtime_strategies_fails_closed_on_malformed_manifest(tmp_path, monkeypatch):
    """The real caller (_known_runtime_strategies) must fail closed on a malformed
    runtime MODEL.toml instead of logging a warning and silently omitting its
    strategies, which would return an incomplete set."""
    from tests.e2e_harness import manifest_loader as ml

    models_dir = tmp_path / "models"
    (models_dir / "bad").mkdir(parents=True)
    (models_dir / "bad" / "MODEL.toml").write_text('runtime_strategies = ["oops\n', encoding="utf-8")

    monkeypatch.setattr(ml, "_runtime_model_manifests_dir", lambda: models_dir)
    monkeypatch.setattr(ml, "_KNOWN_RUNTIME_STRATEGIES_CACHE", None)

    with pytest.raises(ValueError):
        ml._known_runtime_strategies()
