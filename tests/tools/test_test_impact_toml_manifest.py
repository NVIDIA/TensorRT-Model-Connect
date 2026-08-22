# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for test_impact's runtime-strategy MODEL.toml parser (standards-compliant)."""

import sys
from pathlib import Path

import pytest

# Add tools/ to path so we can import test_impact
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import test_impact  # noqa: E402


def test_parse_runtime_model_manifest_toml_edge_cases(tmp_path):
    """A real TOML parser reads quoted #, comments, and multiline lists correctly."""
    manifest = tmp_path / "MODEL.toml"
    manifest.write_text(
        "# leading comment\n"
        'name = "value with a # hash"          # trailing comment\n'
        "runtime_strategies = [\n"
        '    "torch_trt",                      # inline comment\n'
        '    "trt_llm",\n'
        "]\n",
        encoding="utf-8",
    )
    assert test_impact._parse_runtime_model_manifest(manifest) == ["torch_trt", "trt_llm"]


def test_parse_runtime_model_manifest_scalar_and_missing(tmp_path):
    """The scalar ``runtime_strategy`` form is supported and a missing manifest is empty."""
    scalar = tmp_path / "MODEL.toml"
    scalar.write_text('runtime_strategy = "solo"\n', encoding="utf-8")
    assert test_impact._parse_runtime_model_manifest(scalar) == ["solo"]
    assert test_impact._parse_runtime_model_manifest(tmp_path / "missing.toml") == []


def test_parse_runtime_model_manifest_malformed_fails_closed(tmp_path):
    """A malformed manifest raises rather than silently selecting no strategies."""
    bad = tmp_path / "MODEL.toml"
    bad.write_text('runtime_strategies = ["unterminated\n', encoding="utf-8")
    with pytest.raises(test_impact.tomllib.TOMLDecodeError):
        test_impact._parse_runtime_model_manifest(bad)
