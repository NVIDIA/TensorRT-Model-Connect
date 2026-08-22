# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the family ``MODEL.toml`` loader (standards-compliant TOML parsing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorrt_model_connect import families


def test_read_model_toml_parses_edge_cases(tmp_path: Path):
    """quoted #, comments, escapes, and multiline lists parse correctly.

    The previous flat fallback parser truncated any line at the first ``#`` and
    could not decode escapes, so these cases are exactly where it diverged from a
    standards-compliant parser.
    """
    manifest = tmp_path / "MODEL.toml"
    manifest.write_text(
        "# leading comment\n"
        'id = "value with a # hash"            # trailing comment\n'
        "aliases = [\n"
        '    "alpha",                          # inline comment\n'
        '    "beta",\n'
        "]\n"
        'tabbed = "a\\tb"\n',
        encoding="utf-8",
    )

    data = families._read_model_toml(manifest)

    assert data["id"] == "value with a # hash"
    assert data["aliases"] == ["alpha", "beta"]
    assert data["tabbed"] == "a\tb"


def test_read_model_toml_malformed_fails_closed(tmp_path: Path):
    """A malformed manifest raises rather than being silently misread."""
    manifest = tmp_path / "MODEL.toml"
    manifest.write_text('aliases = ["unterminated\n', encoding="utf-8")

    with pytest.raises(families.tomllib.TOMLDecodeError):
        families._read_model_toml(manifest)
