# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named benchmark requests must execute their declared fixture."""

import json
from pathlib import Path

import pytest

from families.clef.tests.benchmark_compile import select_fixtures


ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "manifests/clef-flash.json").read_text())


@pytest.mark.parametrize("name", ["clef-flash-invoice", "invoice"])
def test_manifest_name_and_fixture_stem_select_the_same_case(name):
    assert select_fixtures(ROOT, MANIFEST, None, [name]) == [
        ("clef-flash-invoice", ROOT / "fixtures/invoice.json")
    ]


def test_explicit_directory_preserves_fixture_stem_selection(tmp_path):
    fixture = tmp_path / "custom.json"
    fixture.write_text("{}")
    assert select_fixtures(ROOT, MANIFEST, tmp_path, ["custom"]) == [("custom", fixture)]


def test_unknown_case_is_rejected_even_alongside_a_valid_case():
    with pytest.raises(ValueError, match="unknown benchmark cases: typo"):
        select_fixtures(ROOT, MANIFEST, None, ["clef-flash-invoice", "typo"])


def test_empty_fixture_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no benchmark fixtures selected"):
        select_fixtures(ROOT, MANIFEST, tmp_path, [])
