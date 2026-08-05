# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tools import prepare_full_duplex_bench_validation as prepare_fdb


def test_selection_is_fixed_and_stratified() -> None:
    discovered = {
        category: [Path(category) / str(index) for index in range(50)]
        for category in prepare_fdb.CATEGORY_COUNTS
    }

    first = prepare_fdb.select_samples(
        discovered,
        samples_per_category=30,
    )
    second = prepare_fdb.select_samples(
        {category: list(reversed(rows)) for category, rows in discovered.items()},
        samples_per_category=30,
    )

    assert first == second
    assert all(len(rows) == 30 for rows in first.values())
    assert any(
        rows != discovered[category][:30] for category, rows in first.items()
    )


def test_selection_rejects_an_undersized_category() -> None:
    with pytest.raises(ValueError, match="only 2 are available"):
        prepare_fdb.select_samples(
            {"icc_backchannel": [Path("0"), Path("1")]},
            samples_per_category=3,
        )


def test_public_subset_licenses_are_not_collapsed_to_repository_license() -> None:
    assert prepare_fdb.CATEGORY_LICENSES["synthetic_pause_handling"] == "MIT"
    assert prepare_fdb.CATEGORY_LICENSES["candor_pause_handling"].startswith(
        "CC BY-NC 4.0"
    )
    assert prepare_fdb.CATEGORY_LICENSES["icc_backchannel"].startswith(
        "CC BY-NC 4.0"
    )


def test_source_manifest_must_match_pinned_revision_counts_and_licenses(
    tmp_path: Path,
) -> None:
    manifest = {
        "sample_count": sum(prepare_fdb.CATEGORY_COUNTS.values()),
        "upstream_revision": prepare_fdb.FDB_REVISION,
        "subsets": {
            category: {
                "samples": count,
                "license": prepare_fdb.CATEGORY_LICENSES[category],
            }
            for category, count in prepare_fdb.CATEGORY_COUNTS.items()
        },
    }
    path = tmp_path / "DATASET_MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, loaded_path = prepare_fdb.load_source_manifest(tmp_path)

    assert loaded == manifest
    assert loaded_path == path


def test_source_manifest_rejects_an_unpinned_revision(tmp_path: Path) -> None:
    manifest = {
        "sample_count": sum(prepare_fdb.CATEGORY_COUNTS.values()),
        "upstream_revision": "main",
        "subsets": {},
    }
    (tmp_path / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="expected upstream revision"):
        prepare_fdb.load_source_manifest(tmp_path)


def test_prepared_float_wav_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    audio = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)

    prepare_fdb._write_float32_wav(first, audio)
    prepare_fdb._write_float32_wav(second, audio)

    assert first.read_bytes() == second.read_bytes()
    assert prepare_fdb._sha256(first) == prepare_fdb._sha256(second)
    assert first.read_bytes()[8:16] == b"WAVEfmt "
    assert struct.unpack("<H", first.read_bytes()[20:22])[0] == 3
