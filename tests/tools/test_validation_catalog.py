# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.validation import catalog as validation_catalog


def test_manifest_records_can_be_loaded_by_canonical_name(monkeypatch) -> None:
    models_dir = Path("models")
    records = [
        {"name": "model-b", "family": "family-b"},
        {"name": "model-a", "family": "family-a"},
    ]
    monkeypatch.setattr(
        validation_catalog,
        "load_manifest_records",
        lambda path: records if path == models_dir else [],
    )

    assert validation_catalog.load_manifest_records_by_name(models_dir) == {
        "model-b": records[0],
        "model-a": records[1],
    }
