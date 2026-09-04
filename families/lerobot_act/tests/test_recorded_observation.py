# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the immutable LeRobot ACT recorded-observation fixture."""

from __future__ import annotations

import json
from pathlib import Path

from . import prepare_recorded_observation as recorded


def test_packaged_recorded_observation_matches_qualified_source() -> None:
    assert recorded._is_qualified_observation(recorded._FIXTURE_DIR, 0, 0)


def test_packaged_recorded_observation_materializes_offline(tmp_path: Path) -> None:
    assert recorded._materialize_packaged_observation(tmp_path, 0, 0)
    assert recorded._is_qualified_observation(tmp_path, 0, 0)
    metadata = json.loads((tmp_path / "recorded_observation.json").read_text())
    assert metadata["dataset_revision"] == recorded._DATASET_REVISION
    assert metadata["global_index"] == 0
