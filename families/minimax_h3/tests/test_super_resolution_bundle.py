# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from families.minimax_h3.model import _super_resolution_build_inputs
from families.minimax_h3.provenance import (
    super_resolution_bundle_config,
    super_resolution_source_identity,
    validate_super_resolution_bundle_config,
    validate_super_resolution_source_identity,
)
from families.minimax_h3.runtime_config_schema import Layer, SCHEMA


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    primary = tmp_path / "realesr-general-x4v3.pth"
    weak = tmp_path / "realesr-general-wdn-x4v3.pth"
    primary.write_bytes(b"primary")
    weak.write_bytes(b"weak")
    return primary, weak


def test_super_resolution_checkpoint_options_are_build_only(tmp_path: Path) -> None:
    fields = {field.name: field for field in SCHEMA.fields}
    for name in ("super_resolution_model", "super_resolution_weak_model"):
        assert fields[name].allowed_layers == frozenset(
            {Layer.BUILD_TIME, Layer.SESSION_REQUEST}
        )

    primary, weak = _checkpoints(tmp_path)
    assert _super_resolution_build_inputs(
        {"super_resolution_model": str(primary), "super_resolution_weak_model": str(weak)}
    ) == (primary.absolute(), weak.absolute(), 0.5)
    assert _super_resolution_build_inputs({"super_resolution_model": str(primary)}) == (
        primary.absolute(),
        None,
        1.0,
    )
    with pytest.raises(ValueError, match="requires super_resolution_model"):
        _super_resolution_build_inputs({"super_resolution_weak_model": str(weak)})


def test_super_resolution_bundle_contract_is_path_free(tmp_path: Path) -> None:
    primary, weak = _checkpoints(tmp_path)
    identity = super_resolution_source_identity(
        primary,
        weak,
        denoise_strength=0.5,
    )
    assert validate_super_resolution_source_identity(identity) == identity
    config = super_resolution_bundle_config(identity)
    assert validate_super_resolution_bundle_config(config) == config
    assert config["section"] == "video_super_resolution_plan"
    assert config["source_shape"] == [480, 864]
    assert config["target_shape"] == [720, 1296]
    assert config["batch_profile"] == [1, 4, 8]
    assert all("path" not in source for source in config["sources"])


def test_super_resolution_requires_the_public_checkpoint_names(tmp_path: Path) -> None:
    primary, weak = _checkpoints(tmp_path)
    renamed = tmp_path / "renamed.pth"
    renamed.write_bytes(primary.read_bytes())
    with pytest.raises(ValueError, match="must be named"):
        super_resolution_source_identity(renamed, weak, denoise_strength=0.5)
