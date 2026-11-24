# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from families.minimax_h3.delivery import resolve_super_resolution_sources
from families.minimax_h3.provenance import (
    super_resolution_bundle_config,
    super_resolution_source_identity,
    validate_super_resolution_bundle_config,
    validate_super_resolution_source_identity,
)
from families.minimax_h3.runtime_config_schema import normalize_build_options


def _checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    primary = tmp_path / "realesr-general-x4v3.pth"
    weak = tmp_path / "realesr-general-wdn-x4v3.pth"
    primary.write_bytes(b"primary")
    weak.write_bytes(b"weak")
    return primary, weak


def test_super_resolution_checkpoint_options_are_build_only(tmp_path: Path) -> None:
    primary, weak = _checkpoints(tmp_path)
    options = {
        "super_resolution": True,
        "super_resolution_model": str(primary),
        "super_resolution_weak_model": str(weak),
    }
    assert normalize_build_options(options) == options
    assert resolve_super_resolution_sources(options) == (primary.absolute(), weak.absolute())
    assert resolve_super_resolution_sources({}) == (None, None)
    assert resolve_super_resolution_sources({"super_resolution": False}) == (None, None)
    for implicit in (
        {"super_resolution_model": str(primary)},
        {"super_resolution": False, "super_resolution_weak_model": str(weak)},
    ):
        with pytest.raises(ValueError, match="require super_resolution=true"):
            resolve_super_resolution_sources(implicit)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": True}, "unknown MiniMax-H3 option"),
        ({"first_block_cache": 1}, "invalid MiniMax-H3 option first_block_cache"),
        ({"first_block_cache_threshold": True}, "invalid MiniMax-H3 option"),
        ({"first_block_cache_threshold": float("nan")}, "invalid MiniMax-H3 option"),
        ({"first_block_cache_threshold": 0.0}, "invalid MiniMax-H3 option"),
        ({"super_resolution_model": False}, "invalid MiniMax-H3 option"),
        ({"retain_engines": True}, "runtime-only: retain_engines"),
        ({"retained_tail_weight_budget_gib": 24}, "runtime-only"),
        ({"retained_tail_weight_budget_gib": True}, "invalid MiniMax-H3 option"),
        ({"retained_tail_weight_budget_gib": 2**33}, "invalid MiniMax-H3 option"),
    ],
)
def test_build_options_preserve_type_and_runtime_only_rejections(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_build_options(options)


def test_build_options_return_only_supplied_values() -> None:
    assert normalize_build_options({}) == {}
    options = {"first_block_cache": False, "first_block_cache_threshold": 0.12}
    normalized = normalize_build_options(options)
    assert normalized == options
    assert normalized is not options


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
    assert config["mode"] == "explicit"
    assert config["input_name"] == "frames"
    assert config["output_name"] == "upscaled_frames"
    assert config["source_shape"] == [480, 864]
    assert config["target_shape"] == [720, 1296]
    assert config["batch_profile"] == [1, 4, 8]
    assert all("path" not in source for source in config["sources"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", None),
        ("mode", "automatic"),
        ("input_name", None),
        ("output_name", "wrong"),
        ("source_shape", [768, 1344]),
        ("target_shape", [1080, 1920]),
        ("batch_profile", [1, 4, 16]),
    ],
)
def test_super_resolution_metadata_rejects_legacy_or_mismatched_abi(
    tmp_path: Path, field: str, value: object
) -> None:
    primary, weak = _checkpoints(tmp_path)
    config = super_resolution_bundle_config(
        super_resolution_source_identity(primary, weak, denoise_strength=0.5)
    )
    if value is None:
        del config[field]
    else:
        config[field] = value
    with pytest.raises(ValueError):
        validate_super_resolution_bundle_config(config)


def test_super_resolution_requires_the_public_checkpoint_names(tmp_path: Path) -> None:
    primary, weak = _checkpoints(tmp_path)
    renamed = tmp_path / "renamed.pth"
    renamed.write_bytes(primary.read_bytes())
    with pytest.raises(ValueError, match="must be named"):
        super_resolution_source_identity(renamed, weak, denoise_strength=0.5)
