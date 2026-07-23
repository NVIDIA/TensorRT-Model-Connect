# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.families.openpi.model_config import (
    OPENPI_MODEL_TYPE,
    OPENPI_UPSTREAM_COMMIT,
    config_from_dir,
    get_profile,
    profile_names,
)


def test_profiles_are_explicit_and_pinned() -> None:
    assert profile_names() == ("pi05_droid",)

    droid = get_profile("pi05_droid")
    assert droid.upstream_commit == OPENPI_UPSTREAM_COMMIT
    assert droid.action_horizon == 15
    assert droid.external_state_dim == 8
    assert droid.external_action_dim == 8
    assert droid.discrete_state_input is True
    assert droid.prefix_length == 968
    assert droid.prefix.kv_width == 256
    assert droid.action_expert.kv_width == 256


def test_profile_selection_never_infers_from_path() -> None:
    with pytest.raises(ValueError, match="unsupported OpenPI profile"):
        get_profile("/tmp/checkpoints/pi05_droid")


def test_prepared_config_is_commit_bound(tmp_path) -> None:
    config_path = tmp_path / "openpi_config.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "pi05_droid",
                "upstream_commit": OPENPI_UPSTREAM_COMMIT,
                "conversion_manifest": "openpi_conversion_manifest.json",
                "tokenizer": "assets/paligemma_tokenizer.trtmcbpe",
                "tokenizer_sha256": "1" * 64,
                "tokenizer_source_sha256": "2" * 64,
                "tokenizer_export": {"schema_version": 1, "source_model_type": "BPE"},
                "normalization": "assets/droid/norm_stats.json",
            }
        ),
        encoding="utf-8",
    )
    config = config_from_dir(str(tmp_path))
    assert config is not None
    assert config["model_type"] == OPENPI_MODEL_TYPE
    assert config["head_dim"] == 256
    assert config["openpi"]["action_horizon"] == 15
    assert config["openpi_tokenizer_file"] == "assets/paligemma_tokenizer.trtmcbpe"
    assert config["openpi_tokenizer_sha256"] == "1" * 64
    assert config["openpi_normalization_file"] == "assets/droid/norm_stats.json"
    assert config["openpi_tokenizer_source_sha256"] == "2" * 64
    assert config["openpi_tokenizer_export"]["source_model_type"] == "BPE"

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["upstream_commit"] = "0" * 40
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unaudited upstream commit"):
        config_from_dir(str(tmp_path))
