# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ..builder import _position_embedding_2d
from ..checkpoint import CHECKPOINT_SHA256, load_checkpoint
from ..plugin import (
    DATASET_REVISION,
    LEROBOT_REVISION,
    POLICY_REVISION,
    _validate_initial_policy,
    config_from_dir,
    plugin,
)


def _config() -> dict:
    return {
        "type": "act",
        "n_obs_steps": 1,
        "chunk_size": 100,
        "n_action_steps": 100,
        "vision_backbone": "resnet18",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "pre_norm": False,
        "use_vae": True,
        "latent_dim": 32,
        "temporal_ensemble_coeff": None,
        "input_features": {
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [14]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }


def test_initial_policy_contract_and_metadata() -> None:
    raw = _config()
    _validate_initial_policy(raw)
    metadata = plugin.get_bundle_config_overrides(SimpleNamespace(raw=raw))
    assert metadata["policy_revision"] == POLICY_REVISION
    assert metadata["lerobot_revision"] == LEROBOT_REVISION
    assert metadata["dataset_revision"] == DATASET_REVISION
    assert metadata["control_frequency_hz"] == 50
    assert metadata["action_chunk_size"] == 100
    assert metadata["robotics_safety_validated"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 99),
        ("n_action_steps", 1),
        ("temporal_ensemble_coeff", 0.01),
        ("pre_norm", True),
    ],
)
def test_initial_policy_contract_rejects_semantic_drift(field: str, value: object) -> None:
    raw = _config()
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        _validate_initial_policy(raw)


def test_config_adapter_emits_native_runtime_contract(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"digest checked during weight loading")
    adapted = config_from_dir(tmp_path)
    assert adapted is not None
    assert adapted["model_type"] == "lerobot_act"
    assert adapted["runtime_strategy"] == "lerobot_act_action_chunk"
    assert adapted["requires_tokenizer"] is False


def test_checkpoint_digest_fails_closed(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not the qualified checkpoint")
    with pytest.raises(ValueError, match=CHECKPOINT_SHA256):
        load_checkpoint(tmp_path)


def test_act_2d_position_embedding_is_stable_and_finite() -> None:
    positions = _position_embedding_2d(15, 20, 256)
    assert positions.shape == (300, 512)
    assert positions.dtype == np.float32
    assert np.isfinite(positions).all()
    np.testing.assert_allclose(positions[0, 0], np.sin(2 * np.pi / 15), atol=1.0e-6)
    np.testing.assert_allclose(positions[0, 256], np.sin(2 * np.pi / 20), atol=1.0e-6)
