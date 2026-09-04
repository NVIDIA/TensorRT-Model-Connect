# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

from ..builder import _position_embedding_2d
from ..checkpoint import load_checkpoint
from ..model import _validate_initial_policy
from ..support import describe


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


def test_initial_policy_contract_and_support() -> None:
    raw = _config()
    _validate_initial_policy(raw)
    support = describe(ModelMetadata(config=raw, model_index={}))
    assert support is not None
    assert support.tasks == ("robot_control",)
    assert support.default_task == "robot_control"
    family, resolved = resolve_family(ModelMetadata(config=raw, model_index={}))
    assert family == "lerobot_act"
    assert resolved == support


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


def test_checkpoint_requires_the_declared_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        load_checkpoint(tmp_path)


def test_act_2d_position_embedding_is_stable_and_finite() -> None:
    positions = _position_embedding_2d(15, 20, 256)
    assert positions.shape == (300, 512)
    assert positions.dtype == np.float32
    assert np.isfinite(positions).all()
    np.testing.assert_allclose(positions[0, 0], np.sin(2 * np.pi / 15), atol=1.0e-6)
    np.testing.assert_allclose(positions[0, 256], np.sin(2 * np.pi / 20), atol=1.0e-6)
