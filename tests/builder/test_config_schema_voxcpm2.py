"""Tests for the VoxCPM2 runtime config schema."""

from __future__ import annotations

import pytest

try:
    from tensorrt_model_connect.runtime_config import (
        ConfigBundle,
        Layer,
        LayerContribution,
        clear_for_testing,
        lookup,
    )
    from tensorrt_model_connect.runtime_config.schemas import load_all
except ImportError:  # pragma: no cover
    pytest.skip("tensorrt_model_connect.runtime_config not importable", allow_module_level=True)


@pytest.fixture(autouse=True)
def clean_registry():
    clear_for_testing()
    yield
    clear_for_testing()


def test_audio_voxcpm2_schema_defaults_and_session_values():
    load_all()
    schema = lookup("audio_voxcpm2")
    assert schema is not None

    by_name = {field.name: field for field in schema.fields}
    assert by_name["cfg_value"].default == 2.0
    assert by_name["inference_timesteps"].default == 10
    assert by_name["normalize"].default is True
    assert by_name["denoise"].default is True
    assert by_name["retry_badcase"].default is True
    assert by_name["retry_badcase_max_times"].default == 3
    assert by_name["retry_badcase_ratio_threshold"].default == 6.0
    assert by_name["seed"].default == -1

    bundle = ConfigBundle.build([
        LayerContribution(
            layer=Layer.SESSION_REQUEST,
            values={
                "audio_voxcpm2": {
                    "cfg_value": 1.5,
                    "inference_timesteps": 4,
                    "seed": 123,
                },
            },
        ),
    ])

    assert bundle.get("audio_voxcpm2", "cfg_value") == 1.5
    assert bundle.get("audio_voxcpm2", "inference_timesteps") == 4
    assert bundle.get("audio_voxcpm2", "seed") == 123


def test_audio_voxcpm2_schema_rejects_invalid_timesteps():
    load_all()
    with pytest.raises(ValueError, match="audio_voxcpm2.inference_timesteps"):
        ConfigBundle.build([
            LayerContribution(
                layer=Layer.SESSION_REQUEST,
                values={"audio_voxcpm2": {"inference_timesteps": 0}},
            ),
        ])
