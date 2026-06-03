"""Unit tests for VoxCPM2 detection-only family support.

Trace: ARCH-FAM-001, UD-FAM-WEIGHTS
Intent: Validate openbmb/VoxCPM2 config parsing, family dispatch, audio
metadata, and the explicit unsupported raw TRT build path.
Preconditions: Synthetic VoxCPM2 config metadata is available.
Postconditions: The family matches upstream architecture strings and refuses
runtime build attempts with a clear follow-up blocker.
"""

from __future__ import annotations

import json

import pytest

try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.voxcpm2 as voxcpm2_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg() -> ModelConfig:
    return ModelConfig.from_json(json.dumps({
        "architecture": "voxcpm2",
        "lm_config": {
            "vocab_size": 73448,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
        },
        "encoder_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
        },
        "dit_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
        },
        "audio_vae_config": {
            "sample_rate": 16000,
            "out_sample_rate": 48000,
        },
        "max_length": 8192,
        "dtype": "bfloat16",
    }))


def test_model_config_merges_voxcpm2_lm_config() -> None:
    """Intent: verify upstream architecture/lm_config are parsed.
    Preconditions: config JSON mirrors openbmb/VoxCPM2 key structure.
    Postconditions: ModelConfig exposes model_type and LM dimensions.
    """
    cfg = _cfg()

    assert cfg.model_type == "voxcpm2"
    assert cfg.vocab_size == 73448
    assert cfg.hidden_size == 2048
    assert cfg.intermediate_size == 6144
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.raw["audio_vae_config"]["out_sample_rate"] == 48000


def test_matches_aliases_and_rejects_other_tts_families() -> None:
    """Intent: verify VoxCPM2 dispatch aliases stay scoped.
    Preconditions: plugin object is imported.
    Postconditions: upstream aliases match and Bark/Magpie do not.
    """
    plugin = voxcpm2_mod.plugin

    assert plugin.matches("voxcpm2")
    assert plugin.matches("vox_cpm2")
    assert plugin.matches("vox-cpm2")
    assert not plugin.matches("bark")
    assert not plugin.matches("magpie_tts")


def test_load_weights_records_detection_metadata(tmp_path) -> None:
    """Intent: cover detection-only load_weights behavior.
    Preconditions: no checkpoint tensors are present.
    Postconditions: returned metadata captures sample rates and blocker text.
    """
    weights = voxcpm2_mod.plugin.load_weights(str(tmp_path), _cfg(), precision="bf16")

    assert weights["_model_format"] == "voxcpm2"
    assert weights["_requires_python_package"] == "voxcpm"
    assert weights["_sample_rate"] == 48000
    assert weights["_reference_sample_rate"] == 16000
    assert weights["_precision"] == "bf16"
    assert "LocEnc -> TSLM -> RALM -> LocDiT" in weights["_architecture"]
    assert "raw TensorRT runtime support is not implemented" in (
        weights["_unsupported_runtime_reason"]
    )


def test_audio_config_documents_runtime_blocker() -> None:
    """Intent: verify future bundle metadata is explicit about support status.
    Preconditions: VoxCPM2 config contains AudioVAE sample-rate fields.
    Postconditions: audio config exposes output/reference rates and blocker.
    """
    audio_cfg = voxcpm2_mod.plugin.get_audio_config(_cfg())

    assert audio_cfg["voxcpm2"] is True
    assert audio_cfg["sample_rate"] == 48000
    assert audio_cfg["reference_sample_rate"] == 16000
    assert audio_cfg["requires_python_package"] == "voxcpm"
    assert audio_cfg["runtime_supported"] is False
    assert "text_to_audio_voxcpm2 runtime" in audio_cfg["runtime_blocker"]


def test_build_engine_not_supported_until_runtime_lands() -> None:
    """Intent: reject misleading raw TRT bundles for VoxCPM2.
    Preconditions: plugin has detection metadata but no runtime implementation.
    Postconditions: build_engine raises a clear follow-up blocker.
    """
    with pytest.raises(NotImplementedError, match="LocEnc -> TSLM -> RALM -> LocDiT"):
        voxcpm2_mod.plugin.build_engine(_cfg(), {}, 16)
