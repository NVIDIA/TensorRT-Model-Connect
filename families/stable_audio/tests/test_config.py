# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the checkpoint contract the builder enforces, on written configs only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.stable_audio import config as config_module

# The published checkpoint's component configs, as diffusers' conversion script writes them
# (sizes match the gated repository's files; see the family README).
_INDEX = {
    "_class_name": "StableAudioPipeline",
    "projection_model": ["stable_audio", "StableAudioProjectionModel"],
    "scheduler": ["diffusers", "CosineDPMSolverMultistepScheduler"],
    "text_encoder": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "StableAudioDiTModel"],
    "vae": ["diffusers", "AutoencoderOobleck"],
}
_COMPONENTS = {
    "transformer": {"_class_name": "StableAudioDiTModel", "attention_head_dim": 64, "cross_attention_dim": 768,
                    "cross_attention_input_dim": 768, "global_states_input_dim": 1536, "in_channels": 64,
                    "num_attention_heads": 24, "num_key_value_attention_heads": 12, "num_layers": 24,
                    "out_channels": 64, "sample_size": 1024.0, "time_proj_dim": 256},
    "vae": {"_class_name": "AutoencoderOobleck", "audio_channels": 2, "channel_multiples": [1, 2, 4, 8, 16],
            "decoder_channels": 128, "decoder_input_channels": 64, "downsampling_ratios": [2, 4, 4, 8, 8],
            "encoder_hidden_size": 128, "sampling_rate": 44100},
    "text_encoder": {"model_type": "t5", "d_model": 768, "d_kv": 64, "d_ff": 3072, "num_layers": 12,
                     "num_heads": 12, "relative_attention_num_buckets": 32, "vocab_size": 32128,
                     "feed_forward_proj": "relu", "layer_norm_epsilon": 1e-6},
    "projection_model": {"_class_name": "StableAudioProjectionModel", "conditioning_dim": 768,
                         "max_value": 512, "min_value": 0, "text_encoder_dim": 768},
}
_SCHEDULER = {"_class_name": "CosineDPMSolverMultistepScheduler", "euler_at_final": False,
              "final_sigmas_type": "zero", "lower_order_final": True, "num_train_timesteps": 1000,
              "prediction_type": "v_prediction", "rho": 7.0, "sigma_data": 1.0, "sigma_max": 500,
              "sigma_min": 0.3, "sigma_schedule": "exponential", "solver_order": 2, "solver_type": "midpoint"}


def _write(root: Path, *, index=None, components=None, scheduler=None, tokenizer_length=128) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(index or _INDEX))
    for name, values in (components or _COMPONENTS).items():
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "config.json").write_text(json.dumps(values))
    (root / "scheduler").mkdir(exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler or _SCHEDULER))
    (root / "tokenizer").mkdir(exist_ok=True)
    (root / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"model_max_length": tokenizer_length}))
    added = [{"id": 0, "content": "<pad>"}, {"id": 1, "content": "</s>"}, {"id": 2, "content": "<unk>"}]
    (root / "tokenizer" / "tokenizer.json").write_text(json.dumps({"added_tokens": added}))
    return root


def _with(section: str, **changes) -> dict:
    components = json.loads(json.dumps(_COMPONENTS))
    components[section].update(changes)
    return components


def test_published_layout_resolves(tmp_path):
    cfg = config_module.resolve(_write(tmp_path))
    assert cfg["transformer"]["sample_size"] == 1024  # written as 1024.0
    assert cfg["transformer"]["rotary_dim"] == 32
    assert cfg["vae"]["hop_length"] == 2048
    assert cfg["vae"]["upsampling_ratios"] == [8, 8, 4, 4, 2]
    assert cfg["audio_samples"] == 2_097_152
    assert cfg["projection"]["text_projection"] is False
    assert cfg["tokenizer"] == {"max_length": 128, "pad_id": 0, "eos_id": 1}
    assert cfg["scheduler"]["sigma_schedule"] == "exponential"


@pytest.mark.parametrize(
    "section,changes",
    [
        ("text_encoder", {"feed_forward_proj": "gated-gelu"}),
        ("transformer", {"sample_size": 1024.5}),
        ("transformer", {"num_key_value_attention_heads": 7}),
        ("transformer", {"global_states_input_dim": 1024}),
        ("transformer", {"in_channels": 32, "out_channels": 32}),
        ("vae", {"downsampling_ratios": [3, 4, 4, 8, 8]}),
        ("projection_model", {"max_value": 0}),
    ],
)
def test_unsupported_or_inconsistent_components_are_rejected(tmp_path, section, changes):
    with pytest.raises((ValueError, NotImplementedError)):
        config_module.resolve(_write(tmp_path, components=_with(section, **changes)))


@pytest.mark.parametrize(
    "changes",
    [{"final_sigmas_type": "sigma_min"}, {"solver_order": 3}, {"prediction_type": "sample"},
     {"sigma_schedule": "linear"}, {"solver_type": "dpmsolver"}],
)
def test_unsupported_sampler_settings_are_rejected(tmp_path, changes):
    with pytest.raises(NotImplementedError):
        config_module.resolve(_write(tmp_path, scheduler={**_SCHEDULER, **changes}))


def test_legacy_solver_names_map_to_midpoint(tmp_path):
    cfg = config_module.resolve(_write(tmp_path, scheduler={**_SCHEDULER, "solver_type": "bh2"}))
    assert cfg["scheduler"]["solver_type"] == "midpoint"


def test_other_pipelines_and_components_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        config_module.resolve(_write(tmp_path / "a", index={**_INDEX, "_class_name": "StableAudio3Pipeline"}))
    with pytest.raises(ValueError):
        config_module.resolve(_write(tmp_path / "b", index={**_INDEX, "vae": ["diffusers", "AutoencoderSAME"]}))
