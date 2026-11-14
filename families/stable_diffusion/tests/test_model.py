# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config resolution and scheduler constants for the Stable Diffusion family."""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.stable_diffusion import config as config_module
from families.stable_diffusion.model import _alphas_cumprod


def _components(tmp_path, **overrides):
    unet = {
        "in_channels": 4, "block_out_channels": [320, 640, 1280, 1280],
        "layers_per_block": 2, "attention_head_dim": 8, "cross_attention_dim": 768,
        "norm_num_groups": 32,
    }
    vae = {
        "block_out_channels": [128, 256, 512, 512], "layers_per_block": 2,
        "norm_num_groups": 32, "latent_channels": 4,
    }
    text = {
        "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
        "max_position_embeddings": 77,
    }
    unet.update(overrides.pop("unet", {}))
    vae.update(overrides.pop("vae", {}))
    (tmp_path / "model_index.json").write_text('{"_class_name": "StableDiffusionPipeline"}')
    for name, payload in (("unet", unet), ("vae", vae), ("text_encoder", text)):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / "config.json").write_text(json.dumps(payload))
    (tmp_path / "scheduler").mkdir(exist_ok=True)
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(
        {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
         "beta_schedule": "scaled_linear", "steps_offset": 1}))
    return tmp_path


def test_component_configs_resolve(tmp_path):
    cfg = config_module.resolve(_components(tmp_path), latent_size=64)

    assert cfg["image_size"] == 512
    assert cfg["unet"]["context_length"] == 77
    assert cfg["unet"]["sample_size"] == 64
    # diffusers supplies both of these when the config stays silent.
    assert cfg["scaling_factor"] == pytest.approx(0.18215)
    assert cfg["text_encoder"]["layer_norm_eps"] == pytest.approx(1e-5)


def test_a_latent_width_disagreement_is_refused(tmp_path):
    root = _components(tmp_path, vae={"latent_channels": 8})
    with pytest.raises(ValueError, match="latent width"):
        config_module.resolve(root, latent_size=64)


def test_a_missing_model_index_is_refused(tmp_path):
    root = _components(tmp_path)
    (root / "model_index.json").unlink()
    with pytest.raises(FileNotFoundError, match="model_index.json"):
        config_module.resolve(root, latent_size=64)


def test_alphas_cumprod_matches_the_scaled_linear_schedule():
    cfg = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
           "beta_schedule": "scaled_linear"}
    alphas = _alphas_cumprod(cfg)

    assert len(alphas) == 1000
    # The product decreases monotonically from just under one towards zero; the
    # DDIM update depends on nothing else about the schedule.
    assert alphas[0] < 1.0
    assert all(later <= earlier for earlier, later in zip(alphas, alphas[1:]))
    assert alphas[-1] > 0.0
    expected = np.cumprod(
        1.0 - np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2)
    assert alphas[-1] == pytest.approx(float(expected[-1]), rel=1e-6)


def test_an_unknown_beta_schedule_is_refused():
    with pytest.raises(NotImplementedError, match="beta_schedule"):
        _alphas_cumprod({"num_train_timesteps": 10, "beta_start": 0.1, "beta_end": 0.2,
                         "beta_schedule": "cosine"})
