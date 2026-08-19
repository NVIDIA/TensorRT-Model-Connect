# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the native LTX-Video family plugin."""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.models.ltx_video import model as ltx_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "_class_name": "LTXPipeline",
        "video_height": 480,
        "video_width": 704,
        "video_num_frames": 161,
    }
    payload.update(raw_overrides)
    return ModelConfig(model_type="ltx_video", raw=payload)


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_matches_declared_ltx_video_aliases() -> None:
    assert ltx_mod.matches("ltx_video")
    assert ltx_mod.matches("LTXPipeline")
    assert ltx_mod.matches("ltx-video")
    assert not ltx_mod.matches("LTXConditionPipeline")
    assert not ltx_mod.matches("wan_t2v")


def test_owner_builders_only_call_concrete_graph_ops() -> None:
    from tensorrt_model_connect.models.ltx_video import graph_ops, ltx_dit_builder

    required: set[str] = set()
    for path in Path(ltx_dit_builder.__file__).parent.glob("*.py"):
        if path.name == "graph_ops.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        required.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "graph_ops"
        )

    assert sorted(name for name in required if not hasattr(graph_ops, name)) == []


def test_load_weights_requires_ltx_diffusers_model_index(tmp_path) -> None:
    model_dir = tmp_path / "ltx"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "LTXPipeline"})
    )
    for subdir in ("text_encoder", "transformer", "vae", "scheduler"):
        (model_dir / subdir).mkdir()
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps({"in_channels": 128, "num_attention_heads": 32})
    )
    (model_dir / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"use_dynamic_shifting": True, "base_image_seq_len": 1024})
    )

    cfg = _cfg()
    weights = ltx_mod.load_weights(str(model_dir), cfg)
    assert weights["_model_format"] == "diffusers"
    assert weights["_pipeline_class"] == "LTXPipeline"
    assert weights["_text_encoder_dir"].endswith("text_encoder")
    assert weights["_transformer_dir"].endswith("transformer")
    assert weights["_vae_dir"].endswith("vae")
    assert weights["_transformer_config"]["in_channels"] == 128
    assert cfg.raw["_scheduler_config"]["base_image_seq_len"] == 1024

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        ltx_mod.load_weights(str(bad_dir), _cfg())

    non_ltx = tmp_path / "non_ltx"
    non_ltx.mkdir()
    (non_ltx / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline"})
    )
    with pytest.raises(ValueError, match="Expected LTX pipeline"):
        ltx_mod.load_weights(str(non_ltx), _cfg())


def test_build_components_calls_native_subbuilders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def load_t5_weights(path, **kwargs):
        calls["load_t5_weights"] = {"path": path, **kwargs}
        return {"shared.weight": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls["build_t5_encoder_engine"] = {"weights": weights, **kwargs}
        return b"t5-plan"

    def compile_denoiser(path, **kwargs):
        calls["compile_denoiser"] = {"path": path, **kwargs}
        return b"denoiser-plan"

    def compile_vae(path, **kwargs):
        calls["compile_vae"] = {"path": path, **kwargs}
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.models.ltx_video.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.models.ltx_video.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setattr(ltx_mod, "_compile_ltx_denoiser_engine", compile_denoiser)
    monkeypatch.setattr(ltx_mod, "_compile_ltx_vae_decoder_engine", compile_vae)

    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
        "_text_encoder_config": {
            "d_model": 4096,
            "num_heads": 64,
            "d_kv": 64,
            "d_ff": 10240,
            "num_layers": 24,
            "vocab_size": 32128,
        },
        "_transformer_config": {"in_channels": 128},
    }

    out = ltx_mod.build_components(
        "/model",
        _cfg(
            video_height=256,
            video_width=384,
            video_num_frames=17,
            _fp32_layers=[24],
        ),
        weights,
        precision="fp16",
        verbose=True,
    )

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser"] == b"denoiser-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert "runtime_only" not in out
    assert calls["load_t5_weights"]["precision"] == "fp32"
    assert calls["load_t5_weights"]["fp32_layers"] == ()
    assert calls["build_t5_encoder_engine"]["max_seq_len"] == 128
    assert calls["build_t5_encoder_engine"]["precision"] == "fp32"
    assert calls["build_t5_encoder_engine"]["fp32_layers"] == ()
    assert calls["compile_denoiser"]["latent_frames"] == 3
    assert calls["compile_denoiser"]["latent_height"] == 8
    assert calls["compile_denoiser"]["latent_width"] == 12
    assert calls["compile_denoiser"]["precision"] == "fp16"
    assert calls["compile_vae"]["latent_channels"] == 128
    assert calls["compile_vae"]["precision"] == "fp16"


def test_get_diffusion_config_uses_ltx_scheduler_fields() -> None:
    latents_mean = [0.1, 0.2]
    latents_std = [0.9, 1.1]
    cfg = _cfg(
        video_height=480,
        video_width=704,
        video_num_frames=161,
        num_inference_steps=12,
        guidance_scale=2.5,
        _vae_latents_mean=latents_mean,
        _vae_latents_std=latents_std,
        _transformer_config={
            "in_channels": 128,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "num_layers": 28,
        },
        _vae_config={"scaling_factor": 1.0},
        _scheduler_config={
            "shift": 1.0,
            "use_dynamic_shifting": True,
            "base_shift": 0.95,
            "max_shift": 2.05,
            "base_image_seq_len": 1024,
            "max_image_seq_len": 4096,
            "shift_terminal": 0.1,
        },
    )

    out = ltx_mod.get_diffusion_config(cfg)

    assert out["diffusion_backend_type"] == "ltx_video"
    assert out["video_height"] == 480
    assert out["video_width"] == 704
    assert out["video_num_frames"] == 161
    assert out["num_inference_steps"] == 12
    assert out["guidance_scale"] == 2.5
    assert out["z_dim"] == 128
    assert out["dit_dim"] == 2048
    assert out["dit_num_heads"] == 32
    assert out["scale_factor_temporal"] == 8
    assert out["scale_factor_spatial"] == 32
    assert out["text_seq_len"] == 128
    assert out["use_dynamic_shifting"] == 1
    assert out["base_image_seq_len"] == 1024
    assert out["shift_terminal"] == 0.1
    assert out["latents_mean"] == latents_mean
    assert out["latents_std"] == latents_std
    assert "diffusers_runtime" not in json.dumps(out)


def test_ltx_rope_tables_use_diffusers_frequency_triplet_order() -> None:
    from tensorrt_model_connect.models.ltx_video.ltx_dit_builder import make_ltx_rope_tables

    dim = 14
    latent_frames = 1
    latent_height = 2
    latent_width = 3
    cos, sin = make_ltx_rope_tables(
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        dim=dim,
        frame_rate=25,
        temporal_compression_ratio=8,
        spatial_compression_ratio=32,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
    )

    token = 5  # f=0, h=1, w=2 in the flattened f/h/w grid.
    coords = np.array(
        [
            0.0,
            1.0 * 32.0 / 2048.0,
            2.0 * 32.0 / 2048.0,
        ],
        dtype=np.float32,
    )
    freqs = 10000.0 ** np.linspace(0.0, 1.0, dim // 6, dtype=np.float32)
    freqs = freqs * (np.pi / 2.0)
    expected_angles = np.array(
        [freq * (coord * 2.0 - 1.0) for freq in freqs for coord in coords],
        dtype=np.float32,
    )
    expected_cos = np.concatenate(
        [
            np.ones(dim % 6, dtype=np.float32),
            np.repeat(np.cos(expected_angles), 2).astype(np.float32),
        ]
    )
    expected_sin = np.concatenate(
        [
            np.zeros(dim % 6, dtype=np.float32),
            np.repeat(np.sin(expected_angles), 2).astype(np.float32),
        ]
    )

    np.testing.assert_allclose(cos[token], expected_cos, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(sin[token], expected_sin, rtol=0.0, atol=1e-6)


def test_ltx_rotate_half_matrix_rotates_within_each_head() -> None:
    from tensorrt_model_connect.models.ltx_video.ltx_dit_builder import _make_ltx_rotate_half_matrix

    interleaved = _make_ltx_rotate_half_matrix(8, 2, interleaved=True)
    x = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(
        x @ interleaved,
        np.array([-1.0, 0.0, -3.0, 2.0, -5.0, 4.0, -7.0, 6.0], dtype=np.float32),
    )

    split_half = _make_ltx_rotate_half_matrix(8, 2, interleaved=False)
    np.testing.assert_array_equal(
        x @ split_half,
        np.array([-2.0, -3.0, 0.0, 1.0, -6.0, -7.0, 4.0, 5.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="divisible"):
        _make_ltx_rotate_half_matrix(7, 2, interleaved=True)
