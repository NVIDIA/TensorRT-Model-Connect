# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PixArt family plugin, weight loader, and serializer helpers.

Trace: ARCH-FAM-001, UD-FAM-PIXART
Intent: Validate PixArt diffusion family plugin matching, weight serialization, and preprocessor encoding
Preconditions: Synthetic PixArt model config and weight tensors are available
Postconditions: Plugin matches PixArt aliases, serializes preprocessor weights correctly, and produces valid blobs
"""

from __future__ import annotations

import json
import struct
import sys
import types

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.config import ModelConfig
    import tensorrt_model_connect.families.pixart as pixart_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "pixart",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_height": 256,
        "image_width": 384,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _decode_blob(blob: bytes) -> tuple[dict[str, dict], bytes]:
    idx_len = struct.unpack("<I", blob[:4])[0]
    index = json.loads(blob[4:4 + idx_len].decode("utf-8"))
    payload = blob[4 + idx_len:]
    return index, payload


def _tensor_maker() -> callable:
    cursor = {"v": 1.0}

    def make(*shape: int) -> np.ndarray:
        n = int(np.prod(shape))
        start = cursor["v"]
        arr = np.arange(start, start + n, dtype=np.float32).reshape(shape)
        cursor["v"] += n
        return arr

    return make


def _pixart_tensors() -> dict[str, np.ndarray]:
    m = _tensor_maker()
    d: dict[str, np.ndarray] = {}
    src = "transformer_blocks.0"
    dim = 4

    d[f"{src}.scale_shift_table"] = m(6, dim)

    for proj in ("to_q", "to_k", "to_v"):
        d[f"{src}.attn1.{proj}.weight"] = m(dim, dim)
    d[f"{src}.attn1.to_q.bias"] = m(dim)
    d[f"{src}.attn1.to_out.0.weight"] = m(dim, dim)

    for proj in ("to_q", "to_k", "to_v"):
        d[f"{src}.attn2.{proj}.weight"] = m(dim, dim)
    d[f"{src}.attn2.to_v.bias"] = m(dim)
    d[f"{src}.attn2.to_out.0.weight"] = m(dim, dim)
    d[f"{src}.attn2.to_out.0.bias"] = m(dim)

    d[f"{src}.ff.net.0.proj.weight"] = m(8, dim)
    d[f"{src}.ff.net.0.proj.bias"] = m(8)
    d[f"{src}.ff.net.2.weight"] = m(dim, 8)

    d["scale_shift_table"] = m(2, dim)
    d["proj_out.weight"] = m(8, dim)

    d["pos_embed.proj.weight"] = m(3, 2, 2, 2)
    d["pos_embed.proj.bias"] = m(3)

    d["adaln_single.emb.timestep_embedder.linear_1.weight"] = m(4, 4)
    d["adaln_single.emb.timestep_embedder.linear_1.bias"] = m(4)
    d["adaln_single.linear.weight"] = m(6, 4)

    d["caption_projection.linear_1.weight"] = m(4, 5)
    d["caption_projection.linear_1.bias"] = m(4)

    return d


def test_matches_and_build_engine_not_supported() -> None:
    """Intent: validate alias matching and build_engine rejection for PixArt.

    Preconditions: plugin object is imported.
    Postconditions: declared aliases match; build_engine raises NotImplementedError.
    """
    plugin = pixart_mod.plugin
    assert plugin.matches("pixart")
    assert plugin.matches("pixart_sigma")
    assert plugin.matches("pixart_alpha")
    assert plugin.matches("pixartsigma")
    assert plugin.matches("pixartalpha")
    assert not plugin.matches("flux")

    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_cfg(), {}, 16)


def test_pixart_pipeline_classes_resolve_to_pixart_plugin() -> None:
    """PixArt owns the real Diffusers pipeline class mapping for PixArt models."""
    from tensorrt_model_connect.families import find_diffusion_plugin

    for pipeline_class in ("PixArtSigmaPipeline", "PixArtAlphaPipeline"):
        assert find_diffusion_plugin(pipeline_class) is pixart_mod.plugin


def test_load_weights_success_and_missing_model_index(tmp_path) -> None:
    """Intent: cover load_weights success and failure branches.

    Preconditions: one temp model has model_index and transformer config; another does not.
    Postconditions: success path populates expected keys and parsed transformer config.
    """
    model_dir = tmp_path / "pixart"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}")
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps({"num_attention_heads": 4, "num_layers": 2})
    )

    weights = pixart_mod.plugin.load_weights(str(model_dir), _cfg())
    assert weights["_model_format"] == "diffusers"
    assert weights["_text_encoder_dir"].endswith("text_encoder")
    assert weights["_transformer_config"]["num_layers"] == 2

    bad_dir = tmp_path / "pixart_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        pixart_mod.plugin.load_weights(str(bad_dir), _cfg())


def test_build_components_uses_transformer_and_t5_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: validate orchestration and argument derivation in build_components.

    Preconditions: all imported builders are replaced by deterministic stubs.
    Postconditions: builders receive expected dimensions/patch counts and return plans.
    """
    calls: dict[str, object] = {}

    model_dir = tmp_path / "pixart_model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    (model_dir / "text_encoder" / "config.json").write_text(
        json.dumps({
            "d_model": 1024,
            "num_heads": 16,
            "d_kv": 64,
            "d_ff": 2048,
            "num_layers": 8,
            "vocab_size": 2048,
        })
    )

    def load_t5_weights(path, **kwargs):
        calls["load_t5_weights"] = (path, kwargs)
        return {"t5": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls["build_t5_encoder_engine"] = (weights, kwargs)
        return b"t5-plan"

    def build_standard_dit_engine(weights, **kwargs):
        calls["build_standard_dit_engine"] = (weights, kwargs)
        return b"dit-plan"

    def build_vae_2d_decoder_engine(path, **kwargs):
        calls["build_vae_2d_decoder_engine"] = (path, kwargs)
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.families.pixart.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.standard_dit_builder",
        _module(
            "tensorrt_model_connect.families.pixart.standard_dit_builder",
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.vae_2d_builder",
        _module(
            "tensorrt_model_connect.families.pixart.vae_2d_builder",
            build_vae_2d_decoder_engine=build_vae_2d_decoder_engine,
        ),
    )

    monkeypatch.setattr(
        pixart_mod,
        "_load_pixart_dit_weights",
        lambda *args, **kwargs: (
            calls.setdefault("_load_pixart_dit_weights", []).append((args, kwargs))
            or {
                "pos_embed.proj.weight": np.ones((3, 2, 2, 2), dtype=np.float32),
                "pos_embed.proj.bias": np.ones((3,), dtype=np.float32),
            }
        ),
    )
    monkeypatch.setattr(
        pixart_mod,
        "_serialize_preprocessor_weights",
        lambda dit_weights, t5_dim, dit_dim: (
            calls.setdefault("_serialize_preprocessor_weights", []).append(
                (dit_weights, t5_dim, dit_dim)
            )
            or b"pixart-pre"
        ),
    )

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "num_layers": 2,
            "patch_size": 4,
            "cross_attention_dim": 64,
        },
    }

    config = _cfg(image_height=256, image_width=384)
    config.raw["_fp32_layers"] = [2]
    out = pixart_mod.plugin.build_components(
        str(model_dir),
        config,
        weights,
        precision="fp16",
        verbose=False,
    )

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"pixart-pre"

    # h_lat=32, w_lat=48, patch_size=4 -> 96 patches.
    assert calls["load_t5_weights"][1]["precision"] == "fp16"
    assert calls["build_t5_encoder_engine"][1]["precision"] == "fp16"
    assert calls["build_standard_dit_engine"][1]["precision"] == "fp16"
    assert calls["build_vae_2d_decoder_engine"][1]["precision"] == "fp32"
    assert calls["build_standard_dit_engine"][1]["num_patches"] == 96
    assert calls["build_standard_dit_engine"][1]["context_dim"] == 64
    assert calls["_serialize_preprocessor_weights"][0][1] == 1024
    assert calls["_serialize_preprocessor_weights"][0][2] == 32


def test_build_components_tensor_parallel_uses_tp_dit_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: verify PixArt TP dispatch leaves the single-device DiT builder unused.

    Preconditions: component builders are monkeypatched and TensorRT version is 11.0.
    Postconditions: denoiser_ranks contains one TP DiT plan per rank from the TP builder.
    """
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.parallel_config import ParallelConfig

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")
    calls: dict[str, object] = {"dit_ranks": []}

    model_dir = tmp_path / "pixart_tp"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    def build_standard_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device PixArt DiT builder used for TP build")

    def build_standard_dit_tp_engine(_weights, **kwargs):
        parallel = kwargs["parallel_config"]
        calls["dit_ranks"].append(parallel.rank)
        return f"dit-rank-{parallel.rank}".encode()

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.families.pixart.t5_encoder_builder",
            load_t5_weights=lambda *_args, **_kwargs: {},
            build_t5_encoder_engine=lambda *_args, **_kwargs: b"t5-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.standard_dit_builder",
        _module(
            "tensorrt_model_connect.families.pixart.standard_dit_builder",
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.standard_dit_tp_builder",
        _module(
            "tensorrt_model_connect.families.pixart.standard_dit_tp_builder",
            build_standard_dit_engine=build_standard_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.pixart.vae_2d_builder",
        _module(
            "tensorrt_model_connect.families.pixart.vae_2d_builder",
            build_vae_2d_decoder_engine=lambda *_args, **_kwargs: b"vae-plan",
        ),
    )
    monkeypatch.setattr(
        pixart_mod,
        "_load_pixart_dit_weights",
        lambda *_args, **_kwargs: {"dit": np.array([1], dtype=np.float32)},
    )
    monkeypatch.setattr(
        pixart_mod,
        "_serialize_preprocessor_weights",
        lambda *_args, **_kwargs: b"pixart-pre",
    )

    out = pixart_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=256, image_width=384),
        {
            "_text_encoder_dir": str(model_dir / "text_encoder"),
            "_transformer_dir": str(model_dir / "transformer"),
            "_vae_dir": str(model_dir / "vae"),
            "_transformer_config": {
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "patch_size": 4,
                "cross_attention_dim": 64,
            },
        },
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4),
    )

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser_ranks"] == {
        0: b"dit-rank-0",
        1: b"dit-rank-1",
        2: b"dit-rank-2",
        3: b"dit-rank-3",
    }
    assert "denoiser" not in out
    assert out["vae_decoder"] == b"vae-plan"
    assert calls["dit_ranks"] == [0, 1, 2, 3]


def test_get_diffusion_config_uses_transformer_overrides() -> None:
    """Intent: verify derived diffusion config values from transformer overrides.

    Preconditions: config.raw contains _transformer_config override fields.
    Postconditions: dit_dim/head/layer values are computed from overrides.
    """
    cfg = _cfg(
        _transformer_config={
            "num_attention_heads": 5,
            "attention_head_dim": 6,
            "num_layers": 3,
        },
        image_height=640,
        image_width=832,
    )

    dc = pixart_mod.plugin.get_diffusion_config(cfg)
    assert dc["dit_dim"] == 30
    assert dc["dit_num_heads"] == 5
    assert dc["dit_num_layers"] == 3
    assert dc["image_height"] == 640
    assert dc["image_width"] == 832
    assert dc["use_rope"] == 0


def test_load_pixart_dit_weights_maps_optional_biases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: validate low-level PixArt key remapping with optional bias branches.

    Preconditions: safetensor helpers are mocked over a deterministic key-value tensor map.
    Postconditions: mapped keys exist, 2D weights are transposed, optional absent biases are skipped.
    """
    tensors = _pixart_tensors()

    import importlib

    cm = importlib.import_module("tensorrt_model_connect.families.pixart.checkpoint_mapper")
    monkeypatch.setattr(cm, "_open_safetensors", lambda _p: ["reader"])
    monkeypatch.setattr(cm, "_has_tensor", lambda _r, name: name in tensors)
    monkeypatch.setattr(cm, "_load_tensor", lambda _r, name: tensors[name])

    out = pixart_mod._load_pixart_dit_weights(
        "/unused",
        dim=4,
        num_heads=2,
        num_layers=1,
        ffn_dim=8,
        cross_attn_dim=4,
    )

    np.testing.assert_allclose(
        out["blocks.0.attn1.to_q.weight"],
        tensors["transformer_blocks.0.attn1.to_q.weight"].T,
    )
    assert "blocks.0.attn1.to_q.bias" in out
    assert "blocks.0.attn1.to_k.bias" not in out

    assert out["blocks.0.scale_shift_table"].shape == (1, 6, 4)
    assert out["scale_shift_table"].shape == (1, 2, 4)

    np.testing.assert_allclose(
        out["adaln_single.emb.timestep_embedder.linear_1.weight"],
        tensors["adaln_single.emb.timestep_embedder.linear_1.weight"].T,
    )
    np.testing.assert_allclose(
        out["caption_projection.linear_1.weight"],
        tensors["caption_projection.linear_1.weight"].T,
    )


def test_serialize_preprocessor_weights_flattens_patch_conv() -> None:
    """Intent: validate PixArt preprocessor key mapping and Conv2D flatten+transpose.

    Preconditions: input includes patch conv and selected adaln/caption tensors.
    Postconditions: serialized index uses mapped keys with expected shapes and payload length.
    """
    dit_weights = {
        "pos_embed.proj.weight": np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2),
        "pos_embed.proj.bias": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "adaln_single.emb.timestep_embedder.linear_1.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "caption_projection.linear_2.bias": np.array([7.0, 8.0], dtype=np.float32),
    }

    blob = pixart_mod._serialize_preprocessor_weights(dit_weights, t5_dim=4096, dit_dim=1152)
    index, payload = _decode_blob(blob)

    assert "patch_embedding.weight" in index
    assert index["patch_embedding.weight"]["shape"] == [8, 3]
    assert "condition_embedder.time_embedding.0.weight" in index
    assert "condition_embedder.text_embedding_2.bias" in index

    max_end = 0
    for info in index.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload) == max_end
