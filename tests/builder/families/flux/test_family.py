# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FLUX family plugin and preprocessor serialization.

Trace: ARCH-FAM-001, UD-FAM-FLUX
Intent: Validate FLUX diffusion family plugin matching, weight serialization, and preprocessor blob encoding
Preconditions: Synthetic FLUX model config and weight tensors are available
Postconditions: Plugin matches FLUX aliases, serializes preprocessor weights correctly, and rejects build_engine
"""

from __future__ import annotations

import json
import struct
import sys
import types
from importlib import import_module

import numpy as np
import pytest

try:
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.parallel_config import ParallelConfig
    flux_mod = import_module("tensorrt_model_connect.families.flux.plugin")
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "flux",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_height": 80,
        "image_width": 96,
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


def test_matches_and_build_engine_not_supported() -> None:
    """Intent: verify FLUX model aliases and explicit build_engine rejection.

    Preconditions: plugin object is imported.
    Postconditions: aliases match and build_engine raises NotImplementedError.
    """
    plugin = flux_mod.plugin
    assert plugin.matches("flux")
    assert plugin.matches("flux.1")
    assert plugin.matches("flux.2")
    assert plugin.matches("flux_t2i")
    assert not plugin.matches("wan_t2v")

    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_cfg(), {}, 16)


def test_load_weights_reads_optional_dirs_and_transformer_config(tmp_path) -> None:
    """Intent: cover load_weights success branch with optional subdirs and config JSON.

    Preconditions: model_index.json, transformer/config.json, and selected subdirs exist.
    Postconditions: output WeightDict includes discovered directories and parsed transformer config.
    """
    model_dir = tmp_path / "flux"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}")
    (model_dir / "transformer" / "config.json").write_text(
        json.dumps({"num_attention_heads": 3, "guidance_embeds": True})
    )

    weights = flux_mod.plugin.load_weights(str(model_dir), _cfg())

    assert weights["_model_format"] == "diffusers"
    assert "_text_encoder_dir" in weights
    assert "_text_encoder_2_dir" not in weights
    assert weights["_transformer_config"]["num_attention_heads"] == 3
    assert weights["_vae_dir"].endswith("vae")


def test_load_weights_rejects_missing_model_index(tmp_path) -> None:
    """Intent: cover load_weights failure branch for non-diffusers paths.

    Preconditions: directory exists without model_index.json.
    Postconditions: load_weights raises ValueError.
    """
    bad_dir = tmp_path / "flux_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        flux_mod.plugin.load_weights(str(bad_dir), _cfg())


def test_build_components_with_clip_and_second_t5(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: validate CLIP+T5 orchestration and derived image-token count.

    Preconditions: builder modules are replaced by deterministic stubs and both text encoders exist.
    Postconditions: both encoder plans are returned, denoiser/vae builders are called with derived args.
    """
    calls: dict[str, object] = {}

    model_dir = tmp_path / "flux_model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "text_encoder_2").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    (model_dir / "text_encoder" / "config.json").write_text(
        json.dumps({
            "architectures": ["CLIPTextModel"],
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "intermediate_size": 24,
            "vocab_size": 99,
            "max_position_embeddings": 77,
        })
    )
    (model_dir / "text_encoder_2" / "config.json").write_text(
        json.dumps({
            "d_model": 64,
            "num_heads": 8,
            "d_kv": 8,
            "d_ff": 128,
            "num_layers": 3,
            "vocab_size": 321,
        })
    )

    def load_t5_weights(path, **kwargs):
        calls.setdefault("load_t5_weights", []).append((path, kwargs))
        return {"t5": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls.setdefault("build_t5_encoder_engine", []).append((weights, kwargs))
        return b"t5-plan"

    def load_clip_weights(path, **kwargs):
        calls["load_clip_weights"] = (path, kwargs)
        return {"clip": np.array([2], dtype=np.float32)}

    def build_clip_encoder_engine(weights, **kwargs):
        calls["build_clip_encoder_engine"] = (weights, kwargs)
        return b"clip-plan"

    def load_flux_dit_weights(path, **kwargs):
        calls["load_flux_dit_weights"] = (path, kwargs)
        return {"dit": np.array([3], dtype=np.float32)}

    def build_flux_dit_engine(weights, **kwargs):
        calls["build_flux_dit_engine"] = (weights, kwargs)
        return b"dit-plan"

    def build_flux_vae_decoder_engine(path, **kwargs):
        calls["build_flux_vae_decoder_engine"] = (path, kwargs)
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.t5_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.t5_encoder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.clip_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.clip_encoder",
            load_clip_weights=load_clip_weights,
            build_clip_encoder_engine=build_clip_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux",
            load_flux_dit_weights=load_flux_dit_weights,
            build_flux_dit_engine=build_flux_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux_parallel",
            build_flux_dit_engine=lambda *_a, **_k: b"unused-tp-dit-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=build_flux_vae_decoder_engine,
        ),
    )

    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux_preprocessor",
        lambda dit_weights, guidance_embeds: (
            calls.setdefault("serialize", []).append((dit_weights, guidance_embeds))
            or b"flux-preproc"
        ),
    )

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_text_encoder_2_dir": str(model_dir / "text_encoder_2"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 3,
            "num_single_layers": 4,
            "guidance_embeds": True,
        },
    }

    out = flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=80, image_width=96),
        weights,
        verbose=False,
    )

    assert out["text_encoders"] == [("clip", b"clip-plan"), ("t5", b"t5-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"flux-preproc"

    # h_lat=10, w_lat=12, pack_size=2 -> num_img_tokens=30
    assert calls["build_flux_dit_engine"][1]["num_img_tokens"] == 30
    assert calls["load_flux_dit_weights"][1]["dim"] == 8
    assert calls["serialize"][0][1] is True


def test_build_components_builds_rank_local_flux_dit_for_tp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: verify FLUX.1 TP builds one rank-local DiT plan per rank.

    Preconditions: component builders are replaced by deterministic stubs and TRT version is 11.x.
    Postconditions: text/VAE stay single-copy while denoiser_ranks carries rank-local plans.
    """
    calls: list[ParallelConfig] = []
    model_dir = tmp_path / "flux_tp"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    def load_flux_dit_weights(_path, **_kwargs):
        return {"dit": np.array([3], dtype=np.float32)}

    def build_flux_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device FLUX DiT builder used for TP build")

    def build_flux_dit_tp_engine(_weights, **kwargs):
        parallel = kwargs["parallel_config"]
        calls.append(parallel)
        return f"dit-rank-{parallel.rank}".encode("ascii")

    def build_flux_vae_decoder_engine(_path, **_kwargs):
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.t5_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.t5_encoder",
            load_t5_weights=lambda *_args, **_kwargs: {},
            build_t5_encoder_engine=lambda *_args, **_kwargs: b"t5-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.clip_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.clip_encoder",
            load_clip_weights=lambda *_args, **_kwargs: {},
            build_clip_encoder_engine=lambda *_args, **_kwargs: b"clip-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux",
            load_flux_dit_weights=load_flux_dit_weights,
            build_flux_dit_engine=build_flux_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux_parallel",
            build_flux_dit_engine=build_flux_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=build_flux_vae_decoder_engine,
        ),
    )
    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux_preprocessor",
        lambda _dit_weights, _guidance_embeds: b"flux-preproc",
    )

    from tensorrt_model_connect import trt_compat
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")

    out = flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=80, image_width=96),
        {
            "_transformer_dir": str(model_dir / "transformer"),
            "_vae_dir": str(model_dir / "vae"),
            "_transformer_config": {
                "num_attention_heads": 8,
                "attention_head_dim": 4,
                "num_layers": 1,
                "num_single_layers": 1,
            },
        },
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4),
    )

    assert out["text_encoders"] == []
    assert out["vae_decoder"] == b"vae-plan"
    assert out["denoiser_ranks"] == {
        0: b"dit-rank-0",
        1: b"dit-rank-1",
        2: b"dit-rank-2",
        3: b"dit-rank-3",
    }
    assert "denoiser" not in out
    assert [cfg.rank for cfg in calls] == [0, 1, 2, 3]
    assert all(cfg.tp_size == 4 for cfg in calls)


def test_build_components_treats_text_encoder_as_t5_when_not_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: cover branch where text_encoder directory contains a T5 config.

    Preconditions: text_encoder has non-CLIP architecture and no text_encoder_2 dir.
    Postconditions: T5 loader path is used and CLIP loader path is not used.
    """
    calls: dict[str, int] = {"clip_load": 0, "t5_load": 0}

    model_dir = tmp_path / "flux_model2"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    (model_dir / "text_encoder" / "config.json").write_text(
        json.dumps({
            "architectures": ["T5EncoderModel"],
            "d_model": 48,
            "num_heads": 6,
            "d_kv": 8,
            "d_ff": 96,
            "num_layers": 2,
            "vocab_size": 123,
        })
    )

    def load_t5_weights(_path, **_kwargs):
        calls["t5_load"] += 1
        return {"t5": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(_weights, **_kwargs):
        return b"t5-plan"

    def load_clip_weights(*_args, **_kwargs):
        calls["clip_load"] += 1
        return {"clip": np.array([1], dtype=np.float32)}

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.t5_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.t5_encoder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.clip_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.clip_encoder",
            load_clip_weights=load_clip_weights,
            build_clip_encoder_engine=lambda *_a, **_k: b"clip-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux",
            load_flux_dit_weights=lambda *_a, **_k: {"dit": np.array([2], dtype=np.float32)},
            build_flux_dit_engine=lambda *_a, **_k: b"dit-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux_parallel",
            build_flux_dit_engine=lambda *_a, **_k: b"unused-tp-dit-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=lambda *_a, **_k: b"vae-plan",
        ),
    )
    monkeypatch.setattr(flux_mod, "_serialize_flux_preprocessor", lambda *_a, **_k: b"pre")

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {},
    }

    out = flux_mod.plugin.build_components(str(model_dir), _cfg(), weights, verbose=False)

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert calls["t5_load"] == 1
    assert calls["clip_load"] == 0


def test_build_flux2_components_forwards_precision_to_mistral(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: ensure FLUX.2 build path forwards precision into the Mistral sub-builders.

    Preconditions: FLUX.2 builder modules are replaced with deterministic stubs.
    Postconditions: Mistral load/build receive fp16 precision and DiT stays on fp16 cast dtype.
    """
    calls: dict[str, object] = {}

    model_dir = tmp_path / "flux2_model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    def load_mistral_encoder_weights(path, **kwargs):
        calls["mistral_load"] = (path, kwargs)
        return {"mistral": np.array([1], dtype=np.float16)}

    def build_mistral_encoder_engine(weights, **kwargs):
        calls["mistral_build"] = (weights, kwargs)
        return b"mistral-plan"

    def load_flux2_dit_weights(path, **kwargs):
        calls["dit_load"] = (path, kwargs)
        return {"dit": np.array([2], dtype=np.float32)}

    def build_flux2_dit_engine(weights, **kwargs):
        calls["dit_build"] = (weights, kwargs)
        return b"dit-plan"

    def build_flux_vae_decoder_engine(path, **kwargs):
        calls["vae_build"] = (path, kwargs)
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
            load_mistral_encoder_weights=load_mistral_encoder_weights,
            build_mistral_encoder_engine=build_mistral_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2",
            load_flux2_dit_weights=load_flux2_dit_weights,
            build_flux2_dit_engine=build_flux2_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
            build_flux2_dit_engine=lambda *_a, **_k: b"unused-tp-dit-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=build_flux_vae_decoder_engine,
        ),
    )
    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux2_preprocessor",
        lambda *_a, **_k: b"flux2-preproc",
    )

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {
            "_class_name": "Flux2Transformer2DModel",
            "num_attention_heads": 48,
            "attention_head_dim": 128,
            "num_layers": 8,
            "num_single_layers": 48,
            "timestep_guidance_channels": 256,
        },
        "_text_encoder_config": {
            "text_config": {
                "hidden_size": 5120,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 32768,
                "num_hidden_layers": 40,
                "vocab_size": 131072,
                "rope_theta": 1000000000.0,
            }
        },
        "_vae_config": {
            "latent_channels": 32,
        },
    }

    out = flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=1024, image_width=1024),
        weights,
        precision="fp16",
        verbose=False,
    )

    assert out["text_encoders"] == [("mistral", b"mistral-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"flux2-preproc"
    assert calls["mistral_load"][1]["precision"] == "fp16"
    assert calls["mistral_build"][1]["precision"] == "fp16"
    assert calls["dit_load"][1]["fp8_scales"] is None
    assert calls["dit_build"][1]["cast_dtype"] == "fp16"


def test_build_flux2_components_builds_rank_local_dit_for_tp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: verify FLUX.2 TP builds one rank-local DiT plan per rank.

    Preconditions: FLUX.2 component builders are replaced with deterministic stubs and TRT is 11.x.
    Postconditions: Mistral/VAE stay single-copy while denoiser_ranks carries rank-local plans.
    """
    calls: dict[str, object] = {}
    rank_calls: list[dict[str, object]] = []

    model_dir = tmp_path / "flux2_tp_model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    def load_flux2_dit_weights(path, **kwargs):
        calls["dit_load"] = (path, kwargs)
        return {"dit": np.array([2], dtype=np.float32)}

    def build_flux2_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device FLUX.2 DiT builder used for TP build")

    def build_flux2_dit_tp_engine(_weights, **kwargs):
        parallel = kwargs["parallel_config"]
        rank_calls.append(kwargs)
        return f"flux2-rank-{parallel.rank}".encode("ascii")

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
            load_mistral_encoder_weights=lambda *_a, **_k: {},
            build_mistral_encoder_engine=lambda *_a, **_k: b"mistral-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2",
            load_flux2_dit_weights=load_flux2_dit_weights,
            build_flux2_dit_engine=build_flux2_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
            build_flux2_dit_engine=build_flux2_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=lambda *_a, **_k: b"vae-plan",
        ),
    )
    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux2_preprocessor",
        lambda *_a, **_k: b"flux2-preproc",
    )

    from tensorrt_model_connect import trt_compat
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")

    out = flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=384, image_width=384, max_cache_length=256),
        {
            "_text_encoder_dir": str(model_dir / "text_encoder"),
            "_transformer_dir": str(model_dir / "transformer"),
            "_vae_dir": str(model_dir / "vae"),
            "_transformer_config": {
                "_class_name": "Flux2Transformer2DModel",
                "num_attention_heads": 48,
                "attention_head_dim": 128,
                "num_layers": 8,
                "num_single_layers": 48,
                "timestep_guidance_channels": 256,
            },
            "_text_encoder_config": {
                "text_config": {
                    "hidden_size": 5120,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "intermediate_size": 32768,
                    "num_hidden_layers": 40,
                    "vocab_size": 131072,
                }
            },
            "_vae_config": {"latent_channels": 32},
        },
        precision="fp16",
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4),
        verbose=False,
    )

    assert out["text_encoders"] == [("mistral", b"mistral-plan")]
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"flux2-preproc"
    assert out["denoiser_ranks"] == {
        0: b"flux2-rank-0",
        1: b"flux2-rank-1",
        2: b"flux2-rank-2",
        3: b"flux2-rank-3",
    }
    assert "denoiser" not in out
    assert calls["dit_load"][1]["fp8_scales"] is None
    assert [entry["parallel_config"].rank for entry in rank_calls] == [0, 1, 2, 3]
    assert all(entry["parallel_config"].tp_size == 4 for entry in rank_calls)
    assert all(entry["cast_dtype"] == "fp16" for entry in rank_calls)
    assert all(entry["num_img_tokens"] == 576 for entry in rank_calls)
    assert all(entry["text_seq_len"] == 256 for entry in rank_calls)


def test_build_flux2_components_builds_rank_local_fp8_dit_for_tp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: verify FLUX.2 FP8 TP forwards scales to each rank-local DiT build."""
    calls: dict[str, object] = {}
    rank_calls: list[dict[str, object]] = []

    model_dir = tmp_path / "flux2_fp8_tp_model"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    def load_flux2_dit_weights(path, **kwargs):
        calls["dit_load"] = (path, kwargs)
        return {"dit": np.array([2], dtype=np.float32)}

    def build_flux2_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device FLUX.2 DiT builder used for FP8 TP build")

    def build_flux2_dit_tp_engine(_weights, **kwargs):
        parallel = kwargs["parallel_config"]
        rank_calls.append(kwargs)
        return f"flux2-fp8-rank-{parallel.rank}".encode("ascii")

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
            load_mistral_encoder_weights=lambda *_a, **_k: {},
            build_mistral_encoder_engine=lambda *_a, **_k: b"mistral-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2",
            load_flux2_dit_weights=load_flux2_dit_weights,
            build_flux2_dit_engine=build_flux2_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
            build_flux2_dit_engine=build_flux2_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=lambda *_a, **_k: b"vae-plan",
        ),
    )
    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux2_preprocessor",
        lambda *_a, **_k: b"flux2-preproc",
    )

    from tensorrt_model_connect import trt_compat
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")

    fp8_scales = {
        "transformer_blocks.0.attn.to_q": {
            "input_scale": 0.1,
            "weight_scale": 0.2,
        }
    }

    out = flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=384, image_width=384, max_cache_length=256),
        {
            "_transformer_dir": str(model_dir / "transformer"),
            "_vae_dir": str(model_dir / "vae"),
            "_transformer_config": {
                "_class_name": "Flux2Transformer2DModel",
                "num_attention_heads": 48,
                "attention_head_dim": 128,
                "num_layers": 8,
                "num_single_layers": 48,
                "timestep_guidance_channels": 256,
            },
            "_vae_config": {"latent_channels": 32},
        },
        precision="fp16",
        fp8_scales=fp8_scales,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4),
        verbose=False,
    )

    assert out["denoiser_ranks"] == {
        0: b"flux2-fp8-rank-0",
        1: b"flux2-fp8-rank-1",
        2: b"flux2-fp8-rank-2",
        3: b"flux2-fp8-rank-3",
    }
    assert calls["dit_load"][1]["fp8_scales"] is fp8_scales
    assert [entry["parallel_config"].rank for entry in rank_calls] == [0, 1, 2, 3]
    assert all(entry["fp8_scales"] is fp8_scales for entry in rank_calls)
    assert all(entry["cast_dtype"] == "bf16" for entry in rank_calls)
    assert "denoiser" not in out


def test_build_flux2_components_forwards_fp8_scales_to_dit_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Intent: ensure FLUX.2 FP8 scales are available during DiT weight loading."""
    calls: dict[str, object] = {}

    model_dir = tmp_path / "flux2_model"
    (model_dir / "text_encoder").mkdir(parents=True)
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "vae").mkdir(parents=True)

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
        _module(
            "tensorrt_model_connect.families.flux.model.components.mistral_encoder",
            load_mistral_encoder_weights=lambda *_a, **_k: {},
            build_mistral_encoder_engine=lambda *_a, **_k: b"mistral-plan",
        ),
    )

    def load_flux2_dit_weights(path, **kwargs):
        calls["dit_load"] = (path, kwargs)
        return {"dit": np.array([2], dtype=np.float32)}

    def build_flux2_dit_engine(weights, **kwargs):
        calls["dit_build"] = (weights, kwargs)
        return b"dit-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2",
            load_flux2_dit_weights=load_flux2_dit_weights,
            build_flux2_dit_engine=build_flux2_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
        _module(
            "tensorrt_model_connect.families.flux.model.components.flux2_parallel",
            build_flux2_dit_engine=lambda *_a, **_k: b"unused-tp-dit-plan",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.flux.model.components.vae",
        _module(
            "tensorrt_model_connect.families.flux.model.components.vae",
            build_flux_vae_decoder_engine=lambda *_a, **_k: b"vae-plan",
        ),
    )
    monkeypatch.setattr(
        flux_mod,
        "_serialize_flux2_preprocessor",
        lambda *_a, **_k: b"flux2-preproc",
    )

    weights = {
        "_text_encoder_dir": str(model_dir / "text_encoder"),
        "_transformer_dir": str(model_dir / "transformer"),
        "_vae_dir": str(model_dir / "vae"),
        "_transformer_config": {
            "_class_name": "Flux2Transformer2DModel",
            "num_attention_heads": 48,
            "attention_head_dim": 128,
            "num_layers": 8,
            "num_single_layers": 48,
            "timestep_guidance_channels": 256,
        },
        "_text_encoder_config": {
            "text_config": {
                "hidden_size": 5120,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 32768,
                "num_hidden_layers": 40,
                "vocab_size": 131072,
            }
        },
        "_vae_config": {"latent_channels": 32},
    }
    fp8_scales = {
        "transformer_blocks.0.attn.to_q": {
            "input_scale": 0.1,
            "weight_scale": 0.2,
        }
    }

    flux_mod.plugin.build_components(
        str(model_dir),
        _cfg(image_height=1024, image_width=1024),
        weights,
        fp8_scales=fp8_scales,
        verbose=False,
    )

    assert calls["dit_load"][1]["fp8_scales"] is fp8_scales
    assert calls["dit_build"][1]["fp8_scales"] is fp8_scales
    assert calls["dit_build"][1]["cast_dtype"] == "bf16"


def test_flux2_mha_forwards_fp8_attention_scales(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Intent: ensure FLUX.2 passes BMM scales into native IAttention."""
    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__version__ = "test"
    fake_trt.float16 = object()
    fake_trt.bfloat16 = object()
    fake_trt.float32 = object()
    fake_trt.DataType = types.SimpleNamespace(FP8=object())
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    sys.modules.pop("tensorrt_model_connect.graph_ops", None)
    sys.modules.pop(
        "tensorrt_model_connect.families.flux.model.components.flux2", None)

    def _cleanup_trt_imports() -> None:
        import tensorrt_model_connect.trt_compat as trt_compat
        trt_compat._module = None
        sys.modules.pop("tensorrt_model_connect.graph_ops", None)
        sys.modules.pop(
            "tensorrt_model_connect.families.flux.model.components.flux2", None)

    request.addfinalizer(_cleanup_trt_imports)

    import tensorrt_model_connect.families.flux.model.components.flux2 as flux2_builder

    calls: dict[str, object] = {}

    def fake_attention_from_rows(*args, **kwargs):
        calls["kwargs"] = kwargs
        return "attention-output"

    monkeypatch.setattr(
        flux2_builder.graph_ops,
        "add_attention_from_rows",
        fake_attention_from_rows,
    )
    monkeypatch.setattr(flux2_builder, "_FP8_MODE", True)
    monkeypatch.setattr(
        flux2_builder,
        "_FP8_SCALES",
        {
            "transformer_blocks.0.attn": {
                "q_bmm_scale": 0.1,
                "k_bmm_scale": 0.2,
                "v_bmm_scale": 0.3,
                "softmax_scale": 0.4,
                "bmm2_output_scale": 0.5,
            }
        },
    )

    out = flux2_builder._mha(
        object(), "q", "k", "v", 48, 128, 4096,
        prefix="transformer_blocks.0.attn")

    assert out == "attention-output"
    assert calls["kwargs"]["quant_scales"] == {
        "q_bmm_scale": 0.1,
        "k_bmm_scale": 0.2,
        "v_bmm_scale": 0.3,
        "softmax_scale": 0.4,
        "bmm2_output_scale": 0.5,
    }
    assert calls["kwargs"]["tag"] == "transformer_blocks.0.attn"


def test_get_diffusion_config_guidance_toggle() -> None:
    """Intent: verify guidance-dependent scheduler defaults in diffusion config.

    Preconditions: config.raw contains transformer settings with/without guidance_embeds.
    Postconditions: step count and guidance scale follow the flag.
    """
    cfg_guided = _cfg(
        _transformer_config={
            "guidance_embeds": True,
            "num_attention_heads": 5,
            "attention_head_dim": 6,
        }
    )
    guided = flux_mod.plugin.get_diffusion_config(cfg_guided)
    assert guided["num_inference_steps"] == 28
    assert guided["guidance_scale"] == 3.5
    assert guided["guidance_embeds"] == 1
    assert guided["dit_dim"] == 30

    cfg_fast = _cfg(_transformer_config={"guidance_embeds": False})
    fast = flux_mod.plugin.get_diffusion_config(cfg_fast)
    assert fast["num_inference_steps"] == 4
    assert fast["guidance_scale"] == 0.0
    assert fast["guidance_embeds"] == 0


def test_serialize_flux_preprocessor_guidance_key_control() -> None:
    """Intent: validate key mapping and guidance key gating in serialized preprocessor blob.

    Preconditions: source dict includes base and guidance tensors.
    Postconditions: guidance keys are only serialized when guidance_embeds=True.
    """
    base = {
        "x_embedder.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "x_embedder.bias": np.array([1.0, 2.0], dtype=np.float32),
        "context_embedder.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
        "context_embedder.bias": np.array([3.0, 4.0], dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_1.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.timestep_embedder.linear_1.bias": np.array([5.0, 6.0], dtype=np.float32),
        "time_text_embed.timestep_embedder.linear_2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.timestep_embedder.linear_2.bias": np.array([7.0, 8.0], dtype=np.float32),
        "time_text_embed.text_embedder.linear_1.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.text_embedder.linear_1.bias": np.array([9.0, 10.0], dtype=np.float32),
        "time_text_embed.text_embedder.linear_2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.text_embedder.linear_2.bias": np.array([11.0, 12.0], dtype=np.float32),
        "time_text_embed.guidance_embedder.linear_1.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.guidance_embedder.linear_1.bias": np.array([13.0, 14.0], dtype=np.float32),
        "time_text_embed.guidance_embedder.linear_2.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "time_text_embed.guidance_embedder.linear_2.bias": np.array([15.0, 16.0], dtype=np.float32),
    }

    idx_off, _payload_off = _decode_blob(flux_mod._serialize_flux_preprocessor(base, guidance_embeds=False))
    assert "condition_embedder.guidance_embedding.0.weight" not in idx_off

    idx_on, payload_on = _decode_blob(flux_mod._serialize_flux_preprocessor(base, guidance_embeds=True))
    assert "condition_embedder.guidance_embedding.0.weight" in idx_on

    max_end = 0
    for info in idx_on.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload_on) == max_end
