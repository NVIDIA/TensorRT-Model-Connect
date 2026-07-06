# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Z-Image family plugin and serialization helpers.

Trace: ARCH-FAM-001, UD-FAM-ZIMAGE
Intent: Validate Z-Image diffusion family plugin matching, weight serialization, and image config encoding
Preconditions: Synthetic Z-Image model config and weight tensors are available
Postconditions: Plugin matches Z-Image aliases, serializes preprocessor weights correctly, and rejects build_engine
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
    import tensorrt_model_connect.families.z_image as zimg_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "z_image",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_height": 1024,
        "image_width": 768,
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
    """Intent: verify alias matching and explicit build_engine rejection.

    Preconditions: plugin object is imported.
    Postconditions: all declared aliases match and build_engine raises NotImplementedError.
    """
    plugin = zimg_mod.plugin
    assert plugin.matches("z_image")
    assert plugin.matches("zimage")
    assert plugin.matches("z-image")
    assert plugin.matches("zimagepipeline")
    assert not plugin.matches("pixart")

    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_cfg(), {}, 16)


def test_load_weights_requires_diffusers_model_index(tmp_path) -> None:
    """Intent: cover load_weights success and error branches.

    Preconditions: one model directory has model_index.json and one does not.
    Postconditions: success path populates required directory pointers; error path raises ValueError.
    """
    model_dir = tmp_path / "zimg"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}")

    weights = zimg_mod.plugin.load_weights(str(model_dir), _cfg())
    assert weights["_model_format"] == "diffusers"
    assert weights["_text_encoder_dir"].endswith("text_encoder")
    assert weights["_transformer_dir"].endswith("transformer")
    assert weights["_vae_dir"].endswith("vae")
    assert weights["_tokenizer_dir"].endswith("tokenizer")
    assert weights["_model_dir"] == str(model_dir)

    bad_dir = tmp_path / "zimg_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        zimg_mod.plugin.load_weights(str(bad_dir), _cfg())


@pytest.mark.parametrize(
    ("fp32_layers", "dit_precision", "dit_fp32_layers", "vae_precision"),
    [
        ([1], "fp32", (), "fp16"),
        ([2, 3, 7, 37], "fp16", (0, 4, 34), "fp32"),
    ],
)
def test_build_components_calls_all_subbuilders(
    monkeypatch: pytest.MonkeyPatch,
    fp32_layers: list[int],
    dit_precision: str,
    dit_fp32_layers: tuple[int, ...],
    vae_precision: str,
) -> None:
    """Intent: verify builder orchestration and latent-size derived num_patches.

    Preconditions: imported builder modules are monkeypatched with deterministic stubs.
    Postconditions: returned plans and call arguments reflect Z-Image latent/patch math.
    """
    calls: dict[str, object] = {}

    def load_qwen3_encoder_weights(path, **kwargs):
        calls["load_qwen3_encoder_weights"] = (path, kwargs)
        return {"te": np.array([1], dtype=np.float32)}

    def build_qwen3_encoder_engine(weights, **kwargs):
        calls["build_qwen3_encoder_engine"] = (weights, kwargs)
        return b"te-plan"

    def load_z_image_dit_weights(path, **kwargs):
        calls["load_z_image_dit_weights"] = (path, kwargs)
        return {"dit": np.array([2], dtype=np.float32)}

    def build_z_image_dit_engine(weights, **kwargs):
        calls["build_z_image_dit_engine"] = (weights, kwargs)
        return b"dit-plan"

    def build_vae_2d_decoder_engine(path, **kwargs):
        calls["build_vae_2d_decoder_engine"] = (path, kwargs)
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.qwen3_encoder_builder",
        _module(
            "tensorrt_model_connect.families.z_image.qwen3_encoder_builder",
            load_qwen3_encoder_weights=load_qwen3_encoder_weights,
            build_qwen3_encoder_engine=build_qwen3_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.z_image_dit_builder",
        _module(
            "tensorrt_model_connect.families.z_image.z_image_dit_builder",
            load_z_image_dit_weights=load_z_image_dit_weights,
            build_z_image_dit_engine=build_z_image_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.vae_2d_builder",
        _module(
            "tensorrt_model_connect.families.z_image.vae_2d_builder",
            build_vae_2d_decoder_engine=build_vae_2d_decoder_engine,
        ),
    )
    monkeypatch.setattr(zimg_mod, "_serialize_preprocessor_weights", lambda _w: b"zimg-pre")

    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    out = zimg_mod.plugin.build_components(
        "/model",
        _cfg(image_height=1024, image_width=768, _fp32_layers=fp32_layers),
        weights,
        precision="fp16",
        verbose=True,
    )

    assert out["text_encoders"] == [("qwen3", b"te-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"zimg-pre"

    # h_lat=128, w_lat=96, patch=2x2 -> 3072 patches.
    assert calls["build_z_image_dit_engine"][1]["num_patches"] == 3072
    assert calls["build_qwen3_encoder_engine"][1]["precision"] == "fp16"
    assert calls["build_z_image_dit_engine"][1]["precision"] == dit_precision
    assert calls["build_z_image_dit_engine"][1]["fp32_layers"] == dit_fp32_layers
    assert calls["build_vae_2d_decoder_engine"][1]["precision"] == vae_precision
    assert calls["build_vae_2d_decoder_engine"][1]["h_lat"] == 128
    assert calls["build_vae_2d_decoder_engine"][1]["w_lat"] == 96


def test_build_components_tensor_parallel_builds_rank_denoisers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: verify Z-Image TP packaging builds rank-local DiTs only.

    Preconditions: builder modules are monkeypatched and TensorRT version is 11.0.
    Postconditions: denoiser_ranks contains TP=2 rank plans while text/VAE remain single-copy.
    """
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.parallel_config import ParallelConfig

    calls: dict[str, object] = {"dit_ranks": []}

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")

    def load_qwen3_encoder_weights(path, **kwargs):
        calls["load_qwen3_encoder_weights"] = (path, kwargs)
        return {"te": np.array([1], dtype=np.float32)}

    def build_qwen3_encoder_engine(weights, **kwargs):
        calls["build_qwen3_encoder_engine"] = (weights, kwargs)
        return b"te-plan"

    def load_z_image_dit_weights(path, **kwargs):
        calls["load_z_image_dit_weights"] = (path, kwargs)
        return {"dit": np.array([2], dtype=np.float32)}

    def build_z_image_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device Z-Image DiT builder used for TP build")

    def build_z_image_dit_tp_engine(weights, **kwargs):
        parallel = kwargs["parallel_config"]
        calls["dit_ranks"].append(parallel.rank)
        return f"dit-rank-{parallel.rank}".encode()

    def build_vae_2d_decoder_engine(path, **kwargs):
        calls["build_vae_2d_decoder_engine"] = (path, kwargs)
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.qwen3_encoder_builder",
        _module(
            "tensorrt_model_connect.families.z_image.qwen3_encoder_builder",
            load_qwen3_encoder_weights=load_qwen3_encoder_weights,
            build_qwen3_encoder_engine=build_qwen3_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.z_image_dit_builder",
        _module(
            "tensorrt_model_connect.families.z_image.z_image_dit_builder",
            load_z_image_dit_weights=load_z_image_dit_weights,
            build_z_image_dit_engine=build_z_image_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.z_image_dit_tp_builder",
        _module(
            "tensorrt_model_connect.families.z_image.z_image_dit_tp_builder",
            build_z_image_dit_engine=build_z_image_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.z_image.vae_2d_builder",
        _module(
            "tensorrt_model_connect.families.z_image.vae_2d_builder",
            build_vae_2d_decoder_engine=build_vae_2d_decoder_engine,
        ),
    )
    monkeypatch.setattr(zimg_mod, "_serialize_preprocessor_weights", lambda _w: b"zimg-pre")

    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    out = zimg_mod.plugin.build_components(
        "/model",
        _cfg(image_height=512, image_width=512),
        weights,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2),
    )

    assert out["text_encoders"] == [("qwen3", b"te-plan")]
    assert out["denoiser_ranks"] == {
        0: b"dit-rank-0",
        1: b"dit-rank-1",
    }
    assert "denoiser" not in out
    assert out["vae_decoder"] == b"vae-plan"
    assert calls["dit_ranks"] == [0, 1]


def test_get_diffusion_config_uses_correct_latent_math() -> None:
    """Intent: validate latent-dimension and patch-count wiring in diffusion config.

    Preconditions: image dimensions are overridden in config.raw.
    Postconditions: returned config reflects expected backend type and dimensions.
    """
    dc = zimg_mod.plugin.get_diffusion_config(_cfg(image_height=1024, image_width=768))

    assert dc["diffusion_backend_type"] == "z_image_2d"
    assert dc["video_height"] == 1024
    assert dc["video_width"] == 768
    assert dc["num_inference_steps"] == 9
    assert dc["guidance_scale"] == 0.0
    assert dc["scale_factor_spatial"] == zimg_mod.plugin._VAE_SCALE_FACTOR
    assert dc["dit_num_layers"] == zimg_mod.plugin._DIT_NUM_LAYERS


def test_serialize_preprocessor_weights_maps_and_converts_values() -> None:
    """Intent: validate canonical key mapping, dtype normalization, and payload sizing.

    Preconditions: dit_weights includes ndarray and list-like inputs for mapped keys.
    Postconditions: serialized index contains canonical names with contiguous float32 payload.
    """
    dit_weights = {
        "t_emb.0.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "t_emb.0.bias": np.array([1.0, 2.0], dtype=np.float32),
        "cap_norm.weight": np.array([3.0, 4.0], dtype=np.float32),
        "cap_proj.weight": np.arange(4, dtype=np.float32).reshape(2, 2),
        "cap_proj.bias": np.array([5.0, 6.0], dtype=np.float32),
        "x_embedder.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
        "x_embedder.bias": np.array([7.0, 8.0], dtype=np.float32),
        "cap_pad_token": [0.1, 0.2],
    }

    blob = zimg_mod._serialize_preprocessor_weights(dit_weights)
    index, payload = _decode_blob(blob)

    assert "t_embedder.mlp.0.weight" in index
    assert "cap_embedder.norm.weight" in index
    assert "cap_embedder.proj.weight" in index
    assert "x_embedder.weight" in index
    assert "x_pad_token" not in index  # absent source key should not be serialized

    max_end = 0
    for info in index.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload) == max_end


def test_fp16_vae_builder_reaches_checkpoint_validation(tmp_path) -> None:
    """FP16 dtype selection must not shadow the TensorRT module binding."""
    from tensorrt_model_connect.families.z_image.vae_2d_builder import (
        build_vae_2d_decoder_engine,
    )

    with pytest.raises(FileNotFoundError, match="No safetensors found"):
        build_vae_2d_decoder_engine(
            str(tmp_path), h_lat=8, w_lat=8, precision="fp16")
