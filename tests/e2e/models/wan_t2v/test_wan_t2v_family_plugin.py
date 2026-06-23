"""Unit tests for Wan T2V family plugin and preprocessor serialization.

Trace: ARCH-FAM-001, UD-FAM-WAN-T2V
Intent: Validate Wan T2V diffusion family plugin matching, weight serialization, and video config encoding
Preconditions: Synthetic Wan T2V model config with video dimensions and weight tensors are available
Postconditions: Plugin matches Wan aliases, serializes preprocessor weights correctly, and encodes video parameters
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
    import tensorrt_model_connect.families.wan_t2v as wan_mod
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _cfg(**raw_overrides: object) -> ModelConfig:
    payload = {
        "model_type": "wan_t2v",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "video_height": 64,
        "video_width": 80,
        "video_num_frames": 9,
    }
    payload.update(raw_overrides)
    return ModelConfig.from_json(json.dumps(payload))


def _decode_blob(blob: bytes) -> tuple[dict[str, dict], bytes]:
    idx_len = struct.unpack("<I", blob[:4])[0]
    index = json.loads(blob[4:4 + idx_len].decode("utf-8"))
    payload = blob[4 + idx_len:]
    return index, payload


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_matches_and_build_engine_not_supported() -> None:
    """Intent: validate model-type routing and explicit build_engine rejection.

    Preconditions: plugin is imported.
    Postconditions: supported aliases match and build_engine raises NotImplementedError.
    """
    plugin = wan_mod.plugin
    assert plugin.matches("wan_t2v")
    assert plugin.matches("wan")
    assert plugin.matches("Wan2.1")
    assert not plugin.matches("flux")

    with pytest.raises(NotImplementedError, match="build_components"):
        plugin.build_engine(_cfg(), {}, 16)


def test_wan_pipeline_classes_resolve_to_wan_plugin() -> None:
    """Wan owns the real Diffusers pipeline class mapping for Wan models."""
    from tensorrt_model_connect.families import find_diffusion_plugin

    for pipeline_class in ("WanPipeline", "WanVideoToVideoPipeline"):
        assert find_diffusion_plugin(pipeline_class) is wan_mod.plugin


def test_load_weights_requires_diffusers_model_index(tmp_path) -> None:
    """Intent: cover both diffusers-detection branches in load_weights.

    Preconditions: one temp directory contains model_index.json and another does not.
    Postconditions: success path returns expected subdir keys; failure path raises ValueError.
    """
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}")

    weights = wan_mod.plugin.load_weights(str(model_dir), _cfg())
    assert weights["_model_format"] == "diffusers"
    assert weights["_text_encoder_dir"].endswith("text_encoder")
    assert weights["_transformer_dir"].endswith("transformer")
    assert weights["_vae_dir"].endswith("vae")

    bad_dir = tmp_path / "wan_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        wan_mod.plugin.load_weights(str(bad_dir), _cfg())


def test_build_components_calls_all_subbuilders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify build_components orchestration and computed num_patches.

    Preconditions: all imported builder modules are monkeypatched with deterministic stubs.
    Postconditions: each sub-builder receives expected arguments and return dict shape is correct.
    """
    calls: dict[str, object] = {}

    def load_t5_weights(path, **kwargs):
        calls["load_t5_weights"] = {"path": path, **kwargs}
        return {"t5.weight": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls["build_t5_encoder_engine"] = {"weights": weights, **kwargs}
        return b"t5-plan"

    def load_dit_weights(path, **kwargs):
        calls["load_dit_weights"] = {"path": path, **kwargs}
        return {"dit.weight": np.array([2], dtype=np.float32)}

    def build_standard_dit_engine(weights, **kwargs):
        calls["build_standard_dit_engine"] = {"weights": weights, **kwargs}
        return b"dit-plan"

    def load_vae_weights(path, **kwargs):
        calls["load_vae_weights"] = {"path": path, **kwargs}
        return {"vae.weight": np.array([3], dtype=np.float32)}

    def build_causal_vae_3d_engine(weights, **kwargs):
        calls["build_causal_vae_3d_engine"] = {"weights": weights, **kwargs}
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.standard_dit_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.standard_dit_builder",
            load_dit_weights=load_dit_weights,
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.causal_vae_3d_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.causal_vae_3d_builder",
            load_vae_weights=load_vae_weights,
            build_causal_vae_3d_engine=build_causal_vae_3d_engine,
            count_vae_caches=lambda **_kwargs: 0,
        ),
    )

    monkeypatch.setattr(
        wan_mod,
        "_serialize_preprocessor_weights",
        lambda dit_weights: b"wan-preproc",
    )

    cfg = _cfg(video_height=64, video_width=80, video_num_frames=9)
    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    out = wan_mod.plugin.build_components(
        "/model", cfg, weights, precision="fp16", verbose=True)

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["preprocessor_weights"] == b"wan-preproc"

    # Preconditions ensure 64x80 and 9 frames.
    # Postcondition: num_patches = 60 using Wan's latent+patching math.
    assert calls["load_t5_weights"]["precision"] == "fp16"
    assert calls["build_standard_dit_engine"]["num_patches"] == 60
    assert calls["build_standard_dit_engine"]["context_dim"] == wan_mod.plugin._DIT_DIM


def test_build_components_tensor_parallel_builds_rank_denoisers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intent: verify Wan TP packaging keeps T5/VAE single-copy and builds rank DiTs.

    Preconditions: builder modules are monkeypatched and TensorRT version is 11.0.
    Postconditions: denoiser_ranks contains one rank-local plan per requested TP rank.
    """
    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.parallel_config import ParallelConfig

    calls: dict[str, object] = {"dit_ranks": []}

    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.0.0")

    def load_t5_weights(path, **kwargs):
        calls["load_t5_weights"] = {"path": path, **kwargs}
        return {"t5.weight": np.array([1], dtype=np.float32)}

    def build_t5_encoder_engine(weights, **kwargs):
        calls["build_t5_encoder_engine"] = {"weights": weights, **kwargs}
        return b"t5-plan"

    def load_dit_weights(path, **kwargs):
        calls["load_dit_weights"] = {"path": path, **kwargs}
        return {"dit.weight": np.array([2], dtype=np.float32)}

    def build_standard_dit_engine(_weights, **_kwargs):
        raise AssertionError("single-device Wan DiT builder used for TP build")

    def build_standard_dit_tp_engine(weights, **kwargs):
        parallel = kwargs["parallel_config"]
        calls["dit_ranks"].append(parallel.rank)
        return f"dit-rank-{parallel.rank}".encode()

    def load_vae_weights(path, **kwargs):
        calls["load_vae_weights"] = {"path": path, **kwargs}
        return {"vae.weight": np.array([3], dtype=np.float32)}

    def build_causal_vae_3d_engine(weights, **kwargs):
        calls["build_causal_vae_3d_engine"] = {"weights": weights, **kwargs}
        return b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.t5_encoder_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.standard_dit_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.standard_dit_builder",
            load_dit_weights=load_dit_weights,
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.standard_dit_tp_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.standard_dit_tp_builder",
            build_standard_dit_engine=build_standard_dit_tp_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.causal_vae_3d_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.causal_vae_3d_builder",
            load_vae_weights=load_vae_weights,
            build_causal_vae_3d_engine=build_causal_vae_3d_engine,
            count_vae_caches=lambda **_kwargs: 0,
        ),
    )
    monkeypatch.setattr(
        wan_mod,
        "_serialize_preprocessor_weights",
        lambda dit_weights: b"wan-preproc",
    )

    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }

    out = wan_mod.plugin.build_components(
        "/model",
        _cfg(video_height=64, video_width=80, video_num_frames=9),
        weights,
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


def test_get_diffusion_config_uses_count_vae_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: verify diffusion config wiring and imported cache-count helper call.

    Preconditions: count_vae_caches is monkeypatched to a known constant.
    Postconditions: returned config includes custom image/video dimensions and mocked cache count.
    """
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_model_connect.families.wan_t2v.causal_vae_3d_builder",
        _module(
            "tensorrt_model_connect.families.wan_t2v.causal_vae_3d_builder",
            count_vae_caches=lambda **_kwargs: 13,
        ),
    )

    cfg = _cfg(video_height=96, video_width=160, video_num_frames=13)
    dc = wan_mod.plugin.get_diffusion_config(cfg)

    assert dc["video_height"] == 96
    assert dc["video_width"] == 160
    assert dc["video_num_frames"] == 13
    assert dc["num_vae_caches"] == 13
    assert dc["diffusion_backend_type"] == "wan_3d"


def test_serialize_preprocessor_weights_transforms_patch_weight() -> None:
    """Intent: validate binary serialization, key filtering, and Conv3D flatten+transpose.

    Preconditions: dit_weights includes a Conv3D patch embedding and a subset of listed keys.
    Postconditions: output index maps stored keys with correct shapes and contiguous payload size.
    """
    dit_weights = {
        "patch_embedding.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 1, 2, 2),
        "patch_embedding.bias": np.array([1.0, 2.0], dtype=np.float32),
        "condition_embedder.time_embedding.0.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "condition_embedder.text_embedding_2.bias": np.array([9.0], dtype=np.float32),
    }

    blob = wan_mod._serialize_preprocessor_weights(dit_weights)
    index, payload = _decode_blob(blob)

    assert "patch_embedding.weight" in index
    assert index["patch_embedding.weight"]["shape"] == [12, 2]
    assert "condition_embedder.time_embedding.2.weight" not in index

    max_end = 0
    for info in index.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload) == max_end
