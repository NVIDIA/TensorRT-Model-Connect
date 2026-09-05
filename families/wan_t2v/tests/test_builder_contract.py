# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan-T2V scheduler, tokenizer, component, and precision contracts."""

from __future__ import annotations

import inspect
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for Wan builder tests")

from families.wan_t2v import model  # noqa: E402
from families.wan_t2v.config import ModelConfig  # noqa: E402


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
    index_length = struct.unpack("<I", blob[:4])[0]
    index = json.loads(blob[4 : 4 + index_length].decode("utf-8"))
    return index, blob[4 + index_length :]


def _module(name: str, **attrs) -> types.ModuleType:
    result = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(result, key, value)
    return result


def test_nightly_wan_build_keeps_complete_t5_encoder_in_fp32() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "wan21-t2v-1.3b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp16"
    assert manifest["fp32_layers"] == [24]


def test_load_weights_requires_diffusers_model_index(tmp_path: Path) -> None:
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    weights = model._WanT2VModel().load_weights(str(model_dir), _cfg())
    assert weights["_model_format"] == "diffusers"
    assert weights["_text_encoder_dir"].endswith("text_encoder")
    assert weights["_transformer_dir"].endswith("transformer")
    assert weights["_vae_dir"].endswith("vae")

    bad_dir = tmp_path / "wan_bad"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match="Expected diffusers format"):
        model._WanT2VModel().load_weights(str(bad_dir), _cfg())


def test_load_weights_preserves_checkpoint_scheduler_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "wan"
    scheduler_dir = model_dir / "scheduler"
    scheduler_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "UniPCMultistepScheduler",
                "num_train_timesteps": 1000,
                "flow_shift": 3.0,
                "solver_order": 2,
                "solver_type": "bh2",
                "prediction_type": "flow_prediction",
                "use_flow_sigmas": True,
                "lower_order_final": True,
                "use_dynamic_shifting": False,
            }
        ),
        encoding="utf-8",
    )
    config = _cfg()

    family = model._WanT2VModel()
    family.load_weights(str(model_dir), config)
    diffusion = family.get_diffusion_config(config)

    assert diffusion["scheduler"] == "unipc_multistep"
    assert diffusion["flow_shift"] == pytest.approx(3.0)
    assert diffusion["unipc_lower_order_final"] == 1
    assert diffusion["use_dynamic_shifting"] == 0


def test_wan_rejects_unsupported_unipc_variant() -> None:
    config = _cfg(
        _scheduler_config={
            "_class_name": "UniPCMultistepScheduler",
            "solver_order": 3,
        }
    )
    with pytest.raises(ValueError, match="order-2 BH2 UniPC"):
        model._WanT2VModel().get_diffusion_config(config)


def test_wan_runtime_owns_t5_special_token_framing() -> None:
    source = inspect.getsource(model.build)
    assert '"tokenizer_add_special_tokens": False' in source
    assert '"tokenizer_prefix_ids": []' in source
    assert '"tokenizer_suffix_ids": [1]' in source


def test_build_components_calls_all_subbuilders(monkeypatch: pytest.MonkeyPatch) -> None:
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
        entry = {"weights": weights, **kwargs}
        calls.setdefault("build_causal_vae_3d_engine", []).append(entry)
        return b"vae-first-frame-plan" if kwargs.get("first_frame_only") else b"vae-plan"

    monkeypatch.setitem(
        sys.modules,
        "families.wan_t2v.t5_encoder_builder",
        _module(
            "families.wan_t2v.t5_encoder_builder",
            load_t5_weights=load_t5_weights,
            build_t5_encoder_engine=build_t5_encoder_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "families.wan_t2v.standard_dit_builder",
        _module(
            "families.wan_t2v.standard_dit_builder",
            load_dit_weights=load_dit_weights,
            build_standard_dit_engine=build_standard_dit_engine,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "families.wan_t2v.standard_dit_tp_builder",
        _module(
            "families.wan_t2v.standard_dit_tp_builder",
            build_standard_dit_engine=lambda *_args, **_kwargs: b"unused-tp",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "families.wan_t2v.standard_dit_cp_builder",
        _module(
            "families.wan_t2v.standard_dit_cp_builder",
            build_standard_dit_engine=lambda *_args, **_kwargs: b"unused-cp",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "families.wan_t2v.causal_vae_3d_builder",
        _module(
            "families.wan_t2v.causal_vae_3d_builder",
            load_vae_weights=load_vae_weights,
            build_causal_vae_3d_engine=build_causal_vae_3d_engine,
            count_vae_caches=lambda **_kwargs: 0,
        ),
    )
    monkeypatch.setattr(
        model, "_serialize_preprocessor_weights", lambda dit_weights: b"wan-preproc"
    )

    out = model._WanT2VModel().build_components(
        "/model",
        _cfg(video_height=64, video_width=80, video_num_frames=9, _fp32_layers=[24]),
        {
            "_text_encoder_dir": "/model/text_encoder",
            "_transformer_dir": "/model/transformer",
            "_vae_dir": "/model/vae",
        },
        precision="fp16",
        verbose=True,
    )

    assert out["text_encoders"] == [("t5", b"t5-plan")]
    assert out["denoiser"] == b"dit-plan"
    assert out["vae_decoder"] == b"vae-plan"
    assert out["vae_decoder_first_frame"] == b"vae-first-frame-plan"
    assert out["preprocessor_weights"] == b"wan-preproc"
    assert calls["load_t5_weights"]["precision"] == "fp32"
    assert calls["build_t5_encoder_engine"]["precision"] == "fp32"
    assert calls["build_standard_dit_engine"]["num_patches"] == 60
    assert calls["build_standard_dit_engine"]["context_dim"] == model._WanT2VModel._DIT_DIM
    assert calls["build_standard_dit_engine"]["precision"] == "fp16"
    vae_calls = calls["build_causal_vae_3d_engine"]
    assert len(vae_calls) == 2
    assert vae_calls[0]["precision"] == "fp16"
    assert vae_calls[0].get("first_frame_only") is None
    assert vae_calls[1]["precision"] == "fp16"
    assert vae_calls[1]["first_frame_only"] is True


def test_build_components_rejects_partial_t5_fp32_selectors() -> None:
    weights = {
        "_text_encoder_dir": "/model/text_encoder",
        "_transformer_dir": "/model/transformer",
        "_vae_dir": "/model/vae",
    }
    with pytest.raises(ValueError, match="supports only selector 24"):
        model._WanT2VModel().build_components(
            "/model",
            _cfg(_fp32_layers=[0, 23]),
            weights,
            precision="fp16",
        )


def test_get_diffusion_config_uses_count_vae_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "families.wan_t2v.causal_vae_3d_builder",
        _module(
            "families.wan_t2v.causal_vae_3d_builder",
            count_vae_caches=lambda **_kwargs: 13,
        ),
    )
    config = _cfg(video_height=96, video_width=160, video_num_frames=13)
    diffusion = model._WanT2VModel().get_diffusion_config(config)

    assert diffusion["video_height"] == 96
    assert diffusion["video_width"] == 160
    assert diffusion["video_num_frames"] == 13
    assert diffusion["num_vae_caches"] == 13
    assert diffusion["diffusion_backend_type"] == "wan_3d"


def test_serialize_preprocessor_weights_transforms_patch_weight() -> None:
    dit_weights = {
        "patch_embedding.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 1, 2, 2),
        "patch_embedding.bias": np.array([1.0, 2.0], dtype=np.float32),
        "condition_embedder.time_embedding.0.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "condition_embedder.text_embedding_2.bias": np.array([9.0], dtype=np.float32),
    }

    index, payload = _decode_blob(model._serialize_preprocessor_weights(dit_weights))

    assert "patch_embedding.weight" in index
    assert index["patch_embedding.weight"]["shape"] == [12, 2]
    assert "condition_embedder.time_embedding.2.weight" not in index
    max_end = 0
    for info in index.values():
        nbytes = int(np.prod(info["shape"])) * 4
        max_end = max(max_end, info["offset"] + nbytes)
    assert len(payload) == max_end
