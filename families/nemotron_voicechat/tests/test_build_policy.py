# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused policy tests for the compressed VoiceChat family build."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from families.nemotron_voicechat import model


def _voicechat_sections() -> tuple[dict, dict]:
    stt = {
        "perception": {
            "encoder": {
                "d_model": 1024,
                "n_layers": 24,
                "n_heads": 8,
                "att_context_size": [70, 0],
            },
            "preprocessor": {"features": 128, "preemph": 0.97},
        }
    }
    speech = {
        "tts_config": {
            "backbone_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "head_dim": 72,
                "sliding_window": 7500,
                "sliding_window_pattern": 6,
                "max_position_embeddings": 131072,
            },
            "num_quantizers": 31,
            "codebook_size": 1024,
            "mog_head_config": {"num_predictions": 1024},
        },
        "codec_config": {
            "num_quantizers": 31,
            "codebook_size": 1024,
            "latent_size": 512,
            "wav_to_token_ratio": 1764,
        },
    }
    return stt, speech


def _write_text_asset_fixtures(root: Path) -> None:
    for filename in model._TEXT_ASSETS:
        (root / filename).write_text(f"fixture:{filename}", encoding="utf-8")


def test_quantization_policy_selects_only_supported_runtime_absmax_w8a8() -> None:
    assert model._normalize_quantization(None) is None
    assert model._normalize_quantization("none") is None
    assert model._normalize_quantization("int8") == "int8_sq"
    assert model._normalize_quantization("INT8-SQ") == "int8_sq"
    with pytest.raises(ValueError, match="only int8/int8_sq"):
        model._normalize_quantization("fp8")


def test_thinker_selection_keeps_language_head_out_of_w8a8() -> None:
    names = model._thinker_quantized_weight_names(
        {"_layer_types": ["mamba2", "mlp", "attention"]}
    )
    assert names == [
        "layer.0.mamba_in_proj",
        "layer.0.mamba_out_proj",
        "layer.1.w_up",
        "layer.1.w_down",
        "layer.2.w_q",
        "layer.2.w_k",
        "layer.2.w_v",
        "layer.2.w_o",
        "w_function_head",
    ]
    assert "w_lm_head" not in names


def test_text_asset_download_is_revision_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_text_asset_fixtures(tmp_path)
    received: dict[str, object] = {}
    hub = ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        received.update(kwargs)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert model._resolve_text_assets() == tmp_path
    assert received == {
        "repo_id": model.TEXT_MODEL_ID,
        "revision": model.TEXT_MODEL_REVISION,
        "allow_patterns": list(model._TEXT_ASSETS),
    }


def test_text_asset_download_rejects_partial_snapshot_and_names_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_text_asset_fixtures(tmp_path)
    missing_asset = model._TEXT_ASSETS[-1]
    (tmp_path / missing_asset).unlink()
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **_kwargs: str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    with pytest.raises(FileNotFoundError, match="missing required files") as exc_info:
        model._resolve_text_assets()
    assert missing_asset in str(exc_info.value)


def test_runtime_records_compression_and_omits_it_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_core = ModuleType("families.nemotron_voicechat.native_core")
    native_core._parse_layer_types = lambda pattern: [
        {"M": "mamba2", "-": "mlp", "*": "attention"}[character]
        for character in pattern
    ]
    monkeypatch.setitem(sys.modules, native_core.__name__, native_core)
    monkeypatch.setattr(
        sys.modules["families.nemotron_voicechat"],
        "native_core",
        native_core,
        raising=False,
    )
    stt, speech = _voicechat_sections()
    thinker = SimpleNamespace(
        vocab_size=131072,
        hidden_size=4480,
        num_hidden_layers=56,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
    )

    runtime = model._runtime_config(
        thinker=thinker,
        stt=stt,
        speech=speech,
        precision="fp32",
        quantization="int8_sq",
        tts_linear_precision="fp16",
        max_cache_length=8192,
        mel_length=3000,
    )

    assert runtime["quantization"] == {
        "format": "int8_sq",
        "scheme": "w8a8",
        "scale_source": "runtime_absmax",
        "scope": "thinker_static_gemms_except_lm_head",
    }
    assert runtime["thinker_embedding_precision"] == "fp16"
    assert runtime["thinker_lm_head_precision"] == "fp16"
    assert runtime["tts_linear_precision"] == "fp16"
    assert runtime["tts_sliding_window_pattern"] == 6
    assert runtime["tts_max_position_embeddings"] == 131072
    assert runtime["context_rollover_soft_frames"] == 1125
    assert runtime["context_rollover_hard_frames"] == 1375
    assert runtime["context_memory_max_tokens"] == 96

    default_runtime = model._runtime_config(
        thinker=thinker,
        stt=stt,
        speech=speech,
        precision="fp32",
        quantization=None,
        tts_linear_precision="fp32",
        max_cache_length=8192,
        mel_length=3000,
    )
    assert default_runtime["tts_linear_precision"] == "fp32"
    assert {
        "quantization",
        "quantization_format",
        "quantization_scheme",
        "quantization_scope",
        "quantization_scale_source",
        "quantization_experimental",
        "thinker_embedding_precision",
        "thinker_lm_head_precision",
    }.isdisjoint(default_runtime)
