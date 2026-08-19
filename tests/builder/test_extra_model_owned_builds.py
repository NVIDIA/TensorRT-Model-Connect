# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle-orchestration contracts for families with extra engine sections."""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from tensorrt_model_connect import build_timing, bundle_writer, trt_compat


FAMILY_EXTRA_KWARGS = {
    "bark": {"precision", "verbose", "build_timing", "parallel_config"},
    "elf_flow": {"precision", "verbose", "build_timing"},
    "nemotron_labs_diffusion": {"precision", "quant_ctx", "verbose"},
    "personaplex": {"precision", "verbose", "build_timing"},
    "sana_wm": {"precision", "verbose", "build_timing"},
    "canary": {"precision", "verbose"},
    "magpie_tts": {"precision", "verbose", "build_timing"},
    "nemotron_speech_streaming": {"precision", "verbose"},
    "qwen3_omni": {"precision", "verbose", "build_timing"},
    "sam3": {"precision", "verbose"},
    "whisper": {"precision", "verbose"},
}

VISION_FAMILIES = {
    "canary",
    "magpie_tts",
    "nemotron_speech_streaming",
    "qwen3_omni",
    "sam3",
    "whisper",
}


def _fake_trt() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="11.1.0",
        ElementWiseOperation=types.SimpleNamespace(
            SUM="sum", SUB="sub", PROD="prod", DIV="div", POW="pow"
        ),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh", RELU="relu"),
        ReduceOperation=types.SimpleNamespace(AVG="avg", SUM="sum", MAX="max"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip", EXP="exp", LOG="log"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        BuilderFlag=types.SimpleNamespace(TF32="tf32", DISABLE_TIMING_CACHE="disable"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        TopKOperation=types.SimpleNamespace(MAX="max"),
        SliceMode=types.SimpleNamespace(WRAP="wrap"),
        Permutation=lambda value: tuple(value),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
        int64="int64",
        bool="bool",
    )


@pytest.fixture(autouse=True)
def fake_trt_module(monkeypatch: pytest.MonkeyPatch):
    fake = _fake_trt()
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    monkeypatch.setattr(trt_compat, "_module", fake)


def _config(family: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_type=family,
        raw={"model_type": family, "architectures": []},
        architectures=[],
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _load_model(family: str):
    return importlib.import_module(f"tensorrt_model_connect.models.{family}.model")


@pytest.mark.parametrize("family", FAMILY_EXTRA_KWARGS)
def test_extra_family_owns_complete_bundle_assembly(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = _load_model(family)
    config = _config(family)
    extra_calls: list[dict[str, object]] = []
    vision_calls: list[dict[str, object]] = []
    written = []

    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model, "load_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        model,
        "_build_local_engine",
        lambda *_args, **_kwargs: (b"primary-plan", "single"),
    )

    def extra_engines(*_args, **kwargs):
        extra_calls.append(kwargs)
        return {
            "family_extra_engine_plan": b"extra-plan",
            "family_opaque_metadata.bin": b"opaque-data",
        }

    monkeypatch.setattr(model, "build_extra_engines", extra_engines)
    if family in VISION_FAMILIES:
        monkeypatch.setattr(
            model,
            "build_vision_engine",
            lambda *_args, **kwargs: vision_calls.append(kwargs) or b"vision-plan",
        )

    for hook_name in (
        "get_audio_config",
        "get_vl_config",
        "get_segmentation_config",
        "get_bundle_config_overrides",
        "get_lora_config",
    ):
        if hasattr(model, hook_name):
            monkeypatch.setattr(
                model,
                hook_name,
                lambda _config, marker=f"hook-{family}": {"hook_marker": marker},
            )

    monkeypatch.setattr(build_timing, "new_build_timing", lambda _path: {})
    monkeypatch.setattr(build_timing, "add_build_timing", lambda *_args: None)
    monkeypatch.setattr(build_timing, "write_build_timing", lambda *_args: None)
    monkeypatch.setattr(build_timing, "build_timing_phase", lambda *_args: 0.0)
    monkeypatch.setattr(build_timing, "untracked_phase_time", lambda elapsed, *_args: elapsed)
    monkeypatch.setattr(bundle_writer, "tensorrt_version", lambda: "11.1.0")
    monkeypatch.setattr(bundle_writer, "tensorrt_abi", lambda _version: "11.1")
    monkeypatch.setattr(bundle_writer, "gpu_name", lambda: "test-gpu")
    monkeypatch.setattr(
        bundle_writer,
        "write_bundle",
        lambda path, info, sections: written.append((path, info, list(sections))),
    )
    tokenizer = importlib.import_module("tensorrt_model_connect.tokenizer_conversion")
    monkeypatch.setattr(
        tokenizer,
        "prepare_tokenizer_special_frame",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        tokenizer,
        "detect_tokenizer_special_frame",
        lambda *_args, **_kwargs: ([], []),
    )
    graph_build = importlib.import_module("tensorrt_model_connect.tvm_ffi.graph_build")
    monkeypatch.setattr(graph_build, "kernel_slots_section", lambda: None)

    model.build(
        str(tmp_path),
        str(tmp_path / f"{family}.bundle"),
        precision="fp16",
        verbose=True,
    )

    assert len(written) == 1
    _, info, sections = written[0]
    section_map = {section.name: section.data for section in sections}
    runtime = json.loads(section_map["config.json"])
    assert info.family == model.name
    assert section_map["engine_plan"] == b"primary-plan"
    assert section_map["family_extra_engine_plan"] == b"extra-plan"
    assert section_map["family_opaque_metadata.bin"] == b"opaque-data"
    assert runtime["runtime_strategy"] == model.runtime_strategy
    assert f"hook-{family}" in json.dumps(runtime)
    assert not any(str(key).startswith("_") for key in runtime)
    assert len(extra_calls) == 1
    assert set(extra_calls[0]) == FAMILY_EXTRA_KWARGS[family]
    if family in VISION_FAMILIES:
        assert section_map["vision_engine_plan"] == b"vision-plan"
        assert len(vision_calls) == 1
        assert set(vision_calls[0]) == {"precision", "verbose"}
        assert runtime["has_vision_engine"] is True
    else:
        assert "vision_engine_plan" not in section_map
        assert not vision_calls
