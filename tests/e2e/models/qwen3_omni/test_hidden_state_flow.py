# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_qwen3_omni_thinker_marks_hidden_state_output() -> None:
    source = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()

    assert 'hidden_out.name = "hidden_state"' in source
    assert "network.mark_output(hidden_out)" in source


def test_qwen3_omni_audio_generation_fails_closed_without_native_talker() -> None:
    source = (ROOT / "src/runtime/models/qwen3_omni/pipeline.cpp").read_text()

    assert "native Qwen3-Omni Talker is unavailable" in source
    assert "audio generation is disabled" in source
    assert "Qwen3OmniTalkerRuntime" not in source
    assert "talker_runtime_->run" not in source


def test_qwen3_omni_detects_real_talker_checkpoint_keys() -> None:
    source = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()

    assert "talker.model.codec_embedding.weight" in source
    assert "talker.code_predictor.lm_head" in source
    assert "num_code_groups" in source


def test_qwen3_omni_does_not_build_incomplete_talker_projection() -> None:
    source = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()

    assert "def _build_talker_engine" not in source
    assert 'result["talker_engine_plan"]' not in source
    assert "def _build_code2wav_engine" in source
    assert "def build_extra_engines(" not in source


def test_qwen3_omni_runtime_does_not_load_code2wav_for_native_text() -> None:
    source = (ROOT / "src/runtime/models/qwen3_omni/plugin.cpp").read_text()
    builder = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()

    assert "code2wav_engine_plan" not in source
    assert "code2wav_module" not in source
    assert "required official Code2Wav engine is missing" not in source
    assert "ctx.hf_python" not in source
    assert 'tensor_dtype("cache_k_0")' in source
    assert '"omni_talker_model_id"' not in source
    assert 'overrides["omni_talker_model_id"]' not in builder
    assert 'overrides["omni_talker_model_revision"]' not in builder
    assert 'find_section(ctx.bundle, "talker_engine_plan")' not in source


def test_qwen3_omni_builder_exposes_only_the_thinker_contract() -> None:
    source = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()
    load_weights = source.split("def load_weights(", 1)[1].split(
        "def _detect_audio_encoder(", 1
    )[0]
    build_engine = source.split("def build_engine(", 1)[1].split(
        "def _build_vision_engine(", 1
    )[0]

    assert "self._detect_audio_encoder" not in load_weights
    assert "self._detect_talker" not in load_weights
    assert "self._detect_code2wav" not in load_weights
    assert 'weights[f"audio.' not in load_weights
    assert 'weights[f"vision.' not in load_weights
    assert 'weights[f"code2wav.' not in load_weights
    assert '"input_embed"' not in build_engine
    assert '"use_input_embed"' not in build_engine
    assert "def build_vision_engine(" not in source
    assert "def get_vl_config(" not in source
    assert "def build_extra_engines(" not in source


def test_qwen3_omni_runtime_has_no_retired_talker_recurrent_state() -> None:
    runtime = ROOT / "src/runtime/models/qwen3_omni"
    plugin = (runtime / "plugin.cpp").read_text()

    assert "Qwen3OmniKvCache" in plugin
    assert "Qwen3OmniRecurrentState" not in plugin
    assert "runtime/models/qwen3_omni/recurrent_state.h" not in plugin
    assert not (runtime / "recurrent_state.h").exists()
    assert not (runtime / "recurrent_state.cpp").exists()
