# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def test_qwen3_omni_thinker_marks_hidden_state_output() -> None:
    source = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()

    assert 'hidden_out.name = "hidden_state"' in source
    assert "network.mark_output(hidden_out)" in source


def test_qwen3_omni_runtime_feeds_generated_text_to_official_talker() -> None:
    source = (ROOT / "src/runtime/models/qwen3_omni/pipeline.cpp").read_text()

    assert "tokenizer_->decode(text_tokens)" in source
    assert "run_qwen3_omni_official_talker" in source
    assert "format_omni_chat_prompt(prompt)" in source
    assert "token == config_->thinker_eos_token_id" in source
    assert 'outputs.find("hidden_state")' not in source


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


def test_qwen3_omni_runtime_requires_official_code2wav_and_python_talker() -> None:
    source = (ROOT / "src/runtime/models/qwen3_omni/plugin.cpp").read_text()
    builder = (ROOT / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py").read_text()

    assert "required official Code2Wav engine is missing" in source
    assert "omni_cfg.hf_python = ctx.hf_python" in source
    assert 'tensor_dtype("cache_k_0")' in source
    assert '"omni_talker_model_id"' in source
    assert 'overrides["omni_talker_model_id"] = self._talker_model_id' in builder
    assert 'overrides["omni_talker_model_revision"]' in builder
    assert 'find_section(ctx.bundle, "talker_engine_plan")' not in source
