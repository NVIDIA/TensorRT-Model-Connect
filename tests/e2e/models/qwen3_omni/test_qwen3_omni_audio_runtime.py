# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct

import pytest

from tensorrt_model_connect.families.qwen3_omni.audio_runtime import (
    TalkerRequest,
    _chatml,
    _read_request,
)
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.qwen3_omni.plugin import (
    Qwen3OmniPlugin,
    _talker_model_locator,
)


def _payload(prompt: str, assistant: str) -> bytes:
    prompt_bytes = prompt.encode("utf-8")
    assistant_bytes = assistant.encode("utf-8")
    return (
        struct.pack("<II", len(prompt_bytes), len(assistant_bytes)) + prompt_bytes + assistant_bytes
    )


def test_talker_request_preserves_prompt_and_trims_generated_stop_marker() -> None:
    request = _read_request(_payload("Say hello.", "Hello from Qwen-Omni!<|im_end|>ignored"))

    assert request == TalkerRequest(prompt="Say hello.", assistant_text="Hello from Qwen-Omni!")


def test_talker_request_rejects_empty_assistant_text() -> None:
    with pytest.raises(ValueError, match="no speakable assistant text"):
        _read_request(_payload("Say hello.", "<|im_end|>"))


def test_talker_request_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="expected"):
        _read_request(struct.pack("<II", 3, 4) + b"abc")


def test_talker_chatml_contains_model_roles_and_exact_text() -> None:
    rendered = _chatml(TalkerRequest(prompt="question", assistant_text="answer"))

    assert "<|im_start|>system\n" in rendered
    assert "<|im_start|>user\nquestion<|im_end|>" in rendered
    assert "<|im_start|>assistant\nanswer<|im_end|>" in rendered


def test_talker_model_locator_pins_hugging_face_snapshot(tmp_path) -> None:
    snapshot = tmp_path / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    assert _talker_model_locator(snapshot) == (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "abc123",
    )


def test_talker_model_locator_preserves_deliberate_local_directory(tmp_path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    assert _talker_model_locator(model_dir) == (str(model_dir.resolve()), "")


def test_bundle_config_persists_portable_talker_locator() -> None:
    plugin = Qwen3OmniPlugin()
    plugin._talker_model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    plugin._talker_model_revision = "abc123"

    overrides = plugin.get_bundle_config_overrides(ModelConfig.create_tiny("qwen3_omni"))

    assert overrides["omni_talker_model_id"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert overrides["omni_talker_model_revision"] == "abc123"
