# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from families.qwen.tests.sdk import check_consumers


@pytest.mark.parametrize("chat_template", [False, True])
def test_sdk_token_inputs_do_not_apply_text_preprocessing(monkeypatch, tmp_path, chat_template):
    calls = []
    native = {"text": "native", "token_ids": [11]}

    class Tokenizer:
        def encode(self, text):
            assert text == "prompt"
            return [7, 8]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())),
    )
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path))
    for language in ("c", "cpp"):
        (tmp_path / f"test_qwen_sdk_{language}").touch()

    def run(command, **kwargs):
        controls = dict(value.split("=", 1) for value in command[5:])
        calls.append((command[3], controls))
        if controls["max_new_tokens"] == "0":
            output = {"text": "", "token_ids": []}
        else:
            assert controls["temperature"] == "0.0"
            assert controls["top_k"] == "1"
            if command[3] == "tokens":
                if controls.get("use_chat_template") == "true":
                    raise subprocess.CalledProcessError(
                        1, command, stderr="Qwen token-ID input cannot apply a chat template"
                    )
                assert "enable_thinking" not in controls
                assert (tmp_path / "sdk-prefix.i32").read_bytes() == struct.pack("<2i", 7, 8)
                output = {"text": "tokens", "token_ids": [12]}
            else:
                assert controls["use_chat_template"] == str(chat_template).lower()
                assert controls["enable_thinking"] == "false"
                output = dict(native)
            output.update(setup_ms=0.0, prefill_ms=1.0, decode_ms=2.0)
        return SimpleNamespace(stdout=json.dumps(output))

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("families.qwen.tests.sdk.record_evidence", lambda *a: None)
    check_consumers(
        tmp_path / "model.bundle",
        tmp_path,
        "prompt",
        {
            "max_new_tokens": 2,
            "temperature": 0.0,
            "top_k": 1,
            "use_chat_template": chat_template,
            "enable_thinking": False,
        },
        native,
        tmp_path,
        tmp_path,
    )
    assert [mode for mode, _ in calls] == ["text", "text", "tokens", "tokens", "text", "text"]
