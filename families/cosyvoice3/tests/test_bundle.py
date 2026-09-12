# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle inspection and opt-in native tokenizer comparison (no GPU required)."""
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC


def read_bundle_section(path, name):
    data = Path(path).read_bytes()
    assert data.startswith(BUNDLE_MAGIC)
    header_size = struct.unpack_from("<Q", data, len(BUNDLE_MAGIC))[0]
    payload_start = len(BUNDLE_MAGIC) + 8 + header_size
    header = json.loads(data[len(BUNDLE_MAGIC) + 8:payload_start])
    section = header["sections"][name]
    start = payload_start + section["offset"]
    return data[start:start + section["length"]]


def test_fixed_voice_package_command_is_not_available(capsys):
    from families.cosyvoice3.__main__ import main

    with pytest.raises(SystemExit) as error:
        main(["package"])
    assert error.value.code == 2
    assert "invalid choice: 'package'" in capsys.readouterr().err


def test_native_tokenizer_matches_python(tmp_path):
    binary, model = os.environ.get("COSYVOICE3_CPP_TEST"), os.environ.get("COSYVOICE3_MODEL_DIR")
    if not binary or not model:
        pytest.skip("Set COSYVOICE3_CPP_TEST and COSYVOICE3_MODEL_DIR for native BPE comparison")
    from families.cosyvoice3.tts import encode_request, text_tokenizer

    cases = []
    for text in ("Hello world.", "你好，世界。", "Hello 世界，2026!", "It's a nice day.\nGood morning."):
        for transcript in ("", "This is my voice."):
            packed, _ = encode_request(Path(model), text, prompt_text=transcript, prompt_tokens=[5, 6])
            cases.append(dict(text=text, transcript=transcript, speech=[5, 6], packed=packed))
    path = tmp_path / "tokenizer-cases.json"
    path.write_text(json.dumps(dict(tokenizer=text_tokenizer(model).backend_tokenizer.to_str(), cases=cases)), encoding="utf-8")
    subprocess.run([binary, str(path)], check=True, timeout=60)
