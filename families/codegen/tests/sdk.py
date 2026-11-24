# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct C/C++ consumers for this family's single-device checkpoint cases."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess

from tools.e2e_evidence import record_evidence


def check_consumers(bundle: Path, runtime_root: Path, prompt: str, case: dict,
                    native: dict, model_dir: Path, tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    build = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    controls = {"max_new_tokens": case["max_new_tokens"]}
    for name in ("temperature", "top_k", "top_p", "min_p", "seed", "repetition_penalty",
                 "use_chat_template", "enable_thinking"):
        if name in case:
            controls[name] = case[name]
    arguments = [f"{name}={str(value).lower()}" for name, value in controls.items()]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    # Token input is an independent public input representation. Do not decode
    # and re-tokenize it in the family or require tokenizer implementations to
    # produce bitwise-identical input IDs across frameworks.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    ids = tokenizer.encode(prompt)
    token_file = tmp_path / "sdk-prefix.i32"
    token_file.write_bytes(struct.pack(f"<{len(ids)}i", *ids))

    for mode, source in (("text", prompt), ("tokens", str(token_file))):
        outputs = []
        for language in ("c", "cpp"):
            executable = build / f"test_codegen_sdk_{language}"
            assert executable.is_file(), f"missing family SDK consumer: {executable}"
            completed = subprocess.run(
                [str(executable), str(bundle), str(runtime_root), mode, source, *arguments],
                capture_output=True, text=True, check=True, env=environment, timeout=600,
            )
            output = json.loads(completed.stdout)
            record_evidence(f"sdk_{language}_{mode}", output)
            assert isinstance(output["text"], str)
            assert output["token_ids"] and len(output["token_ids"]) <= int(case["max_new_tokens"])
            assert all(type(value) is int for value in output["token_ids"])
            for name in ("setup_ms", "prefill_ms", "decode_ms"):
                assert math.isfinite(output[name]) and output[name] >= 0
            outputs.append(output)
        assert outputs[0]["token_ids"] == outputs[1]["token_ids"]
        assert outputs[0]["text"] == outputs[1]["text"]
        if mode == "text":
            assert outputs[0]["token_ids"] == native["token_ids"]
            assert outputs[0]["text"] == native["text"]

    for language in ("c", "cpp"):
        completed = subprocess.run(
            [str(build / f"test_codegen_sdk_{language}"), str(bundle), str(runtime_root),
             "text", prompt, "max_new_tokens=0"],
            capture_output=True, text=True, check=True, env=environment, timeout=600,
        )
        empty = json.loads(completed.stdout)
        record_evidence(f"sdk_{language}_zero_tokens", empty)
        assert empty["text"] == "" and empty["token_ids"] == []
