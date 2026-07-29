# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Granite native C++ logits-trace reconstruction contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.models.granite.e2e_plugins.contracts import E2ECase, RunContext
from tests.e2e.models.granite.e2e_plugins.runners import text_generation

TextGenerationCausalRunner = text_generation.TextGenerationCausalRunner


def _case(**metadata) -> E2ECase:
    return E2ECase(
        name="granite-native-trace",
        hf_id="ibm-granite/test",
        family="granite",
        runtime_strategy="granite_decoder_kv_cache",
        inputs={"max_new_tokens": 1},
        metadata=metadata,
    )


def test_cpp_runner_keeps_local_cli_and_jsonl_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        runtime_cli_requires_hf_python=True,
        contract_config={"use_chat_template": True, "enable_thinking": False},
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/opt/trtmc",
        runtime_python="/opt/runtime-python",
        ld_library_path="/opt/trt/lib",
    )
    observed: dict = {}

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text(
            json.dumps({"generated": "native output", "token_ids": ["7", 8]}) + "\n",
            encoding="utf-8",
        )
        stderr = (
            "[trtmc.timing] prefill_ms=10 decode_ms=20 total_ms=30\n"
            "[trtmc.load_timing] load_deserialize_ms=40\n"
        )
        return text_generation.subprocess.CompletedProcess(
            cmd, 0, stdout="stdout fallback\n", stderr=stderr
        )

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    runner = TextGenerationCausalRunner()
    generated, _, meta = runner._run_cpp_binary(
        ctx,
        "/models/granite.trtfb",
        "hello",
        2,
        case=case,
        inputs=case.inputs,
    )

    cmd = observed["cmd"]
    assert cmd[:3] == ["/opt/trtmc", "run", "/models/granite.trtfb"]
    assert cmd[cmd.index("--hf-python") + 1] == "/opt/runtime-python"
    assert "--chat-template" in cmd
    assert "--no-thinking" in cmd
    assert observed["kwargs"]["env"]["LD_LIBRARY_PATH"] == "/opt/trt/lib"
    assert generated == "native output"
    assert meta["token_ids"] == [7, 8]
    assert meta["trt_engine_s"] == pytest.approx(0.03)
    assert meta["trt_load_deserialize_s"] == pytest.approx(0.04)
    assert Path(meta["text_output_path"]).is_file()


def test_cpp_runner_fails_closed_on_trt_error_with_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/opt/trtmc",
    )
    stderr = "[TRT] ERROR: IExecutionContext::enqueueV3 failed\nfull diagnostic\n"

    def fake_run(cmd, **kwargs):
        return text_generation.subprocess.CompletedProcess(
            cmd, 0, stdout="invalid output\n", stderr=stderr
        )

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        text_generation,
        "save_full_stderr",
        lambda *args, **kwargs: ("trimmed", "/artifacts/cpp_binary.stderr.log"),
    )

    _, _, meta = TextGenerationCausalRunner()._run_cpp_binary(
        ctx,
        "/models/granite.trtfb",
        "hello",
        1,
        case=case,
    )

    assert meta["returncode"] == 0
    assert meta["effective_returncode"] == -1
    assert meta["runtime_error_detected"] == stderr.splitlines()[0]
    assert meta["error"] == "TensorRT runtime error detected in stderr"
    assert meta["stderr"] == stderr
    assert meta["stderr_log"] == "/artifacts/cpp_binary.stderr.log"


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        ("prefill", [[1.0, 2.0, 3.0]]),
        ("decode", [[4.0, 5.0, 6.0]]),
        ("full", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ],
)
def test_native_trace_reconstructs_full_vocab_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected: list[list[float]],
) -> None:
    case = _case()
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/unused/trtmc",
    )
    runner = TextGenerationCausalRunner()

    def fake_cpp_run(*args, **kwargs):
        runtime_sets = kwargs["runtime_sets"]
        trace_arg = next(
            value for value in runtime_sets if value.startswith("text_trace.step_trace_path=")
        )
        trace_path = Path(trace_arg.split("=", 1)[1])
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            {
                "phase": "prefill",
                "top_ids": [2, 0, 1],
                "top_logits": [3.0, 1.0, 2.0],
            },
            {
                "phase": "decode",
                "top_ids": [1, 2, 0],
                "top_logits": [5.0, 6.0, 4.0],
            },
        )
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return "generated", 0.1, {"returncode": 0, "token_ids": [2]}

    monkeypatch.setattr(runner, "_run_cpp_binary", fake_cpp_run)
    logits_path, _, meta = runner._run_native_trace_logits(
        ctx,
        "/unused/model.trtfb",
        "prompt",
        max_new_tokens=0 if phase == "prefill" else 1,
        case=case,
        phase=phase,
    )

    assert logits_path is not None
    np.testing.assert_array_equal(np.load(logits_path), np.asarray(expected, dtype=np.float32))
    assert meta["runner"] == "native_cpp_trace"
    assert meta["trace_rows"] == len(expected)
