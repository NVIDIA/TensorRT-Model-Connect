# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mistral native C++ runner contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.models.mistral.e2e_plugins.contracts import E2ECase, RunContext, StageSpec
from tests.e2e.models.mistral.e2e_plugins.runners import text_generation


def _case() -> E2ECase:
    return E2ECase(
        name="mistral-native-runner",
        hf_id="mistralai/test",
        family="mistral",
        runtime_strategy="mistral_decoder_kv_cache",
        reference_family="causal_base_continuation",
        user_contract="continuation_parity",
        ci_lane="acceptance",
        bundle="model.trtfb",
        inputs={"prompt": "hello", "max_new_tokens": 1},
    )


def _context(case: E2ECase, tmp_path: Path) -> RunContext:
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/unused/trtmc",
        engine_dir=str(tmp_path),
    )


def _write_generation_output(command: list[str]) -> None:
    output_path = Path(command[command.index("-o") + 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"generated": " world", "token_ids": [42]}) + "\n",
        encoding="utf-8",
    )


def test_local_cpp_run_preserves_timing_and_runtime_error_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    runtime_error = "[TRT] ERROR: IExecutionContext::enqueueV3 failed"
    stderr = (
        "[trtmc.timing] prefill_ms=2 decode_ms=3 total_ms=5\n"
        "[trtmc.load_timing] load_deserialize_ms=7\n"
        f"{runtime_error}\n"
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        _write_generation_output(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr=stderr)

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    case = _case()
    output = text_generation.TextGenerationCausalRunner().run_stage(
        case,
        StageSpec(name="full_generation"),
        _context(case, tmp_path),
    )
    metadata = output.metadata["cpp"]

    assert len(commands) == 1
    assert commands[0][:3] == ["/unused/trtmc", "run", str(tmp_path / case.bundle)]
    assert output.text == " world"
    assert output.data["cpp_returncode"] == -1
    assert output.data["cpp_runtime_error"] == runtime_error
    assert output.metadata["native_trace"] == {
        "skipped": "contract plugin active in acceptance lane"
    }
    assert metadata["token_ids"] == [42]
    assert metadata["returncode"] == 0
    assert metadata["effective_returncode"] == -1
    assert metadata["runtime_error_detected"] == runtime_error
    assert metadata["trt_engine_prefill_s"] == pytest.approx(0.002)
    assert metadata["trt_engine_decode_s"] == pytest.approx(0.003)
    assert metadata["trt_engine_s"] == pytest.approx(0.005)
    assert metadata["trt_load_deserialize_s"] == pytest.approx(0.007)


def test_native_trace_still_collects_full_vocab_logits_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        _write_generation_output(command)
        trace_arg = next(arg for arg in command if arg.startswith("text_trace.step_trace_path="))
        trace_path = Path(trace_arg.split("=", 1)[1])
        rows = (
            {"phase": "prefill", "top_ids": [2, 0, 1], "top_logits": [30, 10, 20]},
            {"phase": "decode", "top_ids": [1, 2, 0], "top_logits": [50, 60, 40]},
        )
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    case = _case()
    logits_path, _elapsed, metadata = (
        text_generation.TextGenerationCausalRunner()._run_native_trace_logits(
            _context(case, tmp_path),
            str(tmp_path / case.bundle),
            case.inputs["prompt"],
            case.inputs["max_new_tokens"],
            case,
        )
    )

    assert len(commands) == 1
    assert commands[0][0] == "/unused/trtmc"
    assert logits_path is not None
    np.testing.assert_array_equal(
        np.load(logits_path),
        np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.float32),
    )
    assert metadata["phase_counts"] == {"prefill": 1, "decode": 1}
    assert metadata["generated_text"] == " world"
    assert metadata["generated_token_count"] == 1
