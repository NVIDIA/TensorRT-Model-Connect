# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM production-prefill and diagnostic-trace runner contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.models.glm.e2e_plugins.contracts import (
    E2ECase,
    RunContext,
    StageSpec,
)
from tests.e2e.models.glm.e2e_plugins.runners import text_generation


def _case(*, ci_lane: str) -> E2ECase:
    return E2ECase(
        name="glm-native-runner",
        hf_id="zai-org/test",
        family="glm",
        runtime_strategy="glm_decoder_kv_cache",
        reference_family="causal_base_continuation",
        user_contract="continuation_parity",
        ci_lane=ci_lane,
        bundle="model.trtfb",
        inputs={"prompt": "hello", "max_new_tokens": 1},
    )


def _write_generation_output(command: list[str]) -> None:
    output_path = Path(command[command.index("-o") + 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"generated": " world", "token_ids": [42]}) + "\n",
        encoding="utf-8",
    )


def test_acceptance_generation_uses_production_prefill_without_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        _write_generation_output(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    case = _case(ci_lane="acceptance")
    output = text_generation.TextGenerationCausalRunner().run_stage(
        case,
        StageSpec(name="full_generation"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path),
            binary_path="/unused/trtmc",
            engine_dir=str(tmp_path),
        ),
    )

    assert len(commands) == 1
    assert commands[0][0] == "/unused/trtmc"
    assert not any("text_trace.step_trace_path=" in arg for arg in commands[0])
    assert output.text == " world"
    assert output.metadata["native_cpp_trace"] == {
        "skipped": "contract plugin active in acceptance lane"
    }


def test_runtime_error_in_stderr_fails_closed_when_process_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_error = "[TRT] ERROR: IExecutionContext::enqueueV3 failed"

    def fake_run(command, **_kwargs):
        _write_generation_output(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr=runtime_error)

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    case = _case(ci_lane="acceptance")
    output = text_generation.TextGenerationCausalRunner().run_stage(
        case,
        StageSpec(name="full_generation"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path),
            binary_path="/unused/trtmc",
            engine_dir=str(tmp_path),
        ),
    )

    assert output.data["cpp_returncode"] == -1
    assert output.data["cpp_runtime_error"] == runtime_error
    assert output.metadata["cpp"]["returncode"] == 0
    assert output.metadata["cpp"]["effective_returncode"] == -1
    assert output.metadata["cpp"]["error"] == "TensorRT runtime error detected in stderr"


def test_nightly_trace_is_separate_and_full_vocab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        _write_generation_output(command)
        trace_args = [arg for arg in command if arg.startswith("text_trace.step_trace_path=")]
        if trace_args:
            trace_path = Path(trace_args[0].split("=", 1)[1])
            trace_path.write_text(
                json.dumps(
                    {
                        "phase": "prefill",
                        "position": 0,
                        "logits": [1.0, 2.0, 3.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(text_generation.subprocess, "run", fake_run)
    case = _case(ci_lane="nightly_parity")
    output = text_generation.TextGenerationCausalRunner().run_stage(
        case,
        StageSpec(name="full_generation"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path),
            binary_path="/unused/trtmc",
            engine_dir=str(tmp_path),
        ),
    )

    assert len(commands) == 2
    assert not any("text_trace.step_trace_path=" in arg for arg in commands[0])
    assert any("text_trace.step_trace_path=" in arg for arg in commands[1])
    assert output.logits is not None
    np.testing.assert_array_equal(
        np.load(output.logits),
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
