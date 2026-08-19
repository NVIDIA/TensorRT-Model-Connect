# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for E2E orchestrator TRT runtime path validation guard.

Trace: ARCH-E2E-001, UD-E2E-RUNTIME-GUARD
Intent: Validate that the runtime path guard accepts new-runtime markers and rejects legacy runtime markers
Preconditions: StageOutput metadata contains stderr with runtime backend markers
Postconditions: New runtime markers pass validation; legacy runtime markers trigger rejection
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput
from tests.e2e_harness.orchestrator import _validate_trt_runtime_path
from tests.e2e_harness.runtime_strategy_metadata import (
    runtime_strategy_performance_mode,
    runtime_strategy_requires_new_runtime_guard,
)


def _make_case(strategy: str) -> E2ECase:
    return E2ECase(
        name="guard-case",
        hf_id="hf/test-model",
        family="unit",
        runtime_strategy=strategy,
        task_strategy="text_generation_causal",
        bundle="guard-case.bundle",
        stages=[],
    )


def _make_ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        case=_make_case("decoder_kv_cache"),
        artifacts_dir=str(tmp_path),
        binary_path="/tmp/build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir=str(tmp_path),
    )


def _write_runtime_owners(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    for owner, strategy, task in (
        ("decoder", "unit_new_runtime", "text_generation_causal"),
        ("media", "unit_diffusion_runtime", "diffusion_media_generation"),
        ("speech", "unit_enc_dec_runtime", "speech_to_text"),
        ("audio", "unit_multistage_runtime", "text_to_audio"),
    ):
        root = models / owner
        manifest = root / "tests" / "manifests" / "case.json"
        manifest.parent.mkdir(parents=True)
        (root / "MODEL.toml").write_text(
            f'id = "{owner}"\n'
            f'runtime_strategies = ["{strategy}"]\n'
            'test_manifests = ["tests/manifests/case.json"]\n',
            encoding="utf-8",
        )
        manifest.write_text(
            "{\n"
            f'  "name": "{owner}",\n'
            f'  "hf_id": "unit/{owner}",\n'
            f'  "family": "{owner}",\n'
            f'  "runtime_strategy": "{strategy}",\n'
            f'  "task_strategy": "{task}"\n'
            "}\n",
            encoding="utf-8",
        )
    return models


def test_runtime_guard_accepts_new_runtime_marker_in_metadata(tmp_path: Path) -> None:
    case = _make_case("decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.bundle"],
            "stderr": "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=decoder_kv_cache)",
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_strategy_ownership_is_declarative(tmp_path: Path) -> None:
    models = _write_runtime_owners(tmp_path)
    assert runtime_strategy_requires_new_runtime_guard("unit_new_runtime", models)
    assert not runtime_strategy_requires_new_runtime_guard("future_unknown_strategy", models)


def test_runtime_strategy_performance_mode_comes_from_metadata(tmp_path: Path) -> None:
    models = _write_runtime_owners(tmp_path)
    assert runtime_strategy_performance_mode("unit_new_runtime", models) == "decode"
    assert runtime_strategy_performance_mode("unit_diffusion_runtime", models) == "diffusion"
    assert runtime_strategy_performance_mode("unit_enc_dec_runtime", models) == "enc_dec"
    assert runtime_strategy_performance_mode("unit_multistage_runtime", models) == "multi_stage"
    with pytest.raises(ValueError, match="unknown runtime_strategy"):
        runtime_strategy_performance_mode("future_unknown_strategy", models)


def test_runtime_guard_rejects_legacy_runtime_marker(tmp_path: Path) -> None:
    case = _make_case("qwen_decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.bundle"],
            "stderr": "[trtmc] Runtime path: compatibility factory mode",
        },
    )

    message = _validate_trt_runtime_path(case, ctx, output)
    assert message is not None
    assert "legacy compatibility factory mode" in message


def test_runtime_guard_rejects_missing_new_runtime_confirmation(tmp_path: Path) -> None:
    case = _make_case("qwen_decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.bundle"],
            "stderr": "tokens: 1 2 3 4",
        },
    )

    message = _validate_trt_runtime_path(case, ctx, output)
    assert message is not None
    assert "did not confirm the new runtime path" in message


def test_runtime_guard_reads_stderr_log_from_stage_data(tmp_path: Path) -> None:
    case = _make_case("speech_to_text")
    ctx = _make_ctx(tmp_path)
    stderr_log = tmp_path / "speech_to_text_stderr.log"
    stderr_log.write_text(
        "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=speech_to_text)\n",
        encoding="utf-8",
    )
    output = StageOutput(
        stage_name="full_inference",
        data={
            "stderr_log": str(stderr_log),
            "stderr": "truncated tail",
        },
        metadata={
            "command": [ctx.binary_path, "transcribe", "/tmp/model.bundle"],
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_ignores_legacy_marker_from_unrelated_subprocess(tmp_path: Path) -> None:
    case = _make_case("decoder_kv_cache")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "cpp": {
                "command": [ctx.binary_path, "run", "/tmp/model.bundle"],
                "stderr": "[trtmc] Runtime ready (backend=trt_new_runtime_default, strategy=decoder_kv_cache)",
            },
            "debug_runner": {
                "command": [ctx.hf_python, "-m", "tensorrt_model_connect.debug_runner"],
                "stderr": "[trtmc] Runtime path: compatibility factory mode",
            },
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_skips_unknown_strategies(tmp_path: Path) -> None:
    case = _make_case("future_unknown_strategy")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_generation",
        metadata={
            "command": [ctx.binary_path, "run", "/tmp/model.bundle"],
            "stderr": "[trtmc] Runtime path: compatibility factory mode",
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None


def test_runtime_guard_ignores_nonzero_cli_parse_errors_without_runtime_markers(tmp_path: Path) -> None:
    case = _make_case("unit_prompted_segmentation")
    ctx = _make_ctx(tmp_path)
    output = StageOutput(
        stage_name="full_inference",
        metadata={
            "command": [ctx.binary_path, "segment", "/tmp/model.bundle"],
            "returncode": 1,
            "stderr": "Error: Unknown flag: --point",
        },
    )

    assert _validate_trt_runtime_path(case, ctx, output) is None
