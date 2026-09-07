# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.models.k2_horizon.e2e_plugins.comparators.text import TextComparator
from tests.e2e.models.k2_horizon.e2e_plugins.contract import (
    K2HorizonGreedyContinuationPlugin,
)
from tests.e2e.models.k2_horizon.e2e_plugins.runners import (
    text_generation as text_generation_runner,
)
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest


def test_manifest_pins_the_independent_native_family() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "k2-horizon-7b.json"
    case = load_manifest(manifest_path)

    assert case.hf_id == "IFM/K2-Horizon-7B"
    assert case.hf_revision == "586b03f0fd1fbbf2f13eeafc33749e95ae34dd10"
    assert case.family == "k2_horizon"
    assert case.runtime_strategy == "k2_horizon_decoder_kv_cache"
    assert case.task_strategy == "text_generation_causal"
    assert case.user_contract == "continuation_parity"
    assert case.reference_family == "k2_horizon_greedy_continuation"
    assert case.metadata["precision"] == "bf16"
    assert case.inputs["max_cache_length"] == 256
    assert case.metadata["trust_remote_code"] is True
    assert case.execution_profiles["reference"] == "k2_horizon_reference"
    assert case.metadata["contract_config"] == {"use_chat_template": False}
    assert case.metadata["expected_continuation_token_ids"] == [11511, 15, 589, 7169]


def test_e2e_code_uses_only_k2_horizon_owners() -> None:
    root = Path(__file__).resolve().parent
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((root / "e2e_plugins").rglob("*.py"))
    )

    assert "families.qwen" not in source
    assert "qwen_decoder_kv_cache" not in source
    assert "families.k2_horizon.debug_runner" in source


def test_comparator_keeps_logit_parity_as_a_required_gate() -> None:
    reference = np.array([[0.0, 3.0, -1.0]], dtype=np.float32)
    divergent = np.array([[3.0, 0.0, -1.0]], dtype=np.float32)
    result = TextComparator().compare(
        trt=StageOutput(
            stage_name="full_generation",
            data={"cpp_returncode": 0},
            text=" Paris",
            logits=divergent,
        ),
        ref=StageOutput(
            stage_name="full_generation",
            text=" Paris",
            logits=reference,
        ),
        threshold=ThresholdProfile(
            task_strategy="text_generation_causal",
            profile_name="k2-horizon-test",
            metrics={
                "logit_cosine_p5": 0.99,
                "logit_rel_l2_p95": 0.05,
                "stable_top1_match_rate": 0.9,
                "unstable_topk_hit_rate": 0.8,
                "token_agreement_rate": 0.8,
                "normalized_text_edit_distance": 0.2,
            },
        ),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == "failed"
    assert not result.metrics["token_agreement_rate"].passed
    assert "logit_cosine_p5" in result.metrics


@pytest.mark.parametrize(
    ("trt_logits", "ref_logits", "cpp_returncode", "message"),
    [
        (None, np.ones((1, 3), dtype=np.float32), 0, "requires both"),
        (
            np.ones((1, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            0,
            "identical shapes",
        ),
        (
            np.array([[np.nan, 1.0, 0.0]], dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            0,
            "finite values",
        ),
        (
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            None,
            "returncode",
        ),
    ],
)
def test_comparator_fails_closed_when_required_evidence_is_invalid(
    trt_logits: np.ndarray | None,
    ref_logits: np.ndarray | None,
    cpp_returncode: int | None,
    message: str,
) -> None:
    result = TextComparator().compare(
        trt=StageOutput(
            stage_name="full_generation",
            data={"cpp_returncode": cpp_returncode},
            text=" Paris",
            logits=trt_logits,
        ),
        ref=StageOutput(
            stage_name="full_generation",
            text=" Paris",
            logits=ref_logits,
        ),
        threshold=ThresholdProfile(
            task_strategy="text_generation_causal",
            profile_name="k2-horizon-test",
            metrics={},
        ),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == "error"
    assert message in result.message


def _combined_contract_result(*, trt_tokens: list[int], trt_logits: np.ndarray):
    case = load_manifest(Path(__file__).with_name("manifests") / "k2-horizon-7b.json")
    reference_tokens = [11511, 15, 589, 7169]
    reference_logits = np.array(
        [
            [0.0, 3.0, -1.0],
            [0.0, 4.0, -2.0],
            [0.0, 5.0, -3.0],
            [0.0, 6.0, -4.0],
        ],
        dtype=np.float32,
    )
    return K2HorizonGreedyContinuationPlugin().verify(
        trt_output=StageOutput(
            stage_name="full_generation",
            data={"cpp_returncode": 0, "token_ids": trt_tokens},
            text=" Paris. The capital",
            logits=trt_logits,
        ),
        ref_output=StageOutput(
            stage_name="full_generation",
            data={
                "token_ids": reference_tokens,
                "model_revision": case.hf_revision,
            },
            text=" Paris. The capital",
            logits=reference_logits,
        ),
        case=case,
        threshold=ThresholdProfile(
            task_strategy="text_generation_causal",
            profile_name="k2-horizon-test",
            metrics={
                "logit_cosine_p5": 0.99,
                "logit_rel_l2_p95": 0.05,
                "stable_top1_match_rate": 0.9,
                "unstable_topk_hit_rate": 0.8,
                "token_agreement_rate": 0.8,
                "normalized_text_edit_distance": 0.2,
            },
        ),
    )


def test_combined_contract_requires_numeric_exact_and_golden_evidence() -> None:
    logits = np.array(
        [
            [0.0, 3.0, -1.0],
            [0.0, 4.0, -2.0],
            [0.0, 5.0, -3.0],
            [0.0, 6.0, -4.0],
        ],
        dtype=np.float32,
    )
    result = _combined_contract_result(
        trt_tokens=[11511, 15, 589, 7169],
        trt_logits=logits,
    )

    assert result.status == "passed"
    assert result.metrics["logit_cosine_p5"].passed
    assert result.metrics["exact_token_parity"].passed
    assert result.metrics["expected_golden_continuation"].passed


def test_combined_contract_rejects_cpp_token_drift_even_when_logits_match() -> None:
    logits = np.array(
        [
            [0.0, 3.0, -1.0],
            [0.0, 4.0, -2.0],
            [0.0, 5.0, -3.0],
            [0.0, 6.0, -4.0],
        ],
        dtype=np.float32,
    )
    result = _combined_contract_result(
        trt_tokens=[11511, 15, 589, 0],
        trt_logits=logits,
    )

    assert result.status == "failed"
    assert not result.metrics["exact_token_parity"].passed


def test_combined_contract_rejects_logit_drift_even_when_tokens_match() -> None:
    divergent = np.array(
        [
            [3.0, 0.0, -1.0],
            [4.0, 0.0, -2.0],
            [5.0, 0.0, -3.0],
            [6.0, 0.0, -4.0],
        ],
        dtype=np.float32,
    )
    result = _combined_contract_result(
        trt_tokens=[11511, 15, 589, 7169],
        trt_logits=divergent,
    )

    assert result.status == "failed"
    assert not result.metrics["token_agreement_rate"].passed
    assert "logit_cosine_p5" in result.metrics


def test_debug_timeout_preserves_diagnostics(tmp_path, monkeypatch) -> None:
    case = load_manifest(Path(__file__).with_name("manifests") / "k2-horizon-7b.json")
    context = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        runtime_python_path=lambda: "python3",
        ld_library_path="",
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python3"],
            timeout=900,
            stderr=b"partial debug stderr",
        )

    monkeypatch.setattr(text_generation_runner.subprocess, "run", timeout)
    logits, _elapsed, metadata = (
        text_generation_runner.TextGenerationCausalRunner()._run_debug_logits(
            context,
            case,
            "bundle.bundle",
            "The capital of France is",
            4,
        )
    )

    assert logits is None
    assert metadata["returncode"] == -1
    assert metadata["error"] == "timeout"
    assert "partial debug stderr" in metadata["stderr_truncated"]
    assert Path(metadata["stderr_log"]).read_text(encoding="utf-8") == "partial debug stderr"


def test_cpp_timeout_preserves_diagnostics(tmp_path, monkeypatch) -> None:
    case = load_manifest(Path(__file__).with_name("manifests") / "k2-horizon-7b.json")
    context = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        binary_path="trtmc",
        runtime_cli_hf_python=lambda: None,
        ld_library_path="",
    )

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["trtmc"],
            timeout=900,
            stderr=b"partial C++ stderr",
        )

    monkeypatch.setattr(text_generation_runner.subprocess, "run", timeout)
    text, _elapsed, metadata = text_generation_runner.TextGenerationCausalRunner()._run_cpp_binary(
        context,
        case,
        "bundle.bundle",
        "The capital of France is",
        4,
    )

    assert text == ""
    assert metadata["returncode"] == -1
    assert metadata["error"] == "timeout"
    assert "partial C++ stderr" in metadata["stderr_truncated"]
    assert Path(metadata["stderr_log"]).read_text(encoding="utf-8") == "partial C++ stderr"
