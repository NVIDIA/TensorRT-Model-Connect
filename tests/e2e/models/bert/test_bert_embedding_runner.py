# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT-owned tests for embedding runtime dispatch."""

from __future__ import annotations

import subprocess
import json
from pathlib import Path

from tests.e2e.models.bert.e2e_plugins.references import hf_transformers
from tests.e2e.models.bert.e2e_plugins.runners import encoder_only
from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.registry import activate_model_plugins, get_comparator, get_runner


def _make_case() -> E2ECase:
    return E2ECase(
        name="multilingual-e5-small",
        hf_id="intfloat/multilingual-e5-small",
        family="bert",
        runtime_strategy="bert_embedding",
        task_strategy="embedding",
        bundle="multilingual-e5-small.bundle",
        hf_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        inputs={"prompt": "query: example"},
    )


def _make_ctx(case: E2ECase, tmp_path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def test_embedding_runtime_uses_pooled_embed_entrypoint(monkeypatch, tmp_path) -> None:
    """Changing the runtime to BERT embedding must not return the CLS path."""
    case = _make_case()
    ctx = _make_ctx(case, tmp_path)
    captured: dict[str, object] = {}

    def _fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"embedding": [0.6, 0.8]}\n',
            stderr="",
        )

    monkeypatch.setattr(encoder_only.subprocess, "run", _fake_run)

    output = encoder_only.EncoderOnlyRunner().run_stage(
        case, StageSpec(name="full_inference"), ctx
    )

    assert captured["cmd"][1] == "embed"
    assert output.data == {"embedding": [0.6, 0.8]}


def test_bert_registers_embedding_runner_and_comparator() -> None:
    activate_model_plugins(Path(__file__).parent)
    runner = get_runner("embedding")
    comparator = get_comparator("embedding")

    assert runner is not None
    assert comparator is not None

    result = comparator.compare(
        StageOutput(stage_name="full_inference", data={"embedding": [0.6, 0.8]}),
        StageOutput(stage_name="full_inference", data={"embedding": [0.6, 0.8]}),
        ThresholdProfile(
            task_strategy="embedding",
            profile_name="e5",
            metrics={
                "cosine_similarity": 0.999,
                "l2_distance": 0.01,
                "trt_unit_norm_error": 0.001,
                "reference_unit_norm_error": 0.001,
            },
        ),
        StageSpec(name="full_inference"),
    )
    assert result.status == "passed"
    assert set(result.metrics) == {
        "cosine_similarity",
        "l2_distance",
        "trt_unit_norm_error",
        "reference_unit_norm_error",
    }


def test_embedding_reference_uses_pinned_manifest_revision(monkeypatch, tmp_path) -> None:
    case = _make_case()
    ctx = _make_ctx(case, tmp_path)
    captured: dict[str, object] = {}
    marker = object()

    def _capture_reference(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(hf_transformers, "run_reference_subprocess", _capture_reference)
    result = hf_transformers.HfTransformersReference()._run_embedding_ref(
        case, StageSpec(name="full_inference"), ctx
    )

    assert result is marker
    script = captured["command"][2]
    assert f"revision = {case.hf_revision!r}" in script
    assert script.count("revision=revision") == 2
    assert captured["metadata"] == {
        "hf_id": case.hf_id,
        "hf_revision": case.hf_revision,
    }


def test_mean_pool_normalize_manifests_use_embedding_runtime_and_task() -> None:
    manifests_dir = Path(__file__).parent / "manifests"
    checked = ["all-minilm-l6-v2.json", "multilingual-e5-small.json"]
    for filename in checked:
        path = manifests_dir / filename
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["runtime_strategy"] == "bert_embedding"
        assert manifest["task_strategy"] == "embedding"
