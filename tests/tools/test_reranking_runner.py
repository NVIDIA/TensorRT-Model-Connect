"""Tests for the E2E reranking runner."""

from __future__ import annotations

import subprocess

import pytest

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.runners.reranking import (
    RerankingRunner,
    _documents_from_inputs,
)


def _case(inputs: dict) -> E2ECase:
    return E2ECase(
        name="rerank-test",
        hf_id="org/reranker",
        family="example_reranker",
        runtime_strategy="reranking",
        bundle="rerank.trtfb",
        inputs=inputs,
    )


def test_documents_from_inputs_prefers_document_list() -> None:
    docs = _documents_from_inputs({
        "document": "legacy",
        "documents": ["first", "second"],
    })

    assert docs == ["first", "second"]


def test_documents_from_inputs_rejects_non_list_documents() -> None:
    with pytest.raises(TypeError, match="documents"):
        _documents_from_inputs({"documents": "not-a-list"})


def test_runner_scores_each_manifest_document(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):  # noqa: ANN001
        calls.append(cmd)
        idx = len(calls)
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"Relevance score: {idx * 0.25}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    case = _case({
        "prompt": "Which planet is known as the Red Planet?",
        "documents": ["Venus text", "Mars text", "Jupiter text"],
    })
    ctx = RunContext(
        case=case,
        binary_path="/bin/trtmc",
        engine_dir=str(tmp_path),
    )

    output = RerankingRunner().run_stage(case, StageSpec("full_inference"), ctx)

    assert output.data["scores"] == [0.25, 0.5, 0.75]
    assert output.data["documents"] == ["Venus text", "Mars text", "Jupiter text"]
    assert [cmd[6] for cmd in calls] == ["Venus text", "Mars text", "Jupiter text"]
    assert output.metadata["document_count"] == 3
    assert output.metadata["document_1"]["stdout"] == "Relevance score: 0.5"
