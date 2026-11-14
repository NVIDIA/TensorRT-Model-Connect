# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the eagle_vlm Task SDK migration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.eagle_vlm import model  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402


def _checkpoint(tmp_path: Path, reranker: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama_nemotron_vl_rerank" if reranker else "llama_nemotron_vl",
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 512,
                "pooling": "last",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


class _Writer:
    def __init__(self) -> None:
        self.sections: dict[str, object] = {}

    def set_header(self, **header) -> None:
        self.sections["header"] = header

    def add_bytes(self, name: str, value: bytes) -> None:
        self.sections[name] = value

    def add_json(self, name: str, value: object) -> None:
        self.sections[name] = value


def _patch_build(monkeypatch) -> None:
    monkeypatch.setattr(model._EagleModel, "load_weights", lambda self, *a, **k: {})
    monkeypatch.setattr(model._EagleModel, "build_engine", lambda self, *a, **k: b"plan")
    monkeypatch.setattr(
        model,
        "_tokenizer_runtime_contract",
        lambda model_dir: {
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [],
        },
    )


def _build(tmp_path: Path, task: str, reranker: bool | None = None) -> _Writer:
    if reranker is None:
        reranker = task == "text_query_documents_to_relevance"
    writer = _Writer()
    model.build(
        BuildRequest(
            model_dir=_checkpoint(tmp_path, reranker),
            output_path=tmp_path / "unused.bundle",
            family="eagle_vlm",
            task=task,
            precision="fp32",
        ),
        writer,
    )
    return writer


@pytest.mark.parametrize(
    "task",
    ["text_to_pooled_features", "text_to_embedding", "text_query_documents_to_relevance"],
)
def test_build_accepts_every_semantic_task(tmp_path: Path, monkeypatch, task: str) -> None:
    _patch_build(monkeypatch)

    writer = _build(tmp_path, task)

    assert writer.sections["header"]["task"] == task
    assert writer.sections["engine.plan"] == b"plan"
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 1
    assert writer.sections["runtime.json"]["vocab_size"] == 100


def test_only_a_relevance_bundle_records_the_score_pooling(tmp_path: Path, monkeypatch) -> None:
    _patch_build(monkeypatch)

    relevance = _build(tmp_path / "relevance", "text_query_documents_to_relevance")
    embedding = _build(tmp_path / "embedding", "text_to_embedding")

    assert relevance.sections["runtime.json"]["pooling"] == "last"
    assert "pooling" not in embedding.sections["runtime.json"]


@pytest.mark.parametrize("task", ["encoding", "embedding", "reranking", "text_pair_to_relevance"])
def test_build_rejects_the_retired_and_secondary_task_names(
    tmp_path: Path, monkeypatch, task: str
) -> None:
    _patch_build(monkeypatch)

    with pytest.raises(ValueError, match="text_query_documents_to_relevance"):
        _build(tmp_path, task)


def test_build_rejects_an_unrelated_task(tmp_path: Path, monkeypatch) -> None:
    _patch_build(monkeypatch)

    with pytest.raises(ValueError, match="text_query_documents_to_relevance"):
        _build(tmp_path, "classification")


@pytest.mark.parametrize(
    ("task", "reranker"),
    [
        ("text_to_embedding", True),
        ("text_to_pooled_features", True),
        ("text_query_documents_to_relevance", False),
    ],
)
def test_build_rejects_a_task_that_does_not_match_the_checkpoint(
    tmp_path: Path, monkeypatch, task: str, reranker: bool
) -> None:
    _patch_build(monkeypatch)

    with pytest.raises(ValueError, match="does not match this checkpoint"):
        _build(tmp_path, task, reranker)
