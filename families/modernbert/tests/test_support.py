# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from families.modernbert.support import describe
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata

TASKS = ("text_to_pooled_features", "text_to_embedding", "text_pair_to_relevance")


def test_support_exposes_semantic_tasks():
    support = describe(ModelMetadata(config={"model_type": "modernbert"}, model_index={}))
    assert support.tasks == TASKS
    assert support.default_task == TASKS[0]
    assert describe(ModelMetadata(config={"model_type": "bert"}, model_index={})) is None


class Writer:
    def __init__(self):
        self.sections = {}

    def set_header(self, **header):
        self.sections["header"] = header

    def add_bytes(self, name, value):
        self.sections[name] = value

    def add_json(self, name, value):
        self.sections[name] = value


@pytest.fixture
def builder(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    model = importlib.import_module("families.modernbert.model")
    config = dict(
        model_type="modernbert",
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(model._ModernBertModel, "load_weights", lambda *args: {})
    monkeypatch.setattr(model._ModernBertModel, "build_engine", lambda *args, **kwargs: b"plan")
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda path: {})
    return model, tmp_path


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("tp", [1, 4])
def test_build_semantic_bundle_contract(builder, task, tp):
    model, path = builder
    writer = Writer()
    model.build(
        BuildRequest(
            model_dir=path,
            output_path=path / "unused.bundle",
            family="modernbert",
            task=task,
            precision="fp32",
            max_sequence_length=16,
            tensor_parallel_size=tp,
        ),
        writer,
    )
    assert writer.sections["header"]["task"] == task
    runtime = writer.sections["runtime.json"]
    assert runtime["vocab_size"] == 100
    assert runtime["max_sequence_length"] == 16
    assert runtime["tensor_parallel_size"] == tp
    plans = {key for key in writer.sections if key.endswith(".plan")}
    assert plans == ({"engine.plan"} if tp == 1 else {f"engine.rank{i}.plan" for i in range(tp)})


@pytest.mark.parametrize("task", ["encoding", "embedding", "reranking", "text_continuation"])
def test_build_rejects_retired_and_unrelated_tasks(builder, task):
    model, path = builder
    with pytest.raises(ValueError, match="text_to_pooled_features"):
        model.build(
            BuildRequest(
                model_dir=path,
                output_path=path / "unused.bundle",
                family="modernbert",
                task=task,
                precision="fp32",
            ),
            Writer(),
        )
