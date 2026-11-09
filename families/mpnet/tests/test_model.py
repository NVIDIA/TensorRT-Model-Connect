# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the mpnet Task SDK migration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.mpnet import model  # noqa: E402
from tensorrt_model_connect import BuildRequest  # noqa: E402


def _position_checkpoint(tmp_path: Path, rows: int, pad_id: int = 1) -> Path:
    directory = _checkpoint(tmp_path)
    path = directory / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(max_position_embeddings=rows, pad_token_id=pad_id)
    path.write_text(json.dumps(config), encoding="utf-8")
    return directory


def _checkpoint(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mpnet",
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 512,
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
    monkeypatch.setattr(model._MPNetModel, "load_weights", lambda self, *a, **k: {})
    monkeypatch.setattr(model._MPNetModel, "build_engine", lambda self, *a, **k: b"plan")
    monkeypatch.setattr(
        model,
        "_tokenizer_runtime_contract",
        lambda model_dir: {
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [],
        },
    )


@pytest.mark.parametrize("task", ["text_to_pooled_features", "text_to_embedding"])
def test_build_accepts_every_semantic_task(tmp_path: Path, monkeypatch, task: str) -> None:
    _patch_build(monkeypatch)
    writer = _Writer()

    model.build(
        BuildRequest(
            model_dir=_checkpoint(tmp_path),
            output_path=tmp_path / "unused.bundle",
            family="mpnet",
            task=task,
            precision="fp32",
        ),
        writer,
    )

    assert writer.sections["header"]["task"] == task
    assert writer.sections["engine.plan"] == b"plan"
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 1
    assert writer.sections["runtime.json"]["vocab_size"] == 100


@pytest.mark.parametrize(
    "task", ["encoding", "embedding", "reranking", "text_pair_to_relevance"]
)
def test_build_rejects_the_retired_task_names(tmp_path: Path, monkeypatch, task: str) -> None:
    _patch_build(monkeypatch)

    with pytest.raises(ValueError, match="text_to_pooled_features"):
        model.build(
            BuildRequest(
                model_dir=_checkpoint(tmp_path),
                output_path=tmp_path / "unused.bundle",
                family="mpnet",
                task=task,
                precision="fp32",
            ),
            _Writer(),
        )


def test_build_rejects_an_unrelated_task(tmp_path: Path, monkeypatch) -> None:
    _patch_build(monkeypatch)

    with pytest.raises(ValueError, match="text_to_pooled_features"):
        model.build(
            BuildRequest(
                model_dir=_checkpoint(tmp_path),
                output_path=tmp_path / "unused.bundle",
                family="mpnet",
                task="classification",
                precision="fp32",
            ),
            _Writer(),
        )


@pytest.mark.parametrize(
    "rows,pad_id,length", [(514, 1, 513), (514, 1, 514), (16, 1, 15), (16, 0, 16)]
)
@pytest.mark.parametrize("precision,quantization", [("fp32", None), ("fp16", "fp8")])
def test_build_rejects_lengths_in_reserved_position_rows(
    tmp_path, monkeypatch, rows, pad_id, length, precision, quantization
):
    def unexpected_load(*args, **kwargs):
        pytest.fail("an over-capacity request must be refused before loading weights")

    monkeypatch.setattr(model._MPNetModel, "load_weights", unexpected_load)
    with pytest.raises(ValueError, match="position.*capacity"):
        model.build(
            BuildRequest(
                model_dir=_position_checkpoint(tmp_path, rows, pad_id),
                output_path=tmp_path / "unused.bundle",
                family="mpnet",
                task="text_to_embedding",
                precision=precision,
                quantization=quantization,
                max_sequence_length=length,
            ),
            _Writer(),
        )


@pytest.mark.parametrize("rows,pad_id,expected", [(514, 1, 512), (16, 1, 14), (16, 3, 12)])
@pytest.mark.parametrize("use_default", [False, True])
def test_build_accepts_usable_position_boundary_and_default(
    tmp_path, monkeypatch, rows, pad_id, expected, use_default
):
    _patch_build(monkeypatch)
    lengths = []

    def capture(self, config, weights, length, **kwargs):
        lengths.append(length)
        return b"plan"

    monkeypatch.setattr(model._MPNetModel, "build_engine", capture)
    model.build(
        BuildRequest(
            model_dir=_position_checkpoint(tmp_path, rows, pad_id),
            output_path=tmp_path / "unused.bundle",
            family="mpnet",
            task="text_to_embedding",
            precision="fp32",
            max_sequence_length=None if use_default else expected,
        ),
        _Writer(),
    )
    assert lengths == [expected]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_build_engine_checks_the_mapped_position_table_before_graph_work(monkeypatch, tp_size):
    from families.mpnet import tp_builder

    def unexpected_build(*args, **kwargs):
        pytest.fail("over-capacity positions must not reach TensorRT")

    monkeypatch.setattr(model, "build_encoder_engine", unexpected_build)
    monkeypatch.setattr(tp_builder, "build_tp_encoder_engine", unexpected_build)
    config = model.ModelConfig(hidden_size=32, num_attention_heads=2)
    weights = model.WeightDict(position_embedding=np.zeros((512, 32), np.float32))
    with pytest.raises(ValueError, match="position.*capacity"):
        model._MPNetModel().build_engine(
            config, weights, 513, parallel_config=model.ParallelConfig(tp_size=tp_size)
        )


@pytest.mark.parametrize("pad_id", [0, 1, 3])
def test_loader_reserves_pad_and_preceding_position_rows(tmp_path, pad_id):
    directory = _position_checkpoint(tmp_path, 16, pad_id)
    config = model.ModelConfig.from_dir(directory)
    config.num_hidden_layers = 0
    positions = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    save_file(
        {
            "embeddings.word_embeddings.weight": np.zeros((100, 32), np.float32),
            "embeddings.position_embeddings.weight": positions,
            "embeddings.LayerNorm.weight": np.ones(32, np.float32),
            "embeddings.LayerNorm.bias": np.zeros(32, np.float32),
        },
        directory / "model.safetensors",
    )
    weights = model._MPNetModel().load_weights(str(directory), config)
    np.testing.assert_array_equal(weights["position_embedding"], positions[pad_id + 1 :])


def test_build_forwards_fp8_context_at_usable_capacity(tmp_path, monkeypatch):
    from families.mpnet import quantization

    context = object()
    weights = model.WeightDict(position_embedding=np.zeros((512, 32), np.float32))
    monkeypatch.setattr(model._MPNetModel, "load_weights", lambda *args: weights)
    monkeypatch.setattr(model, "_open_safetensors", lambda *args: SimpleNamespace(tensor_map={}))
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda *args: {})
    monkeypatch.setattr(quantization, "refuse_quantized_checkpoint", lambda *args: None)
    monkeypatch.setattr(quantization, "calibrate_mpnet", lambda *args, **kwargs: context)
    seen = {}

    def capture(config, mapped_weights, **kwargs):
        seen.update(kwargs)
        assert mapped_weights is weights
        return b"fp8-plan"

    monkeypatch.setattr(model, "build_encoder_engine", capture)
    writer = _Writer()
    model.build(
        BuildRequest(
            model_dir=_position_checkpoint(tmp_path, 514),
            output_path=tmp_path / "unused.bundle",
            family="mpnet",
            task="text_to_embedding",
            precision="fp16",
            quantization="fp8",
            max_sequence_length=512,
        ),
        writer,
    )
    assert seen["max_seq_length"] == 512
    assert seen["precision"] == "fp16"
    assert seen["quant_ctx"] is context
    assert writer.sections["engine.plan"] == b"fp8-plan"
