# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the LayoutLM builder: refused requests and checkpoint shapes.

No engine is built here; the GPU E2E builds and runs the real checkpoint.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from .. import checkpoint, model  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
_TINY = {"hidden_size": 8, "num_layers": 2, "num_heads": 2, "intermediate_size": 16,
         "vocab_size": 700, "max_position_embeddings": 20, "max_2d_position_embeddings": 1024,
         "type_vocab_size": 1}


def _request(**overrides):
    values = dict(model_dir=Path("/nonexistent"), output_path=Path("/nonexistent/out.bundle"),
                  family="layoutlm", task="images_text_to_text", precision="fp32", backend="trt",
                  max_sequence_length=None, image_height=None, image_width=None,
                  video_num_frames=None, max_batch_size=1, tensor_parallel_size=1,
                  context_parallel_size=1, quantization=None, fp32_layers=(),
                  dynamic_kv_cache=False, verbose=False, graph_transform=None, mtp_seq_len=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class _NoWriter:
    def __getattr__(self, name):
        raise AssertionError(f"a refused request must not write the bundle ({name})")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"task": "text_continuation"}, "task="),
        ({"backend": "trt_rtx"}, "backend=trt"),
        ({"quantization": "fp8"}, "quantization"),
        ({"max_batch_size": 2}, "max_batch_size"),
        ({"tensor_parallel_size": 2}, "one GPU"),
        ({"context_parallel_size": 2}, "one GPU"),
        ({"dynamic_kv_cache": True}, "dynamic_kv_cache"),
        ({"image_height": 224}, "image dimensions"),
        ({"video_num_frames": 4}, "video"),
        ({"fp32_layers": (0,)}, "mixed-precision"),
        ({"mtp_seq_len": 2}, "MTP"),
        ({"precision": "bf16"}, "fp32"),
    ],
)
def test_unsupported_requests_are_refused_before_reading_the_checkpoint(override, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model.build(_request(**override), _NoWriter())


def _checkpoint(directory: Path, config=None, tokenizer_settings=None, tensors=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = {"model_type": "layoutlm", "architectures": ["LayoutLMForQuestionAnswering"],
           "hidden_act": "gelu", "position_embedding_type": "absolute",
           "hidden_size": _TINY["hidden_size"], "num_hidden_layers": _TINY["num_layers"],
           "num_attention_heads": _TINY["num_heads"],
           "intermediate_size": _TINY["intermediate_size"], "vocab_size": _TINY["vocab_size"],
           "max_position_embeddings": _TINY["max_position_embeddings"],
           "max_2d_position_embeddings": 1024, "type_vocab_size": 1, "layer_norm_eps": 1e-5}
    raw.update(config or {})
    (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    shutil.copy(DATA / "tiny_tokenizer.json", directory / "tokenizer.json")
    settings = {"add_prefix_space": True, "model_max_length": 16}
    settings.update(tokenizer_settings or {})
    (directory / "tokenizer_config.json").write_text(json.dumps(settings), encoding="utf-8")
    if tensors is None:
        tensors = _tensors()
    safetensors_numpy.save_file(tensors, str(directory / "model.safetensors"))
    return directory


def _tensors() -> dict:
    rng = np.random.default_rng(0)
    hidden, inner = _TINY["hidden_size"], _TINY["intermediate_size"]

    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    prefix = "layoutlm.embeddings."
    values = {prefix + "word_embeddings.weight": tensor(_TINY["vocab_size"], hidden),
              prefix + "position_embeddings.weight": tensor(_TINY["max_position_embeddings"], hidden),
              prefix + "token_type_embeddings.weight": tensor(1, hidden),
              prefix + "LayerNorm.weight": tensor(hidden), prefix + "LayerNorm.bias": tensor(hidden),
              "layoutlm.embeddings.position_ids": np.arange(20, dtype=np.int64)[None],
              "layoutlm.pooler.dense.weight": tensor(hidden, hidden),
              "layoutlm.pooler.dense.bias": tensor(hidden),
              "qa_outputs.weight": tensor(2, hidden), "qa_outputs.bias": tensor(2)}
    for table in ("x", "y", "h", "w"):
        values[f"{prefix}{table}_position_embeddings.weight"] = tensor(1024, hidden)
    for layer in range(_TINY["num_layers"]):
        base = f"layoutlm.encoder.layer.{layer}."
        for name in ("attention.self.query", "attention.self.key", "attention.self.value",
                     "attention.output.dense"):
            values[base + name + ".weight"] = tensor(hidden, hidden)
            values[base + name + ".bias"] = tensor(hidden)
        values[base + "intermediate.dense.weight"] = tensor(inner, hidden)
        values[base + "intermediate.dense.bias"] = tensor(inner)
        values[base + "output.dense.weight"] = tensor(hidden, inner)
        values[base + "output.dense.bias"] = tensor(hidden)
        for norm in ("attention.output.LayerNorm", "output.LayerNorm"):
            values[base + norm + ".weight"] = tensor(hidden)
            values[base + norm + ".bias"] = tensor(hidden)
    return values


def test_a_complete_checkpoint_is_read(tmp_path):
    directory = _checkpoint(tmp_path / "ok")
    config = checkpoint.read_config(directory)
    assert config["num_layers"] == 2 and config["layer_norm_eps"] == 1e-5
    tokenizer = checkpoint.read_tokenizer(directory)
    assert (tokenizer["cls_token_id"], tokenizer["sep_token_id"]) == (0, 2)
    assert tokenizer["model_max_length"] == 16
    effective = json.loads(tokenizer["json"])
    assert effective["pre_tokenizer"]["add_prefix_space"] is True
    weights = checkpoint.read_weights(directory, config)
    assert "layoutlm.pooler.dense.weight" not in weights
    assert weights["qa_outputs.weight"].shape == (2, 8)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"model_type": "layoutlmv2"}, "model_type"),
        ({"architectures": ["LayoutLMForTokenClassification"]}, "LayoutLMForQuestionAnswering"),
        ({"hidden_act": "gelu_new"}, "GELU"),
        ({"position_embedding_type": "relative_key"}, "absolute"),
        ({"num_attention_heads": 3}, "multiple"),
        ({"max_2d_position_embeddings": 512}, "0..1000"),
    ],
)
def test_other_architectures_are_refused(tmp_path, config, message):
    directory = _checkpoint(tmp_path / "config", config=config)
    with pytest.raises(ValueError, match=message):
        checkpoint.read_config(directory)


def test_pre_split_words_need_the_prefix_space(tmp_path):
    directory = _checkpoint(tmp_path / "prefix", tokenizer_settings={"add_prefix_space": False})
    with pytest.raises(ValueError, match="add_prefix_space"):
        checkpoint.read_tokenizer(directory)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"truncation_side": "left"}, "truncation_side"),
        ({"cls_token": "<pad>"}, "cls_token"),
        ({"sep_token": {"content": "<unk>"}}, "sep_token"),
    ],
)
def test_framing_and_truncation_settings_must_match_the_reference(tmp_path, settings, message):
    directory = _checkpoint(tmp_path / "settings", tokenizer_settings=settings)
    with pytest.raises(ValueError, match=message):
        checkpoint.read_tokenizer(directory)


def test_other_tokenizer_shapes_are_refused(tmp_path):
    directory = _checkpoint(tmp_path / "tokenizer")
    document = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    for mutate, message in (
        (lambda value: value["model"].update(type="WordPiece"), "byte-level BPE"),
        (lambda value: value.update(normalizer={"type": "Lowercase"}), "normalizer"),
        (lambda value: value["post_processor"].update(type="BertProcessing"), "RoBERTa"),
        (lambda value: value["added_tokens"][0].update(normalized=True), "added token"),
    ):
        changed = json.loads(json.dumps(document))
        mutate(changed)
        (directory / "tokenizer.json").write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            checkpoint.read_tokenizer(directory)


def test_tensors_must_match_the_question_answering_head(tmp_path):
    missing = _tensors()
    del missing["qa_outputs.bias"]
    directory = _checkpoint(tmp_path / "missing", tensors=missing)
    config = checkpoint.read_config(directory)
    with pytest.raises(ValueError, match="missing"):
        checkpoint.read_weights(directory, config)
    extra = _tensors()
    extra["classifier.weight"] = np.zeros((3, 8), np.float32)
    directory = _checkpoint(tmp_path / "extra", tensors=extra)
    with pytest.raises(ValueError, match="unexpected"):
        checkpoint.read_weights(directory, config)
    reshaped = _tensors()
    reshaped["qa_outputs.weight"] = np.zeros((3, 8), np.float32)
    directory = _checkpoint(tmp_path / "shape", tensors=reshaped)
    with pytest.raises(ValueError, match="shape"):
        checkpoint.read_weights(directory, config)


def test_windows_must_fit_the_position_table(tmp_path):
    directory = _checkpoint(tmp_path / "window")
    with pytest.raises(ValueError, match="windows hold"):
        model.build(_request(model_dir=directory, max_sequence_length=21), _NoWriter())
