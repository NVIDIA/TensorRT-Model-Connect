# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder contract of ministral3 on a tiny synthetic checkpoint (CPU, no TensorRT)."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("safetensors")
from safetensors.numpy import save_file

from families.ministral3 import model as builder
from families.ministral3.config import ModelConfig
from families.ministral3.tests.test_config import CONFIG, MODULES, POOLING, SENTENCE, SENTENCE_BERT
from families.ministral3.weights import expected_shapes, load_weights

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class Writer:
    def __init__(self):
        self.header, self.sections = None, {}

    def set_header(self, **header):
        self.header = header

    def add_bytes(self, name, value):
        self.sections[name] = value

    def add_json(self, name, value):
        self.sections[name] = value


@pytest.fixture
def checkpoint(tmp_path):
    tokenizer = json.loads((FIXTURES / "tekken_tokenizer.json").read_text(encoding="utf-8"))[
        "tokenizer"
    ]
    config = copy.deepcopy(CONFIG)
    config.update(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=len(tokenizer["model"]["vocab"]),
    )
    pooling = dict(POOLING, word_embedding_dimension=8)
    (tmp_path / "1_Pooling").mkdir()
    for name, value in (
        ("config.json", config),
        ("modules.json", MODULES),
        ("1_Pooling/config.json", pooling),
        ("config_sentence_transformers.json", SENTENCE),
        ("sentence_bert_config.json", SENTENCE_BERT),
        ("tokenizer.json", tokenizer),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    parsed = ModelConfig.from_dir(tmp_path)
    rng = np.random.default_rng(252)
    tensors = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in expected_shapes(parsed).items()
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path, tensors


@pytest.fixture
def engines(monkeypatch):
    calls = []

    def build_engine(config, weights, **options):
        calls.append((config, weights, options))
        return b"plan"

    monkeypatch.setitem(
        sys.modules, "families.ministral3.graph", SimpleNamespace(build_engine=build_engine)
    )
    return calls


def request(model_dir, **overrides):
    values = dict(
        model_dir=model_dir,
        task="text_to_embedding",
        precision="bf16",
        backend="trt",
        max_sequence_length=None,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        dynamic_kv_cache=False,
        verbose=False,
        mtp_seq_len=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bundle_sections_and_runtime_contract(checkpoint, engines):
    model_dir, tensors = checkpoint
    writer = Writer()
    builder.build(request(model_dir, task="text_to_token_features", max_sequence_length=64), writer)
    assert writer.header == {
        "family": "ministral3",
        "task": "text_to_token_features",
        "backend": "trt",
    }
    assert writer.sections["engine.plan"] == b"plan"
    assert writer.sections["tokenizer.json"] == (model_dir / "tokenizer.json").read_bytes()
    assert writer.sections["runtime.json"] == {
        "hidden_size": 8,
        "vocab_size": tensors["embed_tokens.weight"].shape[0],
        "max_sequence_length": 64,
        "precision": "bf16",
        "query_prompt": "query: ",
        "document_prompt": "passage: ",
    }
    ((config, weights, options),) = engines
    assert options == {
        "precision": "bf16",
        "max_sequence_length": 64,
        "opt_sequence_length": 64,
        "verbose": False,
    }
    assert set(weights) == set(tensors)
    np.testing.assert_array_equal(
        weights["layers.1.mlp.down_proj.weight"], tensors["layers.1.mlp.down_proj.weight"]
    )


def test_default_and_maximum_sequence_lengths(checkpoint, engines):
    model_dir, _ = checkpoint
    builder.build(request(model_dir), Writer())
    assert engines[-1][2]["max_sequence_length"] == 8192
    assert engines[-1][2]["opt_sequence_length"] == 512
    builder.build(request(model_dir, max_sequence_length=32768, precision="fp32"), Writer())
    assert engines[-1][2]["max_sequence_length"] == 32768
    with pytest.raises(ValueError, match="max_sequence_length"):
        builder.build(request(model_dir, max_sequence_length=32769), Writer())


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("task", "text_continuation", ValueError),
        ("backend", "trt_rtx", ValueError),
        ("precision", "fp16", ValueError),
        ("quantization", "fp8", NotImplementedError),
        ("fp32_layers", (1,), NotImplementedError),
        ("max_batch_size", 2, NotImplementedError),
        ("tensor_parallel_size", 2, NotImplementedError),
        ("context_parallel_size", 2, ValueError),
        ("dynamic_kv_cache", True, NotImplementedError),
        ("mtp_seq_len", 2, NotImplementedError),
        ("image_height", 224, NotImplementedError),
        ("image_width", 224, NotImplementedError),
        ("video_num_frames", 8, NotImplementedError),
    ],
)
def test_unsupported_build_options_are_rejected(checkpoint, engines, field, value, error):
    model_dir, _ = checkpoint
    with pytest.raises(error):
        builder.build(request(model_dir, **{field: value}), Writer())
    assert engines == []


def test_checkpoint_tensors_must_match_exactly(checkpoint):
    model_dir, tensors = checkpoint
    config = ModelConfig.from_dir(model_dir)
    assert set(load_weights(model_dir, config)) == set(tensors)
    extra = dict(tensors, **{"lm_head.weight": tensors["embed_tokens.weight"]})
    save_file(extra, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="extra"):
        load_weights(model_dir, config)
    missing = {name: value for name, value in tensors.items() if name != "norm.weight"}
    save_file(missing, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        load_weights(model_dir, config)
    reshaped = dict(tensors, **{"norm.weight": np.ones(9, dtype=np.float32)})
    save_file(reshaped, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="shape"):
        load_weights(model_dir, config)


def test_tokenizer_vocabulary_must_match_the_config(checkpoint, engines):
    model_dir, _ = checkpoint
    config = json.loads((model_dir / "config.json").read_text())
    config["vocab_size"] += 1
    (model_dir / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="vocab size"):
        builder.build(request(model_dir), Writer())
    assert engines == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "ignore_merges"), False),
        (("model", "byte_fallback"), True),
        (("normalizer",), {"type": "NFC"}),
        (("pre_tokenizer", "pretokenizers", 0, "pattern", "Regex"), "\\s+"),
        (("post_processor", "single"), [{"SpecialToken": {"id": "<s>", "type_id": 0}}]),
        (("added_tokens", 0, "lstrip"), True),
    ],
)
def test_tokenizer_contract_is_checked_at_build_time(checkpoint, engines, path, value):
    model_dir, _ = checkpoint
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    target = tokenizer
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer))
    with pytest.raises(ValueError, match="Tekken contract"):
        builder.build(request(model_dir), Writer())
    assert engines == []
