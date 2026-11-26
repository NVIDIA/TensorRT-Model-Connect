# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder-side checks of the OpenAI GPT family that need neither TensorRT nor a GPU:
configuration, the tokenizer contract, the checkpoint reader and the build options."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.openai_gpt import model as builder
from families.openai_gpt.config import parse_config, parse_json_object
from families.openai_gpt.tests.synthetic import TOKENIZER, mutations
from families.openai_gpt.tokenizer_contract import check_tokenizer
from families.openai_gpt.weights import expected_shapes, load_weights
from tensorrt_model_connect import BuildRequest

# config.json of openai-community/openai-gpt.
PUBLISHED = {
    "afn": "gelu",
    "architectures": ["OpenAIGPTLMHeadModel"],
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "openai-gpt",
    "n_ctx": 512,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 512,
    "n_special": 0,
    "predict_special_tokens": True,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 50}},
    "vocab_size": 40478,
}
TINY = {**PUBLISHED, "n_embd": 8, "n_head": 2, "n_layer": 2, "n_positions": 16, "vocab_size": 49}


def test_published_config() -> None:
    config = parse_config(PUBLISHED, {"_from_model_config": True})
    assert (config.vocab_size, config.n_positions, config.n_embd) == (40478, 512, 768)
    assert (config.n_layer, config.n_head, config.head_dim, config.inner_size) == (12, 12, 64, 3072)
    assert config.layer_norm_epsilon == float(np.float32(1e-5))
    assert config.eos_token_ids == ()


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "gpt2"},
        {"architectures": ["OpenAIGPTDoubleHeadsModel"]},
        {"architectures": ["OpenAIGPTLMHeadModel", "OpenAIGPTModel"]},
        {"afn": "relu"},
        {"afn": None},
        {"tie_word_embeddings": False},
        {"n_embd": None},
        {"n_embd": 768.0},
        {"n_embd": True},
        {"n_embd": 0},
        {"n_embd": 770},
        {"n_head": -12},
        {"n_layer": 2**31},
        {"vocab_size": "40478"},
        {"layer_norm_epsilon": 0.0},
        {"layer_norm_epsilon": -1e-5},
        {"layer_norm_epsilon": float("nan")},
        {"layer_norm_epsilon": float("inf")},
        {"layer_norm_epsilon": True},
        # Finite in Python, but zero, subnormal or infinite as the engine's float32.
        {"layer_norm_epsilon": 1e-50},
        {"layer_norm_epsilon": 1e-40},
        {"layer_norm_epsilon": 1e100},
    ],
)
def test_config_rejects_what_the_graph_cannot_build(change: dict) -> None:
    raw = {**PUBLISHED, **change}
    for name, value in change.items():
        if value is None:
            del raw[name]
    with pytest.raises(ValueError):
        parse_config(raw, {})


def test_eos_tokens_follow_transformers_generation_config() -> None:
    assert parse_config({**PUBLISHED, "eos_token_id": 3}, {}).eos_token_ids == (3,)
    assert parse_config(PUBLISHED, {"eos_token_id": [5, 4, 5]}).eos_token_ids == (5, 4)
    # generation_config.json wins, an explicit null included.
    assert parse_config({**PUBLISHED, "eos_token_id": 3}, {"eos_token_id": 7}).eos_token_ids == (7,)
    assert parse_config({**PUBLISHED, "eos_token_id": 3}, {"eos_token_id": None}).eos_token_ids == ()
    for bad in (40478, -1, True, 1.0, [1, "2"]):
        with pytest.raises(ValueError):
            parse_config(PUBLISHED, {"eos_token_id": bad})


def test_json_objects_refuse_repeated_keys() -> None:
    assert parse_json_object(b'{"a": 1, "b": {"a": 2}}', "x.json") == {"a": 1, "b": {"a": 2}}
    for text in (b'{"a": 1, "a": 2}', b'{"b": {"a": 1, "a": 1}}'):
        with pytest.raises(ValueError, match="repeats the key"):
            parse_json_object(text, "x.json")
    with pytest.raises(ValueError, match="one JSON object"):
        parse_json_object(b"[1]", "x.json")
    # The runtime's JSON parser refuses NaN and Infinity, so the builder does too.
    for constant in (b"NaN", b"Infinity", b"-Infinity"):
        with pytest.raises(ValueError, match="non-standard constant"):
            parse_json_object(b'{"model": {"dropout": ' + constant + b"}}", "tokenizer.json")
    # Python reads UTF-16 and UTF-32 JSON too; the runtime reads the bundled bytes as UTF-8.
    for data in ('{"a": 1}'.encode("utf-16"), '{"a": 1}'.encode("utf-32"), b'{"a": "\xff"}'):
        with pytest.raises(ValueError, match="not UTF-8"):
            parse_json_object(data, "tokenizer.json")
    with pytest.raises(ValueError):
        parse_json_object(b'\xef\xbb\xbf{"a": 1}', "tokenizer.json")
    for literal in (b"1e400", b"-1e400", b"[1e999]"):
        with pytest.raises(ValueError, match="double range"):
            parse_json_object(b'{"model": {"dropout": ' + literal + b"}}", "tokenizer.json")
    # A lone surrogate, as a key, a value or a list item, is not text the runtime can parse.
    for text in (b'{"\\ud800": 1}', b'{"a": "x\\udfff"}', b'{"a": ["ok", "\\ud834"]}'):
        with pytest.raises(ValueError, match="not valid Unicode"):
            parse_json_object(text, "tokenizer.json")


def test_synthetic_tokenizer_meets_the_contract() -> None:
    check_tokenizer(copy.deepcopy(TOKENIZER), 49)


@pytest.mark.parametrize("name", sorted(mutations()))
def test_tokenizer_contract_refusals(name: str) -> None:
    with pytest.raises(ValueError):
        check_tokenizer(mutations()[name], 49)


@pytest.mark.parametrize("rows", [48, 50])
def test_tokenizer_has_one_token_per_embedding_row(rows: int) -> None:
    # Fewer rows leave tokens without an embedding; more let the model emit ids without a token.
    with pytest.raises(ValueError, match="embedding rows"):
        check_tokenizer(copy.deepcopy(TOKENIZER), rows)


def test_layer_norm_epsilon_is_the_engine_float32() -> None:
    assert parse_config({**PUBLISHED, "layer_norm_epsilon": 1e-5}, {}).layer_norm_epsilon == float(
        np.float32(1e-5)
    )
    smallest = float(np.finfo(np.float32).tiny)
    assert parse_config({**PUBLISHED, "layer_norm_epsilon": smallest}, {}).layer_norm_epsilon == smallest


def _tiny_checkpoint(directory: Path, *, change=None, extra=None, drop=None) -> dict:
    """A random float32 checkpoint of TINY in the published tensor layout."""
    config = parse_config(TINY, {})
    rng = np.random.default_rng(0)
    tensors = {
        name: rng.standard_normal(shape).astype(np.float32)
        for name, shape in expected_shapes(config).items()
    }
    for index in range(config.n_layer):
        tensors[f"h.{index}.attn.bias"] = np.tril(
            np.ones((config.n_positions, config.n_positions), dtype=np.float32)
        ).reshape(1, 1, config.n_positions, config.n_positions)
    if change:
        change(tensors)
    tensors.update(extra or {})
    for name in drop or ():
        del tensors[name]
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(TINY), encoding="utf-8")
    return tensors


def test_weights_read_every_tensor_of_the_graph(tmp_path: Path) -> None:
    tensors = _tiny_checkpoint(tmp_path)
    weights = load_weights(tmp_path, parse_config(TINY, {}))
    np.testing.assert_array_equal(weights.tokens_embed, tensors["tokens_embed.weight"])
    np.testing.assert_array_equal(weights.positions_embed, tensors["positions_embed.weight"])
    layer = weights.layers[1]
    np.testing.assert_array_equal(layer.attn_weight, tensors["h.1.attn.c_attn.weight"])
    np.testing.assert_array_equal(layer.mlp_proj_bias, tensors["h.1.mlp.c_proj.bias"])
    np.testing.assert_array_equal(layer.ln_2_weight, tensors["h.1.ln_2.weight"])
    assert len(weights.layers) == 2


def test_weights_check_the_layer_count_against_the_file_first(tmp_path: Path) -> None:
    """A declared layer count far beyond the file is refused before a name is built per layer."""
    _tiny_checkpoint(tmp_path)
    for layers in (3, 2**31 - 1):
        with pytest.raises(ValueError, match="tensors; n_layer="):
            load_weights(tmp_path, parse_config({**TINY, "n_layer": layers}, {}))
    # Fewer declared layers than stored leave tensors unused, which are named.
    with pytest.raises(ValueError, match="does not use: h.1."):
        load_weights(tmp_path, parse_config({**TINY, "n_layer": 1}, {}))


def _with(name: str, value):
    def change(tensors: dict) -> None:
        tensors[name] = value

    return change


@pytest.mark.parametrize(
    "case, arguments, message",
    [
        ("missing", {"drop": ["h.1.ln_1.bias"]}, "missing tensors"),
        ("untied head", {"extra": {"lm_head.weight": np.zeros((49, 8), np.float32)}}, "does not use"),
        ("third mask", {"extra": {"h.2.attn.bias": np.zeros((1, 1, 16, 16), np.float32)}}, "does not use"),
        ("float16", {"change": _with("h.0.ln_1.weight", np.ones(8, np.float16))}, "stored as F32"),
        ("shape", {"change": _with("h.0.mlp.c_fc.weight", np.ones((32, 8), np.float32))}, "shape"),
        ("transposed", {"change": _with("h.0.attn.c_attn.weight", np.ones((24, 8), np.float32))}, "shape"),
        ("nan", {"change": _with("h.1.attn.c_proj.bias", np.full(8, np.nan, np.float32))}, "NaN"),
        ("inf", {"change": _with("tokens_embed.weight", np.full((49, 8), np.inf, np.float32))}, "infinite"),
    ],
)
def test_weights_refuse_a_checkpoint_the_graph_cannot_read(
    tmp_path: Path, case: str, arguments: dict, message: str
) -> None:
    _tiny_checkpoint(tmp_path, **arguments)
    with pytest.raises(ValueError, match=message):
        load_weights(tmp_path, parse_config(TINY, {}))


def _request(model_dir: Path, **changes) -> BuildRequest:
    fields = {
        "model_dir": model_dir,
        "output_path": model_dir / "out.bundle",
        "family": "openai_gpt",
        "task": "text_continuation",
        "precision": "fp32",
    }
    fields.update(changes)
    return BuildRequest(**fields)


@pytest.mark.parametrize(
    "changes",
    [
        {"task": "text_generation"},
        {"backend": "trt_rtx"},
        {"precision": "fp16"},
        {"precision": "bf16"},
        {"quantization": "fp8"},
        {"image_height": 224},
        {"image_width": 224},
        {"video_num_frames": 8},
        {"mtp_seq_len": 2},
        {"max_batch_size": 2},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"fp32_layers": (0,)},
        {"dynamic_kv_cache": True},
        {"max_sequence_length": 513},
    ],
)
def test_build_rejects_options_it_cannot_honor(tmp_path: Path, changes: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(PUBLISHED), encoding="utf-8")
    with pytest.raises((ValueError, NotImplementedError)):
        builder.build(_request(tmp_path, **changes), writer=None)


def test_runtime_config_carries_the_cache_and_eos_contract() -> None:
    config = parse_config({**PUBLISHED, "eos_token_id": [1, 2]}, {})
    assert builder.runtime_config(config, 300) == {
        "vocab_size": 40478,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "head_dim": 64,
        "max_cache_length": 300,
        "precision": "fp32",
        "eos_token_ids": [1, 2],
    }
