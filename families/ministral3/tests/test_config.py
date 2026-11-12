# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-contract parsing of ministral3 (CPU, no TensorRT)."""

import copy
import json

import pytest

from families.ministral3.config import ModelConfig

# config.json and the Sentence Transformers files of
# nvidia/Nemotron-3-Embed-1B-BF16@c0c9fea93ea424587517f2c59e20db9f1d6bf615.
CONFIG = {
    "architectures": ["Ministral3Model"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "dtype": "bfloat16",
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "is_causal": False,
    "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 16384},
    "max_position_embeddings": 262144,
    "model_type": "ministral3",
    "num_attention_heads": 24,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pad_token_id": 11,
    "pooling": "avg",
    "rms_norm_eps": 1e-05,
    "rope_parameters": {
        "apply_yarn_scaling": False,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 16.0,
        "llama_4_scaling_beta": 0.1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 16384,
        "rope_theta": 1000000.0,
        "rope_type": "yarn",
        "type": "yarn",
    },
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 131072,
}
MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    },
]
POOLING = {
    "word_embedding_dimension": 2048,
    "pooling_mode_cls_token": False,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_tokens": True,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": False,
    "include_prompt": True,
}
SENTENCE = {
    "prompts": {"query": "query: ", "document": "passage: "},
    "default_prompt_name": None,
    "similarity_fn_name": "cosine",
}
SENTENCE_BERT = {"max_seq_length": 32768, "do_lower_case": False}


def parse(
    config=CONFIG, modules=MODULES, pooling=POOLING, sentence=SENTENCE, sentence_bert=SENTENCE_BERT
):
    return ModelConfig.from_dicts(
        config, modules=modules, pooling=pooling, sentence=sentence, sentence_bert=sentence_bert
    )


def test_checkpoint_contract_is_parsed():
    config = parse()
    assert (config.hidden_size, config.num_hidden_layers, config.head_dim) == (2048, 16, 128)
    assert (config.num_attention_heads, config.num_key_value_heads) == (24, 8)
    assert config.attention_size == 3072 and config.kv_attention_size == 1024
    assert config.rope.rope_type == "yarn" and config.rope.factor == 16.0
    assert config.rope.original_max_position_embeddings == 16384
    assert config.rope.llama_4_scaling_beta == 0.1 and config.rope.truncate is True
    assert config.sentence.prompts == {"query": "query: ", "document": "passage: "}
    assert config.sentence.max_seq_length == 32768


def test_checkpoint_files_are_read_from_the_model_directory(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    for name, value in (
        ("config.json", CONFIG),
        ("modules.json", MODULES),
        ("1_Pooling/config.json", POOLING),
        ("config_sentence_transformers.json", SENTENCE),
        ("sentence_bert_config.json", SENTENCE_BERT),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    assert ModelConfig.from_dir(tmp_path) == parse()
    (tmp_path / "modules.json").unlink()
    with pytest.raises(ValueError, match="modules.json"):
        ModelConfig.from_dir(tmp_path)


def test_an_rms_norm_eps_that_overflows_to_infinity_is_rejected(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    for name, value in (
        ("modules.json", MODULES),
        ("1_Pooling/config.json", POOLING),
        ("config_sentence_transformers.json", SENTENCE),
        ("sentence_bert_config.json", SENTENCE_BERT),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    text = json.dumps(CONFIG)
    assert '"rms_norm_eps": 1e-05' in text
    # JSON parsing turns the literal 1e400 into float("inf").
    (tmp_path / "config.json").write_text(
        text.replace('"rms_norm_eps": 1e-05', '"rms_norm_eps": 1e400'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rms_norm_eps"):
        ModelConfig.from_dir(tmp_path)


def edited(source, path, value):
    result = copy.deepcopy(source)
    target = result
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return result


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("is_causal",), True, "bidirectional"),
        (("is_causal",), KeyError, "bidirectional"),
        (("architectures",), ["Ministral3ForCausalLM"], "Ministral3Model"),
        (("sliding_window",), 4096, "sliding-window"),
        (("hidden_act",), "gelu", "silu"),
        (("num_key_value_heads",), 7, "multiple"),
        (("rope_parameters", "rope_type"), "longrope", "rope_type"),
        (("rope_parameters", "apply_yarn_scaling"), True, "apply_yarn_scaling"),
        (("rope_parameters", "llama_4_scaling_beta"), KeyError, "llama_4_scaling_beta"),
        (("rope_parameters", "partial_rotary_factor"), 0.5, "keys"),
        (("llama_4_scaling", "beta"), 0.2, "disagrees"),
        (("rms_norm_eps",), 0, "rms_norm_eps"),
        (("rms_norm_eps",), float("inf"), "rms_norm_eps"),
        (("rms_norm_eps",), float("nan"), "rms_norm_eps"),
        (("rms_norm_eps",), True, "rms_norm_eps"),
        (("partial_rotary_factor",), 0.5, "full rotary"),
        (("original_max_position_embeddings",), 8192, "disagrees"),
    ],
)
def test_unsupported_backbones_are_rejected(path, value, message):
    with pytest.raises(ValueError, match=message):
        parse(config=edited(CONFIG, path, value))


@pytest.mark.parametrize(
    ("argument", "source", "path", "value", "message"),
    [
        ("modules", MODULES, (2,), MODULES[1], "module stack"),
        ("pooling", POOLING, ("pooling_mode_cls_token",), True, "mean pooling"),
        ("pooling", POOLING, ("include_prompt",), False, "prompt"),
        ("pooling", POOLING, ("word_embedding_dimension",), 1024, "hidden_size"),
        ("sentence", SENTENCE, ("prompts",), {"query": "q: "}, "prompts"),
        ("sentence", SENTENCE, ("default_prompt_name",), "query", "default_prompt_name"),
        ("sentence", SENTENCE, ("similarity_fn_name",), "dot", "cosine"),
        ("sentence_bert", SENTENCE_BERT, ("do_lower_case",), True, "lower-case"),
        ("sentence_bert", SENTENCE_BERT, ("max_seq_length",), 0, "max_seq_length"),
    ],
)
def test_unsupported_sentence_heads_are_rejected(argument, source, path, value, message):
    with pytest.raises(ValueError, match=message):
        parse(**{argument: edited(source, path, value)})
