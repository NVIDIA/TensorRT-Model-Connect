# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from families.nomic_bert.cli import BuildRequest, coerce_request
from families.nomic_bert.config import ModelConfig
from families.nomic_bert import model
from tensorrt_model_connect.build import BuildRequest as SharedRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


CONFIG = {
    "model_type": "nomic_bert",
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,
    "activation_function": "swiglu",
    "prenorm": False,
    "parallel_block": False,
    "causal": False,
    "rotary_emb_fraction": 1.0,
    "rotary_emb_interleaved": False,
    "qkv_proj_bias": False,
    "mlp_fc1_bias": False,
    "mlp_fc2_bias": False,
    "type_vocab_size": 2,
    "pad_token_id": 0,
    "vocab_size": 30528,
    "layer_norm_epsilon": 1e-12,
    "rotary_emb_base": 1000,
}


def test_support_resolves_one_owner():
    family, support = resolve_family(ModelMetadata(CONFIG, {}))
    assert family == "nomic_bert"
    assert support.tasks == ("text_to_embedding",)
    assert support.default_task == "text_to_embedding"


def test_official_config_is_accepted(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    assert ModelConfig.from_dir(tmp_path) == ModelConfig(30528, 1e-12, 1000.0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("prenorm", True),
        ("rotary_emb_fraction", 0.5),
        ("qkv_proj_bias", True),
        ("rotary_scaling_factor", 2),
        ("rotary_emb_scale_base", 512),
        ("num_experts", 4),
        ("rotary_head_dim", True),
        ("activation_function", "gelu"),
        ("vocab_size", True),
        ("layer_norm_epsilon", float("nan")),
        ("rotary_emb_base", 0),
        ("causal", True),
    ],
)
def test_unsupported_topologies_fail_before_loading_weights(tmp_path, key, value):
    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, key: value}))
    with pytest.raises(ValueError):
        ModelConfig.from_dir(tmp_path)


@pytest.mark.parametrize("length", [None, True, 0, 1, 2049, 2.5])
def test_invalid_profile_bounds_fail(tmp_path, length):
    with pytest.raises(ValueError):
        BuildRequest(tmp_path, max_sequence_length=length)


def test_shared_build_dispatch_keeps_family_bounds(tmp_path):
    request = SharedRequest(
        tmp_path, tmp_path / "n.bundle", "nomic_bert", "text_to_embedding", "fp32"
    )
    assert coerce_request(request) == BuildRequest(tmp_path)
    assert coerce_request(replace(request, max_sequence_length=128)).max_sequence_length == 128
    for change in (
        {"tensor_parallel_size": 2},
        {"max_batch_size": 2},
        {"dynamic_kv_cache": True},
        {"quantization": "int8"},
        {"precision": "fp16"},
    ):
        with pytest.raises(ValueError):
            coerce_request(replace(request, **change))


@pytest.mark.parametrize(
    "change",
    [
        {"norm_mlp": True},
        {"num_heads_kv": 6},
        {"rope_parameters": {"rope_type": "dynamic", "factor": 2}},
        {"n_head": 12.0},
    ],
)
def test_additional_architecture_variants_fail_closed(tmp_path, change):
    (tmp_path / "config.json").write_text(json.dumps({**CONFIG, **change}))
    with pytest.raises(ValueError):
        ModelConfig.from_dir(tmp_path)


@pytest.mark.parametrize("change", ["pre_tokenizer", "normalizer", "framing", "vocab"])
def test_tokenizer_variants_fail_before_weight_loading(tmp_path, monkeypatch, change):
    vocab = {f"token{index}": index for index in range(103)}
    for token, index in (("[PAD]", 0), ("[UNK]", 100), ("[CLS]", 101), ("[SEP]", 102)):
        del vocab[f"token{index}"]
        vocab[token] = index
    tokenizer = {
        "model": {"type": "WordPiece", "vocab": vocab},
        "normalizer": {"type": "BertNormalizer"},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
        },
    }
    changed = deepcopy(tokenizer)
    if change == "pre_tokenizer":
        changed["pre_tokenizer"]["type"] = "Whitespace"
    elif change == "normalizer":
        changed["normalizer"]["type"] = "NFKC"
    elif change == "framing":
        changed["post_processor"]["single"].reverse()
    else:
        changed["model"]["vocab"]["token1"] = 40000
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    (tmp_path / "tokenizer.json").write_text(json.dumps(changed))

    def unexpected_weights(*args):
        pytest.fail("unsupported tokenizer reached weight loading")

    monkeypatch.setattr(model, "load_weights", unexpected_weights)
    with pytest.raises(ValueError, match="tokenizer"):
        model.build(BuildRequest(tmp_path), None)
