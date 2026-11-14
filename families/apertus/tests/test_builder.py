# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Apertus builder: configuration, RoPE constants, tokenizer and weights."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.apertus.config import RopeConfig, parse_config
from families.apertus.model import tokenizer_contract

# config.json and generation_config.json of swiss-ai/Apertus-8B-Instruct-2509
# (revision b946d40447b2b597999b9c86d44bee0b452c919f).
CHECKPOINT_CONFIG = {
    "architectures": ["ApertusForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "dtype": "bfloat16",
    "eos_token_id": 68,
    "hidden_act": "xielu",
    "hidden_dropout": 0.0,
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 21504,
    "max_position_embeddings": 65536,
    "mlp_bias": False,
    "model_type": "apertus",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pad_token_id": 3,
    "post_norm": False,
    "qk_norm": True,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
        "type": "llama3",
    },
    "rope_theta": 12000000,
    "tie_word_embeddings": False,
    "transformers_version": "4.56.0.dev0",
    "use_cache": False,
    "vocab_size": 131072,
}
CHECKPOINT_GENERATION = {"bos_token_id": 1, "eos_token_id": [2, 68, 72]}
CHECKPOINT_POST_PROCESSOR = {
    "type": "TemplateProcessing",
    "single": [
        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
    ],
    "pair": [
        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<s>", "type_id": 1}},
        {"Sequence": {"id": "B", "type_id": 1}},
    ],
    "special_tokens": {
        "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]},
        "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
    },
}


def test_checkpoint_config_is_parsed():
    config = parse_config(CHECKPOINT_CONFIG, CHECKPOINT_GENERATION)
    assert config.head_dim == 128
    assert (config.num_attention_heads, config.num_key_value_heads) == (32, 8)
    assert config.attention_size == 4096 and config.kv_size == 1024
    assert config.eos_token_ids == (2, 68, 72)
    assert config.bos_token_id == 1 and config.pad_token_id == 3
    assert config.rope == RopeConfig(
        theta=12000000.0,
        rope_type="llama3",
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )


def test_rope_parameters_format_matches_the_legacy_keys():
    raw = copy.deepcopy(CHECKPOINT_CONFIG)
    scaling = raw.pop("rope_scaling")
    raw["rope_parameters"] = {**scaling, "rope_theta": raw.pop("rope_theta")}
    assert parse_config(raw, CHECKPOINT_GENERATION) == parse_config(
        CHECKPOINT_CONFIG, CHECKPOINT_GENERATION
    )


def test_eos_falls_back_to_config_without_generation_config():
    assert parse_config(CHECKPOINT_CONFIG, {}).eos_token_ids == (68,)


@pytest.mark.parametrize("rope", [None, {}])
def test_missing_rope_schedule_is_rejected_instead_of_guessed(rope):
    raw = {key: value for key, value in CHECKPOINT_CONFIG.items() if key != "rope_scaling"}
    if rope is not None:
        raw["rope_scaling"] = rope
    with pytest.raises(ValueError, match="rope_parameters or rope_scaling"):
        parse_config(raw, CHECKPOINT_GENERATION)


@pytest.mark.parametrize(
    "update",
    [
        {"model_type": "llama"},
        {"architectures": ["ApertusForTokenClassification"]},
        {"hidden_act": "silu"},
        {"qk_norm": False},
        {"post_norm": True},
        {"tie_word_embeddings": True},
        {"attention_bias": True},
        {"mlp_bias": True},
        {"num_key_value_heads": 5},
        {"head_dim": 127},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"rope_scaling": {**CHECKPOINT_CONFIG["rope_scaling"], "low_freq_factor": 4.0}},
        {"rope_scaling": {**CHECKPOINT_CONFIG["rope_scaling"], "partial_rotary_factor": 0.5}},
        {"rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}},
        {"rms_norm_eps": 0.0},
        {"eos_token_id": 131072},
    ],
)
def test_unsupported_configurations_are_rejected(update):
    raw = {**CHECKPOINT_CONFIG, **update}
    with pytest.raises(ValueError):
        parse_config(raw, {} if "eos_token_id" in update else CHECKPOINT_GENERATION)


@pytest.mark.parametrize("rope_type", ["llama3", "default"])
def test_rope_inverse_frequencies_match_transformers(rope_type):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from families.apertus.rope import rope_inverse_frequencies

    raw = copy.deepcopy(CHECKPOINT_CONFIG)
    if rope_type == "default":
        raw["rope_scaling"] = {"rope_type": "default"}
    hf_config = transformers.ApertusConfig(
        **{key: value for key, value in raw.items() if key not in {"architectures", "dtype"}}
    )
    config = parse_config(raw, CHECKPOINT_GENERATION)
    if rope_type == "llama3":
        expected, scaling = ROPE_INIT_FUNCTIONS["llama3"](hf_config, "cpu")
    else:
        expected, scaling = (
            transformers.models.apertus.modeling_apertus.ApertusRotaryEmbedding.compute_default_rope_parameters(
                hf_config, "cpu"
            )
        )
    assert scaling == 1.0
    actual = rope_inverse_frequencies(config.head_dim, config.rope)
    np.testing.assert_array_equal(actual, expected.numpy())


def test_rope_tables_hold_cos_and_sin_of_position_times_frequency():
    pytest.importorskip("torch")
    from families.apertus.rope import rope_inverse_frequencies, rope_tables

    config = parse_config(CHECKPOINT_CONFIG, CHECKPOINT_GENERATION)
    cos, sin = rope_tables(config.head_dim, config.rope, 70000)
    inv_freq = rope_inverse_frequencies(config.head_dim, config.rope)
    assert cos.shape == sin.shape == (70000, 64)
    angles = np.float32(65535) * inv_freq
    np.testing.assert_allclose(cos[65535], np.cos(angles.astype(np.float64)), atol=2e-6)
    np.testing.assert_allclose(sin[65535], np.sin(angles.astype(np.float64)), atol=2e-6)


def test_tokenizer_contract_reads_the_post_processor():
    assert tokenizer_contract({"post_processor": CHECKPOINT_POST_PROCESSOR}) == {
        "tokenizer_add_special_tokens": True,
        "tokenizer_prefix_ids": [1],
        "tokenizer_suffix_ids": [],
    }
    assert tokenizer_contract({"post_processor": None})["tokenizer_prefix_ids"] == []
    suffixed = copy.deepcopy(CHECKPOINT_POST_PROCESSOR)
    suffixed["single"].append({"SpecialToken": {"id": "</s>", "type_id": 0}})
    assert tokenizer_contract({"post_processor": suffixed})["tokenizer_suffix_ids"] == [2]
    with pytest.raises(ValueError):
        tokenizer_contract({"post_processor": {"type": "RobertaProcessing"}})


TINY = {
    **CHECKPOINT_CONFIG,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_hidden_layers": 2,
    "vocab_size": 32,
    "max_position_embeddings": 64,
    "rope_scaling": {**CHECKPOINT_CONFIG["rope_scaling"], "original_max_position_embeddings": 16},
}


def _tiny_tensors(ml_dtypes) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(256)

    def bf16(*shape):
        return rng.standard_normal(shape).astype(np.float32).astype(ml_dtypes.bfloat16)

    tensors = {
        "model.embed_tokens.weight": bf16(32, 8),
        "model.norm.weight": bf16(8),
        "lm_head.weight": bf16(32, 8),
    }
    for layer, (alpha_p, alpha_n) in enumerate(((166.0, 40.75), (-0.5, 0.25))):
        prefix = f"model.layers.{layer}"
        tensors.update(
            {
                f"{prefix}.attention_layernorm.weight": bf16(8),
                f"{prefix}.feedforward_layernorm.weight": bf16(8),
                f"{prefix}.self_attn.q_proj.weight": bf16(8, 8),
                f"{prefix}.self_attn.k_proj.weight": bf16(4, 8),
                f"{prefix}.self_attn.v_proj.weight": bf16(4, 8),
                f"{prefix}.self_attn.o_proj.weight": bf16(8, 8),
                f"{prefix}.self_attn.q_norm.weight": bf16(4),
                f"{prefix}.self_attn.k_norm.weight": bf16(4),
                f"{prefix}.mlp.up_proj.weight": bf16(16, 8),
                f"{prefix}.mlp.down_proj.weight": bf16(8, 16),
                f"{prefix}.mlp.act_fn.alpha_p": np.array([alpha_p], dtype=ml_dtypes.bfloat16),
                f"{prefix}.mlp.act_fn.alpha_n": np.array([alpha_n], dtype=ml_dtypes.bfloat16),
                f"{prefix}.mlp.act_fn.beta": np.array(0.5, dtype=ml_dtypes.bfloat16),
                f"{prefix}.mlp.act_fn.eps": np.array(-1e-6, dtype=ml_dtypes.bfloat16),
            }
        )
    return tensors


def _write_checkpoint(directory: Path, tensors: dict[str, np.ndarray]) -> Path:
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(TINY), encoding="utf-8")
    return directory


def test_weights_keep_bf16_bits_and_reparameterize_xielu(tmp_path):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    pytest.importorskip("safetensors")
    from families.apertus.weights import load_weights, softplus

    tensors = _tiny_tensors(ml_dtypes)
    model_dir = _write_checkpoint(tmp_path / "tiny", tensors)
    config = parse_config(TINY, CHECKPOINT_GENERATION | {"eos_token_id": [2]})
    bf16 = load_weights(model_dir, config, "bf16")
    fp32 = load_weights(model_dir, config, "fp32")
    source = tensors["model.layers.1.mlp.up_proj.weight"]
    assert bf16.layers[1].up_proj.dtype == source.dtype
    np.testing.assert_array_equal(bf16.layers[1].up_proj.view(np.uint16), source.view(np.uint16))
    np.testing.assert_array_equal(fp32.layers[1].up_proj, source.astype(np.float32))
    # torch.nn.functional.softplus returns x itself above its threshold of 20.
    assert bf16.layers[0].xielu_alpha_p == 166.0
    assert bf16.layers[0].xielu_alpha_n == pytest.approx(0.5 + 40.75)
    assert bf16.layers[1].xielu_alpha_p == pytest.approx(np.log1p(np.exp(-0.5)), rel=1e-6)
    assert bf16.layers[1].xielu_alpha_n == pytest.approx(0.5 + np.log1p(np.exp(0.25)), rel=1e-6)
    assert bf16.layers[1].xielu_eps == float(np.float32(np.array(-1e-6, ml_dtypes.bfloat16)))
    assert softplus(21.0) == 21.0


@pytest.mark.parametrize("change", ["missing", "extra", "shape"])
def test_weights_reject_a_checkpoint_that_differs_from_the_graph(tmp_path, change):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    pytest.importorskip("safetensors")
    from families.apertus.weights import load_weights

    tensors = _tiny_tensors(ml_dtypes)
    if change == "missing":
        del tensors["model.layers.1.self_attn.k_norm.weight"]
    elif change == "extra":
        tensors["model.layers.1.self_attn.q_proj.bias"] = tensors["model.norm.weight"].copy()
    else:
        tensors["model.layers.0.mlp.up_proj.weight"] = tensors["lm_head.weight"].copy()
    model_dir = _write_checkpoint(tmp_path / change, tensors)
    config = parse_config(TINY, CHECKPOINT_GENERATION | {"eos_token_id": [2]})
    with pytest.raises(ValueError):
        load_weights(model_dir, config, "bf16")


def _request(**updates):
    fields = {
        "model_dir": Path("/nonexistent"),
        "task": "text_continuation",
        "precision": "bf16",
        "backend": "trt",
        "max_sequence_length": None,
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "max_batch_size": 1,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "quantization": None,
        "fp32_layers": (),
        "dynamic_kv_cache": False,
        "verbose": False,
        "mtp_seq_len": None,
    }
    return SimpleNamespace(**{**fields, **updates})


@pytest.mark.parametrize(
    "update",
    [
        {"task": "text_generation"},
        {"backend": "trt_rtx"},
        {"precision": "fp16"},
        {"image_height": 224},
        {"image_width": 224},
        {"video_num_frames": 8},
        {"mtp_seq_len": 2},
        {"max_batch_size": 2},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"quantization": "fp8"},
        {"fp32_layers": (0,)},
        {"dynamic_kv_cache": True},
    ],
)
def test_build_rejects_requests_it_cannot_honor(update):
    pytest.importorskip("tensorrt")
    from families.apertus.model import build

    class Writer:
        def __getattr__(self, name):
            raise AssertionError(f"writer.{name} must not be reached")

    with pytest.raises((ValueError, NotImplementedError)):
        build(_request(**update), Writer())


@pytest.mark.parametrize(
    ("token", "ids"),
    [("<s>", [-1]), ("<s>", [131072]), ("<s>", [True]), ("</s>", [131072])],
)
def test_build_rejects_tokenizer_ids_outside_the_vocabulary(tmp_path, token, ids):
    pytest.importorskip("tensorrt")
    from families.apertus.model import build

    post_processor = copy.deepcopy(CHECKPOINT_POST_PROCESSOR)
    post_processor["single"].append({"SpecialToken": {"id": "</s>", "type_id": 0}})
    post_processor["special_tokens"][token]["ids"] = ids
    (tmp_path / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
    (tmp_path / "generation_config.json").write_text(json.dumps(CHECKPOINT_GENERATION))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"post_processor": post_processor}))

    class Writer:
        def __getattr__(self, name):
            raise AssertionError(f"writer.{name} must not be reached")

    with pytest.raises(ValueError, match=r"tokenizer_(prefix|suffix)_ids must be inside"):
        build(_request(model_dir=tmp_path), Writer())
