# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for TIPSv2, none of which need the checkpoint or a GPU.

Three subjects:

* ``config.resolve``, whose job is to read the geometry from the tensors and from the same
  factory table the reference uses, and to refuse a ``config.json`` that contradicts either. The
  vision head count is the sharp case: the reference **ignores** ``vision_config``'s, so a
  resolver that trusted the file would build a differently-headed model than the file describes;
* ``tokenizer_export``, whose asset the runtime reads without protobuf or SentencePiece;
* the two values that are computed rather than read - the sinusoidal position signal and the
  lowercase table - and the recorded Pillow output the C++ resize is measured against.

The weights here are zero-strided views, because ``config.resolve`` reads shapes and nothing
else; a test that allocated the real so400m tensors would spend 190 MB to learn the same thing.
"""

from __future__ import annotations

import bisect
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tensorrt as trt

from families.tipsv2 import config as config_module, graph, model as model_module
from families.tipsv2 import tokenizer_export as tx
from families.tipsv2 import text_builder, vision_builder
from tensorrt_model_connect import BuildRequest

# The released so400m geometry: the resolver is keyed on it, so the fixture uses it rather than a
# miniature that no factory entry would claim.
_WIDTH = 1152
_INTERMEDIATE = 4304
_LAYERS = 27
_HEADS = 16
_PATCH = 14
_GRID = 32
_VOCAB = 32000
_WINDOW = 64
_IMAGE = _GRID * _PATCH

_PROCESSOR = {
    "image_processor": {
        "do_convert_rgb": True, "do_normalize": False, "do_rescale": True, "do_resize": True,
        "image_processor_type": "Tipsv2ImageProcessor", "resample": 2,
        "rescale_factor": 1.0 / 255.0, "size": {"height": _IMAGE, "width": _IMAGE},
    },
    "processor_class": "Tipsv2Processor",
}

_CONFIG = {
    "architectures": ["TIPSv2Model"],
    "model_type": "tipsv2",
    "temperature_init_value": 0.004638021811842918,
    "text_config": {
        "hidden_act": "relu", "hidden_size": _WIDTH, "intermediate_size": _INTERMEDIATE,
        "max_position_embeddings": _WINDOW, "num_attention_heads": _HEADS,
        "num_hidden_layers": _LAYERS, "pad_token_id": 0, "pooling_epsilon": 1e-08,
        "scale_sqrt_depth": True, "vocab_size": _VOCAB,
    },
    "vision_config": {
        "hidden_act": "gelu", "hidden_size": _WIDTH, "image_size": _IMAGE,
        "num_attention_heads": _HEADS, "num_channels": 3, "num_hidden_layers": _LAYERS,
        "num_register_tokens": 1, "patch_size": _PATCH, "use_swiglu_ffn": False,
    },
}


def _zeros(*shape) -> np.ndarray:
    """A float32 array of this shape that occupies one element of memory."""
    return np.broadcast_to(np.zeros((1,), dtype=np.float32), shape)


def _weights(**overrides) -> dict:
    weights = {
        "vision_encoder.cls_token": _zeros(1, 1, _WIDTH),
        "vision_encoder.register_tokens": _zeros(1, 1, _WIDTH),
        "vision_encoder.pos_embed": _zeros(1, _GRID * _GRID + 1, _WIDTH),
        "vision_encoder.patch_embed.proj.weight": _zeros(_WIDTH, 3, _PATCH, _PATCH),
        "text_encoder.token_embedding.weight": _zeros(_VOCAB, _WIDTH),
    }
    for index in range(_LAYERS):
        weights[f"vision_encoder.blocks.{index}.mlp.fc1.weight"] = _zeros(_INTERMEDIATE, _WIDTH)
        weights[f"vision_encoder.blocks.{index}.attn.qkv.weight"] = _zeros(3 * _WIDTH, _WIDTH)
        weights[f"vision_encoder.blocks.{index}.attn.qkv.bias"] = _zeros(3 * _WIDTH)
        weights[f"text_encoder.transformer.resblocks.{index}.mlp.c_fc.weight"] = _zeros(
            _INTERMEDIATE, _WIDTH)
        weights[f"text_encoder.transformer.resblocks.{index}.attn.in_proj_weight"] = _zeros(
            3 * _WIDTH, _WIDTH)
    weights.update(overrides)
    return weights


def _full_weights(**overrides) -> dict:
    """Every tensor the two graphs read, written out from the checkpoint's own layout.

    `_weights` has only what `config.resolve` reads. This is the fixture `model._require_weights`
    needs, and it is spelled out here rather than taken from `model._expected_shapes` so that the
    tests below check that map rather than agree with it.
    """
    weights = dict(_weights())
    weights["vision_encoder.patch_embed.proj.bias"] = _zeros(_WIDTH)
    weights["vision_encoder.norm.weight"] = _zeros(_WIDTH)
    weights["vision_encoder.norm.bias"] = _zeros(_WIDTH)
    weights["text_encoder.ln_final.weight"] = _zeros(_WIDTH)
    weights["text_encoder.ln_final.bias"] = _zeros(_WIDTH)
    for index in range(_LAYERS):
        block = f"vision_encoder.blocks.{index}"
        weights[f"{block}.norm1.weight"] = _zeros(_WIDTH)
        weights[f"{block}.norm1.bias"] = _zeros(_WIDTH)
        weights[f"{block}.norm2.weight"] = _zeros(_WIDTH)
        weights[f"{block}.norm2.bias"] = _zeros(_WIDTH)
        weights[f"{block}.attn.proj.weight"] = _zeros(_WIDTH, _WIDTH)
        weights[f"{block}.attn.proj.bias"] = _zeros(_WIDTH)
        weights[f"{block}.mlp.fc1.bias"] = _zeros(_INTERMEDIATE)
        weights[f"{block}.mlp.fc2.weight"] = _zeros(_WIDTH, _INTERMEDIATE)
        weights[f"{block}.mlp.fc2.bias"] = _zeros(_WIDTH)
        weights[f"{block}.ls1.gamma"] = _zeros(_WIDTH)
        weights[f"{block}.ls2.gamma"] = _zeros(_WIDTH)
        resblock = f"text_encoder.transformer.resblocks.{index}"
        weights[f"{resblock}.ln_1.weight"] = _zeros(_WIDTH)
        weights[f"{resblock}.ln_1.bias"] = _zeros(_WIDTH)
        weights[f"{resblock}.ln_2.weight"] = _zeros(_WIDTH)
        weights[f"{resblock}.ln_2.bias"] = _zeros(_WIDTH)
        weights[f"{resblock}.attn.in_proj_bias"] = _zeros(3 * _WIDTH)
        weights[f"{resblock}.attn.out_proj.weight"] = _zeros(_WIDTH, _WIDTH)
        weights[f"{resblock}.attn.out_proj.bias"] = _zeros(_WIDTH)
        weights[f"{resblock}.mlp.c_fc.bias"] = _zeros(_INTERMEDIATE)
        weights[f"{resblock}.mlp.c_proj.weight"] = _zeros(_WIDTH, _INTERMEDIATE)
        weights[f"{resblock}.mlp.c_proj.bias"] = _zeros(_WIDTH)
    weights.update(overrides)
    return weights


def _model_dir(tmp_path: Path, config: dict | None = None,
               processor: dict | None = None) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps(_CONFIG if config is None else config), encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(
        json.dumps(_PROCESSOR if processor is None else processor), encoding="utf-8")
    return tmp_path


def _without(section: str, key: str) -> dict:
    config = json.loads(json.dumps(_CONFIG))
    del config[section][key]
    return config


def _with(section: str, **changes) -> dict:
    config = json.loads(json.dumps(_CONFIG))
    config[section].update(changes)
    return config


# ─── config.resolve ───

def test_the_geometry_is_resolved(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    vision, text = cfg["vision"], cfg["text"]
    assert vision["factory"] == "vit_so400m"
    assert (vision["hidden_size"], vision["intermediate_size"]) == (_WIDTH, _INTERMEDIATE)
    assert (vision["num_layers"], vision["num_heads"], vision["head_dim"]) == (_LAYERS, _HEADS, 72)
    assert (vision["patch_size"], vision["grid"], vision["image_size"]) == (_PATCH, _GRID, _IMAGE)
    assert vision["num_positions"] == _GRID * _GRID + 1
    assert vision["num_register_tokens"] == 1
    assert vision["eps"] == 1e-6 and text["eps"] == 1e-5
    assert (text["vocab_size"], text["max_text_len"], text["pad_token_id"]) == (_VOCAB, _WINDOW, 0)
    assert cfg["preprocess"] == {"image_size": _IMAGE, "resample": 2,
                                 "rescale_factor": 1.0 / 255.0}
    assert cfg["embedding_dim"] == _WIDTH


def test_the_vision_head_count_comes_from_the_factory_not_the_file(tmp_path: Path) -> None:
    # The reference picks vit_so400m by (hidden_size, num_hidden_layers) and that factory hard
    # codes 16 heads; it never reads vision_config.num_attention_heads. A file that claims
    # another count describes a different model than the one that would be built, so it is
    # refused rather than believed or ignored.
    directory = _model_dir(tmp_path, _with("vision_config", num_attention_heads=12))
    with pytest.raises(ValueError, match="num_attention_heads"):
        config_module.resolve(directory, _weights())
    # With the claim absent the factory still supplies it.
    directory = _model_dir(tmp_path, _without("vision_config", "num_attention_heads"))
    assert config_module.resolve(directory, _weights())["vision"]["num_heads"] == _HEADS


def test_the_text_head_count_is_required(tmp_path: Path) -> None:
    # This one the reference does read, and no tensor shape carries it: an absent value would
    # silently become the reference's 12-head default on a 16-head checkpoint.
    directory = _model_dir(tmp_path, _without("text_config", "num_attention_heads"))
    with pytest.raises(ValueError, match="num_attention_heads"):
        config_module.resolve(directory, _weights())


@pytest.mark.parametrize("window", [2**31, 2**31 // _WIDTH + 1, 2**62])
def test_a_prompt_window_the_runtime_cannot_hold_is_refused(tmp_path: Path, window: int) -> None:
    """A JSON integer is unbounded; the field it ends up in is an int32.

    `max_position_embeddings` becomes the runtime's `max_text_len` and the length of the engine's
    computed sinusoid, `window x width` float32 entries. A config stating more than either can
    represent has to be refused where the file is still in hand - otherwise the first sign of it
    is a build that tries to materialise the table, or a bundle whose window wrapped on load.
    """
    directory = _model_dir(tmp_path, _with("text_config", max_position_embeddings=window))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        config_module.resolve(directory, _weights())

    # The largest window both can still represent builds, so the bound is the runtime's and not
    # a round number chosen for comfort.
    largest = (2**31 - 1) // _WIDTH
    directory = _model_dir(tmp_path, _with("text_config", max_position_embeddings=largest))
    assert config_module.resolve(directory, _weights())["text"]["max_text_len"] == largest


def test_an_unlisted_vision_geometry_is_refused(tmp_path: Path) -> None:
    # Only so400m is qualified. A B/14 or L/14 checkpoint has a different head count and MLP
    # ratio, which the same code would silently get wrong.
    config = _with("vision_config", hidden_size=1024, num_hidden_layers=24)
    weights = {key: value for key, value in _weights().items()
               if not key.startswith("vision_encoder.blocks.")}
    for index in range(24):
        weights[f"vision_encoder.blocks.{index}.mlp.fc1.weight"] = _zeros(4096, 1024)
        weights[f"vision_encoder.blocks.{index}.attn.qkv.weight"] = _zeros(3072, 1024)
        weights[f"vision_encoder.blocks.{index}.attn.qkv.bias"] = _zeros(3072)
    weights["vision_encoder.patch_embed.proj.weight"] = _zeros(1024, 3, _PATCH, _PATCH)
    weights["vision_encoder.cls_token"] = _zeros(1, 1, 1024)
    weights["vision_encoder.register_tokens"] = _zeros(1, 1, 1024)
    weights["vision_encoder.pos_embed"] = _zeros(1, _GRID * _GRID + 1, 1024)
    with pytest.raises(NotImplementedError, match="qualified for the vision geometries"):
        config_module.resolve(_model_dir(tmp_path, config), weights)


def test_an_mlp_width_the_factory_does_not_expect_is_refused(tmp_path: Path) -> None:
    # int(1152 * 3.736111...) is the number a resolver that recomputed the ratio would get, and
    # it is not necessarily 4304. The stored width decides.
    weights = _weights(**{"vision_encoder.blocks.0.mlp.fc1.weight": _zeros(4303, _WIDTH)})
    with pytest.raises(ValueError, match="expects an MLP"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_class_row_is_expected_in_the_position_table(tmp_path: Path) -> None:
    # 1024 positions is the bare patch grid, SigLIP's layout. TIPSv2 prepends a class token
    # before the table is added, so the table must have one more row than the grid.
    weights = _weights(**{"vision_encoder.pos_embed": _zeros(1, _GRID * _GRID, _WIDTH)})
    with pytest.raises(NotImplementedError, match="square patch grid"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_tanh_gelu_claim_is_refused(tmp_path: Path) -> None:
    # The vision MLP is nn.GELU() with approximate="none"; the tanh approximation is a different
    # function by about 1e-3.
    directory = _model_dir(tmp_path, _with("vision_config", hidden_act="gelu_pytorch_tanh"))
    with pytest.raises(NotImplementedError, match="error-function"):
        config_module.resolve(directory, _weights())


def test_a_non_relu_text_activation_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, _with("text_config", hidden_act="gelu"))
    with pytest.raises(NotImplementedError, match="relu"):
        config_module.resolve(directory, _weights())


def test_swiglu_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, _with("vision_config", use_swiglu_ffn=True))
    with pytest.raises(NotImplementedError, match="use_swiglu_ffn"):
        config_module.resolve(directory, _weights())


def test_scale_sqrt_depth_off_is_refused(tmp_path: Path) -> None:
    # Without it the embedding is not multiplied by sqrt(1152) and every row changes.
    directory = _model_dir(tmp_path, _with("text_config", scale_sqrt_depth=False))
    with pytest.raises(NotImplementedError, match="scale_sqrt_depth"):
        config_module.resolve(directory, _weights())


def test_the_dense_prediction_model_type_is_refused(tmp_path: Path) -> None:
    config = json.loads(json.dumps(_CONFIG))
    config["model_type"] = "tipsv2_dpt"
    with pytest.raises(NotImplementedError, match="tipsv2_dpt"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_an_unpacked_attention_projection_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"vision_encoder.blocks.0.attn.qkv.weight": _zeros(_WIDTH, _WIDTH)})
    with pytest.raises(ValueError, match="packed QKV"):
        config_module.resolve(_model_dir(tmp_path), weights)
    weights = _weights(**{
        "text_encoder.transformer.resblocks.0.attn.in_proj_weight": _zeros(_WIDTH, _WIDTH)})
    with pytest.raises(ValueError, match="packed QKV"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_missing_qkv_bias_is_refused(tmp_path: Path) -> None:
    weights = _weights()
    del weights["vision_encoder.blocks.0.attn.qkv.bias"]
    with pytest.raises(KeyError, match="QKV bias"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_non_contiguous_block_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights(**{f"vision_encoder.blocks.{_LAYERS + 3}.mlp.fc1.weight":
                          _zeros(_INTERMEDIATE, _WIDTH)})
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_non_canonical_block_index_is_refused(tmp_path: Path) -> None:
    # "00" and "+1" are integers to int() and would collapse onto an existing index.
    weights = _weights(**{"vision_encoder.blocks.00.mlp.fc1.weight": _zeros(_INTERMEDIATE,
                                                                           _WIDTH)})
    with pytest.raises(ValueError, match="canonical decimal"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_towers_of_different_widths_are_refused(tmp_path: Path) -> None:
    config = _with("text_config", hidden_size=768, intermediate_size=2048)
    weights = _weights(**{"text_encoder.token_embedding.weight": _zeros(_VOCAB, 768)})
    for index in range(_LAYERS):
        weights[f"text_encoder.transformer.resblocks.{index}.mlp.c_fc.weight"] = _zeros(2048, 768)
        weights[f"text_encoder.transformer.resblocks.{index}.attn.in_proj_weight"] = _zeros(
            3 * 768, 768)
    with pytest.raises(ValueError, match="share a width"):
        config_module.resolve(_model_dir(tmp_path, config), weights)


def test_a_pad_id_outside_the_table_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, _with("text_config", pad_token_id=_VOCAB))
    with pytest.raises(ValueError, match="outside the"):
        config_module.resolve(directory, _weights())


@pytest.mark.parametrize("temperature", [0.0, -1.0, "warm", True])
def test_an_unusable_temperature_is_refused(tmp_path: Path, temperature) -> None:
    config = json.loads(json.dumps(_CONFIG))
    config["temperature_init_value"] = temperature
    with pytest.raises(ValueError, match="temperature"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_normalisation_is_refused(tmp_path: Path) -> None:
    # TIPSv2 takes pixels in [0, 1]. A checkpoint that asks for an ImageNet mean would move every
    # pixel and nothing loudly.
    processor = json.loads(json.dumps(_PROCESSOR))
    processor["image_processor"]["do_normalize"] = True
    with pytest.raises(NotImplementedError, match="do_normalize"):
        config_module.resolve(_model_dir(tmp_path, None, processor), _weights())


def test_a_non_bilinear_resample_is_refused(tmp_path: Path) -> None:
    processor = json.loads(json.dumps(_PROCESSOR))
    processor["image_processor"]["resample"] = 3
    with pytest.raises(NotImplementedError, match="bilinear"):
        config_module.resolve(_model_dir(tmp_path, None, processor), _weights())


def test_a_processor_size_the_table_cannot_cover_is_refused(tmp_path: Path) -> None:
    processor = json.loads(json.dumps(_PROCESSOR))
    processor["image_processor"]["size"] = {"height": 224, "width": 224}
    with pytest.raises(NotImplementedError, match="fixed at the checkpoint"):
        config_module.resolve(_model_dir(tmp_path, None, processor), _weights())


# ─── build request validation ───

def _request(tmp_path: Path, **overrides) -> BuildRequest:
    fields = dict(model_dir=tmp_path, output_path=tmp_path / "tipsv2.bundle", family="tipsv2",
                  task="text_image_to_relevance", precision="fp32")
    fields.update(overrides)
    return BuildRequest(**fields)


@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8", "fp64"])
def test_an_unqualified_precision_is_refused(tmp_path: Path, precision: str) -> None:
    with pytest.raises(NotImplementedError, match="not qualified"):
        model_module.build(_request(tmp_path, precision=precision), writer=None)


def test_quantization_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="no quantization"):
        model_module.build(_request(tmp_path, quantization="fp8"), writer=None)


def test_an_unsupported_task_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supports only task"):
        model_module.build(_request(tmp_path, task="image_to_class_scores"), writer=None)


# ─── the computed position signal ───

def test_the_position_signal_is_sines_then_cosines() -> None:
    signal = graph.sinusoidal_position_signal(_WINDOW, _WIDTH)
    assert signal.shape == (_WINDOW, _WIDTH)
    assert signal.dtype == np.float32
    half = _WIDTH // 2
    # Row 0 is sin(0) for the first half and cos(0) for the second. An interleaved layout - the
    # other common convention - would put ones in every second column instead.
    assert np.allclose(signal[0, :half], 0.0, atol=0.0)
    assert np.allclose(signal[0, half:], 1.0, atol=0.0)
    # Channel 0 has the fastest timescale: sin(position).
    assert np.allclose(signal[:, 0], np.sin(np.arange(_WINDOW, dtype=np.float32)), atol=1e-6)
    assert np.allclose(signal[:, half], np.cos(np.arange(_WINDOW, dtype=np.float32)), atol=1e-6)
    # The slowest timescale is 1 / 10000, so the last sine channel barely moves over 64 rows.
    assert abs(float(signal[_WINDOW - 1, half - 1])) < 1e-2
    assert np.isfinite(signal).all()


def test_an_odd_width_position_signal_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="even width"):
        graph.sinusoidal_position_signal(8, 7)


def test_the_attention_mask_value_is_finite_in_its_own_dtype() -> None:
    # float("-inf") is what the reference writes; in a half-precision graph that value multiplied
    # by a zero weight is a NaN, so each dtype gets a finite stand-in it can represent.
    for dtype in (np.float32, np.float16):
        value = graph.mask_additive_value(dtype)
        assert np.isfinite(np.array(value, dtype=dtype))
        assert np.array(value, dtype=dtype) < -1e3
    with pytest.raises(NotImplementedError):
        graph.mask_additive_value(np.float64)


# ─── the tokenizer asset ───

def _asset(**overrides) -> tx.TokenizerAsset:
    pieces = [("<pad>", 0.0, tx.PIECE_CONTROL), ("</s>", 0.0, tx.PIECE_CONTROL),
              ("<unk>", 0.0, tx.PIECE_UNKNOWN), ("<s>", 0.0, tx.PIECE_CONTROL)]
    pieces += [(f"<0x{value:02X}>", 0.0, tx.PIECE_BYTE) for value in range(256)]
    pieces += [("▁", -1.0, tx.PIECE_NORMAL), ("a", -2.0, tx.PIECE_NORMAL)]
    fields = dict(unknown_id=2, bos_id=3, eos_id=1, pad_id=0, pieces=pieces,
                  charsmap=b"\x04\x00\x00\x00\x00\x00\x00\x00", lowercase=[(0x41, "a")],
                  cased=[(0x41, 0x5A)], case_ignorable=[(0x300, 0x36F)])
    fields.update(overrides)
    return tx.TokenizerAsset(**fields)


def _model_proto(**trainer_overrides):
    """A miniature `tokenizer.model`: the smallest ModelProto `convert_tokenizer_model` accepts.

    Reading a real `tokenizer.model` is the one step of the build that needs protobuf, and the
    generated code sentencepiece ships refuses an older runtime than the one it was generated
    for. Building the proto here rather than parsing a recorded file keeps that dependency
    exercised on every CPU run - the family's `requirements.txt` has to pin it - and covers the
    conversion's own refusals, which no other test reaches.
    """
    from sentencepiece import sentencepiece_model_pb2 as model_pb2

    model = model_pb2.ModelProto()
    for piece, score, kind in _asset().pieces:
        entry = model.pieces.add()
        entry.piece, entry.score, entry.type = piece, score, kind
    model.trainer_spec.model_type = model_pb2.TrainerSpec.ModelType.Value("BPE")
    model.trainer_spec.vocab_size = len(model.pieces)
    model.trainer_spec.byte_fallback = True
    model.trainer_spec.unk_id, model.trainer_spec.bos_id = 2, 3
    model.trainer_spec.eos_id, model.trainer_spec.pad_id = 1, 0
    model.normalizer_spec.name = "nmt_nfkc"
    model.normalizer_spec.add_dummy_prefix = True
    model.normalizer_spec.remove_extra_whitespaces = True
    model.normalizer_spec.escape_whitespaces = True
    model.normalizer_spec.precompiled_charsmap = b"\x04\x00\x00\x00\x00\x00\x00\x00"
    for name, value in trainer_overrides.items():
        setattr(model.trainer_spec, name, value)
    return model


def test_a_tokenizer_model_becomes_the_asset_the_runtime_reads() -> None:
    model = _model_proto()
    data, asset = tx.convert_tokenizer_model(model.SerializeToString())

    assert [piece.piece for piece in model.pieces] == [piece for piece, _, _ in asset.pieces]
    assert (asset.unknown_id, asset.bos_id, asset.eos_id, asset.pad_id) == (2, 3, 1, 0)
    assert asset.charsmap == bytes(model.normalizer_spec.precompiled_charsmap)
    assert asset.byte_fallback and asset.add_dummy_prefix and asset.escape_whitespaces
    # The conversion carries the tables the proto knows nothing about: they come from CPython.
    assert asset.lowercase == tx.lowercase_table() and asset.lowercase
    assert asset.cased and asset.case_ignorable
    assert tx.parse_tokenizer_asset(data).pieces == asset.pieces


def test_a_unigram_tokenizer_model_is_refused() -> None:
    from sentencepiece import sentencepiece_model_pb2 as model_pb2

    model = _model_proto()
    model.trainer_spec.model_type = model_pb2.TrainerSpec.ModelType.Value("UNIGRAM")
    with pytest.raises(NotImplementedError, match="BPE"):
        tx.convert_tokenizer_model(model.SerializeToString())


def test_a_tokenizer_model_without_a_charsmap_is_refused() -> None:
    model = _model_proto()
    model.normalizer_spec.precompiled_charsmap = b""
    with pytest.raises(NotImplementedError, match="charsmap"):
        tx.convert_tokenizer_model(model.SerializeToString())


def test_a_tokenizer_model_with_a_normalization_rule_is_refused() -> None:
    model = _model_proto()
    model.normalizer_spec.normalization_rule_tsv = "rules.tsv"
    with pytest.raises(NotImplementedError, match="normalization_rule_tsv"):
        tx.convert_tokenizer_model(model.SerializeToString())


def test_a_vocab_size_that_disagrees_with_the_pieces_is_refused() -> None:
    model = _model_proto()
    model.trainer_spec.vocab_size = len(model.pieces) + 1
    with pytest.raises(ValueError, match="vocab_size"):
        tx.convert_tokenizer_model(model.SerializeToString())


@pytest.mark.parametrize("charsmap,reason", [
    (b"", "too short"),
    (b"\x04\x00\x00", "too short"),
    (b"\x00\x00\x00\x00abcd", "multiple of four"),
    (b"\x05\x00\x00\x00abcde", "multiple of four"),
    (b"\x08\x00\x00\x00abcd", "carries only"),
])
def test_a_charsmap_the_runtime_could_not_load_is_refused(charsmap: bytes, reason: str) -> None:
    """A charsmap is copied into the asset verbatim and read again only when the bundle opens.

    These are `PrecompiledCharsMap::load`'s own three conditions. Without them here a truncated
    or misaligned map would build, ship and then refuse to open, with nothing between the
    checkpoint and the failure.
    """
    with pytest.raises(ValueError, match=reason):
        tx.serialize_tokenizer_asset(_asset(charsmap=charsmap))


def test_the_asset_survives_its_own_round_trip() -> None:
    asset = _asset()
    parsed = tx.parse_tokenizer_asset(tx.serialize_tokenizer_asset(asset))
    assert parsed.pieces == asset.pieces
    assert parsed.charsmap == asset.charsmap
    assert parsed.lowercase == asset.lowercase
    assert parsed.cased == asset.cased
    assert parsed.case_ignorable == asset.case_ignorable
    assert (parsed.unknown_id, parsed.bos_id, parsed.eos_id, parsed.pad_id) == (2, 3, 1, 0)
    assert parsed.byte_fallback and parsed.add_dummy_prefix and parsed.escape_whitespaces


def test_an_empty_piece_is_refused() -> None:
    # It would match at every position without advancing.
    asset = _asset()
    asset.pieces.append(("", -3.0, tx.PIECE_NORMAL))
    with pytest.raises(ValueError, match="is empty"):
        tx.serialize_tokenizer_asset(asset)


def test_duplicate_pieces_are_refused() -> None:
    asset = _asset()
    asset.pieces.append(("a", -3.0, tx.PIECE_NORMAL))
    with pytest.raises(ValueError, match="both"):
        tx.serialize_tokenizer_asset(asset)


def test_two_unknown_pieces_are_refused() -> None:
    asset = _asset()
    asset.pieces.append(("<unk2>", -3.0, tx.PIECE_UNKNOWN))
    with pytest.raises(ValueError, match="exactly one unknown piece"):
        tx.serialize_tokenizer_asset(asset)


def test_byte_fallback_without_every_byte_is_refused() -> None:
    asset = _asset()
    asset.pieces = [piece for piece in asset.pieces if piece[0] != "<0x41>"]
    with pytest.raises(ValueError, match="one piece per byte"):
        tx.serialize_tokenizer_asset(asset)


def test_a_padding_id_that_is_not_a_control_piece_is_refused() -> None:
    # The text engine derives its mask from this id, so a prompt must not be able to produce it.
    with pytest.raises(ValueError, match="padding id"):
        tx.serialize_tokenizer_asset(_asset(pad_id=260))


def test_special_ids_that_collide_are_refused() -> None:
    with pytest.raises(ValueError, match="not distinct"):
        tx.serialize_tokenizer_asset(_asset(eos_id=0))


def test_an_unsorted_lowercase_table_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        tx.serialize_tokenizer_asset(_asset(lowercase=[(0x42, "b"), (0x41, "a")]))


def test_a_lowercase_entry_mapping_to_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="to nothing"):
        tx.serialize_tokenizer_asset(_asset(lowercase=[(0x41, "")]))


def test_trailing_bytes_are_refused() -> None:
    data = tx.serialize_tokenizer_asset(_asset())
    with pytest.raises(ValueError, match="trailing bytes"):
        tx.parse_tokenizer_asset(data + b"\x00")
    with pytest.raises(ValueError, match="truncated"):
        tx.parse_tokenizer_asset(data[:-1])


def test_the_case_tables_reproduce_this_interpreter_s_final_sigma() -> None:
    """U+03A3 is the only character in Unicode whose lower case depends on its neighbours.

    ``str.lower()`` gives it the final form U+03C2 when a cased character precedes it and none
    follows, skipping case-ignorable characters in both directions, so ``"ΟΣ".lower()`` is
    ``"ος"``. The bundle carries the two property sets rather than the answer, and this replays
    CPython's walk over them. A per-code-point table alone - the obvious implementation - fails
    every case below that ends in a sigma.
    """
    cased, ignorable = tx.case_tables()
    cased_starts = [first for first, _ in cased]
    ignorable_starts = [first for first, _ in ignorable]

    def member(ranges, starts, code_point: int) -> bool:
        index = bisect.bisect_right(starts, code_point)
        return index > 0 and code_point <= ranges[index - 1][1]

    table = dict(tx.lowercase_table())

    def lowered(text: str) -> str:
        out = []
        for index, character in enumerate(text):
            if ord(character) != tx.CAPITAL_SIGMA:
                out.append(table.get(ord(character), character))
                continue
            back = index - 1
            while back >= 0 and member(ignorable, ignorable_starts, ord(text[back])):
                back -= 1
            final = back >= 0 and member(cased, cased_starts, ord(text[back]))
            if final:
                forward = index + 1
                while forward < len(text) and member(ignorable, ignorable_starts, ord(text[forward])):
                    forward += 1
                final = forward == len(text) or not member(cased, cased_starts, ord(text[forward]))
            out.append(tx.FINAL_SIGMA if final else tx.SMALL_SIGMA)
        return "".join(out)

    assert lowered("ΟΣ") == "ος" != "οσ"
    assert lowered("ΣΟ") == "σο"
    assert lowered("ΟΣΟ") == "οσο"
    assert lowered("ΟΣ́") == "ος́"  # a combining mark is case-ignorable
    assert lowered("ΟΣ!") == "ος!"  # punctuation is neither, so the sigma is still final

    rng = random.Random(290)
    alphabet = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψως" + " .,!?-'’́­"
    for _ in range(5000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        assert lowered(text) == text.lower(), repr(text)


def _collection_hook():
    """`conftest.py` is loaded by pytest, not imported; this reads it as a module."""
    path = Path(__file__).resolve().parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("tipsv2_tests_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pytest_collection_modifyitems


def _item(function_name: str, case_name: str, file_name: str = "test_e2e.py"):
    return SimpleNamespace(
        originalname=function_name,
        nodeid=f"families/tipsv2/tests/{file_name}::{function_name}[{case_name}]",
        callspec=SimpleNamespace(params={"case_name": case_name}))


def _config(testcases):
    options = {"--e2e-testcase": testcases, "--e2e-model": [], "--e2e-models-file": None}
    return SimpleNamespace(getoption=lambda name, default=None: options.get(name, default))


def test_only_a_real_e2e_function_satisfies_the_testcase_guard() -> None:
    """What makes an item the E2E is **the function's name**, not the file it sits in.

    The guard exists to catch `--e2e-testcase <typo>`, a selection that makes every family
    deselect itself so the run passes having tested nothing. Matching the file instead would let
    any parametrization inside `test_e2e.py` satisfy it - and would reject a case another family
    legitimately collected from a file named something else.
    """
    hook = _collection_hook()
    e2e = _item("test_official_checkpoint_e2e", "tipsv2-so400m14")
    # A unit test parametrized with a `case_name`, living in the same file.
    neighbour = _item("test_some_unit_case", "a-unit-fixture")

    hook(None, _config(["tipsv2-so400m14"]), [e2e, neighbour])  # the real case: quiet

    with pytest.raises(pytest.UsageError, match="matches none of the E2E cases"):
        hook(None, _config(["a-unit-fixture"]), [e2e, neighbour])


def test_the_testcase_guard_never_judges_another_family(monkeypatch) -> None:
    """A family may only judge its own collection.

    A sibling is free to parametrize on something other than `case_name` - `families/clef` and
    `families/laya` parametrize on the manifest and the case - so its cases are invisible here.
    Aborting the session on that would make this family's presence break a selection that is
    correct for the family owning it, which is what family isolation forbids.
    """
    hook = _collection_hook()
    mine = _item("test_official_checkpoint_e2e", "tipsv2-so400m14")
    # `families/clef` and `families/laya` parametrize on `(manifest, case)` with
    # `id=case["name"]`, so the name is in the parameter id and not in a `case_name` parameter.
    sibling = SimpleNamespace(
        originalname="test_e2e",
        nodeid="families/clef/tests/test_e2e.py::test_e2e[a-clef-case]",
        callspec=SimpleNamespace(params={"manifest": object(), "case": object()}))

    # The sibling's name is in its parameter id even though `case_name` is not, so selecting it
    # is a correct run and must stay quiet.
    hook(None, _config(["a-clef-case"]), [mine, sibling])

    # With this family alone, the same selection is the typo the guard exists for.
    with pytest.raises(pytest.UsageError, match="matches none of the E2E cases"):
        hook(None, _config(["a-clef-case"]), [mine])

    # And a sibling in the session does **not** switch the guard off: a name nobody collected is
    # still a typo, which is what the first attempt at this exemption got wrong.
    with pytest.raises(pytest.UsageError, match="matches none of the E2E cases"):
        hook(None, _config(["neither-families-case"]), [mine, sibling])


def test_an_empty_parametrization_contributes_no_selectable_name() -> None:
    """The leftover of a family that deselected itself is not a case anyone can ask for.

    pytest spells that item's id differently by version - `NOTSET` on 8.x, `case_name0` on the
    7.3 the test extra still allows - so the id cannot identify it and a check that read the id
    would let `--e2e-testcase case_name0` satisfy itself on 7.3 while running nothing. The
    parameter *values* identify it on every version: pytest fills them with its own sentinel.
    """
    hook = _collection_hook()

    class NotSetType:                      # what pytest puts in the params of an empty set
        def __repr__(self):
            return "<NOTSET>"

    for identifier in ("NOTSET", "case_name0"):
        placeholder = SimpleNamespace(
            originalname="test_official_checkpoint_e2e",
            nodeid=f"families/tipsv2/tests/test_e2e.py::test_official_checkpoint_e2e[{identifier}]",
            callspec=SimpleNamespace(params={"case_name": NotSetType()}))
        with pytest.raises(pytest.UsageError, match="matches none of the E2E cases"):
            hook(None, _config([identifier]), [placeholder])


@pytest.mark.parametrize("value", [448.5, 448.0, "448", None])
def test_a_build_option_that_is_not_an_integer_is_refused(value) -> None:
    """``int(448.5)`` is 448: a size this family cannot serve, truncated into the one it can.

    Every number read from the checkpoint's own files is refused when it is not a real integer;
    an option from the caller is no different, and silently building something other than what
    was asked for is the worst of the three outcomes.
    """
    if value is None:  # the one value that means "not given"
        assert model_module._require_integer(None, "image_height") is None
        return
    with pytest.raises(NotImplementedError, match="must be an integer"):
        model_module._require_integer(value, "image_height")


def test_a_boolean_is_not_an_integer_here() -> None:
    # `isinstance(True, int)` is true in Python, and `image_height=True` would become 1.
    with pytest.raises(NotImplementedError, match="must be an integer"):
        model_module._require_integer(True, "image_height")
    assert model_module._require_integer(448, "image_height") == 448


def test_case_table_ranges_must_be_sorted_and_separated() -> None:
    # The runtime binary-searches them; out-of-order or touching ranges would answer wrongly.
    with pytest.raises(ValueError, match="sorted and separated"):
        tx.serialize_tokenizer_asset(_asset(cased=[(0x41, 0x5A), (0x40, 0x42)]))


@pytest.mark.parametrize("empty", [["cased"], ["case_ignorable"], ["cased", "case_ignorable"]])
def test_the_asset_needs_both_case_tables(monkeypatch, empty: list) -> None:
    """The two tables answer one question between them, and neither alone - nor neither at all.

    Whether a capital sigma is word-final is decided by both: a cased character after it, with
    only case-ignorable ones in between. Any other combination gives **every** sigma the same
    form, silently, and changes the ids of every Greek word that ends in one. Both empty is the
    case that reads like "no case rules configured" and is in fact the rule switched off.
    """
    fields = {name: [] for name in empty}
    with pytest.raises(ValueError, match="both case tables"):
        tx.serialize_tokenizer_asset(_asset(**fields))

    # And reading it back, the way the loader does: build the bytes with the check suspended,
    # then parse them. A bundle from an older build is checked, not trusted.
    monkeypatch.setattr(tx, "_validate_case_tables", lambda _asset: None)
    data = tx.serialize_tokenizer_asset(_asset(**fields))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="both case tables"):
        tx.parse_tokenizer_asset(data)


def test_the_lowercase_table_is_this_interpreter_s_own() -> None:
    """Every code point whose ``str.lower()`` differs, and no others.

    The checkpoint's tokenizer lowercases with Python before SentencePiece normalises, so the
    bundle carries the mapping of the interpreter that built it rather than an ASCII fold. This
    fails on a table built from ``str.casefold()`` (which maps U+00DF to "ss") or from an
    ASCII-only rule.
    """
    table = tx.lowercase_table()
    assert dict(table) == {code: chr(code).lower() for code in range(sys.maxunicode + 1)
                           if chr(code).lower() != chr(code)}
    assert (0x0130, "i̇") in table  # the one 1:N mapping
    assert (0x00DF, "ss") not in table  # casefold's, not lower's
    assert len(table) > 1000


# ─── the recorded Pillow output ───

def test_the_recorded_pillow_values_match_the_installed_pillow() -> None:
    """Tie `tests/data/pillow_bilinear.json` to the Pillow `requirements.txt` pins.

    The C++ resize reproduces one Pillow version coefficient for coefficient, and Pillow 13
    changed which axis it resamples first - worth up to one 8-bit level per sample. That is too
    small to fail an embedding gate and large enough to be a real difference, so the drift is
    caught here instead.
    """
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    document = json.loads(
        (Path(__file__).resolve().parent / "data" / "pillow_bilinear.json").read_text("utf-8"))
    assert PIL.__version__ == document["pillow_version"], (
        f"tests/data/pillow_bilinear.json was recorded with Pillow "
        f"{document['pillow_version']} but Pillow {PIL.__version__} is installed; "
        "families/tipsv2/requirements.txt pins the version the C++ resize reproduces")
    for case in document["cases"]:
        shape = (case["input_height"], case["input_width"], 3)
        pixels = np.asarray(case["pixels"], dtype=np.uint8).reshape(shape)
        resized = np.asarray(
            Image.fromarray(pixels, "RGB").resize(
                (case["output_width"], case["output_height"]), Image.Resampling.BILINEAR),
            dtype=np.uint8)
        expected = np.asarray(case["expected"], dtype=np.uint8).reshape(
            case["output_height"], case["output_width"], 3)
        assert np.array_equal(resized, expected), case["note"]


# ─── values that are finite in float64 and unusable in float32 ───

@pytest.mark.parametrize("rescale", [1e38, 1e-50, 1.0])
def test_a_rescale_factor_that_is_not_a_usable_float32_is_refused(tmp_path: Path, rescale) -> None:
    """The runtime multiplies an 8-bit sample by this in float32.

    1e38 sends a white pixel to an infinity, 1e-50 narrows to exactly zero, and 1.0 leaves the
    pixels in [0, 255] instead of [0, 1]. None of them is caught by a float64 finiteness check.
    """
    processor = json.loads(json.dumps(_PROCESSOR))
    processor["image_processor"]["rescale_factor"] = rescale
    with pytest.raises(ValueError, match="rescale_factor"):
        config_module.resolve(_model_dir(tmp_path, None, processor), _weights())


def test_a_pooling_epsilon_that_narrows_to_zero_is_refused(tmp_path: Path) -> None:
    # The pooled average divides by `count + epsilon` in float32; an epsilon that becomes zero
    # lets an all-padding prompt divide zero by zero.
    directory = _model_dir(tmp_path, _with("text_config", pooling_epsilon=1e-50))
    with pytest.raises(ValueError, match="pooling_epsilon"):
        config_module.resolve(directory, _weights())


def test_a_temperature_that_overflows_when_inverted_is_refused(tmp_path: Path) -> None:
    config = json.loads(json.dumps(_CONFIG))
    config["temperature_init_value"] = 1e-45  # finite as a float32, but 1 / it is not
    with pytest.raises(ValueError, match="temperature"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


@pytest.mark.parametrize("claim", [1152.9, "1152", 1152.0])
def test_a_non_integer_claim_is_refused(tmp_path: Path, claim) -> None:
    # int() would accept all three and then agree with a model that is none of them.
    directory = _model_dir(tmp_path, _with("vision_config", hidden_size=claim))
    with pytest.raises(ValueError, match="must be an integer"):
        config_module.resolve(directory, _weights())


def test_an_explicit_null_claim_is_refused(tmp_path: Path) -> None:
    # `null` is a claim of nothing, not an absent key.
    directory = _model_dir(tmp_path, _with("vision_config", patch_size=None))
    with pytest.raises(ValueError, match="null"):
        config_module.resolve(directory, _weights())


# ─── every tensor's shape, not only the ones config.resolve reads ───

def test_a_later_block_with_a_different_mlp_width_is_refused(tmp_path: Path) -> None:
    # config.resolve reads block 0; without this, block 13 could be a different model.
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    weights = _full_weights(**{"vision_encoder.blocks.13.mlp.fc1.weight": _zeros(4096, _WIDTH)})
    with pytest.raises(ValueError, match="expected"):
        model_module._require_weights(weights, cfg)


def test_a_scalar_bias_is_refused(tmp_path: Path) -> None:
    # A one-element bias broadcasts over the row and builds a graph that is quietly wrong.
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    weights = _full_weights(**{"text_encoder.ln_final.bias": _zeros(1)})
    with pytest.raises(ValueError, match="expected"):
        model_module._require_weights(weights, cfg)


def test_a_non_float32_tensor_is_refused(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    weights = _full_weights(**{"vision_encoder.cls_token":
                               np.zeros((1, 1, _WIDTH), dtype=np.float16)})
    with pytest.raises(NotImplementedError, match="float32 checkpoint"):
        model_module._require_weights(weights, cfg)


def test_a_non_finite_tensor_is_refused(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    broken = np.zeros((1, 1, _WIDTH), dtype=np.float32)
    broken[0, 0, 7] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        model_module._require_weights(_full_weights(**{"vision_encoder.cls_token": broken}), cfg)


def test_a_complete_checkpoint_passes_the_weight_check(tmp_path: Path) -> None:
    # The negative cases above only mean something if the baseline fixture passes.
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    model_module._require_weights(_full_weights(), cfg)


def test_an_absent_boolean_claim_is_accepted_and_a_null_one_is_not(tmp_path: Path) -> None:
    # `_ABSENT` and `None` are different answers: a checkpoint may leave a claim out, and one
    # that writes `null` is making a claim of nothing.
    processor = json.loads(json.dumps(_PROCESSOR))
    del processor["image_processor"]["do_convert_rgb"]
    config_module.resolve(_model_dir(tmp_path, None, processor), _weights())
    processor["image_processor"]["do_convert_rgb"] = None
    with pytest.raises(ValueError, match="null"):
        config_module.resolve(_model_dir(tmp_path, None, processor), _weights())


def test_a_range_endpoint_past_unicode_is_refused() -> None:
    # `not 0 <= first and last <= maxunicode` binds as `(not (0 <= first)) and (...)`, which lets
    # this through and produces an asset the runtime then refuses.
    with pytest.raises(ValueError, match="outside Unicode"):
        tx.serialize_tokenizer_asset(_asset(cased=[(0x41, 0x110000)]))


class _RecordedTensor:
    """A tensor in the recording network below: a value, a dtype and a shape."""

    def __init__(self, values, dtype):
        self.values = np.asarray(values)
        self.dtype = dtype

    @property
    def shape(self):
        return self.values.shape


class _RecordingNetwork:
    """Enough of ``trt.INetworkDefinition`` to run one elementwise expression and watch it.

    ``graph.add_gelu_exact`` is the production code; this executes the layers it asks for, in
    NumPy, and records which ones they were. A test against a second implementation of the same
    formula would pass with the production clamp deleted.
    """

    def __init__(self):
        self.operations = []

    @staticmethod
    def _layer(tensor):
        return type("Layer", (), {"get_output": staticmethod(lambda _index: tensor)})

    @staticmethod
    def _require_same_dtype(left, right, what):
        # The networks this family builds are STRONGLY_TYPED: TensorRT refuses an operation whose
        # operands disagree on dtype, and a constant built in the wrong one is the easy way to
        # get there. The recorder enforces the same rule so that every test below checks it.
        if left.dtype != right.dtype:
            raise AssertionError(
                f"{what} mixes {left.dtype} and {right.dtype}; a strongly typed network "
                "refuses that")

    _TRT_DTYPES = {np.dtype(np.float16): trt.float16, np.dtype(np.float32): trt.float32,
                   np.dtype(np.int32): trt.int32, np.dtype(bool): trt.bool}

    def add_constant(self, shape, weights):
        values = np.asarray(weights).reshape(shape)
        # The recorder has to carry the **declared** dtype, not just the storage: the text tower
        # compares ids against an int32 constant, and a recorder that called that float would
        # report a strongly-typed violation the engine never sees.
        dtype = self._TRT_DTYPES.get(values.dtype, trt.float32)
        return self._layer(_RecordedTensor(values, dtype))

    def add_cast(self, tensor, dtype):
        target = np.float32 if dtype == trt.float32 else np.float16
        self.operations.append(("cast", dtype))
        return self._layer(_RecordedTensor(tensor.values.astype(target), dtype))

    def add_elementwise(self, left, right, operation):
        kinds = {
            trt.ElementWiseOperation.SUM: np.add,
            trt.ElementWiseOperation.PROD: np.multiply,
            trt.ElementWiseOperation.MAX: np.maximum,
            trt.ElementWiseOperation.MIN: np.minimum,
            trt.ElementWiseOperation.SUB: np.subtract,
            trt.ElementWiseOperation.DIV: np.divide,
            trt.ElementWiseOperation.EQUAL: np.equal,
        }
        self._require_same_dtype(left, right, f"elementwise {operation}")
        self.operations.append((operation, float(np.ravel(right.values)[0])
                                if right.values.size == 1 else None))
        # An engine overflows to an infinity without saying so, which is the behaviour several of
        # the tests below exist to pin down; NumPy would warn instead.
        with np.errstate(invalid="ignore", over="ignore"):
            values = kinds[operation](left.values, right.values)
        # A comparison answers BOOL whatever it compared; everything else keeps the operand type.
        dtype = trt.bool if operation == trt.ElementWiseOperation.EQUAL else left.dtype
        return self._layer(_RecordedTensor(values, dtype))

    def add_unary(self, tensor, operation):
        self.operations.append((operation, None))
        with np.errstate(invalid="ignore", over="ignore"):
            if operation == trt.UnaryOperation.ERF:
                erf = np.vectorize(math.erf, otypes=[np.float64])
                values = erf(tensor.values.astype(np.float64))
            elif operation == trt.UnaryOperation.SQRT:
                values = np.sqrt(tensor.values)
            elif operation == trt.UnaryOperation.NOT:
                values = np.logical_not(tensor.values)
            else:
                raise AssertionError(f"the recorder does not implement {operation}")
        return self._layer(_RecordedTensor(values, tensor.dtype))

    def add_activation(self, tensor, activation):
        assert activation == trt.ActivationType.RELU, activation
        self.operations.append(("activation", activation))
        return self._layer(_RecordedTensor(np.maximum(tensor.values, 0), tensor.dtype))

    def add_gather(self, data, indices, axis):
        self.operations.append(("gather", axis))
        values = np.take(data.values, np.asarray(indices.values, dtype=np.int64), axis=axis)
        return self._layer(_RecordedTensor(values, data.dtype))

    def add_select(self, condition, then_branch, else_branch):
        # TensorRT broadcasts the three operands; the condition is BOOL and the two branches must
        # agree on dtype, as they would in the engine.
        self._require_same_dtype(then_branch, else_branch, "select branches")
        self.operations.append(("select", None))
        values = np.where(np.asarray(condition.values, dtype=bool), then_branch.values,
                          else_branch.values)
        return self._layer(_RecordedTensor(values.astype(then_branch.values.dtype),
                                           then_branch.dtype))

    def add_reduce(self, tensor, operation, axes, keep_dims):
        self.operations.append(("reduce", operation))
        axis = int(np.log2(axes))
        with np.errstate(invalid="ignore", over="ignore"):
            if operation == trt.ReduceOperation.AVG:
                values = tensor.values.mean(axis=axis, keepdims=keep_dims)
            else:
                values = tensor.values.sum(axis=axis, keepdims=keep_dims)
        return self._layer(_RecordedTensor(values, tensor.dtype))

    def add_shuffle(self, tensor):
        network = self

        class Shuffle:
            # TensorRT applies first_transpose, then reshape_dims, then second_transpose, and the
            # caller sets them after this returns - so the work happens in get_output.
            first_transpose = None
            reshape_dims = None
            second_transpose = None

            @staticmethod
            def get_output(_index):
                values = tensor.values
                if Shuffle.first_transpose is not None:
                    values = np.transpose(values, Shuffle.first_transpose)
                if Shuffle.reshape_dims is not None:
                    values = values.reshape(Shuffle.reshape_dims)
                if Shuffle.second_transpose is not None:
                    values = np.transpose(values, Shuffle.second_transpose)
                network.operations.append(("shuffle", None))
                return _RecordedTensor(values, tensor.dtype)

        return Shuffle

    def add_matrix_multiply(self, left, left_op, right, right_op):
        self._require_same_dtype(left, right, "matmul")
        self.operations.append(("matmul", None))
        first = np.swapaxes(left.values, -1, -2) \
            if left_op == trt.MatrixOperation.TRANSPOSE else left.values
        second = np.swapaxes(right.values, -1, -2) \
            if right_op == trt.MatrixOperation.TRANSPOSE else right.values
        with np.errstate(invalid="ignore", over="ignore"):
            return self._layer(_RecordedTensor(first @ second, left.dtype))

    def add_softmax(self, tensor):
        network = self

        class Softmax:
            axes = None

            @staticmethod
            def get_output(_index):
                axis = int(np.log2(Softmax.axes))
                network.operations.append(("softmax", axis))
                with np.errstate(invalid="ignore", over="ignore"):
                    shifted = tensor.values - tensor.values.max(axis=axis, keepdims=True)
                    weights = np.exp(shifted)
                    return _RecordedTensor(weights / weights.sum(axis=axis, keepdims=True),
                                           tensor.dtype)

        return Softmax


def _gelu_through_the_builder(values, dtype=np.float32, build=None):
    """Run ``graph.add_gelu_exact`` over ``values``.

    ``dtype`` is the tensor the layer is handed; ``build`` is the precision the engine is being
    built at, which is what decides the clamp. They differ only where the test wants to probe
    FP16's clamp with values FP16 could not hold.
    """
    network = _RecordingNetwork()
    source = _RecordedTensor(np.asarray(values, dtype=np.float32),
                             trt.float32 if dtype == np.float32 else trt.float16)
    result = graph.add_gelu_exact(network, source, dtype=build or dtype)
    return result.values, network.operations


def test_the_gelu_holds_its_input_inside_the_dtype_s_finite_range() -> None:
    """``add_gelu_exact`` itself, layer by layer, not a second copy of the formula.

    The clamp exists for one reason: the projection that feeds this runs in the engine's dtype,
    so an FP16 row past +-65504 arrives as an infinity, and both signs end as a NaN -
    `0.5 * -inf * (1 + erf(-inf))` is `-inf * 0` here, and a surviving `+inf` becomes one in the
    next projection, where a zero weight or two mixed signs produce it there. Clamping at the
    dtype's own largest finite value cannot change a value that dtype can carry, which is what
    makes it free; a clamp at any smaller magnitude would cut GELU's linear branch.
    """
    for dtype, bound in ((np.float32, float(np.finfo(np.float32).max)),
                         (np.float16, float(np.finfo(np.float16).max))):
        probes = [-np.inf, -bound, -1.0e4, -5.0, 0.0, 5.0, 1.0e4, bound, np.inf]
        values, operations = _gelu_through_the_builder(probes, dtype=np.float32, build=dtype)
        assert np.isfinite(values).all(), (dtype, values)

        maxima = [value for operation, value in operations
                  if operation == trt.ElementWiseOperation.MAX]
        minima = [value for operation, value in operations
                  if operation == trt.ElementWiseOperation.MIN]
        assert maxima == [-bound], (dtype, maxima)
        assert minima == [bound], (dtype, minima)

        # Below the bound everything is the same signed zero; above it, GELU is x.
        assert values[0] == values[1]
        assert abs(float(values[1])) < 1e-30
        assert abs(float(values[7]) - bound) <= abs(bound) * 1e-6
        assert values[8] == values[7]
        # The clamp does not touch the range the tower actually works in.
        assert abs(float(values[6]) - 1.0e4) < 1e-3
        for index, value in ((3, -5.0), (4, 0.0), (5, 5.0)):
            expected = 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
            assert abs(float(values[index]) - expected) < 1e-6


def test_the_gelu_is_fp32_even_in_an_fp16_graph() -> None:
    # The cancellation in `1 + erf(x)` is the reason: at x = -4 the erf rounds to exactly -1 in
    # FP16 and the activation returns 0 where the answer is -1.27e-4.
    values, operations = _gelu_through_the_builder([-4.0], dtype=np.float16, build=np.float16)
    casts = [dtype for operation, dtype in operations if operation == "cast"]
    assert casts == [trt.float32, trt.float16], casts
    assert abs(float(values[0]) - (-1.2672e-4)) < 1e-6, values


def test_duplicate_configuration_keys_are_refused(tmp_path: Path) -> None:
    # json.loads keeps the last of two identical keys, so a file making two contradictory claims
    # passes validation on whichever came second.
    (tmp_path / "processor_config.json").write_text(json.dumps(_PROCESSOR), encoding="utf-8")
    (tmp_path / "config.json").write_text(
        '{"model_type": "tipsv2", "model_type": "tipsv2_dpt"}', encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        config_module.resolve(tmp_path, _weights())


def test_a_duplicate_key_in_the_processor_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(
        '{"image_processor": {"do_normalize": false, "do_normalize": true}}', encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        config_module.resolve(tmp_path, _weights())


def test_a_contradictory_mlp_ratio_is_refused(tmp_path: Path) -> None:
    # The reference takes the ratio from the factory and never reads this field, so a file that
    # states another one describes a different model than the one that would be built.
    directory = _model_dir(tmp_path, _with("vision_config", mlp_ratio=4.0))
    with pytest.raises(ValueError, match="mlp_ratio"):
        config_module.resolve(directory, _weights())
    # The checkpoint's own value agrees with the factory and is accepted.
    directory = _model_dir(tmp_path, _with("vision_config",
                                           mlp_ratio=_INTERMEDIATE / _WIDTH))
    assert config_module.resolve(directory, _weights())["vision"]["intermediate_size"] == \
        _INTERMEDIATE


@pytest.mark.parametrize("field,value", [("architectures", None), ("architectures", "TIPSv2Model"),
                                         ("model_type", None), ("model_type", 7)])
def test_a_malformed_identity_claim_is_refused(tmp_path: Path, field: str, value) -> None:
    # `null` is not the same as absent, and a bare string is not a list of architectures.
    config = json.loads(json.dumps(_CONFIG))
    config[field] = value
    with pytest.raises(ValueError, match=field):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_identity_claims_may_be_left_out(tmp_path: Path) -> None:
    config = json.loads(json.dumps(_CONFIG))
    del config["architectures"]
    del config["model_type"]
    assert config_module.resolve(_model_dir(tmp_path, config), _weights())["embedding_dim"] == \
        _WIDTH


@pytest.mark.parametrize("section", [None, False, 0, [], "vision"])
def test_a_malformed_configuration_section_is_refused(tmp_path: Path, section) -> None:
    # `config.get(name) or {}` reads every one of these as "not there", which is how an
    # explicitly empty or malformed section would skip every check under it.
    config = json.loads(json.dumps(_CONFIG))
    config["vision_config"] = section
    with pytest.raises(ValueError, match="vision_config"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_a_configuration_section_may_be_left_out(tmp_path: Path) -> None:
    # The vision tower's geometry comes from the tensors and the factory, so the section is
    # optional; only the text tower's four numbers are required.
    config = json.loads(json.dumps(_CONFIG))
    del config["vision_config"]
    assert config_module.resolve(_model_dir(tmp_path, config), _weights())["vision"]["num_heads"] \
        == _HEADS


def test_the_header_is_what_decides_before_anything_is_loaded() -> None:
    # `_require_header` is given names, shapes and dtypes only - no values - so a checkpoint this
    # family cannot build is refused without materialising it.
    cfg = {"vision": {"hidden_size": _WIDTH, "intermediate_size": _INTERMEDIATE,
                      "num_layers": 1, "num_register_tokens": 1,
                      "num_positions": _GRID * _GRID + 1, "patch_size": _PATCH},
           "text": {"hidden_size": _WIDTH, "intermediate_size": _INTERMEDIATE, "num_layers": 1,
                    "vocab_size": _VOCAB}}
    header = {key: (shape, "F32") for key, shape in model_module._expected_shapes(cfg).items()}
    model_module._require_header(header, cfg)

    wrong = dict(header)
    wrong["vision_encoder.blocks.0.mlp.fc1.weight"] = ((4096, _WIDTH), "F32")
    with pytest.raises(ValueError, match="expected"):
        model_module._require_header(wrong, cfg)

    half = dict(header)
    half["vision_encoder.cls_token"] = ((1, 1, _WIDTH), "F16")
    with pytest.raises(NotImplementedError, match="float32 checkpoint"):
        model_module._require_header(half, cfg)


def test_the_attention_clamps_every_projection_before_it_widens() -> None:
    """``add_attention`` itself: an overflowed FP16 projection must not reach the score product.

    Widening an infinity keeps it, and from there the score product, the softmax and the value
    product all produce NaNs. The clamp is at the engine dtype's own largest finite value, so no
    projection that dtype can carry is touched.
    """
    heads, head_dim, seq = 2, 4, 3
    width = heads * head_dim
    bound = float(np.finfo(np.float16).max)

    rng = np.random.default_rng(290)
    healthy = rng.standard_normal((seq, width))
    broken = healthy.copy()
    broken[1, 0] = np.inf
    broken[2, 3] = -np.inf

    def attend(values):
        network = _RecordingNetwork()
        rows = [_RecordedTensor(np.asarray(values, dtype=np.float32), trt.float16)
                for _ in range(3)]
        out = graph.add_attention(network, rows[0], rows[1], rows[2], num_heads=heads,
                                  head_dim=head_dim, dtype=np.float16)
        return out.values, network.operations

    values, operations = attend(broken)
    assert np.isfinite(values).all(), values

    minima = [value for operation, value in operations
              if operation == trt.ElementWiseOperation.MIN]
    maxima = [value for operation, value in operations
              if operation == trt.ElementWiseOperation.MAX]
    # One clamp per projection: query, key and value.
    assert minima == [bound] * 3, minima
    assert maxima == [-bound] * 3, maxima

    # And the clamp is invisible to a projection FP16 could actually hold.
    healthy_values, _ = attend(healthy)
    assert np.isfinite(healthy_values).all()
    assert healthy_values.shape == (seq, width)


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_clamp_to_finite_is_a_no_op_on_everything_the_dtype_can_hold(dtype) -> None:
    """The one rule the FP16 path rests on, tested on its own.

    Clamping at the dtype's own largest finite value is free: it changes no value that dtype can
    carry. Any smaller bound would be a tradeoff, and any larger one would not catch the
    infinity it exists for.
    """
    bound = float(np.finfo(dtype).max)
    tiny = float(np.finfo(dtype).tiny)
    probes = [-np.inf, -bound, -1.0, -tiny, 0.0, tiny, 1.0, bound, np.inf]

    for tensor_dtype, trt_dtype in ((np.float32, trt.float32), (np.float16, trt.float16)):
        # Both the widened call sites and the one that clamps a tensor still in the engine's
        # dtype, and both orders of (tensor dtype, engine dtype). The constants have to follow
        # the **tensor**, or a strongly typed network refuses the operation and the engine never
        # builds; and the bound has to be the narrower of the two, or it becomes an infinity on
        # the way into a narrower constant and clamps nothing.
        # Probes this tensor dtype can actually hold, so the fixture itself is not the overflow.
        carried = float(np.finfo(tensor_dtype).max)
        network = _RecordingNetwork()
        source = _RecordedTensor(
            np.asarray([-np.inf, -carried, -1.0, 0.0, 1.0, carried, np.inf], dtype=tensor_dtype),
            trt_dtype)
        clamped = graph.clamp_to_finite(network, source, dtype)
        assert np.isfinite(clamped.values).all(), (tensor_dtype, dtype, clamped.values)
        for operation, value in network.operations:
            if operation in (trt.ElementWiseOperation.MIN, trt.ElementWiseOperation.MAX):
                assert math.isfinite(value), (tensor_dtype, dtype, value)

    network = _RecordingNetwork()
    source = _RecordedTensor(np.asarray(probes, dtype=np.float32), trt.float32)
    result = graph.clamp_to_finite(network, source, dtype)

    assert np.isfinite(result.values).all(), result.values
    for index in range(1, len(probes) - 1):  # everything but the two infinities
        assert float(result.values[index]) == probes[index], probes[index]
    assert float(result.values[0]) == -bound
    assert float(result.values[-1]) == bound


def test_the_e2e_comparison_record_earns_the_reference_badge(tmp_path: Path) -> None:
    """The record `tests/test_e2e.py` writes has to be one the report can act on.

    A reviewer reads the badge on each case's `report.html`, and the strongest way to earn
    "Reference checks passed" is a complete `reference_comparison` record. A malformed one is
    not an error anywhere - it simply stops counting, and the badge quietly falls back to
    whatever the raw assertions happen to establish. This family wrote one with **three checks
    called `relative_l2`**, and duplicate names void the whole record; `native` and `reference`
    pointing at one artifact void it the same way. Both are pinned here, in the shape the
    recorder stores and the report reads.
    """
    from tools.e2e_report import render_case

    def verdict(case: dict) -> str:
        """The badge on the case's own section - the page's filter menu names every badge."""
        page = render_case(case, tmp_path)
        found = re.search(r'<section class="case"[^>]*\bdata-status="([a-z]+)"', page)
        assert found is not None, page[:400]
        return found.group(1)

    def artifact(name: str) -> dict:
        return {"artifact": f"artifacts/{name}", "size_bytes": 4736}

    # The checks `test_e2e.py` itself builds, from a healthy fp32 run's measured numbers - not a
    # copy of them here, or the production record could regress while this still passed.
    from families.tipsv2.tests.test_e2e import reference_checks, reference_comparison_record

    checks = reference_checks(image_delta=1.1e-07, image_cosine=1.0, text_delta=6.0e-08,
                              logit_delta=6.2e-06, agrees=True, embedding_tolerance=1e-05,
                              logit_tolerance=1e-03, cosine_floor=0.999)
    assert all(check["passed"] for check in checks)
    # The record production writes, not a copy of it: if `test_e2e.py` pointed both fields at one
    # artifact, or named two checks the same, the badge would be lost there and here together.
    record = reference_comparison_record("tipsv2-so400m14",
                                         artifact("native_image_embedding.npy"),
                                         artifact("reference_image_embedding.npy"), checks)
    data = {"status": "passed", "family": "tipsv2", "case": "tipsv2-so400m14", "checks": [],
            "reference_comparison": record}

    assert verdict(data) == "reference"

    # The shape this family actually had: every check named `relative_l2`.
    repeated = {**record, "checks": [{**check, "name": "relative_l2"} for check in checks]}
    assert verdict({**data, "reference_comparison": repeated}) == "unverified"
    # And the other way a record is voided, which the two `.npy` files above also avoid.
    same = {**record, "reference": artifact("native_image_embedding.npy")}
    assert verdict({**data, "reference_comparison": same}) == "unverified"


def test_the_record_builder_refuses_what_voids_the_badge(tmp_path: Path) -> None:
    """The test above can only speak for the arguments **it** passes.

    `test_e2e.py` builds the production record at one call site. A change there - both fields
    pointing at `native_file`, or a check name repeated - would void the record in the real
    report while every test here still passed, because they supply their own fixtures. So the
    builder refuses both, and that call site cannot produce a voided record at all.
    """
    from families.tipsv2.tests.test_e2e import reference_checks, reference_comparison_record

    checks = reference_checks(image_delta=1.1e-07, image_cosine=1.0, text_delta=6.0e-08,
                              logit_delta=6.2e-06, agrees=True, embedding_tolerance=1e-05,
                              logit_tolerance=1e-03, cosine_floor=0.999)
    native = tmp_path / "native_image_embedding.npy"
    reference = tmp_path / "reference_image_embedding.npy"

    # The paths the E2E passes, and the dicts the recorder turns them into.
    with pytest.raises(ValueError, match="two different artifacts"):
        reference_comparison_record("tipsv2-so400m14", native, native, checks)
    with pytest.raises(ValueError, match="two different artifacts"):
        reference_comparison_record("tipsv2-so400m14", {"artifact": "artifacts/native.npy"},
                                    {"artifact": "artifacts/native.npy"}, checks)
    with pytest.raises(ValueError, match="one name per check"):
        reference_comparison_record("tipsv2-so400m14", native, reference,
                                    [{**check, "name": "relative_l2"} for check in checks])

    record = reference_comparison_record("tipsv2-so400m14", native, reference, checks)
    assert record["native"] != record["reference"]


def _tiny_text_cfg(window: int) -> dict:
    return {"hidden_size": 8, "max_text_len": window, "num_layers": 2, "num_heads": 2,
            "head_dim": 4, "vocab_size": 11, "pad_token_id": 0, "eps": 1e-5,
            "pooling_eps": 1e-8}


def _tiny_text_weights(rng, cfg: dict) -> dict:
    width, inner = cfg["hidden_size"], cfg["hidden_size"] * 2
    weights = {f"{text_builder.PREFIX}.token_embedding.weight":
               rng.standard_normal((cfg["vocab_size"], width)).astype(np.float32),
               f"{text_builder.PREFIX}.ln_final.weight": np.ones(width, dtype=np.float32),
               f"{text_builder.PREFIX}.ln_final.bias": np.zeros(width, dtype=np.float32)}
    for index in range(cfg["num_layers"]):
        layer = f"{text_builder.PREFIX}.transformer.resblocks.{index}"
        weights[f"{layer}.attn.in_proj_weight"] = rng.standard_normal(
            (3 * width, width)).astype(np.float32)
        weights[f"{layer}.attn.in_proj_bias"] = rng.standard_normal(3 * width).astype(np.float32)
        weights[f"{layer}.attn.out_proj.weight"] = rng.standard_normal(
            (width, width)).astype(np.float32)
        weights[f"{layer}.attn.out_proj.bias"] = rng.standard_normal(width).astype(np.float32)
        weights[f"{layer}.mlp.c_fc.weight"] = rng.standard_normal((inner, width)).astype(np.float32)
        weights[f"{layer}.mlp.c_fc.bias"] = rng.standard_normal(inner).astype(np.float32)
        weights[f"{layer}.mlp.c_proj.weight"] = rng.standard_normal(
            (width, inner)).astype(np.float32)
        weights[f"{layer}.mlp.c_proj.bias"] = rng.standard_normal(width).astype(np.float32)
        for affine in ("ln_1", "ln_2"):
            weights[f"{layer}.{affine}.weight"] = np.ones(width, dtype=np.float32)
            weights[f"{layer}.{affine}.bias"] = np.zeros(width, dtype=np.float32)
    return weights


def _text_embedding(ids: list, window: int) -> np.ndarray:
    cfg = _tiny_text_cfg(window)
    weights = _tiny_text_weights(np.random.default_rng(290), cfg)
    padded = list(ids) + [cfg["pad_token_id"]] * (window - len(ids))
    network = _RecordingNetwork()
    source = _RecordedTensor(np.asarray(padded, dtype=np.int32), trt.int32)
    return text_builder.build_text_tower(network, source, weights, cfg,
                                         dtype=np.float32).values


def test_the_text_embedding_does_not_depend_on_the_padding_behind_it() -> None:
    """The property the padding mask exists for, run rather than asserted in prose.

    The padding mask is used in the attention keys, in both MLP maskings and in the pooled
    average. **Two** of those three carry this property - the keys and the pooled average - and
    because of them a prompt that fits gives the same embedding at any window at least as long
    as it; dropping either breaks that, silently, in a way no comparison against a single
    reference run would show. The MLP maskings are reference fidelity and do not carry it: the
    two tests below measure that and hold them in place by what they actually do.

    Two windows, the same four real tokens. The production builder runs in NumPy through the
    recorder, so a mask deleted from `text_builder.py` fails this.

    Not *exactly* equal here, and the reason is the harness rather than the tower: the pooled
    sum runs over a different number of rows in the two windows, and NumPy's pairwise summation
    adds them in a different order, which float32 addition is not associative under. That is
    2.7e-07 on this fixture. Disabling either of the two maskings that carry the property costs
    0.26 (the attention keys) or 0.59 (the pooled average) - six orders of magnitude away - so
    the gate sits between them, nowhere near either. The MLP maskings are a third use of the
    same mask and do **not** carry it; the test below that measures them says why.
    """
    ids = [5, 7, 3, 9]
    short, long = _text_embedding(ids, 8), _text_embedding(ids, 24)

    assert short.shape == long.shape == (8,)
    assert np.isfinite(short).all()
    assert float(np.abs(short - long).max()) < 1e-5, np.abs(short - long).max()


@pytest.mark.parametrize("mask_use", ["keys", "pooling"])
def test_dropping_either_masking_that_carries_it_breaks_that(monkeypatch, mask_use: str) -> None:
    """And the same property, with each of the two loadbearing uses disabled in turn.

    Without this the test above would pass on a tower that never masked anything: with no
    padding in the short window the two runs could still agree. Each case here removes one use
    and asserts the embeddings then differ.
    """
    ids = [5, 7, 3, 9]
    if mask_use == "keys":
        # The attention gets no additive mask, so padding rows are attended as keys.
        original = text_builder.g.add_attention
        monkeypatch.setattr(
            text_builder.g, "add_attention",
            lambda *args, additive_mask=None, **kwargs: original(*args, **kwargs))
    else:
        monkeypatch.setattr(text_builder, "_masks", _masks_without_pooling())

    short, long = _text_embedding(ids, 8), _text_embedding(ids, 24)
    # Measured: 0.26 (keys), 0.59 (pooling), against 2.7e-07 for the correct tower.
    assert float(np.abs(short - long).max()) > 0.1, (mask_use, np.abs(short - long).max())


def test_the_mlp_masking_is_reference_fidelity_and_not_what_carries_the_property(
        monkeypatch) -> None:
    """The third use of the mask does **not** carry padding invariance, and saying it did was
    wrong.

    The attention's additive mask already keeps a padding row from being read as a key, and the
    pooled average already drops it from both the sum and the count. Nothing downstream ever
    reads a padded row, so zeroing it in the MLP cannot move the embedding: disabling the two
    selections leaves this at the same 2.7e-07 as a correct tower, with a padded row that
    overflows just as much as with one that does not.

    They are kept because they are what the reference computes (`x * mlp_mask`), and because a
    selection is the one form that could not turn a non-finite padded row into a NaN if a later
    change ever made those rows observable. That is a different claim from the one this file
    used to make, and the number belongs to it.
    """
    ids = [5, 7, 3, 9]
    before = _text_embedding(ids, 8), _text_embedding(ids, 24)
    monkeypatch.setattr(text_builder, "_masked_rows",
                        lambda _network, tensor, _keep, _zero: tensor)
    after = _text_embedding(ids, 8), _text_embedding(ids, 24)

    assert float(np.abs(after[0] - after[1]).max()) < 1e-5     # invariance survives without them
    assert float(np.abs(before[0] - after[0]).max()) == 0.0    # and the embedding is unchanged


def test_the_mlp_masking_zeroes_the_padded_rows_at_both_selections(monkeypatch) -> None:
    """What holds the two MLP selections in place, now that the embedding cannot.

    The test above measures what they do **not** do, and every test that reads only the pooled
    embedding - that one and the invariance pair - passes with `_masked_rows` replaced by the
    identity, because nothing downstream reads a padded row. The selections are still the
    reference's `x * mlp_mask`, so they are checked where they act rather than where they do
    not: each one leaves every padded row zero and no real row touched.
    """
    ids, window = [5, 7, 3, 9], 24
    calls = []
    original = text_builder._masked_rows

    def spy(network, tensor, keep_rows, zero):
        out = original(network, tensor, keep_rows, zero)
        calls.append((np.asarray(tensor.values).copy(), np.asarray(out.values).copy()))
        return out

    monkeypatch.setattr(text_builder, "_masked_rows", spy)
    _text_embedding(ids, window)

    assert len(calls) == 2 * _tiny_text_cfg(window)["num_layers"], "two selections per block"
    # Without this the loop below would hold for a tower whose padded rows were zero already.
    assert any(np.count_nonzero(before[len(ids):]) for before, _ in calls)
    for before, after in calls:
        assert after.shape == before.shape
        np.testing.assert_array_equal(after[:len(ids)], before[:len(ids)])
        assert np.count_nonzero(after[len(ids):]) == 0, after[len(ids):]


def _masks_without_pooling():
    """`_masks` with `valid` all-ones: the pooled average then divides by the window, not by the
    number of real tokens."""
    original = text_builder._masks

    def patched(network, input_ids, cfg):
        keep_rows, additive_mask, valid = original(network, input_ids, cfg)
        ones = _RecordedTensor(np.ones_like(np.asarray(valid.values)), valid.dtype)
        return keep_rows, additive_mask, ones

    return patched


def test_the_residual_stream_cannot_become_a_nan() -> None:
    """Two overflows of opposite sign meeting in a residual addition are `inf + -inf`: a NaN.

    Everything else in this family clamps, and a clamp cannot take a NaN back - it compares, and
    every comparison against a NaN is false. So the residual stream is kept finite at each step
    instead: the stream is finite going in, which is what makes the addition finite-or-infinite
    but never a NaN, and the result is clamped so the next block starts from the invariant again.
    Three blocks here, each overflowing the other way from the stream.
    """
    bound = float(np.finfo(np.float16).max)
    stream = _RecordedTensor(np.full((2, 4), bound, dtype=np.float16), trt.float16)
    network = _RecordingNetwork()

    # Block 1 overflows the stream the positive way, block 2 hands it an overflowed projection
    # of the other sign, block 3 shows the stream is still usable afterwards.
    for value in (bound, -np.inf, 1.0):
        branch = _RecordedTensor(np.full((2, 4), value, dtype=np.float16), trt.float16)
        stream = graph.add_residual(network, stream, branch, np.float16)
        assert np.isfinite(stream.values).all(), (value, stream.values)

    # Unclamped, those same two steps are exactly the NaN this exists for: the stream reaches an
    # infinity by overflowing, and the next branch's own overflow has the opposite sign.
    with np.errstate(invalid="ignore", over="ignore"):
        unprotected = np.float16(bound) + np.float16(bound)
        assert np.isinf(unprotected)
        assert np.isnan(unprotected + np.float16(-np.inf))


def test_the_layer_norm_cannot_be_handed_an_infinity() -> None:
    """The last stage that widens: without the clamp its mean is an infinity and `inf - inf` is a
    NaN that the rest of the tower carries."""
    width = 8
    row = np.ones((2, width), dtype=np.float32)
    row[0, 3] = np.inf

    network = _RecordingNetwork()
    source = _RecordedTensor(row, trt.float16)
    result = graph.add_layer_norm(network, source, np.ones(width, dtype=np.float32),
                                  np.zeros(width, dtype=np.float32), 1e-5)
    assert np.isfinite(result.values).all(), result.values
    # The clamp is the FP16 maximum, because that is the dtype the row arrived in.
    minima = [value for operation, value in network.operations
              if operation == trt.ElementWiseOperation.MIN]
    assert minima == [float(np.finfo(np.float16).max)] * 2, minima


def test_the_layer_norm_cannot_hand_back_an_infinity_either() -> None:
    """The way out, for the reason the way in is guarded.

    The statistics and the affine run in FP32 and the result goes back to the engine's dtype. A
    gamma large enough makes that result finite in FP32 and an infinity in FP16, and the
    projection that reads it next turns the infinity into a NaN - a build that overflowed on the
    way out rather than on the way in, which the input clamp cannot see.
    """
    width = 8
    row = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    gamma = np.full(width, 1.0e5, dtype=np.float32)  # finite in FP32, 1.5x past FP16's maximum

    network = _RecordingNetwork()
    result = graph.add_layer_norm(network, _RecordedTensor(row, trt.float16), gamma,
                                  np.zeros(width, dtype=np.float32), 1e-5)

    assert result.dtype == trt.float16
    assert np.isfinite(result.values).all(), result.values
    assert np.abs(result.values).max() == float(np.finfo(np.float16).max)

    # At fp32 the same layer norm keeps the value: the clamp follows the destination dtype.
    wide = graph.add_layer_norm(_RecordingNetwork(), _RecordedTensor(row, trt.float32), gamma,
                                np.zeros(width, dtype=np.float32), 1e-5)
    assert np.abs(wide.values).max() > float(np.finfo(np.float16).max)


def test_a_weight_that_cannot_survive_the_build_precision_is_refused() -> None:
    """A finite FP32 weight above 65504 becomes an infinity in an FP16 build.

    An infinite constant makes its matmul produce NaNs before any activation clamp can see
    them, in a build that otherwise succeeds. The released checkpoint's largest weight is 1.12,
    so this is a guard for a later variant - and it has to be a refusal, not a clamp: a weight
    the engine cannot represent is not a weight this family can build with.
    """
    network = _RecordingNetwork()
    safe = np.array([-65504.0, -1.0, 0.0, 1.0, 65504.0], dtype=np.float32)
    graph.add_constant(network, safe.shape, safe, dtype=np.float16)  # exactly representable

    too_large = np.array([1.0, 1.0e5], dtype=np.float32)
    with pytest.raises(ValueError, match="becomes an infinity"):
        graph.add_constant(network, too_large.shape, too_large, dtype=np.float16)
    # The same weights build at fp32, which is the point of having two precisions.
    graph.add_constant(network, too_large.shape, too_large, dtype=np.float32)
    # An input that was already infinite is not blamed on the conversion.
    already = np.array([np.inf], dtype=np.float32)
    graph.add_constant(network, already.shape, already, dtype=np.float16)


def test_the_weight_check_sees_every_weight_before_it_is_narrowed() -> None:
    """The guard above is worth nothing unless each weight reaches it **still finite**.

    Almost every weight of the towers goes in through `add_linear` or `add_conv2d`, not through
    `add_constant` directly. A helper that narrows first and hands the result over defeats the
    check completely: by then the value is already an infinity, and "was it finite before?" can
    no longer be answered. So these drive the two helpers themselves.
    """
    too_large = np.full((2, 4), 1.0e5, dtype=np.float32)
    rows = _RecordedTensor(np.ones((3, 4), dtype=np.float16), trt.float16)

    with pytest.raises(ValueError, match="linear weight.*becomes an infinity"):
        graph.add_linear(_RecordingNetwork(), rows, too_large, dtype=np.float16)
    # And the bias, which arrives by a different path in the same call.
    with pytest.raises(ValueError, match="linear bias.*becomes an infinity"):
        graph.add_linear(_RecordingNetwork(), rows, np.ones((2, 4), dtype=np.float32),
                         np.full((2,), 1.0e5, dtype=np.float32), dtype=np.float16)

    class _RefusingNetwork:
        """A convolution must be refused before the layer is ever asked for."""

        @staticmethod
        def add_convolution_nd(*_args, **_kwargs):
            raise AssertionError("tipsv2 built a convolution from a weight it cannot represent")

    with pytest.raises(ValueError, match="convolution weight.*becomes an infinity"):
        graph.add_conv2d(_RefusingNetwork(), rows, np.full((2, 3, 1, 1), 1.0e5, np.float32),
                         dtype=np.float16)
    with pytest.raises(ValueError, match="convolution bias.*becomes an infinity"):
        graph.add_conv2d(_RefusingNetwork(), rows, np.ones((2, 3, 1, 1), dtype=np.float32),
                         np.full((2,), 1.0e5, dtype=np.float32), dtype=np.float16)

    # All four build at fp32, where the same values are representable.
    graph.add_linear(_RecordingNetwork(), _RecordedTensor(np.ones((3, 4), np.float32),
                                                          trt.float32), too_large,
                     np.full((2,), 1.0e5, dtype=np.float32), dtype=np.float32)


def test_layer_scale_cannot_turn_an_overflowed_branch_into_a_nan() -> None:
    """`inf * 0` is a NaN, and a NaN is the one value no later clamp can repair.

    LayerScale multiplies a residual branch by a trained per-channel gamma. On a checkpoint with
    a zero entry there - which the architecture permits, and which is what a pruned channel
    looks like - an FP16 projection that overflowed would meet it and produce a NaN, and the
    residual stream would carry that NaN into every later block and out into the embedding.
    """
    width = 4
    branch = np.ones((2, width), dtype=np.float16)
    branch[0, 1] = np.inf
    gamma = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    # Without the clamp this is what the product would be. `inf * 0` is exactly the invalid
    # operation NumPy warns about, which is the point of the assertion.
    with np.errstate(invalid="ignore"):
        assert np.isnan(branch.astype(np.float32) * gamma).any()

    network = _RecordingNetwork()
    scaled = vision_builder._layer_scaled(
        network, _RecordedTensor(branch, trt.float16), gamma, width, dtype=np.float16)

    assert np.isfinite(scaled.values).all(), scaled.values
    assert float(scaled.values[0, 1]) == 0.0  # the zero channel, zero rather than NaN
    minima = [value for operation, value in network.operations
              if operation == trt.ElementWiseOperation.MIN]
    assert minima == [float(np.finfo(np.float16).max)], minima
