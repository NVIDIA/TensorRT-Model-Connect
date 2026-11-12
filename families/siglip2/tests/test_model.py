# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for SigLIP 2 NaFlex (CPU only, no engine is built).

``config.resolve`` reads the geometry from the tensors and the processor file, since the
published ``config.json`` states almost nothing; ``build`` refuses every option it cannot
honour before it touches the checkpoint; the tokenizer contract refuses tokenizers the
runtime would tokenise differently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.siglip2 import config as config_module, model as model_module
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 256
_LAYERS = 2
_PATCH = 4
_GRID = 4
_VOCAB = 100
_WINDOW = 8

_PREPROCESSOR = {
    "image_processor_type": "Siglip2ImageProcessorFast",
    "do_resize": True,
    "resample": 2,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "patch_size": _PATCH,
    "max_num_patches": 16,
}

_TOKENIZER = {
    "added_tokens": [{"id": 0, "content": "<pad>"}, {"id": 1, "content": "<eos>"}],
    "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
    "pre_tokenizer": None,
    "post_processor": {"type": "TemplateProcessing",
                       "single": [{"Sequence": {"id": "A", "type_id": 0}},
                                  {"SpecialToken": {"id": "<eos>", "type_id": 0}}]},
    "model": {"type": "BPE", "byte_fallback": True, "vocab": {"<pad>": 0, "<eos>": 1, "a": 2},
              "merges": []},
}


def _weights(**overrides) -> dict:
    weights = {
        "text_model.embeddings.token_embedding.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "text_model.embeddings.position_embedding.weight": np.zeros((_WINDOW, _HIDDEN), np.float32),
        "text_model.head.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
        "vision_model.embeddings.patch_embedding.weight":
            np.zeros((_HIDDEN, 3 * _PATCH * _PATCH), np.float32),
        "vision_model.embeddings.position_embedding.weight":
            np.zeros((_GRID * _GRID, _HIDDEN), np.float32),
        "vision_model.head.attention.in_proj_weight": np.zeros((3 * _HIDDEN, _HIDDEN), np.float32),
        "vision_model.head.probe": np.zeros((1, 1, _HIDDEN), np.float32),
        "vision_model.head.attention.in_proj_bias": np.zeros((3 * _HIDDEN,), np.float32),
        "vision_model.embeddings.patch_embedding.bias": np.zeros((_HIDDEN,), np.float32),
        "vision_model.post_layernorm.weight": np.ones((_HIDDEN,), np.float32),
        "text_model.final_layer_norm.weight": np.ones((_HIDDEN,), np.float32),
        "logit_scale": np.zeros((1,), np.float32),
        "logit_bias": np.zeros((1,), np.float32),
    }
    for prefix in ("text_model", "vision_model"):
        for index in range(_LAYERS):
            weights[f"{prefix}.encoder.layers.{index}.mlp.fc1.weight"] = np.zeros(
                (_INTERMEDIATE, _HIDDEN), np.float32)
    weights.update(overrides)
    return weights


def _model_dir(tmp_path: Path, config: dict | None = None, preprocessor: dict | None = None,
               tokenizer: dict | None = None) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(
        config if config is not None else {"architectures": ["Siglip2Model"],
                                           "model_type": "siglip2"}))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(
        preprocessor if preprocessor is not None else _PREPROCESSOR))
    (tmp_path / "tokenizer.json").write_text(json.dumps(
        tokenizer if tokenizer is not None else _TOKENIZER))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"pad_token": "<pad>", "eos_token": "<eos>"}))
    return tmp_path


def test_geometry_comes_from_the_tensors_and_the_processor(tmp_path: Path) -> None:
    cfg = _resolve_with_heads(tmp_path)
    vision, text = cfg["vision"], cfg["text"]
    assert (vision["hidden_size"], vision["intermediate_size"], vision["num_layers"]) == (
        _HIDDEN, _INTERMEDIATE, _LAYERS)
    assert vision["patch_size"] == _PATCH
    assert vision["patch_features"] == 3 * _PATCH * _PATCH
    assert (vision["num_positions"], vision["grid"]) == (_GRID * _GRID, _GRID)
    assert vision["max_num_patches"] == 16
    assert (text["vocab_size"], text["max_text_len"], text["projection_size"]) == (
        _VOCAB, _WINDOW, _HIDDEN)
    assert cfg["resample"] == 2
    assert cfg["rescale_factor"] == pytest.approx(1 / 255)


def test_head_count_defaults_to_the_reference_and_must_divide_the_width(tmp_path: Path) -> None:
    stated = {"architectures": ["Siglip2Model"],
              "text_config": {"num_attention_heads": 4}, "vision_config": {"num_attention_heads": 8}}
    cfg = config_module.resolve(_model_dir(tmp_path, stated), _weights())
    assert (cfg["text"]["num_heads"], cfg["vision"]["num_heads"]) == (4, 8)
    assert cfg["vision"]["head_dim"] == _HIDDEN // 8
    with pytest.raises(ValueError, match="not divisible by 12 heads"):
        config_module.resolve(_model_dir(tmp_path), _weights())


def _resolve_with_heads(tmp_path: Path, config: dict | None = None, **kwargs):
    base = {"architectures": ["Siglip2Model"],
            "text_config": {"num_attention_heads": 4}, "vision_config": {"num_attention_heads": 4}}
    for key, value in (config or {}).items():
        if isinstance(value, dict):
            base.setdefault(key, {}).update(value)
        else:
            base[key] = value
    return config_module.resolve(_model_dir(tmp_path, base, **kwargs), _weights())


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tensors say"):
        _resolve_with_heads(tmp_path, {"vision_config": {"hidden_size": _HIDDEN * 2}})
    with pytest.raises(ValueError, match="tensors say"):
        _resolve_with_heads(tmp_path, {"text_config": {"vocab_size": _VOCAB + 1}})


def test_the_classification_head_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="Siglip2Model"):
        _resolve_with_heads(tmp_path, {"architectures": ["Siglip2ForImageClassification"]})


def test_a_patch_size_the_embedding_cannot_take_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="values per patch"):
        _resolve_with_heads(tmp_path, preprocessor={**_PREPROCESSOR, "patch_size": 8})


def test_a_non_square_position_table_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"vision_model.embeddings.position_embedding.weight":
                          np.zeros((12, _HIDDEN), np.float32)})
    config = {"architectures": ["Siglip2Model"],
              "text_config": {"num_attention_heads": 4}, "vision_config": {"num_attention_heads": 4}}
    with pytest.raises(NotImplementedError, match="square position table"):
        config_module.resolve(_model_dir(tmp_path, config), weights)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"resample": 3}, "bilinear"),
        ({"do_resize": False}, "aspect-preserving resize"),
        ({"do_rescale": False}, "1/255 rescale"),
        ({"rescale_factor": 1 / 127.5}, "1/255 rescale"),
        ({"do_normalize": False}, "normalization"),
        ({"image_processor_type": "SiglipImageProcessor"}, "Siglip2 image processor"),
    ],
)
def test_processor_settings_the_runtime_does_not_reproduce_are_refused(
        tmp_path: Path, override: dict, message: str) -> None:
    with pytest.raises(NotImplementedError, match=message):
        _resolve_with_heads(tmp_path, preprocessor={**_PREPROCESSOR, **override})


def test_the_tokenizer_contract_returns_the_pad_id(tmp_path: Path) -> None:
    assert model_module._tokenizer_contract(_model_dir(tmp_path)) == 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Lowercase"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]}}, "normalizer"),
        ({"pre_tokenizer": {"type": "Metaspace"}}, "pre-tokenizer"),
        ({"post_processor": {"type": "TemplateProcessing",
                             "single": [{"SpecialToken": {"id": "<bos>"}},
                                        {"Sequence": {"id": "A"}}]}}, "one end token"),
        ({"model": {"type": "Unigram", "vocab": []}}, "byte-fallback BPE"),
    ],
)
def test_tokenizers_the_runtime_would_tokenise_differently_are_refused(
        tmp_path: Path, override: dict, message: str) -> None:
    with pytest.raises(NotImplementedError, match=message):
        model_module._tokenizer_contract(_model_dir(tmp_path, tokenizer={**_TOKENIZER, **override}))


class _Writer:
    def __getattr__(self, name):
        raise AssertionError(f"build wrote {name} before refusing")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision", "fp16", "fp32 only"),
        ("backend", "trt_rtx", "backend=trt"),
        ("max_batch_size", 2, "max_batch_size"),
        ("tensor_parallel_size", 2, "tensor parallelism"),
        ("context_parallel_size", 2, "context parallelism"),
        ("video_num_frames", 4, "video_num_frames"),
        ("quantization", "fp8", "quantization"),
        ("fp32_layers", (1,), "mixed-precision"),
        ("dynamic_kv_cache", True, "dynamic_kv_cache"),
        ("mtp_seq_len", 2, "MTP"),
        ("image_height", 224, "image_height/width"),
        ("image_width", 224, "image_height/width"),
        ("task", "image_to_class_scores", "supports only task"),
    ],
)
def test_build_refuses_options_it_cannot_honour(tmp_path: Path, field, value, message) -> None:
    fields = {"model_dir": tmp_path, "output_path": tmp_path / "x.bundle", "family": "siglip2",
              "task": "text_image_to_relevance", "precision": "fp32"}
    fields[field] = value
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model_module.build(BuildRequest(**fields), _Writer())


def _checkpoint(tmp_path: Path, weights: dict) -> Path:
    from safetensors.numpy import save_file

    config = {"architectures": ["Siglip2Model"], "model_type": "siglip2",
              "text_config": {"num_attention_heads": 4}, "vision_config": {"num_attention_heads": 4}}
    model_dir = _model_dir(tmp_path, config)
    save_file(weights, str(model_dir / "model.safetensors"))
    return model_dir


def test_the_prompt_window_is_not_a_capacity(tmp_path: Path) -> None:
    model_dir = _checkpoint(tmp_path, _weights())
    request = BuildRequest(model_dir=model_dir, output_path=tmp_path / "x.bundle", family="siglip2",
                           task="text_to_embedding", precision="fp32",
                           max_sequence_length=_WINDOW * 2)
    with pytest.raises(NotImplementedError, match="window of 8"):
        model_module.build(request, _Writer())


def test_a_reduced_precision_checkpoint_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"text_model.head.weight": np.zeros((_HIDDEN, _HIDDEN), np.float16)})
    model_dir = _checkpoint(tmp_path, weights)
    request = BuildRequest(model_dir=model_dir, output_path=tmp_path / "x.bundle", family="siglip2",
                           task="text_to_embedding", precision="fp32")
    with pytest.raises(NotImplementedError, match="float32 checkpoint"):
        model_module.build(request, _Writer())
