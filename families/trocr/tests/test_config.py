# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract of the TrOCR checkpoint resolution and build request.

The files are the shapes ``microsoft/trocr-base-printed`` publishes, written
into a temporary directory; no weights and no GPU are involved.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.trocr import config as config_module

_CONFIG = {
    "architectures": ["VisionEncoderDecoderModel"],
    "model_type": "vision-encoder-decoder",
    "is_encoder_decoder": True,
    "encoder": {
        "model_type": "vit", "hidden_size": 768, "num_hidden_layers": 12,
        "num_attention_heads": 12, "intermediate_size": 3072, "hidden_act": "gelu",
        "layer_norm_eps": 1e-12, "image_size": 384, "patch_size": 16, "num_channels": 3,
        "qkv_bias": False,
    },
    "decoder": {
        "model_type": "trocr", "d_model": 1024, "decoder_layers": 12,
        "decoder_attention_heads": 16, "decoder_ffn_dim": 4096, "activation_function": "gelu",
        "max_position_embeddings": 512, "cross_attention_hidden_size": 768,
        "vocab_size": 8, "scale_embedding": False, "tie_word_embeddings": True,
        "decoder_start_token_id": 2, "eos_token_id": 2,
    },
}
_GENERATION = {"decoder_start_token_id": 2, "eos_token_id": 2, "pad_token_id": 1}
_PREPROCESSOR = {
    "image_processor_type": "ViTImageProcessor", "do_resize": True, "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "resample": 2, "size": 384,
}
_VOCAB = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "A": 4, "ĠB": 5, ".": 6, "<mask>": 7}
_SPECIAL = {
    "bos_token": {"content": "<s>"}, "eos_token": {"content": "</s>"},
    "unk_token": {"content": "<unk>"}, "sep_token": {"content": "</s>"},
    "pad_token": {"content": "<pad>"}, "cls_token": {"content": "<s>"},
    "mask_token": {"content": "<mask>"},
}


def _write(root: Path, **overrides) -> Path:
    files = {
        "config.json": copy.deepcopy(_CONFIG),
        "generation_config.json": dict(_GENERATION),
        "preprocessor_config.json": dict(_PREPROCESSOR),
        "tokenizer_config.json": {"tokenizer_class": "RobertaTokenizer"},
        "vocab.json": dict(_VOCAB),
        "special_tokens_map.json": dict(_SPECIAL),
    }
    for name, update in overrides.items():
        update(files[name.replace("__", ".")])
    for name, content in files.items():
        (root / name).write_text(json.dumps(content), encoding="utf-8")
    return root


def test_the_published_shape_resolves(tmp_path: Path) -> None:
    cfg = config_module.resolve(_write(tmp_path))
    assert (cfg.encoder.sequence_length, cfg.encoder.head_dim, cfg.encoder.qkv_bias) == (
        577, 64, False)
    assert (cfg.decoder.head_dim, cfg.decoder.layer_norm_eps, cfg.decoder.max_positions) == (
        64, 1e-5, 512)
    assert (cfg.decoder_start_token_id, cfg.eos_token_id) == (2, 2)
    assert cfg.preprocess.resample == 2 and cfg.preprocess.rescale_factor == 1 / 255
    assert (cfg.preprocess.height, cfg.preprocess.width) == (384, 384)
    assert cfg.tokenizer.vocab[5] == "ĠB"
    assert cfg.tokenizer.special_token_ids == (0, 1, 2, 3, 7)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        # trocr-*-stage1 and trocr-large-printed: a different decoder graph.
        ("activation_function", "relu", "activation_function"),
        ("use_learned_position_embeddings", False, "use_learned_position_embeddings"),
        ("layernorm_embedding", False, "layernorm_embedding"),
        ("scale_embedding", True, "scale_embedding"),
        ("tie_word_embeddings", False, "tie_word_embeddings"),
        # A projection between encoder and decoder is not built.
        ("cross_attention_hidden_size", None, "cross_attention_hidden_size"),
    ],
)
def test_decoder_variants_are_refused_by_name(tmp_path, key, value, message) -> None:
    def update(config):
        config["decoder"][key] = value

    with pytest.raises(ValueError, match=message):
        config_module.resolve(_write(tmp_path, config__json=update))


def test_encoder_and_preprocessor_variants_are_refused(tmp_path: Path) -> None:
    cases = {
        "a deit encoder": ("config__json", lambda c: c["encoder"].update(model_type="deit")),
        "a tanh encoder": ("config__json", lambda c: c["encoder"].update(hidden_act="gelu_new")),
        "a bicubic resize": ("preprocessor_config__json", lambda c: c.update(resample=3)),
        "a resize to another size": ("preprocessor_config__json", lambda c: c.update(size=224)),
        "no normalization": ("preprocessor_config__json", lambda c: c.update(do_normalize=False)),
        "a zero rescale factor": ("preprocessor_config__json",
                                  lambda c: c.update(rescale_factor=0.0)),
        "a negative rescale factor": ("preprocessor_config__json",
                                      lambda c: c.update(rescale_factor=-1 / 255)),
        "another processor": ("preprocessor_config__json",
                              lambda c: c.update(image_processor_type="DonutImageProcessor")),
        "another tokenizer": ("tokenizer_config__json",
                              lambda c: c.update(tokenizer_class="XLMRobertaTokenizer")),
        "a vocabulary gap": ("vocab__json", lambda c: c.pop("A")),
    }
    for name, (file, update) in cases.items():
        root = tmp_path / name.replace(" ", "_")
        root.mkdir()
        with pytest.raises(ValueError):
            config_module.resolve(_write(root, **{file: update}))


def test_generation_tokens_fall_back_to_the_decoder_config(tmp_path: Path) -> None:
    root = _write(tmp_path)
    (root / "generation_config.json").unlink()
    cfg = config_module.resolve(root)
    assert (cfg.decoder_start_token_id, cfg.eos_token_id) == (2, 2)


class _Writer:
    def __getattr__(self, name):
        raise AssertionError(f"the request is refused before anything is written ({name})")


def _request(**overrides) -> SimpleNamespace:
    request = dict(
        model_dir=Path("/nonexistent"), task="images_text_to_text", precision="fp32",
        backend="trt", max_sequence_length=None, image_height=None, image_width=None,
        video_num_frames=None, max_batch_size=1, tensor_parallel_size=1,
        context_parallel_size=1, quantization=None, fp32_layers=(), dynamic_kv_cache=False,
        verbose=False, mtp_seq_len=None,
    )
    request.update(overrides)
    return SimpleNamespace(**request)


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "text_continuation"}, {"backend": "trt_rtx"}, {"precision": "bf16"},
        {"precision": "fp16"},
        {"quantization": "fp8"}, {"fp32_layers": (1,)}, {"max_batch_size": 2},
        {"tensor_parallel_size": 2}, {"context_parallel_size": 2}, {"dynamic_kv_cache": True},
        {"video_num_frames": 8}, {"mtp_seq_len": 2},
    ],
)
def test_unsupported_build_options_are_refused(overrides) -> None:
    from families.trocr import model

    with pytest.raises((ValueError, NotImplementedError)):
        model.build(_request(**overrides), _Writer())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_height": 224}, "image_height"),
        ({"image_width": 512}, "image_width"),
        ({"max_sequence_length": 1}, "max_sequence_length"),
        ({"max_sequence_length": 513}, "max_sequence_length"),
    ],
)
def test_shape_options_are_checked_against_the_checkpoint(tmp_path, overrides, message) -> None:
    from families.trocr import model

    with pytest.raises((ValueError, NotImplementedError), match=message):
        model.build(_request(model_dir=_write(tmp_path), **overrides), _Writer())
