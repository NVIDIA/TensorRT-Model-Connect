# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a TrOCR checkpoint into the values the builder and the runtime use.

Every value the graph depends on is read from the checkpoint's files, and every
option this family does not build is refused by name rather than ignored:

* ``config.json``: a ``vit`` encoder (pre-norm, exact GELU) and a ``trocr``
  decoder (post-norm, exact GELU, learned positions offset by two, a layer
  norm on the embeddings, output projection tied to the token embedding).
  The TrOCR checkpoints with sinusoidal positions, ReLU, scaled embeddings and
  an untied projection (``trocr-*-stage1``, ``trocr-large-printed``) and the
  DeiT-encoder ``trocr-small-*`` are a different graph and are refused.
* ``generation_config.json``: the decoder start and end tokens.
* ``preprocessor_config.json``: the ViT image processor (squash to the encoder
  size with Pillow bilinear, rescale, normalize).
* ``tokenizer_config.json``, ``vocab.json``, ``special_tokens_map.json``: the
  byte-level BPE vocabulary the runtime decodes generated ids with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# torch.nn.LayerNorm's default; TrOCR's decoder layer norms do not set it.
_DECODER_LAYER_NORM_EPS = 1e-5
_BILINEAR = 2


@dataclass(frozen=True)
class EncoderConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    layer_norm_eps: float
    qkv_bias: bool
    image_size: int
    patch_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def sequence_length(self) -> int:
        """Patches plus the class token."""
        return self.grid * self.grid + 1


@dataclass(frozen=True)
class DecoderConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_size: int
    vocab_size: int
    max_positions: int
    layer_norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass(frozen=True)
class PreprocessConfig:
    height: int
    width: int
    resample: int
    rescale_factor: float
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]


@dataclass(frozen=True)
class TokenizerConfig:
    vocab: tuple[str, ...]
    special_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrocrConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig
    preprocess: PreprocessConfig
    tokenizer: TokenizerConfig
    decoder_start_token_id: int
    eos_token_id: int


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"trocr checkpoint file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"trocr expects one JSON object in {path}")
    return value


def _int(config: dict, key: str, default: int | None = None) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"trocr config value {key}={value!r} must be a positive integer")
    return value


def _flag(config: dict, key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"trocr config value {key}={value!r} must be a boolean")
    return value


def _token_id(config: dict, key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"trocr {key}={value!r} must be one non-negative token id")
    return value


def _encoder(config: dict) -> EncoderConfig:
    if config.get("model_type") != "vit":
        raise ValueError(f"trocr builds a vit encoder, not {config.get('model_type')!r}")
    if config.get("hidden_act", "gelu") != "gelu":
        raise ValueError(f"trocr encoder hidden_act {config.get('hidden_act')!r} is not gelu")
    if config.get("num_channels", 3) != 3:
        raise ValueError("trocr encoder must read three-channel RGB images")
    image_size = _int(config, "image_size", 224)
    patch_size = _int(config, "patch_size", 16)
    if image_size % patch_size:
        raise ValueError("trocr encoder image_size must be a multiple of patch_size")
    encoder = EncoderConfig(
        hidden_size=_int(config, "hidden_size", 768),
        num_layers=_int(config, "num_hidden_layers", 12),
        num_heads=_int(config, "num_attention_heads", 12),
        intermediate_size=_int(config, "intermediate_size", 3072),
        layer_norm_eps=float(config.get("layer_norm_eps", 1e-12)),
        qkv_bias=_flag(config, "qkv_bias", True),
        image_size=image_size,
        patch_size=patch_size,
    )
    if encoder.hidden_size % encoder.num_heads:
        raise ValueError("trocr encoder hidden_size must be divisible by its heads")
    if not (math.isfinite(encoder.layer_norm_eps) and encoder.layer_norm_eps > 0.0):
        raise ValueError("trocr encoder layer_norm_eps must be positive")
    return encoder


def _decoder(config: dict, encoder: EncoderConfig) -> DecoderConfig:
    if config.get("model_type") != "trocr":
        raise ValueError(f"trocr builds a trocr decoder, not {config.get('model_type')!r}")
    unsupported = {
        "activation_function": (config.get("activation_function", "gelu"), "gelu"),
        "use_learned_position_embeddings": (
            _flag(config, "use_learned_position_embeddings", True), True),
        "layernorm_embedding": (_flag(config, "layernorm_embedding", True), True),
        "scale_embedding": (_flag(config, "scale_embedding", False), False),
        "tie_word_embeddings": (_flag(config, "tie_word_embeddings", True), True),
    }
    for key, (value, built) in unsupported.items():
        if value != built:
            raise ValueError(
                f"trocr builds the decoder with {key}={built!r}; this checkpoint has {value!r}")
    decoder = DecoderConfig(
        hidden_size=_int(config, "d_model", 1024),
        num_layers=_int(config, "decoder_layers", 12),
        num_heads=_int(config, "decoder_attention_heads", 16),
        ffn_size=_int(config, "decoder_ffn_dim", 4096),
        vocab_size=_int(config, "vocab_size", 50265),
        max_positions=_int(config, "max_position_embeddings", 512),
        layer_norm_eps=_DECODER_LAYER_NORM_EPS,
    )
    if decoder.hidden_size % decoder.num_heads:
        raise ValueError("trocr decoder d_model must be divisible by its heads")
    # The decoder's cross-attention reads the encoder states directly; a
    # missing cross_attention_hidden_size would mean a projection in between.
    if config.get("cross_attention_hidden_size") != encoder.hidden_size:
        raise ValueError(
            "trocr decoder cross_attention_hidden_size must equal the encoder hidden_size")
    return decoder


def _preprocess(path: Path, encoder: EncoderConfig) -> PreprocessConfig:
    config = _read_json(path)
    kind = config.get("image_processor_type", config.get("feature_extractor_type"))
    if kind not in {"ViTImageProcessor", "ViTFeatureExtractor"}:
        raise ValueError(f"trocr preprocesses with the ViT image processor, not {kind!r}")
    for key in ("do_resize", "do_rescale", "do_normalize"):
        if not _flag(config, key, True):
            raise ValueError(f"trocr requires preprocessor {key}=true")
    size = config.get("size", 224)
    if isinstance(size, int) and not isinstance(size, bool):
        height = width = size
    elif isinstance(size, dict) and set(size) == {"height", "width"}:
        height, width = size["height"], size["width"]
    else:
        raise ValueError(f"trocr preprocessor size {size!r} is not a height and width")
    if (height, width) != (encoder.image_size, encoder.image_size):
        raise ValueError("trocr preprocessor size must equal the encoder image_size")
    resample = config.get("resample", _BILINEAR)
    if resample != _BILINEAR:
        raise ValueError(f"trocr resizes with Pillow bilinear (2), not resample={resample!r}")
    rescale = float(config.get("rescale_factor", 1.0 / 255.0))

    def triple(key: str) -> tuple[float, float, float]:
        value = config.get(key, 0.5)
        values = [value] * 3 if isinstance(value, (int, float)) else value
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"trocr preprocessor {key} must name three channels")
        return tuple(float(item) for item in values)

    image_std = triple("image_std")
    if any(value == 0.0 for value in image_std):
        raise ValueError("trocr preprocessor image_std must be non-zero")
    return PreprocessConfig(height=height, width=width, resample=resample,
                            rescale_factor=rescale, image_mean=triple("image_mean"),
                            image_std=image_std)


def _tokenizer(model_dir: Path, vocab_size: int) -> TokenizerConfig:
    config = _read_json(model_dir / "tokenizer_config.json")
    if config.get("tokenizer_class") != "RobertaTokenizer":
        raise ValueError(
            "trocr decodes with the byte-level BPE RobertaTokenizer, not "
            f"{config.get('tokenizer_class')!r}")
    raw = _read_json(model_dir / "vocab.json")
    vocab: list[str | None] = [None] * vocab_size
    for token, index in raw.items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < vocab_size:
            raise ValueError(f"trocr vocab.json maps {token!r} outside the decoder vocabulary")
        if vocab[index] is not None:
            raise ValueError(f"trocr vocab.json maps two tokens to id {index}")
        vocab[index] = token
    missing = [index for index, token in enumerate(vocab) if token is None]
    if missing:
        raise ValueError(f"trocr vocab.json has no token for {len(missing)} decoder ids")
    special = _read_json(model_dir / "special_tokens_map.json")
    ids = set()
    for name, value in special.items():
        content = value.get("content") if isinstance(value, dict) else value
        if not isinstance(content, str) or content not in raw:
            raise ValueError(f"trocr special token {name}={content!r} is not in vocab.json")
        ids.add(raw[content])
    return TokenizerConfig(vocab=tuple(vocab), special_token_ids=tuple(sorted(ids)))


def resolve(model_dir: str | Path) -> TrocrConfig:
    """Read and validate every file this family consumes."""
    root = Path(model_dir)
    config = _read_json(root / "config.json")
    if config.get("model_type") != "vision-encoder-decoder":
        raise ValueError("trocr expects a vision-encoder-decoder checkpoint")
    encoder_raw, decoder_raw = config.get("encoder"), config.get("decoder")
    if not isinstance(encoder_raw, dict) or not isinstance(decoder_raw, dict):
        raise ValueError("trocr config.json must carry encoder and decoder configs")
    encoder = _encoder(encoder_raw)
    decoder = _decoder(decoder_raw, encoder)
    generation_path = root / "generation_config.json"
    generation = _read_json(generation_path) if generation_path.is_file() else {}
    start = _token_id(generation, "decoder_start_token_id")
    end = _token_id(generation, "eos_token_id")
    if start is None:
        start = _token_id(decoder_raw, "decoder_start_token_id")
    if end is None:
        end = _token_id(decoder_raw, "eos_token_id")
    if start is None or end is None:
        raise ValueError("trocr needs decoder_start_token_id and eos_token_id")
    for name, value in (("decoder_start_token_id", start), ("eos_token_id", end)):
        if value >= decoder.vocab_size:
            raise ValueError(f"trocr {name}={value} is outside the decoder vocabulary")
    return TrocrConfig(
        encoder=encoder,
        decoder=decoder,
        preprocess=_preprocess(root / "preprocessor_config.json", encoder),
        tokenizer=_tokenizer(root, decoder.vocab_size),
        decoder_start_token_id=start,
        eos_token_id=end,
    )
