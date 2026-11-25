# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and BF16 weights of the original VibeVoice-ASR checkpoint layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

ARCHITECTURE = "VibeVoiceForASRTraining"
SAMPLE_RATE = 24000


@dataclass(frozen=True)
class EncoderConfig:
    """One sigma-VAE tokenizer encoder (acoustic or semantic)."""

    prefix: str
    vae_dim: int
    n_filters: int
    ratios: tuple[int, ...]  # in encoder order (the config lists them reversed)
    depths: tuple[int, ...]
    kernel_size: int
    eps: float

    @property
    def hop_length(self) -> int:
        return int(np.prod(self.ratios))

    def stage_channels(self, stage: int) -> int:
        return self.n_filters * (2**stage)

    def convolutions(self) -> list[tuple[int, int]]:
        """(kernel, stride) of every causal convolution, input to output."""
        layers = [(self.kernel_size, 1)]
        layers += [(self.kernel_size, 1)] * self.depths[0]
        for stage, ratio in enumerate(self.ratios, start=1):
            layers.append((2 * ratio, ratio))
            layers += [(self.kernel_size, 1)] * self.depths[stage]
        layers.append((self.kernel_size, 1))
        return layers

    def history_samples(self) -> int:
        """Input samples before an output frame's own block that the frame depends on.

        Every convolution is causal with left padding ``kernel - stride`` and its
        output frame ends on the last input of its stride block, so the window
        grows backwards by ``R_in = R_out * stride + kernel - stride``.
        """
        reach = 0
        for kernel, stride in reversed(self.convolutions()):
            reach = reach * stride + kernel - stride
        return reach


@dataclass(frozen=True)
class DecoderConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int


@dataclass(frozen=True)
class VibeVoiceAsrConfig:
    acoustic: EncoderConfig
    semantic: EncoderConfig
    decoder: DecoderConfig

    @property
    def hop_length(self) -> int:
        return self.acoustic.hop_length


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"vibevoice_asr: {message}")


def _encoder_config(raw: dict, prefix: str, kind: str, vae_dim: int) -> EncoderConfig:
    _require(isinstance(raw, dict), f"{kind} config is missing")
    expected = {
        "causal": True,
        "channels": 1,
        "conv_bias": True,
        "conv_norm": "none",
        "disable_last_norm": True,
        "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": True,
        "mixer_layer": "depthwise_conv",
        "pad_mode": "constant",
    }
    for key, value in expected.items():
        _require(raw.get(key) == value, f"{kind} {key}={raw.get(key)!r} is not supported")
    _require(raw.get("vae_dim") == vae_dim, f"{kind} vae_dim does not match the connector")
    depths = raw.get("encoder_depths")
    if isinstance(depths, str):
        depths = [int(value) for value in depths.split("-")]
    ratios = list(reversed(raw.get("encoder_ratios") or []))
    _require(
        isinstance(depths, list) and len(depths) == len(ratios) + 1,
        f"{kind} needs one stage depth per downsampling ratio plus the stem",
    )
    _require(
        all(isinstance(value, int) and value > 0 for value in depths + ratios),
        f"{kind} depths and ratios must be positive integers",
    )
    return EncoderConfig(
        prefix=prefix,
        vae_dim=int(vae_dim),
        n_filters=int(raw["encoder_n_filters"]),
        ratios=tuple(int(value) for value in ratios),
        depths=tuple(int(value) for value in depths),
        kernel_size=int(raw.get("kernel_size", 7)),
        eps=float(raw["layernorm_eps"]),
    )


def _decoder_config(raw: dict) -> DecoderConfig:
    _require(
        isinstance(raw, dict) and raw.get("model_type") == "qwen2",
        "decoder_config must be a qwen2 decoder",
    )
    _require(raw.get("hidden_act") == "silu", "decoder hidden_act must be silu")
    _require(not raw.get("rope_scaling"), "decoder rope_scaling is not supported")
    _require(
        not raw.get("use_sliding_window") and not raw.get("use_mrope"),
        "decoder sliding-window or multimodal RoPE is not supported",
    )
    _require(not raw.get("tie_word_embeddings", False), "the checkpoint keeps an untied lm_head")
    layer_types = raw.get("layer_types")
    _require(
        layer_types is None or set(layer_types) == {"full_attention"},
        "every decoder layer must use full attention",
    )
    hidden = int(raw["hidden_size"])
    heads = int(raw["num_attention_heads"])
    kv_heads = int(raw["num_key_value_heads"])
    _require(hidden % heads == 0 and heads % kv_heads == 0, "decoder head layout is invalid")
    return DecoderConfig(
        hidden_size=hidden,
        num_layers=int(raw["num_hidden_layers"]),
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=hidden // heads,
        intermediate_size=int(raw["intermediate_size"]),
        vocab_size=int(raw["vocab_size"]),
        rms_norm_eps=float(raw["rms_norm_eps"]),
        rope_theta=float(raw["rope_theta"]),
        max_position_embeddings=int(raw["max_position_embeddings"]),
    )


def parse_config(raw: dict) -> VibeVoiceAsrConfig:
    _require(
        raw.get("architectures") == [ARCHITECTURE], f"architectures must be [{ARCHITECTURE!r}]"
    )
    acoustic = _encoder_config(
        raw.get("acoustic_tokenizer_config"),
        "model.acoustic_tokenizer.encoder",
        "acoustic tokenizer",
        raw.get("acoustic_vae_dim"),
    )
    semantic = _encoder_config(
        raw.get("semantic_tokenizer_config"),
        "model.semantic_tokenizer.encoder",
        "semantic tokenizer",
        raw.get("semantic_vae_dim"),
    )
    _require(
        acoustic.ratios == semantic.ratios,
        "the acoustic and semantic encoders must share one frame rate",
    )
    return VibeVoiceAsrConfig(
        acoustic=acoustic, semantic=semantic, decoder=_decoder_config(raw.get("decoder_config"))
    )


def load_config(model_dir: Path) -> VibeVoiceAsrConfig:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"vibevoice_asr checkpoint has no config.json: {path}")
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


class Weights:
    """Read-only access to the checkpoint's safetensors shards."""

    def __init__(self, handles: dict[str, object], weight_map: dict[str, str]) -> None:
        self._handles = handles
        self._weight_map = weight_map

    def names(self) -> list[str]:
        return sorted(self._weight_map)

    def bf16(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Return one BF16 tensor exactly as stored."""
        if name not in self._weight_map:
            raise KeyError(f"vibevoice_asr checkpoint is missing {name}")
        value = self._handles[self._weight_map[name]].get_tensor(name)
        if value.dtype != ml_dtypes.bfloat16:
            raise ValueError(f"vibevoice_asr expects BF16 weights, {name} is {value.dtype}")
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}")
        return np.ascontiguousarray(value)

    def close(self) -> None:
        for handle in self._handles.values():
            exit_method = getattr(handle, "__exit__", None)
            if exit_method is not None:
                exit_method(None, None, None)
        self._handles.clear()


def _open_safetensors(model_dir: Path) -> Weights:
    """Open every shard listed by ``model.safetensors.index.json`` (numpy + ml_dtypes BF16)."""
    root = Path(model_dir)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
    elif (root / "model.safetensors").is_file():
        with safe_open(str(root / "model.safetensors"), framework="numpy") as handle:
            weight_map = {name: "model.safetensors" for name in handle.keys()}
    else:
        raise FileNotFoundError(f"vibevoice_asr checkpoint has no safetensors weights: {root}")
    handles = {}
    for shard in sorted(set(weight_map.values())):
        path = root / shard
        if not path.is_file():
            raise FileNotFoundError(f"vibevoice_asr checkpoint shard is missing: {path}")
        handles[shard] = safe_open(str(path), framework="numpy").__enter__()
    return Weights(handles, dict(weight_map))
