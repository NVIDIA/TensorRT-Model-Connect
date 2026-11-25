# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the original VibeVoice safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy for safetensors)
import numpy as np

# The original checkpoint stores every tensor in bfloat16.
CHECKPOINT_DTYPE = "BF16"
_KERNEL = 7


def _open_safetensors(path: Path):
    from safetensors import safe_open

    return safe_open(str(path), framework="numpy")


def _unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"vibevoice model.safetensors.index.json repeats the key {key!r}")
        result[key] = value
    return result


def _block(shapes: dict, prefix: str, channels: int) -> None:
    """One ConvNeXt-style block: RMSNorms, depthwise causal conv, 4x FFN, scales."""
    for name in ("norm.weight", "ffn_norm.weight", "gamma", "ffn_gamma"):
        shapes[prefix + name] = (channels,)
    shapes[prefix + "mixer.conv.conv.conv.weight"] = (channels, 1, _KERNEL)
    shapes[prefix + "mixer.conv.conv.conv.bias"] = (channels,)
    shapes[prefix + "ffn.linear1.weight"] = (4 * channels, channels)
    shapes[prefix + "ffn.linear1.bias"] = (4 * channels,)
    shapes[prefix + "ffn.linear2.weight"] = (channels, 4 * channels)
    shapes[prefix + "ffn.linear2.bias"] = (channels,)


def _encoder(shapes: dict, prefix: str, tokenizer) -> None:
    channels = tokenizer.stage_channels()
    shapes[prefix + "downsample_layers.0.0.conv.conv.weight"] = (channels[0], 1, _KERNEL)
    shapes[prefix + "downsample_layers.0.0.conv.conv.bias"] = (channels[0],)
    for index, ratio in enumerate(tokenizer.ratios):
        layer = f"{prefix}downsample_layers.{index + 1}.0.conv.conv."
        shapes[layer + "weight"] = (channels[index + 1], channels[index], 2 * ratio)
        shapes[layer + "bias"] = (channels[index + 1],)
    for stage, depth in enumerate(tokenizer.depths):
        for block in range(depth):
            _block(shapes, f"{prefix}stages.{stage}.{block}.", channels[stage])
    shapes[prefix + "head.conv.conv.weight"] = (tokenizer.vae_dim, channels[-1], _KERNEL)
    shapes[prefix + "head.conv.conv.bias"] = (tokenizer.vae_dim,)


def _decoder(shapes: dict, prefix: str, tokenizer) -> None:
    channels = list(reversed(tokenizer.stage_channels()))
    depths = list(reversed(tokenizer.depths))
    shapes[prefix + "upsample_layers.0.0.conv.conv.weight"] = (
        channels[0],
        tokenizer.vae_dim,
        _KERNEL,
    )
    shapes[prefix + "upsample_layers.0.0.conv.conv.bias"] = (channels[0],)
    for index, ratio in enumerate(reversed(tokenizer.ratios)):
        layer = f"{prefix}upsample_layers.{index + 1}.0.convtr.convtr."
        shapes[layer + "weight"] = (channels[index], channels[index + 1], 2 * ratio)
        shapes[layer + "bias"] = (channels[index + 1],)
    for stage, depth in enumerate(depths):
        for block in range(depth):
            _block(shapes, f"{prefix}stages.{stage}.{block}.", channels[stage])
    shapes[prefix + "head.conv.conv.weight"] = (tokenizer.channels, channels[-1], _KERNEL)
    shapes[prefix + "head.conv.conv.bias"] = (tokenizer.channels,)


def implied_tensor_count(cfg) -> int:
    """How many tensors expected_tensors(cfg) lists, computed without listing them, so a
    configuration with absurd depths is refused before any list is built."""

    def tower(tokenizer) -> int:
        # The stem and strided convolutions, ten tensors per block, the head convolution.
        return 2 * (1 + len(tokenizer.ratios)) + 10 * sum(tokenizer.depths) + 2

    language_model = 2 + 12 * cfg.num_layers
    prediction_head = 4 + 5 * cfg.head_layers + 2
    connectors_and_factors = 10 + 2
    return (
        language_model
        + 2 * tower(cfg.acoustic)
        + tower(cfg.semantic)
        + prediction_head
        + connectors_and_factors
    )


def expected_tensors(cfg) -> dict[str, tuple[int, ...]]:
    """Every tensor the engines read, with the shape the configuration gives it."""
    shapes: dict[str, tuple[int, ...]] = {}
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    query, kv = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
    base = "model.language_model."
    shapes[base + "embed_tokens.weight"] = (cfg.vocab_size, hidden)
    shapes[base + "norm.weight"] = (hidden,)
    for layer in range(cfg.num_layers):
        prefix = f"{base}layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (hidden,)
        shapes[prefix + "post_attention_layernorm.weight"] = (hidden,)
        shapes[prefix + "self_attn.q_proj.weight"] = (query, hidden)
        shapes[prefix + "self_attn.q_proj.bias"] = (query,)
        for name in ("k_proj", "v_proj"):
            shapes[f"{prefix}self_attn.{name}.weight"] = (kv, hidden)
            shapes[f"{prefix}self_attn.{name}.bias"] = (kv,)
        shapes[prefix + "self_attn.o_proj.weight"] = (hidden, query)
        shapes[prefix + "mlp.gate_proj.weight"] = (inter, hidden)
        shapes[prefix + "mlp.up_proj.weight"] = (inter, hidden)
        shapes[prefix + "mlp.down_proj.weight"] = (hidden, inter)

    _encoder(shapes, "model.acoustic_tokenizer.encoder.", cfg.acoustic)
    _decoder(shapes, "model.acoustic_tokenizer.decoder.", cfg.acoustic)
    _encoder(shapes, "model.semantic_tokenizer.encoder.", cfg.semantic)

    head, latent, head_inter = "model.prediction_head.", cfg.latent_size, cfg.head_intermediate_size
    shapes[head + "noisy_images_proj.weight"] = (hidden, latent)
    shapes[head + "cond_proj.weight"] = (hidden, hidden)
    shapes[head + "t_embedder.mlp.0.weight"] = (hidden, cfg.frequency_embedding_size)
    shapes[head + "t_embedder.mlp.2.weight"] = (hidden, hidden)
    for layer in range(cfg.head_layers):
        prefix = f"{head}layers.{layer}."
        shapes[prefix + "adaLN_modulation.1.weight"] = (3 * hidden, hidden)
        shapes[prefix + "norm.weight"] = (hidden,)
        shapes[prefix + "ffn.gate_proj.weight"] = (head_inter, hidden)
        shapes[prefix + "ffn.up_proj.weight"] = (head_inter, hidden)
        shapes[prefix + "ffn.down_proj.weight"] = (hidden, head_inter)
    shapes[head + "final_layer.adaLN_modulation.1.weight"] = (2 * hidden, hidden)
    shapes[head + "final_layer.linear.weight"] = (latent, hidden)
    for name, inputs in (
        ("acoustic_connector", cfg.acoustic.vae_dim),
        ("semantic_connector", cfg.semantic.vae_dim),
    ):
        shapes[f"model.{name}.fc1.weight"] = (hidden, inputs)
        shapes[f"model.{name}.fc1.bias"] = (hidden,)
        shapes[f"model.{name}.norm.weight"] = (hidden,)
        shapes[f"model.{name}.fc2.weight"] = (hidden, hidden)
        shapes[f"model.{name}.fc2.bias"] = (hidden,)
    shapes["model.speech_scaling_factor"] = ()
    shapes["model.speech_bias_factor"] = ()
    return shapes


class Checkpoint:
    """Name-addressed access to the tensors of one sharded original-layout checkpoint.

    The constructor reads only the safetensors headers and refuses the checkpoint
    unless it holds exactly the tensors ``cfg`` implies, each in bfloat16 with its
    configured shape: the engines take their widths from the weights, so a
    configuration that disagrees with them would otherwise build another model.
    """

    def __init__(self, model_dir: str | Path, cfg) -> None:
        root = Path(model_dir)
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"vibevoice checkpoint has no model.safetensors.index.json: {root}"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not all(
            isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()
        ):
            raise ValueError(f"vibevoice checkpoint index has no weight_map of names: {index_path}")
        self._shard_of = dict(weight_map)
        self._handles = {}
        headers: dict[str, tuple[str, tuple[int, ...]]] = {}
        for shard in sorted(set(weight_map.values())):
            path = root / shard
            if Path(shard).name != shard or not path.is_file():
                raise FileNotFoundError(f"vibevoice checkpoint shard is missing: {path}")
            handle = _open_safetensors(path)
            self._handles[shard] = handle
            listed = {name for name, owner in weight_map.items() if owner == shard}
            stored = set(handle.keys())
            if stored != listed:
                raise ValueError(
                    f"vibevoice checkpoint shard {shard} holds other tensors than its index lists"
                )
            for name in stored:
                tensor = handle.get_slice(name)
                headers[name] = (tensor.get_dtype(), tuple(tensor.get_shape()))
        implied = implied_tensor_count(cfg)
        if implied != len(headers):
            raise ValueError(
                f"vibevoice checkpoint holds {len(headers)} tensors; its configuration implies "
                f"{implied}"
            )
        expected = expected_tensors(cfg)
        missing = sorted(set(expected) - set(headers))
        unexpected = sorted(set(headers) - set(expected))
        if missing or unexpected:
            raise ValueError(
                "vibevoice checkpoint does not hold the tensors its configuration implies: "
                f"missing {missing[:5]}, unexpected {unexpected[:5]}"
            )
        for name, shape in expected.items():
            dtype, stored_shape = headers[name]
            if dtype != CHECKPOINT_DTYPE:
                raise ValueError(f"vibevoice checkpoint tensor {name} is {dtype}, not bfloat16")
            if stored_shape != shape:
                raise ValueError(
                    f"vibevoice checkpoint tensor {name} has shape {stored_shape}; "
                    f"the configuration implies {shape}"
                )

    def names(self) -> list[str]:
        return sorted(self._shard_of)

    def get(self, name: str) -> np.ndarray:
        shard = self._shard_of.get(name)
        if shard is None:
            raise KeyError(f"vibevoice checkpoint is missing tensor {name}")
        return self._handles[shard].get_tensor(name)

    def f32(self, name: str) -> np.ndarray:
        return np.asarray(self.get(name)).astype(np.float32)
