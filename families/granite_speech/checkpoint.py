# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to the safetensors checkpoint against an inventory derived from the config.

The inventory is computed from the geometry the graphs are built for, so a checkpoint with a
missing, extra or differently shaped tensor is refused by name before any engine is built.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 with numpy so safetensors can return it
import numpy as np

from .config import GraniteSpeechConfig

_INDEX = "model.safetensors.index.json"
_SINGLE = "model.safetensors"
_FLOATS = {np.dtype(np.float32), np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16)}


def _open_safetensors(path: Path):
    from safetensors import safe_open

    return safe_open(str(path), framework="numpy")


def _encoder_inventory(cfg: GraniteSpeechConfig, out: dict) -> None:
    e = cfg.encoder
    hidden = e.hidden_dim
    inner = e.num_heads * e.dim_head
    ff = hidden * e.feedforward_mult
    conv = hidden * e.conv_expansion_factor

    def linear(name, rows, cols, bias=True):
        out[name + ".weight"] = (rows, cols)
        if bias:
            out[name + ".bias"] = (rows,)

    def norm(name, width):
        out[name + ".weight"] = (width,)
        out[name + ".bias"] = (width,)

    linear("encoder.input_linear", hidden, e.input_dim)
    for index in range(e.num_layers):
        base = f"encoder.layers.{index}."
        for half in ("ff1", "ff2"):
            norm(base + half + ".pre_norm", hidden)
            linear(base + half + ".up_proj", ff, hidden)
            linear(base + half + ".down_proj", hidden, ff)
        norm(base + "attn.pre_norm", hidden)
        linear(base + "attn.to_q", inner, hidden, bias=False)
        linear(base + "attn.to_kv", 2 * inner, hidden, bias=False)
        linear(base + "attn.to_out", hidden, inner)
        out[base + "attn.rel_pos_emb.weight"] = (2 * e.max_pos_emb + 1, e.dim_head)
        norm(base + "conv.norm", hidden)
        out[base + "conv.up_conv.weight"] = (2 * conv, hidden, 1)
        out[base + "conv.up_conv.bias"] = (2 * conv,)
        out[base + "conv.depth_conv.conv.weight"] = (conv, 1, e.conv_kernel_size)
        norm(base + "conv.batch_norm", conv)
        out[base + "conv.batch_norm.running_mean"] = (conv,)
        out[base + "conv.batch_norm.running_var"] = (conv,)
        out[base + "conv.batch_norm.num_batches_tracked"] = ()
        out[base + "conv.down_conv.weight"] = (hidden, conv, 1)
        out[base + "conv.down_conv.bias"] = (hidden,)
        norm(base + "post_norm", hidden)
    linear("encoder.out", e.output_dim, hidden)
    linear("encoder.out_mid", hidden, e.output_dim)


def _projector_inventory(cfg: GraniteSpeechConfig, out: dict) -> None:
    p = cfg.projector
    width = p.hidden_size
    out["projector.query"] = (1, p.num_queries, width)
    out["projector.qformer.layernorm.weight"] = (width,)
    out["projector.qformer.layernorm.bias"] = (width,)
    for index in range(p.num_layers):
        base = f"projector.qformer.encoder.layer.{index}."
        blocks = ["attention"] + (["crossattention"] if p.has_cross_attention(index) else [])
        for block in blocks:
            source = cfg.encoder.hidden_dim if block == "crossattention" else width
            for name, cols in (("query", width), ("key", source), ("value", source)):
                out[f"{base}{block}.attention.{name}.weight"] = (width, cols)
                out[f"{base}{block}.attention.{name}.bias"] = (width,)
            out[f"{base}{block}.output.dense.weight"] = (width, width)
            out[f"{base}{block}.output.dense.bias"] = (width,)
            out[f"{base}{block}.output.LayerNorm.weight"] = (width,)
            out[f"{base}{block}.output.LayerNorm.bias"] = (width,)
        out[base + "intermediate_query.dense.weight"] = (p.intermediate_size, width)
        out[base + "intermediate_query.dense.bias"] = (p.intermediate_size,)
        out[base + "output_query.dense.weight"] = (width, p.intermediate_size)
        out[base + "output_query.dense.bias"] = (width,)
        out[base + "output_query.LayerNorm.weight"] = (width,)
        out[base + "output_query.LayerNorm.bias"] = (width,)
    out["projector.linear.weight"] = (cfg.text.hidden_size, width)
    out["projector.linear.bias"] = (cfg.text.hidden_size,)


def _text_inventory(cfg: GraniteSpeechConfig, out: dict) -> None:
    t = cfg.text
    hidden = t.hidden_size
    out["language_model.model.embed_tokens.weight"] = (t.vocab_size, hidden)
    for index in range(t.num_layers):
        base = f"language_model.model.layers.{index}."
        out[base + "input_layernorm.weight"] = (hidden,)
        out[base + "post_attention_layernorm.weight"] = (hidden,)
        out[base + "self_attn.q_proj.weight"] = (t.num_heads * t.head_dim, hidden)
        out[base + "self_attn.k_proj.weight"] = (t.kv_width, hidden)
        out[base + "self_attn.v_proj.weight"] = (t.kv_width, hidden)
        out[base + "self_attn.o_proj.weight"] = (hidden, t.num_heads * t.head_dim)
        out[base + "mlp.gate_proj.weight"] = (t.intermediate_size, hidden)
        out[base + "mlp.up_proj.weight"] = (t.intermediate_size, hidden)
        out[base + "mlp.down_proj.weight"] = (hidden, t.intermediate_size)
    out["language_model.model.norm.weight"] = (hidden,)
    out["language_model.lm_head.weight"] = (t.vocab_size, hidden)


def inventory(cfg: GraniteSpeechConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the reference module tree holds, with its shape."""
    out: dict[str, tuple[int, ...]] = {}
    _encoder_inventory(cfg, out)
    _projector_inventory(cfg, out)
    _text_inventory(cfg, out)
    return out


def _weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / _INDEX
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            mapping = json.load(handle).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"granite_speech checkpoint index has no weight_map: {index_path}")
        return {str(name): str(shard) for name, shard in mapping.items()}
    single = model_dir / _SINGLE
    if not single.is_file():
        raise FileNotFoundError(f"granite_speech needs {_INDEX} or {_SINGLE} in {model_dir}")
    with _open_safetensors(single) as handle:
        return {name: _SINGLE for name in handle.keys()}


class Checkpoint:
    """Opened shards plus the expected inventory; tensors are read one at a time."""

    def __init__(self, model_dir: Path, cfg: GraniteSpeechConfig):
        self._dir = Path(model_dir)
        self._expected = inventory(cfg)
        self._map = _weight_map(self._dir)
        missing = sorted(set(self._expected) - set(self._map))
        unexpected = sorted(set(self._map) - set(self._expected))
        if missing or unexpected:
            raise NotImplementedError(
                "granite_speech checkpoint does not match the built module tree: "
                f"missing {missing[:8]}{'...' if len(missing) > 8 else ''}, "
                f"unexpected {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        self._handles: dict[str, object] = {}

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def raw(self, name: str) -> np.ndarray:
        shard = self._map[name]
        handle = self._handles.get(shard)
        if handle is None:
            path = self._dir / shard
            if not path.is_file():
                raise FileNotFoundError(f"granite_speech checkpoint shard is missing: {path}")
            handle = _open_safetensors(path).__enter__()
            self._handles[shard] = handle
        value = handle.get_tensor(name)
        if tuple(value.shape) != self._expected[name]:
            raise NotImplementedError(
                f"granite_speech tensor {name} has shape {tuple(value.shape)}, "
                f"not {self._expected[name]}")
        return value

    def tensor(self, name: str) -> np.ndarray:
        """A floating tensor as float32 (exact for BF16 and FP16 checkpoints)."""
        value = self.raw(name)
        if value.dtype not in _FLOATS:
            raise NotImplementedError(f"granite_speech tensor {name} is {value.dtype}, not floating point")
        return np.asarray(value, dtype=np.float32)
