# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the safetensors checkpoint against an inventory derived from the configuration.

Every tensor's name, dtype and shape is checked from the file headers before any value is read:
a missing, extra, differently typed or differently shaped tensor is refused by name before an
engine is built. Values are read one tensor at a time and must be finite.
"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 with numpy so safetensors can return it
import numpy as np

from .config import GraniteSpeech5Config, read_json

_INDEX = "model.safetensors.index.json"
_SINGLE = "model.safetensors"
_FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})
_COUNTER = "num_batches_tracked"  # BatchNorm's training counter: stored, never used for inference


def _open(path: Path):
    from safetensors import safe_open

    return safe_open(str(path), framework="numpy")


def inventory(cfg: GraniteSpeech5Config) -> dict[str, tuple[int, ...]]:
    """Every tensor the reference module tree stores, with its shape."""
    e = cfg.encoder
    hidden, width, channels = e.hidden_size, e.attention_width, e.conv_channels
    out: dict[str, tuple[int, ...]] = {}

    def linear(name, rows, cols, bias=True):
        out[name + ".weight"] = (rows, cols)
        if bias:
            out[name + ".bias"] = (rows,)

    def norm(name, size):
        out[name + ".weight"] = (size,)
        out[name + ".bias"] = (size,)

    linear("encoder.input_linear", hidden, e.input_width)
    for index in range(e.num_layers):
        base = f"encoder.layers.{index}."
        for block in ("feed_forward1", "feed_forward2"):
            linear(base + block + ".linear1", e.intermediate_size, hidden, e.feed_forward_bias)
            linear(base + block + ".linear2", hidden, e.intermediate_size, e.feed_forward_bias)
        for name in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            norm(base + name, hidden)
        for name in ("q_proj", "k_proj", "v_proj"):
            linear(base + "self_attn." + name, width, hidden, bias=False)
        linear(base + "self_attn.o_proj", hidden, width)
        out[base + "self_attn.rel_pos_emb.weight"] = (2 * e.max_position_embeddings + 1, e.head_dim)
        linear(base + "conv.pointwise_lin1", 2 * channels, hidden)
        out[base + "conv.depthwise_conv.weight"] = (channels, 1, e.conv_kernel_size)
        norm(base + "conv.norm", channels)
        out[base + "conv.norm.running_mean"] = (channels,)
        out[base + "conv.norm.running_var"] = (channels,)
        out[base + "conv.norm." + _COUNTER] = ()
        linear(base + "conv.pointwise_lin2", hidden, channels)
    linear("encoder.out", cfg.vocab_size, hidden)
    linear("encoder.out_mid", hidden, cfg.vocab_size)
    return out


def _weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / _INDEX
    if index_path.is_file():
        mapping = read_json(index_path).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"granite_speech5 checkpoint index has no weight_map: {index_path}")
        if not all(isinstance(shard, str) and shard and Path(shard).name == shard for shard in mapping.values()):
            raise ValueError("granite_speech5 checkpoint index names a shard outside the checkpoint directory")
        return dict(mapping)
    if not (model_dir / _SINGLE).is_file():
        raise FileNotFoundError(f"granite_speech5 needs {_INDEX} or {_SINGLE} in {model_dir}")
    with _open(model_dir / _SINGLE) as handle:
        return {name: _SINGLE for name in handle.keys()}


class Checkpoint:
    """Opened shards, checked against the expected inventory; tensors are read one at a time."""

    def __init__(self, model_dir: Path, cfg: GraniteSpeech5Config):
        self._dir = Path(model_dir)
        self._expected = inventory(cfg)
        self._map = _weight_map(self._dir)
        missing = sorted(set(self._expected) - set(self._map))
        unexpected = sorted(set(self._map) - set(self._expected))
        if missing or unexpected:
            raise NotImplementedError(
                "granite_speech5 checkpoint does not match the module tree the configuration describes: "
                f"missing {missing[:8]}{'...' if len(missing) > 8 else ''}, "
                f"unexpected {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        self._handles: dict[str, object] = {}
        try:
            self._check_headers()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()

    def _handle(self, name: str):
        shard = self._map[name]
        handle = self._handles.get(shard)
        if handle is None:
            path = self._dir / shard
            if not path.is_file():
                raise FileNotFoundError(f"granite_speech5 checkpoint shard is missing: {path}")
            handle = _open(path).__enter__()
            self._handles[shard] = handle
        return handle

    def _check_headers(self) -> None:
        """Dtype and shape of every tensor from the headers, before any value is read."""
        for name, shape in self._expected.items():
            header = self._handle(name).get_slice(name)
            dtype, stored = str(header.get_dtype()), tuple(int(v) for v in header.get_shape())
            allowed = {"I64"} if name.endswith(_COUNTER) else _FLOAT_DTYPES
            if dtype not in allowed:
                raise NotImplementedError(f"granite_speech5 tensor {name} is {dtype}, not {'/'.join(sorted(allowed))}")
            if stored != shape:
                raise NotImplementedError(f"granite_speech5 tensor {name} has shape {stored}, not {shape}")

    def tensor(self, name: str) -> np.ndarray:
        """A floating tensor as float32 (exact for BF16 and FP16 values); refused unless every value is finite."""
        if name.endswith(_COUNTER):
            raise KeyError(f"granite_speech5 never reads {name}")
        value = np.asarray(self._handle(name).get_tensor(name), dtype=np.float32)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"granite_speech5 tensor {name} holds non-finite values")
        return value
