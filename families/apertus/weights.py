# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate the Apertus safetensors checkpoint for one build precision.

Projection matrices keep the Hugging Face ``[out, in]`` layout; the graph
multiplies by their transpose. A BF16 build keeps BF16 tensors bit-exact; an
FP32 build widens them exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

from .config import ApertusConfig

_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))
_SOFTPLUS_THRESHOLD = 20.0


@dataclass(frozen=True)
class LayerWeights:
    attention_norm: np.ndarray
    q_proj: np.ndarray
    k_proj: np.ndarray
    v_proj: np.ndarray
    o_proj: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray
    feedforward_norm: np.ndarray
    up_proj: np.ndarray
    down_proj: np.ndarray
    # xIELU coefficients after their fixed reparameterization, in FP32:
    # alpha_p = softplus(alpha_p_raw), alpha_n = beta + softplus(alpha_n_raw).
    xielu_alpha_p: float
    xielu_alpha_n: float
    xielu_beta: float
    xielu_eps: float


@dataclass(frozen=True)
class ApertusWeights:
    embedding: np.ndarray
    layers: tuple[LayerWeights, ...]
    final_norm: np.ndarray
    lm_head: np.ndarray


def _open_safetensors(model_dir: Path) -> dict[str, object]:
    """Return one numpy safetensors reader per tensor name."""
    single = model_dir / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework="numpy")
        return {name: reader for name in reader.keys()}
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no Apertus safetensors checkpoint in {model_dir}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    readers = {
        shard: safe_open(str(model_dir / shard), framework="numpy")
        for shard in sorted(set(weight_map.values()))
    }
    return {name: readers[shard] for name, shard in weight_map.items()}


def softplus(value: float) -> float:
    """``torch.nn.functional.softplus`` in FP32 with its default beta and threshold."""
    x = np.float32(value)
    if x > _SOFTPLUS_THRESHOLD:
        return float(x)
    return float(np.log1p(np.exp(x, dtype=np.float32), dtype=np.float32))


class _Reader:
    def __init__(self, model_dir: Path, storage: np.dtype) -> None:
        self._readers = _open_safetensors(model_dir)
        self._storage = storage
        self._used: set[str] = set()

    def tensor(
        self, name: str, shape: tuple[int, ...], dtype: np.dtype | None = None
    ) -> np.ndarray:
        reader = self._readers.get(name)
        if reader is None:
            raise ValueError(f"Apertus checkpoint is missing tensor {name}")
        value = reader.get_tensor(name)
        if value.dtype not in _FLOAT_DTYPES:
            raise ValueError(f"Apertus tensor {name} has unsupported dtype {value.dtype}")
        if tuple(value.shape) != shape:
            raise ValueError(f"Apertus tensor {name} must have shape {shape}, got {value.shape}")
        self._used.add(name)
        return np.ascontiguousarray(value, dtype=dtype or self._storage)

    def scalar(self, name: str, shape: tuple[int, ...]) -> float:
        value = self.tensor(name, shape, np.dtype(np.float32)).reshape(-1)
        if not np.isfinite(value[0]):
            raise ValueError(f"Apertus tensor {name} must be finite")
        return float(value[0])

    def require_all_used(self) -> None:
        unused = sorted(set(self._readers) - self._used)
        if unused:
            raise ValueError(
                "Apertus checkpoint has tensors this graph does not use: " + ", ".join(unused[:8])
            )


def load_weights(model_dir: Path, config: ApertusConfig, precision: str) -> ApertusWeights:
    storage = {"bf16": np.dtype(ml_dtypes.bfloat16), "fp32": np.dtype(np.float32)}[precision]
    reader = _Reader(model_dir, storage)
    hidden, kv, attention = config.hidden_size, config.kv_size, config.attention_size
    mlp, head_dim = config.intermediate_size, config.head_dim
    layers = []
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        act = f"{prefix}.mlp.act_fn"
        beta = reader.scalar(f"{act}.beta", ())
        layers.append(
            LayerWeights(
                attention_norm=reader.tensor(f"{prefix}.attention_layernorm.weight", (hidden,)),
                q_proj=reader.tensor(f"{prefix}.self_attn.q_proj.weight", (attention, hidden)),
                k_proj=reader.tensor(f"{prefix}.self_attn.k_proj.weight", (kv, hidden)),
                v_proj=reader.tensor(f"{prefix}.self_attn.v_proj.weight", (kv, hidden)),
                o_proj=reader.tensor(f"{prefix}.self_attn.o_proj.weight", (hidden, attention)),
                q_norm=reader.tensor(f"{prefix}.self_attn.q_norm.weight", (head_dim,)),
                k_norm=reader.tensor(f"{prefix}.self_attn.k_norm.weight", (head_dim,)),
                feedforward_norm=reader.tensor(f"{prefix}.feedforward_layernorm.weight", (hidden,)),
                up_proj=reader.tensor(f"{prefix}.mlp.up_proj.weight", (mlp, hidden)),
                down_proj=reader.tensor(f"{prefix}.mlp.down_proj.weight", (hidden, mlp)),
                xielu_alpha_p=softplus(reader.scalar(f"{act}.alpha_p", (1,))),
                xielu_alpha_n=float(
                    np.float32(beta) + np.float32(softplus(reader.scalar(f"{act}.alpha_n", (1,))))
                ),
                xielu_beta=beta,
                xielu_eps=reader.scalar(f"{act}.eps", ()),
            )
        )
    result = ApertusWeights(
        embedding=reader.tensor("model.embed_tokens.weight", (config.vocab_size, hidden)),
        layers=tuple(layers),
        final_norm=reader.tensor("model.norm.weight", (hidden,)),
        lm_head=reader.tensor("lm_head.weight", (config.vocab_size, hidden)),
    )
    reader.require_all_used()
    return result
