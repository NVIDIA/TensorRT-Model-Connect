# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map Hugging Face LFM2-MoE checkpoints to the family-owned TRT graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .config import Lfm2MoeConfig, validate_lfm2_moe_config


try:
    import ml_dtypes  # noqa: F401
except ImportError:
    pass


class WeightDict(dict):
    """Flat, shape-checked float32 weights consumed by ``model.py``."""


class _TorchBinReader:
    def __init__(self, path: Path):
        import torch

        self._state = torch.load(str(path), map_location="cpu", weights_only=True)

    def keys(self) -> list[str]:
        return list(self._state.keys())

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {}
            for reader in readers:
                for key in reader.keys():
                    tensor_map[key] = reader
        self.tensor_map = tensor_map


def _detect_framework() -> str:
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    framework = _detect_framework()
    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([safe_open(str(single), framework=framework)])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        shard_names = sorted(set(str(value) for value in weight_map.values()))
        readers_by_name = {
            name: safe_open(str(model_dir / name), framework=framework) for name in shard_names
        }
        return _ReaderCollection(
            [readers_by_name[name] for name in shard_names],
            tensor_map={
                str(tensor_name): readers_by_name[str(shard_name)]
                for tensor_name, shard_name in weight_map.items()
            },
        )

    torch_bin = model_dir / "pytorch_model.bin"
    if torch_bin.is_file():
        return _ReaderCollection([_TorchBinReader(torch_bin)])
    raise FileNotFoundError(
        f"No model.safetensors, model.safetensors.index.json, or pytorch_model.bin in {model_dir}"
    )


def _has_tensor(readers: list, name: str) -> bool:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        return name in tensor_map
    return any(name in reader.keys() for reader in readers)


def _to_numpy_fp32(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == np.uint16 or str(array.dtype) == "bfloat16":
        words = array.view(np.uint16).astype(np.uint32) << 16
        return words.view(np.float32)
    return np.asarray(array, dtype=np.float32)


def _load_tensor(readers: list, name: str) -> np.ndarray:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return _to_numpy_fp32(reader.get_tensor(name))
    for reader in readers:
        if name in reader.keys():
            return _to_numpy_fp32(reader.get_tensor(name))
    raise KeyError(f"Tensor not found: {name}")


def _expect(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if tuple(array.shape) != tuple(shape):
        raise ValueError(
            f"LFM2-MoE tensor {name} must have shape {shape}, got {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array)


def _load_exact(readers: list, name: str, shape: tuple[int, ...]) -> np.ndarray:
    return _expect(_load_tensor(readers, name), shape, name)


def _transpose(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"LFM2-MoE tensor {name} must be rank 2")
    return np.ascontiguousarray(array.T, dtype=np.float32)


def _store_conv_biases(
    weights: WeightDict,
    readers: list,
    conv_prefix: str,
    target_prefix: str,
    hidden_size: int,
    *,
    required: bool,
) -> None:
    specs = (
        (f"{conv_prefix}.in_proj.bias", f"{target_prefix}.conv_in_bias", 3 * hidden_size),
        (f"{conv_prefix}.conv.bias", f"{target_prefix}.conv_bias", hidden_size),
        (f"{conv_prefix}.out_proj.bias", f"{target_prefix}.conv_out_bias", hidden_size),
    )
    present = {source_name for source_name, _, _ in specs if _has_tensor(readers, source_name)}
    if required:
        missing = [source_name for source_name, _, _ in specs if source_name not in present]
        if missing:
            raise ValueError(
                "LFM2-MoE conv_bias=true requires all conv bias tensors; missing: "
                + ", ".join(missing)
            )
    elif present:
        raise ValueError(
            "LFM2-MoE conv_bias=false but checkpoint contains conv bias tensors: "
            + ", ".join(sorted(present))
        )
    else:
        return
    for source_name, target_name, width in specs:
        weights[target_name] = _load_exact(readers, source_name, (width,))


def _store_moe_feed_forward(
    weights: WeightDict,
    readers: list,
    hf_prefix: str,
    prefix: str,
    cfg: Lfm2MoeConfig,
) -> None:
    """Stack routed-expert weights into the two gather-ready constants.

    Orientation contract consumed by ``model.py``:

    ``layer.{i}.moe_w13``: shape [num_experts, hidden, 2 * moe_intermediate].
    Per expert, HF ``w1.weight`` [inter, hidden] and ``w3.weight``
    [inter, hidden] are transposed so a row-vector activation multiplies as
    ``x @ W``; the transposed w1 occupies columns [0, inter) (the SiLU gate)
    and the transposed w3 occupies columns [inter, 2*inter) (the up
    projection).

    ``layer.{i}.moe_w2``: shape [num_experts, moe_intermediate, hidden]. Per
    expert, HF ``w2.weight`` [hidden, inter] is transposed for the same
    row-vector convention, so ``(silu(x@w1) * (x@w3)) @ w2`` lands back in
    hidden space.
    """

    hidden = cfg.hidden_size
    inter = cfg.moe_intermediate_size
    experts = cfg.num_experts

    stacked_w13 = np.empty((experts, hidden, 2 * inter), dtype=np.float32)
    stacked_w2 = np.empty((experts, inter, hidden), dtype=np.float32)
    for expert in range(cfg.num_experts):
        expert_prefix = f"{hf_prefix}.feed_forward.experts.{expert}"
        w1 = _load_exact(readers, f"{expert_prefix}.w1.weight", (inter, hidden))
        w3 = _load_exact(readers, f"{expert_prefix}.w3.weight", (inter, hidden))
        w2 = _load_exact(readers, f"{expert_prefix}.w2.weight", (hidden, inter))
        stacked_w13[expert, :, :inter] = w1.T
        stacked_w13[expert, :, inter:] = w3.T
        stacked_w2[expert] = w2.T
    weights[f"{prefix}.moe_w13"] = np.ascontiguousarray(stacked_w13)
    weights[f"{prefix}.moe_w2"] = np.ascontiguousarray(stacked_w2)

    weights[f"{prefix}.router"] = _load_exact(
        readers,
        f"{hf_prefix}.feed_forward.gate.weight",
        (experts, hidden),
    )
    weights[f"{prefix}.expert_bias"] = _load_exact(
        readers,
        f"{hf_prefix}.feed_forward.expert_bias",
        (experts,),
    )


def load_lfm2_moe_weights(
    model_dir: str,
    config: object,
    *,
    precision: str = "bf16",
) -> WeightDict:
    """Load LFM2-MoE checkpoints without importing Transformers."""

    if precision not in {"fp16", "bf16"}:
        raise ValueError("LFM2-MoE supports only fp16 and bf16 builds")
    cfg = validate_lfm2_moe_config(config)
    readers = _open_safetensors(Path(model_dir))
    hidden = cfg.hidden_size
    vocab = cfg.vocab_size
    mlp = cfg.intermediate_size
    attention = cfg.attention_size
    kv_attention = cfg.kv_attention_size

    weights = WeightDict()
    embedding_name = "model.embed_tokens.weight"
    weights["embedding"] = _load_exact(readers, embedding_name, (vocab, hidden))

    for layer_index, layer_type in enumerate(cfg.layer_types):
        hf_prefix = f"model.layers.{layer_index}"
        prefix = f"layer.{layer_index}"
        weights[f"{prefix}.operator_norm"] = _load_exact(
            readers,
            f"{hf_prefix}.operator_norm.weight",
            (hidden,),
        )
        weights[f"{prefix}.ffn_norm"] = _load_exact(
            readers,
            f"{hf_prefix}.ffn_norm.weight",
            (hidden,),
        )

        if layer_type == "conv":
            conv_prefix = f"{hf_prefix}.conv"
            weights[f"{prefix}.conv_in"] = _transpose(
                _load_exact(
                    readers,
                    f"{conv_prefix}.in_proj.weight",
                    (3 * hidden, hidden),
                ),
                f"{conv_prefix}.in_proj.weight",
            )
            conv_weight = _load_tensor(readers, f"{conv_prefix}.conv.weight")
            if tuple(conv_weight.shape) == (hidden, 1, cfg.conv_l_cache):
                conv_weight = conv_weight[:, 0, :]
            weights[f"{prefix}.conv_weight"] = _expect(
                conv_weight,
                (hidden, cfg.conv_l_cache),
                f"{conv_prefix}.conv.weight",
            )
            weights[f"{prefix}.conv_out"] = _transpose(
                _load_exact(
                    readers,
                    f"{conv_prefix}.out_proj.weight",
                    (hidden, hidden),
                ),
                f"{conv_prefix}.out_proj.weight",
            )
            _store_conv_biases(
                weights,
                readers,
                conv_prefix,
                prefix,
                hidden,
                required=cfg.conv_bias,
            )
        else:
            attention_prefix = f"{hf_prefix}.self_attn"
            for source, target, shape in (
                ("q_proj", "w_q", (attention, hidden)),
                ("k_proj", "w_k", (kv_attention, hidden)),
                ("v_proj", "w_v", (kv_attention, hidden)),
                ("out_proj", "w_o", (hidden, attention)),
            ):
                source_name = f"{attention_prefix}.{source}.weight"
                weights[f"{prefix}.{target}"] = _transpose(
                    _load_exact(readers, source_name, shape),
                    source_name,
                )
            weights[f"{prefix}.q_norm"] = _load_exact(
                readers,
                f"{attention_prefix}.q_layernorm.weight",
                (cfg.head_dim,),
            )
            weights[f"{prefix}.k_norm"] = _load_exact(
                readers,
                f"{attention_prefix}.k_layernorm.weight",
                (cfg.head_dim,),
            )

        if cfg.is_moe_layer(layer_index):
            _store_moe_feed_forward(weights, readers, hf_prefix, prefix, cfg)
        else:
            for source, target, shape in (
                ("w1", "w1", (mlp, hidden)),
                ("w3", "w3", (mlp, hidden)),
                ("w2", "w2", (hidden, mlp)),
            ):
                source_name = f"{hf_prefix}.feed_forward.{source}.weight"
                weights[f"{prefix}.{target}"] = _transpose(
                    _load_exact(readers, source_name, shape),
                    source_name,
                )

    weights["final_norm"] = _load_exact(
        readers,
        "model.embedding_norm.weight",
        (hidden,),
    )
    if _has_tensor(readers, "lm_head.weight"):
        weights["w_lm_head"] = _transpose(
            _load_exact(readers, "lm_head.weight", (vocab, hidden)),
            "lm_head.weight",
        )
    elif cfg.tie_word_embeddings:
        weights["w_lm_head"] = np.ascontiguousarray(
            weights["embedding"].T,
            dtype=np.float32,
        )
    else:
        raise ValueError("untied LFM2-MoE checkpoint is missing lm_head.weight")

    weights["_layer_types"] = list(cfg.layer_types)
    weights["_num_attention_layers"] = cfg.num_attention_layers
    weights["_num_conv_layers"] = cfg.num_conv_layers
    weights["_hidden_size"] = cfg.hidden_size
    weights["_intermediate_size"] = cfg.intermediate_size
    weights["_head_dim"] = cfg.head_dim
    weights["_conv_l_cache"] = cfg.conv_l_cache
    weights["_num_experts"] = cfg.num_experts
    weights["_num_experts_per_tok"] = cfg.num_experts_per_tok
    weights["_moe_intermediate_size"] = cfg.moe_intermediate_size
    weights["_num_dense_layers"] = cfg.num_dense_layers
    return weights
