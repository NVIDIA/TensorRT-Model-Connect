# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT family model -- encoder-only transformer with modern design.

ModernBERT differs significantly from classic BERT:
  - PRE-norm with LayerNorm (no bias) -- NOT RMSNorm despite weight naming
  - Fused QKV projection (Wqkv) -- split into Q/K/V
  - GeGLU MLP (fused Wi gate+up, Wo down) -- split Wi into gate/up
  - RoPE position encoding with per-layer theta (full_attention=160000, sliding=10000)
  - No token type embeddings
  - No attention bias, no MLP bias
  - Layer 0 has no attn_norm (identity)
"""

from __future__ import annotations

import json
import re
import tempfile
import time

import sys
from pathlib import Path

import numpy as np

from .config import ModelConfig, resolve_attention_contract
from .weights import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ...parallel_config import (
    ParallelConfig,
    add_all_reduce_sum,
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt() if trt_compat.is_available() else None

graph_ops = sys.modules[__name__]


def _add_layernorm_no_bias(
    network,
    inp,
    hidden_size,
    gamma,
    eps,
    *,
    dtype=np.float32,
):
    """LayerNorm without bias via TRT native normalization.

    ModernBERT uses nn.LayerNorm(bias=False) which still mean-centers,
    unlike RMSNorm which does not.
    """
    beta = np.zeros(hidden_size, dtype=dtype)
    return graph_ops.add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


name = "modernbert"
runtime_strategy = "modernbert_encoder_only"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower().startswith("modernbert")


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate = config.intermediate_size

    weights = WeightDict()

    # Word embedding
    embedding = _load_tensor(readers, "model.embeddings.tok_embeddings.weight")
    assert embedding.shape == (config.vocab_size, hidden)
    weights["embedding"] = embedding.astype(np.float32)

    # Embedding LayerNorm (no bias)
    weights["embed_norm"] = _load_tensor(readers, "model.embeddings.norm.weight").astype(np.float32)

    # Final LayerNorm
    weights["final_norm"] = _load_tensor(readers, "model.final_norm.weight").astype(np.float32)

    # MLM head weights (optional)
    if _has_tensor(readers, "head.dense.weight"):
        weights["head_dense_w"] = np.ascontiguousarray(
            _load_tensor(readers, "head.dense.weight").T.astype(np.float32)
        )
    if _has_tensor(readers, "head.norm.weight"):
        weights["head_norm"] = _load_tensor(readers, "head.norm.weight").astype(np.float32)
    if _has_tensor(readers, "decoder.bias"):
        weights["decoder_bias"] = _load_tensor(readers, "decoder.bias").astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        # Attention LayerNorm (layer 0 has no attn_norm)
        attn_norm_key = f"{hf_prefix}.attn_norm.weight"
        if _has_tensor(readers, attn_norm_key):
            weights[f"{prefix}.attn_norm"] = _load_tensor(readers, attn_norm_key).astype(np.float32)

        # Fused QKV: [3*hidden, hidden] -> split into Q, K, V
        wqkv = _load_tensor(readers, f"{hf_prefix}.attn.Wqkv.weight")
        assert wqkv.shape == (3 * hidden, hidden)
        q_w, k_w, v_w = np.split(wqkv, 3, axis=0)
        weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
        weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
        weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

        # Output projection
        wo = _load_tensor(readers, f"{hf_prefix}.attn.Wo.weight")
        weights[f"{prefix}.w_o"] = np.ascontiguousarray(wo.T.astype(np.float32))

        # MLP LayerNorm
        weights[f"{prefix}.mlp_norm"] = _load_tensor(
            readers, f"{hf_prefix}.mlp_norm.weight"
        ).astype(np.float32)

        # GeGLU MLP: Wi [2*intermediate, hidden] -> split into input, gate
        wi = _load_tensor(readers, f"{hf_prefix}.mlp.Wi.weight")
        assert wi.shape == (2 * intermediate, hidden)
        input_w, gate_w = np.split(wi, 2, axis=0)
        weights[f"{prefix}.w_mlp_input"] = np.ascontiguousarray(input_w.T.astype(np.float32))
        weights[f"{prefix}.w_mlp_gate"] = np.ascontiguousarray(gate_w.T.astype(np.float32))

        # Down projection
        mlp_wo = _load_tensor(readers, f"{hf_prefix}.mlp.Wo.weight")
        weights[f"{prefix}.w_down"] = np.ascontiguousarray(mlp_wo.T.astype(np.float32))

    return weights


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="ModernBERT tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("ModernBERT tensor-parallel builds do not support quantization")

        return build_tp_modernbert_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            verbose=verbose,
            parallel_config=parallel,
        )

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    intermediate = config.intermediate_size
    eps = config.raw.get("norm_eps", config.rms_norm_eps)
    max_seq = max_cache_length
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported ModernBERT precision: {precision}")

    attention_contract = resolve_attention_contract(config)
    layer_types = attention_contract.layer_types
    full_theta = attention_contract.full_rope_theta
    sliding_theta = attention_contract.sliding_rope_theta

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # Inputs
    input_ids = network.add_input("input_ids", trt.int32, (max_seq,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq,))

    # Attention mask: [seq] int -> [1, 1, 1, seq] additive float mask.
    mask_float = network.add_cast(attention_mask_input, work_trt_dtype)
    ones_c = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
    )
    mask_penalty = -1e4 if precision == "fp16" else -1e10
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([mask_penalty], dtype=work_np_dtype), dtype=work_np_dtype
    )
    inv_mask = network.add_elementwise(
        ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )
    pad_mask_4d = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_4d.reshape_dims = (1, 1, 1, max_seq)

    # Pre-compute RoPE tables for both theta values
    rope_tables = {}
    for theta in set([full_theta, sliding_theta]):
        cos = graph_ops.add_constant(
            network,
            (max_seq, head_dim // 2),
            graph_ops.make_rope_table_half_dim(max_seq, head_dim, theta, cosine=True),
            dtype=work_np_dtype,
        )
        sin = graph_ops.add_constant(
            network,
            (max_seq, head_dim // 2),
            graph_ops.make_rope_table_half_dim(max_seq, head_dim, theta, cosine=False),
            dtype=work_np_dtype,
        )
        rope_tables[theta] = (cos, sin)

    pos_indices = graph_ops.add_constant(
        network, (max_seq,), np.arange(max_seq, dtype=np.int32), dtype=np.int32
    )

    # Embedding
    embed_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )
    word_embed = network.add_gather(embed_table, input_ids, 0)
    hidden_state = _add_layernorm_no_bias(
        network, word_embed.get_output(0), hidden, weights["embed_norm"], eps, dtype=work_np_dtype
    )

    # Encoder layers
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Determine RoPE theta for this layer
        lt = layer_types[layer_idx]
        if lt in ("full_attention", "global_attention"):
            theta = full_theta
        else:
            theta = sliding_theta
        cos_table, sin_table = rope_tables[theta]

        # Pre-norm attention
        has_attn_norm = f"{prefix}.attn_norm" in weights
        if has_attn_norm:
            attn_input = _add_layernorm_no_bias(
                network,
                hidden_state,
                hidden,
                weights[f"{prefix}.attn_norm"],
                eps,
                dtype=work_np_dtype,
            )
        else:
            attn_input = hidden_state

        # QKV projections
        q = graph_ops.add_matmul_rhs_constant(
            network, attn_input, hidden, hidden, weights[f"{prefix}.w_q"], dtype=work_np_dtype
        )
        k = graph_ops.add_matmul_rhs_constant(
            network, attn_input, hidden, hidden, weights[f"{prefix}.w_k"], dtype=work_np_dtype
        )
        v = graph_ops.add_matmul_rhs_constant(
            network, attn_input, hidden, hidden, weights[f"{prefix}.w_v"], dtype=work_np_dtype
        )

        # RoPE
        q = graph_ops.add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_table,
            sin_table,
            pos_indices,
            head_dim,
            sequence_length=max_seq,
        )
        k = graph_ops.add_apply_rope_native(
            network,
            k,
            num_heads,
            head_dim,
            cos_table,
            sin_table,
            pos_indices,
            head_dim,
            sequence_length=max_seq,
        )

        context_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=max_seq,
            kv_seq=max_seq,
            mask=pad_mask_4d.get_output(0),
        )

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat, hidden, hidden, weights[f"{prefix}.w_o"], dtype=work_np_dtype
        )

        # Residual
        res1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        hidden_state = res1.get_output(0)

        # Pre-norm GeGLU MLP
        mlp_input = _add_layernorm_no_bias(
            network, hidden_state, hidden, weights[f"{prefix}.mlp_norm"], eps, dtype=work_np_dtype
        )

        # GeGLU: act(input) * gate
        inp_proj = graph_ops.add_matmul_rhs_constant(
            network,
            mlp_input,
            hidden,
            intermediate,
            weights[f"{prefix}.w_mlp_input"],
            dtype=work_np_dtype,
        )
        gate_proj = graph_ops.add_matmul_rhs_constant(
            network,
            mlp_input,
            hidden,
            intermediate,
            weights[f"{prefix}.w_mlp_gate"],
            dtype=work_np_dtype,
        )
        inp_act = graph_ops.add_gelu_erf(network, inp_proj, dtype=work_np_dtype)
        gated = network.add_elementwise(inp_act, gate_proj, trt.ElementWiseOperation.PROD)

        down = graph_ops.add_matmul_rhs_constant(
            network,
            gated.get_output(0),
            intermediate,
            hidden,
            weights[f"{prefix}.w_down"],
            dtype=work_np_dtype,
        )

        res2 = network.add_elementwise(hidden_state, down, trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

    # Final norm
    hidden_state = _add_layernorm_no_bias(
        network, hidden_state, hidden, weights["final_norm"], eps, dtype=work_np_dtype
    )

    public_output = hidden_state
    if public_output.dtype != trt.float32:
        public_output = network.add_cast(public_output, trt.float32).get_output(0)
    public_output.name = "hidden_states"
    network.mark_output(public_output)

    if verbose:
        print(
            f"[trtmc build] Building ModernBERT encoder TRT engine "
            f"({num_layers} layers, hidden={hidden}, seq_len={max_seq}, "
            f"precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_modernbert_tp(
    config: ModelConfig,
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("ModernBERT tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            "ModernBERT tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp})"
        )
    if config.intermediate_size % tp != 0:
        raise ValueError(
            "ModernBERT tensor parallel requires intermediate_size divisible by "
            f"tp_size ({config.intermediate_size} vs {tp})"
        )

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        for key in (f"{prefix}.w_q", f"{prefix}.w_k", f"{prefix}.w_v"):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_o"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_o input dim must be divisible by tp_size")
        for key in (f"{prefix}.w_mlp_input", f"{prefix}.w_mlp_gate"):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_down"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_down input dim must be divisible by tp_size")


def shard_modernbert_weights(
    config: ModelConfig,
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local ModernBERT weights for the TP builder."""
    _validate_modernbert_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_k", ".w_v", ".w_mlp_input", ".w_mlp_gate")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = config.attention_size // parallel.tp_size
    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def build_tp_modernbert_engine(
    config: ModelConfig,
    weights: "WeightDict",
    max_seq_length: int,
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local TensorRT engine plan for ModernBERT."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("build_tp_modernbert_engine requires tensor_parallel mode and tp_size > 1")
    weights = shard_modernbert_weights(config, weights, parallel=parallel)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    full_num_heads = config.num_attention_heads
    num_heads = config.num_attention_heads // parallel.tp_size
    head_dim = hidden // full_num_heads
    attention_size = num_heads * head_dim
    intermediate = config.intermediate_size // parallel.tp_size
    eps = config.raw.get("norm_eps", config.rms_norm_eps)
    max_seq = max_seq_length

    attention_contract = resolve_attention_contract(config)
    layer_types = attention_contract.layer_types
    full_theta = attention_contract.full_rope_theta
    sliding_theta = attention_contract.sliding_rope_theta

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    input_ids = network.add_input("input_ids", trt.int32, (max_seq,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq,))

    mask_float = network.add_cast(attention_mask_input, trt.float32)
    ones_c = graph_ops.add_constant(network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = graph_ops.add_constant(network, (1,), np.array([-1e10], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )
    pad_mask_4d = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_4d.reshape_dims = (1, 1, 1, max_seq)

    rope_tables = {}
    for theta in set([full_theta, sliding_theta]):
        cos = graph_ops.add_constant(
            network,
            (max_seq, head_dim // 2),
            graph_ops.make_rope_table_half_dim(max_seq, head_dim, theta, cosine=True),
        )
        sin = graph_ops.add_constant(
            network,
            (max_seq, head_dim // 2),
            graph_ops.make_rope_table_half_dim(max_seq, head_dim, theta, cosine=False),
        )
        rope_tables[theta] = (cos, sin)

    pos_indices = graph_ops.add_constant(
        network, (max_seq,), np.arange(max_seq, dtype=np.int32), dtype=np.int32
    )

    embed_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"])
    word_embed = network.add_gather(embed_table, input_ids, 0)
    hidden_state = _add_layernorm_no_bias(
        network, word_embed.get_output(0), hidden, weights["embed_norm"], eps
    )

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        lt = layer_types[layer_idx]
        theta = full_theta if lt in ("full_attention", "global_attention") else sliding_theta
        cos_table, sin_table = rope_tables[theta]

        if f"{prefix}.attn_norm" in weights:
            attn_input = _add_layernorm_no_bias(
                network, hidden_state, hidden, weights[f"{prefix}.attn_norm"], eps
            )
        else:
            attn_input = hidden_state

        q = graph_ops.add_matmul_rhs_constant(
            network, attn_input, hidden, attention_size, weights[f"{prefix}.w_q"]
        )
        k = graph_ops.add_matmul_rhs_constant(
            network, attn_input, hidden, attention_size, weights[f"{prefix}.w_k"]
        )
        v = graph_ops.add_matmul_rhs_constant(
            network, attn_input, hidden, attention_size, weights[f"{prefix}.w_v"]
        )

        q = graph_ops.add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_table,
            sin_table,
            pos_indices,
            head_dim,
            sequence_length=max_seq,
        )
        k = graph_ops.add_apply_rope_native(
            network,
            k,
            num_heads,
            head_dim,
            cos_table,
            sin_table,
            pos_indices,
            head_dim,
            sequence_length=max_seq,
        )

        context_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=max_seq,
            kv_seq=max_seq,
            mask=pad_mask_4d.get_output(0),
        )

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat, attention_size, hidden, weights[f"{prefix}.w_o"]
        )
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)

        res1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        hidden_state = res1.get_output(0)

        mlp_input = _add_layernorm_no_bias(
            network, hidden_state, hidden, weights[f"{prefix}.mlp_norm"], eps
        )

        inp_proj = graph_ops.add_matmul_rhs_constant(
            network, mlp_input, hidden, intermediate, weights[f"{prefix}.w_mlp_input"]
        )
        gate_proj = graph_ops.add_matmul_rhs_constant(
            network, mlp_input, hidden, intermediate, weights[f"{prefix}.w_mlp_gate"]
        )
        inp_act = graph_ops.add_gelu_erf(network, inp_proj)
        gated = network.add_elementwise(inp_act, gate_proj, trt.ElementWiseOperation.PROD)

        down = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), intermediate, hidden, weights[f"{prefix}.w_down"]
        )
        down = add_all_reduce_sum(network, down, parallel.tp_size)

        res2 = network.add_elementwise(hidden_state, down, trt.ElementWiseOperation.SUM)
        hidden_state = res2.get_output(0)

    hidden_state = _add_layernorm_no_bias(network, hidden_state, hidden, weights["final_norm"], eps)

    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(
            f"[trtmc build] Building ModernBERT encoder TRT engine "
            f"({num_layers} layers, hidden={hidden}, tp={parallel.tp_size}, "
            f"seq_len={max_seq}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


requires_tokenizer = True


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _ensure_tokenizer_json(model_dir: Path) -> None:
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-tokenizer-") as temporary:
            generated = Path(temporary) / "tokenizer.json"
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is None:
                backend = getattr(tokenizer, "_tokenizer", None)
            if backend is not None and hasattr(backend, "save"):
                backend.save(str(generated))
            if not generated.is_file():
                tokenizer.save_pretrained(temporary)
            if not generated.is_file():
                raise RuntimeError("tokenizer conversion did not create tokenizer.json")
            with tempfile.NamedTemporaryFile(
                dir=model_dir, prefix=".trtmc-tokenizer-", suffix=".json", delete=False
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated.read_bytes())
            temporary_path.replace(tokenizer_path)
    except Exception as exc:
        print(
            "[trtmc build] Warning: could not generate tokenizer.json "
            f"(C++ runtime may fail to create tokenizer): {exc}",
            file=sys.stderr,
        )


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(
    config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete modernbert bundle inside its owning family module."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("modernbert does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("modernbert does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context

        family_graph_ops = sys.modules[__name__]

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="modernbert tensor-parallel builds"
        )
        if quant_ctx is not None:
            raise ValueError("modernbert tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    tokenizer_source = str(options.get("tokenizer_source_model_id_or_path") or model_path)
    tokenizer_frame = _detect_tokenizer_frame(
        tokenizer_source,
        revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )
    _ensure_tokenizer_json(model_path)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else dict(config.raw)
    )
    _apply_generation_config_eos(model_path, runtime_config)
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_gelu_erf(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (exact, erf-based): 0.5 * x * (1 + erf(x / sqrt(2))).

    Constants are cast to ``inp.dtype`` for the same STRONGLY_TYPED reason
    documented on ``add_gelu_new``.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    inv_sqrt2 = _const(1.0 / np.sqrt(2.0))
    x_scaled = network.add_elementwise(inp, inv_sqrt2, trt.ElementWiseOperation.PROD)
    erf_out = network.add_unary(x_scaled.get_output(0), trt.UnaryOperation.ERF)
    one = _const(1.0)
    one_plus_erf = network.add_elementwise(one, erf_out.get_output(0), trt.ElementWiseOperation.SUM)
    half = _const(0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_erf.get_output(0), trt.ElementWiseOperation.PROD
    )
    return result.get_output(0)


def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm via TRT native INormalizationLayer (add_normalization_v2).

    Replaces the manual reduce/elementwise chain in add_layer_norm with a
    single fused layer that TRT can optimize end-to-end. In strongly typed
    networks, input/scale/bias must have identical tensor types; compute
    precision is set to FP32 for numerical stability when the TensorRT Python
    layer exposes that control.

    Note: INormalizationLayer computes (x - mean) / sqrt(var + eps) * gamma + beta.
    This is LayerNorm, NOT RMSNorm.  Use add_rms_norm for RMSNorm models.

    Args:
        inp:         Input tensor [*, hidden_size].
        hidden_size: Size of the normalized dimension (last axis).
        gamma:       Scale weights [hidden_size].
        beta:        Bias weights [hidden_size].
        eps:         Numerical stability epsilon (scalar, not a tensor).
        dtype:       Storage dtype for gamma/beta constants before TRT cast.
    """
    inp_shape = getattr(inp, "shape", None)
    rank = len(tuple(inp_shape)) if inp_shape is not None else 2
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype
    )
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    # axesMask bit i selects axis i as a reduction axis. The normalized
    # hidden dimension is always the last axis for [*, hidden_size] tensors.
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def validate_native_rope_dim(
    rotary_embedding_dim: int,
    *,
    field_name: str = "rotary_embedding_dim",
) -> int:
    """Validate the dimension contract required by TRT native RoPE."""
    rotary_embedding_dim = int(rotary_embedding_dim)
    if rotary_embedding_dim < 2 or rotary_embedding_dim % 2 != 0:
        raise ValueError(
            f"TRT native RoPE requires {field_name} to be an even value >= 2; "
            f"got {rotary_embedding_dim}"
        )
    return rotary_embedding_dim


def make_rope_table_half_dim(
    max_cache_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
    partial_rotary_factor: float = 1.0,
    interleaved: bool = False,
) -> np.ndarray:
    """Build a RoPE cos/sin table of shape [max_cache_length, rotary_ndims // 2].

    IRotaryEmbeddingLayer expects the cos/sin cache with only the *half*
    rotary dimension (it internally handles both halves).  This is different
    from make_rope_table which produces [max_cache_length, hidden_size] by
    repeating the per-head values across all heads.

    Args:
        max_cache_length: Number of positions (rows in the table).
        head_dim:         Full head dimension (D).
        rope_theta:       Base frequency for inverse-frequency computation.
        cosine:           True → cos table, False → sin table.
        partial_rotary_factor: Fraction of head dims that rotate (default 1.0).
        interleaved:      If True, adjacent-pair frequencies (CodeGen/GPT-J).
                          If False, half-split frequencies (LLaMA/Qwen).

    Returns:
        Float32 array [max_cache_length, rotary_ndims // 2].
    """
    rotary_ndims = int(head_dim * partial_rotary_factor)
    rotary_ndims = validate_native_rope_dim(rotary_ndims)
    half = rotary_ndims // 2
    default = 1.0 if cosine else 0.0
    if max_cache_length <= 0 or rope_theta <= 0.0:
        return np.full((max(max_cache_length, 1), max(half, 1)), default, dtype=np.float32)
    table = np.full((max_cache_length, half), default, dtype=np.float32)
    for pos in range(max_cache_length):
        for d in range(half):
            # For both interleaved and rotate-half the frequency index is d
            # (the distinction only affects which input pair is rotated; the
            # freq assignment per half-dim is the same).
            exponent = (2.0 * d) / rotary_ndims
            inv_freq = rope_theta ** (-exponent)
            angle = pos * inv_freq
            table[pos, d] = np.cos(angle) if cosine else np.sin(angle)
    return table


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [S, H * D] rows into [1, H, S, D].

    The transpose is required for S > 1 because each input row contains all
    heads for one token. ``sequence_length=None`` means runtime-dynamic S.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    if tag:
        r1.name = tag + "_s_h_d"
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    if tag:
        r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
    if tag:
        out.name = tag + "_s_h_d"
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (seq_dim, attention_size)
    return out.get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    """Apply RoPE via TRT native IRotaryEmbeddingLayer.

    Handles both single-token decoder steps and dynamic-Sq prefill/decode
    graphs without a manual rotate-half matmul chain.

    Shape contract (IRotaryEmbeddingLayer with position_ids):
      input:           [1, num_heads, Sq, head_dim]  (reshaped internally)
      cos_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      sin_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      position_id:     [Sq] int32, reshaped to [1, Sq] internally
      interleaved:     False → rotate-half (LLaMA/Qwen)
                       True  → adjacent-pair (CodeGen/GPT-J)

    Args:
        inp:                  [Sq, num_heads * head_dim].
        num_heads:            Number of attention heads.
        head_dim:             Per-head dimension.
        cos_cache_2d:         Pre-built 2-D cos table constant.
        sin_cache_2d:         Pre-built 2-D sin table constant.
        position_id:          Runtime position indices, shape [Sq] int32.
        rotary_embedding_dim: Number of head dims that participate in RoPE.
        interleaved:          Frequency layout (see above).
        sequence_length:      Static Sq, or None for runtime-dynamic Sq.

    Returns:
        [Sq, num_heads * head_dim] with RoPE applied.
    """
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim

    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)

    # Reshape position_id [Sq] -> [1, Sq] (batch=1).
    seq_dim = -1 if sequence_length is None else sequence_length
    pos_2d = network.add_shuffle(position_id)
    pos_2d.reshape_dims = (1, seq_dim)

    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_2d,
        sin_cache_2d,
        interleaved,
        rotary_embedding_dim,
    )
    rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, sequence_length)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    fp32_accumulation: bool = False,
) -> trt.ITensor:
    """Scaled dot-product attention via TRT native IAttention layer.

    Replaces the manual Q@K^T → scale → softmax → @V chain.  TRT 10 fuses
    this into a single kernel when a compatible implementation is available;
    decomposable=True ensures a correct fallback to primitives otherwise.

    NOTE: TRT IAttention computes raw BMM1 = Q @ K^T without any built-in
    1/sqrt(D) scaling.  We pre-scale Q by 1/sqrt(D) so that the fused kernel
    computes the standard scaled dot-product attention formula.

    Args:
        q_4d:    Query  [B, H, q_seq, D].
        k_4d:    Key    [B, H, kv_seq, D].
        v_4d:    Value  [B, H, kv_seq, D].
        causal:  Apply causal (autoregressive) mask.  Mutually exclusive
                 with ``mask``.
        mask:    Optional additive float mask [B, H, q_seq, kv_seq] added
                 to scaled logits before softmax.  Cannot be used with
                 causal=True.
        scale:   Optional Q pre-scale factor.  Defaults to 1/sqrt(D).
        fp32_accumulation:
                 Cast Q/K/V to FP32 before IAttention, then cast the context
                 back to the original Q dtype.  TRT may still select a
                 Half-input fused MHA tactic after optimizing the casts, while
                 keeping the IAttention accumulation/output boundary in FP32.

    Returns:
        Context tensor [B, H, q_seq, D].
    """
    output_dtype = q_4d.dtype
    if fp32_accumulation and output_dtype != trt.float32:
        q_4d = network.add_cast(q_4d, trt.float32).get_output(0)
        k_4d = network.add_cast(k_4d, trt.float32).get_output(0)
        v_4d = network.add_cast(v_4d, trt.float32).get_output(0)
        if mask is not None and mask.dtype != trt.float32:
            mask = network.add_cast(mask, trt.float32).get_output(0)

    # Pre-scale Q: TRT IAttention does not apply score scaling itself.
    # Match the scale constant's dtype to Q's dtype: in strongly-typed networks
    # a FP32 constant mixed with a FP16/BF16 Q causes add_elementwise to emit
    # a type-mismatch error and produce a tensor with corrupted dimensions,
    # which makes add_attention return None.
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    # Use FP16 weights directly for FP16; BF16 has no numpy native type so
    # create as FP32 and cast; FP32 falls through to the default.
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]]), dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0),
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    # Allow TRT to decompose into primitive ops when no fused kernel is
    # available (e.g. unsupported head-dim or dtype).  This guarantees
    # correctness on any configuration at the cost of potential performance.
    attn.decomposable = True
    if mask is not None and not causal:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def _scalar_constant_for_trt_dtype(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    dtype: trt.DataType,
) -> trt.ITensor:
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    const = add_constant(network, shape, np.full(shape, value, dtype=np_dtype), dtype=np_dtype)
    if dtype == trt.bfloat16:
        const = network.add_cast(const, trt.bfloat16).get_output(0)
    return const


def add_tanh_softcap(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cap: float,
    *,
    scalar_shape: tuple[int, ...],
) -> trt.ITensor:
    """Apply ``tanh(tensor / cap) * cap`` using scalar broadcasting."""
    cap_t = _scalar_constant_for_trt_dtype(network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


def _repeat_kv_heads_4d(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> trt.ITensor:
    if num_kv_heads == num_heads:
        return x_4d
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0), shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1)
        )
        head_slice.set_input(2, slice_shape.get_output(0))
        repeated.extend([head_slice.get_output(0)] * repeat)

    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _add_attention_core_with_logit_softcap(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: trt.ITensor | None,
    scale: float,
    logit_softcap: float,
) -> trt.ITensor:
    output_dtype = q_4d.dtype
    k_4d = _repeat_kv_heads_4d(
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE, score_k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = network.add_elementwise(scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    scores = add_tanh_softcap(network, scores, logit_softcap, scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM
        ).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE, v_4d, trt.MatrixOperation.NONE
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, context, output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_seq: int | None,
    kv_seq: int | None,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    logit_softcap: float | None = None,
    fp32_accumulation: bool = False,
    tag: str | None = None,
) -> trt.ITensor:
    """Native IAttention for row-major [S, H * D] Q/K/V tensors.

    ``num_kv_heads`` can be smaller than ``num_heads`` for GQA/MQA. TRT
    native IAttention supports this directly, so callers should not expand K/V
    heads unless the model semantics require per-query-head K/V values.
    """
    attention_size = num_heads * head_dim
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    if logit_softcap is not None and float(logit_softcap) > 0.0:
        if causal:
            raise NotImplementedError("logit_softcap attention requires an explicit additive mask")
        ctx_4d = _add_attention_core_with_logit_softcap(
            network,
            q_4d,
            k_4d,
            v_4d,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            mask=mask,
            scale=scale,
            logit_softcap=float(logit_softcap),
        )
    else:
        ctx_4d = add_attention_core(
            network,
            q_4d,
            k_4d,
            v_4d,
            causal=causal,
            mask=mask,
            scale=scale,
            fp32_accumulation=fp32_accumulation,
        )
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )
