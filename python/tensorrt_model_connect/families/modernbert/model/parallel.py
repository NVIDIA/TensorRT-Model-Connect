"""Tensor-parallel ModernBERT encoder builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_ops
from ..config import ModelConfig
from ....parallel_config import add_all_reduce_sum, normalize_parallel_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ....parallel_config import ParallelConfig


def _add_layernorm_no_bias(network, inp, hidden_size, gamma, eps):
    """LayerNorm without bias via TRT native normalization."""
    beta = np.zeros(hidden_size, dtype=np.float32)
    return graph_ops.add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps)


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

    layer_types = config.raw.get("layer_types", [])
    rope_params = config.raw.get("rope_parameters", {})
    full_theta = 160000.0
    sliding_theta = 10000.0
    if rope_params:
        if "full_attention" in rope_params and rope_params["full_attention"]:
            full_theta = rope_params["full_attention"].get("rope_theta", 160000.0)
        if "sliding_attention" in rope_params and rope_params["sliding_attention"]:
            sliding_theta = rope_params["sliding_attention"].get("rope_theta", 10000.0)

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

        if layer_idx < len(layer_types):
            lt = layer_types[layer_idx]
            theta = full_theta if lt in ("full_attention", "global_attention") else sliding_theta
        else:
            theta = full_theta
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
