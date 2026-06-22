"""Tensor-parallel Bark semantic/coarse decoder builder.

This mirrors Bark's single-device ``standard_decoder_builder`` for the
autoregressive semantic and coarse GPT blocks. Tensor parallelism only changes
the decoder projections: Q/K/V and FC1 are column-parallel, output/FC2 are
row-parallel and joined with a TensorRT distributed all-reduce.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import WeightDict
from .config import ModelConfig
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...parallel_config import ParallelConfig


def _slice_first_dim(value: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    if value.shape[0] % tp_size != 0:
        raise ValueError(f"Cannot shard first dimension {value.shape[0]} over TP{tp_size}")
    chunk = value.shape[0] // tp_size
    return np.ascontiguousarray(value[rank * chunk:(rank + 1) * chunk])


def _slice_last_dim(value: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    if value.shape[-1] % tp_size != 0:
        raise ValueError(f"Cannot shard last dimension {value.shape[-1]} over TP{tp_size}")
    chunk = value.shape[-1] // tp_size
    slc = [slice(None)] * value.ndim
    slc[-1] = slice(rank * chunk, (rank + 1) * chunk)
    return np.ascontiguousarray(value[tuple(slc)])


def _validate_bark_tp(
    *,
    sub_model: str,
    hidden: int,
    num_heads: int,
    mlp_size: int,
    parallel: "ParallelConfig",
) -> None:
    if not parallel.enabled:
        raise ValueError("Bark tensor-parallel builder requires an enabled parallel config")
    if hidden % parallel.tp_size != 0:
        raise ValueError(
            f"{sub_model} hidden_size={hidden} must be divisible by TP{parallel.tp_size}")
    if num_heads % parallel.tp_size != 0:
        raise ValueError(
            f"{sub_model} num_heads={num_heads} must be divisible by TP{parallel.tp_size}")
    if mlp_size % parallel.tp_size != 0:
        raise ValueError(
            f"{sub_model} mlp_size={mlp_size} must be divisible by TP{parallel.tp_size}")


def shard_bark_decoder_weights(
    weights: WeightDict,
    *,
    sub_model: str,
    sub_cfg: dict,
    parallel_config=None,
) -> WeightDict:
    """Return rank-local weights for a Bark semantic/coarse decoder."""
    parallel = normalize_parallel_config(parallel_config)
    hidden = int(sub_cfg["hidden_size"])
    num_heads = int(sub_cfg["num_heads"])
    num_layers = int(sub_cfg["num_layers"])
    mlp_size = int(sub_cfg.get("intermediate_size", hidden * 4))
    _validate_bark_tp(
        sub_model=sub_model,
        hidden=hidden,
        num_heads=num_heads,
        mlp_size=mlp_size,
        parallel=parallel,
    )

    sharded = WeightDict()
    for key, value in weights.items():
        if key.startswith("_"):
            continue
        if not isinstance(value, np.ndarray):
            sharded[key] = value
            continue

        if key in {"embedding", "position_embedding", "final_norm", "final_norm_beta", "w_out"}:
            sharded[key] = value
            continue

        handled = False
        for layer_idx in range(num_layers):
            lp = f"layer.{layer_idx}"
            if key in {f"{lp}.w_q", f"{lp}.w_k", f"{lp}.w_v", f"{lp}.w_fc1"}:
                sharded[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
                handled = True
                break
            if key in {f"{lp}.q_bias", f"{lp}.k_bias", f"{lp}.v_bias", f"{lp}.fc1_bias"}:
                sharded[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
                handled = True
                break
            if key in {f"{lp}.w_o", f"{lp}.w_fc2"}:
                sharded[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
                handled = True
                break
            if key in {
                f"{lp}.input_norm",
                f"{lp}.input_norm_beta",
                f"{lp}.post_attn_norm",
                f"{lp}.post_attn_norm_beta",
                f"{lp}.o_bias",
                f"{lp}.fc2_bias",
            }:
                sharded[key] = value
                handled = True
                break
        if not handled:
            sharded[key] = value

    return sharded


def _row_parallel_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    in_width: int,
    out_width: int,
    weight: np.ndarray,
    *,
    tp_size: int,
    bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    out = graph_ops.add_matmul_rhs_constant(
        network, inp, in_width, out_width, weight, dtype=dtype)
    out = add_all_reduce_sum(network, out, tp_size)
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_width, bias, dtype=dtype)
    return out


def _add_bark_tp_decoder_layer(
    network: trt.INetworkDefinition,
    hidden_state: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    *,
    layer_idx: int,
    weights: WeightDict,
    hidden: int,
    local_attention_size: int,
    local_heads: int,
    head_dim: int,
    local_mlp_size: int,
    attention_window: int,
    eps_tensor: trt.ITensor,
    tp_size: int,
    dtype: np.dtype = np.float32,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    lp = f"layer.{layer_idx}"

    normed = graph_ops.add_layer_norm(
        network,
        hidden_state,
        hidden,
        weights[f"{lp}.input_norm"],
        weights[f"{lp}.input_norm_beta"],
        eps_tensor,
        dtype=dtype,
    )

    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden, local_attention_size, weights[f"{lp}.w_q"], dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden, local_attention_size, weights[f"{lp}.w_k"], dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden, local_attention_size, weights[f"{lp}.w_v"], dtype=dtype)
    for name, width in (("q", local_attention_size), ("k", local_attention_size),
                        ("v", local_attention_size)):
        bias = weights.get(f"{lp}.{name}_bias")
        if bias is None:
            continue
        tensor = {"q": q, "k": k, "v": v}[name]
        tensor = graph_ops.add_bias_sum(network, tensor, width, bias, dtype=dtype)
        if name == "q":
            q = tensor
        elif name == "k":
            k = tensor
        else:
            v = tensor

    present_k = k
    present_v = v

    k_row = network.add_shuffle(k)
    k_row.reshape_dims = (1, local_attention_size)
    v_row = network.add_shuffle(v)
    v_row.reshape_dims = (1, local_attention_size)

    all_k = network.add_concatenation([cache_k, k_row.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_row.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    ctx = graph_ops.add_attention_from_rows(
        network,
        q,
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=local_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=mask_4d,
        scale=1.0 / np.sqrt(head_dim),
    )

    attn_out = _row_parallel_linear(
        network,
        ctx,
        local_attention_size,
        hidden,
        weights[f"{lp}.w_o"],
        tp_size=tp_size,
        bias=weights.get(f"{lp}.o_bias"),
        dtype=dtype,
    )
    hidden_state = network.add_elementwise(
        hidden_state, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

    normed2 = graph_ops.add_layer_norm(
        network,
        hidden_state,
        hidden,
        weights[f"{lp}.post_attn_norm"],
        weights[f"{lp}.post_attn_norm_beta"],
        eps_tensor,
        dtype=dtype,
    )
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed2, hidden, local_mlp_size, weights[f"{lp}.w_fc1"], dtype=dtype)
    fc1_bias = weights.get(f"{lp}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, local_mlp_size, fc1_bias, dtype=dtype)
    gelu = graph_ops.add_gelu_new(network, fc1, dtype=dtype)
    fc2 = _row_parallel_linear(
        network,
        gelu,
        local_mlp_size,
        hidden,
        weights[f"{lp}.w_fc2"],
        tp_size=tp_size,
        bias=weights.get(f"{lp}.fc2_bias"),
        dtype=dtype,
    )
    hidden_state = network.add_elementwise(
        hidden_state, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    return hidden_state, present_k, present_v


def build_bark_tp_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    sub_model: str,
    sub_cfg: dict,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local TP engine for a Bark semantic/coarse decoder."""
    del precision  # Bark decoder engines currently use fp32, matching the existing path.
    parallel = normalize_parallel_config(parallel_config)
    hidden = int(sub_cfg["hidden_size"])
    num_heads = int(sub_cfg["num_heads"])
    num_layers = int(sub_cfg["num_layers"])
    vocab = int(sub_cfg["vocab_size"])
    output_vocab = int(sub_cfg.get("output_vocab", vocab))
    max_position = int(sub_cfg.get("max_position", 1024))
    mlp_size = int(sub_cfg.get("intermediate_size", hidden * 4))
    _validate_bark_tp(
        sub_model=sub_model,
        hidden=hidden,
        num_heads=num_heads,
        mlp_size=mlp_size,
        parallel=parallel,
    )

    local_heads = num_heads // parallel.tp_size
    head_dim = hidden // num_heads
    local_attention_size = local_heads * head_dim
    local_mlp_size = mlp_size // parallel.tp_size
    attention_window = max_cache_length + 1
    rank_weights = shard_bark_decoder_weights(
        weights, sub_model=sub_model, sub_cfg=sub_cfg, parallel_config=parallel)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
    input_embed = network.add_input("input_embed", trt.float32, (1, hidden))
    use_input_embed = network.add_input("use_input_embed", trt.float32, (1,))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            trt.float32,
            (max_cache_length, local_attention_size),
        )
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            trt.float32,
            (max_cache_length, local_attention_size),
        )
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["embedding"])
    position_table = graph_ops.add_constant(
        network, (max_position, hidden), rank_weights["position_embedding"])
    token_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)

    flag = network.add_shuffle(use_input_embed)
    flag.reshape_dims = (1, 1)
    one = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    inv_flag = network.add_elementwise(
        one, flag.get_output(0), trt.ElementWiseOperation.SUB).get_output(0)
    token_part = network.add_elementwise(
        token_embed, inv_flag, trt.ElementWiseOperation.PROD).get_output(0)
    embed_part = network.add_elementwise(
        input_embed, flag.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    hidden_state = network.add_elementwise(
        token_part, embed_part, trt.ElementWiseOperation.SUM).get_output(0)
    pos_embed = network.add_gather(position_table, position_id, 0).get_output(0)
    hidden_state = network.add_elementwise(
        hidden_state, pos_embed, trt.ElementWiseOperation.SUM).get_output(0)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        hidden_state, present_k, present_v = _add_bark_tp_decoder_layer(
            network,
            hidden_state,
            cache_k_inputs[layer_idx],
            cache_v_inputs[layer_idx],
            attention_mask,
            layer_idx=layer_idx,
            weights=rank_weights,
            hidden=hidden,
            local_attention_size=local_attention_size,
            local_heads=local_heads,
            head_dim=head_dim,
            local_mlp_size=local_mlp_size,
            attention_window=attention_window,
            eps_tensor=eps_tensor,
            tp_size=parallel.tp_size,
        )
        present_k_outputs.append(present_k)
        present_v_outputs.append(present_v)

    hidden_state = graph_ops.add_layer_norm(
        network,
        hidden_state,
        hidden,
        rank_weights["final_norm"],
        rank_weights["final_norm_beta"],
        eps_tensor,
    )

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, output_vocab, rank_weights["w_out"])
    logits = graph_ops.add_bias_sum(
        network, logits, output_vocab, np.zeros(output_vocab, dtype=np.float32))
    logits.name = "logits"
    network.mark_output(logits)

    for i, (present_k, present_v) in enumerate(zip(present_k_outputs, present_v_outputs)):
        present_k.name = graph_ops.layer_tensor_name("present_k", i)
        present_v.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            f"[trtmc build] Building Bark {sub_model} TP rank {parallel.rank}/"
            f"{parallel.tp_size}: layers={num_layers}, hidden={hidden}, "
            f"local_heads={local_heads}, local_attn={local_attention_size}, "
            f"cache={max_cache_length}",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine build failed for Bark {sub_model} TP rank")
    return bytes(plan)
