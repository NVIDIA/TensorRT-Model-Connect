"""Tensor-parallel RWKV recurrent builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_ops
from ....parallel_config import add_all_reduce_sum, normalize_parallel_config
from ..plugin import _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ..config import ModelConfig
    from ....parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_rwkv_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("RWKV tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    hidden = int(config.hidden_size)
    intermediate = int(weights.get("_intermediate_size", config.intermediate_size))
    if hidden % tp != 0:
        raise ValueError(
            f"RWKV tensor parallel requires hidden_size divisible by tp_size ({hidden} vs {tp})"
        )
    if intermediate % tp != 0:
        raise ValueError(
            "RWKV tensor parallel requires intermediate_size divisible by tp_size "
            f"({intermediate} vs {tp})"
        )

    for layer_idx in range(int(config.num_hidden_layers)):
        prefix = f"layer.{layer_idx}"
        for key in (
            f"{prefix}.time_decay",
            f"{prefix}.time_first",
            f"{prefix}.w_attn_k",
            f"{prefix}.w_attn_v",
            f"{prefix}.w_attn_r",
            f"{prefix}.w_ffn_k",
        ):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim is not divisible by tp_size={tp}")
        for key in (f"{prefix}.w_attn_o", f"{prefix}.w_ffn_v"):
            if weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} input dim is not divisible by tp_size={tp}")


def shard_rwkv_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local RWKV weights for the TP builder."""
    _validate_rwkv_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith(
            (
                ".time_decay",
                ".time_first",
                ".w_attn_k",
                ".w_attn_v",
                ".w_attn_r",
                ".w_ffn_k",
            )
        ):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".w_attn_o", ".w_ffn_v")):
            out[key] = _slice_first_dim(value, rank, tp)
        else:
            out[key] = value

    out["_intermediate_size"] = int(weights["_intermediate_size"]) // tp
    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def _add_rwkv_tp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    attn_state_in: trt.ITensor,
    ff_state_in: trt.ITensor,
    num_state_in: trt.ITensor,
    den_state_in: trt.ITensor,
    max_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    one_const: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    local_hidden: int,
    local_intermediate: int,
    tp_size: int,
) -> dict[str, trt.ITensor]:
    normed_attn = graph_ops.add_layer_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.attn_norm"],
        weights[f"{prefix}.attn_norm_beta"],
        eps_tensor,
    )
    present_attn = normed_attn

    def _time_shift_blend(normed, prev_state, mix_weights_key):
        mix = graph_ops.add_constant(network, (1, hidden_size), weights[mix_weights_key])
        one_minus_mix = network.add_elementwise(one_const, mix, trt.ElementWiseOperation.SUB)
        cur_part = network.add_elementwise(normed, mix, trt.ElementWiseOperation.PROD)
        prev_part = network.add_elementwise(
            prev_state, one_minus_mix.get_output(0), trt.ElementWiseOperation.PROD
        )
        blended = network.add_elementwise(
            cur_part.get_output(0), prev_part.get_output(0), trt.ElementWiseOperation.SUM
        )
        return blended.get_output(0)

    xk = _time_shift_blend(normed_attn, attn_state_in, f"{prefix}.time_mix_key")
    xv = _time_shift_blend(normed_attn, attn_state_in, f"{prefix}.time_mix_value")
    xr = _time_shift_blend(normed_attn, attn_state_in, f"{prefix}.time_mix_receptance")

    r_proj = graph_ops.add_matmul_rhs_constant(
        network, xr, hidden_size, local_hidden, weights[f"{prefix}.w_attn_r"]
    )
    r_gate = network.add_activation(r_proj, trt.ActivationType.SIGMOID)
    k_proj = graph_ops.add_matmul_rhs_constant(
        network, xk, hidden_size, local_hidden, weights[f"{prefix}.w_attn_k"]
    )
    v_proj = graph_ops.add_matmul_rhs_constant(
        network, xv, hidden_size, local_hidden, weights[f"{prefix}.w_attn_v"]
    )

    time_decay = graph_ops.add_constant(network, (1, local_hidden), weights[f"{prefix}.time_decay"])
    time_first = graph_ops.add_constant(network, (1, local_hidden), weights[f"{prefix}.time_first"])

    decay_plus_max = network.add_elementwise(max_state_in, time_decay, trt.ElementWiseOperation.SUM)
    tf_plus_k = network.add_elementwise(time_first, k_proj, trt.ElementWiseOperation.SUM)

    q_out = network.add_elementwise(
        tf_plus_k.get_output(0), max_state_in, trt.ElementWiseOperation.MAX
    )
    tf_k_minus_q = network.add_elementwise(
        tf_plus_k.get_output(0), q_out.get_output(0), trt.ElementWiseOperation.SUB
    )
    exp_tf_k = network.add_unary(tf_k_minus_q.get_output(0), trt.UnaryOperation.EXP)
    ms_minus_q = network.add_elementwise(
        max_state_in, q_out.get_output(0), trt.ElementWiseOperation.SUB
    )
    exp_dpm = network.add_unary(ms_minus_q.get_output(0), trt.UnaryOperation.EXP)

    term1_num = network.add_elementwise(
        exp_tf_k.get_output(0), v_proj, trt.ElementWiseOperation.PROD
    )
    term2_num = network.add_elementwise(
        exp_dpm.get_output(0), num_state_in, trt.ElementWiseOperation.PROD
    )
    wkv_num = network.add_elementwise(
        term1_num.get_output(0), term2_num.get_output(0), trt.ElementWiseOperation.SUM
    )

    term2_den = network.add_elementwise(
        exp_dpm.get_output(0), den_state_in, trt.ElementWiseOperation.PROD
    )
    wkv_den = network.add_elementwise(
        exp_tf_k.get_output(0), term2_den.get_output(0), trt.ElementWiseOperation.SUM
    )
    wkv = network.add_elementwise(
        wkv_num.get_output(0), wkv_den.get_output(0), trt.ElementWiseOperation.DIV
    )

    q2 = network.add_elementwise(k_proj, decay_plus_max.get_output(0), trt.ElementWiseOperation.MAX)
    k_minus_q2 = network.add_elementwise(k_proj, q2.get_output(0), trt.ElementWiseOperation.SUB)
    exp_k_q2 = network.add_unary(k_minus_q2.get_output(0), trt.UnaryOperation.EXP)
    dpm_minus_q2 = network.add_elementwise(
        decay_plus_max.get_output(0), q2.get_output(0), trt.ElementWiseOperation.SUB
    )
    exp_dpm_q2 = network.add_unary(dpm_minus_q2.get_output(0), trt.UnaryOperation.EXP)

    st_term1 = network.add_elementwise(
        exp_k_q2.get_output(0), v_proj, trt.ElementWiseOperation.PROD
    )
    st_term2 = network.add_elementwise(
        exp_dpm_q2.get_output(0), num_state_in, trt.ElementWiseOperation.PROD
    )
    present_num = network.add_elementwise(
        st_term1.get_output(0), st_term2.get_output(0), trt.ElementWiseOperation.SUM
    )
    st_den_term2 = network.add_elementwise(
        exp_dpm_q2.get_output(0), den_state_in, trt.ElementWiseOperation.PROD
    )
    present_den = network.add_elementwise(
        exp_k_q2.get_output(0), st_den_term2.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_max = q2.get_output(0)

    gated = network.add_elementwise(
        r_gate.get_output(0), wkv.get_output(0), trt.ElementWiseOperation.PROD
    )
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), local_hidden, hidden_size, weights[f"{prefix}.w_attn_o"]
    )
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)
    residual_attn = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual_attn.get_output(0)

    normed_ffn = graph_ops.add_layer_norm(
        network,
        hidden_after_attn,
        hidden_size,
        weights[f"{prefix}.ffn_norm"],
        weights[f"{prefix}.ffn_norm_beta"],
        eps_tensor,
    )
    present_ff = normed_ffn

    xk_ffn = _time_shift_blend(normed_ffn, ff_state_in, f"{prefix}.time_mix_ffn_key")
    xr_ffn = _time_shift_blend(normed_ffn, ff_state_in, f"{prefix}.time_mix_ffn_receptance")

    k_ffn = graph_ops.add_matmul_rhs_constant(
        network, xk_ffn, hidden_size, local_intermediate, weights[f"{prefix}.w_ffn_k"]
    )
    k_activated = graph_ops.add_activation(network, k_ffn, "squared_relu")
    r_ffn = graph_ops.add_matmul_rhs_constant(
        network, xr_ffn, hidden_size, hidden_size, weights[f"{prefix}.w_ffn_r"]
    )
    r_ffn_gate = network.add_activation(r_ffn, trt.ActivationType.SIGMOID)
    kv_ffn = graph_ops.add_matmul_rhs_constant(
        network, k_activated, local_intermediate, hidden_size, weights[f"{prefix}.w_ffn_v"]
    )
    kv_ffn = add_all_reduce_sum(network, kv_ffn, tp_size)
    gated_ffn = network.add_elementwise(
        r_ffn_gate.get_output(0), kv_ffn, trt.ElementWiseOperation.PROD
    )
    residual_ffn = network.add_elementwise(
        hidden_after_attn, gated_ffn.get_output(0), trt.ElementWiseOperation.SUM
    )

    return {
        "hidden": residual_ffn.get_output(0),
        "present_attn": present_attn,
        "present_ff": present_ff,
        "present_num": present_num.get_output(0),
        "present_den": present_den.get_output(0),
        "present_max": present_max,
    }


def build_rwkv_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    del max_cache_length, precision, quant_ctx
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("build_rwkv_tp_engine requires tensor_parallel mode with tp_size > 1")

    rank_weights = shard_rwkv_weights(config, weights, parallel=parallel)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    local_hidden = hidden // parallel.tp_size
    local_intermediate = int(rank_weights["_intermediate_size"])

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    attn_state_inputs = []
    ff_state_inputs = []
    num_state_inputs = []
    den_state_inputs = []
    max_state_inputs = []
    for layer_idx in range(num_layers):
        attn_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("attn_state", layer_idx), trt.float32, (1, hidden)
            )
        )
        ff_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("ff_state", layer_idx), trt.float32, (1, hidden)
            )
        )
        num_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("num_state", layer_idx), trt.float32, (1, local_hidden)
            )
        )
        den_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("den_state", layer_idx), trt.float32, (1, local_hidden)
            )
        )
        max_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("max_state", layer_idx), trt.float32, (1, local_hidden)
            )
        )

    embedding_table = graph_ops.add_constant(network, (vocab, hidden), rank_weights["embedding"])
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32)
    )
    one_const = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if "pre_ln_weight" in rank_weights:
        hidden_state = graph_ops.add_layer_norm(
            network,
            hidden_state,
            hidden,
            rank_weights["pre_ln_weight"],
            rank_weights["pre_ln_bias"],
            eps_tensor,
        )
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_attn_outputs = []
    present_ff_outputs = []
    present_num_outputs = []
    present_den_outputs = []
    present_max_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_rwkv_tp_layer(
            network=network,
            hidden=hidden_state,
            attn_state_in=attn_state_inputs[layer_idx],
            ff_state_in=ff_state_inputs[layer_idx],
            num_state_in=num_state_inputs[layer_idx],
            den_state_in=den_state_inputs[layer_idx],
            max_state_in=max_state_inputs[layer_idx],
            eps_tensor=eps_tensor,
            one_const=one_const,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            local_hidden=local_hidden,
            local_intermediate=local_intermediate,
            tp_size=parallel.tp_size,
        )
        hidden_state = result["hidden"]
        present_attn_outputs.append(result["present_attn"])
        present_ff_outputs.append(result["present_ff"])
        present_num_outputs.append(result["present_num"])
        present_den_outputs.append(result["present_den"])
        present_max_outputs.append(result["present_max"])
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    hidden_state = graph_ops.add_layer_norm(
        network,
        hidden_state,
        hidden,
        rank_weights["final_norm"],
        rank_weights["final_norm_beta"],
        eps_tensor,
    )
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_lm_head"]
    )
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(num_layers):
        present_attn_outputs[layer_idx].name = graph_ops.layer_tensor_name(
            "present_attn", layer_idx
        )
        present_ff_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_ff", layer_idx)
        present_num_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_num", layer_idx)
        present_den_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_den", layer_idx)
        present_max_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_max", layer_idx)
        network.mark_output(present_attn_outputs[layer_idx])
        network.mark_output(present_ff_outputs[layer_idx])
        network.mark_output(present_num_outputs[layer_idx])
        network.mark_output(present_den_outputs[layer_idx])
        network.mark_output(present_max_outputs[layer_idx])

    if verbose:
        print(
            "[trtmc build] RWKV TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"h={hidden}, local_h={local_hidden}, local_i={local_intermediate})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT RWKV TP engine build failed")
    return bytes(plan)
