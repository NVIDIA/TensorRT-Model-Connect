"""Tensor-parallel Mamba SSM builder.

The TP policy shards Mamba's ``d_inner`` dimension. Per-rank engines keep local
conv/SSM state, all-reduce input-dependent ``dt/B/C`` projections that need the
full inner reduction, and all-reduce the row-parallel output projection back to
full hidden size.
"""

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


def _validate_mamba_tp(weights: "WeightDict", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Mamba tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    d_inner = int(weights["_d_inner"])
    if d_inner % tp != 0:
        raise ValueError(
            f"Mamba tensor parallel requires d_inner divisible by tp_size ({d_inner} vs {tp})"
        )

    for layer_idx in range(int(weights.get("_num_layers", 0))):
        prefix = f"layer.{layer_idx}"
        for key in (
            f"{prefix}.w_in_x",
            f"{prefix}.w_in_z",
            f"{prefix}.w_dt_out",
            f"{prefix}.dt_proj_bias",
        ):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim is not divisible by tp_size={tp}")
        for key in (
            f"{prefix}.conv1d_weight",
            f"{prefix}.conv1d_bias",
            f"{prefix}.w_dt_in",
            f"{prefix}.w_B",
            f"{prefix}.w_C",
            f"{prefix}.A",
            f"{prefix}.D",
            f"{prefix}.w_out",
        ):
            if weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} input dim is not divisible by tp_size={tp}")


def shard_mamba_weights(
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local Mamba weights for the TP builder."""
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((".w_in_x", ".w_in_z", ".w_dt_out", ".dt_proj_bias")):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(
            (
                ".conv1d_weight",
                ".conv1d_bias",
                ".w_dt_in",
                ".w_B",
                ".w_C",
                ".A",
                ".D",
                ".w_out",
            )
        ):
            out[key] = _slice_first_dim(value, rank, tp)
        else:
            out[key] = value

    out["_d_inner"] = int(weights["_d_inner"]) // tp
    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def _add_mamba_tp_layer(
    *,
    network,
    hidden,
    conv_state_in,
    ssm_state_in,
    eps_tensor,
    weights,
    prefix: str,
    hidden_size: int,
    local_d_inner: int,
    state_size: int,
    conv_kernel: int,
    dt_rank: int,
    tp_size: int,
):
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.norm"], eps_tensor
    )

    x = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, local_d_inner, weights[f"{prefix}.w_in_x"]
    )
    z = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, local_d_inner, weights[f"{prefix}.w_in_z"]
    )

    x_col = network.add_shuffle(x)
    x_col.reshape_dims = (local_d_inner, 1)
    if conv_kernel > 1:
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(local_d_inner, conv_kernel - 1),
            stride=(1, 1),
        )
        new_conv_state = network.add_concatenation([slice_layer.get_output(0), x_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = x_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (local_d_inner, conv_kernel), weights[f"{prefix}.conv1d_weight"]
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, local_d_inner)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), local_d_inner, weights[f"{prefix}.conv1d_bias"]
    )
    conv_activated = graph_ops.add_activation(network, conv_out, "silu")

    dt_in = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, local_d_inner, dt_rank, weights[f"{prefix}.w_dt_in"]
    )
    dt_in = add_all_reduce_sum(network, dt_in, tp_size)
    B = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, local_d_inner, state_size, weights[f"{prefix}.w_B"]
    )
    B = add_all_reduce_sum(network, B, tp_size)
    C = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, local_d_inner, state_size, weights[f"{prefix}.w_C"]
    )
    C = add_all_reduce_sum(network, C, tp_size)

    dt = graph_ops.add_matmul_rhs_constant(
        network, dt_in, dt_rank, local_d_inner, weights[f"{prefix}.w_dt_out"]
    )
    dt = graph_ops.add_bias_sum(network, dt, local_d_inner, weights[f"{prefix}.dt_proj_bias"])

    dt_exp = network.add_unary(dt, trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    dt_exp_p1 = network.add_elementwise(dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt_final = dt_softplus.get_output(0)

    dt_col = network.add_shuffle(dt_final)
    dt_col.reshape_dims = (local_d_inner, 1)
    A_const = graph_ops.add_constant(network, (local_d_inner, state_size), weights[f"{prefix}.A"])
    dtA = network.add_elementwise(dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    A_bar = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    B_reshape = network.add_shuffle(B)
    B_reshape.reshape_dims = (1, state_size)
    dt_B = network.add_elementwise(
        dt_col.get_output(0), B_reshape.get_output(0), trt.ElementWiseOperation.PROD
    )

    x_col2 = network.add_shuffle(conv_activated)
    x_col2.reshape_dims = (local_d_inner, 1)
    dtBx = network.add_elementwise(
        dt_B.get_output(0), x_col2.get_output(0), trt.ElementWiseOperation.PROD
    )

    decay = network.add_elementwise(
        A_bar.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD
    )
    new_ssm = network.add_elementwise(
        decay.get_output(0), dtBx.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_ssm.get_output(0)

    C_reshape = network.add_shuffle(C)
    C_reshape.reshape_dims = (state_size, 1)
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE, C_reshape.get_output(0), trt.MatrixOperation.NONE
    )
    y_flat = network.add_shuffle(y_matmul.get_output(0))
    y_flat.reshape_dims = (1, local_d_inner)

    D_const = graph_ops.add_constant(network, (1, local_d_inner), weights[f"{prefix}.D"])
    Dx = network.add_elementwise(D_const, conv_activated, trt.ElementWiseOperation.PROD)
    y = network.add_elementwise(
        y_flat.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM
    )

    z_activated = graph_ops.add_activation(network, z, "silu")
    gated = network.add_elementwise(y.get_output(0), z_activated, trt.ElementWiseOperation.PROD)

    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), local_d_inner, hidden_size, weights[f"{prefix}.w_out"]
    )
    out = add_all_reduce_sum(network, out, tp_size)
    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def build_mamba_tp_engine(
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
        raise ValueError("build_mamba_tp_engine requires tensor_parallel mode with tp_size > 1")

    weights = type(weights)(weights)
    weights["_num_layers"] = config.num_hidden_layers
    _validate_mamba_tp(weights, parallel)
    rank_weights = shard_mamba_weights(weights, parallel=parallel)

    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    local_d_inner = int(rank_weights["_d_inner"])
    state_size = int(weights["_state_size"])
    conv_kernel = int(weights["_conv_kernel"])
    dt_rank = int(weights["_dt_rank"])

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    conv_state_inputs = []
    ssm_state_inputs = []
    for layer_idx in range(num_layers):
        conv_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("conv_state", layer_idx),
                trt.float32,
                (local_d_inner, conv_kernel),
            )
        )
        ssm_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("ssm_state", layer_idx),
                trt.float32,
                (local_d_inner, state_size),
            )
        )

    embedding_table = graph_ops.add_constant(network, (vocab, hidden), rank_weights["embedding"])
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32)
    )

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_conv_outputs = []
    present_ssm_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_mamba_tp_layer(
            network=network,
            hidden=hidden_state,
            conv_state_in=conv_state_inputs[layer_idx],
            ssm_state_in=ssm_state_inputs[layer_idx],
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            local_d_inner=local_d_inner,
            state_size=state_size,
            conv_kernel=conv_kernel,
            dt_rank=dt_rank,
            tp_size=parallel.tp_size,
        )
        hidden_state = result["hidden"]
        present_conv_outputs.append(result["present_conv"])
        present_ssm_outputs.append(result["present_ssm"])
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(network, hidden_state, hidden, final_norm, eps_tensor)

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_lm_head"]
    )
    logits = graph_ops.add_bias_sum(network, logits, vocab, np.zeros(vocab, dtype=np.float32))
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(num_layers):
        present_conv_outputs[layer_idx].name = graph_ops.layer_tensor_name(
            "present_conv", layer_idx
        )
        present_ssm_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_ssm", layer_idx)
        network.mark_output(present_conv_outputs[layer_idx])
        network.mark_output(present_ssm_outputs[layer_idx])

    if verbose:
        print(
            "[trtmc build] Mamba TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"h={hidden}, local_d_inner={local_d_inner}, state={state_size})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Mamba TP engine build failed")
    return bytes(plan)
