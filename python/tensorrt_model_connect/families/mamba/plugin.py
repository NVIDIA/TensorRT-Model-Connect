"""Mamba family plugin -- Selective State Space Model (SSM).

Mamba replaces attention entirely with a recurrent SSM block. Key differences:
  - NO attention mask, NO position_id, NO KV cache
  - Instead: conv_state + ssm_state per layer (constant memory per step)
  - Input projection splits into x (SSM path) and z (gate)
  - Causal conv1d with cached state for single-step inference
  - Selective scan: input-dependent discretization of continuous SSM

Architecture per layer (single step):
  1. RMSNorm(hidden) -> in_proj -> split into x, z
  2. Conv1d step: shift conv_state left, append x, depthwise convolve
  3. SiLU activation on convolved x
  4. x_proj -> split into dt, B, C (input-dependent SSM params)
  5. dt_proj: dt_rank -> intermediate (with bias + softplus)
  6. Selective scan: ssm_state = exp(dt*A) * ssm_state + dt*B*x; y = C*ssm_state + D*x
  7. Output: y * silu(z) -> out_proj -> residual add

Weight key mapping:
  HF: backbone.layers.{i}.mixer.in_proj.weight    -> [2*d_inner, d_model]
  HF: backbone.layers.{i}.mixer.conv1d.weight      -> [d_inner, 1, conv_kernel]
  HF: backbone.layers.{i}.mixer.conv1d.bias         -> [d_inner]
  HF: backbone.layers.{i}.mixer.x_proj.weight       -> [dt_rank+2*state_size, d_inner]
  HF: backbone.layers.{i}.mixer.dt_proj.weight      -> [d_inner, dt_rank]
  HF: backbone.layers.{i}.mixer.dt_proj.bias        -> [d_inner]
  HF: backbone.layers.{i}.mixer.A_log               -> [d_inner, state_size]
  HF: backbone.layers.{i}.mixer.D                   -> [d_inner]
  HF: backbone.layers.{i}.mixer.out_proj.weight     -> [d_model, d_inner]
  HF: backbone.layers.{i}.norm.weight               -> [d_model]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel


trt = trt_compat.get_trt()

class MambaPlugin:
    name = "mamba"
    runtime_strategy = "mamba_ssm_recurrent"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "mamba"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load Mamba weights from safetensors."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        d_inner = config.raw.get("intermediate_size", config.raw.get("d_inner", hidden * 2))
        state_size = config.raw.get("state_size", 16)
        conv_kernel = config.raw.get("conv_kernel", 4)
        dt_rank = config.raw.get("time_step_rank", 48)

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "backbone.embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"backbone.layers.{layer_idx}"

            # RMSNorm
            norm = _load_tensor(readers, f"{hf_prefix}.norm.weight")
            weights[f"{prefix}.norm"] = norm.astype(np.float32)

            # in_proj: [2*d_inner, hidden] -> split into x_proj [d_inner, hidden] and z_proj [d_inner, hidden]
            in_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.in_proj.weight")
            assert in_proj_raw.shape == (2 * d_inner, hidden), (
                f"in_proj shape {in_proj_raw.shape} != ({2 * d_inner}, {hidden})")
            # Transpose to [hidden, d_inner] for matmul
            weights[f"{prefix}.w_in_x"] = _transpose_2d(
                in_proj_raw[:d_inner, :], "in_proj_x")  # [hidden, d_inner]
            weights[f"{prefix}.w_in_z"] = _transpose_2d(
                in_proj_raw[d_inner:, :], "in_proj_z")  # [hidden, d_inner]
            del in_proj_raw

            # conv1d: [d_inner, 1, conv_kernel] -> flatten to [d_inner, conv_kernel]
            conv1d_weight = _load_tensor(
                readers, f"{hf_prefix}.mixer.conv1d.weight")
            # Shape: [d_inner, 1, conv_kernel] -> squeeze to [d_inner, conv_kernel]
            conv1d_weight = conv1d_weight.reshape(d_inner, conv_kernel).astype(np.float32)
            weights[f"{prefix}.conv1d_weight"] = conv1d_weight

            conv1d_bias = _load_tensor(readers, f"{hf_prefix}.mixer.conv1d.bias")
            weights[f"{prefix}.conv1d_bias"] = conv1d_bias.astype(np.float32)

            # x_proj: [dt_rank + 2*state_size, d_inner] -- projects x to dt, B, C
            x_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.x_proj.weight")
            expected_xproj_rows = dt_rank + 2 * state_size
            assert x_proj_raw.shape == (expected_xproj_rows, d_inner), (
                f"x_proj shape {x_proj_raw.shape} != ({expected_xproj_rows}, {d_inner})")
            # Split into dt_proj_in [dt_rank, d_inner], B_proj [state_size, d_inner], C_proj [state_size, d_inner]
            # Transpose each to [d_inner, out_dim] for matmul
            weights[f"{prefix}.w_dt_in"] = _transpose_2d(
                x_proj_raw[:dt_rank, :], "dt_proj_in")  # [d_inner, dt_rank]
            weights[f"{prefix}.w_B"] = _transpose_2d(
                x_proj_raw[dt_rank:dt_rank+state_size, :], "B_proj")  # [d_inner, state_size]
            weights[f"{prefix}.w_C"] = _transpose_2d(
                x_proj_raw[dt_rank+state_size:, :], "C_proj")  # [d_inner, state_size]
            del x_proj_raw

            # dt_proj: [d_inner, dt_rank] -- projects dt_rank -> d_inner
            dt_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.dt_proj.weight")
            assert dt_proj_raw.shape == (d_inner, dt_rank), (
                f"dt_proj shape {dt_proj_raw.shape} != ({d_inner}, {dt_rank})")
            weights[f"{prefix}.w_dt_out"] = _transpose_2d(
                dt_proj_raw, "dt_proj")  # [dt_rank, d_inner]
            del dt_proj_raw

            dt_proj_bias = _load_tensor(readers, f"{hf_prefix}.mixer.dt_proj.bias")
            weights[f"{prefix}.dt_proj_bias"] = dt_proj_bias.astype(np.float32)

            # A_log: [d_inner, state_size] -- compute A = -exp(A_log)
            A_log = _load_tensor(readers, f"{hf_prefix}.mixer.A_log")
            assert A_log.shape == (d_inner, state_size)
            A = -np.exp(A_log.astype(np.float32))
            weights[f"{prefix}.A"] = A  # [d_inner, state_size]

            # D: [d_inner] -- skip connection scalar
            D = _load_tensor(readers, f"{hf_prefix}.mixer.D")
            weights[f"{prefix}.D"] = D.astype(np.float32)

            # out_proj: [hidden, d_inner] -> transpose to [d_inner, hidden]
            out_proj_raw = _load_tensor(readers, f"{hf_prefix}.mixer.out_proj.weight")
            weights[f"{prefix}.w_out"] = _transpose_2d(
                out_proj_raw, "out_proj")  # [d_inner, hidden]
            del out_proj_raw

        # Final norm
        final_norm_key = "backbone.norm_f.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head (tied to embeddings for mamba-130m-hf)
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_lm_head"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            # Tied embeddings: [vocab, hidden] -> [hidden, vocab]
            weights["w_lm_head"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        # Store Mamba-specific dimensions for engine builder
        weights["_d_inner"] = d_inner  # type: ignore[assignment]
        weights["_state_size"] = state_size  # type: ignore[assignment]
        weights["_conv_kernel"] = conv_kernel  # type: ignore[assignment]
        weights["_dt_rank"] = dt_rank  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build TRT engine for Mamba SSM.

        max_cache_length is accepted for API compatibility but is not used
        by Mamba (SSM state is constant size regardless of sequence length).

        Engine inputs:
          token_id: int32 [1]
          conv_state_0..N: float32 [1, d_inner * conv_kernel]
          ssm_state_0..N: float32 [1, d_inner * state_size]

        Engine outputs:
          logits: float32 [1, vocab]
          present_conv_0..N: float32 [1, d_inner * conv_kernel]
          present_ssm_0..N: float32 [1, d_inner * state_size]
        """
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Mamba tensor-parallel builds")
            from .tp_builder import build_mamba_tp_engine
            return build_mamba_tp_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        d_inner: int = weights["_d_inner"]
        state_size: int = weights["_state_size"]
        conv_kernel: int = weights["_conv_kernel"]
        dt_rank: int = weights["_dt_rank"]

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # -----------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------
        token_id = network.add_input("token_id", trt.int32, (1,))

        conv_state_inputs = []
        ssm_state_inputs = []
        for i in range(num_layers):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", i),
                trt.float32, (d_inner, conv_kernel))
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", i),
                trt.float32, (d_inner, state_size))
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        # -----------------------------------------------------------
        # Shared constants
        # -----------------------------------------------------------
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])

        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32))

        # -----------------------------------------------------------
        # Embedding lookup
        # -----------------------------------------------------------
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)  # [1, hidden]

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # -----------------------------------------------------------
        # Mamba layers
        # -----------------------------------------------------------
        present_conv_outputs = []
        present_ssm_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            result = _add_mamba_layer(
                network=network,
                hidden=hidden_state,
                conv_state_in=conv_state_inputs[layer_idx],
                ssm_state_in=ssm_state_inputs[layer_idx],
                eps_tensor=eps_tensor,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                d_inner=d_inner,
                state_size=state_size,
                conv_kernel=conv_kernel,
                dt_rank=dt_rank,
            )

            hidden_state = result["hidden"]
            present_conv_outputs.append(result["present_conv"])
            present_ssm_outputs.append(result["present_ssm"])

            if debug_layer_outputs:
                _mark_debug_output(
                    network, hidden_state, f"debug_hidden_{layer_idx}")

        # -----------------------------------------------------------
        # Final norm
        # -----------------------------------------------------------
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_ops.add_rms_norm(
                network, hidden_state, hidden, final_norm, eps_tensor)

        # -----------------------------------------------------------
        # LM head (logits)
        # -----------------------------------------------------------
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_lm_head"])
        # Zero bias
        b_out = np.zeros(vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)

        logits.name = "logits"
        network.mark_output(logits)

        # -----------------------------------------------------------
        # Present conv/ssm state outputs
        # -----------------------------------------------------------
        for i in range(num_layers):
            pc = present_conv_outputs[i]
            ps = present_ssm_outputs[i]
            pc.name = graph_ops.layer_tensor_name("present_conv", i)
            ps.name = graph_ops.layer_tensor_name("present_ssm", i)
            network.mark_output(pc)
            network.mark_output(ps)

        # -----------------------------------------------------------
        # Build engine
        # -----------------------------------------------------------
        if verbose:
            print(f"[trtmc build] Building Mamba TRT engine ({num_layers} layers, "
                  f"hidden={hidden}, d_inner={d_inner}, state_size={state_size}, "
                  f"conv_kernel={conv_kernel}, dt_rank={dt_rank}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Mark a tensor as a network output for debug inspection."""
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _add_mamba_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    d_inner: int,
    state_size: int,
    conv_kernel: int,
    dt_rank: int,
) -> dict[str, trt.ITensor]:
    """Add one Mamba SSM layer.

    Returns: {hidden, present_conv, present_ssm}
    """
    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.norm"], eps_tensor)

    # ===== 2. Input projection: normed -> x, z =====
    # x: [1, d_inner]  (SSM path)
    x = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner,
        weights[f"{prefix}.w_in_x"])
    # z: [1, d_inner]  (gate path)
    z = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner,
        weights[f"{prefix}.w_in_z"])

    # ===== 3. Conv1d step =====
    # conv_state_in: [d_inner, conv_kernel]
    # Shift left: drop column 0, append x as new column
    # New conv_state = [conv_state_in[:, 1:], x^T]

    # x reshaped to [d_inner, 1] for concatenation
    x_col = network.add_shuffle(x)
    x_col.reshape_dims = (d_inner, 1)

    if conv_kernel > 1:
        # Slice conv_state_in[:, 1:] using a slice layer
        # conv_state_in shape: [d_inner, conv_kernel]
        # We want columns [1:conv_kernel] -> shape [d_inner, conv_kernel-1]
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(d_inner, conv_kernel - 1),
            stride=(1, 1))

        # Concatenate: [d_inner, conv_kernel-1] + [d_inner, 1] = [d_inner, conv_kernel]
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), x_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)  # [d_inner, conv_kernel]
    else:
        # conv_kernel == 1: conv_state is just x
        present_conv = x_col.get_output(0)

    # Depthwise convolution: element-wise multiply conv_state by weights, then sum over kernel dim
    # conv1d_weight: [d_inner, conv_kernel]
    # present_conv: [d_inner, conv_kernel]
    # Result: sum(present_conv * conv1d_weight, dim=1) + bias -> [d_inner]

    conv_w = graph_ops.add_constant(
        network, (d_inner, conv_kernel),
        weights[f"{prefix}.conv1d_weight"])
    conv_prod = network.add_elementwise(
        present_conv, conv_w, trt.ElementWiseOperation.PROD)
    # Reduce sum over axis 1 (kernel dim): [d_inner, conv_kernel] -> [d_inner, 1]
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM,
        1 << 1, keep_dims=True)
    # Reshape to [1, d_inner]
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, d_inner)

    # Add conv bias
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), d_inner,
        weights[f"{prefix}.conv1d_bias"])

    # SiLU activation on conv output
    conv_activated = graph_ops.add_activation(network, conv_out, "silu")
    # conv_activated: [1, d_inner]

    # ===== 4. SSM parameters from x =====
    # x_proj splits into dt, B, C
    # dt_in: [1, dt_rank]
    dt_in = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, d_inner, dt_rank,
        weights[f"{prefix}.w_dt_in"])
    # B: [1, state_size]
    B = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, d_inner, state_size,
        weights[f"{prefix}.w_B"])
    # C: [1, state_size]
    C = graph_ops.add_matmul_rhs_constant(
        network, conv_activated, d_inner, state_size,
        weights[f"{prefix}.w_C"])

    # ===== 5. dt_proj: dt_rank -> d_inner with bias + softplus =====
    dt = graph_ops.add_matmul_rhs_constant(
        network, dt_in, dt_rank, d_inner,
        weights[f"{prefix}.w_dt_out"])  # [1, d_inner]
    dt = graph_ops.add_bias_sum(
        network, dt, d_inner,
        weights[f"{prefix}.dt_proj_bias"])

    # Softplus: log(1 + exp(x))
    # For numerical stability: softplus(x) = x when x > 20, else log(1+exp(x))
    # In TRT: exp -> add 1 -> log
    dt_exp = network.add_unary(dt, trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(network, (1, 1),
                                  np.array([1.0], dtype=np.float32))
    dt_exp_p1 = network.add_elementwise(
        dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(
        dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt_final = dt_softplus.get_output(0)  # [1, d_inner]

    # ===== 6. Selective scan =====
    # A: [d_inner, state_size] (negative, precomputed as -exp(A_log))
    # dt: [1, d_inner]
    # B: [1, state_size]
    # C: [1, state_size]
    # ssm_state_in: [d_inner, state_size]
    # conv_activated (x after conv): [1, d_inner]

    # Discretize A: A_bar = exp(dt * A)
    # dt: [1, d_inner] -> reshape to [d_inner, 1] for broadcasting with A [d_inner, state_size]
    dt_col = network.add_shuffle(dt_final)
    dt_col.reshape_dims = (d_inner, 1)

    A_const = graph_ops.add_constant(
        network, (d_inner, state_size), weights[f"{prefix}.A"])

    # dt * A: [d_inner, 1] * [d_inner, state_size] -> [d_inner, state_size] (broadcast)
    dtA = network.add_elementwise(
        dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    # exp(dt * A)
    A_bar = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)
    # A_bar: [d_inner, state_size]

    # Discretize B: dt_B = dt * B
    # dt: [d_inner, 1], B: [1, state_size] -> broadcast to [d_inner, state_size]
    B_reshape = network.add_shuffle(B)
    B_reshape.reshape_dims = (1, state_size)
    dt_B = network.add_elementwise(
        dt_col.get_output(0), B_reshape.get_output(0),
        trt.ElementWiseOperation.PROD)
    # dt_B: [d_inner, state_size]

    # x for scan: conv_activated [1, d_inner] -> [d_inner, 1]
    x_col2 = network.add_shuffle(conv_activated)
    x_col2.reshape_dims = (d_inner, 1)

    # dt_B * x: [d_inner, state_size] * [d_inner, 1] -> [d_inner, state_size]
    dtBx = network.add_elementwise(
        dt_B.get_output(0), x_col2.get_output(0),
        trt.ElementWiseOperation.PROD)

    # SSM update: new_ssm = A_bar * old_ssm + dt_B * x
    decay = network.add_elementwise(
        A_bar.get_output(0), ssm_state_in,
        trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dtBx.get_output(0),
        trt.ElementWiseOperation.SUM)
    present_ssm = new_ssm.get_output(0)  # [d_inner, state_size]

    # Output: y = C * new_ssm  (sum over state_size dim)
    # C: [1, state_size] -> [1, state_size], new_ssm: [d_inner, state_size]
    # We want: for each d in d_inner: y[d] = sum_s(C[s] * new_ssm[d, s])
    # = matmul new_ssm @ C^T
    # new_ssm [d_inner, state_size] @ C^T [state_size, 1] -> [d_inner, 1]
    C_reshape2 = network.add_shuffle(C)
    C_reshape2.reshape_dims = (state_size, 1)
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE,
        C_reshape2.get_output(0), trt.MatrixOperation.NONE)
    # y_matmul: [d_inner, 1] -> reshape to [1, d_inner]
    y_flat = network.add_shuffle(y_matmul.get_output(0))
    y_flat.reshape_dims = (1, d_inner)

    # D * x (skip connection): D [d_inner] * conv_activated [1, d_inner]
    D_const = graph_ops.add_constant(
        network, (1, d_inner), weights[f"{prefix}.D"])
    Dx = network.add_elementwise(
        D_const, conv_activated, trt.ElementWiseOperation.PROD)

    # y = y + D*x
    y = network.add_elementwise(
        y_flat.get_output(0), Dx.get_output(0),
        trt.ElementWiseOperation.SUM)
    # y: [1, d_inner]

    # ===== 7. Gate and output =====
    # y * silu(z)
    z_activated = graph_ops.add_activation(network, z, "silu")
    gated = network.add_elementwise(
        y.get_output(0), z_activated,
        trt.ElementWiseOperation.PROD)

    # out_proj: [1, d_inner] @ [d_inner, hidden] -> [1, hidden]
    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size,
        weights[f"{prefix}.w_out"])

    # Residual connection
    residual = network.add_elementwise(
        hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


plugin = MambaPlugin()
