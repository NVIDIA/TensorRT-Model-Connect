"""NemotronH family plugin -- Hybrid Mamba-2 + MLP + Attention decoder.

NemotronH (NVIDIA) uses a heterogeneous layer stack with three layer types
defined by hybrid_override_pattern (e.g. "M-M-M-MM-M-M-M*-..."):
  M = Mamba-2 SSM layer
  - = MLP layer (up_proj -> relu2 -> down_proj)
  * = Attention layer (GQA, no RoPE, no bias)

Key differences from Mamba-1 (existing mamba.py):
  Mamba-2 uses State Space Duality (SSD):
    - in_proj -> split into [gate, hidden_B_C, dt]
    - conv1d over hidden_B_C (d_inner + 2*n_groups*d_state channels)
    - After conv+SiLU, split hidden_B_C -> [hidden, B, C]
    - Multi-head SSM (nheads * headdim = d_inner)
    - A is a scalar per head (not per d_inner like Mamba-1)
    - dt from in_proj directly (no separate x_proj/dt_proj)
    - Gated RMSNorm on SSM output: norm(y) * silu(gate)
    - SSM state: [nheads, headdim, d_state] (headdim-aware)

NemotronH Nano 9B: 56 layers (27 mamba2 + 25 mlp + 4 attention)
  - MLP layers: up_proj -> relu2 -> down_proj (NO gate_proj)
  - Attention layers: q/k/v/o_proj (GQA, no RoPE, no bias)

Weight key mapping (HF -> engine):
  backbone.embeddings.weight                           -> embedding
  backbone.layers.{i}.norm.weight                      -> layer.{i}.norm
  backbone.layers.{i}.mixer.in_proj.weight             -> Mamba-2 in_proj
  backbone.layers.{i}.mixer.conv1d.weight/bias         -> Mamba-2 conv state
  backbone.layers.{i}.mixer.dt_bias                    -> Mamba-2 timestep bias
  backbone.layers.{i}.mixer.A_log                      -> Mamba-2 SSM A
  backbone.layers.{i}.mixer.D                          -> Mamba-2 skip connection
  backbone.layers.{i}.mixer.norm.weight                -> Mamba-2 gated RMSNorm
  backbone.layers.{i}.mixer.out_proj.weight            -> Mamba-2 output proj
  backbone.layers.{i}.mixer.up_proj.weight             -> MLP up
  backbone.layers.{i}.mixer.down_proj.weight           -> MLP down
  backbone.layers.{i}.mixer.q/k/v/o_proj.weight        -> Attention QKV + out
  backbone.norm_f.weight                               -> final_norm
  lm_head.weight                                       -> w_lm_head
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
from . import graph_blocks
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


trt = trt_compat.get_trt()

def _parse_layer_types(pattern: str) -> list[str]:
    """Parse hybrid_override_pattern: M=mamba2, -=mlp, *=attention."""
    mapping = {"M": "mamba2", "-": "mlp", "*": "attention"}
    return [mapping[ch] for ch in pattern if ch in mapping]


class NemotronHPlugin:
    name = "nemotron_h"
    runtime_strategy = "hybrid_mamba_attention"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in {"nemotron_h", "nemotron_hybrid"}

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.head_dim
        raw = config.raw

        # Parse layer types from hybrid_override_pattern
        pattern = raw.get("hybrid_override_pattern", "M" * num_layers)
        layer_types = _parse_layer_types(pattern)
        assert len(layer_types) == num_layers, (
            f"Pattern length {len(layer_types)} != num_hidden_layers {num_layers}")

        # Mamba-2 dimensions
        mamba_num_heads = raw.get("mamba_num_heads", 64)
        mamba_head_dim = raw.get("mamba_head_dim", 64)
        d_inner = mamba_num_heads * mamba_head_dim
        n_groups = raw.get("n_groups", 8)
        d_state = raw.get("ssm_state_size", raw.get("mamba_state_dim", 128))
        d_conv = raw.get("conv_kernel", 4)
        conv_dim = d_inner + 2 * n_groups * d_state

        # MLP dimensions
        mlp_intermediate = config.intermediate_size

        # Attention dimensions
        q_dim = num_heads * head_dim
        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "backbone.embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        mamba_count = 0
        attn_count = 0

        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx]
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"backbone.layers.{layer_idx}"

            # RMSNorm (all layer types)
            norm = _load_tensor(readers, f"{hf_prefix}.norm.weight")
            weights[f"{prefix}.input_norm"] = norm.astype(np.float32)

            if lt == "mamba2":
                # in_proj: [proj_size, hidden] where proj_size = d_inner + conv_dim + mamba_num_heads
                in_proj_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.in_proj.weight")
                weights[f"{prefix}.mamba_in_proj"] = _transpose_2d(
                    in_proj_raw, "mamba_in_proj")

                # conv1d: [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
                conv1d_w = _load_tensor(
                    readers, f"{hf_prefix}.mixer.conv1d.weight")
                weights[f"{prefix}.conv1d_weight"] = conv1d_w.reshape(
                    conv_dim, d_conv).astype(np.float32)

                conv1d_b = _load_tensor(
                    readers, f"{hf_prefix}.mixer.conv1d.bias")
                weights[f"{prefix}.conv1d_bias"] = conv1d_b.astype(np.float32)

                # out_proj: [hidden, d_inner]
                out_proj_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.out_proj.weight")
                weights[f"{prefix}.mamba_out_proj"] = _transpose_2d(
                    out_proj_raw, "mamba_out_proj")

                # A_log: [mamba_num_heads]
                A_log = _load_tensor(readers, f"{hf_prefix}.mixer.A_log")
                A = -np.exp(A_log.astype(np.float32))
                weights[f"{prefix}.A"] = A

                # D: [mamba_num_heads]
                D = _load_tensor(readers, f"{hf_prefix}.mixer.D")
                weights[f"{prefix}.D"] = D.astype(np.float32)

                # dt_bias: [mamba_num_heads]
                dt_bias = _load_tensor(readers, f"{hf_prefix}.mixer.dt_bias")
                weights[f"{prefix}.dt_bias"] = dt_bias.astype(np.float32)

                # Gated RMSNorm: [d_inner]
                norm_key = f"{hf_prefix}.mixer.norm.weight"
                if _has_tensor(readers, norm_key):
                    weights[f"{prefix}.mamba_norm"] = _load_tensor(
                        readers, norm_key).astype(np.float32)
                else:
                    weights[f"{prefix}.mamba_norm"] = np.ones(
                        d_inner, dtype=np.float32)

                mamba_count += 1

            elif lt == "mlp":
                # MLP: up_proj -> relu2 -> down_proj (NO gate_proj)
                up_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.up_proj.weight")
                down_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.down_proj.weight")
                weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
                weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")

            elif lt == "attention":
                # Attention: q/k/v/o projections (no bias, no RoPE)
                q_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.q_proj.weight")
                k_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.k_proj.weight")
                v_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.v_proj.weight")
                o_raw = _load_tensor(
                    readers, f"{hf_prefix}.mixer.o_proj.weight")

                q_t = _transpose_2d(q_raw, "q_proj")
                k_t = _transpose_2d(k_raw, "k_proj")
                v_t = _transpose_2d(v_raw, "v_proj")
                o_t = _transpose_2d(o_raw, "o_proj")

                # Compact GQA/MQA K/V

                weights[f"{prefix}.w_q"] = q_t
                weights[f"{prefix}.w_k"] = k_t
                weights[f"{prefix}.w_v"] = v_t
                weights[f"{prefix}.w_o"] = o_t

                attn_count += 1

        # Final norm
        final_norm_key = "backbone.norm_f.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_lm_head"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_lm_head"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        # Metadata for engine builder
        weights["_layer_types"] = layer_types
        weights["_d_inner"] = d_inner
        weights["_d_state"] = d_state
        weights["_d_conv"] = d_conv
        weights["_conv_dim"] = conv_dim
        weights["_mamba_num_heads"] = mamba_num_heads
        weights["_mamba_head_dim"] = mamba_head_dim
        weights["_n_groups"] = n_groups
        weights["_num_mamba_layers"] = mamba_count
        weights["_num_attention_layers"] = attn_count
        weights["_attention_size"] = q_dim
        weights["_mlp_size"] = mlp_intermediate

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build hybrid TRT engine with heterogeneous layer stack."""
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Nemotron-H tensor-parallel builds")
            from .tp_builder import build_nemotron_h_tp_engine

            return build_nemotron_h_tp_engine(
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

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_state: int = weights["_d_state"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        mamba_num_heads: int = weights["_mamba_num_heads"]
        mamba_head_dim: int = weights["_mamba_head_dim"]
        n_groups: int = weights["_n_groups"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attention_size: int = weights["_attention_size"]
        mlp_size: int = weights["_mlp_size"]

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attention_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # --- Inputs ---
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi),
                trt.float32, (conv_dim, d_conv))
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32, (mamba_num_heads, mamba_head_dim, d_state))
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                trt.float32, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                trt.float32, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32))

        # --- Embedding ---
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # --- Layer stack ---
        present_conv_outputs = []
        present_ssm_outputs = []
        present_k_outputs = []
        present_v_outputs = []
        mamba_counter = 0
        attn_counter = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            lt = layer_types[layer_idx]

            if lt == "mamba2":
                result = _add_mamba2_layer(
                    network=network,
                    hidden=hidden_state,
                    conv_state_in=conv_state_inputs[mamba_counter],
                    ssm_state_in=ssm_state_inputs[mamba_counter],
                    eps_tensor=eps_tensor,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    d_inner=d_inner,
                    d_state=d_state,
                    d_conv=d_conv,
                    conv_dim=conv_dim,
                    mamba_num_heads=mamba_num_heads,
                    mamba_head_dim=mamba_head_dim,
                    n_groups=n_groups,
                )
                hidden_state = result["hidden"]
                present_conv_outputs.append(result["present_conv"])
                present_ssm_outputs.append(result["present_ssm"])
                mamba_counter += 1

            elif lt == "mlp":
                result = _add_mlp_layer(
                    network=network,
                    hidden=hidden_state,
                    eps_tensor=eps_tensor,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    mlp_size=mlp_size,
                )
                hidden_state = result["hidden"]

            elif lt == "attention":
                result = graph_blocks.add_attention_block(
                    network, hidden_state,
                    cache_k_inputs[attn_counter],
                    cache_v_inputs[attn_counter],
                    attention_mask, position_id,
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    attention_size=attention_size,
                    kv_attention_size=kv_attention_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_cache_length=max_cache_length,
                    eps_tensor=eps_tensor,
                    position_type="none",
                )
                # add_attention_block does NOT apply residual
                residual = network.add_elementwise(
                    hidden_state, result["attn_out"],
                    trt.ElementWiseOperation.SUM)
                hidden_state = residual.get_output(0)
                present_k_outputs.append(result["present_k"])
                present_v_outputs.append(result["present_v"])
                attn_counter += 1

            if debug_layer_outputs:
                _mark_debug_output(
                    network, hidden_state, f"debug_hidden_{layer_idx}")

        # --- Final norm ---
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_ops.add_rms_norm(
                network, hidden_state, hidden, final_norm, eps_tensor)

        # --- LM head ---
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_lm_head"])
        b_out = np.zeros(vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)
        logits.name = "logits"
        network.mark_output(logits)

        # --- Present state outputs ---
        for mi in range(num_mamba):
            pc = present_conv_outputs[mi]
            ps = present_ssm_outputs[mi]
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            pk = present_k_outputs[ai]
            pv = present_v_outputs[ai]
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        # --- Build ---
        if verbose:
            print(f"[trtmc build] Building NemotronH hybrid TRT engine "
                  f"({num_layers} layers: {num_mamba} mamba2 + "
                  f"{sum(1 for t in layer_types if t == 'mlp')} mlp + "
                  f"{num_attn} attention, "
                  f"hidden={hidden}, d_inner={d_inner}, "
                  f"d_state={d_state}, nheads={mamba_num_heads}, "
                  f"cache={max_cache_length}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Inject hybrid-specific config fields into the bundle."""
        raw = config.raw
        pattern = raw.get("hybrid_override_pattern", "")
        layer_types = _parse_layer_types(pattern)

        mamba_num_heads = raw.get("mamba_num_heads", 64)
        mamba_head_dim = raw.get("mamba_head_dim", 64)
        d_inner = mamba_num_heads * mamba_head_dim
        n_groups = raw.get("n_groups", 8)
        d_state = raw.get("ssm_state_size", raw.get("mamba_state_dim", 128))
        d_conv = raw.get("conv_kernel", 4)

        num_mamba = sum(1 for lt in layer_types if lt == "mamba2")
        num_attn = sum(1 for lt in layer_types if lt == "attention")

        conv_dim = d_inner + 2 * n_groups * d_state

        return {
            "layer_types": layer_types,
            "num_mamba_layers": num_mamba,
            "num_attention_layers": num_attn,
            "d_inner": d_inner,
            "mamba_d_state": d_state,
            "mamba_d_conv": d_conv,
            "mamba_nheads": mamba_num_heads,
            "mamba_head_dim": mamba_head_dim,
            "conv_dim": conv_dim,
            "n_groups": n_groups,
        }


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _add_mamba2_layer(
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
    d_state: int,
    d_conv: int,
    conv_dim: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    n_groups: int,
) -> dict[str, trt.ITensor]:
    """Add one Mamba-2 SSD layer (single-step decode).

    Mamba-2 in_proj splits: [gate(d_inner), hidden_B_C(conv_dim), dt(nheads)]
    Conv1d operates on hidden_B_C (d_inner + 2*n_groups*d_state channels).
    After conv+SiLU, split: hidden[d_inner], B[n_groups*d_state], C[n_groups*d_state].
    SSM state shape: [nheads, headdim, d_state] for full headdim-aware state.

    Returns: {hidden, present_conv, present_ssm}
    """
    groups_state_size = n_groups * d_state

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    # ===== 2. Input projection =====
    proj_dim = d_inner + conv_dim + mamba_num_heads
    projected = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, proj_dim,
        weights[f"{prefix}.mamba_in_proj"])  # [1, proj_dim]

    # Split: gate [d_inner], hidden_B_C [conv_dim], dt [nheads]
    offset = 0
    gate_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    gate = gate_slice.get_output(0)
    offset += d_inner

    hbc_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, conv_dim), stride=(1, 1))
    hidden_B_C = hbc_slice.get_output(0)
    offset += conv_dim

    dt_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, mamba_num_heads), stride=(1, 1))
    dt_raw = dt_slice.get_output(0)

    # ===== 3. Conv1d step on hidden_B_C =====
    # conv_state_in: [conv_dim, d_conv]
    # hidden_B_C: [1, conv_dim] -> [conv_dim, 1]
    hbc_col = network.add_shuffle(hidden_B_C)
    hbc_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(conv_dim, d_conv - 1),
            stride=(1, 1))
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), hbc_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = hbc_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"])
    conv_prod = network.add_elementwise(
        present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM,
        1 << 1, keep_dims=True)
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim,
        weights[f"{prefix}.conv1d_bias"])
    hbc_activated = graph_ops.add_activation(network, conv_out, "silu")

    # ===== 4. Split hidden, B, C from activated output =====
    hidden_x_slice = network.add_slice(
        hbc_activated, start=(0, 0), shape=(1, d_inner), stride=(1, 1))
    hidden_x = hidden_x_slice.get_output(0)

    B_raw_slice = network.add_slice(
        hbc_activated, start=(0, d_inner),
        shape=(1, groups_state_size), stride=(1, 1))
    B_raw = B_raw_slice.get_output(0)

    C_raw_slice = network.add_slice(
        hbc_activated, start=(0, d_inner + groups_state_size),
        shape=(1, groups_state_size), stride=(1, 1))
    C_raw = C_raw_slice.get_output(0)

    # ===== 5. dt: add bias + softplus =====
    dt_bias_const = graph_ops.add_constant(
        network, (1, mamba_num_heads), weights[f"{prefix}.dt_bias"])
    dt_biased = network.add_elementwise(
        dt_raw, dt_bias_const, trt.ElementWiseOperation.SUM)
    dt_exp = network.add_unary(dt_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    dt_exp_p1 = network.add_elementwise(
        dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(
        dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt = dt_softplus.get_output(0)  # [1, mamba_num_heads]

    # ===== 6. Multi-head SSM step =====
    # A: [nheads] -> [nheads, 1, 1] for broadcast
    A_const = graph_ops.add_constant(
        network, (mamba_num_heads, 1, 1),
        weights[f"{prefix}.A"].reshape(mamba_num_heads, 1, 1))

    # dt: [1, nheads] -> [nheads, 1, 1]
    dt_col = network.add_shuffle(dt)
    dt_col.reshape_dims = (mamba_num_heads, 1, 1)

    # dA = exp(dt * A): broadcast to [nheads, headdim, d_state]
    dtA = network.add_elementwise(
        dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    dA = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    # B: [1, n_groups*d_state] -> [n_groups, d_state] -> expand to [nheads, d_state]
    B_grouped = network.add_shuffle(B_raw)
    B_grouped.reshape_dims = (n_groups, d_state)
    heads_per_group = mamba_num_heads // n_groups

    if heads_per_group > 1:
        B_3d = network.add_shuffle(B_grouped.get_output(0))
        B_3d.reshape_dims = (n_groups, 1, d_state)
        tile_ones = graph_ops.add_constant(
            network, (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=np.float32))
        B_tiled = network.add_elementwise(
            B_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        B_heads_s = network.add_shuffle(B_tiled.get_output(0))
        B_heads_s.reshape_dims = (mamba_num_heads, d_state)
        B_heads = B_heads_s.get_output(0)
    else:
        B_heads = B_grouped.get_output(0)

    # C: same group expansion
    C_grouped = network.add_shuffle(C_raw)
    C_grouped.reshape_dims = (n_groups, d_state)

    if heads_per_group > 1:
        C_3d = network.add_shuffle(C_grouped.get_output(0))
        C_3d.reshape_dims = (n_groups, 1, d_state)
        C_tiled = network.add_elementwise(
            C_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        C_heads_s = network.add_shuffle(C_tiled.get_output(0))
        C_heads_s.reshape_dims = (mamba_num_heads, d_state)
        C_heads = C_heads_s.get_output(0)
    else:
        C_heads = C_grouped.get_output(0)

    # x: [1, d_inner] -> [nheads, headdim]
    x_heads = network.add_shuffle(hidden_x)
    x_heads.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # dBx[h,d,s] = dt[h] * B[h,s] * x[h,d]
    # dt_B: [nheads, 1, 1] * [nheads, 1, d_state] -> [nheads, 1, d_state]
    B_3d_expand = network.add_shuffle(B_heads)
    B_3d_expand.reshape_dims = (mamba_num_heads, 1, d_state)
    dt_B = network.add_elementwise(
        dt_col.get_output(0), B_3d_expand.get_output(0),
        trt.ElementWiseOperation.PROD)

    # x: [nheads, headdim] -> [nheads, headdim, 1]
    x_3d = network.add_shuffle(x_heads.get_output(0))
    x_3d.reshape_dims = (mamba_num_heads, mamba_head_dim, 1)

    # dBx: [nheads, headdim, 1] * [nheads, 1, d_state] -> [nheads, headdim, d_state]
    dBx = network.add_elementwise(
        x_3d.get_output(0), dt_B.get_output(0),
        trt.ElementWiseOperation.PROD)

    # SSM update: new_ssm = dA * ssm_state + dBx
    # ssm_state_in: [nheads, headdim, d_state]
    decay = network.add_elementwise(
        dA.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dBx.get_output(0),
        trt.ElementWiseOperation.SUM)
    present_ssm = new_ssm.get_output(0)  # [nheads, headdim, d_state]

    # y[h,d] = sum_s(ssm_state[h,d,s] * C[h,s])
    # C: [nheads, d_state] -> [nheads, d_state, 1]
    C_col = network.add_shuffle(C_heads)
    C_col.reshape_dims = (mamba_num_heads, d_state, 1)
    # batch matmul: [nheads, headdim, d_state] @ [nheads, d_state, 1] -> [nheads, headdim, 1]
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE,
        C_col.get_output(0), trt.MatrixOperation.NONE)
    y_squeeze = network.add_shuffle(y_matmul.get_output(0))
    y_squeeze.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # D skip: D[h] * x[h,d]
    D_const = graph_ops.add_constant(
        network, (mamba_num_heads, 1),
        weights[f"{prefix}.D"].reshape(mamba_num_heads, 1))
    Dx = network.add_elementwise(
        D_const, x_heads.get_output(0), trt.ElementWiseOperation.PROD)

    y_plus_D = network.add_elementwise(
        y_squeeze.get_output(0), Dx.get_output(0),
        trt.ElementWiseOperation.SUM)
    # [nheads, headdim] -> [1, d_inner]
    y_flat = network.add_shuffle(y_plus_D.get_output(0))
    y_flat.reshape_dims = (1, d_inner)

    # ===== 7. Gated Group RMSNorm (norm_before_gate=False) =====
    # HF: output = weight * group_rms_norm(y * silu(gate))
    # Gate is applied BEFORE normalization. RMSNorm is per-group,
    # with group_size = d_inner // n_groups.
    mamba_norm_w = weights[f"{prefix}.mamba_norm"]
    eps_small = graph_ops.add_constant(
        network, (1, 1),
        np.array([1e-5], dtype=np.float32))

    # Step 1: Apply silu(gate) to y BEFORE norm
    gate_activated = graph_ops.add_activation(network, gate, "silu")
    y_gated = network.add_elementwise(
        y_flat.get_output(0), gate_activated, trt.ElementWiseOperation.PROD)

    # Step 2: Group RMSNorm — reshape to [n_groups, group_size], norm per group
    group_size = d_inner // n_groups
    y_grouped = network.add_shuffle(y_gated.get_output(0))
    y_grouped.reshape_dims = (n_groups, group_size)

    sq = network.add_elementwise(
        y_grouped.get_output(0), y_grouped.get_output(0),
        trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        y_grouped.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    # Reshape back to [1, d_inner] and apply weight
    y_flat_normed = network.add_shuffle(normalized.get_output(0))
    y_flat_normed.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), mamba_norm_w)
    gated = network.add_elementwise(
        y_flat_normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    # ===== 8. Output projection + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size,
        weights[f"{prefix}.mamba_out_proj"])

    residual = network.add_elementwise(
        hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_mlp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
) -> dict[str, trt.ITensor]:
    """Add MLP layer: RMSNorm -> up -> relu2 -> down -> residual."""
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    up = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, mlp_size,
        weights[f"{prefix}.w_up"])
    activated = graph_ops.add_activation(network, up, "relu2")
    down = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_size, hidden_size,
        weights[f"{prefix}.w_down"])

    residual = network.add_elementwise(
        hidden, down, trt.ElementWiseOperation.SUM)

    return {"hidden": residual.get_output(0)}


plugin = NemotronHPlugin()
