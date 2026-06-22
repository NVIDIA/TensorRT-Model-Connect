"""ModernBERT family plugin -- encoder-only transformer with modern design.

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

import sys
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from . import graph_ops
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)

from tensorrt_model_connect import trt_compat



trt = trt_compat.get_trt() if trt_compat.is_available() else None

def _add_layernorm_no_bias(network, inp, hidden_size, gamma, eps):
    """LayerNorm without bias via TRT native normalization.

    ModernBERT uses nn.LayerNorm(bias=False) which still mean-centers,
    unlike RMSNorm which does not.
    """
    beta = np.zeros(hidden_size, dtype=np.float32)
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps)


class ModernbertPlugin:
    name = "modernbert"
    runtime_strategy = "encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("modernbert")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
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
        weights["embed_norm"] = _load_tensor(
            readers, "model.embeddings.norm.weight").astype(np.float32)

        # Final LayerNorm
        weights["final_norm"] = _load_tensor(
            readers, "model.final_norm.weight").astype(np.float32)

        # MLM head weights (optional)
        if _has_tensor(readers, "head.dense.weight"):
            weights["head_dense_w"] = np.ascontiguousarray(
                _load_tensor(readers, "head.dense.weight").T.astype(np.float32))
        if _has_tensor(readers, "head.norm.weight"):
            weights["head_norm"] = _load_tensor(
                readers, "head.norm.weight").astype(np.float32)
        if _has_tensor(readers, "decoder.bias"):
            weights["decoder_bias"] = _load_tensor(
                readers, "decoder.bias").astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Attention LayerNorm (layer 0 has no attn_norm)
            attn_norm_key = f"{hf_prefix}.attn_norm.weight"
            if _has_tensor(readers, attn_norm_key):
                weights[f"{prefix}.attn_norm"] = _load_tensor(
                    readers, attn_norm_key).astype(np.float32)

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
                readers, f"{hf_prefix}.mlp_norm.weight").astype(np.float32)

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
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="ModernBERT tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("ModernBERT tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_modernbert_engine
            return build_tp_modernbert_engine(
                config, weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads
        intermediate = config.intermediate_size
        eps = config.raw.get("norm_eps", config.rms_norm_eps)
        max_seq = max_cache_length

        # Per-layer RoPE theta from layer_types
        layer_types = config.raw.get("layer_types", [])
        rope_params = config.raw.get("rope_parameters", {})
        # Default theta values
        full_theta = 160000.0
        sliding_theta = 10000.0
        if rope_params:
            if "full_attention" in rope_params and rope_params["full_attention"]:
                full_theta = rope_params["full_attention"].get("rope_theta", 160000.0)
            if "sliding_attention" in rope_params and rope_params["sliding_attention"]:
                sliding_theta = rope_params["sliding_attention"].get("rope_theta", 10000.0)

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        # Inputs
        input_ids = network.add_input("input_ids", trt.int32, (max_seq,))
        attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq,))

        # Attention mask: [seq] int -> [1, 1, 1, seq] additive float mask.
        mask_float = network.add_cast(attention_mask_input, trt.float32)
        ones_c = graph_ops.add_constant(network, (1,), np.array([1.0], dtype=np.float32))
        neg_large = graph_ops.add_constant(network, (1,), np.array([-1e10], dtype=np.float32))
        inv_mask = network.add_elementwise(
            ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB)
        pad_penalty = network.add_elementwise(
            inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD)
        pad_mask_4d = network.add_shuffle(pad_penalty.get_output(0))
        pad_mask_4d.reshape_dims = (1, 1, 1, max_seq)

        # Pre-compute RoPE tables for both theta values
        rope_tables = {}
        for theta in set([full_theta, sliding_theta]):
            cos = graph_ops.add_constant(
                network, (max_seq, head_dim // 2),
                graph_ops.make_rope_table_half_dim(
                    max_seq, head_dim, theta, cosine=True))
            sin = graph_ops.add_constant(
                network, (max_seq, head_dim // 2),
                graph_ops.make_rope_table_half_dim(
                    max_seq, head_dim, theta, cosine=False))
            rope_tables[theta] = (cos, sin)

        pos_indices = graph_ops.add_constant(
            network, (max_seq,), np.arange(max_seq, dtype=np.int32),
            dtype=np.int32)

        # Embedding
        embed_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"])
        word_embed = network.add_gather(embed_table, input_ids, 0)
        hidden_state = _add_layernorm_no_bias(
            network, word_embed.get_output(0), hidden, weights["embed_norm"], eps)

        # Encoder layers
        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            # Determine RoPE theta for this layer
            if layer_idx < len(layer_types):
                lt = layer_types[layer_idx]
                if lt in ("full_attention", "global_attention"):
                    theta = full_theta
                else:
                    theta = sliding_theta
            else:
                theta = full_theta
            cos_table, sin_table = rope_tables[theta]

            # Pre-norm attention
            has_attn_norm = f"{prefix}.attn_norm" in weights
            if has_attn_norm:
                attn_input = _add_layernorm_no_bias(
                    network, hidden_state, hidden,
                    weights[f"{prefix}.attn_norm"], eps)
            else:
                attn_input = hidden_state

            # QKV projections
            q = graph_ops.add_matmul_rhs_constant(network, attn_input, hidden, hidden, weights[f"{prefix}.w_q"])
            k = graph_ops.add_matmul_rhs_constant(network, attn_input, hidden, hidden, weights[f"{prefix}.w_k"])
            v = graph_ops.add_matmul_rhs_constant(network, attn_input, hidden, hidden, weights[f"{prefix}.w_v"])

            # RoPE
            q = graph_ops.add_apply_rope_native(
                network, q, num_heads, head_dim,
                cos_table, sin_table, pos_indices, head_dim,
                sequence_length=max_seq)
            k = graph_ops.add_apply_rope_native(
                network, k, num_heads, head_dim,
                cos_table, sin_table, pos_indices, head_dim,
                sequence_length=max_seq)

            context_flat = graph_ops.add_attention_from_rows(
                network, q, k, v,
                num_heads=num_heads, head_dim=head_dim,
                q_seq=max_seq, kv_seq=max_seq,
                mask=pad_mask_4d.get_output(0))

            attn_out = graph_ops.add_matmul_rhs_constant(
                network, context_flat, hidden, hidden, weights[f"{prefix}.w_o"])

            # Residual
            res1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = res1.get_output(0)

            # Pre-norm GeGLU MLP
            mlp_input = _add_layernorm_no_bias(
                network, hidden_state, hidden, weights[f"{prefix}.mlp_norm"], eps)

            # GeGLU: act(input) * gate
            inp_proj = graph_ops.add_matmul_rhs_constant(
                network, mlp_input, hidden, intermediate, weights[f"{prefix}.w_mlp_input"])
            gate_proj = graph_ops.add_matmul_rhs_constant(
                network, mlp_input, hidden, intermediate, weights[f"{prefix}.w_mlp_gate"])
            inp_act = graph_ops.add_gelu_erf(network, inp_proj)
            gated = network.add_elementwise(inp_act, gate_proj, trt.ElementWiseOperation.PROD)

            down = graph_ops.add_matmul_rhs_constant(
                network, gated.get_output(0), intermediate, hidden, weights[f"{prefix}.w_down"])

            res2 = network.add_elementwise(hidden_state, down, trt.ElementWiseOperation.SUM)
            hidden_state = res2.get_output(0)

        # Final norm
        hidden_state = _add_layernorm_no_bias(
            network, hidden_state, hidden, weights["final_norm"], eps)

        hidden_state.name = "hidden_states"
        network.mark_output(hidden_state)

        if verbose:
            print(f"[trtmc build] Building ModernBERT encoder TRT engine "
                  f"({num_layers} layers, hidden={hidden}, seq_len={max_seq}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        return bytes(plan)


plugin = ModernbertPlugin()
