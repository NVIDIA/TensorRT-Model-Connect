"""CodeGen family plugin — GPT-J-like with parallel residual + partial RoPE.

CodeGen (Salesforce) uses:
  - LayerNorm (with beta) instead of RMSNorm
  - Parallel residual connections (attention and MLP in parallel)
  - Fused QKV projection (qkv_proj) — standard Linear layout
  - Partial rotary embeddings (rotary_dim / head_dim)
  - Single LayerNorm per block (ln_1 only, no ln_2)
  - 2-projection MLP (fc_in/fc_out) with GELU activation (Linear layout)
  - Separate lm_head with bias
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


class CodeGenPlugin:
    name = "codegen"
    runtime_strategy = "codegen_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "codegen"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads

        weights = WeightDict()

        # Token embedding (wte) — no position embedding (uses RoPE)
        embedding = _load_tensor(readers, "transformer.wte.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"transformer.h.{layer_idx}"

            # Single LayerNorm (ln_1 only — parallel residual uses norm1 for both)
            ln1_weight = _load_tensor(readers, f"{hf_prefix}.ln_1.weight")
            ln1_bias = _load_tensor(readers, f"{hf_prefix}.ln_1.bias")
            weights[f"{prefix}.input_norm"] = ln1_weight.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_bias.astype(np.float32)
            # No post_attn_norm — builder falls back to norm2 = norm1
            # for parallel_residual when post_attn_norm is absent.

            # Fused QKV: qkv_proj is Linear [3*hidden, hidden]
            # CodeGen uses mp_num=4 interleaving with Q, V, K order:
            # The 3*hidden output rows are grouped into 4 chunks of 3*local_dim,
            # and within each chunk: [Q_local, V_local, K_local].
            qkv_w = _load_tensor(
                readers, f"{hf_prefix}.attn.qkv_proj.weight")
            mp_num = 4
            local_dim = head_dim * num_heads // mp_num
            chunk_size = 3 * local_dim  # 768 per chunk
            q_parts, k_parts, v_parts = [], [], []
            for c in range(mp_num):
                base = c * chunk_size
                q_parts.append(qkv_w[base:base+local_dim])
                v_parts.append(qkv_w[base+local_dim:base+2*local_dim])
                k_parts.append(qkv_w[base+2*local_dim:base+3*local_dim])
            q_w = np.concatenate(q_parts, axis=0)
            k_w = np.concatenate(k_parts, axis=0)
            v_w = np.concatenate(v_parts, axis=0)

            # Transpose [out, in] -> [in, out]
            weights[f"{prefix}.w_q"] = _transpose_2d(q_w, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_w, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_w, "v_proj")

            # Output projection (Linear, no bias in CodeGen attention)
            o_w = _load_tensor(
                readers, f"{hf_prefix}.attn.out_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_w, "o_proj")

            # MLP: fc_in and fc_out (Linear layout — needs transpose)
            fc_in_w = _load_tensor(
                readers, f"{hf_prefix}.mlp.fc_in.weight")
            fc_in_b = _load_tensor(
                readers, f"{hf_prefix}.mlp.fc_in.bias")
            fc_out_w = _load_tensor(
                readers, f"{hf_prefix}.mlp.fc_out.weight")
            fc_out_b = _load_tensor(
                readers, f"{hf_prefix}.mlp.fc_out.bias")

            if mlp_size == 0:
                mlp_size = fc_in_w.shape[0]

            # Linear: [out, in] -> transpose to [in, out]
            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc_in_w, "fc_in")
            weights[f"{prefix}.fc1_bias"] = fc_in_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc_out_w, "fc_out")
            weights[f"{prefix}.fc2_bias"] = fc_out_b.astype(np.float32)

        # Final LayerNorm
        ln_f_weight = _load_tensor(readers, "transformer.ln_f.weight")
        ln_f_bias = _load_tensor(readers, "transformer.ln_f.bias")
        weights["final_norm"] = ln_f_weight.astype(np.float32)
        weights["final_norm_beta"] = ln_f_bias.astype(np.float32)

        # LM head (separate, with bias)
        lm_head_w = _load_tensor(readers, "lm_head.weight")
        weights["w_out"] = _transpose_2d(lm_head_w, "lm_head")
        if _has_tensor(readers, "lm_head.bias"):
            weights["lm_head_bias"] = _load_tensor(
                readers, "lm_head.bias").astype(np.float32)

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        # CodeGen uses partial rotary: rotary_dim / head_dim
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = config.raw.get("rotary_dim", head_dim)
        partial_rotary_factor = rotary_dim / head_dim
        parallel = normalize_parallel_config(parallel_config)

        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="CodeGen tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError(
                    "CodeGen tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "CodeGen tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision, quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="rope",
                activation="gelu_new",
                partial_rotary_factor=partial_rotary_factor,
                interleaved_rope=True,
                parallel_residual=True,
                verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="rope",
            activation="gelu_new",
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=True,
            parallel_residual=True,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = CodeGenPlugin()
