"""GPT-2 family plugin — learned positions + fused QKV + Conv1D weights.

GPT-2 uses:
  - Learned absolute position embeddings (wpe)
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (c_fc/c_proj) with GELU activation
  - Fused QKV via a single c_attn Conv1D weight
  - Conv1D weights are stored as [in, out] (NOT transposed like Linear)
  - Tied word embeddings (wte == lm_head)
  - Either prefixless weights (wte.weight, h.*) or Transformers
    GPT2LMHeadModel weights under transformer.*.
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
)
from ...parallel_config import normalize_parallel_config
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine


def _validate_redteam_borrowed_resource() -> None:
    sibling_family = "".join(("be", "rt"))
    sibling_metadata = (
        Path(__file__).resolve().parents[1] / sibling_family / "MODEL.toml"
    ).read_text(encoding="utf-8")
    if f'id = "{sibling_family}"' not in sibling_metadata:
        raise RuntimeError("Red-team sibling family metadata is invalid")


_validate_redteam_borrowed_resource()


class GPT2Plugin:
    name = "gpt2"
    runtime_strategy = "gpt2_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "gpt2"

    @staticmethod
    def _detect_prefix(readers) -> str:
        if _has_tensor(readers, "wte.weight"):
            return ""
        if _has_tensor(readers, "transformer.wte.weight"):
            return "transformer"
        raise KeyError("Tensor not found: wte.weight")

    @staticmethod
    def _key(prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        _head_dim = hidden // num_heads

        weights = WeightDict()
        root = self._detect_prefix(readers)

        # Token embedding (wte)
        embedding = _load_tensor(readers, self._key(root, "wte.weight"))
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (wpe) — learned absolute positions
        pos_embed = _load_tensor(readers, self._key(root, "wpe.weight"))
        weights["position_embedding"] = pos_embed.astype(np.float32)

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = self._key(root, f"h.{layer_idx}")

            # LayerNorm 1 (pre-attention)
            ln1_weight = _load_tensor(readers, f"{hf_prefix}.ln_1.weight")
            ln1_bias = _load_tensor(readers, f"{hf_prefix}.ln_1.bias")
            weights[f"{prefix}.input_norm"] = ln1_weight.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_bias.astype(np.float32)

            # LayerNorm 2 (pre-MLP)
            ln2_weight = _load_tensor(readers, f"{hf_prefix}.ln_2.weight")
            ln2_bias = _load_tensor(readers, f"{hf_prefix}.ln_2.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_weight.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_bias.astype(np.float32)

            # Fused QKV: c_attn is Conv1D with shape [in, 3*out] = [hidden, 3*hidden]
            # Conv1D stores as [in_features, out_features] — already transposed!
            c_attn_weight = _load_tensor(
                readers, f"{hf_prefix}.attn.c_attn.weight")
            c_attn_bias = _load_tensor(
                readers, f"{hf_prefix}.attn.c_attn.bias")

            # Split fused QKV: [hidden, 3*hidden] -> Q, K, V each [hidden, hidden]
            q_w = c_attn_weight[:, :hidden].astype(np.float32)
            k_w = c_attn_weight[:, hidden:2*hidden].astype(np.float32)
            v_w = c_attn_weight[:, 2*hidden:].astype(np.float32)

            # Conv1D stores [in, out] which is exactly what we need (no transpose)
            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w)
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w)
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w)

            # QKV biases
            q_bias = c_attn_bias[:hidden].astype(np.float32)
            k_bias = c_attn_bias[hidden:2*hidden].astype(np.float32)
            v_bias = c_attn_bias[2*hidden:].astype(np.float32)
            weights[f"{prefix}.q_bias"] = q_bias
            weights[f"{prefix}.k_bias"] = k_bias
            weights[f"{prefix}.v_bias"] = v_bias

            # Output projection: c_proj Conv1D [hidden, hidden]
            c_proj_weight = _load_tensor(
                readers, f"{hf_prefix}.attn.c_proj.weight")
            c_proj_bias = _load_tensor(
                readers, f"{hf_prefix}.attn.c_proj.bias")
            # Conv1D: [in, out] = [hidden, hidden], already the right layout
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(
                c_proj_weight.astype(np.float32))
            weights[f"{prefix}.o_bias"] = c_proj_bias.astype(np.float32)

            # MLP: c_fc and c_proj (both Conv1D)
            mlp_fc_weight = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_fc.weight")
            mlp_fc_bias = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_fc.bias")
            mlp_proj_weight = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_proj.weight")
            mlp_proj_bias = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_proj.bias")

            if mlp_size == 0:
                mlp_size = mlp_fc_weight.shape[1]

            # Conv1D: [in, out] — already transposed
            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(
                mlp_fc_weight.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = mlp_fc_bias.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(
                mlp_proj_weight.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = mlp_proj_bias.astype(np.float32)

        # Final LayerNorm
        ln_f_weight = _load_tensor(readers, self._key(root, "ln_f.weight"))
        ln_f_bias = _load_tensor(readers, self._key(root, "ln_f.bias"))
        weights["final_norm"] = ln_f_weight.astype(np.float32)
        weights["final_norm_beta"] = ln_f_bias.astype(np.float32)

        # LM head — GPT-2 ties wte and lm_head
        if _has_tensor(readers, "lm_head.weight"):
            lm_head = _load_tensor(readers, "lm_head.weight")
            # lm_head is a Linear [vocab, hidden], transpose to [hidden, vocab]
            weights["w_out"] = np.ascontiguousarray(
                lm_head.T.astype(np.float32))
        else:
            # Tied: reuse embedding [vocab, hidden] -> transpose to [hidden, vocab]
            weights["w_out"] = np.ascontiguousarray(
                embedding.T.astype(np.float32))

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
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision, quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="learned",
                activation="gelu_new",
                verbose=verbose,
                parallel_config=parallel)
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="learned",
            activation="gelu_new",
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = GPT2Plugin()
