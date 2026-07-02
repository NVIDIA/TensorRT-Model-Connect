# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V2 family plugin — Multi-head Latent Attention + Mixture of Experts.

DeepSeek-V2 uses Multi-head Latent Attention (MLA) which compresses KV cache
via latent projections, plus Mixture of Experts with shared experts. This
plugin implements the "naive" MLA approach: decompress K/V fully, then cache
the decompressed values using the standard KV cache runtime.

Key architecture details:
  - MLA compresses KV into a latent space (kv_lora_rank) then decompresses
  - Partial RoPE: only qk_rope_head_dim dimensions get RoPE
  - K and V have different per-head sizes (K: nope+rope=192, V: v_head_dim=128)
  - V is zero-padded to match K size for uniform cache_state_size
  - First first_k_dense_replace layers use dense SwiGLU MLP
  - Remaining layers use MoE with shared experts that are always active
  - Standard top-k softmax routing with renormalization for routed experts

V2-Lite specifics:
  - q_lora_rank is null: Q is a direct projection, no LoRA compression
  - num_attention_heads == num_key_value_heads == 16 (after decompression)

Weight key mapping:
  HF: model.layers.{i}.self_attn.q_proj.weight     -> w_q [num_heads*(nope+rope), hidden]
  HF: model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight -> w_kv_a [kv_lora_rank+rope, hidden]
  HF: model.layers.{i}.self_attn.kv_a_layernorm.weight     -> kv_a_norm [kv_lora_rank]
  HF: model.layers.{i}.self_attn.kv_b_proj.weight          -> w_kv_b [num_heads*(nope+v), kv_lora_rank]
  HF: model.layers.{i}.self_attn.o_proj.weight              -> w_o [hidden, num_heads*v_head_dim]
  HF: model.layers.{i}.mlp.gate.weight -> moe_gate [n_routed_experts, hidden]
  HF: model.layers.{i}.mlp.experts.{e}.gate_proj.weight -> expert gate
  HF: model.layers.{i}.mlp.experts.{e}.up_proj.weight   -> expert up
  HF: model.layers.{i}.mlp.experts.{e}.down_proj.weight -> expert down
  HF: model.layers.{i}.mlp.shared_experts.gate_proj.weight -> shared gate
  HF: model.layers.{i}.mlp.shared_experts.up_proj.weight   -> shared up
  HF: model.layers.{i}.mlp.shared_experts.down_proj.weight -> shared down
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
from .standard_decoder_builder import _apply_norm, _mark_debug_output


trt = trt_compat.get_trt()

class DeepSeekV2Plugin:
    name = "deepseek_v2"
    runtime_strategy = "deepseek_v2_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("deepseek_v2", "deepseek_v3")

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        """Inject head_dim into bundle config.json for C++ runtime.

        The C++ fast_path_config parser computes:
            head_dim = config["head_dim"] or (hidden_size / num_attention_heads)
            attention_size = num_heads * head_dim

        For DeepSeek-V2/MLA, the effective head_dim for K cache is
        qk_nope_head_dim + qk_rope_head_dim (e.g. 128 + 64 = 192), not the
        default hidden_size / num_heads (e.g. 2048 / 16 = 128). We inject
        the correct head_dim so the C++ runtime allocates the right cache.
        """
        raw = config.raw
        qk_nope = raw.get("qk_nope_head_dim", 128)
        qk_rope = raw.get("qk_rope_head_dim", 64)
        return {"head_dim": qk_nope + qk_rope}

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads

        # MLA-specific dimensions from config
        raw = config.raw
        qk_nope_head_dim = raw.get("qk_nope_head_dim", 128)
        qk_rope_head_dim = raw.get("qk_rope_head_dim", 64)
        v_head_dim = raw.get("v_head_dim", 128)
        kv_lora_rank = raw.get("kv_lora_rank", 512)
        q_lora_rank = raw.get("q_lora_rank", None)  # None for V2-Lite

        # MoE config
        n_routed_experts = raw.get("n_routed_experts", 64)
        n_shared_experts = raw.get("n_shared_experts", 2)
        num_experts_per_tok = raw.get("num_experts_per_tok", 6)
        first_k_dense_replace = raw.get("first_k_dense_replace", 1)
        moe_layer_freq = raw.get("moe_layer_freq", 1)
        intermediate_size = config.intermediate_size

        # Shared expert intermediate size: proportional to n_shared_experts
        moe_intermediate_size = raw.get("moe_intermediate_size", intermediate_size)
        shared_intermediate = moe_intermediate_size * n_shared_experts

        # K has nope + rope dims per head; V has v_head_dim per head.
        # For uniform cache, we pad V to match K size.
        k_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        attention_size = num_heads * k_head_dim  # cache_state_size for both K and V

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # RMSNorm weights
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # --- MLA attention weights ---

            # Q projection: direct for V2-Lite (q_lora_rank is None)
            # Shape: [num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden]
            if q_lora_rank is not None and q_lora_rank > 0:
                # V2 full: Q goes through LoRA compression
                q_a_raw = _load_tensor(
                    readers, f"{hf_prefix}.self_attn.q_a_proj.weight")
                weights[f"{prefix}.w_q_a"] = _transpose_2d(q_a_raw, "q_a_proj")
                del q_a_raw

                q_a_norm = _load_tensor(
                    readers, f"{hf_prefix}.self_attn.q_a_layernorm.weight")
                weights[f"{prefix}.q_a_norm"] = q_a_norm.astype(np.float32)

                q_b_raw = _load_tensor(
                    readers, f"{hf_prefix}.self_attn.q_b_proj.weight")
                weights[f"{prefix}.w_q_b"] = _transpose_2d(q_b_raw, "q_b_proj")
                del q_b_raw
            else:
                # V2-Lite: direct Q projection
                q_raw = _load_tensor(
                    readers, f"{hf_prefix}.self_attn.q_proj.weight")
                weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
                del q_raw

            # KV-A projection with MQA (kv_lora_rank + qk_rope_head_dim, hidden)
            kv_a_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.kv_a_proj_with_mqa.weight")
            weights[f"{prefix}.w_kv_a"] = _transpose_2d(kv_a_raw, "kv_a_proj")
            del kv_a_raw

            # KV-A LayerNorm on the latent (kv_lora_rank dims)
            kv_a_norm = _load_tensor(
                readers, f"{hf_prefix}.self_attn.kv_a_layernorm.weight")
            weights[f"{prefix}.kv_a_norm"] = kv_a_norm.astype(np.float32)

            # KV-B projection: decompresses latent to per-head K_nope and V
            # Shape: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            kv_b_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.kv_b_proj.weight")
            weights[f"{prefix}.w_kv_b"] = _transpose_2d(kv_b_raw, "kv_b_proj")
            del kv_b_raw

            # Output projection: [hidden, num_heads * v_head_dim]
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # --- MLP weights (dense or MoE depending on layer) ---
            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and (layer_idx - first_k_dense_replace) % moe_layer_freq == 0
            )

            if is_moe_layer:
                # Router weight
                router_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate.weight")
                weights[f"{prefix}.router"] = _transpose_2d(
                    router_raw, "router")
                del router_raw

                # Per-expert weights
                for e in range(n_routed_experts):
                    exp_hf = f"{hf_prefix}.mlp.experts.{e}"
                    gate_raw = _load_tensor(
                        readers, f"{exp_hf}.gate_proj.weight")
                    up_raw = _load_tensor(
                        readers, f"{exp_hf}.up_proj.weight")
                    down_raw = _load_tensor(
                        readers, f"{exp_hf}.down_proj.weight")

                    weights[f"{prefix}.expert.{e}.w_gate"] = _transpose_2d(
                        gate_raw, f"expert_{e}_gate")
                    weights[f"{prefix}.expert.{e}.w_up"] = _transpose_2d(
                        up_raw, f"expert_{e}_up")
                    weights[f"{prefix}.expert.{e}.w_down"] = _transpose_2d(
                        down_raw, f"expert_{e}_down")
                    del gate_raw, up_raw, down_raw

                # Shared expert weights (always active)
                shared_hf = f"{hf_prefix}.mlp.shared_experts"
                s_gate_raw = _load_tensor(
                    readers, f"{shared_hf}.gate_proj.weight")
                s_up_raw = _load_tensor(
                    readers, f"{shared_hf}.up_proj.weight")
                s_down_raw = _load_tensor(
                    readers, f"{shared_hf}.down_proj.weight")

                weights[f"{prefix}.shared.w_gate"] = _transpose_2d(
                    s_gate_raw, "shared_gate")
                weights[f"{prefix}.shared.w_up"] = _transpose_2d(
                    s_up_raw, "shared_up")
                weights[f"{prefix}.shared.w_down"] = _transpose_2d(
                    s_down_raw, "shared_down")
                del s_gate_raw, s_up_raw, s_down_raw
            else:
                # Dense MLP
                gate_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate_proj.weight")
                up_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.up_proj.weight")
                down_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.down_proj.weight")

                weights[f"{prefix}.w_gate"] = _transpose_2d(
                    gate_raw, "gate_proj")
                weights[f"{prefix}.w_up"] = _transpose_2d(
                    up_raw, "up_proj")
                weights[f"{prefix}.w_down"] = _transpose_2d(
                    down_raw, "down_proj")
                del gate_raw, up_raw, down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        # Store metadata for engine builder
        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_qk_nope_head_dim"] = qk_nope_head_dim  # type: ignore[assignment]
        weights["_qk_rope_head_dim"] = qk_rope_head_dim  # type: ignore[assignment]
        weights["_v_head_dim"] = v_head_dim  # type: ignore[assignment]
        weights["_kv_lora_rank"] = kv_lora_rank  # type: ignore[assignment]
        weights["_q_lora_rank"] = q_lora_rank  # type: ignore[assignment]
        weights["_n_routed_experts"] = n_routed_experts  # type: ignore[assignment]
        weights["_n_shared_experts"] = n_shared_experts  # type: ignore[assignment]
        weights["_num_experts_per_tok"] = num_experts_per_tok  # type: ignore[assignment]
        weights["_first_k_dense_replace"] = first_k_dense_replace  # type: ignore[assignment]
        weights["_moe_layer_freq"] = moe_layer_freq  # type: ignore[assignment]
        weights["_moe_intermediate_size"] = moe_intermediate_size  # type: ignore[assignment]
        weights["_shared_intermediate_size"] = shared_intermediate  # type: ignore[assignment]
        weights["_norm_topk_prob"] = raw.get("norm_topk_prob", False)  # type: ignore[assignment]
        weights["_routed_scaling_factor"] = raw.get("routed_scaling_factor", 1.0)  # type: ignore[assignment]

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
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="DeepSeek-V2 tensor-parallel builds")
            from .tp_builder import build_deepseek_v2_tp_engine
            return build_deepseek_v2_tp_engine(
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
        num_heads = config.num_attention_heads

        # MLA dimensions
        qk_nope_head_dim: int = weights["_qk_nope_head_dim"]
        qk_rope_head_dim: int = weights["_qk_rope_head_dim"]
        v_head_dim: int = weights["_v_head_dim"]
        kv_lora_rank: int = weights["_kv_lora_rank"]
        q_lora_rank = weights["_q_lora_rank"]

        # MoE dimensions
        n_routed_experts: int = weights["_n_routed_experts"]
        n_shared_experts: int = weights["_n_shared_experts"]
        num_experts_per_tok: int = weights["_num_experts_per_tok"]
        first_k_dense_replace: int = weights["_first_k_dense_replace"]
        moe_layer_freq: int = weights["_moe_layer_freq"]
        moe_intermediate: int = weights["_moe_intermediate_size"]
        shared_intermediate: int = weights["_shared_intermediate_size"]
        norm_topk_prob: bool = weights["_norm_topk_prob"]
        routed_scaling_factor: float = weights["_routed_scaling_factor"]
        dense_intermediate = config.intermediate_size

        # K head dim = nope + rope; this is the per-head cache dimension
        k_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        attention_size = num_heads * k_head_dim  # uniform cache size
        attention_window = max_cache_length + 1

        # Attention scale: 1 / sqrt(full_head_dim) where full = nope + rope
        # HF uses: self.scaling = self.qk_head_dim ** (-0.5)
        # YaRN mscale is handled via rope_utils attention_factor which scales
        # cos/sin directly. For V2-Lite, mscale == mscale_all_dim so they
        # cancel out (attention_factor = 1.0). No adjustment needed here.
        attn_scale = 1.0 / np.sqrt(max(k_head_dim, 1))

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # -----------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                trt.float32, (max_cache_length, attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                trt.float32, (max_cache_length, attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # -----------------------------------------------------------
        # Shared constants
        # -----------------------------------------------------------
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])

        # DeepSeek-V2 uses complex (interleaved) RoPE: adjacent dims (d, d+1)
        # share a frequency, matching HF's apply_rotary_emb with torch.polar.
        rope_scaling = config.raw.get("rope_scaling")
        if rope_scaling and rope_scaling.get("type") == "yarn":
            yarn_kwargs = dict(
                scaling_factor=rope_scaling["factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
                beta_fast=rope_scaling["beta_fast"],
                beta_slow=rope_scaling["beta_slow"],
            )
            cos_half_np = graph_ops.make_yarn_rope_table_half_dim(
                attention_window, qk_rope_head_dim,
                config.rope_theta, True, **yarn_kwargs, interleaved=True)
            sin_half_np = graph_ops.make_yarn_rope_table_half_dim(
                attention_window, qk_rope_head_dim,
                config.rope_theta, False, **yarn_kwargs, interleaved=True)
        else:
            cos_half_np = graph_ops.make_rope_table_half_dim(
                attention_window, qk_rope_head_dim,
                config.rope_theta, True, interleaved=True)
            sin_half_np = graph_ops.make_rope_table_half_dim(
                attention_window, qk_rope_head_dim,
                config.rope_theta, False, interleaved=True)

        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32))

        # -----------------------------------------------------------
        # Embedding lookup
        # -----------------------------------------------------------
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # -----------------------------------------------------------
        # Decoder layers
        # -----------------------------------------------------------
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            is_moe_layer = (
                layer_idx >= first_k_dense_replace
                and (layer_idx - first_k_dense_replace) % moe_layer_freq == 0
            )

            result = _add_deepseek_v2_decoder_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx],
                cache_v=cache_v_inputs[layer_idx],
                attention_mask=attention_mask,
                position_id=position_id,
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                attn_scale=attn_scale,
                eps_tensor=eps_tensor,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                attention_size=attention_size,
                max_cache_length=max_cache_length,
                is_moe_layer=is_moe_layer,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate=moe_intermediate,
                shared_intermediate=shared_intermediate,
                dense_intermediate=dense_intermediate,
                norm_topk_prob=norm_topk_prob,
                routed_scaling_factor=routed_scaling_factor,
            )

            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])

            if debug_layer_outputs:
                _mark_debug_output(
                    network, result["post_attn"],
                    f"debug_post_attn_{layer_idx}")
                _mark_debug_output(
                    network, hidden_state,
                    f"debug_hidden_{layer_idx}")

        # -----------------------------------------------------------
        # Final norm
        # -----------------------------------------------------------
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = _apply_norm(
                network, hidden_state, hidden, final_norm,
                None, eps_tensor, "rmsnorm")

        # -----------------------------------------------------------
        # LM head (logits)
        # -----------------------------------------------------------
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"])
        b_out = np.zeros(vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)

        logits.name = "logits"
        network.mark_output(logits)

        # -----------------------------------------------------------
        # Present K/V outputs
        # -----------------------------------------------------------
        for i in range(num_layers):
            pk = present_k_outputs[i]
            pv = present_v_outputs[i]
            pk.name = graph_ops.layer_tensor_name("present_k", i)
            pv.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(pk)
            network.mark_output(pv)

        # -----------------------------------------------------------
        # Build engine
        # -----------------------------------------------------------
        if verbose:
            print(f"[trtmc build] Building DeepSeek-V2 TRT engine "
                  f"({num_layers} layers, hidden={hidden}, "
                  f"attn={attention_size}, heads={num_heads}, "
                  f"kv_lora_rank={kv_lora_rank}, "
                  f"nope={qk_nope_head_dim}, rope={qk_rope_head_dim}, "
                  f"v_dim={v_head_dim}, "
                  f"experts={n_routed_experts}, shared={n_shared_experts}, "
                  f"top_k={num_experts_per_tok}, "
                  f"cache={max_cache_length}) ...", file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)


# ---------------------------------------------------------------------------
# MLA Attention Block
# ---------------------------------------------------------------------------

def _add_mla_attention_block(
    *,
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale: float,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank,
    attention_size: int,
    max_cache_length: int,
) -> dict[str, trt.ITensor]:
    """Multi-head Latent Attention (MLA) block with naive KV cache.

    Implements the full MLA mechanism:
      1. Q path: direct projection (V2-Lite) or LoRA compression (V2 full)
      2. KV path: compress -> norm -> decompress -> split K_nope / V
      3. Partial RoPE on Q_rope and K_rope
      4. Broadcast K_rope from single-head to all heads
      5. Assemble full K = [K_nope, K_rope], Q = [Q_nope, Q_rope]
      6. Pad V with zeros to match K head dim for uniform cache
      7. Standard scaled dot-product attention with KV cache

    Returns {"attn_out", "present_k", "present_v"}.
    """
    attention_window = max_cache_length + 1
    k_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192 for V2-Lite
    q_total = num_heads * k_head_dim
    rope_total = num_heads * qk_rope_head_dim

    # ===== Q path =====
    if q_lora_rank is not None and q_lora_rank > 0:
        # V2 full: Q goes through LoRA compression
        # hidden -> q_a_proj -> q_a_layernorm -> q_b_proj
        q_compressed = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_lora_rank,
            weights[f"{prefix}.w_q_a"])  # [1, q_lora_rank]
        q_compressed = graph_ops.add_rms_norm(
            network, q_compressed, q_lora_rank,
            weights[f"{prefix}.q_a_norm"], eps_tensor)  # [1, q_lora_rank]
        q = graph_ops.add_matmul_rhs_constant(
            network, q_compressed, q_lora_rank, q_total,
            weights[f"{prefix}.w_q_b"])  # [1, q_total]
    else:
        # V2-Lite: direct Q projection
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_total,
            weights[f"{prefix}.w_q"])  # [1, num_heads * (nope + rope)]

    # Split Q into nope and rope parts per head:
    # q shape: [1, num_heads * (nope + rope)]
    # Reshape to [num_heads, nope + rope] then split
    q_reshaped = network.add_shuffle(q)
    q_reshaped.reshape_dims = (num_heads, k_head_dim)

    # Q_nope: [num_heads, qk_nope_head_dim]
    q_nope_slice = network.add_slice(
        q_reshaped.get_output(0),
        start=(0, 0),
        shape=(num_heads, qk_nope_head_dim),
        stride=(1, 1))
    q_nope = q_nope_slice.get_output(0)

    # Q_rope: [num_heads, qk_rope_head_dim]
    q_rope_slice = network.add_slice(
        q_reshaped.get_output(0),
        start=(0, qk_nope_head_dim),
        shape=(num_heads, qk_rope_head_dim),
        stride=(1, 1))

    # Flatten Q_rope for RoPE application: [1, num_heads * qk_rope_head_dim]
    q_rope_flat = network.add_shuffle(q_rope_slice.get_output(0))
    q_rope_flat.reshape_dims = (1, rope_total)

    # Apply native RoPE to Q_rope
    q_rope_roped = graph_ops.add_apply_rope_native(
        network,
        q_rope_flat.get_output(0),
        num_heads,
        qk_rope_head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        qk_rope_head_dim,
        interleaved=True)

    # Reshape back to [num_heads, qk_rope_head_dim]
    q_rope_heads = network.add_shuffle(q_rope_roped)
    q_rope_heads.reshape_dims = (num_heads, qk_rope_head_dim)

    # Assemble full Q: [num_heads, k_head_dim] = [Q_nope, Q_rope]
    q_full_cat = network.add_concatenation([q_nope, q_rope_heads.get_output(0)])
    q_full_cat.axis = 1  # concat on head_dim axis
    # q_full: [num_heads, k_head_dim]

    # ===== KV path =====
    # Step 1: KV-A projection with MQA
    # hidden -> [1, kv_lora_rank + qk_rope_head_dim]
    kv_a_dim = kv_lora_rank + qk_rope_head_dim
    c_kv = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_a_dim,
        weights[f"{prefix}.w_kv_a"])

    # Split into latent and k_rope_pass
    # c_kv_latent: [1, kv_lora_rank]
    c_kv_latent_slice = network.add_slice(
        c_kv, start=(0, 0), shape=(1, kv_lora_rank), stride=(1, 1))
    c_kv_latent = c_kv_latent_slice.get_output(0)

    # k_rope_pass: [1, qk_rope_head_dim] -- single-head rope input for K
    k_rope_pass_slice = network.add_slice(
        c_kv, start=(0, kv_lora_rank), shape=(1, qk_rope_head_dim), stride=(1, 1))
    k_rope_pass = k_rope_pass_slice.get_output(0)

    # Step 2: RMSNorm on latent
    c_kv_normed = graph_ops.add_rms_norm(
        network, c_kv_latent, kv_lora_rank,
        weights[f"{prefix}.kv_a_norm"], eps_tensor)

    # Step 3: KV-B projection: decompress
    # [1, kv_lora_rank] -> [1, num_heads * (qk_nope_head_dim + v_head_dim)]
    kv_b_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_expanded = graph_ops.add_matmul_rhs_constant(
        network, c_kv_normed, kv_lora_rank, kv_b_out_dim,
        weights[f"{prefix}.w_kv_b"])

    # Split into K_nope and V per head
    # Reshape to [num_heads, qk_nope_head_dim + v_head_dim]
    kv_per_head = network.add_shuffle(kv_expanded)
    kv_per_head.reshape_dims = (num_heads, qk_nope_head_dim + v_head_dim)

    # K_nope: [num_heads, qk_nope_head_dim]
    k_nope_slice = network.add_slice(
        kv_per_head.get_output(0),
        start=(0, 0),
        shape=(num_heads, qk_nope_head_dim),
        stride=(1, 1))
    k_nope = k_nope_slice.get_output(0)

    # V: [num_heads, v_head_dim]
    v_slice = network.add_slice(
        kv_per_head.get_output(0),
        start=(0, qk_nope_head_dim),
        shape=(num_heads, v_head_dim),
        stride=(1, 1))
    v_heads = v_slice.get_output(0)

    # Step 4: Apply native RoPE to the shared K-rope head, then broadcast.
    k_rope_roped = graph_ops.add_apply_rope_native(
        network,
        k_rope_pass,
        1,
        qk_rope_head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        qk_rope_head_dim,
        interleaved=True)
    k_rope_copies = [k_rope_roped for _ in range(num_heads)]
    k_rope_broadcast = network.add_concatenation(k_rope_copies)
    k_rope_broadcast.axis = 0
    k_rope_heads = k_rope_broadcast.get_output(0)

    # Assemble full K: [num_heads, k_head_dim] = [K_nope, K_rope]
    k_full_cat = network.add_concatenation([k_nope, k_rope_heads])
    k_full_cat.axis = 1

    # Step 5: Pad V to match K head dim for uniform cache
    # V is [num_heads, v_head_dim], pad to [num_heads, k_head_dim]
    pad_size = k_head_dim - v_head_dim
    if pad_size > 0:
        zero_pad = graph_ops.add_constant(
            network, (num_heads, pad_size),
            np.zeros((num_heads, pad_size), dtype=np.float32))
        v_padded_cat = network.add_concatenation([v_heads, zero_pad])
        v_padded_cat.axis = 1
        v_padded = v_padded_cat.get_output(0)  # [num_heads, k_head_dim]
    else:
        v_padded = v_heads

    # Flatten K and V for cache: [1, attention_size]
    k_flat = network.add_shuffle(k_full_cat.get_output(0))
    k_flat.reshape_dims = (1, attention_size)
    v_flat = network.add_shuffle(v_padded)
    v_flat.reshape_dims = (1, attention_size)

    # Save present K/V (for cache update)
    present_k = k_flat.get_output(0)
    present_v = v_flat.get_output(0)

    # ===== Standard attention with cache =====

    # Concatenate with cache
    all_k = network.add_concatenation([cache_k, k_flat.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_flat.get_output(0)])
    all_v.axis = 0

    q_flat = network.add_shuffle(q_full_cat.get_output(0))
    q_flat.reshape_dims = (1, attention_size)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    attn_context = graph_ops.add_attention_from_rows(
        network,
        q_flat.get_output(0),
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=num_heads,
        head_dim=k_head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=mask_4d,
        scale=attn_scale)

    # Slice out only the v_head_dim portion (remove zero-padding)
    if pad_size > 0:
        context_heads = network.add_shuffle(attn_context)
        context_heads.reshape_dims = (num_heads, k_head_dim)
        context_sliced = network.add_slice(
            context_heads.get_output(0),
            start=(0, 0),
            shape=(num_heads, v_head_dim),
            stride=(1, 1))
        context_for_proj = context_sliced.get_output(0)
        context_flat = network.add_shuffle(context_for_proj)
        context_flat.reshape_dims = (1, num_heads * v_head_dim)
        attn_context = context_flat.get_output(0)

    # Output projection: [1, num_heads * v_head_dim] -> [1, hidden_size]
    v_total = num_heads * v_head_dim
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, attn_context,
        v_total, hidden_size,
        weights[f"{prefix}.w_o"])

    return {
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


# ---------------------------------------------------------------------------
# MoE Block with Shared Experts
# ---------------------------------------------------------------------------

def _add_swiglu_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> trt.ITensor:
    """Compute a single SwiGLU expert: down(silu(gate(x)) * up(x))."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up)

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down)
    return down


def _add_moe_with_shared_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    n_routed_experts: int,
    moe_intermediate: int,
    num_experts_per_tok: int,
    shared_intermediate: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
) -> trt.ITensor:
    """MoE block with shared experts (DeepSeek-V2 style).

    1. Router logits -> softmax -> top-k selection
    2. Scale weights: renormalize (norm_topk_prob=True) or multiply by
       routed_scaling_factor (norm_topk_prob=False)
    3. Compute all routed expert outputs, select top-k, weighted sum
    4. Compute shared expert output (always active)
    5. Final = routed_output + shared_output
    """
    # 1. Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, n_routed_experts,
        weights[f"{prefix}.router"])

    # 2. Softmax over router logits
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1

    # 3. TopK selection
    topk = network.add_topk(
        sm.get_output(0), trt.TopKOperation.MAX,
        num_experts_per_tok, 1 << 1)
    top_values = topk.get_output(0)   # [1, top_k]
    top_indices = topk.get_output(1)  # [1, top_k]

    # 4. Scale routing weights (matches HF route_tokens_to_experts)
    if norm_topk_prob:
        # Renormalize: values / sum(values)
        sum_val = network.add_reduce(
            top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
        scaled_weights = network.add_elementwise(
            top_values, sum_val.get_output(0),
            trt.ElementWiseOperation.DIV).get_output(0)  # [1, top_k]
    elif routed_scaling_factor != 1.0:
        # Multiply by routed_scaling_factor
        scale_c = graph_ops.add_constant(
            network, (1, 1),
            np.array([[routed_scaling_factor]], dtype=np.float32))
        scaled_weights = network.add_elementwise(
            top_values, scale_c,
            trt.ElementWiseOperation.PROD).get_output(0)
    else:
        # V2-Lite: use raw softmax top-k values directly (scaling=1.0)
        scaled_weights = top_values

    # 5. Compute ALL routed expert outputs and stack
    expert_outputs = []
    for e in range(n_routed_experts):
        exp_out = _add_swiglu_expert(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{e}.w_gate"],
            weights[f"{prefix}.expert.{e}.w_up"],
            weights[f"{prefix}.expert.{e}.w_down"],
        )
        expert_outputs.append(exp_out)

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)  # [n_routed_experts, hidden_size]

    # 6. Gather selected experts, scale, and sum
    result = None
    for k in range(num_experts_per_tok):
        # Extract index k
        idx_slice = network.add_slice(
            top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        # Extract weight k
        w_slice = network.add_slice(
            scaled_weights,
            start=(0, k), shape=(1, 1), stride=(1, 1))

        # Gather expert output
        expert_out = network.add_gather(
            stacked_out, idx_flat.get_output(0), 0)

        # Scale
        scaled_expert = network.add_elementwise(
            expert_out.get_output(0), w_slice.get_output(0),
            trt.ElementWiseOperation.PROD)

        if result is None:
            result = scaled_expert.get_output(0)
        else:
            sum_layer = network.add_elementwise(
                result, scaled_expert.get_output(0),
                trt.ElementWiseOperation.SUM)
            result = sum_layer.get_output(0)

    # 7. Shared expert output (always active)
    shared_out = _add_swiglu_expert(
        network, inp, hidden_size, shared_intermediate,
        weights[f"{prefix}.shared.w_gate"],
        weights[f"{prefix}.shared.w_up"],
        weights[f"{prefix}.shared.w_down"],
    )

    # 8. Combine: routed_output + shared_output
    combined = network.add_elementwise(
        result, shared_out, trt.ElementWiseOperation.SUM)

    return combined.get_output(0)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

def _add_deepseek_v2_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale: float,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank,
    attention_size: int,
    max_cache_length: int,
    is_moe_layer: bool,
    n_routed_experts: int,
    n_shared_experts: int,
    num_experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dense_intermediate: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
) -> dict[str, trt.ITensor]:
    """Add one DeepSeek-V2 decoder layer: MLA attention + (dense MLP or MoE)."""

    # Pre-attention RMSNorm
    norm1 = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        None, eps_tensor, "rmsnorm")

    # MLA attention block
    attn = _add_mla_attention_block(
        network=network,
        normed=norm1,
        cache_k=cache_k,
        cache_v=cache_v,
        attention_mask=attention_mask,
        position_id=position_id,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        attn_scale=attn_scale,
        eps_tensor=eps_tensor,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        attention_size=attention_size,
        max_cache_length=max_cache_length,
    )
    attn_out = attn["attn_out"]

    # Residual connection after attention
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None, eps_tensor, "rmsnorm")

    # MLP: either dense or MoE with shared experts
    if is_moe_layer:
        mlp_out = _add_moe_with_shared_experts(
            network, norm2, weights, prefix,
            hidden_size, n_routed_experts, moe_intermediate,
            num_experts_per_tok, shared_intermediate,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor)
    else:
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            mlp_size=dense_intermediate)

    # Residual connection after MLP
    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


plugin = DeepSeekV2Plugin()
