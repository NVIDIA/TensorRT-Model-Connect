# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 family-owned Hybrid Gated DeltaNet + self-attention builder.

Qwen3.8 is an architectural re-release of Qwen3.5: the checkpoints keep
`model_type: "qwen3_5"` and `architectures: ["Qwen3_5ForConditionalGeneration"]`,
and the tensor layout is unchanged. It is a separate family here because the
model family is this project's unit of ownership (see AGENTS.md) -- Qwen3.8 must
be implementable, validatable and revertable without touching qwen3_5.

Family dispatch therefore cannot key on `model_type`; see
`support.py` for the config-body discriminator.

Qwen3.8 uses a heterogeneous layer stack with two layer types defined by
text_config.layer_types (list of strings):
  "linear_attention" = Gated DeltaNet layer (linear attention with delta rule)
  "full_attention"   = Standard self-attention layer (GQA, partial RoPE, output gating)

Qwen3.8-27B: 64 layers (48 DeltaNet + 16 self-attention), hidden 5120,
24 query heads / 4 KV heads, head_dim 256, vocab 248320, untied lm_head.
Full-attention layers appear every 4th layer (indices 3, 7, 11, ...), i.e.
`full_attention_interval: 4`.

DeltaNet head expansion: `linear_num_key_heads: 16` carries Q/K while
`linear_num_value_heads: 48` carries V, so Q/K are broadcast 3x to meet V.
(Qwen3.5-9B already runs this path at 2x and Qwen3.5-2B at 1x; only the ratio
differs here.)

Config keys Qwen3.8 adds that this graph deliberately ignores:
  - `output_gate_type: "swish"` -- inert. transformers v5.8.0 (the version the
    checkpoint declares) has no such field in Qwen3_5Config; the reference
    gates the DeltaNet norm with `config.hidden_act` ("silu", == swish) and the
    attention output with `sigmoid`, which is exactly what this graph does.
  - `mtp_num_hidden_layers: 1` -- the `mtp.*` speculative-decoding head is
    present in the checkpoint and is not part of the decoder graph.
  - vision tower (`model.visual.*`) and mrope image/video sections -- the
    text-only decoder path does not consume them.

Key architecture details:

  DeltaNet layers:
    - in_proj_qkv -> conv1d step -> SiLU -> split Q[nkv x dim], K[nkv x dim], V[nheads x dim]
    - L2-norm Q and K
    - keep compact Q,K from num_kv_heads -> num_heads
    - Delta rule state update: state [nheads, head_dim, head_dim]
    - Gated RMSNorm with separate gate projection (in_proj_z)
    - Decay: -exp(A_log) * softplus(in_proj_a(x) + dt_bias) per head
    - Beta (write strength): sigmoid(in_proj_b(x)) per head

  Full attention layers:
    - q_proj [2*attn_size, hidden] -> split query + gate
    - QK-norm with (1+weight) centering
    - Partial RoPE (partial_rotary_factor=0.25, 64 of 256 dims)
    - KV cache + scaled dot-product attention
    - Output gating: attn_out * sigmoid(gate)

Weight key mapping (HF -> engine), verified against Qwen/Qwen3.8-27B:
  model.language_model.embed_tokens.weight                        -> embedding
  model.language_model.layers.{i}.input_layernorm.weight          -> layer.{i}.input_norm
  --- DeltaNet (linear_attention) layers ---
  model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight  -> deltanet_in_proj_qkv
  model.language_model.layers.{i}.linear_attn.in_proj_z.weight    -> deltanet_z_proj (gate)
  model.language_model.layers.{i}.linear_attn.in_proj_a.weight    -> deltanet_a_proj (decay)
  model.language_model.layers.{i}.linear_attn.in_proj_b.weight    -> deltanet_b_proj (beta)
  model.language_model.layers.{i}.linear_attn.A_log               -> A
  model.language_model.layers.{i}.linear_attn.dt_bias             -> dt_bias
  model.language_model.layers.{i}.linear_attn.conv1d.weight/bias  -> conv1d
  model.language_model.layers.{i}.linear_attn.norm.weight         -> deltanet_norm
  model.language_model.layers.{i}.linear_attn.out_proj.weight     -> deltanet_out_proj
  --- Full attention layers ---
  model.language_model.layers.{i}.self_attn.q_proj.weight         -> split: w_q + w_gate_attn
  model.language_model.layers.{i}.self_attn.k_proj.weight         -> w_k (keep compacted)
  model.language_model.layers.{i}.self_attn.v_proj.weight         -> w_v (keep compacted)
  model.language_model.layers.{i}.self_attn.o_proj.weight         -> w_o
  model.language_model.layers.{i}.self_attn.q_norm.weight         -> q_norm ((1+w) tiled)
  model.language_model.layers.{i}.self_attn.k_norm.weight         -> k_norm ((1+w) tiled)
  --- SwiGLU MLP (both layer types) ---
  model.language_model.layers.{i}.mlp.gate_proj.weight            -> w_gate
  model.language_model.layers.{i}.mlp.up_proj.weight              -> w_up
  model.language_model.layers.{i}.mlp.down_proj.weight            -> w_down
  model.language_model.layers.{i}.post_attention_layernorm.weight -> post_attn_norm
  --- Final ---
  model.language_model.norm.weight                                -> final_norm
  lm_head.weight                                                  -> w_lm_head
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

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


def _parse_layer_types(raw_types: list[str]) -> list[str]:
    """Normalize layer type strings to 'deltanet' or 'attention'."""
    mapping = {
        "linear": "deltanet",
        "linear_attention": "deltanet",
        "full": "attention",
        "full_attention": "attention",
    }
    return [mapping.get(t.lower(), t.lower()) for t in raw_types]


def _prepare_runtime_inputs(
    network,
    work_trt_dtype,
    attention_mask,
    conv_state_inputs,
    ssm_state_inputs,
    cache_k_inputs,
    cache_v_inputs,
):
    """Cast storage tensors while preserving DeltaNet recurrence in FP32."""
    if work_trt_dtype == trt.float32:
        return (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        )

    def cast_all(tensors):
        return [
            network.add_cast(tensor, work_trt_dtype).get_output(0)
            for tensor in tensors
        ]

    return (
        network.add_cast(attention_mask, work_trt_dtype).get_output(0),
        cast_all(conv_state_inputs),
        # HF keeps the DeltaNet recurrent state in FP32. Never quantize this
        # persistent input before the per-token recurrence.
        ssm_state_inputs,
        cast_all(cache_k_inputs),
        cast_all(cache_v_inputs),
    )


def _owned(quant_ctx, name: str) -> bool:
    """True if quant_ctx will source this weight from the checkpoint's own
    packed bytes at graph-build time, so load_weights() must not materialize
    a dequantized copy that would never be read."""
    return quant_ctx is not None and quant_ctx.profile.should_quantize(name)


class Qwen38Model:
    def load_weights(
        self, model_dir: str, config: ModelConfig, *, precision: str = "fp32",
        quant_ctx=None, readers=None,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        if readers is None:
            readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        raw = config.raw

        # Text config may be nested under text_config
        text_cfg = raw.get("text_config", raw)

        # Parse layer types
        raw_layer_types = text_cfg.get("layer_types", ["linear"] * num_layers)
        layer_types = _parse_layer_types(raw_layer_types)
        assert len(layer_types) == num_layers, (
            f"layer_types length {len(layer_types)} != num_hidden_layers {num_layers}")

        # Full attention dimensions
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        attn_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        # DeltaNet dimensions (from text_config linear_* fields)
        deltanet_num_heads = text_cfg.get("linear_num_value_heads", 32)
        deltanet_num_kv_heads = text_cfg.get("linear_num_key_heads", 16)
        deltanet_head_dim = text_cfg.get("linear_value_head_dim",
                                         text_cfg.get("linear_key_head_dim", 128))
        d_inner = deltanet_num_heads * deltanet_head_dim
        deltanet_qk_dim = deltanet_num_kv_heads * deltanet_head_dim
        conv_dim = deltanet_qk_dim + deltanet_qk_dim + d_inner  # Q + K + V
        d_conv = text_cfg.get("linear_conv_kernel_dim", 4)

        # MLP dimensions
        mlp_size = config.intermediate_size

        # RoPE config for full attention layers
        # rope_parameters may be nested in text_config
        rope_params = text_cfg.get("rope_parameters", {})
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor",
            text_cfg.get("partial_rotary_factor", 0.25))
        rope_theta = rope_params.get(
            "rope_theta",
            text_cfg.get("rope_theta", config.rope_theta))

        weights = WeightDict()

        # Embedding
        embed_key = "model.language_model.embed_tokens.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.embed_tokens.weight"
        embedding = _load_tensor(readers, embed_key)
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        deltanet_count = 0
        attn_count = 0

        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx]
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.language_model.layers.{layer_idx}"

            # Input layernorm (all layer types)
            # Qwen3.8 uses (1+weight) centering in RMSNorm
            norm_key = f"{hf_prefix}.input_layernorm.weight"
            if _has_tensor(readers, norm_key):
                weights[f"{prefix}.input_norm"] = (
                    1.0 + _load_tensor(readers, norm_key).astype(np.float32))
            else:
                weights[f"{prefix}.input_norm"] = np.ones(
                    hidden, dtype=np.float32)

            # Post-attention layernorm (all layer types)
            post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
            if _has_tensor(readers, post_norm_key):
                weights[f"{prefix}.post_attn_norm"] = (
                    1.0 + _load_tensor(readers, post_norm_key).astype(np.float32))
            else:
                weights[f"{prefix}.post_attn_norm"] = np.ones(
                    hidden, dtype=np.float32)

            if lt == "deltanet":
                self._load_deltanet_weights(
                    readers, weights, prefix, hf_prefix,
                    hidden, d_inner, conv_dim, d_conv,
                    deltanet_num_heads, deltanet_num_kv_heads,
                    deltanet_head_dim, precision=precision, quant_ctx=quant_ctx)
                deltanet_count += 1

            elif lt == "attention":
                self._load_attention_weights(
                    readers, weights, prefix, hf_prefix,
                    hidden, attn_size, kv_size,
                    num_heads, num_kv_heads, head_dim, precision=precision,
                    quant_ctx=quant_ctx)
                attn_count += 1

            # SwiGLU MLP (all layer types)
            self._load_mlp_weights(
                readers, weights, prefix, hf_prefix,
                hidden, mlp_size, precision=precision, quant_ctx=quant_ctx)

        # Final norm (also uses (1+weight) centering)
        final_norm_key = "model.language_model.norm.weight"
        if not _has_tensor(readers, final_norm_key):
            final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = (
                1.0 + _load_tensor(readers, final_norm_key).astype(np.float32))
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        if not _owned(quant_ctx, "w_lm_head"):
            lm_head_key = "lm_head.weight"
            if _has_tensor(readers, lm_head_key):
                weights["w_lm_head"] = _transpose_2d(
                    _load_tensor(readers, lm_head_key), "lm_head", precision)
            else:
                weights["w_lm_head"] = _transpose_2d(
                    embedding, "embedding_tied", precision)

        # MTP (multi-token-prediction) draft head: optional, top-level
        # `mtp.*` keys (not under `model.language_model.`). Not every
        # checkpoint ships this -- only load it when present.
        if _has_tensor(readers, "mtp.fc.weight"):
            self._load_mtp_weights(
                readers, weights, hidden, attn_size,
                num_heads, num_kv_heads, head_dim, precision=precision)

        # Metadata for engine builder
        weights["_layer_types"] = layer_types
        weights["_d_inner"] = d_inner
        weights["_d_conv"] = d_conv
        weights["_conv_dim"] = conv_dim
        weights["_deltanet_num_heads"] = deltanet_num_heads
        weights["_deltanet_num_kv_heads"] = deltanet_num_kv_heads
        weights["_deltanet_head_dim"] = deltanet_head_dim
        weights["_num_mamba_layers"] = deltanet_count
        weights["_num_attention_layers"] = attn_count
        weights["_attn_size"] = attn_size
        weights["_mlp_size"] = mlp_size
        weights["_partial_rotary_factor"] = partial_rotary_factor
        weights["_rope_theta"] = rope_theta

        return weights

    def _load_deltanet_weights(
        self, readers, weights, prefix, hf_prefix,
        hidden, d_inner, conv_dim, d_conv,
        num_heads, num_kv_heads, head_dim, *, precision: str = "fp32",
        quant_ctx=None,
    ):
        """Load DeltaNet (linear attention) layer weights."""
        attn_prefix = f"{hf_prefix}.linear_attn"

        # in_proj_qkv (QKV combined): [conv_dim, hidden] -> transpose.
        # Skip if quant_ctx already owns this weight -- it will be sourced
        # from the checkpoint's own packed bytes at graph-build time, so a
        # dequantized copy here would never be read.
        if not _owned(quant_ctx, f"{prefix}.deltanet_in_proj_qkv"):
            in_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_qkv.weight")
            weights[f"{prefix}.deltanet_in_proj_qkv"] = _transpose_2d(
                in_proj_raw, "deltanet_in_proj_qkv", precision)

        # Gate projection (z): [d_inner, hidden] -> transpose
        if not _owned(quant_ctx, f"{prefix}.deltanet_z_proj"):
            z_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_z.weight")
            weights[f"{prefix}.deltanet_z_proj"] = _transpose_2d(
                z_proj_raw, "deltanet_z_proj", precision)

        # Decay projection (a): [num_heads, hidden] -> transpose
        a_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_a.weight")
        weights[f"{prefix}.deltanet_a_proj"] = _transpose_2d(
            a_proj_raw, "deltanet_a_proj", precision)

        # Beta projection (b): [num_heads, hidden] -> transpose
        b_proj_raw = _load_tensor(readers, f"{attn_prefix}.in_proj_b.weight")
        weights[f"{prefix}.deltanet_b_proj"] = _transpose_2d(
            b_proj_raw, "deltanet_b_proj", precision)

        # A_log: [num_heads] -> precompute -exp(A_log)
        A_log = _load_tensor(readers, f"{attn_prefix}.A_log")
        weights[f"{prefix}.A"] = -np.exp(A_log.astype(np.float32))

        # dt_bias: [num_heads]
        dt_bias = _load_tensor(readers, f"{attn_prefix}.dt_bias")
        weights[f"{prefix}.dt_bias"] = dt_bias.astype(np.float32)

        # conv1d: [conv_dim, 1, d_conv] -> reshape to [conv_dim, d_conv]
        conv_w = _load_tensor(readers, f"{attn_prefix}.conv1d.weight")
        weights[f"{prefix}.conv1d_weight"] = conv_w.reshape(
            conv_dim, d_conv).astype(np.float32)

        conv_b_key = f"{attn_prefix}.conv1d.bias"
        if _has_tensor(readers, conv_b_key):
            weights[f"{prefix}.conv1d_bias"] = _load_tensor(
                readers, conv_b_key).astype(np.float32)
        else:
            weights[f"{prefix}.conv1d_bias"] = np.zeros(
                conv_dim, dtype=np.float32)

        # Gated RMSNorm weight: [head_dim] -> tile to [d_inner]
        norm_key = f"{attn_prefix}.norm.weight"
        if _has_tensor(readers, norm_key):
            norm_raw = _load_tensor(readers, norm_key).astype(np.float32)
            # If weight is per-head (head_dim), tile to d_inner
            if norm_raw.shape[0] == head_dim and head_dim < d_inner:
                norm_raw = np.tile(norm_raw, num_heads)
            weights[f"{prefix}.deltanet_norm"] = norm_raw
        else:
            weights[f"{prefix}.deltanet_norm"] = np.ones(
                d_inner, dtype=np.float32)

        # Output projection: [hidden, d_inner] -> transpose
        if not _owned(quant_ctx, f"{prefix}.deltanet_out_proj"):
            out_raw = _load_tensor(readers, f"{attn_prefix}.out_proj.weight")
            weights[f"{prefix}.deltanet_out_proj"] = _transpose_2d(
                out_raw, "deltanet_out_proj", precision)

    def _load_attention_weights(
        self, readers, weights, prefix, hf_prefix,
        hidden, attn_size, kv_size,
        num_heads, num_kv_heads, head_dim, *, precision: str = "fp32",
        quant_ctx=None,
    ):
        """Load full self-attention layer weights."""
        attn_prefix = f"{hf_prefix}.self_attn"

        # q_proj: [2*attn_size, hidden] -> split per head into query + gate
        # HF does: q_proj(x).view(B, seq, num_heads, 2*head_dim).chunk(2, dim=-1)
        # This interleaves: for each head, first head_dim dims are query, next are gate
        # calibrate_qwen3_8_nvfp4 always registers w_q and w_gate_attn together
        # (same checkpoint tensor, same input_scale), so checking w_q alone
        # is sufficient to skip both.
        if not _owned(quant_ctx, f"{prefix}.w_q"):
            q_raw = _load_tensor(readers, f"{attn_prefix}.q_proj.weight")
            # q_raw: [num_heads * 2 * head_dim, hidden] = [8192, 4096]
            # Reshape to [num_heads, 2*head_dim, hidden], split, reshape back
            q_reshaped = q_raw.reshape(num_heads, 2 * head_dim, hidden)
            q_part = q_reshaped[:, :head_dim, :].reshape(attn_size, hidden)
            gate_part = q_reshaped[:, head_dim:, :].reshape(attn_size, hidden)
            weights[f"{prefix}.w_q"] = _transpose_2d(q_part, "q_proj", precision)
            weights[f"{prefix}.w_gate_attn"] = _transpose_2d(gate_part, "gate_proj", precision)

        # k_proj: [kv_size, hidden] -> keep compact
        if not _owned(quant_ctx, f"{prefix}.w_k"):
            k_raw = _load_tensor(readers, f"{attn_prefix}.k_proj.weight")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj", precision)

        # v_proj: [kv_size, hidden] -> keep compact
        if not _owned(quant_ctx, f"{prefix}.w_v"):
            v_raw = _load_tensor(readers, f"{attn_prefix}.v_proj.weight")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj", precision)

        # o_proj: [hidden, attn_size] -> transpose
        if not _owned(quant_ctx, f"{prefix}.w_o"):
            o_raw = _load_tensor(readers, f"{attn_prefix}.o_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj", precision)

        # QK-norm with (1+weight) centering, tiled to num_heads
        q_norm_key = f"{attn_prefix}.q_norm.weight"
        if _has_tensor(readers, q_norm_key):
            q_norm_raw = _load_tensor(readers, q_norm_key).astype(np.float32)
            q_norm_centered = 1.0 + q_norm_raw  # (1+weight) centering
            weights[f"{prefix}.q_norm"] = np.tile(
                q_norm_centered, num_heads)
        k_norm_key = f"{attn_prefix}.k_norm.weight"
        if _has_tensor(readers, k_norm_key):
            k_norm_raw = _load_tensor(readers, k_norm_key).astype(np.float32)
            k_norm_centered = 1.0 + k_norm_raw
            weights[f"{prefix}.k_norm"] = np.tile(
                k_norm_centered, num_kv_heads)

    def _load_mlp_weights(
        self, readers, weights, prefix, hf_prefix,
        hidden, mlp_size, *, precision: str = "fp32",
        quant_ctx=None,
    ):
        """Load SwiGLU MLP weights."""
        gate_key = f"{hf_prefix}.mlp.gate_proj.weight"
        up_key = f"{hf_prefix}.mlp.up_proj.weight"
        down_key = f"{hf_prefix}.mlp.down_proj.weight"

        if _has_tensor(readers, gate_key):
            if not _owned(quant_ctx, f"{prefix}.w_gate"):
                weights[f"{prefix}.w_gate"] = _transpose_2d(
                    _load_tensor(readers, gate_key), "gate_proj", precision)
            if not _owned(quant_ctx, f"{prefix}.w_up"):
                weights[f"{prefix}.w_up"] = _transpose_2d(
                    _load_tensor(readers, up_key), "up_proj", precision)
            if not _owned(quant_ctx, f"{prefix}.w_down"):
                weights[f"{prefix}.w_down"] = _transpose_2d(
                    _load_tensor(readers, down_key), "down_proj", precision)

    def _load_mtp_weights(
        self, readers, weights,
        hidden, attn_size, num_heads, num_kv_heads, head_dim,
        *, precision: str = "fp32",
    ):
        """Load the MTP (multi-token-prediction) draft-head weights.

        Checkpoint keys live at a top-level `mtp.*` prefix (not under
        `model.language_model.`), and are plain bf16 -- never NVFP4/FP8
        packed in any published checkpoint seen so far (no `.weight_scale`
        companion tensors) -- so this always uses the plain `_load_tensor`
        path, no `quant_ctx`/`_owned()` skip logic. Stored under the
        `mtp_layer.*` prefix in `weights` so it can be fed straight into
        `_add_full_attention_layer` (same shapes/roles as a normal
        full-attention layer's weights) plus `mtp_layer.fc`,
        `mtp_layer.pre_fc_norm_embedding`, `mtp_layer.pre_fc_norm_hidden`,
        and the shared `mtp_final_norm`.
        """
        prefix = "mtp_layer"

        fc_raw = _load_tensor(readers, "mtp.fc.weight")
        weights[f"{prefix}.fc"] = _transpose_2d(fc_raw, "mtp.fc", precision)

        # (1+weight) centering, same convention as every other norm in this
        # family (see input_norm/final_norm above).
        weights[f"{prefix}.pre_fc_norm_embedding"] = (
            1.0 + _load_tensor(
                readers, "mtp.pre_fc_norm_embedding.weight").astype(np.float32))
        weights[f"{prefix}.pre_fc_norm_hidden"] = (
            1.0 + _load_tensor(
                readers, "mtp.pre_fc_norm_hidden.weight").astype(np.float32))

        layer_prefix = "mtp.layers.0"
        weights[f"{prefix}.input_norm"] = (
            1.0 + _load_tensor(
                readers, f"{layer_prefix}.input_layernorm.weight"
            ).astype(np.float32))
        weights[f"{prefix}.post_attn_norm"] = (
            1.0 + _load_tensor(
                readers, f"{layer_prefix}.post_attention_layernorm.weight"
            ).astype(np.float32))

        attn_prefix = f"{layer_prefix}.self_attn"
        q_raw = _load_tensor(readers, f"{attn_prefix}.q_proj.weight")
        q_reshaped = q_raw.reshape(num_heads, 2 * head_dim, hidden)
        q_part = q_reshaped[:, :head_dim, :].reshape(attn_size, hidden)
        gate_part = q_reshaped[:, head_dim:, :].reshape(attn_size, hidden)
        weights[f"{prefix}.w_q"] = _transpose_2d(q_part, "mtp.q_proj", precision)
        weights[f"{prefix}.w_gate_attn"] = _transpose_2d(
            gate_part, "mtp.gate_proj", precision)

        weights[f"{prefix}.w_k"] = _transpose_2d(
            _load_tensor(readers, f"{attn_prefix}.k_proj.weight"),
            "mtp.k_proj", precision)
        weights[f"{prefix}.w_v"] = _transpose_2d(
            _load_tensor(readers, f"{attn_prefix}.v_proj.weight"),
            "mtp.v_proj", precision)
        weights[f"{prefix}.w_o"] = _transpose_2d(
            _load_tensor(readers, f"{attn_prefix}.o_proj.weight"),
            "mtp.o_proj", precision)

        q_norm_raw = _load_tensor(
            readers, f"{attn_prefix}.q_norm.weight").astype(np.float32)
        weights[f"{prefix}.q_norm"] = np.tile(1.0 + q_norm_raw, num_heads)
        k_norm_raw = _load_tensor(
            readers, f"{attn_prefix}.k_norm.weight").astype(np.float32)
        weights[f"{prefix}.k_norm"] = np.tile(1.0 + k_norm_raw, num_kv_heads)

        mlp_prefix = f"{layer_prefix}.mlp"
        weights[f"{prefix}.w_gate"] = _transpose_2d(
            _load_tensor(readers, f"{mlp_prefix}.gate_proj.weight"),
            "mtp.mlp.gate_proj", precision)
        weights[f"{prefix}.w_up"] = _transpose_2d(
            _load_tensor(readers, f"{mlp_prefix}.up_proj.weight"),
            "mtp.mlp.up_proj", precision)
        weights[f"{prefix}.w_down"] = _transpose_2d(
            _load_tensor(readers, f"{mlp_prefix}.down_proj.weight"),
            "mtp.mlp.down_proj", precision)

        weights["mtp_final_norm"] = (
            1.0 + _load_tensor(readers, "mtp.norm.weight").astype(np.float32))

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        """Build hybrid TRT engine with DeltaNet + attention layers."""
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        deltanet_num_heads: int = weights["_deltanet_num_heads"]
        deltanet_num_kv_heads: int = weights["_deltanet_num_kv_heads"]
        deltanet_head_dim: int = weights["_deltanet_head_dim"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "bf16":
            # Constants are staged as FP16 bytes (TensorRT's Weights constructor
            # does not accept ml_dtypes.bfloat16 arrays directly) and explicitly
            # cast to BF16 in-graph by graph_ops._cast_back_to_trt_dtype, which
            # every constant-building helper already calls to match its
            # activation's runtime dtype -- mirroring families/qwen's own
            # "storage np_dtype is fp16, runtime trt_dtype is bfloat16" pattern.
            work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Qwen3.8 precision {precision!r}; expected fp32, fp16, or bf16")
        requested_fp32_layers = frozenset(
            int(layer) for layer in config.raw.get("_fp32_layers", ()))
        invalid_fp32_layers = sorted(
            layer for layer in requested_fp32_layers
            if layer < 0 or layer >= num_layers)
        if invalid_fp32_layers:
            raise ValueError(
                "fp32_layers contains out-of-range indices: "
                f"{invalid_fp32_layers}")

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim,
            quant_ctx=quant_ctx)
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if quant_ctx is not None and getattr(quant_ctx, "disable_dual_gemm_fusion", False):
            # Works around a TRT dual-GEMM fusion bug with plain scalar-scale
            # FP8 (see quantization.py::calibrate_qwen3_8_fp8's docstring).
            trt_config.build_route = "-peep:match_dual_gemm=off"

        # --- Inputs ---
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        # DeltaNet state inputs (conv + ssm per DeltaNet layer)
        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi),
                trt.float32, (conv_dim, d_conv))
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32, (deltanet_num_heads, deltanet_head_dim, deltanet_head_dim))
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        # Attention KV cache inputs
        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                work_trt_dtype, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                work_trt_dtype, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        ) = _prepare_runtime_inputs(
            network,
            work_trt_dtype,
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        )

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"],
            dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)

        rope_theta: float = weights["_rope_theta"]
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)

        # RoPE tables for full attention layers (partial rotary)
        cos_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)

        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype)

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
            layer_is_fp32 = (
                precision in ("fp16", "bf16") and layer_idx in requested_fp32_layers)
            layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
            layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

            def layer_cast(tensor):
                if tensor.dtype == layer_trt_dtype:
                    return tensor
                return network.add_cast(
                    tensor, layer_trt_dtype).get_output(0)

            if lt == "deltanet":
                result = _add_deltanet_layer(
                    network=network,
                    hidden=layer_cast(hidden_state),
                    conv_state_in=layer_cast(
                        conv_state_inputs[mamba_counter]),
                    # Transformers casts the DeltaNet recurrence and its
                    # persistent state to FP32 even for FP16 checkpoints.
                    ssm_state_in=ssm_state_inputs[mamba_counter],
                    eps_tensor=layer_cast(eps_tensor),
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    d_inner=d_inner,
                    d_conv=d_conv,
                    conv_dim=conv_dim,
                    num_heads=deltanet_num_heads,
                    num_kv_heads=deltanet_num_kv_heads,
                    head_dim=deltanet_head_dim,
                    mlp_size=mlp_size,
                    dtype=layer_np_dtype,
                    quant_ctx=quant_ctx,
                )
                hidden_state = result["hidden"]
                present_conv_outputs.append(result["present_conv"])
                present_ssm_outputs.append(result["present_ssm"])
                mamba_counter += 1

            elif lt == "attention":
                result = _add_full_attention_layer(
                    network=network,
                    hidden=layer_cast(hidden_state),
                    cache_k=layer_cast(cache_k_inputs[attn_counter]),
                    cache_v=layer_cast(cache_v_inputs[attn_counter]),
                    attention_mask=layer_cast(attention_mask),
                    position_id=position_id,
                    cos_half_tensor=layer_cast(cos_half_tensor),
                    sin_half_tensor=layer_cast(sin_half_tensor),
                    eps_tensor=layer_cast(eps_tensor),
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    attn_size=attn_size,
                    kv_attention_size=kv_attention_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rotary_embedding_dim=rotary_embedding_dim,
                    max_cache_length=max_cache_length,
                    mlp_size=mlp_size,
                    dtype=layer_np_dtype,
                    quant_ctx=quant_ctx,
                )
                hidden_state = result["hidden"]
                present_k_outputs.append(result["present_k"])
                present_v_outputs.append(result["present_v"])
                attn_counter += 1

            if debug_layer_outputs:
                _mark_debug_output(
                    network, hidden_state, f"debug_hidden_{layer_idx}")

        # --- Final norm ---
        if hidden_state.dtype != work_trt_dtype:
            hidden_state = network.add_cast(
                hidden_state, work_trt_dtype).get_output(0)
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_ops.add_rms_norm(
                network, hidden_state, hidden, final_norm, eps_tensor,
                dtype=work_np_dtype)

        # --- MTP hidden-state tap ---
        # Exposed unconditionally (cheap: one identity+cast) so a paired MTP
        # engine (build_mtp_engine) can consume it -- build_engine itself
        # doesn't know whether model.py will build one.
        hidden_state_out = hidden_state
        if hidden_state_out.dtype != trt.float32:
            hidden_state_out = network.add_cast(
                hidden_state_out, trt.float32).get_output(0)
        hidden_state_out = network.add_identity(hidden_state_out).get_output(0)
        hidden_state_out.name = "hidden_state"
        network.mark_output(hidden_state_out)

        # --- LM head ---
        lm_head_matmul = graph_blocks.make_matmul_fn(network, work_np_dtype, quant_ctx)
        logits = lm_head_matmul(
            hidden_state, hidden, vocab, weights.get("w_lm_head"), "w_lm_head")
        b_out = np.zeros(vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, vocab, b_out, dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        # --- Present state outputs ---
        for mi in range(num_mamba):
            pc = present_conv_outputs[mi]
            ps = present_ssm_outputs[mi]
            if pc.dtype != trt.float32:
                pc = network.add_cast(pc, trt.float32).get_output(0)
            if ps.dtype != trt.float32:
                ps = network.add_cast(ps, trt.float32).get_output(0)
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            pk = present_k_outputs[ai]
            pv = present_v_outputs[ai]
            if pk.dtype != work_trt_dtype:
                pk = network.add_cast(pk, work_trt_dtype).get_output(0)
            if pv.dtype != work_trt_dtype:
                pv = network.add_cast(pv, work_trt_dtype).get_output(0)
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        # --- Build ---
        if verbose:
            print(f"[trtmc build] Building Qwen3.8 hybrid TRT engine "
                  f"({num_layers} layers: {num_mamba} deltanet + "
                  f"{num_attn} attention, "
                  f"hidden={hidden}, d_inner={d_inner}, "
                  f"nheads_dn={deltanet_num_heads}, "
                  f"head_dim_dn={deltanet_head_dim}, "
                  f"cache={max_cache_length}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)

    def build_mtp_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        """Build the MTP (multi-token-prediction) draft-head TRT engine.

        Mirrors vLLM's Qwen3-Next MTP forward pass (`qwen3_next_mtp.py`):
            embeds = pre_fc_norm_embedding(embed_tokens(next_token_id))
            hs     = pre_fc_norm_hidden(hidden_state)   # main engine's final hidden state at pos n
            fused  = fc(cat([embeds, hs], dim=-1))      # embeds first, hidden second
            fused  = one ordinary decoder layer (mtp.layers.0) -- real
                     self-attention with its OWN KV cache, not stateless
            logits = lm_head(norm(fused))                # draft token n+2

        `quant_ctx` should be the SAME context passed to the main engine's
        `build_engine()` (or None for an unquantized build). MTP's own
        `mtp_layer.*` weights are never registered in it (see
        `_load_mtp_weights` -- unquantized in every checkpoint seen so far,
        `quant_ctx.maybe_quantized_matmul` falls back to a plain constant
        matmul for any unregistered name), so passing it through is a no-op
        for the decoder layer. It matters for `lm_head`: when the main
        model is NVFP4-quantized, `w_lm_head` is *shared* with the main
        engine and is NVFP4-owned too -- `load_weights()` never
        materializes a plain copy for it (`_owned(quant_ctx, "w_lm_head")`),
        so building this engine with `quant_ctx=None` regardless of the
        main build's quantization would read an empty weight for `lm_head`.

        Caller contract: `next_token_id` is the token the main engine just
        produced at position n (i.e. token n+1); `position_id` must be the
        position that token would use on a normal decode step (n+1);
        `mtp_cache_k`/`mtp_cache_v` are this engine's OWN persistent cache,
        separate from the main engine's per-layer caches, fed back from
        `mtp_present_k`/`mtp_present_v` every step (not skippable -- MTP's
        attention history desyncs from the main model's token stream
        otherwise).
        """
        if "mtp_layer.fc" not in weights:
            raise ValueError(
                "weights has no MTP head -- checkpoint does not ship mtp.* tensors")

        hidden = config.hidden_size
        vocab = config.vocab_size
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        rope_theta: float = weights["_rope_theta"]

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = num_kv_heads * head_dim
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)
        attention_window = max_cache_length + 1

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "bf16":
            # Constants are staged as FP16 bytes (TensorRT's Weights constructor
            # does not accept ml_dtypes.bfloat16 arrays directly) and explicitly
            # cast to BF16 in-graph by graph_ops._cast_back_to_trt_dtype, which
            # every constant-building helper already calls to match its
            # activation's runtime dtype -- mirroring families/qwen's own
            # "storage np_dtype is fp16, runtime trt_dtype is bfloat16" pattern.
            work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Qwen3.8 precision {precision!r}; expected fp32, fp16, or bf16")

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # --- Inputs ---
        next_token_id = network.add_input("next_token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        hidden_state_in = network.add_input(
            "hidden_state", trt.float32, (1, hidden))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))
        cache_k = network.add_input(
            "mtp_cache_k", work_trt_dtype, (max_cache_length, kv_attention_size))
        cache_v = network.add_input(
            "mtp_cache_v", work_trt_dtype, (max_cache_length, kv_attention_size))

        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(
                attention_mask, work_trt_dtype).get_output(0)
            hidden_state_in = network.add_cast(
                hidden_state_in, work_trt_dtype).get_output(0)

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)
        cos_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype)

        # --- inputs_embeds = pre_fc_norm_embedding(embed_tokens(next_token_id)) ---
        gather = network.add_gather(embedding_table, next_token_id, 0)
        gather_out = gather.get_output(0)
        if work_trt_dtype != trt.float32:
            # embedding_table is stored at work_np_dtype (fp16 even for bf16
            # builds -- see the precision dispatch above), so the gather
            # output is still Half here. Cast up to work_trt_dtype before
            # apply_norm, matching hidden_state_in's cast above -- apply_norm
            # preserves its input's actual runtime dtype through to output,
            # so leaving this uncast made inputs_embeds come out Half while
            # hs (built from the already-cast hidden_state_in) came out
            # BFloat16, and the concat below rejected the mixed types.
            gather_out = network.add_cast(gather_out, work_trt_dtype).get_output(0)
        inputs_embeds = graph_blocks.apply_norm(
            network, gather_out, hidden,
            weights["mtp_layer.pre_fc_norm_embedding"], None,
            eps_tensor, "rmsnorm", dtype=work_np_dtype)

        # --- hidden_states = pre_fc_norm_hidden(hidden_state) ---
        hs = graph_blocks.apply_norm(
            network, hidden_state_in, hidden,
            weights["mtp_layer.pre_fc_norm_hidden"], None,
            eps_tensor, "rmsnorm", dtype=work_np_dtype)

        # --- fused = fc(cat([inputs_embeds, hidden_states])) ---
        fused_cat = network.add_concatenation([inputs_embeds, hs])
        fused_cat.axis = 1
        matmul = graph_blocks.make_matmul_fn(network, work_np_dtype, quant_ctx)
        fused = matmul(
            fused_cat.get_output(0), 2 * hidden, hidden,
            weights["mtp_layer.fc"], "mtp_layer.fc")

        # --- one ordinary decoder layer: real self-attention + its own KV cache ---
        # Mirrors build_engine's per-layer `layer_cast`: every tensor entering
        # _add_full_attention_layer must match work_trt_dtype exactly (TensorRT
        # requires RotaryEmbedding's cosCache/sinCache and IAttention's inputs
        # to share one runtime dtype), and cos/sin/eps constants are staged at
        # work_np_dtype (fp16 storage even for bf16 builds) so they still need
        # an explicit cast for bf16 precision.
        def _layer_cast(tensor):
            if tensor.dtype == work_trt_dtype:
                return tensor
            return network.add_cast(tensor, work_trt_dtype).get_output(0)

        result = _add_full_attention_layer(
            network=network,
            hidden=_layer_cast(fused),
            cache_k=_layer_cast(cache_k),
            cache_v=_layer_cast(cache_v),
            attention_mask=_layer_cast(attention_mask),
            position_id=position_id,
            cos_half_tensor=_layer_cast(cos_half_tensor),
            sin_half_tensor=_layer_cast(sin_half_tensor),
            eps_tensor=_layer_cast(eps_tensor),
            weights=weights,
            prefix="mtp_layer",
            hidden_size=hidden,
            attn_size=attn_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_embedding_dim=rotary_embedding_dim,
            max_cache_length=max_cache_length,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
        )
        mtp_hidden = result["hidden"]
        present_k = result["present_k"]
        present_v = result["present_v"]

        # Debug/verification output: the pre-final-norm hidden state, used
        # to cross-check build_mtp_draft_chain_engine's in-graph self-
        # chaining against manually self-chained calls to this (already
        # proven) single-step engine as an independent reference.
        mtp_hidden_state_out = mtp_hidden
        if mtp_hidden_state_out.dtype != trt.float32:
            mtp_hidden_state_out = network.add_cast(
                mtp_hidden_state_out, trt.float32).get_output(0)
        mtp_hidden_state_out.name = "mtp_hidden_state"
        network.mark_output(mtp_hidden_state_out)

        # --- norm -> lm_head ---
        if mtp_hidden.dtype != work_trt_dtype:
            mtp_hidden = network.add_cast(mtp_hidden, work_trt_dtype).get_output(0)
        mtp_hidden = graph_ops.add_rms_norm(
            network, mtp_hidden, hidden, weights["mtp_final_norm"], eps_tensor,
            dtype=work_np_dtype)

        lm_head_matmul = graph_blocks.make_matmul_fn(network, work_np_dtype, quant_ctx)
        mtp_logits = lm_head_matmul(
            mtp_hidden, hidden, vocab, weights.get("w_lm_head"), "w_lm_head")
        if mtp_logits.dtype != trt.float32:
            mtp_logits = network.add_cast(mtp_logits, trt.float32).get_output(0)
        mtp_logits.name = "mtp_logits"
        network.mark_output(mtp_logits)

        if present_k.dtype != work_trt_dtype:
            present_k = network.add_cast(present_k, work_trt_dtype).get_output(0)
        if present_v.dtype != work_trt_dtype:
            present_v = network.add_cast(present_v, work_trt_dtype).get_output(0)
        present_k.name = "mtp_present_k"
        present_v.name = "mtp_present_v"
        network.mark_output(present_k)
        network.mark_output(present_v)

        if verbose:
            print(f"[trtmc build] Building Qwen3.8 MTP draft-head TRT engine "
                  f"(hidden={hidden}, attn_size={attn_size}, mlp={mlp_size}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT MTP engine build failed")

        return bytes(plan)

    def build_mtp_draft_chain_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, num_draft_tokens: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        """Build an MTP engine that drafts `num_draft_tokens` tokens in one
        call, by repeating the single trained mtp.layers.0 block
        `num_draft_tokens` times against the SAME weights -- no extra
        trained parameters. This is the standard technique mainstream
        frameworks (vLLM/SGLang) use for single-layer MTP/EAGLE drafting
        beyond depth 1: repeat t's own hidden state and in-graph-argmax
        token id feed repeat t+1, entirely inside one engine, with zero
        host round-trips between repeats (repeat 0's real-hidden-state
        input is identical to build_mtp_engine's whole body -- this
        function generalizes it to num_draft_tokens>=1 repeats).

        Caller contract for inputs: identical to build_mtp_engine's single
        step (next_token_id/position_id/hidden_state are the REAL,
        already-confirmed anchor; mtp_cache_k/v are MTP's persistent cache
        before this call).

        Outputs are per-repeat stacked tensors, row t = repeat t's result:
          mtp_draft_token_ids:     (num_draft_tokens,) int32 -- in-graph
            argmax, directly usable as (part of) the verify engine's
            token_ids input, no host argmax needed.
          mtp_draft_hidden_states: (num_draft_tokens, hidden) fp32 -- MTP's
            own hidden state per repeat, needed to seed the NEXT round's
            draft-chain call after a reject/partial-accept resync.
          mtp_draft_logits:        (num_draft_tokens, vocab) fp32 -- mainly
            diagnostic; the in-graph argmax already drives the chain.
          mtp_present_k/v:         (num_draft_tokens, kv_dim) -- MTP's own
            new cache rows. SPECULATIVE beyond whatever prefix length
            verification eventually confirms -- caller commits only the
            accepted-prefix rows into MTP's persistent cache, discards
            the rest (same partial-prefix-commit requirement as the main
            model's Qwen38HybridState already has for N>1 draft depth).
        """
        if "mtp_layer.fc" not in weights:
            raise ValueError(
                "weights has no MTP head -- checkpoint does not ship mtp.* tensors")
        if num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")

        hidden = config.hidden_size
        vocab = config.vocab_size
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        rope_theta: float = weights["_rope_theta"]

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = num_kv_heads * head_dim
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)
        attention_window = max_cache_length + 1

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "bf16":
            work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Qwen3.8 precision {precision!r}; expected fp32, fp16, or bf16")

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # --- Inputs (identical to build_mtp_engine's single-step inputs) ---
        next_token_id = network.add_input("next_token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        hidden_state_in = network.add_input("hidden_state", trt.float32, (1, hidden))
        attention_mask_in = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))
        cache_k_in = network.add_input(
            "mtp_cache_k", work_trt_dtype, (max_cache_length, kv_attention_size))
        cache_v_in = network.add_input(
            "mtp_cache_v", work_trt_dtype, (max_cache_length, kv_attention_size))

        def _layer_cast(tensor):
            if tensor.dtype == work_trt_dtype:
                return tensor
            return network.add_cast(tensor, work_trt_dtype).get_output(0)

        attention_mask = _layer_cast(attention_mask_in)
        hidden_state_in = _layer_cast(hidden_state_in)

        # --- Shared constants. RoPE table sized for the worst case: the
        # last repeat's furthest position. ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(
            network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)
        table_len = attention_window + num_draft_tokens
        cos_half = graph_ops.make_rope_table_half_dim(
            table_len, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            table_len, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)
        cos_half_tensor = _layer_cast(graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype))
        sin_half_tensor = _layer_cast(graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype))
        eps_tensor = _layer_cast(eps_tensor)

        matmul = graph_blocks.make_matmul_fn(network, work_np_dtype, quant_ctx)

        running_cache_k = _layer_cast(cache_k_in)
        running_cache_v = _layer_cast(cache_v_in)
        cur_token_id = next_token_id       # (1,) int32
        cur_hidden_in = hidden_state_in    # (1, hidden); real for repeat 0, self-chained after

        draft_token_ids = []
        draft_hidden_states = []
        draft_logits = []
        draft_present_k = []
        draft_present_v = []

        for t in range(num_draft_tokens):
            # --- inputs_embeds = pre_fc_norm_embedding(embed_tokens(cur_token_id)) ---
            gather = network.add_gather(embedding_table, cur_token_id, 0)
            gather_out = _layer_cast(gather.get_output(0))
            inputs_embeds = graph_blocks.apply_norm(
                network, gather_out, hidden,
                weights["mtp_layer.pre_fc_norm_embedding"], None,
                eps_tensor, "rmsnorm", dtype=work_np_dtype)

            # --- hidden_states = pre_fc_norm_hidden(cur_hidden_in) ---
            hs = graph_blocks.apply_norm(
                network, _layer_cast(cur_hidden_in), hidden,
                weights["mtp_layer.pre_fc_norm_hidden"], None,
                eps_tensor, "rmsnorm", dtype=work_np_dtype)

            # --- fused = fc(cat([inputs_embeds, hs])) ---
            fused_cat = network.add_concatenation([inputs_embeds, hs])
            fused_cat.axis = 1
            fused = matmul(
                fused_cat.get_output(0), 2 * hidden, hidden,
                weights["mtp_layer.fc"], "mtp_layer.fc")

            # --- this repeat's absolute RoPE position = position_id + t ---
            if t == 0:
                step_position_id = position_id
            else:
                t_const = graph_ops.add_constant(
                    network, (1,), np.array([t], dtype=np.int32), dtype=np.int32)
                step_pos = network.add_elementwise(
                    position_id, t_const, trt.ElementWiseOperation.SUM)
                step_position_id = step_pos.get_output(0)

            # --- this repeat's mask: attention_mask (the input) already
            # has width max_cache_length+1 -- it bakes in the self-attend
            # column for repeat 0, unlike build_engine_multi_token's base
            # mask (width max_cache_length, no self-attend column). Each
            # later repeat t needs exactly t MORE always-valid columns, one
            # per token generated by repeats 0..t-1 (repeat t's own
            # self-attend is already the "+1" baked into attention_mask).
            # Rebuilt fresh from the fixed persistent prefix each repeat,
            # not an incrementally-grown running mask -- same pattern as
            # build_engine_multi_token's per-substep zeros_t extension. ---
            if t == 0:
                step_mask = attention_mask
            else:
                zeros_t = graph_ops.add_constant(
                    network, (1, t), np.zeros((1, t), dtype=work_np_dtype),
                    dtype=work_np_dtype)
                zeros_t = _layer_cast(zeros_t)
                mask_concat = network.add_concatenation([attention_mask, zeros_t])
                mask_concat.axis = 1
                step_mask = mask_concat.get_output(0)

            result = _add_full_attention_layer(
                network=network,
                hidden=_layer_cast(fused),
                cache_k=running_cache_k,
                cache_v=running_cache_v,
                attention_mask=step_mask,
                position_id=step_position_id,
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                eps_tensor=eps_tensor,
                weights=weights,
                prefix="mtp_layer",
                hidden_size=hidden,
                attn_size=attn_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_embedding_dim=rotary_embedding_dim,
                max_cache_length=max_cache_length + t,
                mlp_size=mlp_size,
                dtype=work_np_dtype,
                quant_ctx=quant_ctx,
            )
            step_hidden = result["hidden"]
            new_k = _layer_cast(result["present_k"])
            new_v = _layer_cast(result["present_v"])

            # --- norm -> lm_head for this repeat's logits ---
            normed_out = _layer_cast(step_hidden)
            normed_out = graph_ops.add_rms_norm(
                network, normed_out, hidden, weights["mtp_final_norm"], eps_tensor,
                dtype=work_np_dtype)
            lm_head_matmul = graph_blocks.make_matmul_fn(network, work_np_dtype, quant_ctx)
            logits_t = lm_head_matmul(
                normed_out, hidden, vocab, weights.get("w_lm_head"), "w_lm_head")
            if logits_t.dtype != trt.float32:
                logits_t = network.add_cast(logits_t, trt.float32).get_output(0)
            draft_logits.append(logits_t)

            # --- in-graph argmax: this repeat's draft token ---
            topk = network.add_topk(logits_t, trt.TopKOperation.MAX, 1, 1 << 1)
            idx_reshape = network.add_shuffle(topk.get_output(1))
            idx_reshape.reshape_dims = (1,)
            token_id_t = idx_reshape.get_output(0)
            draft_token_ids.append(token_id_t)

            step_hidden_out = step_hidden
            if step_hidden_out.dtype != trt.float32:
                step_hidden_out = network.add_cast(step_hidden_out, trt.float32).get_output(0)
            draft_hidden_states.append(step_hidden_out)

            draft_present_k.append(new_k)
            draft_present_v.append(new_v)

            # --- grow cache (real K/V values, must accumulate -- unlike
            # the mask, which is cheaply rebuildable from scratch) ---
            grow_k = network.add_concatenation([running_cache_k, new_k])
            grow_k.axis = 0
            grow_v = network.add_concatenation([running_cache_v, new_v])
            grow_v.axis = 0
            running_cache_k = grow_k.get_output(0)
            running_cache_v = grow_v.get_output(0)

            cur_token_id = token_id_t
            cur_hidden_in = step_hidden  # self-chained, NOT the real main-model hidden_state

        # --- Outputs: stack per-repeat tensors ---
        ids_cat = network.add_concatenation(draft_token_ids)
        ids_cat.axis = 0
        ids_out = ids_cat.get_output(0)
        ids_out.name = "mtp_draft_token_ids"
        network.mark_output(ids_out)

        hs_cat = network.add_concatenation(draft_hidden_states)
        hs_cat.axis = 0
        hs_out = hs_cat.get_output(0)
        hs_out.name = "mtp_draft_hidden_states"
        network.mark_output(hs_out)

        logits_cat = network.add_concatenation(draft_logits)
        logits_cat.axis = 0
        logits_out = logits_cat.get_output(0)
        logits_out.name = "mtp_draft_logits"
        network.mark_output(logits_out)

        pk_cat = network.add_concatenation(draft_present_k)
        pk_cat.axis = 0
        pk_out = pk_cat.get_output(0)
        pk_out.name = "mtp_present_k"
        network.mark_output(pk_out)

        pv_cat = network.add_concatenation(draft_present_v)
        pv_cat.axis = 0
        pv_out = pv_cat.get_output(0)
        pv_out.name = "mtp_present_v"
        network.mark_output(pv_out)

        if verbose:
            print(f"[trtmc build] Building Qwen3.8 MTP draft-chain TRT engine "
                  f"(num_draft_tokens={num_draft_tokens}, hidden={hidden}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT MTP draft-chain engine build failed")
        return bytes(plan)

    def build_engine_multi_token(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, seq_len: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        """Build a TRT engine that processes `seq_len` NEW tokens in one call,
        emitting `seq_len` rows of logits -- one prediction per position.

        Exists to let MTP's draft token(s) be verified against what the main
        model actually predicts, in one engine call instead of `seq_len`
        separate ones (the real speculative-decoding throughput win; see
        `build_mtp_engine`'s docstring and the PR discussion for why a
        single-token-only main engine can't do this).

        Implementation: NOT a rewrite of the per-layer graph code. This
        chains `seq_len` ordinary single-token sub-steps together inside one
        TensorRT network, reusing `_add_deltanet_layer`/
        `_add_full_attention_layer` verbatim per sub-step -- the same
        composition every existing multi-step *driver loop* in this family
        already does across separate engine calls, just fused into one
        `build_serialized_network()` call so it runs as a single kernel
        launch sequence with no host round-trip between sub-steps.

        DeltaNet layers need no change at all: their recurrence
        (`conv_state`/`ssm_state`) is already expressed as "one call = one
        step", so chaining `result["present_conv"/"present_ssm"]` from
        sub-step t into sub-step t+1's `conv_state_in`/`ssm_state_in` is
        exactly what the existing function already supports.

        Full-attention layers also need no change to `_add_full_attention_layer`
        itself: sub-step t is called with `max_cache_length=max_cache_length+t`
        (so its internal `attention_window = max_cache_length+t+1` grows by
        one each sub-step) and a `cache_k`/`cache_v` tensor of matching
        width, built by concatenating the previous sub-step's cache with
        this sub-step's own new K/V row. Since every value here (`seq_len`,
        `t`, `max_cache_length`) is a Python-level constant at graph-build
        time, these growing shapes are ordinary static TRT shapes, not
        dynamic ones.

        Attention-mask contract: `attention_mask` input covers only the
        *persistent* cache (`max_cache_length` columns, same convention as
        `build_engine`'s single-token mask). No external input is needed for
        the `seq_len` new tokens' mutual causal visibility -- sub-step t's
        attention only ever reads a cache tensor of width
        `max_cache_length+t+1` (this sub-step's own row plus everything
        already grown), so any "future" new-token column is structurally
        unreachable, not merely masked. Each sub-step appends `t+1` zero
        (unmasked) columns onto the external mask itself for this reason.

        Outputs:
          - `logits`: shape `(seq_len, vocab)`, one row per sub-step.
          - `hidden_states`: shape `(seq_len, hidden_size)`, the final-normed
            hidden state behind each `logits` row -- reusable to derive the
            next MTP draft without another main-engine call.
          - `present_conv_{i}`/`present_ssm_{i}`: FINAL state after all
            `seq_len` sub-steps (DeltaNet layers). CAUTION: only valid to
            commit as the new persistent state if every sub-step's token was
            real/accepted -- if a later sub-step's token turns out to be a
            rejected draft, this reflects the wrong recurrence and must be
            discarded (re-run the single-token engine on the accepted
            prefix instead).
          - `present_k_{i}`/`present_v_{i}`: shape `(seq_len, kv_attention_size)`
            -- one new K/V row per sub-step, for the caller to write back
            into the persistent cache at consecutive positions.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        deltanet_num_heads: int = weights["_deltanet_num_heads"]
        deltanet_num_kv_heads: int = weights["_deltanet_num_kv_heads"]
        deltanet_head_dim: int = weights["_deltanet_head_dim"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        rope_theta: float = weights["_rope_theta"]

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "bf16":
            # Constants are staged as FP16 bytes (TensorRT's Weights constructor
            # does not accept ml_dtypes.bfloat16 arrays directly) and explicitly
            # cast to BF16 in-graph by graph_ops._cast_back_to_trt_dtype, which
            # every constant-building helper already calls to match its
            # activation's runtime dtype -- mirroring families/qwen's own
            # "storage np_dtype is fp16, runtime trt_dtype is bfloat16" pattern.
            work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Qwen3.8 precision {precision!r}; expected fp32, fp16, or bf16")

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim,
            quant_ctx=quant_ctx)
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # --- Inputs ---
        token_ids = network.add_input("token_ids", trt.int32, (seq_len,))
        position_ids = network.add_input("position_ids", trt.int32, (seq_len,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, max_cache_length))

        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi),
                trt.float32, (conv_dim, d_conv))
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32, (deltanet_num_heads, deltanet_head_dim, deltanet_head_dim))
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                work_trt_dtype, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                work_trt_dtype, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        ) = _prepare_runtime_inputs(
            network, work_trt_dtype, attention_mask,
            conv_state_inputs, ssm_state_inputs,
            cache_k_inputs, cache_v_inputs,
        )

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)

        # Sized for the worst case: the last sub-step's furthest position.
        table_len = max_cache_length + seq_len
        cos_half = graph_ops.make_rope_table_half_dim(
            table_len, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            table_len, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype)

        # Constants are staged at work_np_dtype (fp16 storage even for bf16
        # builds), so for bf16 they still need an explicit cast up to
        # work_trt_dtype before reaching RotaryEmbedding/IAttention layers,
        # which require all their input tensors to share one runtime dtype.
        if work_trt_dtype != trt.float32:
            eps_tensor = network.add_cast(eps_tensor, work_trt_dtype).get_output(0)
            cos_half_tensor = network.add_cast(cos_half_tensor, work_trt_dtype).get_output(0)
            sin_half_tensor = network.add_cast(sin_half_tensor, work_trt_dtype).get_output(0)

        # --- Embed all seq_len tokens up front ---
        gather = network.add_gather(embedding_table, token_ids, 0)
        embeds = gather.get_output(0)  # (seq_len, hidden)
        if work_trt_dtype != trt.float32:
            embeds = network.add_cast(embeds, work_trt_dtype).get_output(0)

        running_conv = list(conv_state_inputs)
        running_ssm = list(ssm_state_inputs)
        running_cache_k = list(cache_k_inputs)
        running_cache_v = list(cache_v_inputs)
        per_step_logits = []
        per_step_hidden = []
        per_layer_new_k = [[] for _ in range(num_attn)]
        per_layer_new_v = [[] for _ in range(num_attn)]

        for t in range(seq_len):
            hidden_slice = network.add_slice(
                embeds, start=(t, 0), shape=(1, hidden), stride=(1, 1))
            hidden_state = hidden_slice.get_output(0)
            pos_slice = network.add_slice(
                position_ids, start=(t,), shape=(1,), stride=(1,))
            position_id_t = pos_slice.get_output(0)

            # This sub-step's own t+1 new-token columns are always causally
            # valid (unmasked) -- see docstring above.
            zeros_t = graph_ops.add_constant(
                network, (1, t + 1),
                np.zeros((1, t + 1), dtype=work_np_dtype), dtype=work_np_dtype)
            if zeros_t.dtype != attention_mask.dtype:
                zeros_t = network.add_cast(zeros_t, attention_mask.dtype).get_output(0)
            mask_concat = network.add_concatenation([attention_mask, zeros_t])
            mask_concat.axis = 1
            mask_t = mask_concat.get_output(0)

            mamba_counter = 0
            attn_counter = 0
            for layer_idx in range(num_layers):
                prefix = f"layer.{layer_idx}"
                lt = layer_types[layer_idx]

                if lt == "deltanet":
                    result = _add_deltanet_layer(
                        network=network,
                        hidden=hidden_state,
                        conv_state_in=running_conv[mamba_counter],
                        ssm_state_in=running_ssm[mamba_counter],
                        eps_tensor=eps_tensor,
                        weights=weights,
                        prefix=prefix,
                        hidden_size=hidden,
                        d_inner=d_inner,
                        d_conv=d_conv,
                        conv_dim=conv_dim,
                        num_heads=deltanet_num_heads,
                        num_kv_heads=deltanet_num_kv_heads,
                        head_dim=deltanet_head_dim,
                        mlp_size=mlp_size,
                        dtype=work_np_dtype,
                        quant_ctx=quant_ctx,
                    )
                    hidden_state = result["hidden"]
                    running_conv[mamba_counter] = result["present_conv"]
                    running_ssm[mamba_counter] = result["present_ssm"]
                    mamba_counter += 1

                elif lt == "attention":
                    result = _add_full_attention_layer(
                        network=network,
                        hidden=hidden_state,
                        cache_k=running_cache_k[attn_counter],
                        cache_v=running_cache_v[attn_counter],
                        attention_mask=mask_t,
                        position_id=position_id_t,
                        cos_half_tensor=cos_half_tensor,
                        sin_half_tensor=sin_half_tensor,
                        eps_tensor=eps_tensor,
                        weights=weights,
                        prefix=prefix,
                        hidden_size=hidden,
                        attn_size=attn_size,
                        kv_attention_size=kv_attention_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        rotary_embedding_dim=rotary_embedding_dim,
                        max_cache_length=max_cache_length + t,
                        mlp_size=mlp_size,
                        dtype=work_np_dtype,
                        quant_ctx=quant_ctx,
                    )
                    hidden_state = result["hidden"]
                    new_k = result["present_k"]
                    new_v = result["present_v"]
                    per_layer_new_k[attn_counter].append(new_k)
                    per_layer_new_v[attn_counter].append(new_v)

                    grow_k = network.add_concatenation(
                        [running_cache_k[attn_counter], new_k])
                    grow_k.axis = 0
                    grow_v = network.add_concatenation(
                        [running_cache_v[attn_counter], new_v])
                    grow_v.axis = 0
                    running_cache_k[attn_counter] = grow_k.get_output(0)
                    running_cache_v[attn_counter] = grow_v.get_output(0)
                    attn_counter += 1

            # --- Final norm + lm_head for this sub-step ---
            hs = hidden_state
            if hs.dtype != work_trt_dtype:
                hs = network.add_cast(hs, work_trt_dtype).get_output(0)
            final_norm = weights.get("final_norm")
            if final_norm is not None and len(final_norm) > 0:
                hs = graph_ops.add_rms_norm(
                    network, hs, hidden, final_norm, eps_tensor,
                    dtype=work_np_dtype)
            hs_out = hs
            if hs_out.dtype != trt.float32:
                hs_out = network.add_cast(hs_out, trt.float32).get_output(0)
            per_step_hidden.append(hs_out)

            lm_head_matmul = graph_blocks.make_matmul_fn(network, work_np_dtype, quant_ctx)
            logits_t = lm_head_matmul(
                hs, hidden, vocab, weights.get("w_lm_head"), "w_lm_head")
            b_out = np.zeros(vocab, dtype=work_np_dtype)
            logits_t = graph_ops.add_bias_sum(
                network, logits_t, vocab, b_out, dtype=work_np_dtype)
            if logits_t.dtype != trt.float32:
                logits_t = network.add_cast(logits_t, trt.float32).get_output(0)
            per_step_logits.append(logits_t)

        # --- Outputs ---
        logits_cat = network.add_concatenation(per_step_logits)
        logits_cat.axis = 0
        logits_out = logits_cat.get_output(0)
        logits_out.name = "logits"
        network.mark_output(logits_out)

        # hidden_states[t] = final-normed hidden state used to produce
        # logits[t] -- lets a caller re-derive the NEXT MTP draft from any
        # sub-step's result without an extra main-engine call (e.g. after
        # accept, bootstrap the following round from hidden_states[-1]; on
        # reject, hidden_states[0] is still valid since sub-step 0's token
        # was real regardless of what happened at later sub-steps).
        hidden_states_cat = network.add_concatenation(per_step_hidden)
        hidden_states_cat.axis = 0
        hidden_states_out = hidden_states_cat.get_output(0)
        hidden_states_out.name = "hidden_states"
        network.mark_output(hidden_states_out)

        for mi in range(num_mamba):
            pc = running_conv[mi]
            ps = running_ssm[mi]
            if pc.dtype != trt.float32:
                pc = network.add_cast(pc, trt.float32).get_output(0)
            if ps.dtype != trt.float32:
                ps = network.add_cast(ps, trt.float32).get_output(0)
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            stacked_k = network.add_concatenation(per_layer_new_k[ai])
            stacked_k.axis = 0
            stacked_v = network.add_concatenation(per_layer_new_v[ai])
            stacked_v.axis = 0
            pk = stacked_k.get_output(0)
            pv = stacked_v.get_output(0)
            if pk.dtype != work_trt_dtype:
                pk = network.add_cast(pk, work_trt_dtype).get_output(0)
            if pv.dtype != work_trt_dtype:
                pv = network.add_cast(pv, work_trt_dtype).get_output(0)
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        if verbose:
            print(f"[trtmc build] Building Qwen3.8 multi-token ({seq_len}-token) "
                  f"TRT engine (hidden={hidden}, cache={max_cache_length}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT multi-token engine build failed")

        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Inject hybrid-specific config fields into the bundle."""
        raw = config.raw
        text_cfg = raw.get("text_config", raw)

        raw_layer_types = text_cfg.get("layer_types", [])
        layer_types = _parse_layer_types(raw_layer_types)

        deltanet_num_heads = text_cfg.get("linear_num_value_heads", 32)
        deltanet_head_dim = text_cfg.get("linear_value_head_dim",
                                         text_cfg.get("linear_key_head_dim", 128))
        deltanet_num_kv_heads = text_cfg.get("linear_num_key_heads", 16)
        d_inner = deltanet_num_heads * deltanet_head_dim
        d_conv = text_cfg.get("linear_conv_kernel_dim", 4)
        deltanet_qk_dim = deltanet_num_kv_heads * deltanet_head_dim
        conv_dim = deltanet_qk_dim + deltanet_qk_dim + d_inner

        num_mamba = sum(1 for lt in layer_types if lt == "deltanet")
        num_attn = sum(1 for lt in layer_types if lt == "attention")

        # Qwen3.8 keeps every decoder dimension under `text_config`, but the
        # C++ runtime reads the bundle config with a top-level nlohmann lookup
        # (`extract_json_int` -> `j.find(key)`), not a recursive search. Left
        # nested, `hidden_size`/`num_attention_heads`/`num_key_value_heads`/
        # `head_dim` all resolve to their fallbacks, `compute_kv_dim()` returns
        # 0, and the KV cache allocates zero-sized tensors -- `ok()` is false
        # and pipeline construction fails with "Failed to create Qwen38KvCache".
        # Publishing flat copies here is the family-owned fix: bundle config
        # overrides are emitted ahead of the raw config body, so the runtime
        # sees real dimensions while `text_config` stays intact for the
        # Python side.
        #
        # EOS IDs are emitted separately from generation_config.json because it
        # carries both stop tokens while text_config holds only one.
        flat_dims = {}
        for key in ("vocab_size", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "intermediate_size", "max_position_embeddings",
                    "rms_norm_eps", "bos_token_id"):
            value = text_cfg.get(key)
            if value is not None:
                flat_dims[key] = value
        if "head_dim" not in flat_dims:
            heads = flat_dims.get("num_attention_heads", 0)
            hidden = flat_dims.get("hidden_size", 0)
            if heads and hidden:
                flat_dims["head_dim"] = hidden // heads

        return {
            **flat_dims,
            "layer_types": layer_types,
            "num_mamba_layers": num_mamba,
            "num_attention_layers": num_attn,
            "d_inner": d_inner,
            "mamba_d_state": deltanet_head_dim,
            "mamba_d_conv": d_conv,
            "mamba_nheads": deltanet_num_heads,
            "mamba_head_dim": deltanet_head_dim,
            "conv_dim": conv_dim,
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


def _add_deltanet_layer(
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
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx=None,
) -> dict[str, trt.ITensor]:
    """Add one Gated DeltaNet layer (single-step decode).

    DeltaNet uses delta-rule linear attention with:
      - Conv1d on QKV projection output
      - L2-normalized Q,K with Compact GQA/MQA K/V
      - Per-head decay (A_log + softplus(a + dt_bias))
      - Per-head beta (write strength via sigmoid)
      - Delta rule state update: S' = decay*S + outer(k, (v - S@k)*beta)
      - Gated output: norm(S'@q) * silu(z) * norm_weight

    Returns: {hidden, present_conv, present_ssm}
    """
    qk_dim = num_kv_heads * head_dim  # Q and K dimension before Compact GQA/MQA K/V

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype)

    # ===== 2. Input projections =====
    matmul = graph_blocks.make_matmul_fn(network, dtype, quant_ctx)

    # QKV combined: [1, hidden] -> [1, conv_dim]
    qkv = matmul(
        normed, hidden_size, conv_dim,
        weights.get(f"{prefix}.deltanet_in_proj_qkv"), f"{prefix}.deltanet_in_proj_qkv")

    # Gate (z): [1, hidden] -> [1, d_inner]
    z = matmul(
        normed, hidden_size, d_inner,
        weights.get(f"{prefix}.deltanet_z_proj"), f"{prefix}.deltanet_z_proj")

    # Decay projection (a): [1, hidden] -> [1, num_heads]
    a_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads,
        weights[f"{prefix}.deltanet_a_proj"], dtype=dtype)

    # Beta projection (b): [1, hidden] -> [1, num_heads]
    b_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads,
        weights[f"{prefix}.deltanet_b_proj"], dtype=dtype)

    # ===== 3. Conv1d step on QKV =====
    # conv_state_in: [conv_dim, d_conv]
    # qkv: [1, conv_dim] -> [conv_dim, 1]
    qkv_col = network.add_shuffle(qkv)
    qkv_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(conv_dim, d_conv - 1),
            stride=(1, 1))
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), qkv_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = qkv_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"],
        dtype=dtype)
    if conv_w.dtype != present_conv.dtype:
        conv_w = network.add_cast(conv_w, present_conv.dtype).get_output(0)
    conv_prod = network.add_elementwise(
        present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM,
        1 << 1, keep_dims=True)
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim,
        weights[f"{prefix}.conv1d_bias"], dtype=dtype)
    qkv_activated = graph_ops.add_activation(
        network, conv_out, "silu", dtype=dtype)

    # ===== 4. Split Q, K, V from activated output =====
    offset = 0
    q_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    q_raw_t = q_slice.get_output(0)
    offset += qk_dim

    k_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    k_raw_t = k_slice.get_output(0)
    offset += qk_dim

    v_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    v_raw = v_slice.get_output(0)

    # ===== 5. L2-normalize Q and K =====
    # Reshape to [num_kv_heads, head_dim], normalize per-head, reshape back
    q_heads_in = network.add_shuffle(q_raw_t)
    q_heads_in.reshape_dims = (num_kv_heads, head_dim)
    q_normed = graph_ops.add_l2_norm(
        network, q_heads_in.get_output(0), 1, eps=1e-6, dtype=dtype)

    k_heads_in = network.add_shuffle(k_raw_t)
    k_heads_in.reshape_dims = (num_kv_heads, head_dim)
    k_normed = graph_ops.add_l2_norm(
        network, k_heads_in.get_output(0), 1, eps=1e-6, dtype=dtype)

    # ===== 6. keep compact Q,K from num_kv_heads -> num_heads =====
    heads_per_group = num_heads // num_kv_heads

    if heads_per_group > 1:
        # Q: [num_kv_heads, head_dim] -> [num_kv_heads, 1, head_dim] ->
        #    tile -> [num_kv_heads, heads_per_group, head_dim] ->
        #    [num_heads, head_dim]
        q_3d = network.add_shuffle(q_normed)
        q_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        tile_ones = graph_ops.add_constant(
            network, (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=dtype), dtype=dtype)
        if tile_ones.dtype != q_3d.get_output(0).dtype:
            tile_ones = network.add_cast(tile_ones, q_3d.get_output(0).dtype).get_output(0)
        q_tiled = network.add_elementwise(
            q_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        q_expanded_s = network.add_shuffle(q_tiled.get_output(0))
        q_expanded_s.reshape_dims = (num_heads, head_dim)
        q_expanded = q_expanded_s.get_output(0)

        k_3d = network.add_shuffle(k_normed)
        k_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        k_tiled = network.add_elementwise(
            k_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        k_t_s = network.add_shuffle(k_tiled.get_output(0))
        k_t_s.reshape_dims = (num_heads, head_dim)
        k_t = k_t_s.get_output(0)
    else:
        q_expanded = q_normed
        k_t = k_normed

    # V: [1, d_inner] -> [num_heads, head_dim]
    v_heads = network.add_shuffle(v_raw)
    v_heads.reshape_dims = (num_heads, head_dim)
    v_t = v_heads.get_output(0)

    # ===== 7. Compute decay: -exp(A_log) * softplus(a + dt_bias) per head =====
    # Transformers performs the decay and recurrent rule in FP32 even when
    # the model projections use FP16.  Keeping these tensors in the model
    # storage dtype quantizes the persistent state again on every token.
    recurrent_dtype = trt.float32

    def recurrent_cast(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == recurrent_dtype:
            return tensor
        return network.add_cast(tensor, recurrent_dtype).get_output(0)

    # A: [num_heads] (precomputed as -exp(A_log))
    A_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.A"], dtype=np.float32)

    # dt_bias: [num_heads]
    dt_bias_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.dt_bias"], dtype=np.float32)
    a_biased = network.add_elementwise(
        recurrent_cast(a_raw), dt_bias_const, trt.ElementWiseOperation.SUM)

    # softplus(a + dt_bias): log(1 + exp(x))
    a_exp = network.add_unary(a_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32)
    a_exp_p1 = network.add_elementwise(
        a_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    a_softplus = network.add_unary(
        a_exp_p1.get_output(0), trt.UnaryOperation.LOG)

    # decay = A * softplus(...) per head: [1, num_heads]
    decay_flat = network.add_elementwise(
        A_const, a_softplus.get_output(0), trt.ElementWiseOperation.PROD)
    # exp(decay) for the state update: [1, num_heads] -> [num_heads, 1, 1]
    decay_reshaped = network.add_shuffle(decay_flat.get_output(0))
    decay_reshaped.reshape_dims = (num_heads, 1, 1)
    decay_exp = network.add_unary(
        decay_reshaped.get_output(0), trt.UnaryOperation.EXP)

    # ===== 8. Compute beta: sigmoid(b) per head =====
    # b_raw: [1, num_heads]
    beta = network.add_activation(b_raw, trt.ActivationType.SIGMOID)
    # [1, num_heads] -> [num_heads, 1]
    beta_reshaped = network.add_shuffle(recurrent_cast(beta.get_output(0)))
    beta_reshaped.reshape_dims = (num_heads, 1)

    # ===== 9. Delta rule state update =====
    # HF state layout: [H, K_dim, V_dim]
    # ssm_state_in: [num_heads, head_dim, head_dim]  (K on axis -2, V on axis -1)
    # k: [num_heads, head_dim], q: [num_heads, head_dim], v: [num_heads, head_dim]

    # 9a. Decay state first: state = state * exp(g)
    decayed_state = network.add_elementwise(
        decay_exp.get_output(0), recurrent_cast(ssm_state_in),
        trt.ElementWiseOperation.PROD)

    # 9b. kv_mem = state^T @ k: read old value for this key
    # [H, V, K] @ [H, K, 1] = [H, V, 1]  (transpose state to swap K/V axes)
    k_recurrent = recurrent_cast(k_t)
    v_recurrent = recurrent_cast(v_t)
    q_recurrent = recurrent_cast(q_expanded)

    k_col = network.add_shuffle(k_recurrent)
    k_col.reshape_dims = (num_heads, head_dim, 1)
    kv_old_3d = network.add_matrix_multiply(
        decayed_state.get_output(0), trt.MatrixOperation.TRANSPOSE,
        k_col.get_output(0), trt.MatrixOperation.NONE)
    kv_old = network.add_shuffle(kv_old_3d.get_output(0))
    kv_old.reshape_dims = (num_heads, head_dim)

    # 9c. delta = (v - kv_mem) * beta
    v_minus_old = network.add_elementwise(
        v_recurrent, kv_old.get_output(0), trt.ElementWiseOperation.SUB)
    v_delta = network.add_elementwise(
        v_minus_old.get_output(0), beta_reshaped.get_output(0),
        trt.ElementWiseOperation.PROD)

    # 9d. state_new = decayed_state + outer(k, delta)
    # outer: k[:, :, None] * delta[:, None, :] = [H, K, 1] @ [H, 1, V] = [H, K, V]
    k_col2 = network.add_shuffle(k_recurrent)
    k_col2.reshape_dims = (num_heads, head_dim, 1)
    v_delta_row = network.add_shuffle(v_delta.get_output(0))
    v_delta_row.reshape_dims = (num_heads, 1, head_dim)
    outer_prod = network.add_matrix_multiply(
        k_col2.get_output(0), trt.MatrixOperation.NONE,
        v_delta_row.get_output(0), trt.MatrixOperation.NONE)

    new_state = network.add_elementwise(
        decayed_state.get_output(0), outer_prod.get_output(0),
        trt.ElementWiseOperation.SUM)
    present_ssm = new_state.get_output(0)

    # 9e. output = state_new^T @ (q * scale)
    # HF applies: query *= 1/sqrt(k_dim)
    q_scale = graph_ops.add_constant(
        network, (1, 1),
        np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32), dtype=np.float32)
    q_scaled = network.add_elementwise(
        q_recurrent, q_scale, trt.ElementWiseOperation.PROD)
    # [H, V, K] @ [H, K, 1] = [H, V, 1]
    q_col = network.add_shuffle(q_scaled.get_output(0))
    q_col.reshape_dims = (num_heads, head_dim, 1)
    output_3d = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.TRANSPOSE,
        q_col.get_output(0), trt.MatrixOperation.NONE)
    output_flat = network.add_shuffle(output_3d.get_output(0))
    output_flat.reshape_dims = (1, d_inner)

    # ===== 10. Gated RMSNorm per-head: weight * norm(output) * silu(z) =====
    # The reference recurrent kernel returns the attention output in the model
    # storage dtype before Qwen3_5RMSNormGated casts it back to FP32.
    recurrent_output = output_flat.get_output(0)
    if recurrent_output.dtype != hidden.dtype:
        recurrent_output = network.add_cast(
            recurrent_output, hidden.dtype).get_output(0)

    # HF norm operates per head_v_dim: reshape to [num_heads, head_dim], norm, reshape back
    deltanet_norm_w = weights[f"{prefix}.deltanet_norm"]
    # Use same eps as HF Qwen3_5RMSNormGated (config.rms_norm_eps = 1e-6)
    eps_small = graph_ops.add_constant(
        network, (1, 1),
        np.array([1e-6], dtype=np.float32), dtype=np.float32)

    # Reshape output and z to [num_heads, head_dim] for per-head norm
    output_heads = network.add_shuffle(recurrent_output)
    output_heads.reshape_dims = (num_heads, head_dim)
    norm_input = output_heads.get_output(0)
    norm_output_dtype = norm_input.dtype
    if dtype != np.float32:
        norm_input = network.add_cast(norm_input, trt.float32).get_output(0)

    # Per-head RMSNorm: norm each head independently
    sq = network.add_elementwise(
        norm_input, norm_input,
        trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        norm_input, recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    # Reshape back and apply weight
    norm_flat = network.add_shuffle(normalized.get_output(0))
    norm_flat.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(
        network, (1, d_inner), deltanet_norm_w, dtype=np.float32)
    normed_output = network.add_elementwise(
        norm_flat.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    normed_output_tensor = normed_output.get_output(0)
    if normed_output_tensor.dtype != norm_output_dtype:
        normed_output_tensor = network.add_cast(
            normed_output_tensor, norm_output_dtype).get_output(0)

    # Gate: multiply by silu(z)
    z_activated = graph_ops.add_activation(network, z, "silu", dtype=dtype)
    gated = network.add_elementwise(
        normed_output_tensor, z_activated,
        trt.ElementWiseOperation.PROD)

    # ===== 11. Output projection + residual =====
    out = matmul(
        gated.get_output(0), d_inner, hidden_size,
        weights.get(f"{prefix}.deltanet_out_proj"), f"{prefix}.deltanet_out_proj")

    residual = network.add_elementwise(
        hidden, out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # ===== 12. Post-attention norm + SwiGLU MLP + residual =====
    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor, dtype=dtype)

    mlp_out = graph_blocks.add_swiglu_mlp(
        network, post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
        quant_ctx=quant_ctx,
    )

    mlp_residual = network.add_elementwise(
        hidden_after_attn, mlp_out,
        trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_full_attention_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attn_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    max_cache_length: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx=None,
) -> dict[str, trt.ITensor]:
    """Add one full self-attention layer with output gating.

    Qwen3.8 full attention has:
      - QK-norm with (1+weight) centering
      - Partial RoPE (25% of dims)
      - Output gating: context * sigmoid(gate) BEFORE o_proj
      - SwiGLU MLP after attention

    Returns: {hidden, present_k, present_v}
    """
    attention_window = max_cache_length + 1

    # Pre-attention norm
    normed = graph_blocks.apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor, "rmsnorm", dtype=dtype)

    # QKV projections
    matmul = graph_blocks.make_matmul_fn(network, dtype, quant_ctx)
    q = matmul(
        normed, hidden_size, attn_size,
        weights.get(f"{prefix}.w_q"), f"{prefix}.w_q")
    k = matmul(
        normed, hidden_size, kv_attention_size,
        weights.get(f"{prefix}.w_k"), f"{prefix}.w_k")
    v = matmul(
        normed, hidden_size, kv_attention_size,
        weights.get(f"{prefix}.w_v"), f"{prefix}.w_v")

    # Per-head QK norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor,
            dtype=dtype)

    # Native RoPE
    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, rotary_embedding_dim)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, rotary_embedding_dim)

    # Save present K/V
    present_k = k
    present_v = v

    # Reshape K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation(
        [cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = graph_ops.add_attention_from_rows(
        network, q, all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
        q_seq=1, kv_seq=attention_window,
        mask=mask_4d)

    # Gate: applied BEFORE o_proj (HF order). `weights` won't hold this
    # tensor when quant_ctx already owns it (load_weights() skips
    # materializing a dequantized copy in that case), so presence must be
    # checked via quant_ctx too -- otherwise a quantized checkpoint would
    # silently skip gating instead of applying it.
    gate_attn_name = f"{prefix}.w_gate_attn"
    gate_attn_w = weights.get(gate_attn_name)
    attn_out = context_flat
    if gate_attn_w is not None or _owned(quant_ctx, gate_attn_name):
        gate = matmul(
            normed, hidden_size, attn_size, gate_attn_w, gate_attn_name)
        gate_sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        gated = network.add_elementwise(
            attn_out, gate_sigmoid.get_output(0),
            trt.ElementWiseOperation.PROD)
        attn_out = gated.get_output(0)

    # Output projection (AFTER gate)
    attn_out = matmul(
        attn_out, attn_size, hidden_size,
        weights.get(f"{prefix}.w_o"), f"{prefix}.w_o")

    # Residual after attention
    residual = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # Post-attention norm + SwiGLU MLP + residual
    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor, dtype=dtype)

    mlp_out = graph_blocks.add_swiglu_mlp(
        network, post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
        quant_ctx=quant_ctx,
    )

    mlp_residual = network.add_elementwise(
        hidden_after_attn, mlp_out,
        trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }
