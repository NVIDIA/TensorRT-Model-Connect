"""Shared DiT (Diffusion Transformer) engine builder.

Builds a TensorRT engine for a DiT-style denoiser. Parameterized for
variant selection, mirroring how standard_decoder_builder.py works.

Reusable by: Wan2.1, Hunyuan Video, CogVideoX, FLUX (with conditioning_type).

Engine I/O:
    Inputs:
        hidden_states [1, num_patches, dim] float32 (patchified latent)
        timestep_embedding [1, dim * 6] float32 (from external timestep MLP)
        encoder_hidden_states [1, text_seq_len, context_dim] float32
        rotary_cos [1, num_patches, 1, head_dim] float32 (precomputed)
        rotary_sin [1, num_patches, 1, head_dim] float32 (precomputed)
    Outputs:
        output [1, num_patches, dim] float32

The timestep embedding, patch embedding, and RoPE are computed externally
(in the DiffusionRunner / C++ backend) and passed as inputs. This keeps
the TRT engine focused on the core transformer computation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


def build_standard_dit_engine(
    weights: WeightDict,
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    context_dim: int,
    num_patches: int,
    text_seq_len: int = 512,
    qk_norm: bool = True,
    cross_attn_norm: bool = True,
    ffn_activation: str = "gelu_new",
    use_rope: bool = True,
    eps: float = 1e-6,
    verbose: bool = False,
) -> bytes:
    """Build DiT denoiser TRT engine plan.

    Args:
        weights: Weight dict with DiT weights. Expected keys per layer:
            - blocks.{i}.attn1.to_q/to_k/to_v.weight/bias (self-attn)
            - blocks.{i}.attn1.to_out.0.weight/bias
            - blocks.{i}.attn1.norm_q/norm_k.weight (QK norm)
            - blocks.{i}.norm1 (no weight — elementwise_affine=False)
            - blocks.{i}.attn2.to_q/to_k/to_v.weight/bias (cross-attn)
            - blocks.{i}.attn2.to_out.0.weight/bias
            - blocks.{i}.attn2.norm_q/norm_k.weight
            - blocks.{i}.attn2.add_k_proj/add_v_proj.weight/bias (if context needs projection)
            - blocks.{i}.norm2.weight/bias (cross-attn norm, if enabled)
            - blocks.{i}.ffn.net.0.proj.weight/bias (GELU)
            - blocks.{i}.ffn.net.2.weight/bias (output proj)
            - blocks.{i}.norm3 (no weight — elementwise_affine=False)
            - blocks.{i}.scale_shift_table [1, 6, dim]
            Global:
            - norm_out (no weight — elementwise_affine=False)
            - proj_out.weight/bias
            - scale_shift_table [1, 2, dim]
        dim: Hidden dimension of the DiT.
        num_heads: Number of attention heads.
        num_layers: Number of DiT blocks.
        ffn_dim: Feed-forward inner dimension.
        context_dim: Text encoder output dimension (before projection).
        num_patches: Total number of patches (T/pt * H/ph * W/pw).
        text_seq_len: Maximum text sequence length.
        qk_norm: Apply RMSNorm to Q and K.
        cross_attn_norm: Apply LayerNorm before cross-attention.
        ffn_activation: Activation for FFN.
        use_rope: Apply RoPE to self-attention Q/K. When False, the engine
            omits rotary_cos/rotary_sin inputs (suitable for models that use
            fixed position embeddings, e.g. PixArt).
        eps: LayerNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    head_dim = dim // num_heads

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Inputs
    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (num_patches, dim))
    # Block modulation temb from external embedder: [1, 6 * dim]
    temb_inp = network.add_input(
        "timestep_embedding", trt.float32, (1, 6 * dim))
    # Time embed for final output modulation: [1, dim]
    time_embed_inp = network.add_input(
        "time_embed", trt.float32, (1, dim))
    encoder_hidden = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, context_dim))

    # Optional cross-attention mask: [1, 1, text_seq_len] float32.
    # 0.0 for valid tokens, -10000.0 for padding.
    # Only added when use_rope=False (PixArt, etc.) since Wan models
    # don't need it (fixed-length text with no padding).
    cross_attn_mask = None
    if not use_rope:
        cross_attn_mask = network.add_input(
            "encoder_attention_mask", trt.float32, (1, 1, text_seq_len))

    # Constants
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))

    # RoPE inputs and precomputation (only when use_rope=True)
    rotary_cos = rotary_sin = None
    if use_rope:
        rotary_cos = network.add_input(
            "rotary_cos", trt.float32, (num_patches, head_dim))
        rotary_sin = network.add_input(
            "rotary_sin", trt.float32, (num_patches, head_dim))

    hidden = hidden_inp

    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"

        # --- AdaLN modulation ---
        # scale_shift_table: [1, 6, dim] -> flatten to [1, 6*dim]
        sst = weights[f"{prefix}.scale_shift_table"]
        sst_const = graph_ops.add_constant(
            network, (1, 6 * dim), sst.reshape(1, 6 * dim))

        # modulation = sst + temb
        modulation = network.add_elementwise(
            sst_const, temb_inp, trt.ElementWiseOperation.SUM)

        # Chunk into 6 parts: shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff
        chunks = []
        for i in range(6):
            s = network.add_slice(
                modulation.get_output(0),
                start=(0, i * dim),
                shape=(1, dim),
                stride=(1, 1),
            )
            chunks.append(s.get_output(0))
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = chunks

        # === 1. Self-attention with AdaLN + RoPE ===
        normed = graph_ops.add_adaptive_layernorm(
            network, hidden, scale_sa, shift_sa, dim, eps)

        # QKV projections
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, dim, dim,
            weights[f"{prefix}.attn1.to_q.weight"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, dim, dim,
            weights[f"{prefix}.attn1.to_k.weight"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, dim, dim,
            weights[f"{prefix}.attn1.to_v.weight"])

        # Biases
        q_bias = weights.get(f"{prefix}.attn1.to_q.bias")
        if q_bias is not None:
            q = graph_ops.add_bias_sum(network, q, dim, q_bias)
        k_bias = weights.get(f"{prefix}.attn1.to_k.bias")
        if k_bias is not None:
            k = graph_ops.add_bias_sum(network, k, dim, k_bias)
        v_bias = weights.get(f"{prefix}.attn1.to_v.bias")
        if v_bias is not None:
            v = graph_ops.add_bias_sum(network, v, dim, v_bias)

        # QK norm
        if qk_norm:
            q_norm_w = weights.get(f"{prefix}.attn1.norm_q.weight")
            k_norm_w = weights.get(f"{prefix}.attn1.norm_k.weight")
            if q_norm_w is not None:
                q = graph_ops.add_rms_norm(
                    network, q, dim, q_norm_w, eps_t)
            if k_norm_w is not None:
                k = graph_ops.add_rms_norm(
                    network, k, dim, k_norm_w, eps_t)

        # Apply RoPE to Q and K (skip when use_rope=False)
        if use_rope:
            q = graph_ops.add_apply_rope_native_from_full_cache(
                network, q, num_heads, head_dim,
                rotary_cos, rotary_sin, num_patches,
                interleaved=True)
            k = graph_ops.add_apply_rope_native_from_full_cache(
                network, k, num_heads, head_dim,
                rotary_cos, rotary_sin, num_patches,
                interleaved=True)

        # Multi-head attention
        context_flat = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=num_patches, kv_seq=num_patches,
            tag=f"{prefix}.attn1")

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat, dim, dim,
            weights[f"{prefix}.attn1.to_out.0.weight"])
        o_bias = weights.get(f"{prefix}.attn1.to_out.0.bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(network, attn_out, dim, o_bias)

        # Gate + residual
        gated = network.add_elementwise(
            attn_out, gate_sa, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

        # === 2. Cross-attention ===
        if cross_attn_norm:
            cross_norm_w = weights.get(f"{prefix}.norm2.weight")
            cross_norm_b = weights.get(f"{prefix}.norm2.bias")
            if cross_norm_w is not None:
                cross_normed = graph_ops.add_layer_norm(
                    network, hidden, dim,
                    cross_norm_w,
                    cross_norm_b if cross_norm_b is not None else np.zeros(dim, dtype=np.float32),
                    eps_t)
            else:
                cross_normed = hidden
        else:
            cross_normed = hidden

        # Cross-attention: Q from hidden, K/V from encoder_hidden
        cross_q = graph_ops.add_matmul_rhs_constant(
            network, cross_normed, dim, dim,
            weights[f"{prefix}.attn2.to_q.weight"])
        cq_bias = weights.get(f"{prefix}.attn2.to_q.bias")
        if cq_bias is not None:
            cross_q = graph_ops.add_bias_sum(network, cross_q, dim, cq_bias)

        # K/V from encoder (may use add_k_proj/add_v_proj for context projection)
        add_k_proj_w = weights.get(f"{prefix}.attn2.add_k_proj.weight")
        if add_k_proj_w is not None:
            # Context needs projection: [text_seq, context_dim] -> [text_seq, dim]
            cross_k = graph_ops.add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim, add_k_proj_w)
            add_k_bias = weights.get(f"{prefix}.attn2.add_k_proj.bias")
            if add_k_bias is not None:
                cross_k = graph_ops.add_bias_sum(network, cross_k, dim, add_k_bias)
            cross_v = graph_ops.add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim,
                weights[f"{prefix}.attn2.add_v_proj.weight"])
            add_v_bias = weights.get(f"{prefix}.attn2.add_v_proj.bias")
            if add_v_bias is not None:
                cross_v = graph_ops.add_bias_sum(network, cross_v, dim, add_v_bias)
        else:
            # Direct K/V from already-projected context
            cross_k = graph_ops.add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim,
                weights[f"{prefix}.attn2.to_k.weight"])
            ck_bias = weights.get(f"{prefix}.attn2.to_k.bias")
            if ck_bias is not None:
                cross_k = graph_ops.add_bias_sum(network, cross_k, dim, ck_bias)
            cross_v = graph_ops.add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim,
                weights[f"{prefix}.attn2.to_v.weight"])
            cv_bias = weights.get(f"{prefix}.attn2.to_v.bias")
            if cv_bias is not None:
                cross_v = graph_ops.add_bias_sum(network, cross_v, dim, cv_bias)

        # QK norm for cross-attention
        if qk_norm:
            cq_norm = weights.get(f"{prefix}.attn2.norm_q.weight")
            ck_norm = weights.get(f"{prefix}.attn2.norm_k.weight")
            if cq_norm is not None:
                cross_q = graph_ops.add_rms_norm(
                    network, cross_q, dim, cq_norm, eps_t)
            # For cross-attn with add_k_proj, the K norm is norm_added_k
            ck_added_norm = weights.get(f"{prefix}.attn2.norm_added_k.weight")
            if ck_added_norm is not None:
                cross_k = graph_ops.add_rms_norm(
                    network, cross_k, dim, ck_added_norm, eps_t)
            elif ck_norm is not None:
                cross_k = graph_ops.add_rms_norm(
                    network, cross_k, dim, ck_norm, eps_t)

        # Cross multi-head attention (no RoPE)
        cross_mask_4d = None
        if cross_attn_mask is not None:
            cross_mask = network.add_shuffle(cross_attn_mask)
            cross_mask.reshape_dims = (1, 1, 1, text_seq_len)
            cross_mask_4d = cross_mask.get_output(0)
        c_context_flat = graph_ops.add_attention_from_rows(
            network, cross_q, cross_k, cross_v,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=num_patches, kv_seq=text_seq_len,
            mask=cross_mask_4d,
            tag=f"{prefix}.attn2")

        cross_out = graph_ops.add_matmul_rhs_constant(
            network, c_context_flat, dim, dim,
            weights[f"{prefix}.attn2.to_out.0.weight"])
        co_bias = weights.get(f"{prefix}.attn2.to_out.0.bias")
        if co_bias is not None:
            cross_out = graph_ops.add_bias_sum(network, cross_out, dim, co_bias)

        # Cross-attn residual (no gate in Wan)
        hidden = network.add_elementwise(
            hidden, cross_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # === 3. FFN with AdaLN ===
        ffn_normed = graph_ops.add_adaptive_layernorm(
            network, hidden, scale_ff, shift_ff, dim, eps)

        # GELU FFN: Linear(dim, ffn_dim) -> GELU -> Linear(ffn_dim, dim)
        ffn_fc1 = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, dim, ffn_dim,
            weights[f"{prefix}.ffn.net.0.proj.weight"])
        fc1_bias = weights.get(f"{prefix}.ffn.net.0.proj.bias")
        if fc1_bias is not None:
            ffn_fc1 = graph_ops.add_bias_sum(network, ffn_fc1, ffn_dim, fc1_bias)

        ffn_act = graph_ops.add_gelu_new(network, ffn_fc1)

        ffn_fc2 = graph_ops.add_matmul_rhs_constant(
            network, ffn_act, ffn_dim, dim,
            weights[f"{prefix}.ffn.net.2.weight"])
        fc2_bias = weights.get(f"{prefix}.ffn.net.2.bias")
        if fc2_bias is not None:
            ffn_fc2 = graph_ops.add_bias_sum(network, ffn_fc2, dim, fc2_bias)

        # Gate + residual
        gated_ff = network.add_elementwise(
            ffn_fc2, gate_ff, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated_ff.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

    # --- Final output: AdaLN modulation + projection ---
    # HF: shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    # scale_shift_table: [1, 2, dim], time_embed: [1, dim] -> unsqueeze -> [1, 1, dim]
    # Result after add: [1, 2, dim], chunk -> shift [1, 1, dim], scale [1, 1, dim]
    final_sst = weights["scale_shift_table"]  # [1, 2, dim]
    final_sst_const = graph_ops.add_constant(
        network, (1, 2 * dim), final_sst.reshape(1, 2 * dim))

    # time_embed_inp: [1, dim] -> tile to [1, 2*dim] for broadcast add
    time_embed_tiled = network.add_concatenation(
        [time_embed_inp, time_embed_inp])
    time_embed_tiled.axis = 1  # [1, 2*dim]

    final_modulation = network.add_elementwise(
        final_sst_const, time_embed_tiled.get_output(0),
        trt.ElementWiseOperation.SUM)

    final_shift = network.add_slice(
        final_modulation.get_output(0),
        start=(0, 0), shape=(1, dim), stride=(1, 1))
    final_scale = network.add_slice(
        final_modulation.get_output(0),
        start=(0, dim), shape=(1, dim), stride=(1, 1))

    # Final LayerNorm + AdaLN modulation
    hidden = graph_ops.add_adaptive_layernorm(
        network, hidden, final_scale.get_output(0),
        final_shift.get_output(0), dim, eps)

    # Final projection: [num_patches, dim] -> [num_patches, out_channels * prod(patch_size)]
    proj_out_w = weights["proj_out.weight"]
    out_dim = proj_out_w.shape[1]  # [dim, out_channels * prod(patch_size)]
    output = graph_ops.add_matmul_rhs_constant(
        network, hidden, dim, out_dim, proj_out_w)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = graph_ops.add_bias_sum(network, output, out_dim, proj_out_b)

    # Mark output
    cast_output = network.add_cast(output, trt.float32)
    output_final = cast_output.get_output(0)
    output_final.name = "output"
    network.mark_output(output_final)

    # --- Build ---
    print(f"[dit-builder] Building TRT engine "
          f"(dim={dim}, layers={num_layers}, patches={num_patches}) ...",
          file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for DiT")
    return bytes(plan)


def load_dit_weights(
    model_dir: str,
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    context_dim: int,
) -> WeightDict:
    """Load DiT weights from a diffusers-format transformer directory.

    Expects: model_dir/diffusion_pytorch_model.safetensors (or sharded).
    Returns WeightDict with transposed projections for TRT matmul.
    """
    from pathlib import Path
    from ...checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _t(name: str) -> np.ndarray:
        """Load and transpose [out, in] -> [in, out]."""
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name: str) -> np.ndarray:
        """Load flat (1D) weight."""
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe_t(name: str) -> np.ndarray | None:
        if _has_tensor(readers, name):
            return _t(name)
        return None

    def _maybe_f(name: str) -> np.ndarray | None:
        if _has_tensor(readers, name):
            return _f(name)
        return None

    for i in range(num_layers):
        p = f"blocks.{i}"

        # Self-attention
        weights[f"{p}.attn1.to_q.weight"] = _t(f"{p}.attn1.to_q.weight")
        weights[f"{p}.attn1.to_k.weight"] = _t(f"{p}.attn1.to_k.weight")
        weights[f"{p}.attn1.to_v.weight"] = _t(f"{p}.attn1.to_v.weight")
        weights[f"{p}.attn1.to_out.0.weight"] = _t(f"{p}.attn1.to_out.0.weight")

        for proj in ("to_q", "to_k", "to_v"):
            b = _maybe_f(f"{p}.attn1.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn1.{proj}.bias"] = b
        b = _maybe_f(f"{p}.attn1.to_out.0.bias")
        if b is not None:
            weights[f"{p}.attn1.to_out.0.bias"] = b

        # QK norm
        for norm in ("norm_q", "norm_k"):
            w = _maybe_f(f"{p}.attn1.{norm}.weight")
            if w is not None:
                weights[f"{p}.attn1.{norm}.weight"] = w

        # Cross-attention
        weights[f"{p}.attn2.to_q.weight"] = _t(f"{p}.attn2.to_q.weight")
        weights[f"{p}.attn2.to_out.0.weight"] = _t(f"{p}.attn2.to_out.0.weight")

        for proj in ("to_q", "to_k", "to_v"):
            b = _maybe_f(f"{p}.attn2.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn2.{proj}.bias"] = b
        b = _maybe_f(f"{p}.attn2.to_out.0.bias")
        if b is not None:
            weights[f"{p}.attn2.to_out.0.bias"] = b

        # Cross-attn K/V: either to_k/to_v or add_k_proj/add_v_proj
        if _has_tensor(readers, f"{p}.attn2.add_k_proj.weight"):
            weights[f"{p}.attn2.add_k_proj.weight"] = _t(f"{p}.attn2.add_k_proj.weight")
            weights[f"{p}.attn2.add_v_proj.weight"] = _t(f"{p}.attn2.add_v_proj.weight")
            b = _maybe_f(f"{p}.attn2.add_k_proj.bias")
            if b is not None:
                weights[f"{p}.attn2.add_k_proj.bias"] = b
            b = _maybe_f(f"{p}.attn2.add_v_proj.bias")
            if b is not None:
                weights[f"{p}.attn2.add_v_proj.bias"] = b
        else:
            weights[f"{p}.attn2.to_k.weight"] = _t(f"{p}.attn2.to_k.weight")
            weights[f"{p}.attn2.to_v.weight"] = _t(f"{p}.attn2.to_v.weight")

        # Cross-attn QK norm
        for norm in ("norm_q", "norm_k", "norm_added_k"):
            w = _maybe_f(f"{p}.attn2.{norm}.weight")
            if w is not None:
                weights[f"{p}.attn2.{norm}.weight"] = w

        # Cross-attn norm (LayerNorm)
        w = _maybe_f(f"{p}.norm2.weight")
        if w is not None:
            weights[f"{p}.norm2.weight"] = w
        b = _maybe_f(f"{p}.norm2.bias")
        if b is not None:
            weights[f"{p}.norm2.bias"] = b

        # FFN
        weights[f"{p}.ffn.net.0.proj.weight"] = _t(f"{p}.ffn.net.0.proj.weight")
        weights[f"{p}.ffn.net.2.weight"] = _t(f"{p}.ffn.net.2.weight")
        b = _maybe_f(f"{p}.ffn.net.0.proj.bias")
        if b is not None:
            weights[f"{p}.ffn.net.0.proj.bias"] = b
        b = _maybe_f(f"{p}.ffn.net.2.bias")
        if b is not None:
            weights[f"{p}.ffn.net.2.bias"] = b

        # Scale-shift table
        sst = _load_tensor(readers, f"{p}.scale_shift_table")
        weights[f"{p}.scale_shift_table"] = sst.astype(np.float32)

    # Final output
    weights["scale_shift_table"] = _load_tensor(
        readers, "scale_shift_table").astype(np.float32)
    weights["proj_out.weight"] = _t("proj_out.weight")
    b = _maybe_f("proj_out.bias")
    if b is not None:
        weights["proj_out.bias"] = b

    # Patch embedding (loaded but used externally, not in the TRT engine)
    if _has_tensor(readers, "patch_embedding.weight"):
        weights["patch_embedding.weight"] = _load_tensor(
            readers, "patch_embedding.weight").astype(np.float32)
    if _has_tensor(readers, "patch_embedding.bias"):
        weights["patch_embedding.bias"] = _load_tensor(
            readers, "patch_embedding.bias").astype(np.float32)

    # Timestep/text embedder weights (used externally)
    # Map canonical internal names -> possible safetensors names
    _embedder_aliases = {
        "condition_embedder.time_embedding.0.weight": [
            "condition_embedder.time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_1.weight",
        ],
        "condition_embedder.time_embedding.0.bias": [
            "condition_embedder.time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.bias",
        ],
        "condition_embedder.time_embedding.2.weight": [
            "condition_embedder.time_embedding.2.weight",
            "condition_embedder.time_embedder.linear_2.weight",
        ],
        "condition_embedder.time_embedding.2.bias": [
            "condition_embedder.time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.bias",
        ],
        "condition_embedder.text_embedding.weight": [
            "condition_embedder.text_embedding.weight",
            "condition_embedder.text_embedder.linear_1.weight",
        ],
        "condition_embedder.text_embedding.bias": [
            "condition_embedder.text_embedding.bias",
            "condition_embedder.text_embedder.linear_1.bias",
        ],
        "condition_embedder.text_embedding_2.weight": [
            "condition_embedder.text_embedding_2.weight",
            "condition_embedder.text_embedder.linear_2.weight",
        ],
        "condition_embedder.text_embedding_2.bias": [
            "condition_embedder.text_embedding_2.bias",
            "condition_embedder.text_embedder.linear_2.bias",
        ],
    }
    for key in ("condition_embedder.time_proj.weight",
                "condition_embedder.time_proj.bias"):
        if _has_tensor(readers, key):
            w = _load_tensor(readers, key).astype(np.float32)
            if w.ndim == 2:
                weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)
            else:
                weights[key] = w

    for canonical, aliases in _embedder_aliases.items():
        for alias in aliases:
            if _has_tensor(readers, alias):
                w = _load_tensor(readers, alias).astype(np.float32)
                if w.ndim == 2:
                    weights[canonical] = np.ascontiguousarray(w.T, dtype=np.float32)
                else:
                    weights[canonical] = w
                break

    return weights
