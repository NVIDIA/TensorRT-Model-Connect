# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternViT vision encoder builder for InternVL3.

Builds the InternViT-300M-448px vision encoder as a single TRT engine:
  1. 2D Patch Embedding (Conv2D: [C, H, W] -> [num_patches, embed_dim])
  2. Learned position embedding (cls_token + absolute position)
  3. N ViT transformer blocks (bidirectional self-attention, no causal mask):
     - LayerNorm + multi-head self-attention (with layer scale)
     - LayerNorm + GELU MLP (with layer scale)
  4. Pixel-shuffle downsampling (2x2 concat: 1024 patches -> 256 tokens)
  5. MLP projector: LayerNorm + Linear + GELU + Linear (4*vit_dim -> llm_hidden)

Weight prefix variants (auto-detected):
  - InternVL3-8B-hf:  vision_tower.encoder.layer.{i}.*
                       vision_tower.embeddings.{cls_token,position_embeddings,patch_embeddings}
                       multi_modal_projector.{layer_norm,linear_1,linear_2}

Engine I/O (fixed shapes for a specific image size):
  Input:  pixel_values [C, fixed_H, fixed_W] float32
  Output: image_features [num_output_tokens, llm_hidden_size] float32

With fixed_image_size=448, patch_size=14, downsample_ratio=0.5:
  num_patches = (448/14)^2 = 1024
  num_output_tokens = 1024 * 0.5^2 = 256
  Input: [3, 448, 448], Output: [256, 3584]
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .utils import create_builder_context

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def _get_weight(weights: 'WeightDict', *keys: str):
    """Return the first non-None weight found among the given keys."""
    for k in keys:
        v = weights.get(k)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Weight prefix detection
# ---------------------------------------------------------------------------

def _detect_vit_prefix(weights: 'WeightDict') -> str:
    """Detect ViT layer weight prefix from available keys.

    Returns the prefix without the layer index, e.g.:
      "vision_tower.encoder.layer" or "visual.encoder.layers"
    """
    candidates = [
        "vision_tower.encoder.layer",       # InternVL3-8B-hf
        "visual.encoder.layers",             # Older InternVL
        "visual.blocks",                     # Alternative
    ]
    for prefix in candidates:
        if weights.get(f"{prefix}.0.layernorm_before.weight") is not None:
            return prefix
        if weights.get(f"{prefix}.0.ln1.weight") is not None:
            return prefix
    raise RuntimeError(
        f"Cannot detect InternViT layer prefix. "
        f"Tried: {candidates}")


def _detect_ln_style(weights: 'WeightDict', layer_prefix: str) -> str:
    """Detect LayerNorm naming: 'hf' (layernorm_before/after) or 'legacy' (ln1/ln2)."""
    if weights.get(f"{layer_prefix}.0.layernorm_before.weight") is not None:
        return "hf"
    if weights.get(f"{layer_prefix}.0.ln1.weight") is not None:
        return "legacy"
    raise RuntimeError("Cannot detect LayerNorm naming style")


def _detect_attn_style(weights: 'WeightDict', layer_prefix: str) -> str:
    """Detect attention naming: 'hf' (attention.q_proj) or 'legacy' (attn.q_proj/attn.qkv)."""
    if weights.get(f"{layer_prefix}.0.attention.q_proj.weight") is not None:
        return "hf"
    if weights.get(f"{layer_prefix}.0.attn.q_proj.weight") is not None:
        return "legacy_sep"
    if weights.get(f"{layer_prefix}.0.attn.qkv.weight") is not None:
        return "legacy_fused"
    raise RuntimeError("Cannot detect attention naming style")


# ---------------------------------------------------------------------------
# ViT layer builders
# ---------------------------------------------------------------------------

def _add_vit_attention(
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    weights: 'WeightDict',
    prefix: str,
    embed_dim: int,
    num_heads: int,
    seq_len: int,
    attn_style: str,
) -> trt.ITensor:
    """Add bidirectional self-attention for one InternViT layer."""

    if attn_style == "hf":
        # InternVL3-8B-hf: attention.q_proj, attention.k_proj, attention.v_proj, attention.projection_layer
        w_q = weights[f"{prefix}.attention.q_proj.weight"].astype(np.float32).T.copy()
        w_k = weights[f"{prefix}.attention.k_proj.weight"].astype(np.float32).T.copy()
        w_v = weights[f"{prefix}.attention.v_proj.weight"].astype(np.float32).T.copy()
        q_bias = weights.get(f"{prefix}.attention.q_proj.bias")
        k_bias = weights.get(f"{prefix}.attention.k_proj.bias")
        v_bias = weights.get(f"{prefix}.attention.v_proj.bias")

        # Output projection: "projection_layer" in HF naming
        w_o = weights[f"{prefix}.attention.projection_layer.weight"].astype(np.float32).T.copy()
        o_bias = weights.get(f"{prefix}.attention.projection_layer.bias")
    elif attn_style == "legacy_sep":
        w_q = weights[f"{prefix}.attn.q_proj.weight"].astype(np.float32).T.copy()
        w_k = weights[f"{prefix}.attn.k_proj.weight"].astype(np.float32).T.copy()
        w_v = weights[f"{prefix}.attn.v_proj.weight"].astype(np.float32).T.copy()
        q_bias = weights.get(f"{prefix}.attn.q_proj.bias")
        k_bias = weights.get(f"{prefix}.attn.k_proj.bias")
        v_bias = weights.get(f"{prefix}.attn.v_proj.bias")
        w_o = weights[f"{prefix}.attn.proj.weight"].astype(np.float32).T.copy()
        o_bias = weights.get(f"{prefix}.attn.proj.bias")
    else:  # legacy_fused
        qkv_w = weights[f"{prefix}.attn.qkv.weight"].astype(np.float32)
        w_q = qkv_w[:embed_dim, :].T.copy()
        w_k = qkv_w[embed_dim:2*embed_dim, :].T.copy()
        w_v = qkv_w[2*embed_dim:, :].T.copy()
        qkv_b = weights.get(f"{prefix}.attn.qkv.bias")
        if qkv_b is not None:
            qkv_b = qkv_b.astype(np.float32)
            q_bias = qkv_b[:embed_dim].copy()
            k_bias = qkv_b[embed_dim:2*embed_dim].copy()
            v_bias = qkv_b[2*embed_dim:].copy()
        else:
            q_bias, k_bias, v_bias = None, None, None
        w_o = weights[f"{prefix}.attn.proj.weight"].astype(np.float32).T.copy()
        o_bias = weights.get(f"{prefix}.attn.proj.bias")

    return graph_ops.add_self_attention_block(
        network, normed,
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o,
        hidden_size=embed_dim, num_heads=num_heads,
        seq_length=seq_len,
        q_bias=q_bias.astype(np.float32) if q_bias is not None else None,
        k_bias=k_bias.astype(np.float32) if k_bias is not None else None,
        v_bias=v_bias.astype(np.float32) if v_bias is not None else None,
        o_bias=o_bias.astype(np.float32) if o_bias is not None else None,
    )


def _add_gelu_mlp(
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    weights: 'WeightDict',
    prefix: str,
    embed_dim: int,
    mlp_hidden: int,
) -> trt.ITensor:
    """Add GELU MLP for one InternViT layer (fc1 + GELU + fc2)."""
    fc1_w = weights[f"{prefix}.mlp.fc1.weight"].astype(np.float32).T.copy()
    fc1_b = weights.get(f"{prefix}.mlp.fc1.bias")
    fc2_w = weights[f"{prefix}.mlp.fc2.weight"].astype(np.float32).T.copy()
    fc2_b = weights.get(f"{prefix}.mlp.fc2.bias")

    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed, embed_dim, mlp_hidden, fc1_w)
    if fc1_b is not None:
        fc1 = graph_ops.add_bias_sum(
            network, fc1, mlp_hidden, fc1_b.astype(np.float32))
    activated = graph_ops.add_gelu_new(network, fc1)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_hidden, embed_dim, fc2_w)
    if fc2_b is not None:
        fc2 = graph_ops.add_bias_sum(
            network, fc2, embed_dim, fc2_b.astype(np.float32))
    return fc2


def _add_layer_scale(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    scale_weights: np.ndarray,
    embed_dim: int,
) -> trt.ITensor:
    """Apply layer scale: x * lambda (element-wise per-channel)."""
    scale_const = graph_ops.add_constant(
        network, (1, embed_dim),
        scale_weights.astype(np.float32).reshape(1, embed_dim))
    scaled = network.add_elementwise(
        inp, scale_const, trt.ElementWiseOperation.PROD)
    return scaled.get_output(0)


# ---------------------------------------------------------------------------
# Pixel-shuffle downsampling
# ---------------------------------------------------------------------------

def _add_pixel_shuffle_downsample(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    grid_h: int,
    grid_w: int,
    embed_dim: int,
    downsample_ratio: float,
) -> tuple[trt.ITensor, int, int]:
    """Pixel-shuffle downsampling: concat adjacent patches.

    E.g., with downsample_ratio=0.5 and grid 32x32:
      - Scale factor = 1/0.5 = 2
      - Group 2x2 adjacent patches -> concat -> [H/2, W/2, 4*embed_dim]
      - Output: [256, 4096]

    Returns: (output_tensor, num_output_tokens, concat_dim)
    """
    scale = int(1.0 / downsample_ratio)
    new_h = grid_h // scale
    new_w = grid_w // scale
    num_output = new_h * new_w
    concat_dim = embed_dim * scale * scale

    # [num_patches, embed_dim] -> [grid_h, grid_w, embed_dim]
    reshape1 = network.add_shuffle(hidden)
    reshape1.reshape_dims = (grid_h, grid_w, embed_dim)

    # [grid_h, grid_w, embed_dim] -> [new_h, scale, new_w, scale, embed_dim]
    reshape2 = network.add_shuffle(reshape1.get_output(0))
    reshape2.reshape_dims = (new_h, scale, new_w, scale, embed_dim)

    # Transpose to [new_h, new_w, scale, scale, embed_dim]
    transpose = network.add_shuffle(reshape2.get_output(0))
    transpose.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
    transpose.reshape_dims = (num_output, concat_dim)

    return transpose.get_output(0), num_output, concat_dim


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_internvit_vision_engine(
    top_config: dict,
    vision_config: dict,
    weights: 'WeightDict',
    *,
    fixed_image_size: int = 448,
    verbose: bool = False,
) -> bytes:
    """Build InternViT vision encoder + MLP projector as a single TRT engine.

    Args:
        top_config: Full model config dict (for downsample_ratio, etc.).
        vision_config: The "vision_config" dict from the HF config.json.
        weights: Weight dict containing vision + projector keys.
        fixed_image_size: Image height/width the engine is compiled for.
        verbose: Print detailed logs.

    Returns:
        Serialized TRT engine plan bytes.
    """
    embed_dim = vision_config.get("hidden_size", 1024)
    num_heads = vision_config.get("num_attention_heads", 16)
    num_layers = vision_config.get("num_hidden_layers", 24)
    mlp_hidden = vision_config.get("intermediate_size", embed_dim * 4)
    in_channels = vision_config.get("num_channels", 3)
    patch_size_raw = vision_config.get("patch_size", 14)
    patch_size = patch_size_raw[0] if isinstance(patch_size_raw, (list, tuple)) else patch_size_raw
    eps_val = vision_config.get("layer_norm_eps", 1e-6)
    downsample_ratio = top_config.get("downsample_ratio", 0.5)
    select_layer = top_config.get("vision_feature_layer", -1)

    grid_h = fixed_image_size // patch_size
    grid_w = fixed_image_size // patch_size
    num_patches = grid_h * grid_w

    # Detect weight naming style
    layer_prefix = _detect_vit_prefix(weights)
    ln_style = _detect_ln_style(weights, layer_prefix)
    attn_style = _detect_attn_style(weights, layer_prefix)

    # Detect CLS token
    has_cls = (weights.get("vision_tower.embeddings.cls_token") is not None
               or weights.get("visual.cls_token") is not None)
    seq_len = num_patches + 1 if has_cls else num_patches

    # Detect layer scale (lambda_1, lambda_2)
    has_layer_scale = (weights.get(f"{layer_prefix}.0.lambda_1") is not None)

    # Resolve select_layer
    actual_select_layer = (select_layer if select_layer >= 0
                           else num_layers + select_layer)

    # Compute output dimensions
    scale = int(1.0 / downsample_ratio)
    num_output_tokens = num_patches // (scale * scale)

    if verbose:
        print(f"[trtmc build] InternViT: image={fixed_image_size}, "
              f"patch={patch_size}, grid={grid_h}x{grid_w}, "
              f"patches={num_patches}, output_tokens={num_output_tokens}, "
              f"embed={embed_dim}, "
              f"layers={num_layers}, select_layer={actual_select_layer}, "
              f"cls_token={has_cls}, layer_scale={has_layer_scale}, "
              f"downsample_ratio={downsample_ratio}, "
              f"attn_style={attn_style}, ln_style={ln_style}",
              file=sys.stderr)

    builder_context = create_builder_context(
        verbose=verbose,
        workspace_bytes=2 << 30,
    )
    trt_builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([eps_val], dtype=np.float32))

    # ---------------------------------------------------------------
    # Input: pixel_values [C, H, W]
    # ---------------------------------------------------------------
    pixel_values = network.add_input(
        "pixel_values", trt.float32,
        (in_channels, fixed_image_size, fixed_image_size))

    # ---------------------------------------------------------------
    # Stage 1: 2D Patch Embedding (Conv2D)
    # [C, H, W] -> [num_patches, embed_dim]
    # ---------------------------------------------------------------
    patch_embed_w = _get_weight(
        weights,
        "vision_tower.embeddings.patch_embeddings.projection.weight",
        "visual.patch_embed.proj.weight")
    patch_embed_b = _get_weight(
        weights,
        "vision_tower.embeddings.patch_embeddings.projection.bias",
        "visual.patch_embed.proj.bias")

    if patch_embed_w is None:
        raise RuntimeError("Missing InternViT patch embedding weight")

    # Reshape input: [C, H, W] -> [1, C, H, W]
    reshape_in = network.add_shuffle(pixel_values)
    reshape_in.reshape_dims = (1, in_channels, fixed_image_size, fixed_image_size)

    conv_w = trt.Weights(np.ascontiguousarray(
        patch_embed_w.astype(np.float32)))
    conv_b = trt.Weights()
    if patch_embed_b is not None:
        conv_b = trt.Weights(np.ascontiguousarray(
            patch_embed_b.astype(np.float32)))

    conv = network.add_convolution_nd(
        reshape_in.get_output(0),
        num_output_maps=embed_dim,
        kernel_shape=(patch_size, patch_size),
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = (patch_size, patch_size)

    # [1, embed_dim, H', W'] -> [num_patches, embed_dim]
    reshape_conv = network.add_shuffle(conv.get_output(0))
    reshape_conv.first_transpose = trt.Permutation([0, 2, 3, 1])
    reshape_conv.reshape_dims = (num_patches, embed_dim)

    hidden = reshape_conv.get_output(0)

    # ---------------------------------------------------------------
    # Stage 2: CLS token + Position Embedding
    # ---------------------------------------------------------------
    if has_cls:
        cls_token_np = _get_weight(
            weights, "vision_tower.embeddings.cls_token", "visual.cls_token")
        if cls_token_np is not None:
            cls_const = graph_ops.add_constant(
                network, (1, embed_dim),
                cls_token_np.astype(np.float32).reshape(1, embed_dim))
            concat = network.add_concatenation([cls_const, hidden])
            concat.axis = 0
            hidden = concat.get_output(0)
        else:
            has_cls = False
            seq_len = num_patches

    pos_embed_w = _get_weight(
        weights,
        "vision_tower.embeddings.position_embeddings",
        "visual.pos_embed",
        "visual.position_embedding.weight")

    if pos_embed_w is not None:
        pos_embed_np = pos_embed_w.astype(np.float32)
        if pos_embed_np.ndim == 3:
            pos_embed_np = pos_embed_np.reshape(-1, embed_dim)
        # Truncate or pad to match seq_len
        if pos_embed_np.shape[0] > seq_len:
            pos_embed_np = pos_embed_np[:seq_len]
        elif pos_embed_np.shape[0] < seq_len:
            padded = np.zeros((seq_len, embed_dim), dtype=np.float32)
            padded[:pos_embed_np.shape[0]] = pos_embed_np
            pos_embed_np = padded

        pos_const = graph_ops.add_constant(
            network, (seq_len, embed_dim), pos_embed_np)
        pos_add = network.add_elementwise(
            hidden, pos_const, trt.ElementWiseOperation.SUM)
        hidden = pos_add.get_output(0)

    # ---------------------------------------------------------------
    # Stage 3: ViT Transformer blocks
    # ---------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"{layer_prefix}.{layer_idx}"

        # Pre-attention LayerNorm
        if ln_style == "hf":
            ln1_gamma = weights[f"{prefix}.layernorm_before.weight"].astype(np.float32)
            ln1_beta_raw = weights.get(f"{prefix}.layernorm_before.bias")
        else:
            ln1_gamma = weights[f"{prefix}.ln1.weight"].astype(np.float32)
            ln1_beta_raw = weights.get(f"{prefix}.ln1.bias")
        ln1_beta = (ln1_beta_raw.astype(np.float32) if ln1_beta_raw is not None
                     else np.zeros(embed_dim, dtype=np.float32))
        normed = graph_ops.add_layer_norm(
            network, hidden, embed_dim, ln1_gamma, ln1_beta, eps_tensor)

        # Self-attention (bidirectional, no RoPE)
        attn_out = _add_vit_attention(
            network, normed, weights, prefix,
            embed_dim, num_heads, seq_len, attn_style)

        # Layer scale on attention output
        if has_layer_scale:
            lambda_1 = weights.get(f"{prefix}.lambda_1")
            if lambda_1 is not None:
                attn_out = _add_layer_scale(
                    network, attn_out, lambda_1, embed_dim)

        # Residual
        res1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)

        # Post-attention LayerNorm
        if ln_style == "hf":
            ln2_gamma = weights[f"{prefix}.layernorm_after.weight"].astype(np.float32)
            ln2_beta_raw = weights.get(f"{prefix}.layernorm_after.bias")
        else:
            ln2_gamma = weights[f"{prefix}.ln2.weight"].astype(np.float32)
            ln2_beta_raw = weights.get(f"{prefix}.ln2.bias")
        ln2_beta = (ln2_beta_raw.astype(np.float32) if ln2_beta_raw is not None
                     else np.zeros(embed_dim, dtype=np.float32))
        normed2 = graph_ops.add_layer_norm(
            network, res1.get_output(0), embed_dim,
            ln2_gamma, ln2_beta, eps_tensor)

        # GELU MLP
        mlp_out = _add_gelu_mlp(
            network, normed2, weights, prefix, embed_dim, mlp_hidden)

        # Layer scale on MLP output
        if has_layer_scale:
            lambda_2 = weights.get(f"{prefix}.lambda_2")
            if lambda_2 is not None:
                mlp_out = _add_layer_scale(
                    network, mlp_out, lambda_2, embed_dim)

        # Residual
        res2 = network.add_elementwise(
            res1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden = res2.get_output(0)

        # Break early at select_layer
        if layer_idx == actual_select_layer:
            break

    # ---------------------------------------------------------------
    # Stage 4: Remove CLS token if present
    # [cls + num_patches, embed_dim] -> [num_patches, embed_dim]
    # ---------------------------------------------------------------
    if has_cls:
        slice_out = network.add_slice(
            hidden,
            start=(1, 0),
            shape=(num_patches, embed_dim),
            stride=(1, 1))
        hidden = slice_out.get_output(0)

    # ---------------------------------------------------------------
    # Stage 5: Pixel-shuffle downsampling
    # [num_patches, embed_dim] -> [num_output_tokens, concat_dim]
    # ---------------------------------------------------------------
    if downsample_ratio < 1.0:
        hidden, _, concat_dim = _add_pixel_shuffle_downsample(
            network, hidden, grid_h, grid_w, embed_dim, downsample_ratio)
    else:
        concat_dim = embed_dim

    # ---------------------------------------------------------------
    # Stage 6: MLP Projector
    # Try multi_modal_projector.* (InternVL3-8B-hf) first, then mlp1.* (legacy)
    # ---------------------------------------------------------------
    proj_ln_w = _get_weight(
        weights, "multi_modal_projector.layer_norm.weight",
        "mlp1.1.weight", "visual.mlp1.1.weight")
    proj_ln_b = _get_weight(
        weights, "multi_modal_projector.layer_norm.bias",
        "mlp1.1.bias", "visual.mlp1.1.bias")

    proj_fc1_w = _get_weight(
        weights, "multi_modal_projector.linear_1.weight",
        "mlp1.0.weight", "visual.mlp1.0.weight")
    proj_fc1_b = _get_weight(
        weights, "multi_modal_projector.linear_1.bias",
        "mlp1.0.bias", "visual.mlp1.0.bias")

    proj_fc2_w = _get_weight(
        weights, "multi_modal_projector.linear_2.weight",
        "mlp1.3.weight", "visual.mlp1.3.weight")
    proj_fc2_b = _get_weight(
        weights, "multi_modal_projector.linear_2.bias",
        "mlp1.3.bias", "visual.mlp1.3.bias")

    if proj_fc1_w is None:
        raise RuntimeError(
            "Missing MLP projector weights "
            "(multi_modal_projector.linear_1.weight or mlp1.0.weight)")

    # Detect projector structure:
    # InternVL3-8B-hf: LayerNorm(concat_dim) -> Linear(concat_dim, llm_hidden) -> GELU -> Linear(llm_hidden, llm_hidden)
    # Legacy: Linear(embed_dim, proj_fc1_out_dim) -> LayerNorm -> GELU -> Linear(proj_fc1_out_dim, llm_hidden)
    proj_fc1_out_dim = proj_fc1_w.shape[0]
    proj_fc1_in_dim = proj_fc1_w.shape[1]

    # If projector has LayerNorm and it matches concat_dim, apply LN first (InternVL3-8B-hf style)
    is_hf_projector = (proj_ln_w is not None and proj_ln_w.shape[0] == concat_dim
                       and proj_fc1_in_dim == concat_dim)

    if is_hf_projector:
        # InternVL3-8B-hf: LayerNorm -> fc1 -> GELU -> fc2
        proj_ln_beta = (proj_ln_b.astype(np.float32)
                        if proj_ln_b is not None
                        else np.zeros(concat_dim, dtype=np.float32))
        hidden = graph_ops.add_layer_norm(
            network, hidden, concat_dim,
            proj_ln_w.astype(np.float32), proj_ln_beta, eps_tensor)

        fc1 = graph_ops.add_matmul_rhs_constant(
            network, hidden, concat_dim, proj_fc1_out_dim,
            proj_fc1_w.astype(np.float32).T.copy())
        if proj_fc1_b is not None:
            fc1 = graph_ops.add_bias_sum(
                network, fc1, proj_fc1_out_dim, proj_fc1_b.astype(np.float32))

        activated = graph_ops.add_gelu_new(network, fc1)

        llm_hidden = proj_fc2_w.shape[0]
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, activated, proj_fc1_out_dim, llm_hidden,
            proj_fc2_w.astype(np.float32).T.copy())
        if proj_fc2_b is not None:
            fc2 = graph_ops.add_bias_sum(
                network, fc2, llm_hidden, proj_fc2_b.astype(np.float32))
    else:
        # Legacy: fc1 -> LayerNorm -> GELU -> fc2
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, hidden, concat_dim, proj_fc1_out_dim,
            proj_fc1_w.astype(np.float32).T.copy())
        if proj_fc1_b is not None:
            fc1 = graph_ops.add_bias_sum(
                network, fc1, proj_fc1_out_dim, proj_fc1_b.astype(np.float32))

        if proj_ln_w is not None:
            proj_ln_beta = (proj_ln_b.astype(np.float32)
                            if proj_ln_b is not None
                            else np.zeros(proj_fc1_out_dim, dtype=np.float32))
            fc1 = graph_ops.add_layer_norm(
                network, fc1, proj_fc1_out_dim,
                proj_ln_w.astype(np.float32), proj_ln_beta, eps_tensor)

        activated = graph_ops.add_gelu_new(network, fc1)

        llm_hidden = proj_fc2_w.shape[0]
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, activated, proj_fc1_out_dim, llm_hidden,
            proj_fc2_w.astype(np.float32).T.copy())
        if proj_fc2_b is not None:
            fc2 = graph_ops.add_bias_sum(
                network, fc2, llm_hidden, proj_fc2_b.astype(np.float32))

    # ---------------------------------------------------------------
    # Output: image_features [num_output_tokens, llm_hidden_size]
    # ---------------------------------------------------------------
    fc2.name = "image_features"
    network.mark_output(fc2)

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building InternViT vision TRT engine "
              f"({actual_select_layer + 1} layers, embed={embed_dim}, "
              f"patches={num_patches}, output_tokens={num_output_tokens}, "
              f"llm_hidden={llm_hidden}) ...", file=sys.stderr)

    plan = trt_builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT InternViT vision engine build failed")

    return bytes(plan)
