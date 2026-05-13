"""Vision encoder builder for Phi-4-multimodal.

Builds a ViT vision encoder + image projection adapter as a single TRT engine.

The Phi-4-multimodal vision pipeline:
  1. Patch embedding: Conv2D [C, H, W] -> [num_patches, embed_dim]
  2. Learned position embedding (added to patch embeddings)
  3. N ViT transformer blocks: LayerNorm + self-attention + LayerNorm + GELU MLP
  4. Image projection: LayerNorm -> Linear -> GELU -> Linear
     projects [num_patches, embed_dim] -> [num_patches, text_hidden_size]

Engine I/O (fixed shapes for a specific image size):
  Input:  pixel_values [C, fixed_H, fixed_W] float32
  Output: image_features [num_patches, text_hidden_size] float32
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


def build_phi4mm_vision_engine(
    vision_config: dict,
    weights: WeightDict,
    *,
    text_hidden_size: int,
    fixed_image_size: int = 336,
    verbose: bool = False,
) -> bytes:
    """Build Phi-4-multimodal vision encoder TRT engine.

    Args:
        vision_config: The vision encoder config dict (from img_processor
            or vision_config in HF config.json).
        weights: Full weight dict (only vision_embed_tokens.* keys used).
        text_hidden_size: Text decoder hidden size (projection target dim).
        fixed_image_size: Image size the engine is compiled for.
        verbose: Print detailed logs.

    Returns:
        Serialized TRT engine plan bytes.
    """
    embed_dim = vision_config.get("hidden_size", vision_config.get(
        "embed_dim", 1024))
    num_heads = vision_config.get("num_attention_heads", vision_config.get(
        "num_heads", 16))
    num_layers = vision_config.get("num_hidden_layers", vision_config.get(
        "depth", 24))
    mlp_hidden = vision_config.get("intermediate_size", embed_dim * 4)
    in_channels = vision_config.get("num_channels", 3)
    patch_size = vision_config.get("patch_size", 14)
    eps_val = vision_config.get("layer_norm_eps", 1e-6)

    grid_h = fixed_image_size // patch_size
    grid_w = fixed_image_size // patch_size
    num_patches = grid_h * grid_w

    if verbose:
        print(f"[trtmc-build] Phi4MM Vision: image={fixed_image_size}, "
              f"patch={patch_size}, grid={grid_h}x{grid_w}, "
              f"patches={num_patches}, embed={embed_dim}, "
              f"text_hidden={text_hidden_size}", file=sys.stderr)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([eps_val], dtype=np.float32))

    # ---------------------------------------------------------------
    # Input: pixel_values [C, H, W]
    # ---------------------------------------------------------------
    pixel_values = network.add_input(
        "pixel_values", trt.float32,
        (in_channels, fixed_image_size, fixed_image_size))

    # ---------------------------------------------------------------
    # Stage 1: Patch Embedding (Conv2D)
    # [C, H, W] -> [num_patches, embed_dim]
    # ---------------------------------------------------------------
    # We use the patch_embed weight if available
    vit_prefix = "model.vision_embed_tokens.img_processor"
    patch_embed_w = weights.get(
        f"{vit_prefix}.vision_model.embeddings.patch_embedding.weight")
    if patch_embed_w is None:
        patch_embed_w = weights.get(
            f"{vit_prefix}.embeddings.patch_embedding.weight")
    if patch_embed_w is None:
        # Try direct naming
        patch_embed_w = weights.get(
            "model.vision_embed_tokens.img_processor.patch_embed.proj.weight")

    if patch_embed_w is None:
        raise RuntimeError(
            "Missing vision patch embedding weight. Available keys: "
            + ", ".join(k for k in weights if "patch" in k.lower())[:200])

    patch_embed_w = patch_embed_w.astype(np.float32)

    # Conv2D: [embed_dim, C, patch_size, patch_size] -> stride=patch_size
    if patch_embed_w.shape == (embed_dim, in_channels, patch_size, patch_size):
        conv_w = patch_embed_w
    elif patch_embed_w.ndim == 2:
        # Linear patch embed: reshape [embed_dim, C*patch*patch]
        conv_w = patch_embed_w.reshape(
            embed_dim, in_channels, patch_size, patch_size)
    else:
        conv_w = patch_embed_w

    conv_weights = trt.Weights(np.ascontiguousarray(conv_w))

    # Look for bias
    patch_embed_b = weights.get(
        f"{vit_prefix}.vision_model.embeddings.patch_embedding.bias")
    if patch_embed_b is None:
        patch_embed_b = weights.get(
            f"{vit_prefix}.embeddings.patch_embedding.bias")
    if patch_embed_b is None:
        patch_embed_b = weights.get(
            "model.vision_embed_tokens.img_processor.patch_embed.proj.bias")

    conv_bias = trt.Weights(
        np.ascontiguousarray(patch_embed_b.astype(np.float32))
    ) if patch_embed_b is not None else trt.Weights()

    conv_layer = network.add_convolution_nd(
        pixel_values, embed_dim, (patch_size, patch_size),
        conv_weights, conv_bias)
    conv_layer.stride_nd = (patch_size, patch_size)
    # Output: [embed_dim, grid_h, grid_w]

    # Reshape to [num_patches, embed_dim]: transpose [embed_dim, grid_h*grid_w]
    # -> [grid_h*grid_w, embed_dim]
    conv_out = conv_layer.get_output(0)
    reshape1 = network.add_shuffle(conv_out)
    reshape1.reshape_dims = (embed_dim, num_patches)

    # Transpose [embed_dim, num_patches] -> [num_patches, embed_dim]
    reshape1.second_transpose = (1, 0)
    hidden = reshape1.get_output(0)

    # ---------------------------------------------------------------
    # Stage 2: Add learned position embedding
    # ---------------------------------------------------------------
    # Look for CLS token + position embeddings
    cls_token = weights.get(
        f"{vit_prefix}.vision_model.embeddings.class_embedding")
    if cls_token is None:
        cls_token = weights.get(
            f"{vit_prefix}.embeddings.class_embedding")

    pos_embed_w = weights.get(
        f"{vit_prefix}.vision_model.embeddings.position_embedding.weight")
    if pos_embed_w is None:
        pos_embed_w = weights.get(
            f"{vit_prefix}.embeddings.position_embedding.weight")

    if pos_embed_w is not None:
        pos_embed_w = pos_embed_w.astype(np.float32)

        if cls_token is not None:
            # Has CLS token: prepend it, then add position embeddings
            # CLS: [1, embed_dim], pos_embed: [1 + num_patches, embed_dim]
            cls_const = graph_ops.add_constant(
                network, (1, embed_dim), cls_token.astype(np.float32))
            # Concatenate CLS + patches: [1+num_patches, embed_dim]
            concat = network.add_concatenation([cls_const, hidden])
            concat.axis = 0

            # Position embed should cover CLS + patches
            n_pos = min(pos_embed_w.shape[0], 1 + num_patches)
            pos_slice = pos_embed_w[:n_pos, :embed_dim]
            if pos_slice.shape[0] < 1 + num_patches:
                # Pad if needed
                pad = np.zeros(
                    (1 + num_patches - pos_slice.shape[0], embed_dim),
                    dtype=np.float32)
                pos_slice = np.concatenate([pos_slice, pad], axis=0)

            pos_const = graph_ops.add_constant(
                network, (1 + num_patches, embed_dim), pos_slice)
            pos_add = network.add_elementwise(
                concat.get_output(0), pos_const,
                trt.ElementWiseOperation.SUM)
            hidden = pos_add.get_output(0)
            # Update num_patches to include CLS
            seq_len = 1 + num_patches
        else:
            # No CLS token: just add position embeddings to patches
            pos_slice = pos_embed_w[:num_patches, :embed_dim]
            if pos_slice.shape[0] < num_patches:
                pad = np.zeros(
                    (num_patches - pos_slice.shape[0], embed_dim),
                    dtype=np.float32)
                pos_slice = np.concatenate([pos_slice, pad], axis=0)

            pos_const = graph_ops.add_constant(
                network, (num_patches, embed_dim), pos_slice)
            pos_add = network.add_elementwise(
                hidden, pos_const, trt.ElementWiseOperation.SUM)
            hidden = pos_add.get_output(0)
            seq_len = num_patches
    else:
        seq_len = num_patches

    # ---------------------------------------------------------------
    # Stage 3: ViT Transformer blocks
    # LayerNorm + self-attention + LayerNorm + MLP (GELU)
    # ---------------------------------------------------------------
    for layer_idx in range(num_layers):
        # Try multiple naming conventions
        layer_prefix = None
        for candidate in [
            f"{vit_prefix}.vision_model.encoder.layers.{layer_idx}",
            f"{vit_prefix}.encoder.layers.{layer_idx}",
            f"{vit_prefix}.blocks.{layer_idx}",
        ]:
            ln1_key = f"{candidate}.layer_norm1.weight"
            if ln1_key not in weights:
                ln1_key = f"{candidate}.norm1.weight"
            if ln1_key in weights:
                layer_prefix = candidate
                break

        if layer_prefix is None:
            raise RuntimeError(
                f"Cannot find layer {layer_idx} weights. "
                f"Available keys: "
                + ", ".join(k for k in weights
                            if f".{layer_idx}." in k)[:300])

        # Pre-attention LayerNorm
        ln1_w = _try_load(weights, layer_prefix, "layer_norm1.weight",
                          "norm1.weight")
        ln1_b = _try_load(weights, layer_prefix, "layer_norm1.bias",
                          "norm1.bias")
        if ln1_w is None:
            ln1_w = np.ones(embed_dim, dtype=np.float32)
        if ln1_b is None:
            ln1_b = np.zeros(embed_dim, dtype=np.float32)

        normed = graph_ops.add_layer_norm(
            network, hidden, embed_dim,
            ln1_w.astype(np.float32),
            ln1_b.astype(np.float32),
            eps_tensor)

        # Self-attention (Q, K, V, O projections)
        w_q, w_k, w_v, q_bias, k_bias, v_bias = _load_attn_weights(
            weights, layer_prefix, embed_dim)

        w_o_raw = _try_load(weights, layer_prefix,
                            "self_attn.out_proj.weight",
                            "attn.proj.weight")
        o_bias = _try_load(weights, layer_prefix,
                           "self_attn.out_proj.bias",
                           "attn.proj.bias")

        w_o_np = (w_o_raw.astype(np.float32).T.copy()
                  if w_o_raw is not None
                  else np.zeros((embed_dim, embed_dim), dtype=np.float32))
        o_bias_np = (o_bias.astype(np.float32)
                     if o_bias is not None else None)

        attn_out = graph_ops.add_self_attention_block_with_rope(
            network, normed,
            w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o_np,
            hidden_size=embed_dim, num_heads=num_heads,
            seq_length=seq_len,
            cos_table=np.ones((seq_len, embed_dim), dtype=np.float32),
            sin_table=np.zeros((seq_len, embed_dim), dtype=np.float32),
            q_bias=q_bias, k_bias=k_bias, v_bias=v_bias,
            o_bias=o_bias_np)

        # Residual
        res1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)

        # Post-attention LayerNorm
        ln2_w = _try_load(weights, layer_prefix, "layer_norm2.weight",
                          "norm2.weight")
        ln2_b = _try_load(weights, layer_prefix, "layer_norm2.bias",
                          "norm2.bias")
        if ln2_w is None:
            ln2_w = np.ones(embed_dim, dtype=np.float32)
        if ln2_b is None:
            ln2_b = np.zeros(embed_dim, dtype=np.float32)

        normed2 = graph_ops.add_layer_norm(
            network, res1.get_output(0), embed_dim,
            ln2_w.astype(np.float32),
            ln2_b.astype(np.float32),
            eps_tensor)

        # MLP: fc1 -> GELU -> fc2
        fc1_w, fc1_b, fc2_w, fc2_b = _load_mlp_weights(
            weights, layer_prefix, embed_dim, mlp_hidden)

        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, embed_dim, mlp_hidden,
            fc1_w.astype(np.float32).T.copy())
        if fc1_b is not None:
            fc1 = graph_ops.add_bias_sum(
                network, fc1, mlp_hidden, fc1_b.astype(np.float32))

        activated = graph_ops.add_gelu_new(network, fc1)

        fc2 = graph_ops.add_matmul_rhs_constant(
            network, activated, mlp_hidden, embed_dim,
            fc2_w.astype(np.float32).T.copy())
        if fc2_b is not None:
            fc2 = graph_ops.add_bias_sum(
                network, fc2, embed_dim, fc2_b.astype(np.float32))

        # Residual
        res2 = network.add_elementwise(
            res1.get_output(0), fc2, trt.ElementWiseOperation.SUM)
        hidden = res2.get_output(0)

    # ---------------------------------------------------------------
    # Stage 4: Post-encoder LayerNorm (if present)
    # ---------------------------------------------------------------
    post_ln_w = _try_load_direct(weights,
                                 f"{vit_prefix}.vision_model.post_layernorm.weight",
                                 f"{vit_prefix}.post_layernorm.weight",
                                 f"{vit_prefix}.norm.weight")
    post_ln_b = _try_load_direct(weights,
                                 f"{vit_prefix}.vision_model.post_layernorm.bias",
                                 f"{vit_prefix}.post_layernorm.bias",
                                 f"{vit_prefix}.norm.bias")
    if post_ln_w is not None:
        if post_ln_b is None:
            post_ln_b = np.zeros(embed_dim, dtype=np.float32)
        hidden = graph_ops.add_layer_norm(
            network, hidden, embed_dim,
            post_ln_w.astype(np.float32),
            post_ln_b.astype(np.float32),
            eps_tensor)

    # If we added a CLS token, remove it before projection
    if cls_token is not None:
        # Slice off CLS: [1+num_patches, embed_dim] -> [num_patches, embed_dim]
        # Use gather to select indices 1..num_patches
        indices = np.arange(1, 1 + num_patches, dtype=np.int32)
        # Need int32 for gather index
        idx_w = trt.Weights(np.ascontiguousarray(indices))
        idx_layer = network.add_constant((num_patches,), idx_w)
        idx_cast = network.add_cast(idx_layer.get_output(0), trt.int32)
        gather = network.add_gather(hidden, idx_cast.get_output(0), 0)
        hidden = gather.get_output(0)
        seq_len = num_patches

    # ---------------------------------------------------------------
    # Stage 5: Image Projection
    # Linear -> GELU -> Linear to project from embed_dim to text_hidden_size
    # ---------------------------------------------------------------
    proj_prefix = "model.vision_embed_tokens.img_projection"

    # Try loading projection weights with multiple naming conventions
    proj_w1 = _try_load_direct(weights,
                                f"{proj_prefix}.0.weight",
                                f"{proj_prefix}.linear_1.weight")
    proj_b1 = _try_load_direct(weights,
                                f"{proj_prefix}.0.bias",
                                f"{proj_prefix}.linear_1.bias")
    proj_w2 = _try_load_direct(weights,
                                f"{proj_prefix}.2.weight",
                                f"{proj_prefix}.linear_2.weight")
    proj_b2 = _try_load_direct(weights,
                                f"{proj_prefix}.2.bias",
                                f"{proj_prefix}.linear_2.bias")

    if proj_w1 is not None and proj_w2 is not None:
        proj_hidden = proj_w1.shape[0]

        proj1 = graph_ops.add_matmul_rhs_constant(
            network, hidden, embed_dim, proj_hidden,
            proj_w1.astype(np.float32).T.copy())
        if proj_b1 is not None:
            proj1 = graph_ops.add_bias_sum(
                network, proj1, proj_hidden, proj_b1.astype(np.float32))

        proj1_act = graph_ops.add_gelu_new(network, proj1)

        output_dim = proj_w2.shape[0]
        proj2 = graph_ops.add_matmul_rhs_constant(
            network, proj1_act, proj_hidden, output_dim,
            proj_w2.astype(np.float32).T.copy())
        if proj_b2 is not None:
            proj2 = graph_ops.add_bias_sum(
                network, proj2, output_dim, proj_b2.astype(np.float32))

        hidden = proj2
        output_feature_dim = output_dim
    else:
        output_feature_dim = embed_dim

    # ---------------------------------------------------------------
    # Output: image_features [num_patches, output_dim]
    # ---------------------------------------------------------------
    hidden.name = "image_features"
    network.mark_output(hidden)

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------
    if verbose:
        print(f"[trtmc-build] Building Phi4MM vision TRT engine "
              f"({num_layers} layers, embed={embed_dim}, "
              f"patches={num_patches}, "
              f"output_dim={output_feature_dim}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Phi4MM vision engine build failed")

    return bytes(plan)


# ---------------------------------------------------------------------------
# Weight loading helpers with flexible key naming
# ---------------------------------------------------------------------------

def _try_load(weights: WeightDict, prefix: str,
              *suffixes: str) -> np.ndarray | None:
    """Try loading a weight with multiple suffix candidates."""
    for suffix in suffixes:
        key = f"{prefix}.{suffix}"
        if key in weights:
            return weights[key]
    return None


def _try_load_direct(weights: WeightDict,
                     *keys: str) -> np.ndarray | None:
    """Try loading a weight with multiple full key candidates."""
    for key in keys:
        if key in weights:
            return weights[key]
    return None


def _load_attn_weights(
    weights: WeightDict, prefix: str, embed_dim: int,
) -> tuple:
    """Load attention Q/K/V weights with multiple naming conventions.

    Returns: (w_q, w_k, w_v, q_bias, k_bias, v_bias)
    All weights are [in, out] format ready for TRT matmul.
    """
    # Try separate Q/K/V projections first
    w_q = _try_load(weights, prefix,
                    "self_attn.q_proj.weight",
                    "attn.q_proj.weight")
    w_k = _try_load(weights, prefix,
                    "self_attn.k_proj.weight",
                    "attn.k_proj.weight")
    w_v = _try_load(weights, prefix,
                    "self_attn.v_proj.weight",
                    "attn.v_proj.weight")

    if w_q is not None and w_k is not None and w_v is not None:
        w_q = w_q.astype(np.float32).T.copy()
        w_k = w_k.astype(np.float32).T.copy()
        w_v = w_v.astype(np.float32).T.copy()
    else:
        # Try fused QKV
        qkv_w = _try_load(weights, prefix,
                          "self_attn.qkv.weight",
                          "attn.qkv.weight")
        if qkv_w is not None:
            qkv_w = qkv_w.astype(np.float32)
            w_q = qkv_w[:embed_dim, :].T.copy()
            w_k = qkv_w[embed_dim:2*embed_dim, :].T.copy()
            w_v = qkv_w[2*embed_dim:, :].T.copy()
        else:
            # Try in_proj_weight (single fused weight)
            in_proj = _try_load(weights, prefix,
                                "self_attn.in_proj_weight",
                                "attn.in_proj_weight")
            if in_proj is not None:
                in_proj = in_proj.astype(np.float32)
                w_q = in_proj[:embed_dim, :].T.copy()
                w_k = in_proj[embed_dim:2*embed_dim, :].T.copy()
                w_v = in_proj[2*embed_dim:, :].T.copy()
            else:
                w_q = np.eye(embed_dim, dtype=np.float32)
                w_k = np.eye(embed_dim, dtype=np.float32)
                w_v = np.eye(embed_dim, dtype=np.float32)

    # Biases
    q_bias = _try_load(weights, prefix,
                       "self_attn.q_proj.bias",
                       "attn.q_proj.bias")
    k_bias = _try_load(weights, prefix,
                       "self_attn.k_proj.bias",
                       "attn.k_proj.bias")
    v_bias = _try_load(weights, prefix,
                       "self_attn.v_proj.bias",
                       "attn.v_proj.bias")

    if q_bias is None:
        # Try fused QKV bias
        qkv_b = _try_load(weights, prefix,
                          "self_attn.qkv.bias",
                          "attn.qkv.bias")
        if qkv_b is not None:
            qkv_b = qkv_b.astype(np.float32)
            q_bias = qkv_b[:embed_dim].copy()
            k_bias = qkv_b[embed_dim:2*embed_dim].copy()
            v_bias = qkv_b[2*embed_dim:].copy()
        else:
            in_proj_b = _try_load(weights, prefix,
                                  "self_attn.in_proj_bias",
                                  "attn.in_proj_bias")
            if in_proj_b is not None:
                in_proj_b = in_proj_b.astype(np.float32)
                q_bias = in_proj_b[:embed_dim].copy()
                k_bias = in_proj_b[embed_dim:2*embed_dim].copy()
                v_bias = in_proj_b[2*embed_dim:].copy()

    if q_bias is not None:
        q_bias = q_bias.astype(np.float32)
    if k_bias is not None:
        k_bias = k_bias.astype(np.float32)
    if v_bias is not None:
        v_bias = v_bias.astype(np.float32)

    return w_q, w_k, w_v, q_bias, k_bias, v_bias


def _load_mlp_weights(
    weights: WeightDict, prefix: str,
    embed_dim: int, mlp_hidden: int,
) -> tuple:
    """Load MLP weights with multiple naming conventions.

    Returns: (fc1_w, fc1_b, fc2_w, fc2_b)
    """
    fc1_w = _try_load(weights, prefix,
                      "mlp.fc1.weight",
                      "mlp.linear_fc1.weight")
    fc1_b = _try_load(weights, prefix,
                      "mlp.fc1.bias",
                      "mlp.linear_fc1.bias")
    fc2_w = _try_load(weights, prefix,
                      "mlp.fc2.weight",
                      "mlp.linear_fc2.weight")
    fc2_b = _try_load(weights, prefix,
                      "mlp.fc2.bias",
                      "mlp.linear_fc2.bias")

    if fc1_w is None:
        raise RuntimeError(
            f"Missing MLP weights at {prefix}. Available keys: "
            + ", ".join(k for k in weights if prefix in k)[:300])

    return fc1_w, fc1_b, fc2_w, fc2_b
