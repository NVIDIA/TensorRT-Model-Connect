# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete vision encoder builder for Qwen2.5-VL.

Builds the FULL vision pipeline as a single TRT engine, exactly matching HF's
Qwen2_5_VisionTransformerPretrainedModel forward pass:
  1. 3D Patch Embedding (conv: [T*C, H, W] -> [num_patches, embed_dim])
  2. 2D RoPE with spatial merge permutation + window_index reordering
  3. N ViT transformer blocks (full self-attention with RoPE)
  4. Spatial merge: RMSNorm -> view(-1, merged_dim) -> MLP -> reverse reorder

Engine I/O (fixed shapes for a specific image size):
  Input:  pixel_values [T*C, fixed_H, fixed_W] float32
  Output: image_features [num_merged_tokens, text_hidden_size] float32

With fixed_image_size=448, patch_size=14, merge_size=2:
  num_patches = (448/14)^2 = 1024
  num_merged = 1024/4 = 256
  Input: [6, 448, 448], Output: [256, 2048]
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from .. import model as graph_ops
from ..model import create_builder_context

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...weights import WeightDict


# ---------------------------------------------------------------------------
# Vision RoPE + window index (exact port of HF Qwen2.5-VL)
# ---------------------------------------------------------------------------

def _compute_vision_rope_tables(
    grid_h: int,
    grid_w: int,
    embed_dim: int,
    num_heads: int,
    merge_size: int = 2,
    window_size: int = 112,
    patch_size: int = 14,
    rope_theta: float = 10000.0,
    return_window_patch_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Exact port of HF's rot_pos_emb() + get_window_index() for a fixed image.

    Matches HuggingFace's Qwen2_5_VisionTransformerPretrainedModel:
      1. 2D position IDs (height, width) with spatial merge permutation
      2. inv_freq with dim = head_dim // 2 (not head_dim // 3)
      3. Frequency duplication: cat(emb, emb) to fill full head_dim
      4. Window index computation for merged-group reordering
      5. Reorder position embeddings by window_index before cos/sin

    Returns:
        cos_table: [num_patches, embed_dim] float32
        sin_table: [num_patches, embed_dim] float32
        window_index: [num_merged] int32
        reverse_indices: [num_merged] int32
    """
    head_dim = embed_dim // num_heads
    num_patches = grid_h * grid_w
    merge_unit = merge_size * merge_size
    num_merged = num_patches // merge_unit

    # --- Step 1: inv_freq (HF Qwen2_5_VisionRotaryEmbedding) ---
    # dim = head_dim // 2; inv_freq shape = (dim // 2,) = (head_dim // 4,)
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (
        np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim
    ))

    # --- Step 2: Frequency lookup table ---
    max_grid = max(grid_h, grid_w)
    freqs = np.outer(
        np.arange(max_grid, dtype=np.float64), inv_freq
    )  # [max_grid, rope_dim // 2]

    # --- Step 3: Position IDs with spatial merge permutation (HF rot_pos_emb) ---
    hpos = (np.arange(grid_h, dtype=np.int32).reshape(-1, 1)
            * np.ones((1, grid_w), dtype=np.int32))
    hpos = hpos.reshape(
        grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
    hpos = hpos.transpose(0, 2, 1, 3).flatten()

    wpos = (np.ones((grid_h, 1), dtype=np.int32)
            * np.arange(grid_w, dtype=np.int32).reshape(1, -1))
    wpos = wpos.reshape(
        grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
    wpos = wpos.transpose(0, 2, 1, 3).flatten()

    # Index: h_freqs [num_patches, rope_dim//2], w_freqs [num_patches, rope_dim//2]
    h_freqs = freqs[hpos]
    w_freqs = freqs[wpos]
    pos_emb = np.concatenate([h_freqs, w_freqs], axis=1)  # [num_patches, rope_dim]

    # --- Step 4: Window index (HF get_window_index) ---
    llm_grid_h = grid_h // merge_size
    llm_grid_w = grid_w // merge_size
    vit_merger_window_size = window_size // merge_size // patch_size

    pad_h = (-llm_grid_h) % vit_merger_window_size
    pad_w = (-llm_grid_w) % vit_merger_window_size
    num_win_h = (llm_grid_h + pad_h) // vit_merger_window_size
    num_win_w = (llm_grid_w + pad_w) // vit_merger_window_size

    # Single frame (grid_t = 1)
    index = np.arange(
        llm_grid_h * llm_grid_w, dtype=np.int64
    ).reshape(1, llm_grid_h, llm_grid_w)

    index_padded = np.full(
        (1, llm_grid_h + pad_h, llm_grid_w + pad_w), -100, dtype=np.int64)
    index_padded[:, :llm_grid_h, :llm_grid_w] = index

    index_padded = index_padded.reshape(
        1, num_win_h, vit_merger_window_size,
        num_win_w, vit_merger_window_size)
    index_padded = index_padded.transpose(0, 1, 3, 2, 4).reshape(
        1, num_win_h * num_win_w, vit_merger_window_size, vit_merger_window_size)

    window_group_counts = (index_padded != -100).sum(axis=(2, 3)).reshape(-1)
    window_group_counts = window_group_counts[window_group_counts > 0].astype(np.int32)
    index_flat = index_padded.reshape(-1)
    window_index = index_flat[index_flat != -100].astype(np.int32)
    reverse_indices = np.argsort(window_index).astype(np.int32)

    # --- Step 5: Reorder pos_emb by window_index at merge-group level ---
    pos_emb_grouped = pos_emb.reshape(num_merged, merge_unit, rope_dim)
    pos_emb_grouped = pos_emb_grouped[window_index]
    pos_emb = pos_emb_grouped.reshape(num_patches, rope_dim)

    # --- Step 6: Duplicate to full head_dim: cat(emb, emb) ---
    pos_emb_full = np.concatenate(
        [pos_emb, pos_emb], axis=1)  # [num_patches, head_dim]

    # --- Step 7: cos/sin, tile across all heads ---
    cos = np.cos(pos_emb_full).astype(np.float32)
    sin = np.sin(pos_emb_full).astype(np.float32)
    cos_table = np.tile(cos, (1, num_heads))  # [num_patches, embed_dim]
    sin_table = np.tile(sin, (1, num_heads))

    if return_window_patch_counts:
        return cos_table, sin_table, window_index, reverse_indices, (
            window_group_counts * merge_unit
        ).astype(np.int32)
    return cos_table, sin_table, window_index, reverse_indices


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_qwen_vl_vision_engine(
    vision_config: dict,
    weights: WeightDict,
    *,
    fixed_image_size: int = 448,
    fixed_image_height: int | None = None,
    fixed_image_width: int | None = None,
    verbose: bool = False,
) -> bytes:
    """Build complete Qwen2.5-VL vision encoder TRT engine.

    The engine includes ALL learned operations from pixel input to
    text-hidden-size output features, for a fixed image size.

    Args:
        vision_config: The "vision_config" dict from the HF config.json.
        weights: Full weight dict (only "visual.*" keys are used).
        fixed_image_size: Square image height/width fallback.
        fixed_image_height: Optional rectangular image height.
        fixed_image_width: Optional rectangular image width.
        verbose: Print detailed logs.

    Returns:
        Serialized TRT engine plan bytes.
    """
    embed_dim = vision_config.get("embed_dim", vision_config.get("hidden_size", 1280))
    num_heads = vision_config.get("num_heads", vision_config.get("num_attention_heads", 16))
    num_layers = vision_config.get("depth", vision_config.get("num_hidden_layers", 32))
    # Intermediate size: prefer explicit value, fallback to mlp_ratio * embed_dim
    mlp_hidden = vision_config.get("intermediate_size", 0)
    if mlp_hidden == 0:
        mlp_ratio = vision_config.get("mlp_ratio", 4.0)
        mlp_hidden = int(embed_dim * mlp_ratio)
    in_channels = vision_config.get("in_channels", 3)
    temporal_patch_size = vision_config.get("temporal_patch_size", 2)
    patch_size = vision_config.get("patch_size", 14)
    merge_size = vision_config.get("spatial_merge_size", 2)
    eps_val = vision_config.get("layer_norm_eps", 1e-6)
    rope_theta = float(vision_config.get("rope_theta", 10000.0))

    fixed_h = int(fixed_image_height if fixed_image_height is not None else fixed_image_size)
    fixed_w = int(fixed_image_width if fixed_image_width is not None else fixed_image_size)
    if fixed_h <= 0 or fixed_w <= 0:
        raise ValueError("fixed image dimensions must be positive")
    if fixed_h % patch_size or fixed_w % patch_size:
        raise ValueError("fixed image dimensions must be divisible by patch_size")

    # Compute grid dimensions for fixed image size
    grid_h = fixed_h // patch_size
    grid_w = fixed_w // patch_size
    num_patches = grid_h * grid_w
    num_merged = num_patches // (merge_size * merge_size)

    # Text hidden size: determined by merger MLP output dimension.
    # The merger fc2 weight shape is [output_dim, hidden_dim] — output_dim = text_hidden.
    merger_fc2_w = weights.get("visual.merger.mlp.2.weight")
    if merger_fc2_w is not None:
        text_hidden_size = merger_fc2_w.shape[0]
    else:
        # Fallback: try to infer from config
        text_hidden_size = vision_config.get("text_hidden_size", embed_dim)

    if verbose:
        print(f"[trtmc build] Vision: image={fixed_h}x{fixed_w}, "
              f"grid={grid_h}x{grid_w}, patches={num_patches}, "
              f"merged={num_merged}, embed={embed_dim}, "
              f"text_hidden={text_hidden_size}", file=sys.stderr)

    builder_context = create_builder_context(
        verbose=verbose,
        workspace_bytes=2 << 30,
    )
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([eps_val], dtype=np.float32))

    # ---------------------------------------------------------------
    # Input: pixel_values [T*C, H, W]
    # For temporal_patch_size=2, T*C = 2*3 = 6
    # ---------------------------------------------------------------
    input_channels = temporal_patch_size * in_channels
    pixel_values = network.add_input(
        "pixel_values", trt.float32,
        (input_channels, fixed_h, fixed_w))

    # ---------------------------------------------------------------
    # Stage 1: 3D Patch Embedding (conv)
    # [T*C, H, W] -> [num_patches, embed_dim]
    # ---------------------------------------------------------------
    patch_embed_w = weights.get("visual.patch_embed.proj.weight")
    patch_embed_b = weights.get("visual.patch_embed.proj.bias")

    if patch_embed_w is None:
        raise RuntimeError("Missing visual.patch_embed.proj.weight")

    hidden = graph_ops.add_patch_embed_3d(
        network, pixel_values,
        patch_embed_w.astype(np.float32),
        patch_embed_b.astype(np.float32) if patch_embed_b is not None else None,
        in_channels=in_channels,
        embed_dim=embed_dim,
        temporal_patch_size=temporal_patch_size,
        patch_size=patch_size)

    # ---------------------------------------------------------------
    # Stage 2: Precompute RoPE tables + window index
    # ---------------------------------------------------------------
    window_size = int(vision_config.get("window_size", 112))
    merge_unit = merge_size * merge_size

    cos_table, sin_table, window_index, reverse_indices, window_patch_counts = \
        _compute_vision_rope_tables(
            grid_h, grid_w, embed_dim, num_heads,
            merge_size=merge_size, window_size=window_size,
            patch_size=patch_size, rope_theta=rope_theta,
            return_window_patch_counts=True)

    # Windowed vs full attention config
    vit_merger_window_size = window_size // merge_size // patch_size
    # Number of merged groups per window (e.g. 4x4 = 16)
    merged_per_window = vit_merger_window_size * vit_merger_window_size
    # Number of patches per window (e.g. 16 * 4 = 64)
    patches_per_window = merged_per_window * merge_unit
    # Number of real, non-empty windows. Edge windows can be partial when the
    # processor's smart-resized image is not window-aligned.
    num_windows = int(len(window_patch_counts))

    fullatt_block_indexes = set(
        vision_config.get("fullatt_block_indexes", [7, 15, 23, 31]))

    if verbose:
        print(f"[trtmc build] Vision RoPE: head_dim={embed_dim // num_heads}, "
              f"rope_dim={embed_dim // num_heads // 2}, "
              f"window_size={window_size}, "
              f"vit_merger_window_size={vit_merger_window_size}, "
              f"num_windows={num_windows}, "
              f"patches_per_window={patches_per_window}, "
              f"actual_window_patch_counts={window_patch_counts.tolist()}, "
              f"fullatt_blocks={sorted(fullatt_block_indexes)}",
              file=sys.stderr)

    # ---------------------------------------------------------------
    # Stage 2b: Reorder patches by window_index (at merge-group level)
    # hidden: [num_patches, embed_dim] -> reorder groups of merge_unit
    # ---------------------------------------------------------------
    reshp_win = network.add_shuffle(hidden)
    reshp_win.reshape_dims = (num_merged, merge_unit, embed_dim)

    win_idx_weights = trt.Weights(np.ascontiguousarray(window_index))
    win_idx_layer = network.add_constant((num_merged,), win_idx_weights)
    win_idx_cast = network.add_cast(win_idx_layer.get_output(0), trt.int32)

    gathered_win = network.add_gather(
        reshp_win.get_output(0), win_idx_cast.get_output(0), 0)

    reshp_back = network.add_shuffle(gathered_win.get_output(0))
    reshp_back.reshape_dims = (num_patches, embed_dim)
    hidden = reshp_back.get_output(0)

    # ---------------------------------------------------------------
    # Stage 3: ViT Transformer blocks
    # ---------------------------------------------------------------
    for layer_idx in range(num_layers):
        prefix = f"visual.blocks.{layer_idx}"

        # Pre-attention RMSNorm (Qwen2.5-VL uses Qwen2RMSNorm, not LayerNorm)
        ln1_gamma = weights.get(f"{prefix}.norm1.weight")
        if ln1_gamma is None:
            raise RuntimeError(f"Missing {prefix}.norm1.weight")

        normed = graph_ops.add_rms_norm(
            network, hidden, embed_dim,
            ln1_gamma.astype(np.float32),
            eps_tensor)

        # Self-attention with 3D RoPE
        # Handle fused QKV weights
        w_q = weights.get(f"{prefix}.attn.qkv.weight_q")
        w_k = weights.get(f"{prefix}.attn.qkv.weight_k")
        w_v = weights.get(f"{prefix}.attn.qkv.weight_v")
        w_o = weights.get(f"{prefix}.attn.proj.weight")

        if w_q is None:
            qkv_w = weights.get(f"{prefix}.attn.qkv.weight")
            if qkv_w is not None:
                qkv_w = qkv_w.astype(np.float32)
                w_q = qkv_w[:embed_dim, :].T.copy()
                w_k = qkv_w[embed_dim:2*embed_dim, :].T.copy()
                w_v = qkv_w[2*embed_dim:, :].T.copy()
            else:
                raise RuntimeError(f"Missing {prefix}.attn.qkv.weight")

        # Handle fused QKV biases
        q_bias = weights.get(f"{prefix}.attn.qkv.bias_q")
        k_bias = weights.get(f"{prefix}.attn.qkv.bias_k")
        v_bias = weights.get(f"{prefix}.attn.qkv.bias_v")
        o_bias = weights.get(f"{prefix}.attn.proj.bias")

        if q_bias is None:
            qkv_b = weights.get(f"{prefix}.attn.qkv.bias")
            if qkv_b is not None:
                qkv_b = qkv_b.astype(np.float32)
                q_bias = qkv_b[:embed_dim].copy()
                k_bias = qkv_b[embed_dim:2*embed_dim].copy()
                v_bias = qkv_b[2*embed_dim:].copy()

        use_full_attn = (layer_idx in fullatt_block_indexes)
        w_o_np = (w_o.astype(np.float32).T.copy() if w_o is not None
                  else np.zeros((embed_dim, embed_dim), dtype=np.float32))
        attn_kwargs = dict(
            w_q=w_q.astype(np.float32),
            w_k=w_k.astype(np.float32),
            w_v=w_v.astype(np.float32),
            w_o=w_o_np,
            hidden_size=embed_dim,
            num_heads=num_heads,
            seq_length=num_patches,
            cos_table=cos_table,
            sin_table=sin_table,
            q_bias=q_bias.astype(np.float32) if q_bias is not None else None,
            k_bias=k_bias.astype(np.float32) if k_bias is not None else None,
            v_bias=v_bias.astype(np.float32) if v_bias is not None else None,
            o_bias=o_bias.astype(np.float32) if o_bias is not None else None,
        )

        if use_full_attn:
            attn_out = graph_ops.add_self_attention_block_with_rope(
                network, normed, **attn_kwargs)
        else:
            attn_out = graph_ops.add_windowed_self_attention_with_rope(
                network,
                normed,
                num_windows=num_windows,
                window_patch_counts=window_patch_counts,
                **attn_kwargs,
            )

        # Residual
        res1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)

        # Post-attention RMSNorm
        ln2_gamma = weights.get(f"{prefix}.norm2.weight")
        normed2 = graph_ops.add_rms_norm(
            network, res1.get_output(0), embed_dim,
            ln2_gamma.astype(np.float32) if ln2_gamma is not None
                else np.ones(embed_dim, dtype=np.float32),
            eps_tensor)

        # SwiGLU MLP: gate_proj + up_proj + SiLU + down_proj (with biases)
        gate_w = weights.get(f"{prefix}.mlp.gate_proj.weight")
        up_w = weights.get(f"{prefix}.mlp.up_proj.weight")
        down_w = weights.get(f"{prefix}.mlp.down_proj.weight")
        gate_b = weights.get(f"{prefix}.mlp.gate_proj.bias")
        up_b = weights.get(f"{prefix}.mlp.up_proj.bias")
        down_b = weights.get(f"{prefix}.mlp.down_proj.bias")

        # Fallback: try old fc1/fc2 naming (for compatibility)
        if gate_w is None:
            gate_w = weights.get(f"{prefix}.mlp.fc1.weight")
            up_w = None  # fc1/fc2 style, not SwiGLU
        if gate_w is None:
            raise RuntimeError(f"Missing {prefix}.mlp weights")

        if up_w is not None:
            # SwiGLU: gate * sigmoid(gate) * up, then down
            gate = graph_ops.add_matmul_rhs_constant(
                network, normed2, embed_dim, mlp_hidden,
                gate_w.astype(np.float32).T.copy())
            if gate_b is not None:
                gate = graph_ops.add_bias_sum(network, gate, mlp_hidden,
                                              gate_b.astype(np.float32))
            up = graph_ops.add_matmul_rhs_constant(
                network, normed2, embed_dim, mlp_hidden,
                up_w.astype(np.float32).T.copy())
            if up_b is not None:
                up = graph_ops.add_bias_sum(network, up, mlp_hidden,
                                            up_b.astype(np.float32))

            # SiLU(gate) = gate * sigmoid(gate)
            sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
            swish = network.add_elementwise(
                gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
            # gated = SiLU(gate) * up
            gated = network.add_elementwise(
                swish.get_output(0), up, trt.ElementWiseOperation.PROD)

            fc2 = graph_ops.add_matmul_rhs_constant(
                network, gated.get_output(0), mlp_hidden, embed_dim,
                down_w.astype(np.float32).T.copy())
            if down_b is not None:
                fc2 = graph_ops.add_bias_sum(network, fc2, embed_dim,
                                             down_b.astype(np.float32))
        else:
            # GELU fc1/fc2 fallback
            fc1_b = weights.get(f"{prefix}.mlp.fc1.bias")
            fc2_w = weights.get(f"{prefix}.mlp.fc2.weight")
            fc2_b = weights.get(f"{prefix}.mlp.fc2.bias")
            fc1 = graph_ops.add_matmul_rhs_constant(
                network, normed2, embed_dim, mlp_hidden,
                gate_w.astype(np.float32).T.copy())
            if fc1_b is not None:
                fc1 = graph_ops.add_bias_sum(network, fc1, mlp_hidden,
                                             fc1_b.astype(np.float32))
            activated = graph_ops.add_gelu_new(network, fc1)
            fc2 = graph_ops.add_matmul_rhs_constant(
                network, activated, mlp_hidden, embed_dim,
                fc2_w.astype(np.float32).T.copy())
            if fc2_b is not None:
                fc2 = graph_ops.add_bias_sum(network, fc2, embed_dim,
                                             fc2_b.astype(np.float32))

        # Residual
        res2 = network.add_elementwise(
            res1.get_output(0), fc2, trt.ElementWiseOperation.SUM)
        hidden = res2.get_output(0)

    # ---------------------------------------------------------------
    # Stage 4: Spatial Merge (HF Qwen2_5_VLPatchMerger)
    # RMSNorm -> view(-1, merged_dim) -> Linear -> GELU -> Linear
    # [num_patches, embed_dim] -> [num_merged, text_hidden_size]
    # ---------------------------------------------------------------
    merged_dim = embed_dim * merge_unit  # 1280 * 4 = 5120

    # Merger weights
    merger_ln_w = weights.get("visual.merger.ln_q.weight")
    merger_fc1_w = weights.get("visual.merger.mlp.0.weight")
    merger_fc1_b = weights.get("visual.merger.mlp.0.bias")
    merger_fc2_b = weights.get("visual.merger.mlp.2.bias")

    if merger_ln_w is None or merger_fc1_w is None or merger_fc2_w is None:
        raise RuntimeError(
            "Missing merger weights (visual.merger.ln_q.weight, "
            "visual.merger.mlp.{0,2}.weight)")

    # 1. RMSNorm per-patch (NOT LayerNorm — HF uses Qwen2_5_VLRMSNorm)
    normed_patches = graph_ops.add_rms_norm(
        network, hidden, embed_dim,
        merger_ln_w.astype(np.float32),
        eps_tensor)

    # 2. Group every merge_unit consecutive patches (matches HF's view(-1, hidden_size))
    reshape_merge = network.add_shuffle(normed_patches)
    reshape_merge.reshape_dims = (num_merged, merged_dim)
    merged_features = reshape_merge.get_output(0)

    # 3. MLP: [num_merged, merged_dim] -> [num_merged, text_hidden_size]
    merger_fc1_hidden = merger_fc1_w.shape[0]
    fc1_out = graph_ops.add_matmul_rhs_constant(
        network, merged_features, merged_dim, merger_fc1_hidden,
        merger_fc1_w.astype(np.float32).T.copy())
    if merger_fc1_b is not None:
        fc1_out = graph_ops.add_bias_sum(network, fc1_out, merger_fc1_hidden,
                                          merger_fc1_b.astype(np.float32))

    # GELU activation
    activated_merged = graph_ops.add_gelu_new(network, fc1_out)

    # MLP layer 2: [num_merged, fc1_hidden] -> [num_merged, text_hidden_size]
    fc2_out = graph_ops.add_matmul_rhs_constant(
        network, activated_merged, merger_fc1_hidden, text_hidden_size,
        merger_fc2_w.astype(np.float32).T.copy())
    if merger_fc2_b is not None:
        fc2_out = graph_ops.add_bias_sum(network, fc2_out, text_hidden_size,
                                          merger_fc2_b.astype(np.float32))

    # 4. Reverse window reorder: restore original spatial order
    rev_idx_weights = trt.Weights(np.ascontiguousarray(reverse_indices))
    rev_idx_layer = network.add_constant((num_merged,), rev_idx_weights)
    rev_idx_cast = network.add_cast(rev_idx_layer.get_output(0), trt.int32)

    reversed_out = network.add_gather(
        fc2_out, rev_idx_cast.get_output(0), 0)

    # ---------------------------------------------------------------
    # Output: image_features [num_merged, text_hidden_size]
    # ---------------------------------------------------------------
    reversed_out.get_output(0).name = "image_features"
    network.mark_output(reversed_out.get_output(0))

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building Qwen VL vision TRT engine "
              f"({num_layers} layers, embed={embed_dim}, "
              f"patches={num_patches}, merged={num_merged}, "
              f"text_hidden={text_hidden_size}) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT vision engine build failed")

    return bytes(plan)
