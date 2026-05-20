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

from ... import graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


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

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

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
                network, normed, window_patch_counts=window_patch_counts, **attn_kwargs)

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


# ---------------------------------------------------------------------------
# Qwen3-VL vision engine builder (learned positions, LayerNorm, GELU MLP,
# full attention, multi-level DeepStack outputs)
# ---------------------------------------------------------------------------

def _add_deepstack_merger(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: WeightDict,
    merger_prefix: str,
    embed_dim: int,
    merge_unit: int,
    num_merged: int,
    text_hidden_size: int,
    eps_tensor: trt.ITensor,
    reverse_indices: np.ndarray,
) -> trt.ITensor:
    """Apply a DeepStack PatchMerger: group → LayerNorm → fc1 → GELU → fc2 → reverse.

    Input: [num_patches, embed_dim]
    Output: [num_merged, text_hidden_size]

    Note: The DeepStack merger LayerNorm operates on the merged dimension
    (embed_dim * merge_unit), not embed_dim. This matches HF's PatchMerger
    which applies norm AFTER grouping patches.
    """
    merged_dim = embed_dim * merge_unit

    # Group first: [num_patches, embed_dim] -> [num_merged, merged_dim]
    reshape_grp = network.add_shuffle(hidden)
    reshape_grp.reshape_dims = (num_merged, merged_dim)

    # LayerNorm (with bias) on merged dimension
    norm_w = weights[f"{merger_prefix}.norm.weight"].astype(np.float32)
    norm_b = weights[f"{merger_prefix}.norm.bias"].astype(np.float32)
    normed = graph_ops.add_layer_norm(
        network, reshape_grp.get_output(0), merged_dim, norm_w, norm_b, eps_tensor)

    # Group: [num_patches, embed_dim] -> [num_merged, merged_dim]
    reshape_grp = network.add_shuffle(normed)
    reshape_grp.reshape_dims = (num_merged, merged_dim)

    # fc1: [num_merged, merged_dim] -> [num_merged, fc1_hidden]
    fc1_w = weights[f"{merger_prefix}.linear_fc1.weight"].astype(np.float32)
    fc1_b = weights[f"{merger_prefix}.linear_fc1.bias"].astype(np.float32)
    fc1_hidden = fc1_w.shape[0]
    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed, merged_dim, fc1_hidden,
        fc1_w.T.copy())
    fc1 = graph_ops.add_bias_sum(network, fc1, fc1_hidden, fc1_b)

    # GELU activation
    activated = graph_ops.add_gelu_new(network, fc1)

    # fc2: [num_merged, fc1_hidden] -> [num_merged, text_hidden_size]
    fc2_w = weights[f"{merger_prefix}.linear_fc2.weight"].astype(np.float32)
    fc2_b = weights[f"{merger_prefix}.linear_fc2.bias"].astype(np.float32)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, fc1_hidden, text_hidden_size, fc2_w.T.copy())
    fc2 = graph_ops.add_bias_sum(network, fc2, text_hidden_size, fc2_b)

    # Reverse window reorder
    rev_idx = trt.Weights(np.ascontiguousarray(reverse_indices))
    rev_layer = network.add_constant((num_merged,), rev_idx)
    rev_cast = network.add_cast(rev_layer.get_output(0), trt.int32)
    reversed_out = network.add_gather(fc2, rev_cast.get_output(0), 0)

    return reversed_out.get_output(0)


def build_qwen3_vl_vision_engine(
    vision_config: dict,
    weights: WeightDict,
    *,
    fixed_image_size: int = 448,
    verbose: bool = False,
) -> bytes:
    """Build Qwen3-VL vision encoder TRT engine with DeepStack multi-level outputs.

    Differences from Qwen2.5-VL:
      - Learned position embedding (no 3D RoPE)
      - LayerNorm with bias (not RMSNorm)
      - GELU FC MLP: linear_fc1 → GELU → linear_fc2 (not SwiGLU)
      - Full attention only (no windowed attention)
      - DeepStack: branch off at specified ViT layers → PatchMerger → extra outputs

    Engine outputs:
      - image_features [num_merged, text_hidden_size] — main merged features
      - deepstack_features_0..N [num_merged, text_hidden_size] — per-level features
    """
    embed_dim = vision_config.get("embed_dim", vision_config.get("hidden_size", 1024))
    num_heads = vision_config.get("num_heads", vision_config.get("num_attention_heads", 16))
    num_layers = vision_config.get("depth", vision_config.get("num_hidden_layers", 24))
    mlp_hidden = vision_config.get("intermediate_size", 4096)
    in_channels = vision_config.get("in_channels", 3)
    temporal_patch_size = vision_config.get("temporal_patch_size", 2)
    patch_size = vision_config.get("patch_size", 16)
    merge_size = vision_config.get("spatial_merge_size", 2)
    eps_val = vision_config.get("layer_norm_eps", 1e-6)
    deepstack_indexes = vision_config.get("deepstack_visual_indexes", [])
    text_hidden_size = vision_config.get("out_hidden_size", embed_dim)

    grid_h = fixed_image_size // patch_size
    grid_w = fixed_image_size // patch_size
    num_patches = grid_h * grid_w
    merge_unit = merge_size * merge_size
    num_merged = num_patches // merge_unit

    if verbose:
        print(f"[trtmc build] Qwen3-VL Vision: image={fixed_image_size}, "
              f"patch={patch_size}, grid={grid_h}x{grid_w}, "
              f"patches={num_patches}, merged={num_merged}, "
              f"embed={embed_dim}, text_hidden={text_hidden_size}, "
              f"deepstack={deepstack_indexes}", file=sys.stderr)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([eps_val], dtype=np.float32))

    # ---------------------------------------------------------------
    # Input: pixel_values [T*C, H, W]
    # ---------------------------------------------------------------
    input_channels = temporal_patch_size * in_channels
    pixel_values = network.add_input(
        "pixel_values", trt.float32,
        (input_channels, fixed_image_size, fixed_image_size))

    # ---------------------------------------------------------------
    # Stage 1: 3D Patch Embedding
    # ---------------------------------------------------------------
    patch_embed_w = weights["visual.patch_embed.proj.weight"].astype(np.float32)
    patch_embed_b = weights.get("visual.patch_embed.proj.bias")

    hidden = graph_ops.add_patch_embed_3d(
        network, pixel_values, patch_embed_w,
        patch_embed_b.astype(np.float32) if patch_embed_b is not None else None,
        in_channels=in_channels, embed_dim=embed_dim,
        temporal_patch_size=temporal_patch_size, patch_size=patch_size)

    # ---------------------------------------------------------------
    # Stage 2: Learned position embedding (fast_pos_embed_interpolate)
    # The qwen_merge_group preprocessor already reorders pixels so that
    # the conv output is in merge-group order (matching HF's patch ordering).
    # For exact integer grid (no interpolation needed), just index directly.
    # pos_embed.weight: [num_grid_per_side^2, embed_dim]
    # ---------------------------------------------------------------
    pos_embed_w = weights["visual.pos_embed.weight"].astype(np.float32)
    num_grid_per_side = int(np.sqrt(pos_embed_w.shape[0]))

    # For fixed_image_size grid, compute bilinear interpolation indices
    h_idxs = np.linspace(0, num_grid_per_side - 1, grid_h).astype(np.float32)
    w_idxs = np.linspace(0, num_grid_per_side - 1, grid_w).astype(np.float32)

    h_floor = np.floor(h_idxs).astype(int)
    w_floor = np.floor(w_idxs).astype(int)
    h_ceil = np.minimum(h_floor + 1, num_grid_per_side - 1)
    w_ceil = np.minimum(w_floor + 1, num_grid_per_side - 1)
    dh = h_idxs - h_floor.astype(np.float32)
    dw = w_idxs - w_floor.astype(np.float32)

    # Bilinear interpolation of position embeddings in MERGE-GROUP order.
    # Patches arrive in merge-group order from the conv (because qwen_merge_group
    # preprocessor reorders pixels). The position embedding must match this order.
    merged_h_pos = grid_h // merge_size
    merged_w_pos = grid_w // merge_size
    pos_embed_interp = np.zeros((num_patches, pos_embed_w.shape[1]), dtype=np.float32)
    idx = 0
    for bh in range(merged_h_pos):
        for bw in range(merged_w_pos):
            for ih in range(merge_size):
                for iw in range(merge_size):
                    hi = bh * merge_size + ih
                    wi = bw * merge_size + iw
                    w00 = (1 - dh[hi]) * (1 - dw[wi])
                    w01 = (1 - dh[hi]) * dw[wi]
                    w10 = dh[hi] * (1 - dw[wi])
                    w11 = dh[hi] * dw[wi]
                    i00 = h_floor[hi] * num_grid_per_side + w_floor[wi]
                    i01 = h_floor[hi] * num_grid_per_side + w_ceil[wi]
                    i10 = h_ceil[hi] * num_grid_per_side + w_floor[wi]
                    i11 = h_ceil[hi] * num_grid_per_side + w_ceil[wi]
                    pos_embed_interp[idx] = (w00 * pos_embed_w[i00] + w01 * pos_embed_w[i01]
                                             + w10 * pos_embed_w[i10] + w11 * pos_embed_w[i11])
                    idx += 1

    pos_const = graph_ops.add_constant(
        network, (num_patches, embed_dim), pos_embed_interp)
    pos_add = network.add_elementwise(
        hidden, pos_const, trt.ElementWiseOperation.SUM)
    hidden = pos_add.get_output(0)

    # ---------------------------------------------------------------
    # Stage 2b: Compute 2D RoPE tables (merge-group ordered, Qwen3-VL style)
    # Qwen3-VL uses rot_pos_emb() which computes merge-group-ordered 2D
    # position IDs, then uses them as RoPE in attention blocks.
    # NO window_index reordering (unlike Qwen2.5-VL).
    # ---------------------------------------------------------------
    rope_theta = float(vision_config.get("rope_theta", 10000.0))
    head_dim = embed_dim // num_heads
    rope_dim = head_dim // 2  # dim parameter to RotaryEmbedding

    # inv_freq: 1.0 / (theta ** (arange(0, rope_dim, 2) / rope_dim))
    inv_freq = 1.0 / (rope_theta ** (
        np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim))

    # Build frequency table: freq_table[pos] = outer(pos, inv_freq)
    max_hw = max(grid_h, grid_w)
    freq_table = np.outer(np.arange(max_hw, dtype=np.float64), inv_freq)

    # Compute 2D position IDs in the SAME order as patches arrive at attention.
    # HF's rot_pos_emb uses merge-group ordering: for each merged block (bh, bw),
    # iterate over intra-block offsets (ih, iw). The patches from our conv are in
    # raster order, so we must match: the RoPE table is applied element-wise to
    # the sequence, meaning position[i] describes the spatial location of patch[i].
    # Our conv outputs patches in raster order: (0,0),(0,1),...,(0,W-1),(1,0),...
    # HF's unfold+linear also produces raster order patches.
    # The RoPE positions must be in raster order too.
    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size

    # Merge-group-ordered 2D position IDs (matches HF rot_pos_emb)
    row_idx = np.zeros(num_patches, dtype=np.int64)
    col_idx = np.zeros(num_patches, dtype=np.int64)
    idx = 0
    for bh in range(merged_h):
        for bw in range(merged_w):
            for ih in range(merge_size):
                for iw in range(merge_size):
                    row_idx[idx] = bh * merge_size + ih
                    col_idx[idx] = bw * merge_size + iw
                    idx += 1

    # Lookup freqs and concatenate: [h_freqs, w_freqs] -> [num_patches, rope_dim]
    h_freqs = freq_table[row_idx]  # [num_patches, rope_dim//2]
    w_freqs = freq_table[col_idx]  # [num_patches, rope_dim//2]
    pos_emb = np.concatenate([h_freqs, w_freqs], axis=1)  # [num_patches, rope_dim]

    # Duplicate to full head_dim: cat(emb, emb) -> [num_patches, head_dim]
    pos_emb_full = np.concatenate([pos_emb, pos_emb], axis=1)

    # cos/sin tables, tile across all heads -> [num_patches, embed_dim]
    cos_table = np.tile(np.cos(pos_emb_full).astype(np.float32), (1, num_heads))
    sin_table = np.tile(np.sin(pos_emb_full).astype(np.float32), (1, num_heads))

    # Reverse indices for merger output reordering (identity — no window reorder)
    reverse_indices = np.arange(num_merged, dtype=np.int32)

    # ---------------------------------------------------------------
    # Stage 3: ViT Transformer blocks (full attention, LayerNorm, GELU FC MLP)
    # ---------------------------------------------------------------
    deepstack_index_set = set(deepstack_indexes)
    deepstack_branches = {}  # layer_idx -> hidden tensor at that point

    for layer_idx in range(num_layers):
        prefix = f"visual.blocks.{layer_idx}"

        # Pre-attention LayerNorm (with bias)
        ln1_gamma = weights[f"{prefix}.norm1.weight"].astype(np.float32)
        ln1_beta = weights[f"{prefix}.norm1.bias"].astype(np.float32)
        normed = graph_ops.add_layer_norm(
            network, hidden, embed_dim, ln1_gamma, ln1_beta, eps_tensor)

        # Self-attention with 2D RoPE (Qwen3-VL uses both learned pos + RoPE)
        w_q, w_k, w_v, q_bias, k_bias, v_bias = None, None, None, None, None, None
        qkv_w = weights.get(f"{prefix}.attn.qkv.weight")
        if qkv_w is not None:
            qkv_w = qkv_w.astype(np.float32)
            w_q = qkv_w[:embed_dim, :].T.copy()
            w_k = qkv_w[embed_dim:2*embed_dim, :].T.copy()
            w_v = qkv_w[2*embed_dim:, :].T.copy()

        qkv_b = weights.get(f"{prefix}.attn.qkv.bias")
        if qkv_b is not None:
            qkv_b = qkv_b.astype(np.float32)
            q_bias = qkv_b[:embed_dim].copy()
            k_bias = qkv_b[embed_dim:2*embed_dim].copy()
            v_bias = qkv_b[2*embed_dim:].copy()

        w_o = weights.get(f"{prefix}.attn.proj.weight")
        w_o_np = w_o.astype(np.float32).T.copy() if w_o is not None else None
        o_bias = weights.get(f"{prefix}.attn.proj.bias")
        o_bias_np = o_bias.astype(np.float32) if o_bias is not None else None

        attn_out = graph_ops.add_self_attention_block_with_rope(
            network, normed,
            w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o_np,
            hidden_size=embed_dim, num_heads=num_heads,
            seq_length=num_patches,
            cos_table=cos_table, sin_table=sin_table,
            q_bias=q_bias, k_bias=k_bias, v_bias=v_bias,
            o_bias=o_bias_np)

        # Residual
        res1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM)

        # Post-attention LayerNorm
        ln2_gamma = weights[f"{prefix}.norm2.weight"].astype(np.float32)
        ln2_beta = weights[f"{prefix}.norm2.bias"].astype(np.float32)
        normed2 = graph_ops.add_layer_norm(
            network, res1.get_output(0), embed_dim,
            ln2_gamma, ln2_beta, eps_tensor)

        # GELU FC MLP: linear_fc1 → GELU → linear_fc2
        fc1_w = weights[f"{prefix}.mlp.linear_fc1.weight"].astype(np.float32)
        fc1_b = weights[f"{prefix}.mlp.linear_fc1.bias"].astype(np.float32)
        fc2_w = weights[f"{prefix}.mlp.linear_fc2.weight"].astype(np.float32)
        fc2_b = weights[f"{prefix}.mlp.linear_fc2.bias"].astype(np.float32)

        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, embed_dim, mlp_hidden, fc1_w.T.copy())
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_hidden, fc1_b)
        activated = graph_ops.add_gelu_new(network, fc1)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network, activated, mlp_hidden, embed_dim, fc2_w.T.copy())
        fc2 = graph_ops.add_bias_sum(network, fc2, embed_dim, fc2_b)

        # Residual
        res2 = network.add_elementwise(
            res1.get_output(0), fc2, trt.ElementWiseOperation.SUM)
        hidden = res2.get_output(0)

        # DeepStack branch: save hidden state at specified layers
        if layer_idx in deepstack_index_set:
            deepstack_branches[layer_idx] = hidden

    # ---------------------------------------------------------------
    # Stage 4: Main spatial merge (same as Qwen2.5-VL but with LayerNorm naming)
    # ---------------------------------------------------------------
    merger_norm_w = weights["visual.merger.norm.weight"].astype(np.float32)
    merger_norm_b = weights["visual.merger.norm.bias"].astype(np.float32)
    normed_patches = graph_ops.add_layer_norm(
        network, hidden, embed_dim, merger_norm_w, merger_norm_b, eps_tensor)

    merged_dim = embed_dim * merge_unit
    reshape_merge = network.add_shuffle(normed_patches)
    reshape_merge.reshape_dims = (num_merged, merged_dim)

    merger_fc1_w = weights["visual.merger.linear_fc1.weight"].astype(np.float32)
    merger_fc1_b = weights["visual.merger.linear_fc1.bias"].astype(np.float32)
    merger_fc2_w = weights["visual.merger.linear_fc2.weight"].astype(np.float32)
    merger_fc2_b = weights["visual.merger.linear_fc2.bias"].astype(np.float32)

    fc1_hidden = merger_fc1_w.shape[0]
    fc1_out = graph_ops.add_matmul_rhs_constant(
        network, reshape_merge.get_output(0), merged_dim, fc1_hidden,
        merger_fc1_w.T.copy())
    fc1_out = graph_ops.add_bias_sum(network, fc1_out, fc1_hidden, merger_fc1_b)
    activated = graph_ops.add_gelu_new(network, fc1_out)
    fc2_out = graph_ops.add_matmul_rhs_constant(
        network, activated, fc1_hidden, text_hidden_size,
        merger_fc2_w.T.copy())
    fc2_out = graph_ops.add_bias_sum(network, fc2_out, text_hidden_size, merger_fc2_b)

    # Reverse window reorder
    rev_idx = trt.Weights(np.ascontiguousarray(reverse_indices))
    rev_layer = network.add_constant((num_merged,), rev_idx)
    rev_cast = network.add_cast(rev_layer.get_output(0), trt.int32)
    main_features = network.add_gather(fc2_out, rev_cast.get_output(0), 0)

    main_features.get_output(0).name = "image_features"
    network.mark_output(main_features.get_output(0))

    # ---------------------------------------------------------------
    # Stage 5: DeepStack merger outputs
    # Each deepstack branch: hidden at ViT layer → PatchMerger → output
    # ---------------------------------------------------------------
    for ds_idx, layer_idx in enumerate(sorted(deepstack_branches.keys())):
        ds_hidden = deepstack_branches[layer_idx]
        merger_prefix = f"visual.deepstack_merger_list.{ds_idx}"

        ds_features = _add_deepstack_merger(
            network, ds_hidden, weights, merger_prefix,
            embed_dim, merge_unit, num_merged, text_hidden_size,
            eps_tensor, reverse_indices)

        ds_features.name = f"deepstack_features_{ds_idx}"
        network.mark_output(ds_features)

    # ---------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------
    if verbose:
        print(f"[trtmc build] Building Qwen3-VL vision TRT engine "
              f"({num_layers} layers, embed={embed_dim}, "
              f"patches={num_patches}, merged={num_merged}, "
              f"text_hidden={text_hidden_size}, "
              f"deepstack_levels={len(deepstack_branches)}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen3-VL vision engine build failed")

    return bytes(plan)
