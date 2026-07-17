# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed 1280x704/121-frame TensorRT denoiser for Wan2.2 TI2V-5B."""

from __future__ import annotations

import ctypes
import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from .checkpoint_mapper import (
    convert_transformer_state_dict,
    load_native_transformer_state_dict,
)
from .model_config import WAN22_TI2V_5B
from . import trt_ops as op

trt = trt_compat.get_trt()


def _numpy_state(model_dir: str) -> dict[str, np.ndarray]:
    state = convert_transformer_state_dict(load_native_transformer_state_dict(model_dir))
    return {name: tensor.detach().float().cpu().numpy() for name, tensor in state.items()}


def _wan_rope(latent_frames: int, latent_height: int, latent_width: int):
    cfg = WAN22_TI2V_5B
    grid = (
        latent_frames // cfg.patch_size[0],
        latent_height // cfg.patch_size[1],
        latent_width // cfg.patch_size[2],
    )
    half = cfg.head_dim // 2
    parts = (half - 2 * (half // 3), half // 3, half // 3)
    tables = []
    for length, complex_dim in zip(grid, parts):
        real_dim = complex_dim * 2
        inv = np.power(
            10000.0,
            -np.arange(0, real_dim, 2, dtype=np.float64) / real_dim,
        )
        tables.append(np.outer(np.arange(length, dtype=np.float64), inv))
    phase = np.concatenate(
        [
            np.broadcast_to(tables[0][:, None, None, :], (*grid, parts[0])),
            np.broadcast_to(tables[1][None, :, None, :], (*grid, parts[1])),
            np.broadcast_to(tables[2][None, None, :, :], (*grid, parts[2])),
        ],
        axis=-1,
    ).reshape(-1, half)
    return np.cos(phase), np.sin(phase)


def _slice_chunks(network, tensor, count: int, width: int):
    return [
        network.add_slice(tensor, (0, index * width), (1, width), (1, 1)).get_output(0)
        for index in range(count)
    ]


def _mark_debug_rows(network, tensor, name: str, *, rows: int = 8):
    """Expose a small row sample without materializing a multi-GiB debug output."""

    width = int(tensor.shape[-1])
    sample = network.add_slice(tensor, (0, 0), (rows, width), (1, 1)).get_output(0)
    sample = network.add_identity(op.cast(network, sample, trt.float32)).get_output(0)
    sample.name = name
    network.mark_output(sample)


def _mark_debug_tensor(network, tensor, name: str):
    """Expose a complete tensor only for explicit full-attention qualification."""

    debug = network.add_identity(op.cast(network, tensor, trt.float32)).get_output(0)
    debug.name = name
    network.mark_output(debug)


def _patchify(network, latent, weight, bias, cfg):
    source = op.source_patch_embedding(
        network, latent, weight, bias, cfg.patch_size, cfg.num_patches, cfg.dim
    )
    if source is not None:
        return source
    pt, ph, pw = cfg.patch_size
    patches = network.add_shuffle(latent)
    patches.reshape_dims = (
        1,
        cfg.in_channels,
        cfg.latent_frames // pt,
        pt,
        cfg.latent_height // ph,
        ph,
        cfg.latent_width // pw,
        pw,
    )
    patches.second_transpose = trt.Permutation([0, 2, 4, 6, 1, 3, 5, 7])
    rows = network.add_shuffle(patches.get_output(0))
    rows.reshape_dims = (cfg.num_patches, cfg.in_channels * pt * ph * pw)
    return op.linear(
        network,
        rows.get_output(0),
        weight.reshape(cfg.dim, -1),
        bias,
        bf16=True,
    )


def _unpatchify(network, rows, cfg):
    pt, ph, pw = cfg.patch_size
    reshape = network.add_shuffle(rows)
    reshape.reshape_dims = (
        cfg.latent_frames // pt,
        cfg.latent_height // ph,
        cfg.latent_width // pw,
        pt,
        ph,
        pw,
        cfg.out_channels,
    )
    reshape.second_transpose = trt.Permutation([6, 0, 3, 1, 4, 2, 5])
    output = network.add_shuffle(reshape.get_output(0))
    output.reshape_dims = (
        1,
        cfg.out_channels,
        cfg.latent_frames,
        cfg.latent_height,
        cfg.latent_width,
    )
    return output.get_output(0)


def build_dit_engine(
    model_dir: str,
    *,
    latent_frames: int | None = None,
    latent_height: int | None = None,
    latent_width: int | None = None,
    num_layers: int | None = None,
    round_residual_bf16: bool = False,
    emulate_bf16_gemm: bool = False,
    debug_layers: tuple[int, ...] = (),
    debug_sub_layers: tuple[int, ...] = (),
    debug_full_attention_layers: tuple[int, ...] = (),
    debug_full_norm_layers: tuple[int, ...] = (),
    debug_full_substage_layers: tuple[int, ...] = (),
    debug_cross_k_norm_layers: tuple[int, ...] = (),
    debug_embeddings: bool = False,
    debug_final_stages: bool = False,
    cross_attention_fp32: bool = False,
    source_attention_plugin: str | None = None,
    cuda_bf16_plugin: str | None = None,
    dit_cuda_plugin: str | None = None,
    dit_bf16_linear: bool = True,
    dit_time_silu: bool = False,
    dit_time_linear2: bool = False,
    dit_time_projection: bool = False,
    dit_block_layer_norm: bool = False,
    dit_adaptive_norm: bool = False,
    dit_rms_norm: bool = False,
    dit_self_gated_residual: bool = False,
    dit_ffn_gated_residual: bool = False,
    dit_cross_affine_layer_norm: bool = False,
    dit_final_projection: bool = False,
    verbose: bool = False,
) -> bytes:
    cfg = WAN22_TI2V_5B
    lf = cfg.latent_frames if latent_frames is None else latent_frames
    lh = cfg.latent_height if latent_height is None else latent_height
    lw = cfg.latent_width if latent_width is None else latent_width
    if (lf, lh, lw) != (cfg.latent_frames, cfg.latent_height, cfg.latent_width):
        # Small component probes are allowed, while production is the exact
        # fixed profile.  Derive only shape-related values locally.
        cfg = type(cfg)(
            video_num_frames=(lf - 1) * WAN22_TI2V_5B.scale_factor_temporal + 1,
            video_height=lh * WAN22_TI2V_5B.scale_factor_spatial,
            video_width=lw * WAN22_TI2V_5B.scale_factor_spatial,
        )
    layers = cfg.num_layers if num_layers is None else int(num_layers)
    if layers < 1 or layers > cfg.num_layers:
        raise ValueError(f"num_layers must be in [1, {cfg.num_layers}], got {layers}")

    weights = _numpy_state(model_dir)
    op.set_bf16_gemm_emulation(emulate_bf16_gemm)
    op.set_source_attention_plugin(source_attention_plugin is not None)
    op.set_cuda_bf16_barriers(cuda_bf16_plugin is not None)
    op.set_dit_cuda_numerics(dit_cuda_plugin is not None)
    op.set_dit_bf16_linear(dit_cuda_plugin is not None and dit_bf16_linear)
    op.set_dit_time_silu(dit_cuda_plugin is not None and dit_time_silu)
    op.set_dit_time_linear2(dit_cuda_plugin is not None and dit_time_linear2)
    op.set_dit_time_projection(dit_cuda_plugin is not None and dit_time_projection)
    op.set_dit_block_layer_norm(dit_cuda_plugin is not None and dit_block_layer_norm)
    op.set_dit_adaptive_norm(dit_cuda_plugin is not None and dit_adaptive_norm)
    op.set_dit_rms_norm(dit_cuda_plugin is not None and dit_rms_norm)
    op.set_dit_self_gated_residual(dit_cuda_plugin is not None and dit_self_gated_residual)
    op.set_dit_ffn_gated_residual(dit_cuda_plugin is not None and dit_ffn_gated_residual)
    op.set_dit_cross_affine_layer_norm(dit_cuda_plugin is not None and dit_cross_affine_layer_norm)
    op.set_dit_final_projection(dit_cuda_plugin is not None and dit_final_projection)
    if source_attention_plugin is not None:
        ctypes.CDLL(source_attention_plugin, mode=ctypes.RTLD_GLOBAL)
    if cuda_bf16_plugin is not None:
        ctypes.CDLL(cuda_bf16_plugin, mode=ctypes.RTLD_GLOBAL)
    if dit_cuda_plugin is not None:
        ctypes.CDLL(dit_cuda_plugin, mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 96 << 30)

    latent = network.add_input(
        "latents",
        trt.float32,
        (1, cfg.in_channels, cfg.latent_frames, cfg.latent_height, cfg.latent_width),
    )
    time_features = network.add_input("time_features", trt.float32, (1, cfg.freq_dim))
    text = network.add_input(
        "encoder_hidden_states", trt.float32, (1, cfg.text_seq_len, cfg.text_dim)
    )
    text_rows = network.add_shuffle(text)
    text_rows.reshape_dims = (cfg.text_seq_len, cfg.text_dim)

    hidden = _patchify(
        network,
        latent,
        weights["patch_embedding.weight"],
        weights["patch_embedding.bias"],
        cfg,
    )
    if debug_embeddings:
        debug = network.add_identity(op.cast(network, hidden, trt.float32)).get_output(0)
        debug.name = "patch_hidden"
        network.mark_output(debug)
    # Upstream expands the scalar diffusion timestep to every latent token
    # *before* running the FP32 time MLP.  A singleton MLP invocation is not
    # numerically equivalent on CUDA because GEMM dispatch depends on the row
    # count.  Materialize the same [num_patches, freq_dim] input here so the
    # family-owned source operators see the exact upstream shape.
    expanded_time_features = network.add_elementwise(
        time_features,
        op.constant(
            network,
            np.zeros((cfg.num_patches, cfg.freq_dim), dtype=np.float32),
        ),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    time_linear1 = op.source_time_linear1(
        network,
        expanded_time_features,
        weights["condition_embedder.time_embedder.linear_1.weight"],
        weights["condition_embedder.time_embedder.linear_1.bias"],
    )
    if time_linear1 is None:
        time_linear1 = op.linear(
            network,
            expanded_time_features,
            weights["condition_embedder.time_embedder.linear_1.weight"],
            weights["condition_embedder.time_embedder.linear_1.bias"],
            bf16=False,
        )
    if debug_embeddings:
        debug = network.add_identity(op.cast(network, time_linear1, trt.float32)).get_output(0)
        debug.name = "time_linear1"
        network.mark_output(debug)
    time_silu = op.silu(network, time_linear1)
    if debug_embeddings:
        debug = network.add_identity(op.cast(network, time_silu, trt.float32)).get_output(0)
        debug.name = "time_silu"
        network.mark_output(debug)
    time_embed = op.source_time_linear2(
        network,
        time_silu,
        weights["condition_embedder.time_embedder.linear_2.weight"],
        weights["condition_embedder.time_embedder.linear_2.bias"],
    )
    if time_embed is None:
        time_embed = op.linear(
            network,
            time_silu,
            weights["condition_embedder.time_embedder.linear_2.weight"],
            weights["condition_embedder.time_embedder.linear_2.bias"],
            bf16=False,
        )
    if debug_embeddings:
        debug = network.add_identity(op.cast(network, time_embed, trt.float32)).get_output(0)
        debug.name = "time_embed"
        network.mark_output(debug)
    time_proj = op.silu(network, time_embed)
    projected_time = op.source_time_projection(
        network,
        time_proj,
        weights["condition_embedder.time_proj.weight"],
        weights["condition_embedder.time_proj.bias"],
    )
    if projected_time is None:
        projected_time = op.linear(
            network,
            time_proj,
            weights["condition_embedder.time_proj.weight"],
            weights["condition_embedder.time_proj.bias"],
            bf16=False,
        )
    time_proj = projected_time
    if debug_embeddings:
        debug = network.add_identity(op.cast(network, time_proj, trt.float32)).get_output(0)
        debug.name = "time_proj"
        network.mark_output(debug)

    text_hidden = op.linear(
        network,
        text_rows.get_output(0),
        weights["condition_embedder.text_embedder.linear_1.weight"],
        weights["condition_embedder.text_embedder.linear_1.bias"],
        bf16=True,
    )
    text_hidden = op.gelu_tanh(network, text_hidden)
    text_hidden = op.linear(
        network,
        text_hidden,
        weights["condition_embedder.text_embedder.linear_2.weight"],
        weights["condition_embedder.text_embedder.linear_2.bias"],
        bf16=True,
    )
    if debug_embeddings:
        debug = network.add_identity(op.cast(network, text_hidden, trt.float32)).get_output(0)
        debug.name = "text_hidden"
        network.mark_output(debug)

    rope_cos, rope_sin = _wan_rope(cfg.latent_frames, cfg.latent_height, cfg.latent_width)
    for index in range(layers):
        prefix = f"blocks.{index}"
        if index in debug_full_norm_layers:
            _mark_debug_tensor(network, hidden, f"block_{index}_input")
        elif index in debug_sub_layers:
            _mark_debug_rows(network, hidden, f"block_{index}_input")
        table = weights[f"{prefix}.scale_shift_table"].reshape(1, 6 * cfg.dim)
        modulation = network.add_elementwise(
            op.constant(network, table), time_proj, trt.ElementWiseOperation.SUM
        ).get_output(0)
        if index in debug_sub_layers:
            _mark_debug_rows(network, modulation, f"block_{index}_modulation")
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = _slice_chunks(
            network, modulation, 6, cfg.dim
        )

        normalized = op.layer_norm(
            network,
            hidden,
            cfg.dim,
            cfg.eps,
            round_bf16=(index == 0 or round_residual_bf16),
        )
        if index in debug_full_norm_layers:
            _mark_debug_tensor(network, normalized, f"block_{index}_self_norm")
        qkv_input = op.adaptive_norm(network, normalized, shift_sa, scale_sa)
        if index in debug_full_norm_layers:
            _mark_debug_tensor(network, qkv_input, f"block_{index}_self_input")
        elif index in debug_sub_layers:
            _mark_debug_rows(network, qkv_input, f"block_{index}_self_input")
        q = op.linear(
            network,
            qkv_input,
            weights[f"{prefix}.attn1.to_q.weight"],
            weights[f"{prefix}.attn1.to_q.bias"],
        )
        if index in debug_sub_layers or index in debug_full_attention_layers:
            marker = (
                _mark_debug_tensor if index in debug_full_attention_layers else _mark_debug_rows
            )
            marker(network, q, f"block_{index}_self_q_linear")
        k = op.linear(
            network,
            qkv_input,
            weights[f"{prefix}.attn1.to_k.weight"],
            weights[f"{prefix}.attn1.to_k.bias"],
        )
        if index in debug_sub_layers or index in debug_full_attention_layers:
            marker = (
                _mark_debug_tensor if index in debug_full_attention_layers else _mark_debug_rows
            )
            marker(network, k, f"block_{index}_self_k_linear")
        v = op.linear(
            network,
            qkv_input,
            weights[f"{prefix}.attn1.to_v.weight"],
            weights[f"{prefix}.attn1.to_v.bias"],
        )
        if index in debug_sub_layers or index in debug_full_attention_layers:
            marker = (
                _mark_debug_tensor if index in debug_full_attention_layers else _mark_debug_rows
            )
            marker(network, v, f"block_{index}_self_v_linear")
        q = op.rms_norm(network, q, weights[f"{prefix}.attn1.norm_q.weight"], cfg.dim, cfg.eps)
        k = op.rms_norm(network, k, weights[f"{prefix}.attn1.norm_k.weight"], cfg.dim, cfg.eps)
        if index in debug_sub_layers or index in debug_full_attention_layers:
            marker = (
                _mark_debug_tensor if index in debug_full_attention_layers else _mark_debug_rows
            )
            marker(network, q, f"block_{index}_self_q_norm")
            marker(network, k, f"block_{index}_self_k_norm")
        q = op.rotary(network, q, rope_cos, rope_sin, cfg.num_patches, cfg.num_heads, cfg.head_dim)
        k = op.rotary(network, k, rope_cos, rope_sin, cfg.num_patches, cfg.num_heads, cfg.head_dim)
        if index in debug_sub_layers or index in debug_full_attention_layers:
            marker = (
                _mark_debug_tensor if index in debug_full_attention_layers else _mark_debug_rows
            )
            marker(network, q, f"block_{index}_self_q_rotated")
            marker(network, k, f"block_{index}_self_k_rotated")
        attended = op.attention(
            network,
            q,
            k,
            v,
            q_seq=cfg.num_patches,
            kv_seq=cfg.num_patches,
            heads=cfg.num_heads,
            head_dim=cfg.head_dim,
        )
        if index in debug_sub_layers or index in debug_full_attention_layers:
            marker = (
                _mark_debug_tensor if index in debug_full_attention_layers else _mark_debug_rows
            )
            marker(network, attended, f"block_{index}_self_attention")
        attended = op.linear(
            network,
            attended,
            weights[f"{prefix}.attn1.to_out.0.weight"],
            weights[f"{prefix}.attn1.to_out.0.bias"],
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, attended, f"block_{index}_self_projection")
        if index in debug_sub_layers:
            debug = network.add_identity(op.cast(network, attended, trt.float32)).get_output(0)
            debug.name = f"block_{index}_self_update"
            network.mark_output(debug)
        hidden = op.add_fp32_residual(
            network,
            hidden,
            attended,
            gate_sa,
            round_bf16=round_residual_bf16,
            source_exact_gated_stage="self_attention",
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, hidden, f"block_{index}_post_self")

        cross_input = op.affine_layer_norm(
            network,
            hidden,
            weights[f"{prefix}.norm2.weight"],
            weights[f"{prefix}.norm2.bias"],
            cfg.dim,
            cfg.eps,
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, cross_input, f"block_{index}_cross_norm")
        if index in debug_sub_layers:
            _mark_debug_rows(network, cross_input, f"block_{index}_cross_input")
        cq = op.linear(
            network,
            cross_input,
            weights[f"{prefix}.attn2.to_q.weight"],
            weights[f"{prefix}.attn2.to_q.bias"],
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, cq, f"block_{index}_cross_q_linear")
        if index in debug_sub_layers:
            _mark_debug_rows(network, cq, f"block_{index}_cross_q_linear")
        ck = op.linear(
            network,
            text_hidden,
            weights[f"{prefix}.attn2.to_k.weight"],
            weights[f"{prefix}.attn2.to_k.bias"],
        )
        if index in debug_full_substage_layers or index in debug_cross_k_norm_layers:
            _mark_debug_tensor(network, ck, f"block_{index}_cross_k_linear")
        cv = op.linear(
            network,
            text_hidden,
            weights[f"{prefix}.attn2.to_v.weight"],
            weights[f"{prefix}.attn2.to_v.bias"],
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, cv, f"block_{index}_cross_v_linear")
        cq = op.rms_norm(network, cq, weights[f"{prefix}.attn2.norm_q.weight"], cfg.dim, cfg.eps)
        ck = op.rms_norm(
            network,
            ck,
            weights[f"{prefix}.attn2.norm_k.weight"],
            cfg.dim,
            cfg.eps,
            debug_weight_name=(
                f"block_{index}_cross_k_weight" if index in debug_cross_k_norm_layers else None
            ),
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, cq, f"block_{index}_cross_q_norm")
        if index in debug_cross_k_norm_layers:
            _mark_debug_tensor(network, ck, f"block_{index}_cross_k_norm")
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, ck, f"block_{index}_cross_k_norm")
        cross = op.attention(
            network,
            cq,
            ck,
            cv,
            q_seq=cfg.num_patches,
            kv_seq=cfg.text_seq_len,
            heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            fp32_accumulation=cross_attention_fp32,
        )
        if index in debug_sub_layers:
            _mark_debug_rows(network, cross, f"block_{index}_cross_attention")
        elif index in debug_full_substage_layers:
            _mark_debug_tensor(network, cross, f"block_{index}_cross_attention")
        cross = op.linear(
            network,
            cross,
            weights[f"{prefix}.attn2.to_out.0.weight"],
            weights[f"{prefix}.attn2.to_out.0.bias"],
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, cross, f"block_{index}_cross_projection")
        if index in debug_sub_layers:
            debug = network.add_identity(op.cast(network, cross, trt.float32)).get_output(0)
            debug.name = f"block_{index}_cross_update"
            network.mark_output(debug)
        hidden = op.add_fp32_residual(network, hidden, cross, round_bf16=round_residual_bf16)
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, hidden, f"block_{index}_post_cross")

        normalized = op.layer_norm(network, hidden, cfg.dim, cfg.eps)
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, normalized, f"block_{index}_ffn_norm")
        ffn_input = op.adaptive_norm(network, normalized, shift_ff, scale_ff)
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, ffn_input, f"block_{index}_ffn_input")
        if index in debug_sub_layers:
            _mark_debug_rows(network, ffn_input, f"block_{index}_ffn_input")
        ffn = op.linear(
            network,
            ffn_input,
            weights[f"{prefix}.ffn.net.0.proj.weight"],
            weights[f"{prefix}.ffn.net.0.proj.bias"],
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, ffn, f"block_{index}_ffn_linear1")
        if index in debug_sub_layers:
            _mark_debug_rows(network, ffn, f"block_{index}_ffn_linear1")
        ffn = op.gelu_tanh(network, ffn)
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, ffn, f"block_{index}_ffn_gelu")
        if index in debug_sub_layers:
            _mark_debug_rows(network, ffn, f"block_{index}_ffn_gelu")
        ffn = op.linear(
            network,
            ffn,
            weights[f"{prefix}.ffn.net.2.weight"],
            weights[f"{prefix}.ffn.net.2.bias"],
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, ffn, f"block_{index}_ffn_projection")
        if index in debug_sub_layers:
            debug = network.add_identity(op.cast(network, ffn, trt.float32)).get_output(0)
            debug.name = f"block_{index}_ffn_update"
            network.mark_output(debug)
        hidden = op.add_fp32_residual(
            network,
            hidden,
            ffn,
            gate_ff,
            round_bf16=round_residual_bf16,
            source_exact_gated_stage="ffn",
        )
        if index in debug_full_substage_layers:
            _mark_debug_tensor(network, hidden, f"block_{index}_post_ffn")
        if index in debug_layers:
            debug = network.add_identity(op.cast(network, hidden, trt.float32)).get_output(0)
            debug.name = f"block_{index}_hidden"
            network.mark_output(debug)

    final_table = weights["scale_shift_table"].reshape(1, 2 * cfg.dim)
    final_time = network.add_concatenation([time_embed, time_embed])
    final_time.axis = 1
    final_mod = network.add_elementwise(
        op.constant(network, final_table),
        final_time.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    final_shift, final_scale = _slice_chunks(network, final_mod, 2, cfg.dim)
    hidden = op.layer_norm(network, hidden, cfg.dim, cfg.eps)
    if debug_embeddings or debug_final_stages:
        debug = network.add_identity(op.cast(network, hidden, trt.float32)).get_output(0)
        debug.name = "final_norm"
        network.mark_output(debug)
    hidden = op.adaptive_norm(network, hidden, final_shift, final_scale)
    if debug_embeddings or debug_final_stages:
        debug = network.add_identity(op.cast(network, hidden, trt.float32)).get_output(0)
        debug.name = "final_input"
        network.mark_output(debug)
    rows = op.source_final_projection(
        network, hidden, weights["proj_out.weight"], weights["proj_out.bias"]
    )
    if rows is None:
        rows = op.linear(
            network,
            hidden,
            weights["proj_out.weight"],
            weights["proj_out.bias"],
            bf16=False,
        )
    if debug_embeddings or debug_final_stages:
        debug = network.add_identity(op.cast(network, rows, trt.float32)).get_output(0)
        debug.name = "final_rows"
        network.mark_output(debug)
    # Head.forward temporarily disables the outer BF16 autocast and returns
    # FP32 rows.  The outer autocast becomes active again in unpatchify, where
    # torch.einsum quantizes those rows to BF16.  Preserve that source dtype
    # boundary before the layout-only TensorRT shuffles.
    rows = op.cast(network, rows, trt.bfloat16)
    output = _unpatchify(network, rows, cfg)
    output = op.cast(network, output, trt.float32)
    output.name = "noise_prediction"
    network.mark_output(output)

    print(
        f"[wan2.2-ti2v] building DiT: layers={layers}, patches={cfg.num_patches}, "
        f"latent={cfg.latent_frames}x{cfg.latent_height}x{cfg.latent_width}",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build Wan2.2 TI2V denoiser")
    return bytes(plan)
