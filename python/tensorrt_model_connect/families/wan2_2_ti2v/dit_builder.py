# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-free TensorRT denoiser for Wan2.2 TI2V-5B."""

from __future__ import annotations

import math
import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import trt_ops as op
from .checkpoint_mapper import (
    convert_transformer_state_dict,
    load_native_transformer_state_dict,
)
from .model_config import (
    SUPPORTED_GENERATION_PROFILES,
    WAN22_TI2V_5B,
    Wan22TI2VConfig,
)


trt = trt_compat.get_trt()


def _numpy_state(model_dir: str) -> dict[str, np.ndarray]:
    state = convert_transformer_state_dict(load_native_transformer_state_dict(model_dir))
    return {name: tensor.detach().float().cpu().numpy() for name, tensor in state.items()}


def _ffn_fp8_layer_names(profile: Wan22TI2VConfig) -> tuple[str, ...]:
    return tuple(
        name
        for index in range(profile.num_layers)
        for name in (
            f"blocks.{index}.ffn.net.0.proj",
            f"blocks.{index}.ffn.net.2",
        )
    )


def _validated_ffn_fp8_scales(
    weights: dict[str, np.ndarray],
    profile: Wan22TI2VConfig,
    scales: dict | None,
) -> dict[str, dict[str, float]] | None:
    """Validate a fail-closed FFN-only FP8 scale map.

    Every FFN projection must be present and no non-FFN layer is accepted.
    Weight scales may be omitted; in that case the checkpoint's exact
    per-tensor absolute maximum is used.
    """

    if scales is None:
        return None
    if not isinstance(scales, dict):
        raise TypeError("Wan2.2 FFN FP8 scales must be a dictionary")

    expected = set(_ffn_fp8_layer_names(profile))
    provided = set(scales)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise ValueError(
            "Wan2.2 FFN FP8 scales must cover exactly the two FFN projections "
            f"in every block; missing={missing}, unexpected={unexpected}"
        )

    result: dict[str, dict[str, float]] = {}
    for name in sorted(expected):
        entry = scales[name]
        if not isinstance(entry, dict):
            raise TypeError(f"Wan2.2 FFN FP8 scale entry for {name} must be a dictionary")
        input_scale = float(entry.get("input_scale", 0.0))
        if not math.isfinite(input_scale) or input_scale <= 0.0:
            raise ValueError(f"Wan2.2 FFN FP8 input_scale for {name} must be positive and finite")

        weight = weights[f"{name}.weight"]
        minimum_weight_scale = op.fp8_e4m3_weight_scale(weight)
        weight_scale = float(entry.get("weight_scale", minimum_weight_scale))
        if not math.isfinite(weight_scale) or weight_scale <= 0.0:
            raise ValueError(f"Wan2.2 FFN FP8 weight_scale for {name} must be positive and finite")
        if weight_scale < minimum_weight_scale * (1.0 - 1.0e-6):
            raise ValueError(
                f"Wan2.2 FFN FP8 weight_scale for {name} would overflow E4M3: "
                f"provided={weight_scale}, minimum={minimum_weight_scale}"
            )
        result[name] = {
            "input_scale": input_scale,
            "weight_scale": weight_scale,
        }
    return result


def _wan_rope(profile: Wan22TI2VConfig):
    grid = (
        profile.latent_frames // profile.patch_size[0],
        profile.latent_height // profile.patch_size[1],
        profile.latent_width // profile.patch_size[2],
    )
    half = profile.head_dim // 2
    parts = (half - 2 * (half // 3), half // 3, half // 3)
    tables = []
    for length, complex_dim in zip(grid, parts):
        real_dim = complex_dim * 2
        inverse = np.power(
            10000.0,
            -np.arange(0, real_dim, 2, dtype=np.float64) / real_dim,
        )
        tables.append(np.outer(np.arange(length, dtype=np.float64), inverse))
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


def _patchify(network, latent, weight, bias, profile: Wan22TI2VConfig):
    pt, ph, pw = profile.patch_size
    patches = network.add_shuffle(latent)
    patches.reshape_dims = (
        1,
        profile.in_channels,
        profile.latent_frames // pt,
        pt,
        profile.latent_height // ph,
        ph,
        profile.latent_width // pw,
        pw,
    )
    patches.second_transpose = trt.Permutation([0, 2, 4, 6, 1, 3, 5, 7])
    rows = network.add_shuffle(patches.get_output(0))
    rows.reshape_dims = (
        profile.num_patches,
        profile.in_channels * pt * ph * pw,
    )
    return op.linear(
        network,
        rows.get_output(0),
        weight.reshape(profile.dim, -1),
        bias,
    )


def _unpatchify(network, rows, profile: Wan22TI2VConfig):
    pt, ph, pw = profile.patch_size
    reshape = network.add_shuffle(rows)
    reshape.reshape_dims = (
        profile.latent_frames // pt,
        profile.latent_height // ph,
        profile.latent_width // pw,
        pt,
        ph,
        pw,
        profile.out_channels,
    )
    reshape.second_transpose = trt.Permutation([6, 0, 3, 1, 4, 2, 5])
    output = network.add_shuffle(reshape.get_output(0))
    output.reshape_dims = (
        1,
        profile.out_channels,
        profile.latent_frames,
        profile.latent_height,
        profile.latent_width,
    )
    return output.get_output(0)


def build_dit_engine(
    model_dir: str,
    *,
    profile: Wan22TI2VConfig = WAN22_TI2V_5B,
    ffn_fp8_scales: dict | None = None,
    verbose: bool = False,
) -> bytes:
    """Build a DiT plan for an explicitly qualified profile."""

    if profile not in SUPPORTED_GENERATION_PROFILES:
        raise ValueError("Wan2.2 DiT profile is not one of the qualified generation profiles")
    weights = _numpy_state(model_dir)
    ffn_fp8_scales = _validated_ffn_fp8_scales(weights, profile, ffn_fp8_scales)
    # TensorRT consumes raw pointers for explicit FP8 constants. Keep their
    # NumPy owners alive until build_serialized_network() has returned.
    fp8_weight_refs: list[np.ndarray] = []
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 96 << 30)

    latent = network.add_input(
        "latents",
        trt.float32,
        (
            1,
            profile.in_channels,
            profile.latent_frames,
            profile.latent_height,
            profile.latent_width,
        ),
    )
    time_features = network.add_input("time_features", trt.float32, (1, profile.freq_dim))
    text = network.add_input(
        "encoder_hidden_states",
        trt.float32,
        (1, profile.text_seq_len, profile.text_dim),
    )
    text_rows = network.add_shuffle(text)
    text_rows.reshape_dims = (profile.text_seq_len, profile.text_dim)

    hidden = _patchify(
        network,
        latent,
        weights["patch_embedding.weight"],
        weights["patch_embedding.bias"],
        profile,
    )
    # Upstream expands the scalar timestep before the FP32 MLP. The row count
    # influences GEMM dispatch, so materialize the same shape here.
    expanded_time_features = network.add_elementwise(
        time_features,
        op.constant(
            network,
            np.zeros((profile.num_patches, profile.freq_dim), dtype=np.float32),
        ),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    time_linear1 = op.linear(
        network,
        expanded_time_features,
        weights["condition_embedder.time_embedder.linear_1.weight"],
        weights["condition_embedder.time_embedder.linear_1.bias"],
        bf16=False,
    )
    time_embed = op.linear(
        network,
        op.silu(network, time_linear1),
        weights["condition_embedder.time_embedder.linear_2.weight"],
        weights["condition_embedder.time_embedder.linear_2.bias"],
        bf16=False,
    )
    time_proj = op.linear(
        network,
        op.silu(network, time_embed),
        weights["condition_embedder.time_proj.weight"],
        weights["condition_embedder.time_proj.bias"],
        bf16=False,
    )

    text_hidden = op.linear(
        network,
        text_rows.get_output(0),
        weights["condition_embedder.text_embedder.linear_1.weight"],
        weights["condition_embedder.text_embedder.linear_1.bias"],
    )
    text_hidden = op.gelu_tanh(network, text_hidden)
    text_hidden = op.linear(
        network,
        text_hidden,
        weights["condition_embedder.text_embedder.linear_2.weight"],
        weights["condition_embedder.text_embedder.linear_2.bias"],
    )

    rope_cos, rope_sin = _wan_rope(profile)
    for index in range(profile.num_layers):
        prefix = f"blocks.{index}"
        table = weights[f"{prefix}.scale_shift_table"].reshape(1, 6 * profile.dim)
        modulation = network.add_elementwise(
            op.constant(network, table),
            time_proj,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = _slice_chunks(
            network, modulation, 6, profile.dim
        )

        normalized = op.layer_norm(
            network,
            hidden,
            profile.dim,
            profile.eps,
            round_bf16=index == 0,
        )
        qkv_input = op.adaptive_norm(network, normalized, shift_sa, scale_sa)
        q, k, v = op.fused_qkv_linear(
            network,
            qkv_input,
            weights[f"{prefix}.attn1.to_q.weight"],
            weights[f"{prefix}.attn1.to_q.bias"],
            weights[f"{prefix}.attn1.to_k.weight"],
            weights[f"{prefix}.attn1.to_k.bias"],
            weights[f"{prefix}.attn1.to_v.weight"],
            weights[f"{prefix}.attn1.to_v.bias"],
            rows=profile.num_patches,
            hidden_size=profile.dim,
        )
        q = op.rms_norm(
            network,
            q,
            weights[f"{prefix}.attn1.norm_q.weight"],
            profile.dim,
            profile.eps,
        )
        k = op.rms_norm(
            network,
            k,
            weights[f"{prefix}.attn1.norm_k.weight"],
            profile.dim,
            profile.eps,
        )
        q = op.rotary(
            network,
            q,
            rope_cos,
            rope_sin,
            profile.num_patches,
            profile.num_heads,
            profile.head_dim,
        )
        k = op.rotary(
            network,
            k,
            rope_cos,
            rope_sin,
            profile.num_patches,
            profile.num_heads,
            profile.head_dim,
        )
        attended = op.attention(
            network,
            q,
            k,
            v,
            q_seq=profile.num_patches,
            kv_seq=profile.num_patches,
            heads=profile.num_heads,
            head_dim=profile.head_dim,
        )
        attended = op.linear(
            network,
            attended,
            weights[f"{prefix}.attn1.to_out.0.weight"],
            weights[f"{prefix}.attn1.to_out.0.bias"],
        )
        hidden = op.add_fp32_residual(network, hidden, attended, gate_sa)

        cross_input = op.affine_layer_norm(
            network,
            hidden,
            weights[f"{prefix}.norm2.weight"],
            weights[f"{prefix}.norm2.bias"],
            profile.dim,
            profile.eps,
        )
        cq = op.linear(
            network,
            cross_input,
            weights[f"{prefix}.attn2.to_q.weight"],
            weights[f"{prefix}.attn2.to_q.bias"],
        )
        ck = op.linear(
            network,
            text_hidden,
            weights[f"{prefix}.attn2.to_k.weight"],
            weights[f"{prefix}.attn2.to_k.bias"],
        )
        cv = op.linear(
            network,
            text_hidden,
            weights[f"{prefix}.attn2.to_v.weight"],
            weights[f"{prefix}.attn2.to_v.bias"],
        )
        cq = op.rms_norm(
            network,
            cq,
            weights[f"{prefix}.attn2.norm_q.weight"],
            profile.dim,
            profile.eps,
        )
        ck = op.rms_norm(
            network,
            ck,
            weights[f"{prefix}.attn2.norm_k.weight"],
            profile.dim,
            profile.eps,
        )
        cross = op.attention(
            network,
            cq,
            ck,
            cv,
            q_seq=profile.num_patches,
            kv_seq=profile.text_seq_len,
            heads=profile.num_heads,
            head_dim=profile.head_dim,
        )
        cross = op.linear(
            network,
            cross,
            weights[f"{prefix}.attn2.to_out.0.weight"],
            weights[f"{prefix}.attn2.to_out.0.bias"],
        )
        hidden = op.add_fp32_residual(network, hidden, cross)

        normalized = op.layer_norm(network, hidden, profile.dim, profile.eps)
        ffn_input = op.adaptive_norm(network, normalized, shift_ff, scale_ff)
        ffn_up_name = f"{prefix}.ffn.net.0.proj"
        if ffn_fp8_scales is None:
            ffn = op.linear(
                network,
                ffn_input,
                weights[f"{ffn_up_name}.weight"],
                weights[f"{ffn_up_name}.bias"],
            )
        else:
            ffn = op.linear_fp8_e4m3(
                network,
                ffn_input,
                weights[f"{ffn_up_name}.weight"],
                weights[f"{ffn_up_name}.bias"],
                input_scale=ffn_fp8_scales[ffn_up_name]["input_scale"],
                weight_scale=ffn_fp8_scales[ffn_up_name]["weight_scale"],
                weight_refs=fp8_weight_refs,
            )
        ffn = op.gelu_tanh(network, ffn)
        ffn_down_name = f"{prefix}.ffn.net.2"
        if ffn_fp8_scales is None:
            ffn = op.linear(
                network,
                ffn,
                weights[f"{ffn_down_name}.weight"],
                weights[f"{ffn_down_name}.bias"],
            )
        else:
            ffn = op.linear_fp8_e4m3(
                network,
                ffn,
                weights[f"{ffn_down_name}.weight"],
                weights[f"{ffn_down_name}.bias"],
                input_scale=ffn_fp8_scales[ffn_down_name]["input_scale"],
                weight_scale=ffn_fp8_scales[ffn_down_name]["weight_scale"],
                weight_refs=fp8_weight_refs,
            )
        hidden = op.add_fp32_residual(network, hidden, ffn, gate_ff)

    final_table = weights["scale_shift_table"].reshape(1, 2 * profile.dim)
    final_time = network.add_concatenation([time_embed, time_embed])
    final_time.axis = 1
    final_modulation = network.add_elementwise(
        op.constant(network, final_table),
        final_time.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    final_shift, final_scale = _slice_chunks(network, final_modulation, 2, profile.dim)
    hidden = op.layer_norm(network, hidden, profile.dim, profile.eps)
    hidden = op.adaptive_norm(network, hidden, final_shift, final_scale)
    rows = op.linear(
        network,
        hidden,
        weights["proj_out.weight"],
        weights["proj_out.bias"],
        bf16=False,
    )
    # Head.forward returns FP32, then the source unpatchify einsum runs under
    # BF16 autocast. Preserve that boundary before the layout-only shuffles.
    rows = op.cast(network, rows, trt.bfloat16)
    output = op.cast(network, _unpatchify(network, rows, profile), trt.float32)
    output.name = "noise_prediction"
    network.mark_output(output)

    print(
        f"[wan2.2-ti2v] building DiT: layers={profile.num_layers}, "
        f"patches={profile.num_patches}, "
        f"latent={profile.latent_frames}x{profile.latent_height}x{profile.latent_width}, "
        f"ffn_fp8={'enabled' if ffn_fp8_scales is not None else 'disabled'}",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build Wan2.2 TI2V denoiser")
    return bytes(plan)
