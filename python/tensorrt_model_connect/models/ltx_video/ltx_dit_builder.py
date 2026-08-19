# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-Video DiT engine builder using the raw TensorRT network API.

This builder targets diffusers' ``LTXVideoTransformer3DModel`` directly.
It does not use ONNX, Torch-TensorRT, or a runtime Python dependency.

Engine I/O:
    Inputs:
        hidden_states             [1, S, 128]    fp16/fp32
        encoder_hidden_states     [1, 128, 4096] fp16/fp32
        timestep                  [1]            fp32
        encoder_attention_mask    [1, 128]       fp32, 1 = valid token
    Output:
        sample                    [1, S, 128]    fp32
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tensorrt_model_connect import trt_compat
from .checkpoint_mapper import WeightDict, _has_tensor, _load_tensor, _open_safetensors

if TYPE_CHECKING:
    from collections.abc import Mapping

trt: Any = None
graph_ops: Any = None


def _ensure_trt() -> Any:
    global trt
    if trt is None:
        trt = trt_compat.get_trt()
    return trt


def _ensure_graph_ops() -> Any:
    global graph_ops
    if graph_ops is None:
        from . import graph_ops as graph_ops_module

        graph_ops = graph_ops_module
    return graph_ops


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision == "fp16" else np.float32


def _trt_dtype(precision: str) -> trt.DataType:
    trt_module = _ensure_trt()
    return trt_module.float16 if precision == "fp16" else trt_module.float32


def _cast_back(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _transpose(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(arr.T, dtype=dtype)


def _array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=dtype)


def load_ltx_dit_weights(
    model_dir: str | Path,
    *,
    num_layers: int = 28,
    precision: str = "fp16",
) -> WeightDict:
    """Load LTX denoiser weights from a diffusers transformer directory."""
    readers = _open_safetensors(Path(model_dir))
    dtype = _target_np_dtype(precision)
    weights = WeightDict()

    def t(name: str) -> np.ndarray:
        return _transpose(_load_tensor(readers, name), dtype)

    def f(name: str, *, norm: bool = False) -> np.ndarray:
        return _array(_load_tensor(readers, name), np.float32 if norm else dtype)

    def maybe(name: str, *, norm: bool = False) -> np.ndarray | None:
        if not _has_tensor(readers, name):
            return None
        return f(name, norm=norm)

    weights["scale_shift_table"] = f("scale_shift_table")
    weights["proj_in.weight"] = t("proj_in.weight")
    weights["proj_in.bias"] = f("proj_in.bias")

    for layer in ("linear_1", "linear_2"):
        p = f"time_embed.emb.timestep_embedder.{layer}"
        weights[f"{p}.weight"] = t(f"{p}.weight")
        weights[f"{p}.bias"] = f(f"{p}.bias")
    weights["time_embed.linear.weight"] = t("time_embed.linear.weight")
    weights["time_embed.linear.bias"] = f("time_embed.linear.bias")

    for layer in ("linear_1", "linear_2"):
        p = f"caption_projection.{layer}"
        weights[f"{p}.weight"] = t(f"{p}.weight")
        weights[f"{p}.bias"] = f(f"{p}.bias")

    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        weights[f"{p}.scale_shift_table"] = f(f"{p}.scale_shift_table")

        for attn in ("attn1", "attn2"):
            ap = f"{p}.{attn}"
            weights[f"{ap}.norm_q.weight"] = f(f"{ap}.norm_q.weight", norm=True)
            weights[f"{ap}.norm_k.weight"] = f(f"{ap}.norm_k.weight", norm=True)
            for proj in ("to_q", "to_k", "to_v"):
                weights[f"{ap}.{proj}.weight"] = t(f"{ap}.{proj}.weight")
                bias = maybe(f"{ap}.{proj}.bias")
                if bias is not None:
                    weights[f"{ap}.{proj}.bias"] = bias
            weights[f"{ap}.to_out.0.weight"] = t(f"{ap}.to_out.0.weight")
            bias = maybe(f"{ap}.to_out.0.bias")
            if bias is not None:
                weights[f"{ap}.to_out.0.bias"] = bias

        weights[f"{p}.ff.net.0.proj.weight"] = t(f"{p}.ff.net.0.proj.weight")
        weights[f"{p}.ff.net.0.proj.bias"] = f(f"{p}.ff.net.0.proj.bias")
        weights[f"{p}.ff.net.2.weight"] = t(f"{p}.ff.net.2.weight")
        weights[f"{p}.ff.net.2.bias"] = f(f"{p}.ff.net.2.bias")

    weights["proj_out.weight"] = t("proj_out.weight")
    weights["proj_out.bias"] = f("proj_out.bias")
    return weights


def build_ltx_dit_engine(
    weights: "Mapping[str, np.ndarray]",
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    text_seq_len: int = 128,
    text_dim: int = 4096,
    in_channels: int = 128,
    dim: int = 2048,
    num_heads: int = 32,
    num_layers: int = 28,
    frame_rate: int = 25,
    temporal_compression_ratio: int = 8,
    spatial_compression_ratio: int = 32,
    eps: float = 1e-6,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    """Build the LTX transformer denoiser as a TensorRT plan."""
    if precision not in ("fp16", "fp32"):
        raise ValueError("LTX DiT raw builder currently supports fp16 or fp32")

    _ensure_trt()
    _ensure_graph_ops()
    seq_len = latent_frames * latent_height * latent_width
    head_dim = dim // num_heads
    trt_dtype = _trt_dtype(precision)
    np_dtype = _target_np_dtype(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    hidden_in = network.add_input(
        "hidden_states", trt_dtype, (1, seq_len, in_channels)
    )
    encoder_in = network.add_input(
        "encoder_hidden_states", trt_dtype, (1, text_seq_len, text_dim)
    )
    timestep_in = network.add_input("timestep", trt.float32, (1,))
    encoder_mask = network.add_input(
        "encoder_attention_mask", trt.float32, (1, text_seq_len)
    )

    block_eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32)
    )
    qk_eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32)
    )

    hidden = _drop_batch(network, hidden_in, (seq_len, in_channels))
    encoder_hidden = _drop_batch(network, encoder_in, (text_seq_len, text_dim))

    hidden = _linear(network, hidden, in_channels, dim, weights, "proj_in", np_dtype)

    time_proj = graph_ops.add_timestep_embedding(
        network, timestep_in, dim=256, dtype=np.float32
    )
    embedded_timestep = _linear(
        network,
        time_proj,
        256,
        dim,
        weights,
        "time_embed.emb.timestep_embedder.linear_1",
        np_dtype,
    )
    embedded_timestep = graph_ops.add_silu(network, embedded_timestep)
    embedded_timestep = _linear(
        network,
        embedded_timestep,
        dim,
        dim,
        weights,
        "time_embed.emb.timestep_embedder.linear_2",
        np_dtype,
    )
    temb = graph_ops.add_silu(network, embedded_timestep)
    temb = _linear(
        network, temb, dim, 6 * dim, weights, "time_embed.linear", np_dtype
    )

    context = _linear(
        network,
        encoder_hidden,
        text_dim,
        dim,
        weights,
        "caption_projection.linear_1",
        np_dtype,
    )
    context = graph_ops.add_gelu_new(network, context, dtype=np_dtype)
    context = _linear(
        network,
        context,
        dim,
        dim,
        weights,
        "caption_projection.linear_2",
        np_dtype,
    )

    rotary_cos, rotary_sin = make_ltx_rope_tables(
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        dim=dim,
        frame_rate=frame_rate,
        temporal_compression_ratio=temporal_compression_ratio,
        spatial_compression_ratio=spatial_compression_ratio,
    )
    rotary_cos_t = graph_ops.add_constant(
        network, (seq_len, dim), rotary_cos, dtype=np.float32
    )
    rotary_sin_t = graph_ops.add_constant(
        network, (seq_len, dim), rotary_sin, dtype=np.float32
    )
    rot_half = graph_ops.add_constant(
        network,
        (dim, dim),
        _make_ltx_rotate_half_matrix(dim, num_heads, interleaved=True),
        dtype=np.float32,
    )

    cross_mask = _make_cross_attention_mask(
        network, encoder_mask, text_seq_len=text_seq_len
    )

    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            _ltx_block_modulation(network, temb, weights[f"{p}.scale_shift_table"], dim)
        )

        norm_hidden = graph_ops.add_rms_norm(
            network,
            hidden,
            dim,
            np.ones(dim, dtype=np.float32),
            block_eps_t,
            dtype=np_dtype,
        )
        norm_hidden = _modulate(network, norm_hidden, scale_msa, shift_msa)

        attn_hidden = _ltx_attention(
            network,
            norm_hidden,
            None,
            None,
            weights,
            f"{p}.attn1",
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq_len=seq_len,
            kv_seq_len=seq_len,
            eps_t=qk_eps_t,
            dtype=np_dtype,
            rotary_cos=rotary_cos_t,
            rotary_sin=rotary_sin_t,
            rot_half=rot_half,
        )
        hidden = _residual_gated(network, hidden, attn_hidden, gate_msa)

        cross_hidden = _ltx_attention(
            network,
            hidden,
            context,
            cross_mask,
            weights,
            f"{p}.attn2",
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq_len=seq_len,
            kv_seq_len=text_seq_len,
            eps_t=qk_eps_t,
            dtype=np_dtype,
        )
        hidden = network.add_elementwise(
            hidden, cross_hidden, trt.ElementWiseOperation.SUM
        ).get_output(0)

        ff_norm = graph_ops.add_rms_norm(
            network,
            hidden,
            dim,
            np.ones(dim, dtype=np.float32),
            block_eps_t,
            dtype=np_dtype,
        )
        ff_norm = _modulate(network, ff_norm, scale_mlp, shift_mlp)
        ff_out = _ffn(network, ff_norm, weights, p, dim, np_dtype)
        hidden = _residual_gated(network, hidden, ff_out, gate_mlp)

    shift, scale = _final_modulation(
        network, embedded_timestep, weights["scale_shift_table"], dim
    )
    out = graph_ops.add_layer_norm(
        network,
        hidden,
        dim,
        np.ones(dim, dtype=np.float32),
        np.zeros(dim, dtype=np.float32),
        block_eps_t,
        dtype=np_dtype,
    )
    out = _modulate(network, out, scale, shift)
    out = _linear(network, out, dim, in_channels, weights, "proj_out", np_dtype)

    out_batched = network.add_shuffle(out)
    out_batched.reshape_dims = (1, seq_len, in_channels)
    out_fp32 = network.add_cast(out_batched.get_output(0), trt.float32).get_output(0)
    out_fp32.name = "sample"
    network.mark_output(out_fp32)

    print(
        "[ltx-dit] Building TRT engine "
        f"(precision={precision}, tokens={seq_len}, layers={num_layers}, "
        f"dim={dim}, heads={num_heads}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for LTX DiT")
    return bytes(plan)


def make_ltx_rope_tables(
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    dim: int,
    frame_rate: int,
    temporal_compression_ratio: int = 8,
    spatial_compression_ratio: int = 32,
    base_num_frames: int = 20,
    base_height: int = 2048,
    base_width: int = 2048,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute LTX's 3-axis RoPE tables for the fixed latent grid."""
    grid_f, grid_h, grid_w = np.meshgrid(
        np.arange(latent_frames, dtype=np.float32),
        np.arange(latent_height, dtype=np.float32),
        np.arange(latent_width, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([grid_f, grid_h, grid_w], axis=-1).reshape(-1, 3)
    coords[:, 0] *= (temporal_compression_ratio / float(frame_rate)) / base_num_frames
    coords[:, 1] *= spatial_compression_ratio / base_height
    coords[:, 2] *= spatial_compression_ratio / base_width

    freq_count = dim // 6
    freqs = theta ** np.linspace(
        math.log(1.0, theta),
        math.log(theta, theta),
        freq_count,
        dtype=np.float32,
    )
    freqs = freqs * (math.pi / 2.0)
    # Diffusers flattens the RoPE features as frequency triplets:
    # [t_freq0, h_freq0, w_freq0, t_freq1, h_freq1, w_freq1, ...].
    # Keeping the [token, freq, axis] order here is required before the
    # final repeat-interleave over real/imaginary pairs.
    angles = freqs[None, :, None] * (coords[:, None, :] * 2.0 - 1.0)
    angles = angles.reshape(coords.shape[0], -1)
    cos = np.repeat(np.cos(angles), 2, axis=-1).astype(np.float32)
    sin = np.repeat(np.sin(angles), 2, axis=-1).astype(np.float32)
    pad = dim % 6
    if pad:
        cos = np.concatenate(
            [np.ones((coords.shape[0], pad), dtype=np.float32), cos], axis=-1
        )
        sin = np.concatenate(
            [np.zeros((coords.shape[0], pad), dtype=np.float32), sin], axis=-1
        )
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _make_ltx_rotate_half_matrix(
    dim: int,
    num_heads: int,
    *,
    interleaved: bool,
) -> np.ndarray:
    """Return a row-vector matrix for RoPE rotate-half within each attention head."""
    if num_heads <= 0 or dim % num_heads != 0:
        raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

    head_dim = dim // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE rotate-half requires even head_dim, got {head_dim}")

    matrix = np.zeros((dim, dim), dtype=np.float32)
    for head in range(num_heads):
        start = head * head_dim
        if interleaved:
            for offset in range(0, head_dim, 2):
                a = start + offset
                b = a + 1
                matrix[b, a] = -1.0
                matrix[a, b] = 1.0
        else:
            half = head_dim // 2
            for offset in range(half):
                a = start + offset
                b = a + half
                matrix[b, a] = -1.0
                matrix[a, b] = 1.0
    return matrix


def _drop_batch(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    shape: tuple[int, ...],
) -> trt.ITensor:
    s = network.add_shuffle(tensor)
    s.reshape_dims = shape
    return s.get_output(0)


def _linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    in_dim: int,
    out_dim: int,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    dtype: np.dtype,
) -> trt.ITensor:
    out = graph_ops.add_matmul_rhs_constant(
        network, inp, in_dim, out_dim, weights[f"{prefix}.weight"], dtype=dtype
    )
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, bias, dtype=dtype)
    return out


def _modulate(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    scale: trt.ITensor,
    shift: trt.ITensor,
) -> trt.ITensor:
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32)
    )
    one = _cast_back(network, one, x.dtype)
    scale = _cast_back(network, scale, x.dtype)
    shift = _cast_back(network, shift, x.dtype)
    scale_plus_one = network.add_elementwise(
        one, scale, trt.ElementWiseOperation.SUM
    ).get_output(0)
    scaled = network.add_elementwise(
        x, scale_plus_one, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(
        scaled, shift, trt.ElementWiseOperation.SUM
    ).get_output(0)


def _ltx_block_modulation(
    network: trt.INetworkDefinition,
    temb: trt.ITensor,
    table: np.ndarray,
    dim: int,
) -> list[trt.ITensor]:
    chunks: list[trt.ITensor] = []
    for i in range(6):
        t = network.add_slice(temb, (0, i * dim), (1, dim), (1, 1)).get_output(0)
        c = graph_ops.add_constant(
            network, (1, dim), table[i].reshape(1, dim), dtype=table.dtype
        )
        c = _cast_back(network, c, t.dtype)
        chunks.append(
            network.add_elementwise(t, c, trt.ElementWiseOperation.SUM).get_output(0)
        )
    return chunks


def _final_modulation(
    network: trt.INetworkDefinition,
    embedded_timestep: trt.ITensor,
    table: np.ndarray,
    dim: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    out = []
    for i in range(2):
        c = graph_ops.add_constant(
            network, (1, dim), table[i].reshape(1, dim), dtype=table.dtype
        )
        c = _cast_back(network, c, embedded_timestep.dtype)
        out.append(
            network.add_elementwise(
                embedded_timestep, c, trt.ElementWiseOperation.SUM
            ).get_output(0)
        )
    return out[0], out[1]


def _make_cross_attention_mask(
    network: trt.INetworkDefinition,
    mask: trt.ITensor,
    *,
    text_seq_len: int,
) -> trt.ITensor:
    one = graph_ops.add_constant(
        network, (1, text_seq_len), np.ones((1, text_seq_len), dtype=np.float32)
    )
    inv = network.add_elementwise(one, mask, trt.ElementWiseOperation.SUB).get_output(0)
    neg = graph_ops.add_constant(
        network, (1, 1), np.array([-10000.0], dtype=np.float32)
    )
    additive = network.add_elementwise(inv, neg, trt.ElementWiseOperation.PROD)
    mask4 = network.add_shuffle(additive.get_output(0))
    mask4.reshape_dims = (1, 1, 1, text_seq_len)
    return mask4.get_output(0)


def _to_attention_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    x3 = network.add_shuffle(x)
    x3.reshape_dims = (seq_len, num_heads, head_dim)
    x3.second_transpose = trt.Permutation([1, 0, 2])
    x4 = network.add_shuffle(x3.get_output(0))
    x4.reshape_dims = (1, num_heads, seq_len, head_dim)
    return x4.get_output(0)


def _from_attention_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    x3 = network.add_shuffle(x)
    x3.reshape_dims = (num_heads, seq_len, head_dim)
    flat = network.add_shuffle(x3.get_output(0))
    flat.first_transpose = trt.Permutation([1, 0, 2])
    flat.reshape_dims = (seq_len, num_heads * head_dim)
    return flat.get_output(0)


def _apply_ltx_rope(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    cos: trt.ITensor,
    sin: trt.ITensor,
    rot_half: trt.ITensor,
) -> trt.ITensor:
    out_dtype = x.dtype
    x_fp32 = x if x.dtype == trt.float32 else network.add_cast(x, trt.float32).get_output(0)
    rotated = network.add_matrix_multiply(
        x_fp32, trt.MatrixOperation.NONE, rot_half, trt.MatrixOperation.NONE
    ).get_output(0)
    x_cos = network.add_elementwise(x_fp32, cos, trt.ElementWiseOperation.PROD).get_output(0)
    rot_sin = network.add_elementwise(rotated, sin, trt.ElementWiseOperation.PROD).get_output(0)
    out = network.add_elementwise(x_cos, rot_sin, trt.ElementWiseOperation.SUM).get_output(0)
    return _cast_back(network, out, out_dtype)


def _ltx_attention(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    context: trt.ITensor | None,
    mask: trt.ITensor | None,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    *,
    dim: int,
    num_heads: int,
    head_dim: int,
    q_seq_len: int,
    kv_seq_len: int,
    eps_t: trt.ITensor,
    dtype: np.dtype,
    rotary_cos: trt.ITensor | None = None,
    rotary_sin: trt.ITensor | None = None,
    rot_half: trt.ITensor | None = None,
) -> trt.ITensor:
    kv_source = hidden if context is None else context
    q = _linear(network, hidden, dim, dim, weights, f"{prefix}.to_q", dtype)
    k = _linear(network, kv_source, dim, dim, weights, f"{prefix}.to_k", dtype)
    v = _linear(network, kv_source, dim, dim, weights, f"{prefix}.to_v", dtype)

    q = graph_ops.add_rms_norm(
        network, q, dim, weights[f"{prefix}.norm_q.weight"], eps_t, dtype=dtype
    )
    k = graph_ops.add_rms_norm(
        network, k, dim, weights[f"{prefix}.norm_k.weight"], eps_t, dtype=dtype
    )

    if rotary_cos is not None and rotary_sin is not None and rot_half is not None:
        q = _apply_ltx_rope(network, q, rotary_cos, rotary_sin, rot_half)
        k = _apply_ltx_rope(network, k, rotary_cos, rotary_sin, rot_half)

    q4 = _to_attention_4d(
        network, q, seq_len=q_seq_len, num_heads=num_heads, head_dim=head_dim
    )
    k4 = _to_attention_4d(
        network, k, seq_len=kv_seq_len, num_heads=num_heads, head_dim=head_dim
    )
    v4 = _to_attention_4d(
        network, v, seq_len=kv_seq_len, num_heads=num_heads, head_dim=head_dim
    )
    if mask is not None:
        mask = _cast_back(network, mask, q4.dtype)
    ctx4 = graph_ops._add_attention_core(  # noqa: SLF001 - shared TRT primitive
        network, q4, k4, v4, causal=False, mask=mask
    )
    ctx = _from_attention_4d(
        network, ctx4, seq_len=q_seq_len, num_heads=num_heads, head_dim=head_dim
    )
    return _linear(network, ctx, dim, dim, weights, f"{prefix}.to_out.0", dtype)


def _residual_gated(
    network: trt.INetworkDefinition,
    residual: trt.ITensor,
    branch: trt.ITensor,
    gate: trt.ITensor,
) -> trt.ITensor:
    gate = _cast_back(network, gate, branch.dtype)
    gated = network.add_elementwise(branch, gate, trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        residual, gated.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)


def _ffn(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    dim: int,
    dtype: np.dtype,
) -> trt.ITensor:
    fc1_w = weights[f"{prefix}.ff.net.0.proj.weight"]
    ffn_dim = fc1_w.shape[1]
    x = graph_ops.add_matmul_rhs_constant(
        network, hidden, dim, ffn_dim, fc1_w, dtype=dtype
    )
    x = graph_ops.add_bias_sum(
        network, x, ffn_dim, weights[f"{prefix}.ff.net.0.proj.bias"], dtype=dtype
    )
    x = graph_ops.add_gelu_new(network, x, dtype=dtype)
    x = graph_ops.add_matmul_rhs_constant(
        network, x, ffn_dim, dim, weights[f"{prefix}.ff.net.2.weight"], dtype=dtype
    )
    return graph_ops.add_bias_sum(
        network, x, dim, weights[f"{prefix}.ff.net.2.bias"], dtype=dtype
    )
