"""SANA-WM LTX-2 refiner denoiser builder using the raw TensorRT API.

This builder targets the video-only path used by
``diffusion/refiner/diffusers_ltx2_refiner.py`` in the public SANA-WM source.
It does not use ONNX, Torch-TensorRT, or a Python runtime bridge.

Engine I/O:
    Inputs:
        latent              [1, S, C]       fp16/bf16/fp32, sink + current tokens
        clean_latent        [1, S, C]       fp16/bf16/fp32, accepted for runtime ABI
        denoise_mask        [1, S, 1]       fp32, 0 for sink and 1 for current
        positions           [1, 3, S, 2]    fp16/bf16/fp32, accepted for runtime ABI
        v_context           [1, T, D_txt]   fp16/bf16/fp32, connector text embeddings
        v_attention_mask    [1, T]          fp32, 1 = valid token
        sigma               [1]             fp32
    Output:
        denoised            [1, S-current, C] fp32
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tensorrt_model_connect import trt_compat
from ...checkpoint_mapper import WeightDict
from ..ltx_video import ltx_dit_builder as ltx

if TYPE_CHECKING:
    from collections.abc import Mapping

trt: Any = None
graph_ops: Any = None


@dataclass(frozen=True)
class SanaWmRefinerShape:
    latent_frames: int
    latent_height: int
    latent_width: int
    context_tokens: int
    current_tokens: int
    total_tokens: int
    text_seq_len: int
    text_dim: int
    in_channels: int
    dim: int
    num_heads: int
    num_layers: int
    fps: int
    temporal_compression_ratio: int
    spatial_compression_ratio: int
    timestep_scale_multiplier: float
    rope_type: str


def _ensure_trt() -> Any:
    global trt
    if trt is None:
        trt = trt_compat.get_trt()
    ltx.trt = trt
    return trt


def _ensure_graph_ops() -> Any:
    global graph_ops
    if graph_ops is None:
        from ... import graph_ops as graph_ops_module

        graph_ops = graph_ops_module
    ltx.graph_ops = graph_ops
    return graph_ops


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision == "fp16" else np.float32


def _op_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision in ("fp16", "bf16") else np.float32


def _trt_dtype(precision: str) -> trt.DataType:
    trt_module = _ensure_trt()
    if precision == "fp16":
        return trt_module.float16
    if precision == "bf16":
        return trt_module.bfloat16
    return trt_module.float32


def load_sana_wm_refiner_dit_weights(
    transformer_dir: str | Path,
    *,
    num_layers: int = 48,
    precision: str = "fp16",
) -> WeightDict:
    """Load the SANA-WM LTX-2 refiner transformer weights in TRT layout."""
    return ltx.load_ltx_dit_weights(
        transformer_dir,
        num_layers=num_layers,
        precision=precision,
    )


def refiner_shape_from_config(
    raw_config: dict,
    transformer_config: dict | None = None,
) -> SanaWmRefinerShape:
    transformer_config = transformer_config or {}
    vae = raw_config.get("vae", {})
    if not isinstance(vae, dict):
        vae = {}
    vae_stride = raw_config.get("vae_stride", vae.get("vae_stride", (8, 32, 32)))
    if not isinstance(vae_stride, (list, tuple)):
        vae_stride = (8, 32, 32)
    stride_values = [int(v) for v in vae_stride]
    if len(stride_values) == 1:
        stride_values = [stride_values[0], stride_values[0], stride_values[0]]
    if len(stride_values) == 2:
        stride_values = [stride_values[0], stride_values[1], stride_values[1]]

    video_num_frames = int(raw_config.get("video_num_frames", 321))
    video_height = int(raw_config.get("video_height", 704))
    video_width = int(raw_config.get("video_width", 1280))
    latent_frames = (video_num_frames - 1) // stride_values[0] + 1
    latent_height = video_height // stride_values[-1]
    latent_width = video_width // stride_values[-1]
    context_tokens = latent_height * latent_width
    total_tokens = latent_frames * latent_height * latent_width

    num_heads = int(transformer_config.get("num_attention_heads", 32))
    head_dim = int(transformer_config.get("attention_head_dim", 128))
    dim = num_heads * head_dim
    text_dim = int(
        raw_config.get(
            "sana_wm_refiner_text_dim",
            transformer_config.get("caption_channels", 3840),
        )
    )
    return SanaWmRefinerShape(
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        context_tokens=context_tokens,
        current_tokens=total_tokens - context_tokens,
        total_tokens=total_tokens,
        text_seq_len=int(raw_config.get("sana_wm_refiner_text_max_length", 1024)),
        text_dim=text_dim,
        in_channels=int(
            transformer_config.get(
                "in_channels",
                vae.get("vae_latent_dim", raw_config.get("vae_latent_dim", 128)),
            )
        ),
        dim=dim,
        num_heads=num_heads,
        num_layers=int(transformer_config.get("num_layers", 48)),
        fps=int(raw_config.get("fps", 16)),
        temporal_compression_ratio=int(stride_values[0]),
        spatial_compression_ratio=int(stride_values[-1]),
        timestep_scale_multiplier=float(
            transformer_config.get("timestep_scale_multiplier", 1000.0)
        ),
        rope_type=str(transformer_config.get("rope_type", "split")),
    )


def build_sana_wm_refiner_dit_engine(
    weights: "Mapping[str, np.ndarray]",
    raw_config: dict,
    transformer_config: dict | None = None,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    """Build the SANA-WM LTX-2 refiner denoiser as a TensorRT plan."""
    if precision not in ("fp16", "bf16", "fp32"):
        raise ValueError("SANA-WM refiner DiT raw builder supports fp16, bf16, or fp32")

    trt_module = _ensure_trt()
    graph = _ensure_graph_ops()
    shape = refiner_shape_from_config(raw_config, transformer_config)
    head_dim = shape.dim // shape.num_heads
    trt_dtype = _trt_dtype(precision)
    op_dtype = _op_np_dtype(precision)
    weight_dtype = _target_np_dtype(precision)

    logger = trt_module.Logger(
        trt_module.Logger.VERBOSE if verbose else trt_module.Logger.WARNING
    )
    builder = trt_module.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt_module.MemoryPoolType.WORKSPACE, 64 << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 0

    network = builder.create_network(
        1 << int(trt_module.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    latent_in = network.add_input(
        "latent", trt_dtype, (1, shape.total_tokens, shape.in_channels)
    )
    clean_latent_in = network.add_input(
        "clean_latent", trt_dtype, (1, shape.total_tokens, shape.in_channels)
    )
    denoise_mask_in = network.add_input(
        "denoise_mask", trt_module.float32, (1, shape.total_tokens, 1)
    )
    positions_in = network.add_input(
        "positions", trt_dtype, (1, 3, shape.total_tokens, 2)
    )
    context_in = network.add_input(
        "v_context", trt_dtype, (1, shape.text_seq_len, shape.text_dim)
    )
    context_mask_in = network.add_input(
        "v_attention_mask", trt_module.float32, (1, shape.text_seq_len)
    )
    sigma_in = network.add_input("sigma", trt_module.float32, (1,))

    del clean_latent_in, positions_in

    block_eps_t = graph.add_constant(
        network, (1, 1), np.array([1.0e-6], dtype=np.float32)
    )
    qk_eps_t = graph.add_constant(
        network, (1, 1), np.array([1.0e-5], dtype=np.float32)
    )

    latent = ltx._drop_batch(network, latent_in, (shape.total_tokens, shape.in_channels))
    raw_timestep = _raw_timestep(network, denoise_mask_in, sigma_in, shape.total_tokens)
    model_timestep = _scale_timestep(
        network, raw_timestep, shape.timestep_scale_multiplier
    )

    hidden = ltx._linear(
        network,
        latent,
        shape.in_channels,
        shape.dim,
        weights,
        "proj_in",
        op_dtype,
        constant_dtype=weight_dtype,
    )
    timestep_embed = _add_timestep_embedding_rows(
        network, model_timestep, freq_dim=256, dtype=np.float32
    )
    embedded_timestep = ltx._linear(
        network,
        timestep_embed,
        256,
        shape.dim,
        weights,
        "time_embed.emb.timestep_embedder.linear_1",
        op_dtype,
        constant_dtype=weight_dtype,
    )
    embedded_timestep = graph.add_silu(network, embedded_timestep)
    embedded_timestep = ltx._linear(
        network,
        embedded_timestep,
        shape.dim,
        shape.dim,
        weights,
        "time_embed.emb.timestep_embedder.linear_2",
        op_dtype,
        constant_dtype=weight_dtype,
    )
    temb = graph.add_silu(network, embedded_timestep)
    temb = ltx._linear(
        network,
        temb,
        shape.dim,
        6 * shape.dim,
        weights,
        "time_embed.linear",
        op_dtype,
        constant_dtype=weight_dtype,
    )

    context = ltx._drop_batch(
        network, context_in, (shape.text_seq_len, shape.text_dim)
    )
    context = ltx._linear(
        network,
        context,
        shape.text_dim,
        shape.dim,
        weights,
        "caption_projection.linear_1",
        op_dtype,
        constant_dtype=weight_dtype,
    )
    context = graph.add_gelu_new(network, context, dtype=weight_dtype)
    context = ltx._linear(
        network,
        context,
        shape.dim,
        shape.dim,
        weights,
        "caption_projection.linear_2",
        op_dtype,
        constant_dtype=weight_dtype,
    )

    rotary_cos, rotary_sin = _make_refiner_rope_tables(shape, transformer_config)
    rotary_cos_t = graph.add_constant(
        network, (shape.total_tokens, shape.dim), rotary_cos, dtype=np.float32
    )
    rotary_sin_t = graph.add_constant(
        network, (shape.total_tokens, shape.dim), rotary_sin, dtype=np.float32
    )
    rot_half = graph.add_constant(
        network,
        (shape.dim, shape.dim),
        ltx._make_ltx_rotate_half_matrix(
            shape.dim, shape.num_heads, interleaved=shape.rope_type == "interleaved"
        ),
        dtype=np.float32,
    )
    self_mask = graph.add_constant(
        network,
        (1, 1, shape.total_tokens, shape.total_tokens),
        _streaming_self_attention_mask(shape.total_tokens, shape.context_tokens),
        dtype=np.float32,
    )
    cross_mask = ltx._make_cross_attention_mask(
        network, context_mask_in, text_seq_len=shape.text_seq_len
    )

    for i in range(shape.num_layers):
        p = f"transformer_blocks.{i}"
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            _refiner_block_modulation(network, temb, weights[f"{p}.scale_shift_table"], shape.dim)
        )

        norm_hidden = graph.add_rms_norm(
            network,
            hidden,
            shape.dim,
            np.ones(shape.dim, dtype=np.float32),
            block_eps_t,
            dtype=op_dtype,
        )
        norm_hidden = ltx._modulate(network, norm_hidden, scale_msa, shift_msa)
        attn_hidden = ltx._ltx_attention(
            network,
            norm_hidden,
            None,
            self_mask,
            weights,
            f"{p}.attn1",
            dim=shape.dim,
            num_heads=shape.num_heads,
            head_dim=head_dim,
            q_seq_len=shape.total_tokens,
            kv_seq_len=shape.total_tokens,
            eps_t=qk_eps_t,
            dtype=op_dtype,
            rotary_cos=rotary_cos_t,
            rotary_sin=rotary_sin_t,
            rot_half=rot_half,
            constant_dtype=weight_dtype,
        )
        hidden = ltx._residual_gated(network, hidden, attn_hidden, gate_msa)

        cross_norm = graph.add_rms_norm(
            network,
            hidden,
            shape.dim,
            np.ones(shape.dim, dtype=np.float32),
            block_eps_t,
            dtype=op_dtype,
        )
        cross_hidden = ltx._ltx_attention(
            network,
            cross_norm,
            context,
            cross_mask,
            weights,
            f"{p}.attn2",
            dim=shape.dim,
            num_heads=shape.num_heads,
            head_dim=head_dim,
            q_seq_len=shape.total_tokens,
            kv_seq_len=shape.text_seq_len,
            eps_t=qk_eps_t,
            dtype=op_dtype,
            constant_dtype=weight_dtype,
        )
        hidden = network.add_elementwise(
            hidden, cross_hidden, trt_module.ElementWiseOperation.SUM
        ).get_output(0)

        ff_norm = graph.add_rms_norm(
            network,
            hidden,
            shape.dim,
            np.ones(shape.dim, dtype=np.float32),
            block_eps_t,
            dtype=op_dtype,
        )
        ff_norm = ltx._modulate(network, ff_norm, scale_mlp, shift_mlp)
        ff_out = ltx._ffn(
            network,
            ff_norm,
            weights,
            p,
            shape.dim,
            op_dtype,
            constant_dtype=weight_dtype,
        )
        hidden = ltx._residual_gated(network, hidden, ff_out, gate_mlp)

    shift, scale = _refiner_final_modulation(
        network, embedded_timestep, weights["scale_shift_table"], shape.dim
    )
    velocity = graph.add_layer_norm(
        network,
        hidden,
        shape.dim,
        np.ones(shape.dim, dtype=np.float32),
        np.zeros(shape.dim, dtype=np.float32),
        block_eps_t,
        dtype=op_dtype,
    )
    velocity = ltx._modulate(network, velocity, scale, shift)
    velocity = ltx._linear(
        network,
        velocity,
        shape.dim,
        shape.in_channels,
        weights,
        "proj_out",
        op_dtype,
        constant_dtype=weight_dtype,
    )
    denoised = _denoised_x0(network, latent, velocity, raw_timestep)
    current = network.add_slice(
        denoised,
        (shape.context_tokens, 0),
        (shape.current_tokens, shape.in_channels),
        (1, 1),
    ).get_output(0)
    current_batched = network.add_shuffle(current)
    current_batched.reshape_dims = (1, shape.current_tokens, shape.in_channels)
    out = network.add_cast(current_batched.get_output(0), trt_module.float32).get_output(0)
    out.name = "denoised"
    network.mark_output(out)

    print(
        "[sana-wm-refiner] Building TRT engine "
        f"(precision={precision}, tokens={shape.total_tokens}, "
        f"context={shape.context_tokens}, layers={shape.num_layers}, "
        f"dim={shape.dim}, text_seq={shape.text_seq_len}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for SANA-WM refiner DiT")
    return bytes(plan)


def _raw_timestep(
    network: trt.INetworkDefinition,
    denoise_mask: trt.ITensor,
    sigma: trt.ITensor,
    total_tokens: int,
) -> trt.ITensor:
    mask = network.add_shuffle(denoise_mask)
    mask.reshape_dims = (total_tokens, 1)
    sigma_2d = network.add_shuffle(sigma)
    sigma_2d.reshape_dims = (1, 1)
    return network.add_elementwise(
        mask.get_output(0), sigma_2d.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


def _scale_timestep(
    network: trt.INetworkDefinition,
    raw_timestep: trt.ITensor,
    scale: float,
) -> trt.ITensor:
    if math.isclose(scale, 1.0):
        return raw_timestep
    graph = _ensure_graph_ops()
    scale_const = graph.add_constant(
        network, (1, 1), np.array([scale], dtype=np.float32)
    )
    return network.add_elementwise(
        raw_timestep, scale_const, trt.ElementWiseOperation.PROD
    ).get_output(0)


def _make_refiner_rope_tables(
    shape: SanaWmRefinerShape,
    transformer_config: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    transformer_config = transformer_config or {}
    rope_type = shape.rope_type
    if rope_type == "interleaved":
        return ltx.make_ltx_rope_tables(
            latent_frames=shape.latent_frames,
            latent_height=shape.latent_height,
            latent_width=shape.latent_width,
            dim=shape.dim,
            frame_rate=shape.fps,
            temporal_compression_ratio=shape.temporal_compression_ratio,
            spatial_compression_ratio=shape.spatial_compression_ratio,
            base_num_frames=int(transformer_config.get("pos_embed_max_pos", 20)),
            base_height=int(transformer_config.get("base_height", 2048)),
            base_width=int(transformer_config.get("base_width", 2048)),
            theta=float(transformer_config.get("rope_theta", 10000.0)),
        )
    if rope_type != "split":
        raise ValueError(f"Unsupported SANA-WM refiner RoPE type: {rope_type!r}")

    num_pos_dims = 3
    num_rope_elems = num_pos_dims * 2
    head_dim = shape.dim // shape.num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"Split RoPE requires even head dimension, got {head_dim}")

    grid_f, grid_h, grid_w = np.meshgrid(
        np.arange(shape.latent_frames, dtype=np.float32),
        np.arange(shape.latent_height, dtype=np.float32),
        np.arange(shape.latent_width, dtype=np.float32),
        indexing="ij",
    )
    start_f = np.maximum(
        grid_f * shape.temporal_compression_ratio + 1 - shape.temporal_compression_ratio,
        0.0,
    )
    end_f = np.maximum(
        (grid_f + 1.0) * shape.temporal_compression_ratio
        + 1
        - shape.temporal_compression_ratio,
        0.0,
    )
    coord_f = ((start_f + end_f) * 0.5 / float(shape.fps)) / float(
        transformer_config.get("pos_embed_max_pos", 20)
    )
    coord_h = (
        (grid_h + 0.5)
        * shape.spatial_compression_ratio
        / float(transformer_config.get("base_height", 2048))
    )
    coord_w = (
        (grid_w + 0.5)
        * shape.spatial_compression_ratio
        / float(transformer_config.get("base_width", 2048))
    )
    coords = np.stack([coord_f, coord_h, coord_w], axis=-1).reshape(-1, num_pos_dims)

    freq_dtype = (
        np.float64 if bool(transformer_config.get("rope_double_precision", True)) else np.float32
    )
    freq_count = shape.dim // num_rope_elems
    freqs = np.power(
        float(transformer_config.get("rope_theta", 10000.0)),
        np.linspace(0.0, 1.0, freq_count, dtype=freq_dtype),
    )
    freqs = (freqs * (math.pi / 2.0)).astype(np.float32)
    angles = (coords[:, None, :] * 2.0 - 1.0) * freqs[None, :, None]
    angles = angles.reshape(coords.shape[0], -1)
    expected_freqs = shape.dim // 2
    pad_size = expected_freqs - angles.shape[-1]
    if pad_size < 0:
        raise ValueError("SANA-WM refiner split RoPE produced too many frequencies")
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    if pad_size:
        cos_half = np.concatenate(
            [np.ones((coords.shape[0], pad_size), dtype=np.float32), cos_half], axis=-1
        )
        sin_half = np.concatenate(
            [np.zeros((coords.shape[0], pad_size), dtype=np.float32), sin_half], axis=-1
        )

    cos_heads = cos_half.reshape(coords.shape[0], shape.num_heads, head_dim // 2)
    sin_heads = sin_half.reshape(coords.shape[0], shape.num_heads, head_dim // 2)
    cos = np.concatenate([cos_heads, cos_heads], axis=2).reshape(coords.shape[0], shape.dim)
    sin = np.concatenate([sin_heads, sin_heads], axis=2).reshape(coords.shape[0], shape.dim)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _add_timestep_embedding_rows(
    network: trt.INetworkDefinition,
    timestep_col: trt.ITensor,
    *,
    freq_dim: int = 256,
    max_period: float = 10000.0,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    graph = _ensure_graph_ops()
    half = freq_dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)
    freqs_const = graph.add_constant(
        network, (1, half), freqs.reshape(1, -1), dtype=dtype
    )
    args = network.add_elementwise(
        timestep_col, freqs_const, trt.ElementWiseOperation.PROD
    )
    cos_part = network.add_unary(args.get_output(0), trt.UnaryOperation.COS)
    sin_part = network.add_unary(args.get_output(0), trt.UnaryOperation.SIN)
    embed = network.add_concatenation([cos_part.get_output(0), sin_part.get_output(0)])
    embed.axis = 1
    return embed.get_output(0)


def _streaming_self_attention_mask(total_tokens: int, context_tokens: int) -> np.ndarray:
    mask = np.zeros((1, 1, total_tokens, total_tokens), dtype=np.float32)
    if 0 < context_tokens < total_tokens:
        mask[:, :, :context_tokens, context_tokens:] = -10000.0
    return mask


def _refiner_block_modulation(
    network: trt.INetworkDefinition,
    temb: trt.ITensor,
    table: np.ndarray,
    dim: int,
) -> list[trt.ITensor]:
    graph = _ensure_graph_ops()
    chunks: list[trt.ITensor] = []
    seq_len = int(tuple(temb.shape)[0])
    for i in range(6):
        t = network.add_slice(temb, (0, i * dim), (seq_len, dim), (1, 1)).get_output(0)
        c = graph.add_constant(
            network, (1, dim), table[i].reshape(1, dim), dtype=table.dtype
        )
        c = ltx._cast_back(network, c, t.dtype)
        chunks.append(network.add_elementwise(t, c, trt.ElementWiseOperation.SUM).get_output(0))
    return chunks


def _refiner_final_modulation(
    network: trt.INetworkDefinition,
    embedded_timestep: trt.ITensor,
    table: np.ndarray,
    dim: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    graph = _ensure_graph_ops()
    out = []
    for i in range(2):
        c = graph.add_constant(
            network, (1, dim), table[i].reshape(1, dim), dtype=table.dtype
        )
        c = ltx._cast_back(network, c, embedded_timestep.dtype)
        out.append(
            network.add_elementwise(
                embedded_timestep, c, trt.ElementWiseOperation.SUM
            ).get_output(0)
        )
    return out[0], out[1]


def _denoised_x0(
    network: trt.INetworkDefinition,
    latent: trt.ITensor,
    velocity: trt.ITensor,
    raw_timestep: trt.ITensor,
) -> trt.ITensor:
    latent_fp32 = latent if latent.dtype == trt.float32 else network.add_cast(latent, trt.float32).get_output(0)
    velocity_fp32 = (
        velocity
        if velocity.dtype == trt.float32
        else network.add_cast(velocity, trt.float32).get_output(0)
    )
    scaled_velocity = network.add_elementwise(
        velocity_fp32, raw_timestep, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(
        latent_fp32, scaled_velocity, trt.ElementWiseOperation.SUB
    ).get_output(0)
