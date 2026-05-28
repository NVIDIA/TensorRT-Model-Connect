"""SANA-WM LTX-2 refiner text connector builder using the raw TensorRT API.

This builder targets ``diffusers.pipelines.ltx2.LTX2TextConnectors`` for the
video connector path used by SANA-WM. It consumes the packed Gemma/Gemma3
all-hidden-state tensor produced by the refiner text encoder plan and emits the
``v_context``/``v_attention_mask`` tensors consumed by the native refiner DiT.
It does not use ONNX, Torch-TensorRT, or a Python runtime bridge.
"""

from __future__ import annotations

import json
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
class SanaWmRefinerTextConnectorShape:
    seq_len: int
    caption_channels: int
    text_proj_in_factor: int
    inner_dim: int
    num_heads: int
    head_dim: int
    num_layers: int
    num_learnable_registers: int
    rope_base_seq_len: int
    rope_theta: float
    rope_double_precision: bool
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


def _trt_dtype(precision: str):
    trt_module = _ensure_trt()
    return trt_module.float16 if precision == "fp16" else trt_module.float32


def _read_config(path: Path) -> dict:
    config_path = path / "config.json"
    if not config_path.is_file():
        return {}
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _transpose(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(arr.T, dtype=dtype)


def _array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=dtype)


def load_sana_wm_refiner_text_connector_weights(
    connectors_dir: str | Path,
    *,
    num_layers: int = 2,
    precision: str = "fp32",
) -> WeightDict:
    """Load LTX-2 text connector weights in TensorRT matmul layout."""
    readers = ltx._open_safetensors(Path(connectors_dir))  # noqa: SLF001
    dtype = _target_np_dtype(precision)
    weights = WeightDict()

    def has(name: str) -> bool:
        return ltx._has_tensor(readers, name)  # noqa: SLF001

    def t(name: str) -> np.ndarray:
        return _transpose(ltx._load_tensor(readers, name), dtype)  # noqa: SLF001

    def f(name: str, *, norm: bool = False) -> np.ndarray:
        return _array(ltx._load_tensor(readers, name), np.float32 if norm else dtype)  # noqa: SLF001

    def maybe(name: str) -> np.ndarray | None:
        if not has(name):
            return None
        return f(name)

    weights["text_proj_in.weight"] = t("text_proj_in.weight")
    bias = maybe("text_proj_in.bias")
    if bias is not None:
        weights["text_proj_in.bias"] = bias

    if has("video_connector.learnable_registers"):
        weights["video_connector.learnable_registers"] = f(
            "video_connector.learnable_registers"
        )

    for i in range(num_layers):
        p = f"video_connector.transformer_blocks.{i}"
        ap = f"{p}.attn1"
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

    return weights


def refiner_text_connector_shape_from_config(
    raw_config: dict,
    connector_config: dict | None = None,
) -> SanaWmRefinerTextConnectorShape:
    connector_config = connector_config or {}
    if bool(connector_config.get("per_modality_projections", False)):
        raise ValueError(
            "SANA-WM refiner text connector builder currently targets LTX-2.0 "
            "shared text_proj_in connectors, not per-modality LTX-2.3 projections"
        )
    caption_channels = int(connector_config.get("caption_channels", 3840))
    text_proj_in_factor = int(connector_config.get("text_proj_in_factor", 49))
    num_heads = int(connector_config.get("video_connector_num_attention_heads", 30))
    head_dim = int(connector_config.get("video_connector_attention_head_dim", 128))
    num_registers = int(
        connector_config.get("video_connector_num_learnable_registers", 128) or 0
    )
    rope_type = str(connector_config.get("rope_type", "interleaved"))
    if rope_type not in {"interleaved", "split"}:
        raise ValueError(
            "SANA-WM refiner text connector builder supports interleaved or split RoPE"
        )
    return SanaWmRefinerTextConnectorShape(
        seq_len=int(raw_config.get("sana_wm_refiner_text_max_length", 1024)),
        caption_channels=caption_channels,
        text_proj_in_factor=text_proj_in_factor,
        inner_dim=num_heads * head_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=int(connector_config.get("video_connector_num_layers", 2)),
        num_learnable_registers=num_registers,
        rope_base_seq_len=int(connector_config.get("connector_rope_base_seq_len", 4096)),
        rope_theta=float(connector_config.get("rope_theta", 10000.0)),
        rope_double_precision=bool(connector_config.get("rope_double_precision", True)),
        rope_type=rope_type,
    )


def build_sana_wm_refiner_text_connector_engine(
    weights: "Mapping[str, np.ndarray]",
    raw_config: dict,
    connector_config: dict | None = None,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build the LTX-2 video text connector as a TensorRT plan.

    Keep FP32 as the default: Gemma3's pre-final all-hidden-state stack can
    legitimately exceed FP16 range, and the connector consumes that packed
    stack before projecting it down to caption channels.
    """
    if precision not in ("fp16", "fp32"):
        raise ValueError("SANA-WM refiner text connector builder supports fp16 or fp32")

    trt_module = _ensure_trt()
    graph = _ensure_graph_ops()
    shape = refiner_text_connector_shape_from_config(raw_config, connector_config)
    trt_dtype = _trt_dtype(precision)
    np_dtype = _target_np_dtype(precision)

    text_encoder_dim = shape.caption_channels * shape.text_proj_in_factor
    logger = trt_module.Logger(
        trt_module.Logger.VERBOSE if verbose else trt_module.Logger.WARNING
    )
    builder = trt_module.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt_module.MemoryPoolType.WORKSPACE, 16 << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 0

    network = builder.create_network(
        1 << int(trt_module.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    hidden_in = network.add_input(
        "text_hidden_states", trt_dtype, (1, shape.seq_len, text_encoder_dim)
    )
    attention_mask_in = network.add_input(
        "attention_mask", trt_module.float32, (1, shape.seq_len)
    )

    eps_t = graph.add_constant(network, (1, 1), np.array([1.0e-6], dtype=np.float32))
    hidden = _per_layer_masked_mean_norm(
        network,
        hidden_in,
        attention_mask_in,
        shape.seq_len,
        shape.caption_channels,
        shape.text_proj_in_factor,
    )
    hidden = ltx._linear(  # noqa: SLF001 - shared TRT linear lowering
        network,
        hidden,
        text_encoder_dim,
        shape.caption_channels,
        weights,
        "text_proj_in",
        np_dtype,
    )
    hidden = _apply_learnable_registers(
        network,
        hidden,
        attention_mask_in,
        weights,
        shape,
        np_dtype,
    )
    hidden = ltx._drop_batch(  # noqa: SLF001 - shared [1, T, D] -> [T, D] shuffle
        network,
        hidden,
        (shape.seq_len, shape.inner_dim),
    )

    rotary_cos, rotary_sin = _make_text_rope_tables(shape)
    rotary_cos_t = graph.add_constant(network, (shape.seq_len, shape.inner_dim), rotary_cos)
    rotary_sin_t = graph.add_constant(network, (shape.seq_len, shape.inner_dim), rotary_sin)
    rot_half = graph.add_constant(
        network,
        (shape.inner_dim, shape.inner_dim),
        ltx._make_ltx_rotate_half_matrix(  # noqa: SLF001 - shared LTX RoPE helper
            shape.inner_dim,
            shape.num_heads,
            interleaved=shape.rope_type == "interleaved",
        ),
    )

    ones = np.ones((shape.inner_dim,), dtype=np.float32)
    for layer_idx in range(shape.num_layers):
        p = f"video_connector.transformer_blocks.{layer_idx}"
        normed = graph.add_rms_norm(
            network, hidden, shape.inner_dim, ones, eps_t, dtype=np_dtype
        )
        attn = ltx._ltx_attention(  # noqa: SLF001 - shared LTX attention lowering
            network,
            normed,
            None,
            None,
            weights,
            f"{p}.attn1",
            dim=shape.inner_dim,
            num_heads=shape.num_heads,
            head_dim=shape.head_dim,
            q_seq_len=shape.seq_len,
            kv_seq_len=shape.seq_len,
            eps_t=eps_t,
            dtype=np_dtype,
            rotary_cos=rotary_cos_t,
            rotary_sin=rotary_sin_t,
            rot_half=rot_half,
        )
        hidden = network.add_elementwise(
            hidden, attn, trt_module.ElementWiseOperation.SUM
        ).get_output(0)
        normed = graph.add_rms_norm(
            network, hidden, shape.inner_dim, ones, eps_t, dtype=np_dtype
        )
        ff = ltx._ffn(  # noqa: SLF001 - shared LTX feed-forward lowering
            network, normed, weights, p, shape.inner_dim, np_dtype
        )
        hidden = network.add_elementwise(
            hidden, ff, trt_module.ElementWiseOperation.SUM
        ).get_output(0)

    hidden = graph.add_rms_norm(network, hidden, shape.inner_dim, ones, eps_t, dtype=np_dtype)
    out = network.add_shuffle(hidden)
    out.reshape_dims = (1, shape.seq_len, shape.inner_dim)
    out_t = out.get_output(0)
    out_t.name = "v_context"
    network.mark_output(out_t)

    output_mask = graph.add_constant(
        network, (1, shape.seq_len), np.ones((1, shape.seq_len), dtype=np.float32)
    )
    output_mask.name = "v_attention_mask"
    network.mark_output(output_mask)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine serialization failed for SANA-WM refiner text connector")
    return bytes(serialized)


def _per_layer_masked_mean_norm(
    network,
    hidden,
    attention_mask,
    seq_len: int,
    caption_channels: int,
    text_proj_in_factor: int,
):
    trt_module = _ensure_trt()
    graph = _ensure_graph_ops()
    hidden4 = network.add_shuffle(hidden)
    hidden4.reshape_dims = (1, seq_len, caption_channels, text_proj_in_factor)
    hidden4_t = hidden4.get_output(0)
    hidden4_t = ltx._cast_back(network, hidden4_t, trt_module.float32)  # noqa: SLF001

    mask4 = network.add_shuffle(attention_mask)
    mask4.reshape_dims = (1, seq_len, 1, 1)
    mask4_t = mask4.get_output(0)
    mask4_t = ltx._cast_back(network, mask4_t, trt_module.float32)  # noqa: SLF001

    masked = network.add_elementwise(
        hidden4_t, mask4_t, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    reduce_axes = (1 << 1) | (1 << 2)
    summed = network.add_reduce(
        masked, trt_module.ReduceOperation.SUM, reduce_axes, True
    ).get_output(0)
    seq_lengths = network.add_reduce(
        attention_mask, trt_module.ReduceOperation.SUM, 1 << 1, True
    ).get_output(0)
    denom = graph.add_constant(
        network,
        (1, 1),
        np.array([float(caption_channels)], dtype=np.float32),
    )
    denom = network.add_elementwise(
        seq_lengths, denom, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    eps = graph.add_constant(network, (1, 1), np.array([1.0e-6], dtype=np.float32))
    denom = network.add_elementwise(
        denom, eps, trt_module.ElementWiseOperation.SUM
    ).get_output(0)
    denom4 = network.add_shuffle(denom)
    denom4.reshape_dims = (1, 1, 1, 1)
    mean = network.add_elementwise(
        summed, denom4.get_output(0), trt_module.ElementWiseOperation.DIV
    ).get_output(0)

    one = graph.add_constant(network, (1, 1, 1, 1), np.ones((1, 1, 1, 1), dtype=np.float32))
    inv_mask = network.add_elementwise(
        one, mask4_t, trt_module.ElementWiseOperation.SUB
    ).get_output(0)
    large = graph.add_constant(
        network, (1, 1, 1, 1), np.array([1.0e20], dtype=np.float32)
    )
    neg_large = graph.add_constant(
        network, (1, 1, 1, 1), np.array([-1.0e20], dtype=np.float32)
    )
    min_pad = network.add_elementwise(
        inv_mask, large, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    max_pad = network.add_elementwise(
        inv_mask, neg_large, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    min_src = network.add_elementwise(
        hidden4_t, min_pad, trt_module.ElementWiseOperation.SUM
    ).get_output(0)
    max_src = network.add_elementwise(
        hidden4_t, max_pad, trt_module.ElementWiseOperation.SUM
    ).get_output(0)
    x_min = network.add_reduce(
        min_src, trt_module.ReduceOperation.MIN, reduce_axes, True
    ).get_output(0)
    x_max = network.add_reduce(
        max_src, trt_module.ReduceOperation.MAX, reduce_axes, True
    ).get_output(0)

    centered = network.add_elementwise(
        hidden4_t, mean, trt_module.ElementWiseOperation.SUB
    ).get_output(0)
    range_t = network.add_elementwise(
        x_max, x_min, trt_module.ElementWiseOperation.SUB
    ).get_output(0)
    eps4 = graph.add_constant(network, (1, 1, 1, 1), np.array([1.0e-6], dtype=np.float32))
    range_t = network.add_elementwise(
        range_t, eps4, trt_module.ElementWiseOperation.SUM
    ).get_output(0)
    normed = network.add_elementwise(
        centered, range_t, trt_module.ElementWiseOperation.DIV
    ).get_output(0)
    scale = graph.add_constant(network, (1, 1, 1, 1), np.array([8.0], dtype=np.float32))
    normed = network.add_elementwise(
        normed, scale, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    normed = network.add_elementwise(
        normed, mask4_t, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    flat = network.add_shuffle(normed)
    flat.reshape_dims = (1, seq_len, caption_channels * text_proj_in_factor)
    return ltx._cast_back(network, flat.get_output(0), hidden.dtype)  # noqa: SLF001


def _apply_learnable_registers(
    network,
    hidden,
    attention_mask,
    weights: "Mapping[str, np.ndarray]",
    shape: SanaWmRefinerTextConnectorShape,
    dtype: np.dtype,
):
    trt_module = _ensure_trt()
    graph = _ensure_graph_ops()
    if shape.num_learnable_registers <= 0:
        return hidden
    if shape.seq_len % shape.num_learnable_registers != 0:
        raise ValueError(
            f"Text sequence length {shape.seq_len} must be divisible by "
            f"learnable registers {shape.num_learnable_registers}"
        )
    registers = weights.get("video_connector.learnable_registers")
    if registers is None:
        raise ValueError("SANA-WM connector weights are missing learnable registers")
    repeats = shape.seq_len // shape.num_learnable_registers
    registers = np.tile(np.asarray(registers, dtype=dtype), (repeats, 1))
    registers_t = graph.add_constant(
        network, (1, shape.seq_len, shape.inner_dim), registers.reshape(1, shape.seq_len, -1),
        dtype=dtype,
    )
    mask3 = network.add_shuffle(attention_mask)
    mask3.reshape_dims = (1, shape.seq_len, 1)
    mask_t = ltx._cast_back(network, mask3.get_output(0), hidden.dtype)  # noqa: SLF001
    hidden_part = network.add_elementwise(
        hidden, mask_t, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    one = graph.add_constant(network, (1, 1, 1), np.ones((1, 1, 1), dtype=np.float32))
    one = ltx._cast_back(network, one, hidden.dtype)  # noqa: SLF001
    inv_mask = network.add_elementwise(
        one, mask_t, trt_module.ElementWiseOperation.SUB
    ).get_output(0)
    register_part = network.add_elementwise(
        registers_t, inv_mask, trt_module.ElementWiseOperation.PROD
    ).get_output(0)
    hidden = network.add_elementwise(
        hidden_part, register_part, trt_module.ElementWiseOperation.SUM
    ).get_output(0)
    gather_indices = graph.add_constant(
        network,
        (shape.seq_len,),
        np.arange(shape.seq_len - 1, -1, -1, dtype=np.int32),
        dtype=np.int32,
    )
    gather = network.add_gather(hidden, gather_indices, 1)
    return gather.get_output(0)


def _make_text_rope_tables(shape: SanaWmRefinerTextConnectorShape) -> tuple[np.ndarray, np.ndarray]:
    freqs_dtype = np.float64 if shape.rope_double_precision else np.float32
    grid = np.arange(shape.seq_len, dtype=np.float32) / float(shape.rope_base_seq_len)
    pow_indices = shape.rope_theta ** np.linspace(
        0.0,
        1.0,
        shape.inner_dim // 2,
        dtype=freqs_dtype,
    )
    freqs = (pow_indices * math.pi / 2.0).astype(np.float32)
    angles = (grid[:, None] * 2.0 - 1.0) * freqs[None, :]
    cos_freq = np.cos(angles).astype(np.float32)
    sin_freq = np.sin(angles).astype(np.float32)
    if shape.rope_type == "interleaved":
        cos = np.repeat(cos_freq, 2, axis=-1)
        sin = np.repeat(sin_freq, 2, axis=-1)
        if cos.shape[1] != shape.inner_dim:
            pad = shape.inner_dim - cos.shape[1]
            cos = np.concatenate([np.ones((shape.seq_len, pad), dtype=np.float32), cos], axis=1)
            sin = np.concatenate([np.zeros((shape.seq_len, pad), dtype=np.float32), sin], axis=1)
        return np.ascontiguousarray(cos), np.ascontiguousarray(sin)

    expected_freqs = shape.inner_dim // 2
    if cos_freq.shape[1] != expected_freqs:
        pad = expected_freqs - cos_freq.shape[1]
        cos_freq = np.concatenate([np.ones((shape.seq_len, pad), dtype=np.float32), cos_freq], axis=1)
        sin_freq = np.concatenate([np.zeros((shape.seq_len, pad), dtype=np.float32), sin_freq], axis=1)
    head_half = shape.head_dim // 2
    cos_heads = cos_freq.reshape(shape.seq_len, shape.num_heads, head_half)
    sin_heads = sin_freq.reshape(shape.seq_len, shape.num_heads, head_half)
    cos = np.concatenate([cos_heads, cos_heads], axis=-1).reshape(
        shape.seq_len,
        shape.inner_dim,
    )
    sin = np.concatenate([sin_heads, sin_heads], axis=-1).reshape(
        shape.seq_len,
        shape.inner_dim,
    )
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def build_from_directory(
    connectors_dir: str | Path,
    raw_config: dict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    connectors_dir = Path(connectors_dir)
    connector_config = _read_config(connectors_dir)
    shape = refiner_text_connector_shape_from_config(raw_config, connector_config)
    weights = load_sana_wm_refiner_text_connector_weights(
        connectors_dir,
        num_layers=shape.num_layers,
        precision=precision,
    )
    return build_sana_wm_refiner_text_connector_engine(
        weights,
        raw_config,
        connector_config,
        precision=precision,
        verbose=verbose,
    )


__all__ = [
    "SanaWmRefinerTextConnectorShape",
    "build_from_directory",
    "build_sana_wm_refiner_text_connector_engine",
    "load_sana_wm_refiner_text_connector_weights",
    "refiner_text_connector_shape_from_config",
]
