# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT builder surface for SANA-WM Stage-1 DiT.

This module is intentionally owned by the SANA-WM family.  The public model's
Stage-1 denoiser is ``SanaMSVideoCamCtrl`` with BidirectionalGDN and UCPE
camera-control blocks, so it cannot reuse the existing standard attention DiT
builders.  The code here starts from raw safetensors and the TensorRT network
API; it does not use ONNX, tracing, Torch-TensorRT, or a Python runtime bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorrt as trt

from .components.gemma.checkpoint_mapper import (
    WeightDict,
    _ReaderCollection,
    _load_tensor,
    _open_safetensors,
)
from .builder_lifetime import get_process_trt_logger

_BF16_WEIGHT_REFS: list[np.ndarray] = []
_STAGE1_OUTPUT_GATE_DEBUG_RETURNS = {
    "output_gate_linear",
    "output_gate_linear_fp32",
    "output_gate",
    "output_gate_core",
    "output_gated",
    "output_gated_cast",
}
_STAGE1_GDN_PHASE_DEBUG_RETURNS = {
    "gdn_raw_q_inv",
    "gdn_raw_k_inv",
    "gdn_raw_k",
    "gdn_raw_k_rot",
    "gdn_raw_beta_v",
    "gdn_raw_beta_k_rot",
    "gdn_phase_a_i_p_kv",
    "gdn_phase_a_a",
    "gdn_phase_a_i_p_z",
    "gdn_phase_a_b_z",
    "gdn_phase_b_hist_kv",
    "gdn_phase_b_hist_z",
    "gdn_phase_c_num",
    "gdn_phase_c_den",
    "gdn_phase_c_precast",
    "gdn_phase_c_output",
}
_STAGE1_POST_ATTENTION_DEBUG_RETURNS = {
    "attn_gated",
    "post_attn_residual",
    "plucker",
    "post_plucker",
    "cross_q_linear",
    "cross_k_linear",
    "cross_v_linear",
    "cross_q_norm",
    "cross_k_norm",
    "cross_scores",
    "cross_scores_masked",
    "cross_softmax",
    "cross_pre_proj",
    "cross",
    "post_cross",
    "norm2",
    "x_mlp_in",
    "mlp_inverted_pre_silu",
    "mlp_inverted_pre_silu_sample",
    "mlp_inverted",
    "mlp_inverted_sample",
    "mlp_depth",
    "mlp_depth_sample",
    "mlp_value_sample",
    "mlp_gate_pre_silu",
    "mlp_gate_pre_silu_sample",
    "mlp_gate",
    "mlp_gate_sample",
    "mlp_gated_spatial",
    "mlp_gated_spatial_sample",
    "mlp_point_sample",
    "mlp_point_tokens",
    "mlp_temporal_sample",
    "mlp_temporal_sum_sample",
    "mlp",
    "mlp_gated",
    "block_output",
}


@dataclass(frozen=True)
class SanaWmStage1Shape:
    batch_size: int
    latent_channels: int
    latent_frames: int
    latent_height: int
    latent_width: int
    text_max_length: int
    text_embed_dim: int
    chunk_plucker_channels: int
    raymap_width: int = 20


@dataclass(frozen=True)
class SanaWmStage1Frontend:
    x_tokens: Any
    token_count: int
    hidden_size: int
    plucker_tokens: Any | None = None


@dataclass(frozen=True)
class SanaWmStage1Conditioning:
    t: Any
    t0: Any
    y: Any
    mask: Any
    t_freq: Any | None = None


@dataclass(frozen=True)
class SanaWmStage1FinalOutput:
    tokens: Any
    latents: Any


@dataclass(frozen=True)
class SanaWmStage1BlockModulation:
    shift_msa: Any
    scale_msa: Any
    gate_msa: Any
    shift_mlp: Any
    scale_mlp: Any
    gate_mlp: Any


@dataclass(frozen=True)
class SanaWmStage1BlockPreamble:
    x_msa_in: Any
    qkv: Any
    qkv_heads: Any
    q: Any
    k: Any
    q_rot: Any
    k_rot: Any
    v: Any
    beta: Any
    decay: Any
    num_heads: int
    head_dim: int
    modulation: SanaWmStage1BlockModulation
    norm1: Any | None = None
    x_msa_4d: Any | None = None
    x_frame: Any | None = None
    gate: Any | None = None
    gate_dt: Any | None = None
    q_raw: Any | None = None
    k_raw: Any | None = None
    k_conv: Any | None = None
    v_raw: Any | None = None


@dataclass(frozen=True)
class SanaWmStage1CameraPreamble:
    q: Any
    k: Any
    v: Any
    num_heads: int
    head_dim: int
    q_raw: Any | None = None
    k_raw: Any | None = None
    v_raw: Any | None = None
    q_norm_weight: np.ndarray | None = None
    k_norm_weight: np.ndarray | None = None


@dataclass(frozen=True)
class SanaWmStage1CameraUcpe:
    q_rot: Any
    k_rot: Any
    v: Any
    beta: Any
    num_heads: int
    head_dim: int
    inflation_sq: Any | None = None


@dataclass(frozen=True)
class SanaWmStage1GdnComponents:
    num: Any
    den: Any


@dataclass(frozen=True)
class SanaWmStage1GdnCore:
    tokens: Any
    num: Any
    den: Any


def _bf16_np_dtype() -> Any:
    try:
        import ml_dtypes
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SANA-WM Stage-1 bf16 TensorRT builds require the ml_dtypes package"
        ) from exc
    return ml_dtypes.bfloat16


def _is_bf16_dtype(dtype: Any) -> bool:
    return str(np.dtype(dtype)) == "bfloat16"


def _round_float32_to_bf16_values(value: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    bits = arr.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32).reshape(arr.shape)


def _fp32_parameter_values_for_runtime_dtype(value: np.ndarray, dtype: Any) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if _is_bf16_dtype(dtype):
        return _round_float32_to_bf16_values(arr)
    if np.dtype(dtype) == np.dtype(np.float16):
        return np.ascontiguousarray(arr.astype(np.float16).astype(np.float32))
    return arr


def _target_np_dtype(precision: str) -> Any:
    if precision == "bf16":
        return _bf16_np_dtype()
    return np.float16 if precision == "fp16" else np.float32


def _target_trt_dtype(trt_module: Any, precision: str) -> Any:
    if precision == "bf16":
        return trt_module.bfloat16
    return trt_module.float16 if precision == "fp16" else trt_module.float32


def _trt_dtype_for_np(trt_module: Any, dtype: Any) -> Any:
    if _is_bf16_dtype(dtype):
        return _target_trt_dtype(trt_module, "bf16")
    return trt_module.float16 if dtype == np.float16 else trt_module.float32


def _open_stage1_safetensors(path: Path) -> _ReaderCollection:
    if path.is_file():
        from safetensors import safe_open

        return _ReaderCollection([safe_open(str(path), framework="numpy")])
    return _open_safetensors(path)


def _reader_keys(readers: Iterable[Any]) -> list[str]:
    keys: set[str] = set()
    for reader in readers:
        keys.update(str(key) for key in reader.keys())
    return sorted(keys)


def _is_linear_weight(name: str, value: np.ndarray) -> bool:
    return name.endswith(".weight") and value.ndim == 2


def _is_fp32_parameter(name: str) -> bool:
    parts = name.split(".")
    parent = parts[-2] if len(parts) >= 2 else ""
    return (
        name.endswith(".norm.weight")
        or name.endswith("_norm.weight")
        or (name.endswith(".weight") and parent.startswith("norm"))
        or name.endswith(".A_log")
        or name.endswith(".dt_bias")
        or name.endswith(".scale_shift_table")
        or name == "final_layer.scale_shift_table"
    )


def _to_trt_layout(name: str, value: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    dtype = np.float32 if _is_fp32_parameter(name) else target_dtype
    if _is_linear_weight(name, value):
        return np.ascontiguousarray(value.T, dtype=dtype)
    return np.ascontiguousarray(value, dtype=dtype)


def load_sana_wm_stage1_dit_weights(
    dit_path: str | Path,
    *,
    precision: str = "fp16",
) -> WeightDict:
    """Load SANA-WM Stage-1 DiT weights into TRT-friendly tensor layouts.

    Linear weights are transposed from PyTorch ``[out, in]`` to TensorRT
    matmul RHS ``[in, out]``.  Convolution kernels and modulation tables retain
    their source layout.
    """
    readers = _open_stage1_safetensors(Path(dit_path))
    target_dtype = _target_np_dtype(precision)
    weights = WeightDict()
    for name in _reader_keys(readers):
        if name == "__metadata__":
            continue
        weights[name] = _to_trt_layout(name, _load_tensor(readers, name), target_dtype)
    weights["_source_path"] = str(dit_path)
    weights["_precision"] = precision
    return weights


def _tuple3(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
        if len(vals) == 1:
            return vals[0], vals[0], vals[0]
        if len(vals) == 2:
            return vals[0], vals[1], vals[1]
        if len(vals) >= 3:
            return vals[0], vals[1], vals[2]
    return fallback


def _model_dict(raw_config: dict) -> dict:
    model = raw_config.get("model", {})
    return model if isinstance(model, dict) else {}


def _model_bool(raw_config: dict, key: str, fallback: bool = False) -> bool:
    model = _model_dict(raw_config)
    if key in raw_config:
        return bool(raw_config[key])
    if key in model:
        return bool(model[key])
    return fallback


def _stage1_patch_size(raw_config: dict) -> tuple[int, int, int]:
    model = _model_dict(raw_config)
    return _tuple3(raw_config.get("patch_size", model.get("patch_size")), (1, 1, 1))


def _stage1_uses_chunk_plucker(raw_config: dict, weights: WeightDict) -> bool:
    return (
        _model_bool(raw_config, "use_chunk_plucker_input")
        or _model_bool(raw_config, "use_chunk_plucker_post_attn")
        or "plucker_embedder.proj.weight" in weights
    )


def _stage1_linear_head_dim(raw_config: dict, hidden_size: int) -> int:
    model = _model_dict(raw_config)
    value = raw_config.get("linear_head_dim", model.get("linear_head_dim"))
    return int(value if value is not None else hidden_size // 20)


def _stage1_cam_attn_compress(raw_config: dict) -> int:
    model = _model_dict(raw_config)
    value = raw_config.get("cam_attn_compress", model.get("cam_attn_compress", 1))
    compress = int(value)
    if compress <= 0:
        raise ValueError(f"SANA-WM cam_attn_compress must be positive, got {compress}")
    return compress


def _stage1_mlp_ratio(raw_config: dict) -> float:
    model = _model_dict(raw_config)
    return float(raw_config.get("mlp_ratio", model.get("mlp_ratio", 4.0)))


def _stage1_ffn_type(raw_config: dict) -> str:
    model = _model_dict(raw_config)
    return str(raw_config.get("ffn_type", model.get("ffn_type", "mlp")))


def _stage1_t_kernel_size(raw_config: dict) -> int:
    model = _model_dict(raw_config)
    return int(raw_config.get("t_kernel_size", model.get("t_kernel_size", 3)))


def _stage1_cross_norm(raw_config: dict) -> bool:
    return _model_bool(raw_config, "cross_norm", False)


def _stage1_use_trt_attention(raw_config: dict) -> bool:
    return _model_bool(raw_config, "use_trt_attention", False)


def _stage1_norm_eps(raw_config: dict) -> float:
    model = _model_dict(raw_config)
    return float(raw_config.get("norm_eps", model.get("norm_eps", 1.0e-5)))


def _stage1_attention_eps(raw_config: dict) -> float:
    model = _model_dict(raw_config)
    return float(raw_config.get("attention_eps", model.get("attention_eps", 1.0e-15)))


def _stage1_camctrl_type(raw_config: dict) -> str:
    model = _model_dict(raw_config)
    return str(
        raw_config.get(
            "camctrl_type",
            model.get("camctrl_type", "BidirectionalGDNUCPESinglePathLiteLABothTriton"),
        )
    )


def _stage1_softmax_every_n(raw_config: dict) -> int:
    model = _model_dict(raw_config)
    return int(raw_config.get("softmax_every_n", model.get("softmax_every_n", 0)))


def _stage1_depth_from_weights(raw_config: dict, weights: WeightDict) -> int:
    model = _model_dict(raw_config)
    configured = raw_config.get("depth", model.get("depth"))
    if configured is not None:
        return int(configured)
    block_ids = []
    for name in weights:
        if not name.startswith("blocks.") or ".scale_shift_table" not in name:
            continue
        parts = name.split(".")
        if len(parts) >= 3 and parts[1].isdigit():
            block_ids.append(int(parts[1]))
    if not block_ids:
        raise ValueError("SANA-WM Stage-1 checkpoint does not contain any transformer blocks")
    return max(block_ids) + 1


def _stage1_block_uses_softmax(raw_config: dict, block_index: int) -> bool:
    camctrl_type = _stage1_camctrl_type(raw_config)
    if camctrl_type == "BidirectionalSoftmaxUCPESinglePathLiteLA":
        return True
    softmax_every_n = _stage1_softmax_every_n(raw_config)
    return (
        camctrl_type == "BidirectionalGDNUCPESinglePathLiteLABothTriton"
        and softmax_every_n > 0
        and (block_index + 1) % softmax_every_n == 0
    )


def _stage1_stabilizes_camera_ucpe(raw_config: dict, *, softmax_attention: bool) -> bool:
    if softmax_attention:
        return False
    return _stage1_camctrl_type(raw_config) != "BidirectionalGDNUCPESinglePathLiteLABothTriton"


def stage1_shape_from_config(
    raw_config: dict,
    weights: WeightDict | None = None,
) -> SanaWmStage1Shape:
    vae = raw_config.get("vae", {}) if isinstance(raw_config.get("vae"), dict) else {}
    stage1_summary = raw_config.get("_sana_wm_stage1_dit_summary", {})
    if not isinstance(stage1_summary, dict):
        stage1_summary = {}
    if weights is not None and "x_embedder.proj.weight" in weights:
        x_weight = np.asarray(weights["x_embedder.proj.weight"])
        latent_channels = int(x_weight.shape[1] if x_weight.ndim == 5 else x_weight.shape[0])
    else:
        latent_channels = int(stage1_summary.get("latent_channels", vae.get("vae_latent_dim", 128)))

    video_frames = int(raw_config.get("video_num_frames", 321))
    video_height = int(raw_config.get("video_height", 704))
    video_width = int(raw_config.get("video_width", 1280))
    vae_stride = _tuple3(
        vae.get("vae_stride", raw_config.get("vae_stride")),
        (8, 32, 32),
    )
    latent_frames = (video_frames - 1) // vae_stride[0] + 1
    latent_height = video_height // vae_stride[-1]
    latent_width = video_width // vae_stride[-1]

    text_encoder = (
        raw_config.get("text_encoder", {})
        if isinstance(raw_config.get("text_encoder"), dict)
        else {}
    )
    text_max_length = int(
        stage1_summary.get("text_max_length", text_encoder.get("model_max_length", 300))
    )
    text_embed_dim = int(
        stage1_summary.get("text_embed_dim", raw_config.get("text_encoder_dim", 2304))
    )
    chunk_plucker_channels = int(
        stage1_summary.get(
            "chunk_plucker_channels",
            raw_config.get("chunk_plucker_channels", 48),
        )
    )
    return SanaWmStage1Shape(
        batch_size=2,
        latent_channels=latent_channels,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        text_max_length=text_max_length,
        text_embed_dim=text_embed_dim,
        chunk_plucker_channels=chunk_plucker_channels,
    )


def _required_weight(weights: WeightDict, name: str) -> np.ndarray:
    value = weights.get(name)
    if value is None:
        raise KeyError(f"SANA-WM Stage-1 DiT checkpoint is missing tensor {name!r}")
    return np.asarray(value)


def _optional_weight(weights: WeightDict, name: str) -> np.ndarray | None:
    value = weights.get(name)
    return None if value is None else np.asarray(value)


def _trt_weights(trt_module: Any, value: np.ndarray | None, dtype: Any) -> Any:
    if value is None:
        return trt_module.Weights()
    if _is_bf16_dtype(dtype):
        bf16_value = np.ascontiguousarray(np.asarray(value, dtype=np.float32).astype(dtype))
        _BF16_WEIGHT_REFS.append(bf16_value)
        return trt_module.Weights(
            trt_module.bfloat16,
            bf16_value.ctypes.data,
            bf16_value.size,
        )
    return trt_module.Weights(np.ascontiguousarray(value, dtype=dtype))


def _set_tensor_name(tensor: Any, name: str) -> Any:
    tensor.name = name
    return tensor


def _stage1_debug_stop_after_block() -> int | None:
    return None


def _stage1_debug_block_return(block_index: int | None = None) -> str | None:
    del block_index
    return None


def _stage1_debug_block_input_index() -> int | None:
    return None


def _stage1_debug_mlp_input_index() -> int | None:
    return None


def _stage1_explicit_softmax_attention() -> bool:
    return False


def _stage1_decomposable_softmax_attention() -> bool:
    return True


def _stage1_pad_softmax_head_dim() -> bool:
    return True


def _cast_to_dtype(network: Any, tensor: Any, trt_dtype: Any) -> Any:
    return network.add_cast(tensor, trt_dtype).get_output(0)


def _add_constant(
    network: Any,
    trt_module: Any,
    shape: tuple[int, ...],
    value: np.ndarray,
    *,
    dtype: Any,
) -> Any:
    if _is_bf16_dtype(dtype):
        bf16_value = np.ascontiguousarray(
            np.asarray(value, dtype=np.float32).reshape(shape).astype(dtype)
        )
        _BF16_WEIGHT_REFS.append(bf16_value)
        layer = network.add_constant(
            shape,
            trt_module.Weights(
                trt_module.bfloat16,
                bf16_value.ctypes.data,
                bf16_value.size,
            ),
        )
        return layer.get_output(0)
    layer = network.add_constant(
        shape,
        trt_module.Weights(np.ascontiguousarray(value, dtype=dtype).reshape(shape)),
    )
    return layer.get_output(0)


def _linear_rhs_shape(batch_prefix_rank: int, input_dim: int, output_dim: int) -> tuple[int, ...]:
    return (1,) * batch_prefix_rank + (input_dim, output_dim)


def _linear_bias_shape(batch_prefix_rank: int, output_dim: int) -> tuple[int, ...]:
    return (1,) * batch_prefix_rank + (1, output_dim)


def _add_linear(
    network: Any,
    inp: Any,
    *,
    weights: WeightDict,
    prefix: str,
    input_dim: int,
    output_dim: int,
    batch_prefix_rank: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    weight = _required_weight(weights, f"{prefix}.weight")
    bias = _optional_weight(weights, f"{prefix}.bias")
    if weight.shape != (input_dim, output_dim):
        raise ValueError(
            f"{prefix}.weight must have TRT matmul shape "
            f"({input_dim}, {output_dim}), got {weight.shape}"
        )
    if bias is not None and bias.shape != (output_dim,):
        raise ValueError(f"{prefix}.bias must have shape ({output_dim},), got {bias.shape}")

    rhs_shape = _linear_rhs_shape(batch_prefix_rank, input_dim, output_dim)
    rhs = _add_constant(
        network,
        trt_module,
        rhs_shape,
        weight,
        dtype=dtype,
    )
    matmul = network.add_matrix_multiply(
        inp,
        trt_module.MatrixOperation.NONE,
        rhs,
        trt_module.MatrixOperation.NONE,
    )
    out = matmul.get_output(0)
    if bias is not None:
        bias_shape = _linear_bias_shape(batch_prefix_rank, output_dim)
        bias_t = _add_constant(
            network,
            trt_module,
            bias_shape,
            bias,
            dtype=dtype,
        )
        out = network.add_elementwise(
            out,
            bias_t,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
    return _set_tensor_name(out, f"{prefix}.output")


def _add_silu(network: Any, inp: Any, *, trt_module: Any) -> Any:
    sigmoid = network.add_activation(inp, trt_module.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp,
        sigmoid.get_output(0),
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)


def _add_silu_for_stage1_mlp(
    network: Any,
    inp: Any,
    *,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    if dtype != np.float32:
        fp32 = _cast_to_dtype(network, inp, trt_module.float32)
        out = _add_silu(network, fp32, trt_module=trt_module)
        return _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    return _add_silu(network, inp, trt_module=trt_module)


def _add_ranked_scalar(
    network: Any,
    trt_module: Any,
    rank: int,
    value: float,
    *,
    dtype: np.dtype,
) -> Any:
    return _add_constant(
        network,
        trt_module,
        (1,) * rank,
        np.array([value], dtype=np.float32),
        dtype=dtype,
    )


def _add_int32_scalar(network: Any, trt_module: Any, value: int) -> Any:
    layer = network.add_constant((), trt_module.Weights(np.array(value, dtype=np.int32)))
    return layer.get_output(0)


def _add_int32_shape(network: Any, trt_module: Any, values: tuple[int, ...]) -> Any:
    value = np.asarray(values, dtype=np.int32)
    layer = network.add_constant((len(values),), trt_module.Weights(value))
    return layer.get_output(0)


def _can_use_trt_loop(network: Any, trt_module: Any) -> bool:
    del network, trt_module
    return False


def _get_sana_wm_plugin_creator(trt_module: Any, plugin_name: str) -> Any:
    creator = trt_module.get_plugin_registry().get_creator(plugin_name, "1", "")
    if creator is None:
        raise RuntimeError(f"SANA-WM plugin creator {plugin_name} is missing")
    return creator


def _create_sana_wm_plugin(creator: Any, name: str, fields: Any) -> Any:
    plugin = creator.create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create SANA-WM plugin {name}")
    return plugin


def _get_sana_wm_gdn_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmGdnScan")


def _get_sana_wm_timestep_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmTimestepEmbed")


def _get_sana_wm_gate_proj_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmGateProj")


def _get_sana_wm_decay_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmDecay")


def _get_sana_wm_frame_mean_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmFrameMean")


def _get_sana_wm_t2i_modulate_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmT2IModulate")


def _get_sana_wm_caption_embed_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmCaptionEmbed")


def _get_sana_wm_cross_attention_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmCrossAttention")


def _get_sana_wm_softmax_attention_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(
        trt_module,
        "SanaWmSoftmaxAttention",
    )


def _get_sana_wm_frame_gate_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmFrameGate")


def _get_sana_wm_layer_norm_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmLayerNorm")


def _get_sana_wm_short_conv_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmShortConv")


def _get_sana_wm_cam_prep_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmCamPrep")


def _get_sana_wm_torch_cam_prep_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmTorchCamPrep")


def _get_sana_wm_camera_beta_discount_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(
        trt_module,
        "SanaWmCameraBetaDiscount",
    )


def _get_sana_wm_ucpe_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmUcpe")


def _get_sana_wm_torch_conv2d_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmTorchConv2d")


def _get_sana_wm_glumbconvtemp_plugin_creator(trt_module: Any) -> Any | None:
    return _get_sana_wm_plugin_creator(trt_module, "SanaWmGlumbconvTemp")


def _create_sana_wm_gdn_plugin(
    trt_module: Any,
    *,
    mode: int,
    reverse_output: bool,
    eps: float | None = None,
    frames: int | None = None,
    head_dim: int | None = None,
    norm_eps: float | None = None,
) -> Any | None:
    creator = _get_sana_wm_gdn_plugin_creator(trt_module)
    mode_value = np.asarray([mode], dtype=np.int32)
    reverse_value = np.asarray([1 if reverse_output else 0], dtype=np.int32)
    fields = [
        trt_module.PluginField("mode", mode_value, trt_module.PluginFieldType.INT32),
        trt_module.PluginField(
            "reverse_output",
            reverse_value,
            trt_module.PluginFieldType.INT32,
        ),
    ]
    float32_field = trt_module.PluginFieldType.FLOAT32
    if eps is not None:
        eps_value = np.asarray([eps], dtype=np.float32)
        fields.append(trt_module.PluginField("eps", eps_value, float32_field))
    if frames is not None:
        frames_value = np.asarray([frames], dtype=np.int32)
        fields.append(
            trt_module.PluginField("frames", frames_value, trt_module.PluginFieldType.INT32)
        )
    if head_dim is not None:
        head_dim_value = np.asarray([head_dim], dtype=np.int32)
        fields.append(
            trt_module.PluginField("head_dim", head_dim_value, trt_module.PluginFieldType.INT32)
        )
    if norm_eps is not None:
        norm_eps_value = np.asarray([norm_eps], dtype=np.float32)
        fields.append(trt_module.PluginField("norm_eps", norm_eps_value, float32_field))
    collection = trt_module.PluginFieldCollection(fields)
    return _create_sana_wm_plugin(
        creator, f"sana_wm_gdn_{mode}_{int(reverse_output)}", collection
    )


def _create_sana_wm_timestep_plugin(
    trt_module: Any,
    *,
    weights: WeightDict,
    frequency_dim: int,
    hidden_size: int,
) -> Any | None:
    creator = _get_sana_wm_timestep_plugin_creator(trt_module)
    int32_type = trt_module.PluginFieldType.INT32
    float32_type = trt_module.PluginFieldType.FLOAT32

    refs: list[np.ndarray] = []
    fields = []
    for name, value in (
        ("frequency_dim", frequency_dim),
        ("hidden_size", hidden_size),
    ):
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, int32_type))
    import torch

    half = frequency_dim // 2
    freqs_t = torch.exp(
        -float(np.log(10000.0))
        * torch.arange(start=0, end=half, dtype=torch.float32, device="cuda")
        / float(half)
    )
    freqs = np.ascontiguousarray(freqs_t.cpu().numpy().astype(np.float32))
    refs.append(freqs)
    fields.append(trt_module.PluginField("freqs", freqs, float32_type))
    for field_name, weight_name in (
        ("w0", "t_embedder.mlp.0.weight"),
        ("b0", "t_embedder.mlp.0.bias"),
        ("w1", "t_embedder.mlp.2.weight"),
        ("b1", "t_embedder.mlp.2.bias"),
        ("w2", "t_block.1.weight"),
        ("b2", "t_block.1.bias"),
    ):
        value = np.ascontiguousarray(
            np.asarray(_required_weight(weights, weight_name), dtype=np.float32).reshape(-1)
        )
        refs.append(value)
        fields.append(trt_module.PluginField(field_name, value, float32_type))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_timestep_embed", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _create_sana_wm_gate_proj_plugin(
    trt_module: Any,
    *,
    weights: WeightDict,
    prefix: str,
    input_dim: int,
    output_dim: int,
    activation: int = 0,
    use_matmul_bias: bool = False,
) -> Any | None:
    creator = _get_sana_wm_gate_proj_plugin_creator(trt_module)
    int32_type = trt_module.PluginFieldType.INT32
    float32_type = trt_module.PluginFieldType.FLOAT32

    refs: list[np.ndarray] = []
    fields = []
    for name, value in (
        ("input_dim", input_dim),
        ("output_dim", output_dim),
        ("activation", activation),
        ("use_matmul_bias", int(use_matmul_bias)),
    ):
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, int32_type))
    weight = np.ascontiguousarray(
        np.asarray(_required_weight(weights, f"{prefix}.weight"), dtype=np.float32).reshape(-1)
    )
    refs.append(weight)
    fields.append(trt_module.PluginField("weight", weight, float32_type))
    bias = _optional_weight(weights, f"{prefix}.bias")
    if bias is not None:
        bias_value = np.ascontiguousarray(np.asarray(bias, dtype=np.float32).reshape(-1))
        refs.append(bias_value)
        fields.append(trt_module.PluginField("bias", bias_value, float32_type))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_gate_proj", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _apply_sana_wm_frame_mean_plugin(
    network: Any,
    x_msa: Any,
    *,
    frames: int,
    spatial: int,
    trt_module: Any,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    creator = _get_sana_wm_frame_mean_plugin_creator(trt_module)
    int32_type = trt_module.PluginFieldType.INT32
    frames_value = np.asarray([frames], dtype=np.int32)
    spatial_value = np.asarray([spatial], dtype=np.int32)
    collection = trt_module.PluginFieldCollection(
        [
            trt_module.PluginField("frames", frames_value, int32_type),
            trt_module.PluginField("spatial", spatial_value, int32_type),
        ]
    )
    plugin = _create_sana_wm_plugin(creator, "sana_wm_frame_mean", collection)
    _BF16_WEIGHT_REFS.extend([frames_value, spatial_value])
    layer = add_plugin([x_msa], plugin)
    return _set_tensor_name(layer.get_output(0), name)


def _apply_sana_wm_decay_plugin(
    network: Any,
    gate_dt: Any,
    *,
    a_log_values: np.ndarray,
    num_heads: int,
    trt_module: Any,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    creator = _get_sana_wm_decay_plugin_creator(trt_module)
    int32_type = trt_module.PluginFieldType.INT32
    float32_type = trt_module.PluginFieldType.FLOAT32
    heads_value = np.asarray([num_heads], dtype=np.int32)
    a_log_value = np.ascontiguousarray(np.asarray(a_log_values, dtype=np.float32).reshape(-1))
    fields = [
        trt_module.PluginField("heads", heads_value, int32_type),
        trt_module.PluginField("a_log", a_log_value, float32_type),
    ]
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_decay", collection)
    _BF16_WEIGHT_REFS.extend([heads_value, a_log_value])
    layer = add_plugin([gate_dt], plugin)
    return _set_tensor_name(layer.get_output(0), name)


def _add_sana_wm_bf16_linear_plugin(
    network: Any,
    inp: Any,
    *,
    weights: WeightDict,
    prefix: str,
    input_dim: int,
    output_dim: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
    activation: int = 0,
    use_matmul_bias: bool = False,
    enabled: bool = True,
) -> Any | None:
    if not enabled or not _is_bf16_dtype(dtype):
        return None
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_gate_proj_plugin(
        trt_module,
        weights=weights,
        prefix=prefix,
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        use_matmul_bias=use_matmul_bias,
    )
    layer = add_plugin([inp], plugin)
    return _set_tensor_name(layer.get_output(0), f"{name}.output")


def _add_sana_wm_output_gate_plugin(
    network: Any,
    tokens: Any,
    gate_input: Any,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any | None:
    if not _is_bf16_dtype(dtype):
        return None
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_gate_proj_plugin(
        trt_module,
        weights=weights,
        prefix=prefix,
        input_dim=hidden_size,
        output_dim=hidden_size,
        activation=2,
    )
    layer = add_plugin([tokens, gate_input], plugin)
    return _set_tensor_name(layer.get_output(0), name)


def _create_sana_wm_t2i_modulate_plugin(trt_module: Any) -> Any | None:
    creator = _get_sana_wm_t2i_modulate_plugin_creator(trt_module)
    collection = trt_module.PluginFieldCollection([])
    return _create_sana_wm_plugin(creator, "sana_wm_t2i_modulate", collection)


def _create_sana_wm_frame_gate_plugin(trt_module: Any) -> Any | None:
    creator = _get_sana_wm_frame_gate_plugin_creator(trt_module)
    collection = trt_module.PluginFieldCollection([])
    return _create_sana_wm_plugin(creator, "sana_wm_frame_gate", collection)


def _create_sana_wm_caption_embed_plugin(
    trt_module: Any,
    *,
    hidden_size: int,
) -> Any | None:
    creator = _get_sana_wm_caption_embed_plugin_creator(trt_module)
    hidden_value = np.asarray([hidden_size], dtype=np.int32)
    collection = trt_module.PluginFieldCollection(
        [
            trt_module.PluginField(
                "hidden_size",
                hidden_value,
                trt_module.PluginFieldType.INT32,
            )
        ]
    )
    plugin = _create_sana_wm_plugin(creator, "sana_wm_caption_embed", collection)
    _BF16_WEIGHT_REFS.append(hidden_value)
    return plugin


def _apply_sana_wm_caption_embed_plugin(
    network: Any,
    y: Any,
    *,
    weights: WeightDict,
    text_embed_dim: int,
    hidden_size: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any | None:
    if not _is_bf16_dtype(dtype):
        return None
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_caption_embed_plugin(
        trt_module,
        hidden_size=hidden_size,
    )
    constants = [
        _add_constant(
            network,
            trt_module,
            shape,
            _required_weight(weights, name),
            dtype=dtype,
        )
        for name, shape in (
            ("y_embedder.y_proj.fc1.weight", (text_embed_dim, hidden_size)),
            ("y_embedder.y_proj.fc1.bias", (hidden_size,)),
            ("y_embedder.y_proj.fc2.weight", (hidden_size, hidden_size)),
            ("y_embedder.y_proj.fc2.bias", (hidden_size,)),
            ("attention_y_norm.weight", (hidden_size,)),
        )
    ]
    layer = add_plugin([y, *constants], plugin)
    return _set_tensor_name(layer.get_output(0), "attention_y_norm.output")


def _apply_sana_wm_cross_attention_plugin(
    network: Any,
    x: Any,
    conditioning: SanaWmStage1Conditioning,
    *,
    weights: WeightDict,
    block_index: int,
    num_heads: int,
    hidden_size: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any | None:
    if not _is_bf16_dtype(dtype):
        return None
    add_plugin = network.add_plugin_v2
    creator = _get_sana_wm_cross_attention_plugin_creator(trt_module)
    heads_value = np.asarray([num_heads], dtype=np.int32)
    collection = trt_module.PluginFieldCollection(
        [
            trt_module.PluginField(
                "num_heads",
                heads_value,
                trt_module.PluginFieldType.INT32,
            )
        ]
    )
    plugin = _create_sana_wm_plugin(creator, "sana_wm_cross_attention", collection)
    _BF16_WEIGHT_REFS.append(heads_value)
    prefix = f"blocks.{block_index}.cross_attn"
    constants = [
        _add_constant(
            network,
            trt_module,
            shape,
            _required_weight(weights, name),
            dtype=dtype,
        )
        for name, shape in (
            (f"{prefix}.q_linear.weight", (hidden_size, hidden_size)),
            (f"{prefix}.q_linear.bias", (hidden_size,)),
            (f"{prefix}.kv_linear.weight", (hidden_size, 2 * hidden_size)),
            (f"{prefix}.kv_linear.bias", (2 * hidden_size,)),
            (f"{prefix}.q_norm.weight", (hidden_size,)),
            (f"{prefix}.k_norm.weight", (hidden_size,)),
            (f"{prefix}.proj.weight", (hidden_size, hidden_size)),
            (f"{prefix}.proj.bias", (hidden_size,)),
        )
    ]
    layer = add_plugin(
        [x, conditioning.y, conditioning.mask, *constants],
        plugin,
    )
    return _set_tensor_name(layer.get_output(0), f"{prefix}.output")


def _create_sana_wm_torch_conv2d_plugin(
    trt_module: Any,
    *,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    in_channels: int,
    kernel_shape: tuple[int, int],
    padding: tuple[int, int],
    groups: int,
) -> Any | None:
    creator = _get_sana_wm_torch_conv2d_plugin_creator(trt_module)
    float32_type = trt_module.PluginFieldType.FLOAT32
    int32_type = trt_module.PluginFieldType.INT32
    refs: list[np.ndarray] = []
    fields = []
    int_fields = {
        "out_channels": out_channels,
        "in_channels": in_channels,
        "kernel_h": kernel_shape[0],
        "kernel_w": kernel_shape[1],
        "pad_h": padding[0],
        "pad_w": padding[1],
        "groups": groups,
    }
    for name, value in int_fields.items():
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, int32_type))
    weight_f32 = np.ascontiguousarray(np.asarray(weight, dtype=np.float32).reshape(-1))
    refs.append(weight_f32)
    fields.append(trt_module.PluginField("weight", weight_f32, float32_type))
    if bias is not None:
        bias_f32 = np.ascontiguousarray(np.asarray(bias, dtype=np.float32).reshape(-1))
    else:
        bias_f32 = np.empty((0,), dtype=np.float32)
    refs.append(bias_f32)
    fields.append(trt_module.PluginField("bias", bias_f32, float32_type))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_torch_conv2d", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _add_sana_wm_torch_conv2d_plugin(
    network: Any,
    inp: Any,
    *,
    weights: WeightDict,
    prefix: str,
    in_channels: int,
    out_channels: int,
    kernel_shape: tuple[int, int],
    padding: tuple[int, int],
    groups: int,
    trt_module: Any,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    weight = _required_weight(weights, f"{prefix}.weight")
    bias = _optional_weight(weights, f"{prefix}.bias")
    expected_weight_shape = (
        out_channels,
        in_channels // groups,
        kernel_shape[0],
        kernel_shape[1],
    )
    if weight.shape != expected_weight_shape:
        raise ValueError(
            f"{prefix}.weight must have shape {expected_weight_shape}, got {weight.shape}"
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError(f"{prefix}.bias must have shape ({out_channels},), got {bias.shape}")
    plugin = _create_sana_wm_torch_conv2d_plugin(
        trt_module,
        weight=weight,
        bias=bias,
        out_channels=out_channels,
        in_channels=in_channels,
        kernel_shape=kernel_shape,
        padding=padding,
        groups=groups,
    )
    layer = add_plugin([inp], plugin)
    return _set_tensor_name(layer.get_output(0), name)


def _create_sana_wm_glumbconvtemp_plugin(
    trt_module: Any,
    *,
    shape: SanaWmStage1Shape,
    channels: int,
    hidden: int,
    t_kernel: int,
    inverted_weight: np.ndarray,
    inverted_bias: np.ndarray | None,
    depth_weight: np.ndarray,
    depth_bias: np.ndarray | None,
    point_weight: np.ndarray,
    t_weight: np.ndarray,
) -> Any | None:
    creator = _get_sana_wm_glumbconvtemp_plugin_creator(trt_module)
    float32_type = trt_module.PluginFieldType.FLOAT32
    int32_type = trt_module.PluginFieldType.INT32
    refs: list[np.ndarray] = []
    fields = []
    int_fields = {
        "batch": shape.batch_size,
        "frames": shape.latent_frames,
        "height": shape.latent_height,
        "width": shape.latent_width,
        "channels": channels,
        "hidden": hidden,
        "t_kernel": t_kernel,
    }
    for name, value in int_fields.items():
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, int32_type))
    float_fields = {
        "inverted_weight": inverted_weight,
        "inverted_bias": np.empty((0,), dtype=np.float32)
        if inverted_bias is None
        else inverted_bias,
        "depth_weight": depth_weight,
        "depth_bias": np.empty((0,), dtype=np.float32) if depth_bias is None else depth_bias,
        "point_weight": point_weight,
        "t_weight": t_weight,
    }
    for name, value in float_fields.items():
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, float32_type))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_glumbconvtemp", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _add_sana_wm_glumbconvtemp_plugin(
    network: Any,
    inp: Any,
    *,
    shape: SanaWmStage1Shape,
    weights: WeightDict,
    prefix: str,
    channels: int,
    hidden: int,
    t_kernel: int,
    trt_module: Any,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    expanded = hidden * 2
    inverted_weight = _required_weight(weights, f"{prefix}.inverted_conv.conv.weight")
    inverted_bias = _optional_weight(weights, f"{prefix}.inverted_conv.conv.bias")
    depth_weight = _required_weight(weights, f"{prefix}.depth_conv.conv.weight")
    depth_bias = _optional_weight(weights, f"{prefix}.depth_conv.conv.bias")
    point_weight = _required_weight(weights, f"{prefix}.point_conv.conv.weight")
    point_bias = _optional_weight(weights, f"{prefix}.point_conv.conv.bias")
    t_weight = _required_weight(weights, f"{prefix}.t_conv.weight")
    t_bias = _optional_weight(weights, f"{prefix}.t_conv.bias")
    expected_shapes = {
        f"{prefix}.inverted_conv.conv.weight": (expanded, channels, 1, 1),
        f"{prefix}.depth_conv.conv.weight": (expanded, 1, 3, 3),
        f"{prefix}.point_conv.conv.weight": (channels, hidden, 1, 1),
        f"{prefix}.t_conv.weight": (channels, channels, t_kernel, 1),
    }
    actual_shapes = {
        f"{prefix}.inverted_conv.conv.weight": inverted_weight.shape,
        f"{prefix}.depth_conv.conv.weight": depth_weight.shape,
        f"{prefix}.point_conv.conv.weight": point_weight.shape,
        f"{prefix}.t_conv.weight": t_weight.shape,
    }
    for weight_name, expected in expected_shapes.items():
        actual = actual_shapes[weight_name]
        if actual != expected:
            raise ValueError(f"{weight_name} must have shape {expected}, got {actual}")
    if inverted_bias is not None and inverted_bias.shape != (expanded,):
        raise ValueError(
            f"{prefix}.inverted_conv.conv.bias must have shape ({expanded},), got {inverted_bias.shape}"
        )
    if depth_bias is not None and depth_bias.shape != (expanded,):
        raise ValueError(
            f"{prefix}.depth_conv.conv.bias must have shape ({expanded},), got {depth_bias.shape}"
        )
    if point_bias is not None:
        raise ValueError(f"{prefix}.point_conv.conv.bias is not supported by the MLP plugin")
    if t_bias is not None:
        raise ValueError(f"{prefix}.t_conv.bias is not supported by the MLP plugin")
    plugin = _create_sana_wm_glumbconvtemp_plugin(
        trt_module,
        shape=shape,
        channels=channels,
        hidden=hidden,
        t_kernel=t_kernel,
        inverted_weight=inverted_weight,
        inverted_bias=inverted_bias,
        depth_weight=depth_weight,
        depth_bias=depth_bias,
        point_weight=point_weight,
        t_weight=t_weight,
    )
    layer = add_plugin([inp], plugin)
    return _set_tensor_name(layer.get_output(0), name)


def _create_sana_wm_layer_norm_plugin(trt_module: Any, *, eps: float) -> Any | None:
    creator = _get_sana_wm_layer_norm_plugin_creator(trt_module)
    float32_type = trt_module.PluginFieldType.FLOAT32
    eps_value = np.asarray([eps], dtype=np.float32)
    collection = trt_module.PluginFieldCollection(
        [trt_module.PluginField("eps", eps_value, float32_type)]
    )
    plugin = _create_sana_wm_plugin(creator, "sana_wm_layer_norm", collection)
    _BF16_WEIGHT_REFS.append(eps_value)
    return plugin


def _create_sana_wm_short_conv_plugin(
    trt_module: Any,
    *,
    frames: int,
    spatial: int,
    channels: int,
    weight: np.ndarray,
    bias: np.ndarray | None,
) -> Any | None:
    creator = _get_sana_wm_short_conv_plugin_creator(trt_module)
    float32_type = trt_module.PluginFieldType.FLOAT32
    refs: list[np.ndarray] = []
    fields = []
    for name, value in (
        ("frames", frames),
        ("spatial", spatial),
        ("channels", channels),
        ("kernel_size", int(weight.shape[2])),
    ):
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, trt_module.PluginFieldType.INT32))
    weight_f32 = np.ascontiguousarray(np.asarray(weight[:, 0, :], dtype=np.float32).reshape(-1))
    refs.append(weight_f32)
    fields.append(trt_module.PluginField("weight", weight_f32, float32_type))
    if bias is None:
        bias_f32 = np.empty((0,), dtype=np.float32)
    else:
        bias_f32 = np.ascontiguousarray(np.asarray(bias, dtype=np.float32).reshape(-1))
    refs.append(bias_f32)
    fields.append(trt_module.PluginField("bias", bias_f32, float32_type))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_short_conv", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _sana_wm_camera_prep_plugin_mode() -> str:
    return "hybrid"


def _create_sana_wm_softmax_attention_plugin(
    trt_module: Any,
    *,
    frames: int,
    spatial: int,
    heads: int,
    head_dim: int,
    norm_eps: float,
    camera: bool,
) -> Any | None:
    creator = _get_sana_wm_softmax_attention_plugin_creator(trt_module)
    refs: list[np.ndarray] = []
    fields = []
    for name, value in (
        ("frames", frames),
        ("spatial", spatial),
        ("heads", heads),
        ("head_dim", head_dim),
        ("camera", int(camera)),
    ):
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, trt_module.PluginFieldType.INT32))
    eps_value = np.asarray([norm_eps], dtype=np.float32)
    refs.append(eps_value)
    fields.append(
        trt_module.PluginField(
            "norm_eps",
            eps_value,
            trt_module.PluginFieldType.FLOAT32,
        )
    )
    plugin = _create_sana_wm_plugin(
        creator, "sana_wm_softmax_attention", trt_module.PluginFieldCollection(fields)
    )
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _create_sana_wm_cam_prep_plugin(
    trt_module: Any,
    *,
    frames: int,
    spatial: int,
    heads: int,
    head_dim: int,
    norm_eps: float,
    q_norm_weight: np.ndarray,
    k_norm_weight: np.ndarray,
    use_torch_plugin: bool | None = None,
) -> Any | None:
    if use_torch_plugin is None:
        use_torch_plugin = False
    creator = (
        _get_sana_wm_torch_cam_prep_plugin_creator(trt_module)
        if use_torch_plugin
        else _get_sana_wm_cam_prep_plugin_creator(trt_module)
    )
    float32_type = trt_module.PluginFieldType.FLOAT32
    channels = heads * head_dim
    q_norm_weight = np.asarray(q_norm_weight)
    k_norm_weight = np.asarray(k_norm_weight)
    if q_norm_weight.shape != (channels,):
        raise ValueError(
            f"SANA-WM q_norm_cam.weight must have shape ({channels},), got {q_norm_weight.shape}"
        )
    if k_norm_weight.shape != (channels,):
        raise ValueError(
            f"SANA-WM k_norm_cam.weight must have shape ({channels},), got {k_norm_weight.shape}"
        )
    refs: list[np.ndarray] = []
    fields = []
    for name, value in (
        ("frames", frames),
        ("spatial", spatial),
        ("heads", heads),
        ("head_dim", head_dim),
    ):
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, trt_module.PluginFieldType.INT32))
    eps_value = np.asarray([norm_eps], dtype=np.float32)
    refs.append(eps_value)
    fields.append(trt_module.PluginField("norm_eps", eps_value, float32_type))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(
        creator,
        "sana_wm_torch_cam_prep" if use_torch_plugin else "sana_wm_cam_prep",
        collection,
    )
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _create_sana_wm_camera_beta_discount_plugin(
    trt_module: Any,
    *,
    frames: int,
    spatial: int,
    heads: int,
) -> Any | None:
    creator = _get_sana_wm_camera_beta_discount_plugin_creator(trt_module)
    refs: list[np.ndarray] = []
    fields = []
    for field_name, value in (
        ("frames", frames),
        ("spatial", spatial),
        ("heads", heads),
    ):
        field_value = np.asarray([value], dtype=np.int32)
        refs.append(field_value)
        fields.append(
            trt_module.PluginField(
                field_name,
                field_value,
                trt_module.PluginFieldType.INT32,
            )
        )
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_camera_beta_discount", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _create_sana_wm_ucpe_plugin(
    trt_module: Any,
    *,
    frames: int,
    spatial: int,
    heads: int,
    head_dim: int,
    inverse: bool,
    tree_reduce: bool = True,
    downscale: bool = False,
    double_rope: bool = False,
    rope_only: bool = False,
) -> Any | None:
    creator = _get_sana_wm_ucpe_plugin_creator(trt_module)
    refs: list[np.ndarray] = []
    fields = []
    for name, value in (
        ("frames", frames),
        ("spatial", spatial),
        ("heads", heads),
        ("head_dim", head_dim),
        ("inverse", int(inverse)),
        ("tree_reduce", int(tree_reduce)),
        ("downscale", int(downscale)),
        ("double_rope", int(double_rope)),
        ("rope_only", int(rope_only)),
    ):
        arr = np.asarray([value], dtype=np.int32)
        refs.append(arr)
        fields.append(trt_module.PluginField(name, arr, trt_module.PluginFieldType.INT32))
    collection = trt_module.PluginFieldCollection(fields)
    plugin = _create_sana_wm_plugin(creator, "sana_wm_ucpe", collection)
    _BF16_WEIGHT_REFS.extend(refs)
    return plugin


def _add_trt_count_loop(network: Any, trt_module: Any, trip_count: int) -> tuple[Any, Any]:
    loop = network.add_loop()
    count = _add_int32_scalar(network, trt_module, trip_count)
    loop.add_trip_limit(count, trt_module.TripLimit.COUNT)
    return loop, count


def _add_gelu_tanh(network: Any, inp: Any, *, trt_module: Any, rank: int, dtype: np.dtype) -> Any:
    x_sq = network.add_elementwise(
        inp,
        inp,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    x_cu = network.add_elementwise(
        x_sq,
        inp,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    coeff = _add_ranked_scalar(network, trt_module, rank, 0.044715, dtype=dtype)
    scaled_cube = network.add_elementwise(
        x_cu,
        coeff,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    inner_sum = network.add_elementwise(
        inp,
        scaled_cube,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    sqrt_2_over_pi = _add_ranked_scalar(
        network,
        trt_module,
        rank,
        float(np.sqrt(2.0 / np.pi)),
        dtype=dtype,
    )
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi,
        inner_sum,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    tanh_l = network.add_activation(tanh_arg, trt_module.ActivationType.TANH)
    one = _add_ranked_scalar(network, trt_module, rank, 1.0, dtype=dtype)
    one_plus_tanh = network.add_elementwise(
        one,
        tanh_l.get_output(0),
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    half = _add_ranked_scalar(network, trt_module, rank, 0.5, dtype=dtype)
    half_x = network.add_elementwise(
        half,
        inp,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    return network.add_elementwise(
        half_x,
        one_plus_tanh,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)


def _conv_output_dim(size: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def _add_patch_embed3d(
    network: Any,
    inp: Any,
    *,
    weights: WeightDict,
    prefix: str,
    input_shape: tuple[int, int, int, int, int],
    patch_size: tuple[int, int, int],
    trt_module: Any,
    dtype: np.dtype,
) -> tuple[Any, int, int]:
    """Lower upstream ``PatchEmbedMS3D`` to TensorRT Conv3D + flatten."""
    weight = _required_weight(weights, f"{prefix}.proj.weight")
    bias = _optional_weight(weights, f"{prefix}.proj.bias")
    if weight.ndim != 5:
        raise ValueError(f"{prefix}.proj.weight must be rank-5, got shape {weight.shape}")

    batch, in_channels, frames, height, width = input_shape
    out_channels, conv_in_channels, kernel_t, kernel_h, kernel_w = weight.shape
    if conv_in_channels != in_channels:
        raise ValueError(
            f"{prefix}.proj.weight expects {conv_in_channels} input channels, "
            f"but the runtime contract provides {in_channels}"
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError(f"{prefix}.proj.bias must have shape ({out_channels},), got {bias.shape}")

    out_frames = _conv_output_dim(frames, kernel_t, patch_size[0])
    out_height = _conv_output_dim(height, kernel_h, patch_size[1])
    out_width = _conv_output_dim(width, kernel_w, patch_size[2])
    token_count = out_frames * out_height * out_width

    if not _is_bf16_dtype(dtype) or patch_size != (1, 1, 1) or (
        kernel_t,
        kernel_h,
        kernel_w,
    ) != (1, 1, 1):
        raise ValueError("SANA-WM patch embedding requires BF16 1x1x1 convolution")
    input_bfchw = network.add_shuffle(inp)
    input_bfchw.first_transpose = trt_module.Permutation([0, 2, 1, 3, 4])
    input_bfchw.reshape_dims = (batch * frames, in_channels, height, width)
    torch_weights: WeightDict = WeightDict()
    torch_weights[f"{prefix}.proj_torch_conv.weight"] = np.ascontiguousarray(
        np.asarray(weight[:, :, 0, :, :], dtype=np.float32)
    )
    if bias is not None:
        torch_weights[f"{prefix}.proj_torch_conv.bias"] = bias
    conv2d = _add_sana_wm_torch_conv2d_plugin(
        network,
        input_bfchw.get_output(0),
        weights=torch_weights,
        prefix=f"{prefix}.proj_torch_conv",
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_shape=(kernel_h, kernel_w),
        padding=(0, 0),
        groups=1,
        trt_module=trt_module,
        name=f"{prefix}.proj_torch_conv.output",
    )
    if conv2d is None:
        raise RuntimeError("SANA-WM patch embedding plugin did not produce an output")
    restore = network.add_shuffle(conv2d)
    restore.reshape_dims = (batch, out_frames, out_channels, out_height, out_width)
    restore.second_transpose = trt_module.Permutation([0, 1, 3, 4, 2])
    flatten = network.add_shuffle(restore.get_output(0))
    flatten.reshape_dims = (batch, token_count, out_channels)
    return _set_tensor_name(flatten.get_output(0), f"{prefix}.tokens"), token_count, out_channels


def lower_sana_wm_stage1_frontend(
    network: Any,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    weights: WeightDict,
    raw_config: dict,
    *,
    trt_module: Any,
    dtype: np.dtype,
) -> SanaWmStage1Frontend:
    """Lower the Stage-1 pre-transformer embedder stack with TRT APIs."""
    patch_size = _stage1_patch_size(raw_config)
    x_tokens, token_count, hidden_size = _add_patch_embed3d(
        network,
        inputs["x"],
        weights=weights,
        prefix="x_embedder",
        input_shape=(
            shape.batch_size,
            shape.latent_channels,
            shape.latent_frames,
            shape.latent_height,
            shape.latent_width,
        ),
        patch_size=patch_size,
        trt_module=trt_module,
        dtype=dtype,
    )

    plucker_tokens = None
    if _stage1_uses_chunk_plucker(raw_config, weights):
        plucker_tokens, plucker_count, plucker_hidden = _add_patch_embed3d(
            network,
            inputs["chunk_plucker"],
            weights=weights,
            prefix="plucker_embedder",
            input_shape=(
                shape.batch_size,
                shape.chunk_plucker_channels,
                shape.latent_frames,
                shape.latent_height,
                shape.latent_width,
            ),
            patch_size=patch_size,
            trt_module=trt_module,
            dtype=dtype,
        )
        if plucker_count != token_count or plucker_hidden != hidden_size:
            raise ValueError(
                "SANA-WM plucker embedder output shape does not match x embedder: "
                f"x=({token_count}, {hidden_size}) plucker=({plucker_count}, "
                f"{plucker_hidden})"
            )

    return SanaWmStage1Frontend(
        x_tokens=x_tokens,
        token_count=token_count,
        hidden_size=hidden_size,
        plucker_tokens=plucker_tokens,
    )


def _add_timestep_frequency_embedding(
    network: Any,
    timestep: Any,
    shape: SanaWmStage1Shape,
    *,
    frequency_embedding_size: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    half = frequency_embedding_size // 2
    if half * 2 != frequency_embedding_size:
        raise ValueError("SANA-WM timestep embedding size must be even")

    timestep_4d = network.add_shuffle(timestep)
    timestep_4d.reshape_dims = (shape.batch_size, 1, shape.latent_frames, 1)
    freqs = np.exp(-np.log(10000.0) * np.arange(0, half, dtype=np.float32) / float(half))
    freqs_t = _add_constant(
        network,
        trt_module,
        (1, 1, 1, half),
        freqs.reshape(1, 1, 1, half),
        dtype=np.float32,
    )
    args = network.add_elementwise(
        timestep_4d.get_output(0),
        freqs_t,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    cos = network.add_unary(args, trt_module.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(args, trt_module.UnaryOperation.SIN).get_output(0)
    concat = network.add_concatenation([cos, sin])
    concat.axis = 3
    t_freq = concat.get_output(0)
    if dtype != np.float32:
        t_freq = _cast_to_dtype(network, t_freq, _trt_dtype_for_np(trt_module, dtype))
    return _set_tensor_name(t_freq, "timestep.frequency_embedding")


def lower_sana_wm_stage1_conditioning(
    network: Any,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    weights: WeightDict,
    *,
    hidden_size: int,
    trt_module: Any,
    dtype: np.dtype,
) -> SanaWmStage1Conditioning:
    """Lower timestep and caption embedders used before the first block."""
    if not _is_bf16_dtype(dtype):
        raise ValueError("SANA-WM Stage-1 conditioning requires BF16")
    frequency_embedding_size = int(_required_weight(weights, "t_embedder.mlp.0.weight").shape[0])
    t_freq = _add_timestep_frequency_embedding(
        network,
        inputs["timestep"],
        shape,
        frequency_embedding_size=frequency_embedding_size,
        trt_module=trt_module,
        dtype=np.float32,
    )
    t_freq_for_mlp = t_freq
    t_freq_debug = t_freq
    if dtype != np.float32:
        t_freq_for_mlp = _set_tensor_name(
            _cast_to_dtype(network, t_freq, _trt_dtype_for_np(trt_module, dtype)),
            "timestep.frequency_embedding.bf16",
        )
        t_freq_debug = t_freq_for_mlp
    plugin = _create_sana_wm_timestep_plugin(
        trt_module,
        weights=weights,
        frequency_dim=frequency_embedding_size,
        hidden_size=hidden_size,
    )
    plugin_layer = network.add_plugin_v2([inputs["timestep"]], plugin)
    if plugin_layer is None:
        raise RuntimeError("TensorRT failed to add SANA-WM timestep plugin")
    t = _set_tensor_name(plugin_layer.get_output(0), "t_embedder.output")
    t0 = _set_tensor_name(plugin_layer.get_output(1), "t_block.output")

    y = _apply_sana_wm_caption_embed_plugin(
        network,
        inputs["y"],
        weights=weights,
        text_embed_dim=shape.text_embed_dim,
        hidden_size=hidden_size,
        trt_module=trt_module,
        dtype=dtype,
    )
    if y is None:
        raise RuntimeError("SANA-WM caption embedding plugin did not produce an output")
    return SanaWmStage1Conditioning(t_freq=t_freq_debug, t=t, t0=t0, y=y, mask=inputs["mask"])


def _add_layernorm_no_affine(
    network: Any,
    inp: Any,
    *,
    rank: int,
    eps: float,
    trt_module: Any,
    dtype: Any,
    name: str,
) -> Any:
    del rank
    if not _is_bf16_dtype(dtype):
        raise ValueError("SANA-WM layer normalization requires BF16")
    plugin = _create_sana_wm_layer_norm_plugin(trt_module, eps=eps)
    layer = network.add_plugin_v2([inp], plugin)
    if layer is None:
        raise RuntimeError("TensorRT failed to add SANA-WM layer normalization plugin")
    return _set_tensor_name(layer.get_output(0), name)


def _add_slice(
    network: Any,
    inp: Any,
    *,
    start: tuple[int, ...],
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    name: str,
) -> Any:
    layer = network.add_slice(inp, start=start, shape=shape, stride=stride)
    return _set_tensor_name(layer.get_output(0), name)


def _add_t2i_modulate(
    network: Any,
    inp: Any,
    shift: Any,
    scale: Any,
    *,
    rank: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    if rank != 4 or not _is_bf16_dtype(dtype):
        raise ValueError("SANA-WM T2I modulation requires rank-4 BF16 tensors")
    plugin = _create_sana_wm_t2i_modulate_plugin(trt_module)
    layer = network.add_plugin_v2([inp, shift, scale], plugin)
    if layer is None:
        raise RuntimeError("TensorRT failed to add SANA-WM T2I modulation plugin")
    return _set_tensor_name(layer.get_output(0), name)


def _add_rmsnorm(
    network: Any,
    inp: Any,
    weight: np.ndarray,
    *,
    rank: int,
    eps: float,
    trt_module: Any,
    dtype: Any,
    name: str,
    keep_fp32_output: bool = False,
) -> Any:
    if weight.ndim != 1:
        raise ValueError(f"{name} RMSNorm weight must be rank-1, got {weight.shape}")
    output_dtype = getattr(inp, "dtype", None)
    if dtype != np.float32:
        inp = _cast_to_dtype(network, inp, trt_module.float32)
    squared = network.add_elementwise(
        inp,
        inp,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    mean_square = network.add_reduce(
        squared,
        trt_module.ReduceOperation.AVG,
        1 << (rank - 1),
        True,
    ).get_output(0)
    eps_t = _add_ranked_scalar(network, trt_module, rank, eps, dtype=np.float32)
    variance_eps = network.add_elementwise(
        mean_square,
        eps_t,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    rms = network.add_unary(variance_eps, trt_module.UnaryOperation.SQRT).get_output(0)
    inv_rms = network.add_unary(rms, trt_module.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        inp,
        inv_rms,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    weight_shape = (1,) * (rank - 1) + (int(weight.shape[0]),)
    gamma_values = (
        np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
        if keep_fp32_output
        else _fp32_parameter_values_for_runtime_dtype(weight, dtype)
    )
    gamma = _add_constant(
        network,
        trt_module,
        weight_shape,
        gamma_values.reshape(weight_shape),
        dtype=np.float32,
    )
    result = network.add_elementwise(
        normalized,
        gamma,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    if dtype != np.float32 and output_dtype is not None and not keep_fp32_output:
        result = _cast_to_dtype(network, result, output_dtype)
    return _set_tensor_name(result, name)


def _add_relu(network: Any, inp: Any, *, trt_module: Any, name: str) -> Any:
    return _set_tensor_name(
        network.add_activation(inp, trt_module.ActivationType.RELU).get_output(0),
        name,
    )


def _reshape_qkv_component(
    network: Any,
    qkv_heads: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    component: int,
    num_heads: int,
    head_dim: int,
    name: str,
) -> Any:
    sliced = _add_slice(
        network,
        qkv_heads,
        start=(0, 0, component, 0, 0),
        shape=(shape.batch_size, frontend.token_count, 1, num_heads, head_dim),
        stride=(1, 1, 1, 1, 1),
        name=f"{name}.slice",
    )
    flat = network.add_shuffle(sliced)
    flat.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        frontend.hidden_size,
    )
    return _set_tensor_name(flat.get_output(0), name)


def _permute_bnhd_to_bhdn(
    network: Any,
    inp: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    name: str,
) -> Any:
    heads = network.add_shuffle(inp)
    heads.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        num_heads,
        head_dim,
    )
    permuted = network.add_shuffle(heads.get_output(0))
    permuted.first_transpose = trt_module.Permutation([0, 2, 3, 1])
    return _set_tensor_name(permuted.get_output(0), name)


def _stage1_pos_embed_type(raw_config: dict) -> str:
    model = _model_dict(raw_config)
    return str(raw_config.get("pos_embed_type", model.get("pos_embed_type", "wan_rope")))


def _stage1_rope_fhw_dim(raw_config: dict) -> tuple[int, int, int] | None:
    model = _model_dict(raw_config)
    value = raw_config.get("rope_fhw_dim", model.get("rope_fhw_dim"))
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"SANA-WM rope_fhw_dim must be a length-3 list, got {value!r}")
    return int(value[0]), int(value[1]), int(value[2])


def _wan_rope_axis_angles(
    dim: int,
    max_seq_len: int,
    *,
    theta: float,
) -> np.ndarray:
    if dim % 2 != 0:
        raise ValueError(f"Wan RoPE axis dimension must be even, got {dim}")
    positions = np.arange(max_seq_len, dtype=np.float64)
    dims = np.arange(0, dim, 2, dtype=np.float64)[: dim // 2]
    inv_freq = 1.0 / (theta ** (dims / float(dim)))
    return np.outer(positions, inv_freq)


def _wan_rope_angles(
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    head_dim: int,
    max_seq_len: int = 1024,
    theta: float = 10000.0,
    fhw_dim: tuple[int, int, int] | None = None,
) -> np.ndarray:
    if head_dim % 2 != 0:
        raise ValueError(f"Wan RoPE head dimension must be even, got {head_dim}")
    if max(latent_frames, latent_height, latent_width) > max_seq_len:
        raise ValueError(
            "SANA-WM Wan RoPE max_seq_len is smaller than the latent F/H/W "
            f"shape: max_seq_len={max_seq_len}, "
            f"FHW=({latent_frames}, {latent_height}, {latent_width})"
        )

    if fhw_dim is None:
        h_dim = w_dim = 2 * (head_dim // 6)
        t_dim = head_dim - h_dim - w_dim
    else:
        t_dim, h_dim, w_dim = fhw_dim
        if t_dim + h_dim + w_dim != head_dim:
            raise ValueError(
                f"SANA-WM rope_fhw_dim must sum to the attention head dim: {fhw_dim} vs {head_dim}"
            )

    split = (
        head_dim // 2 - 2 * (head_dim // 6),
        head_dim // 6,
        head_dim // 6,
    )
    if (t_dim // 2, h_dim // 2, w_dim // 2) != split:
        raise NotImplementedError(
            "SANA-WM native builder currently supports the public Wan RoPE "
            f"split {split}; got real dims {(t_dim, h_dim, w_dim)}"
        )

    f_angles = _wan_rope_axis_angles(t_dim, max_seq_len, theta=theta)
    h_angles = _wan_rope_axis_angles(h_dim, max_seq_len, theta=theta)
    w_angles = _wan_rope_axis_angles(w_dim, max_seq_len, theta=theta)
    freqs_f = np.broadcast_to(
        f_angles[:latent_frames].reshape(latent_frames, 1, 1, split[0]),
        (latent_frames, latent_height, latent_width, split[0]),
    )
    freqs_h = np.broadcast_to(
        h_angles[:latent_height].reshape(1, latent_height, 1, split[1]),
        (latent_frames, latent_height, latent_width, split[1]),
    )
    freqs_w = np.broadcast_to(
        w_angles[:latent_width].reshape(1, 1, latent_width, split[2]),
        (latent_frames, latent_height, latent_width, split[2]),
    )
    return np.concatenate([freqs_f, freqs_h, freqs_w], axis=-1).reshape(
        latent_frames * latent_height * latent_width,
        head_dim // 2,
    )


def _ucpe_cam_rope_angles(full_angles: np.ndarray, head_dim: int) -> np.ndarray:
    """Slice Wan RoPE frequencies exactly like upstream ``_slice_rope_for_cam``."""
    rope_dim = head_dim // 2
    orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
    orig_h_size = head_dim // 6
    new_t_size = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h_size = rope_dim // 6
    new_w_size = rope_dim // 6
    if rope_dim % 2 != 0:
        raise ValueError(f"SANA-WM UCPE RoPE dim must be even, got {rope_dim}")
    return np.concatenate(
        [
            full_angles[..., :new_t_size],
            full_angles[..., orig_t_size : orig_t_size + new_h_size],
            full_angles[
                ...,
                orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size,
            ],
        ],
        axis=-1,
    )


def _add_wan_rope_constants(
    network: Any,
    shape: SanaWmStage1Shape,
    head_dim: int,
    raw_config: dict,
    *,
    trt_module: Any,
) -> tuple[Any, Any]:
    model = _model_dict(raw_config)
    max_seq_len = int(raw_config.get("max_seq_len", model.get("max_seq_len", 1024)))
    theta = float(raw_config.get("theta", model.get("theta", 10000.0)))
    angles = _wan_rope_angles(
        latent_frames=shape.latent_frames,
        latent_height=shape.latent_height,
        latent_width=shape.latent_width,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        theta=theta,
        fhw_dim=_stage1_rope_fhw_dim(raw_config),
    )
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    const_shape = (1, 1, head_dim // 2, 1, token_count)
    cos = np.cos(angles).T.reshape(const_shape)
    sin = np.sin(angles).T.reshape(const_shape)
    return (
        _add_constant(network, trt_module, const_shape, cos, dtype=np.float32),
        _add_constant(network, trt_module, const_shape, sin, dtype=np.float32),
    )


def _ucpe_cam_rope_arrays(
    shape: SanaWmStage1Shape,
    head_dim: int,
    raw_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    model = _model_dict(raw_config)
    max_seq_len = int(raw_config.get("max_seq_len", model.get("max_seq_len", 1024)))
    theta = float(raw_config.get("theta", model.get("theta", 10000.0)))
    full_angles = _wan_rope_angles(
        latent_frames=shape.latent_frames,
        latent_height=shape.latent_height,
        latent_width=shape.latent_width,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        theta=theta,
        fhw_dim=_stage1_rope_fhw_dim(raw_config),
    )
    angles = _ucpe_cam_rope_angles(full_angles, head_dim)
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    rope_dim = head_dim // 2
    const_shape = (1, 1, token_count, rope_dim // 2, 1)
    cos = np.cos(angles).reshape(const_shape)
    sin = np.sin(angles).reshape(const_shape)
    return cos, sin


def _add_ucpe_cam_rope_constants(
    network: Any,
    shape: SanaWmStage1Shape,
    head_dim: int,
    raw_config: dict,
    *,
    trt_module: Any,
) -> tuple[Any, Any]:
    cos, sin = _ucpe_cam_rope_arrays(shape, head_dim, raw_config)
    return (
        _add_constant(network, trt_module, cos.shape, cos, dtype=np.float32),
        _add_constant(network, trt_module, sin.shape, sin, dtype=np.float32),
    )


def _add_ucpe_cam_rope_residual_constants(
    network: Any,
    shape: SanaWmStage1Shape,
    head_dim: int,
    raw_config: dict,
    *,
    trt_module: Any,
) -> tuple[Any, Any]:
    cos, sin = _ucpe_cam_rope_arrays(shape, head_dim, raw_config)
    cos_residual = (cos - cos.astype(np.float32).astype(np.float64)).astype(np.float32)
    sin_residual = (sin - sin.astype(np.float32).astype(np.float64)).astype(np.float32)
    return (
        _add_constant(
            network,
            trt_module,
            cos_residual.shape,
            cos_residual,
            dtype=np.float32,
        ),
        _add_constant(
            network,
            trt_module,
            sin_residual.shape,
            sin_residual,
            dtype=np.float32,
        ),
    )


def _add_softmax_rope_high_low_constants(
    network: Any,
    shape: SanaWmStage1Shape,
    head_dim: int,
    raw_config: dict,
    *,
    camera: bool,
    trt_module: Any,
) -> tuple[Any, Any, Any, Any]:
    if camera:
        cos, sin = _ucpe_cam_rope_arrays(shape, head_dim, raw_config)
    else:
        model = _model_dict(raw_config)
        angles = _wan_rope_angles(
            latent_frames=shape.latent_frames,
            latent_height=shape.latent_height,
            latent_width=shape.latent_width,
            head_dim=head_dim,
            max_seq_len=int(raw_config.get("max_seq_len", model.get("max_seq_len", 1024))),
            theta=float(raw_config.get("theta", model.get("theta", 10000.0))),
            fhw_dim=_stage1_rope_fhw_dim(raw_config),
        )
        cos = np.cos(angles)
        sin = np.sin(angles)
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    pair_count = head_dim // (4 if camera else 2)
    cos = np.ascontiguousarray(cos.reshape(token_count, pair_count), dtype=np.float64)
    sin = np.ascontiguousarray(sin.reshape(token_count, pair_count), dtype=np.float64)
    cos_high = np.ascontiguousarray(cos, dtype=np.float32)
    sin_high = np.ascontiguousarray(sin, dtype=np.float32)
    cos_low = np.ascontiguousarray(cos - cos_high.astype(np.float64), dtype=np.float32)
    sin_low = np.ascontiguousarray(sin - sin_high.astype(np.float64), dtype=np.float32)
    const_shape = (token_count, pair_count)
    return tuple(
        _add_constant(network, trt_module, const_shape, value, dtype=np.float32)
        for value in (cos_high, sin_high, cos_low, sin_low)
    )


def _apply_wan_rope_to_bhdn(
    network: Any,
    hidden: Any,
    cos: Any,
    sin: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    if head_dim % 2 != 0:
        raise ValueError(f"{name} Wan RoPE head dimension must be even")
    hidden_fp32 = hidden
    if dtype != np.float32:
        hidden_fp32 = _cast_to_dtype(network, hidden, trt_module.float32)
    pairs = network.add_shuffle(hidden_fp32)
    pairs.reshape_dims = (
        shape.batch_size,
        num_heads,
        head_dim // 2,
        2,
        frontend.token_count,
    )
    pair_shape = (
        shape.batch_size,
        num_heads,
        head_dim // 2,
        1,
        frontend.token_count,
    )
    even = _add_slice(
        network,
        pairs.get_output(0),
        start=(0, 0, 0, 0, 0),
        shape=pair_shape,
        stride=(1, 1, 1, 1, 1),
        name=f"{name}.even",
    )
    odd = _add_slice(
        network,
        pairs.get_output(0),
        start=(0, 0, 0, 1, 0),
        shape=pair_shape,
        stride=(1, 1, 1, 1, 1),
        name=f"{name}.odd",
    )
    even_cos = network.add_elementwise(
        even,
        cos,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    odd_sin = network.add_elementwise(
        odd,
        sin,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    even_rot = network.add_elementwise(
        even_cos,
        odd_sin,
        trt_module.ElementWiseOperation.SUB,
    ).get_output(0)
    even_sin = network.add_elementwise(
        even,
        sin,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    odd_cos = network.add_elementwise(
        odd,
        cos,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    odd_rot = network.add_elementwise(
        even_sin,
        odd_cos,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)

    rotated_pairs = network.add_concatenation([even_rot, odd_rot])
    rotated_pairs.axis = 3
    rotated = network.add_shuffle(rotated_pairs.get_output(0))
    rotated.reshape_dims = (
        shape.batch_size,
        num_heads,
        head_dim,
        frontend.token_count,
    )
    out = rotated.get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    return _set_tensor_name(out, name)


def _lower_sana_wm_wan_rope_qk(
    network: Any,
    q: Any,
    k: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    raw_config: dict,
    *,
    prefix: str,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    dtype: np.dtype,
) -> tuple[Any, Any, Any, Any, Any]:
    if not _model_bool(raw_config, "use_pe", True):
        return q, k
    pos_embed_type = _stage1_pos_embed_type(raw_config)
    if pos_embed_type != "wan_rope":
        raise NotImplementedError(
            "SANA-WM native Stage-1 builder currently supports only "
            f"pos_embed_type='wan_rope', got {pos_embed_type!r}"
        )
    cos, sin = _add_wan_rope_constants(
        network,
        shape,
        head_dim,
        raw_config,
        trt_module=trt_module,
    )
    return (
        _apply_wan_rope_to_bhdn(
            network,
            q,
            cos,
            sin,
            shape,
            frontend,
            num_heads=num_heads,
            head_dim=head_dim,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.attn.q_rot",
        ),
        _apply_wan_rope_to_bhdn(
            network,
            k,
            cos,
            sin,
            shape,
            frontend,
            num_heads=num_heads,
            head_dim=head_dim,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.attn.k_rot",
        ),
    )


def _add_bidirectional_short_conv1d(
    network: Any,
    inp: Any,
    weight: np.ndarray,
    bias: np.ndarray | None,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
    channels: int | None = None,
) -> Any:
    channels = frontend.hidden_size if channels is None else channels
    if weight.ndim != 3 or weight.shape[0] != channels or weight.shape[1] != 1:
        raise ValueError(f"{name}.weight must have shape ({channels}, 1, K), got {weight.shape}")
    if bias is not None and bias.shape != (channels,):
        raise ValueError(f"{name}.bias must have shape ({channels},), got {bias.shape}")
    kernel_size = int(weight.shape[2])
    if kernel_size <= 0:
        raise ValueError(f"{name}.weight kernel size must be positive")
    if not _is_bf16_dtype(dtype):
        raise ValueError("SANA-WM bidirectional short convolution requires BF16")
    spatial_tokens = shape.latent_height * shape.latent_width
    plugin = _create_sana_wm_short_conv_plugin(
        trt_module,
        frames=shape.latent_frames,
        spatial=spatial_tokens,
        channels=channels,
        weight=weight,
        bias=bias,
    )
    layer = network.add_plugin_v2([inp], plugin)
    if layer is None:
        raise RuntimeError("TensorRT failed to add SANA-WM short convolution plugin")
    return _set_tensor_name(layer.get_output(0), f"{name}.output")


def _lower_sana_wm_gdn_frame_gates(
    network: Any,
    x_msa: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    *,
    block_index: int,
    num_heads: int,
    trt_module: Any,
    dtype: np.dtype,
) -> tuple[Any, Any, Any, Any, Any]:
    if not _is_bf16_dtype(dtype):
        raise ValueError("SANA-WM GDN frame gates require BF16")
    prefix = f"blocks.{block_index}.attn"
    beta = _add_sana_wm_bf16_linear_plugin(
        network,
        x_msa,
        weights=weights,
        prefix=f"{prefix}.beta_proj",
        input_dim=frontend.hidden_size,
        output_dim=num_heads,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.beta_proj",
        activation=1,
    )
    if beta is None:
        raise RuntimeError("SANA-WM beta projection plugin did not produce an output")
    beta_4d = network.add_shuffle(beta)
    beta_4d.reshape_dims = (
        shape.batch_size,
        shape.latent_frames,
        shape.latent_height * shape.latent_width,
        num_heads,
    )
    beta_bhts = network.add_shuffle(beta_4d.get_output(0))
    beta_bhts.first_transpose = trt_module.Permutation([0, 3, 1, 2])
    beta_t = _set_tensor_name(beta_bhts.get_output(0), f"{prefix}.beta")

    spatial_tokens = shape.latent_height * shape.latent_width
    x_frame = _apply_sana_wm_frame_mean_plugin(
        network,
        x_msa,
        frames=shape.latent_frames,
        spatial=spatial_tokens,
        trt_module=trt_module,
        name=f"{prefix}.frame_mean_plugin",
    )
    gate = _add_sana_wm_bf16_linear_plugin(
        network,
        x_frame,
        weights=weights,
        prefix=f"{prefix}.gate_proj",
        input_dim=frontend.hidden_size,
        output_dim=num_heads,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.gate_proj",
    )
    if gate is None:
        raise RuntimeError("SANA-WM gate projection plugin did not produce an output")
    gate = _cast_to_dtype(network, gate, trt_module.float32)

    dt_bias = _required_weight(weights, f"{prefix}.dt_bias")
    a_log = _required_weight(weights, f"{prefix}.A_log")
    if dt_bias.shape != (num_heads,) or a_log.shape != (num_heads,):
        raise ValueError(
            f"{prefix}.dt_bias and A_log must have shape ({num_heads},), got "
            f"{dt_bias.shape} and {a_log.shape}"
        )
    runtime_dt_bias = _fp32_parameter_values_for_runtime_dtype(dt_bias, dtype)
    runtime_a_log = _fp32_parameter_values_for_runtime_dtype(a_log, dtype)
    dt = _add_constant(
        network,
        trt_module,
        (1, 1, num_heads),
        runtime_dt_bias.reshape(1, 1, num_heads),
        dtype=np.float32,
    )
    gate_dt = network.add_elementwise(
        gate,
        dt,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    decay = _apply_sana_wm_decay_plugin(
        network,
        gate_dt,
        a_log_values=runtime_a_log,
        num_heads=num_heads,
        trt_module=trt_module,
        name=f"{prefix}.decay_plugin",
    )
    if decay is None:
        raise RuntimeError("SANA-WM decay plugin did not produce an output")
    decay_bht = network.add_shuffle(decay)
    decay_bht.first_transpose = trt_module.Permutation([0, 2, 1])
    return (
        beta_t,
        _set_tensor_name(decay_bht.get_output(0), f"{prefix}.decay"),
        _set_tensor_name(x_frame, f"{prefix}.x_frame"),
        _set_tensor_name(gate, f"{prefix}.gate"),
        _set_tensor_name(gate_dt, f"{prefix}.gate_dt"),
    )


def lower_sana_wm_stage1_camera_preamble(
    network: Any,
    x_msa: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
    softmax_attention: bool = False,
) -> SanaWmStage1CameraPreamble:
    """Lower the regular camera-branch QKV path up to UCPE transforms."""
    prefix = f"blocks.{block_index}.attn"
    compress = _stage1_cam_attn_compress(raw_config)
    if frontend.hidden_size % compress != 0:
        raise ValueError(
            "SANA-WM camera hidden size must divide by cam_attn_compress: "
            f"{frontend.hidden_size} / {compress}"
        )
    cam_dim = frontend.hidden_size // compress
    main_head_dim = _stage1_linear_head_dim(raw_config, frontend.hidden_size)
    main_heads = frontend.hidden_size // main_head_dim
    if main_heads % compress != 0:
        raise ValueError(
            f"SANA-WM camera heads must divide by cam_attn_compress: {main_heads} / {compress}"
        )
    cam_heads = main_heads // compress
    if cam_dim % cam_heads != 0:
        raise ValueError(f"SANA-WM camera dim {cam_dim} must divide by heads {cam_heads}")
    cam_head_dim = cam_dim // cam_heads
    if cam_head_dim % 4 != 0:
        raise ValueError(f"SANA-WM UCPE camera head dim must divide by 4, got {cam_head_dim}")
    both_triton_gdn = (
        not softmax_attention
        and _stage1_camctrl_type(raw_config) == "BidirectionalGDNUCPESinglePathLiteLABothTriton"
    )

    q_cam = _add_linear(
        network,
        x_msa,
        weights=weights,
        prefix=f"{prefix}.q_proj_cam",
        input_dim=frontend.hidden_size,
        output_dim=cam_dim,
        batch_prefix_rank=1,
        trt_module=trt_module,
        dtype=dtype,
    )
    k_cam = _add_linear(
        network,
        x_msa,
        weights=weights,
        prefix=f"{prefix}.k_proj_cam",
        input_dim=frontend.hidden_size,
        output_dim=cam_dim,
        batch_prefix_rank=1,
        trt_module=trt_module,
        dtype=dtype,
    )
    v_cam = _add_linear(
        network,
        x_msa,
        weights=weights,
        prefix=f"{prefix}.v_proj_cam",
        input_dim=frontend.hidden_size,
        output_dim=cam_dim,
        batch_prefix_rank=1,
        trt_module=trt_module,
        dtype=dtype,
    )

    q_conv_weight = _optional_weight(weights, f"{prefix}.conv_q_cam.weight")
    if q_conv_weight is not None:
        q_cam = _add_bidirectional_short_conv1d(
            network,
            q_cam,
            q_conv_weight,
            _optional_weight(weights, f"{prefix}.conv_q_cam.bias"),
            shape,
            frontend,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.conv_q_cam",
            channels=cam_dim,
        )
    k_conv_weight = _optional_weight(weights, f"{prefix}.conv_k_cam.weight")
    if k_conv_weight is not None:
        k_cam = _add_bidirectional_short_conv1d(
            network,
            k_cam,
            k_conv_weight,
            _optional_weight(weights, f"{prefix}.conv_k_cam.bias"),
            shape,
            frontend,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.conv_k_cam",
            channels=cam_dim,
        )
    v_conv_weight = _optional_weight(weights, f"{prefix}.conv_v_cam.weight")
    if v_conv_weight is not None:
        v_cam = _add_bidirectional_short_conv1d(
            network,
            v_cam,
            v_conv_weight,
            _optional_weight(weights, f"{prefix}.conv_v_cam.bias"),
            shape,
            frontend,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.conv_v_cam",
            channels=cam_dim,
        )

    q_cam_raw = q_cam
    k_cam_raw = k_cam
    v_cam_raw = v_cam
    q_norm_weight_values = _fp32_parameter_values_for_runtime_dtype(
        _required_weight(weights, f"{prefix}.q_norm_cam.weight"),
        dtype,
    )
    k_norm_weight_values = _fp32_parameter_values_for_runtime_dtype(
        _required_weight(weights, f"{prefix}.k_norm_cam.weight"),
        dtype,
    )

    q_cam = _add_rmsnorm(
        network,
        q_cam,
        _required_weight(weights, f"{prefix}.q_norm_cam.weight"),
        rank=3,
        eps=_stage1_norm_eps(raw_config),
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.q_norm_cam.output",
        keep_fp32_output=both_triton_gdn,
    )
    k_cam = _add_rmsnorm(
        network,
        k_cam,
        _required_weight(weights, f"{prefix}.k_norm_cam.weight"),
        rank=3,
        eps=_stage1_norm_eps(raw_config),
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.k_norm_cam.output",
        keep_fp32_output=both_triton_gdn,
    )
    if not softmax_attention:
        q_cam = _add_relu(
            network,
            q_cam,
            trt_module=trt_module,
            name=f"{prefix}.q_cam_relu",
        )
        k_cam = _add_relu(
            network,
            k_cam,
            trt_module=trt_module,
            name=f"{prefix}.k_cam_relu",
        )
        k_scale = _add_ranked_scalar(
            network,
            trt_module,
            3,
            (cam_head_dim**-0.5) * ((shape.latent_height * shape.latent_width) ** -0.5),
            dtype=np.float32 if both_triton_gdn else dtype,
        )
        k_cam = _set_tensor_name(
            network.add_elementwise(
                k_cam,
                k_scale,
                trt_module.ElementWiseOperation.PROD,
            ).get_output(0),
            f"{prefix}.k_cam_scaled",
        )

    return SanaWmStage1CameraPreamble(
        q=_permute_bnhd_to_bhdn(
            network,
            q_cam,
            shape,
            frontend,
            num_heads=cam_heads,
            head_dim=cam_head_dim,
            trt_module=trt_module,
            name=f"{prefix}.q_cam_bhdn",
        ),
        k=_permute_bnhd_to_bhdn(
            network,
            k_cam,
            shape,
            frontend,
            num_heads=cam_heads,
            head_dim=cam_head_dim,
            trt_module=trt_module,
            name=f"{prefix}.k_cam_bhdn",
        ),
        v=_permute_bnhd_to_bhdn(
            network,
            v_cam,
            shape,
            frontend,
            num_heads=cam_heads,
            head_dim=cam_head_dim,
            trt_module=trt_module,
            name=f"{prefix}.v_cam_bhdn",
        ),
        num_heads=cam_heads,
        head_dim=cam_head_dim,
        q_raw=q_cam_raw,
        k_raw=k_cam_raw,
        v_raw=v_cam_raw,
        q_norm_weight=q_norm_weight_values,
        k_norm_weight=k_norm_weight_values,
    )


def _transpose_bhdn_to_bhnd(network: Any, tensor: Any, trt_module: Any, *, name: str) -> Any:
    transposed = network.add_shuffle(tensor)
    transposed.first_transpose = trt_module.Permutation([0, 1, 3, 2])
    return _set_tensor_name(transposed.get_output(0), name)


def _transpose_bhnd_to_bhdn(network: Any, tensor: Any, trt_module: Any, *, name: str) -> Any:
    transposed = network.add_shuffle(tensor)
    transposed.first_transpose = trt_module.Permutation([0, 1, 3, 2])
    return _set_tensor_name(transposed.get_output(0), name)


def _transpose_last_two_4x4(network: Any, tensor: Any, trt_module: Any, *, name: str) -> Any:
    transposed = network.add_shuffle(tensor)
    transposed.first_transpose = trt_module.Permutation([0, 1, 3, 2])
    return _set_tensor_name(transposed.get_output(0), name)


def _invert_se3_4x4(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    trt_module: Any,
    *,
    dtype: np.dtype,
    name: str,
) -> Any:
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    use_bf16 = _is_bf16_dtype(dtype)
    work_trt_dtype = trt_module.bfloat16 if use_bf16 else trt_module.float32
    work_np_dtype = _bf16_np_dtype() if use_bf16 else np.float32
    tensor_work = _cast_to_dtype(network, tensor, work_trt_dtype)
    transposed = _transpose_last_two_4x4(network, tensor_work, trt_module, name=f"{name}.T")
    r_t = _add_slice(
        network,
        transposed,
        start=(0, 0, 0, 0),
        shape=(shape.batch_size, token_count, 3, 3),
        stride=(1, 1, 1, 1),
        name=f"{name}.R_T",
    )
    t = _add_slice(
        network,
        tensor_work,
        start=(0, 0, 0, 3),
        shape=(shape.batch_size, token_count, 3, 1),
        stride=(1, 1, 1, 1),
        name=f"{name}.t",
    )
    r_t_t = network.add_matrix_multiply(
        r_t,
        trt_module.MatrixOperation.NONE,
        t,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    if use_bf16:
        r_t_t = _cast_to_dtype(network, r_t_t, trt_module.bfloat16)
    neg = _add_constant(
        network,
        trt_module,
        (1, 1, 1, 1),
        np.array([-1.0], dtype=np.float32),
        dtype=work_np_dtype,
    )
    inv_t = network.add_elementwise(
        r_t_t,
        neg,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    if use_bf16:
        inv_t = _cast_to_dtype(network, inv_t, trt_module.bfloat16)
    top = network.add_concatenation([r_t, inv_t])
    top.axis = 3
    bottom_value = np.zeros((shape.batch_size, token_count, 1, 4), dtype=np.float32)
    bottom_value[..., 3] = 1.0
    bottom = _add_constant(
        network,
        trt_module,
        bottom_value.shape,
        bottom_value,
        dtype=work_np_dtype,
    )
    out = network.add_concatenation([top.get_output(0), bottom])
    out.axis = 2
    return _set_tensor_name(out.get_output(0), name)


def _add_clamp_min(
    network: Any,
    tensor: Any,
    value: float,
    *,
    rank: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    floor = _add_ranked_scalar(network, trt_module, rank, value, dtype=dtype)
    return _set_tensor_name(
        network.add_elementwise(
            tensor,
            floor,
            trt_module.ElementWiseOperation.MAX,
        ).get_output(0),
        name,
    )


def _add_clamp_max(
    network: Any,
    tensor: Any,
    value: float,
    *,
    rank: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    ceiling = _add_ranked_scalar(network, trt_module, rank, value, dtype=dtype)
    return _set_tensor_name(
        network.add_elementwise(
            tensor,
            ceiling,
            trt_module.ElementWiseOperation.MIN,
        ).get_output(0),
        name,
    )


def _downscale_to_reference_rms_bhnd(
    network: Any,
    ref: Any,
    transformed: Any,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    """Lower upstream camera UCPE post-transform RMS stabilization."""
    ref_fp32 = ref if dtype == np.float32 else _cast_to_dtype(network, ref, trt_module.float32)
    transformed_fp32 = (
        transformed
        if dtype == np.float32
        else _cast_to_dtype(network, transformed, trt_module.float32)
    )
    ref_sq = network.add_elementwise(
        ref_fp32,
        ref_fp32,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    tr_sq = network.add_elementwise(
        transformed_fp32,
        transformed_fp32,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    ref_mean = network.add_reduce(
        ref_sq,
        trt_module.ReduceOperation.AVG,
        1 << 3,
        True,
    ).get_output(0)
    tr_mean = network.add_reduce(
        tr_sq,
        trt_module.ReduceOperation.AVG,
        1 << 3,
        True,
    ).get_output(0)
    eps = _add_ranked_scalar(network, trt_module, 4, 1.0e-6, dtype=np.float32)
    ref_rms = network.add_unary(
        network.add_elementwise(
            ref_mean,
            eps,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0),
        trt_module.UnaryOperation.SQRT,
    ).get_output(0)
    tr_rms = network.add_unary(
        network.add_elementwise(
            tr_mean,
            eps,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0),
        trt_module.UnaryOperation.SQRT,
    ).get_output(0)
    tr_rms = _add_clamp_min(
        network,
        tr_rms,
        1.0e-6,
        rank=4,
        trt_module=trt_module,
        dtype=np.float32,
        name=f"{name}.tr_rms_clamped",
    )
    scale = network.add_elementwise(
        ref_rms,
        tr_rms,
        trt_module.ElementWiseOperation.DIV,
    ).get_output(0)
    scale = _add_clamp_max(
        network,
        scale,
        1.0,
        rank=4,
        trt_module=trt_module,
        dtype=np.float32,
        name=f"{name}.scale",
    )
    out = network.add_elementwise(
        transformed_fp32,
        scale,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    return _set_tensor_name(out, name)


def _apply_ray_projection_to_bhnd(
    network: Any,
    feats: Any,
    matrix: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    trt_module: Any,
    name: str,
    bf16_projection: bool = False,
) -> Any:
    if channels % 4 != 0:
        raise ValueError(f"{name} UCPE projection channels must divide by 4, got {channels}")
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    if bf16_projection:
        feats = _cast_to_dtype(network, feats, trt_module.bfloat16)
        matrix = _cast_to_dtype(network, matrix, trt_module.bfloat16)
    else:
        feats = _cast_to_dtype(network, feats, trt_module.float32)
        matrix = _cast_to_dtype(network, matrix, trt_module.float32)
    groups = channels // 4
    grouped = network.add_shuffle(feats)
    grouped.reshape_dims = (
        shape.batch_size,
        num_heads,
        token_count,
        groups,
        4,
        1,
    )
    matrix_broadcast = network.add_shuffle(matrix)
    matrix_broadcast.reshape_dims = (
        shape.batch_size,
        1,
        token_count,
        1,
        4,
        4,
    )
    projected = network.add_matrix_multiply(
        matrix_broadcast.get_output(0),
        trt_module.MatrixOperation.NONE,
        grouped.get_output(0),
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    flat = network.add_shuffle(projected)
    flat.reshape_dims = (shape.batch_size, num_heads, token_count, channels)
    return _set_tensor_name(flat.get_output(0), name)


def _apply_ucpe_rope_to_bhnd(
    network: Any,
    hidden: Any,
    cos: Any,
    sin: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    inverse: bool,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    if channels % 2 != 0:
        raise ValueError(f"{name} UCPE RoPE channels must be even, got {channels}")
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    hidden_fp32 = (
        hidden if dtype == np.float32 else _cast_to_dtype(network, hidden, trt_module.float32)
    )
    pairs = network.add_shuffle(hidden_fp32)
    pairs.reshape_dims = (
        shape.batch_size,
        num_heads,
        token_count,
        channels // 2,
        2,
    )
    pair_shape = (shape.batch_size, num_heads, token_count, channels // 2, 1)
    even = _add_slice(
        network,
        pairs.get_output(0),
        start=(0, 0, 0, 0, 0),
        shape=pair_shape,
        stride=(1, 1, 1, 1, 1),
        name=f"{name}.even",
    )
    odd = _add_slice(
        network,
        pairs.get_output(0),
        start=(0, 0, 0, 0, 1),
        shape=pair_shape,
        stride=(1, 1, 1, 1, 1),
        name=f"{name}.odd",
    )
    even_cos = network.add_elementwise(
        even,
        cos,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    odd_sin = network.add_elementwise(
        odd,
        sin,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    even_sin = network.add_elementwise(
        even,
        sin,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    odd_cos = network.add_elementwise(odd, cos, trt_module.ElementWiseOperation.PROD).get_output(0)
    if inverse:
        even_rot = network.add_elementwise(
            even_cos,
            odd_sin,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
        odd_rot = network.add_elementwise(
            odd_cos,
            even_sin,
            trt_module.ElementWiseOperation.SUB,
        ).get_output(0)
    else:
        even_rot = network.add_elementwise(
            even_cos,
            odd_sin,
            trt_module.ElementWiseOperation.SUB,
        ).get_output(0)
        odd_rot = network.add_elementwise(
            even_sin,
            odd_cos,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)

    rotated_pairs = network.add_concatenation([even_rot, odd_rot])
    rotated_pairs.axis = 4
    rotated = network.add_shuffle(rotated_pairs.get_output(0))
    rotated.reshape_dims = (shape.batch_size, num_heads, token_count, channels)
    return _set_tensor_name(rotated.get_output(0), name)


def _apply_ucpe_inverse_rope_high_low_plugin_to_bhnd(
    network: Any,
    hidden: Any,
    matrix: Any,
    cos: Any,
    sin: Any,
    cos_residual: Any,
    sin_residual: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_ucpe_plugin(
        trt_module,
        frames=shape.latent_frames,
        spatial=shape.latent_height * shape.latent_width,
        heads=num_heads,
        head_dim=channels,
        inverse=True,
        double_rope=True,
        rope_only=True,
    )
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    tensor_shape = (1, 1, token_count, channels // 2, 1)
    plugin_inputs = [
        _cast_to_dtype(network, hidden, trt_module.float32),
        _cast_to_dtype(network, matrix, trt_module.float32),
    ]
    for value in (cos, sin, cos_residual, sin_residual):
        flattened = network.add_shuffle(value)
        flattened.reshape_dims = tensor_shape
        plugin_inputs.append(flattened.get_output(0))
    layer = add_plugin(plugin_inputs, plugin)
    out = layer.get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    return _set_tensor_name(out, name)


def _apply_ucpe_block_diagonal_to_bhnd(
    network: Any,
    feats: Any,
    matrix: Any,
    cos: Any,
    sin: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    head_dim: int,
    inverse_rope: bool,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
    keep_fp32_output: bool = False,
    rms_downscale: bool = False,
    rope_cos_residual: Any | None = None,
    rope_sin_residual: Any | None = None,
) -> Any:
    token_count = shape.latent_frames * shape.latent_height * shape.latent_width
    geom_dim = head_dim // 2
    rope_dim = head_dim - geom_dim
    add_plugin = network.add_plugin_v2
    split_inverse_rope = (
        inverse_rope and rope_cos_residual is not None and rope_sin_residual is not None
    )
    if not split_inverse_rope:
        plugin = _create_sana_wm_ucpe_plugin(
            trt_module,
            frames=shape.latent_frames,
            spatial=shape.latent_height * shape.latent_width,
            heads=num_heads,
            head_dim=head_dim,
            inverse=inverse_rope,
            tree_reduce=not keep_fp32_output,
            downscale=rms_downscale,
            double_rope=(
                inverse_rope and rope_cos_residual is not None and rope_sin_residual is not None
            ),
        )
        if plugin is not None:
            feats_fp32 = _cast_to_dtype(network, feats, trt_module.float32)
            matrix_fp32 = _cast_to_dtype(network, matrix, trt_module.float32)
            cos_flat = network.add_shuffle(cos)
            cos_flat.reshape_dims = (1, 1, token_count, rope_dim // 2, 1)
            sin_flat = network.add_shuffle(sin)
            sin_flat.reshape_dims = (1, 1, token_count, rope_dim // 2, 1)
            plugin_inputs = [
                feats_fp32,
                matrix_fp32,
                cos_flat.get_output(0),
                sin_flat.get_output(0),
            ]
            if rope_cos_residual is not None and rope_sin_residual is not None:
                cos_residual_flat = network.add_shuffle(rope_cos_residual)
                cos_residual_flat.reshape_dims = (
                    1,
                    1,
                    token_count,
                    rope_dim // 2,
                    1,
                )
                sin_residual_flat = network.add_shuffle(rope_sin_residual)
                sin_residual_flat.reshape_dims = (
                    1,
                    1,
                    token_count,
                    rope_dim // 2,
                    1,
                )
                plugin_inputs.extend(
                    [
                        cos_residual_flat.get_output(0),
                        sin_residual_flat.get_output(0),
                    ]
                )
            layer = add_plugin(plugin_inputs, plugin)
            out = layer.get_output(0)
            if dtype != np.float32 and not keep_fp32_output:
                out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
            return _set_tensor_name(out, name)
    geom = _add_slice(
        network,
        feats,
        start=(0, 0, 0, 0),
        shape=(shape.batch_size, num_heads, token_count, geom_dim),
        stride=(1, 1, 1, 1),
        name=f"{name}.geom",
    )
    rope = _add_slice(
        network,
        feats,
        start=(0, 0, 0, geom_dim),
        shape=(shape.batch_size, num_heads, token_count, rope_dim),
        stride=(1, 1, 1, 1),
        name=f"{name}.rope",
    )
    geom_out = _apply_ray_projection_to_bhnd(
        network,
        geom,
        matrix,
        shape,
        num_heads=num_heads,
        channels=geom_dim,
        trt_module=trt_module,
        name=f"{name}.ray_projected",
        bf16_projection=inverse_rope and dtype != np.float32,
    )
    rope_out = None
    if split_inverse_rope:
        rope_out = _apply_ucpe_inverse_rope_high_low_plugin_to_bhnd(
            network,
            rope,
            matrix,
            cos,
            sin,
            rope_cos_residual,
            rope_sin_residual,
            shape,
            num_heads=num_heads,
            channels=rope_dim,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.rope_projected",
        )
    if rope_out is None:
        rope_out = _apply_ucpe_rope_to_bhnd(
            network,
            rope,
            cos,
            sin,
            shape,
            num_heads=num_heads,
            channels=rope_dim,
            inverse=inverse_rope,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.rope_projected",
        )
    if inverse_rope and dtype != np.float32:
        geom_out = _cast_to_dtype(network, geom_out, _trt_dtype_for_np(trt_module, dtype))
        rope_out = _cast_to_dtype(network, rope_out, _trt_dtype_for_np(trt_module, dtype))
    concat = network.add_concatenation([geom_out, rope_out])
    concat.axis = 3
    out = concat.get_output(0)
    if dtype != np.float32 and not keep_fp32_output:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    if rms_downscale:
        out = _downscale_to_reference_rms_bhnd(
            network,
            feats,
            out,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.rms_downscaled",
        )
    return _set_tensor_name(out, name)


def _squared_norm_bhnd(
    network: Any,
    tensor: Any,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    tensor_fp32 = (
        tensor if dtype == np.float32 else _cast_to_dtype(network, tensor, trt_module.float32)
    )
    squared = network.add_elementwise(
        tensor_fp32,
        tensor_fp32,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    summed = network.add_reduce(
        squared,
        trt_module.ReduceOperation.SUM,
        1 << 3,
        True,
    ).get_output(0)
    return _set_tensor_name(summed, name)


def _discount_camera_beta_by_ucpe_inflation(
    network: Any,
    beta: Any,
    pre_ucpe_k: Any,
    post_ucpe_k: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    pre = _add_clamp_min(
        network,
        _squared_norm_bhnd(
            network,
            pre_ucpe_k,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.pre_k_norm_sq",
        ),
        1.0e-12,
        rank=4,
        trt_module=trt_module,
        dtype=np.float32,
        name=f"{name}.pre_k_norm_sq_clamped",
    )
    post = _add_clamp_min(
        network,
        _squared_norm_bhnd(
            network,
            post_ucpe_k,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.post_k_norm_sq",
        ),
        1.0e-12,
        rank=4,
        trt_module=trt_module,
        dtype=np.float32,
        name=f"{name}.post_k_norm_sq_clamped",
    )
    ratio = network.add_elementwise(
        post,
        pre,
        trt_module.ElementWiseOperation.DIV,
    ).get_output(0)
    spatial_tokens = shape.latent_height * shape.latent_width
    inflation = network.add_shuffle(ratio)
    inflation.reshape_dims = (
        shape.batch_size,
        num_heads,
        shape.latent_frames,
        spatial_tokens,
    )
    frame_inflation = network.add_reduce(
        inflation.get_output(0),
        trt_module.ReduceOperation.AVG,
        1 << 3,
        False,
    ).get_output(0)
    frame_inflation = _add_clamp_min(
        network,
        frame_inflation,
        1.0,
        rank=3,
        trt_module=trt_module,
        dtype=np.float32,
        name=f"{name}.frame_inflation_clamped",
    )
    frame_inflation_4d = network.add_shuffle(frame_inflation)
    frame_inflation_4d.reshape_dims = (
        shape.batch_size,
        num_heads,
        shape.latent_frames,
        1,
    )
    beta_fp32 = beta if dtype == np.float32 else _cast_to_dtype(network, beta, trt_module.float32)
    return _set_tensor_name(
        network.add_elementwise(
            beta_fp32,
            frame_inflation_4d.get_output(0),
            trt_module.ElementWiseOperation.DIV,
        ).get_output(0),
        name,
    )


def _discount_camera_beta_by_inflation_sq(
    network: Any,
    beta: Any,
    inflation_sq: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    inflation = network.add_shuffle(inflation_sq)
    inflation.reshape_dims = (
        shape.batch_size,
        num_heads,
        shape.latent_frames,
        spatial_tokens,
    )
    frame_inflation = network.add_reduce(
        inflation.get_output(0),
        trt_module.ReduceOperation.AVG,
        1 << 3,
        False,
    ).get_output(0)
    frame_inflation = _add_clamp_min(
        network,
        frame_inflation,
        1.0,
        rank=3,
        trt_module=trt_module,
        dtype=np.float32,
        name=f"{name}.frame_inflation_clamped",
    )
    frame_inflation_4d = network.add_shuffle(frame_inflation)
    frame_inflation_4d.reshape_dims = (
        shape.batch_size,
        num_heads,
        shape.latent_frames,
        1,
    )
    beta_fp32 = beta if dtype == np.float32 else _cast_to_dtype(network, beta, trt_module.float32)
    return _set_tensor_name(
        network.add_elementwise(
            beta_fp32,
            frame_inflation_4d.get_output(0),
            trt_module.ElementWiseOperation.DIV,
        ).get_output(0),
        name,
    )


def _discount_camera_beta_with_plugin(
    network: Any,
    beta: Any,
    inflation_sq: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    trt_module: Any,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_camera_beta_discount_plugin(
        trt_module,
        frames=shape.latent_frames,
        spatial=shape.latent_height * shape.latent_width,
        heads=num_heads,
    )
    layer = add_plugin([beta, inflation_sq], plugin)
    return _set_tensor_name(layer.get_output(0), name)


def _lower_sana_wm_camera_prep_plugin(
    network: Any,
    camera: SanaWmStage1CameraPreamble,
    preamble: SanaWmStage1BlockPreamble,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    raw_config: dict,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
    discount_beta: bool,
    use_torch_plugin: bool | None = None,
) -> SanaWmStage1CameraUcpe | None:
    if not discount_beta or not _is_bf16_dtype(dtype):
        return None
    if _stage1_camctrl_type(raw_config) != "BidirectionalGDNUCPESinglePathLiteLABothTriton":
        return None
    if (
        camera.q_raw is None
        or camera.k_raw is None
        or camera.v_raw is None
        or camera.q_norm_weight is None
        or camera.k_norm_weight is None
    ):
        return None
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_cam_prep_plugin(
        trt_module,
        frames=shape.latent_frames,
        spatial=shape.latent_height * shape.latent_width,
        heads=camera.num_heads,
        head_dim=camera.head_dim,
        norm_eps=_stage1_norm_eps(raw_config),
        q_norm_weight=camera.q_norm_weight,
        k_norm_weight=camera.k_norm_weight,
        use_torch_plugin=use_torch_plugin,
    )
    p_t = _cast_to_dtype(
        network,
        _transpose_last_two_4x4(network, inputs["raymats"], trt_module, name=f"{name}.P_T"),
        trt_module.float32,
    )
    p_inv = _cast_to_dtype(
        network,
        _invert_se3_4x4(
            network,
            inputs["raymats"],
            shape,
            trt_module,
            dtype=dtype,
            name=f"{name}.P_inv",
        ),
        trt_module.float32,
    )
    cos, sin = _add_ucpe_cam_rope_constants(
        network,
        shape,
        camera.head_dim,
        raw_config,
        trt_module=trt_module,
    )
    channels = camera.num_heads * camera.head_dim
    q_norm_weight = _add_constant(
        network,
        trt_module,
        (channels,),
        np.ascontiguousarray(camera.q_norm_weight.reshape(channels), dtype=np.float32),
        dtype=np.float32,
    )
    k_norm_weight = _add_constant(
        network,
        trt_module,
        (channels,),
        np.ascontiguousarray(camera.k_norm_weight.reshape(channels), dtype=np.float32),
        dtype=np.float32,
    )
    layer = add_plugin(
        [
            camera.q_raw,
            camera.k_raw,
            camera.v_raw,
            p_t,
            p_inv,
            cos,
            sin,
            q_norm_weight,
            k_norm_weight,
        ],
        plugin,
    )
    q = layer.get_output(0)
    k = layer.get_output(1)
    v = layer.get_output(2)
    if dtype != np.float32:
        trt_dtype = _trt_dtype_for_np(trt_module, dtype)
        q = _cast_to_dtype(network, q, trt_dtype)
        k = _cast_to_dtype(network, k, trt_dtype)
        v = _cast_to_dtype(network, v, trt_dtype)
    beta = _discount_camera_beta_by_inflation_sq(
        network,
        preamble.beta,
        layer.get_output(3),
        shape,
        num_heads=camera.num_heads,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.beta_discounted",
    )
    return SanaWmStage1CameraUcpe(
        q_rot=_set_tensor_name(q, f"{name}.q"),
        k_rot=_set_tensor_name(k, f"{name}.k"),
        v=_set_tensor_name(v, f"{name}.v"),
        beta=beta,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inflation_sq=_set_tensor_name(layer.get_output(3), f"{name}.inflation_sq"),
    )


def lower_sana_wm_stage1_camera_ucpe(
    network: Any,
    camera: SanaWmStage1CameraPreamble,
    preamble: SanaWmStage1BlockPreamble,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    raw_config: dict,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str = "blocks.0.attn.cam_ucpe",
    discount_beta: bool = True,
    stabilize_transforms: bool = True,
) -> SanaWmStage1CameraUcpe:
    """Apply upstream UCPE Q/K/V block-diagonal transforms for the camera branch."""
    if camera.num_heads != preamble.num_heads:
        raise NotImplementedError(
            "SANA-WM native Stage-1 builder currently supports camera-head "
            "beta discounting only when cam_attn_compress=1"
        )
    camera_prep_plugin_mode = _sana_wm_camera_prep_plugin_mode()
    torch_fused = None
    cuda_fused = None
    hybrid_eligible = (
        discount_beta
        and _is_bf16_dtype(dtype)
        and _stage1_camctrl_type(raw_config) == "BidirectionalGDNUCPESinglePathLiteLABothTriton"
        and camera.q_raw is not None
        and camera.k_raw is not None
        and camera.v_raw is not None
        and camera.q_norm_weight is not None
        and camera.k_norm_weight is not None
    )
    if camera_prep_plugin_mode == "hybrid" and hybrid_eligible:
        torch_fused = _lower_sana_wm_camera_prep_plugin(
            network,
            camera,
            preamble,
            inputs,
            shape,
            raw_config,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.torch_prep",
            discount_beta=discount_beta,
            use_torch_plugin=True,
        )
        cuda_fused = _lower_sana_wm_camera_prep_plugin(
            network,
            camera,
            preamble,
            inputs,
            shape,
            raw_config,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.cuda_prep",
            discount_beta=discount_beta,
            use_torch_plugin=False,
        )
        fused = None
    else:
        fused = _lower_sana_wm_camera_prep_plugin(
            network,
            camera,
            preamble,
            inputs,
            shape,
            raw_config,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.prep_plugin" if camera_prep_plugin_mode == "v" else name,
            discount_beta=discount_beta,
        )
    if fused is not None and camera_prep_plugin_mode != "v":
        return fused

    camera_q = camera.q
    camera_k = camera.k
    q_bhnd = _transpose_bhdn_to_bhnd(network, camera_q, trt_module, name=f"{name}.q_bhnd")
    k_bhnd = _transpose_bhdn_to_bhnd(network, camera_k, trt_module, name=f"{name}.k_bhnd")
    v_bhnd = _transpose_bhdn_to_bhnd(network, camera.v, trt_module, name=f"{name}.v_bhnd")
    p_t = _transpose_last_two_4x4(network, inputs["raymats"], trt_module, name=f"{name}.P_T")
    p_inv = _invert_se3_4x4(
        network,
        inputs["raymats"],
        shape,
        trt_module,
        dtype=dtype,
        name=f"{name}.P_inv",
    )
    cos, sin = _add_ucpe_cam_rope_constants(
        network,
        shape,
        camera.head_dim,
        raw_config,
        trt_module=trt_module,
    )
    keep_inflation_k_fp32 = discount_beta and not stabilize_transforms

    q_trans_raw = _apply_ucpe_block_diagonal_to_bhnd(
        network,
        q_bhnd,
        p_t,
        cos,
        sin,
        shape,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inverse_rope=False,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.q",
        rms_downscale=stabilize_transforms,
    )
    k_trans_for_inflation = _apply_ucpe_block_diagonal_to_bhnd(
        network,
        k_bhnd,
        p_inv,
        cos,
        sin,
        shape,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inverse_rope=False,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.k",
        keep_fp32_output=keep_inflation_k_fp32,
        rms_downscale=stabilize_transforms,
    )
    k_trans_raw = k_trans_for_inflation
    if keep_inflation_k_fp32 and dtype != np.float32:
        k_trans_raw = _cast_to_dtype(
            network,
            k_trans_for_inflation,
            _trt_dtype_for_np(trt_module, dtype),
        )
    v_trans_raw = _apply_ucpe_block_diagonal_to_bhnd(
        network,
        v_bhnd,
        p_inv,
        cos,
        sin,
        shape,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inverse_rope=False,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.v",
        rms_downscale=stabilize_transforms,
    )
    beta = None
    if discount_beta:
        beta = _discount_camera_beta_by_ucpe_inflation(
            network,
            preamble.beta,
            k_bhnd,
            k_trans_for_inflation,
            shape,
            num_heads=camera.num_heads,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.beta_discounted",
        )
    q_trans = q_trans_raw
    k_trans = k_trans_raw
    v_trans = v_trans_raw
    v_bhdn = (
        fused.v
        if fused is not None and camera_prep_plugin_mode == "v"
        else _transpose_bhnd_to_bhdn(network, v_trans, trt_module, name=f"{name}.v_bhdn")
    )
    q_bhdn = _transpose_bhnd_to_bhdn(network, q_trans, trt_module, name=f"{name}.q_bhdn")
    k_bhdn = _transpose_bhnd_to_bhdn(network, k_trans, trt_module, name=f"{name}.k_bhdn")
    if camera_prep_plugin_mode == "hybrid" and hybrid_eligible:
        if torch_fused is None or cuda_fused is None:
            raise RuntimeError("SANA-WM hybrid camera prep requires both Torch and CUDA plugins")
        q_bhdn = torch_fused.q_rot
        k_bhdn = torch_fused.k_rot
        v_bhdn = cuda_fused.v
        if torch_fused.inflation_sq is None:
            raise RuntimeError("SANA-WM Torch camera prep did not return inflation_sq")
        plugin_beta = _discount_camera_beta_with_plugin(
            network,
            preamble.beta,
            torch_fused.inflation_sq,
            shape,
            num_heads=camera.num_heads,
            trt_module=trt_module,
            name=f"{name}.beta_discounted_plugin",
        )
        if plugin_beta is None:
            raise RuntimeError("SANA-WM camera beta discount plugin did not produce an output")
        beta = plugin_beta
    return SanaWmStage1CameraUcpe(
        q_rot=q_bhdn,
        k_rot=k_bhdn,
        v=v_bhdn,
        beta=beta,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inflation_sq=(
            torch_fused.inflation_sq
            if camera_prep_plugin_mode == "hybrid" and hybrid_eligible
            else None
        ),
    )


def _ensure_fp32(network: Any, tensor: Any, trt_module: Any, dtype: Any) -> Any:
    if dtype == np.float32:
        return tensor
    return _cast_to_dtype(network, tensor, trt_module.float32)


def _reshape_bhdn_to_bhtds(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    if frontend.token_count != shape.latent_frames * spatial_tokens:
        raise ValueError(
            "SANA-WM GDN token count does not match F*H*W: "
            f"{frontend.token_count} vs {shape.latent_frames * spatial_tokens}"
        )
    five_d = network.add_shuffle(tensor)
    five_d.reshape_dims = (
        shape.batch_size,
        num_heads,
        head_dim,
        shape.latent_frames,
        spatial_tokens,
    )
    framed = network.add_shuffle(five_d.get_output(0))
    framed.first_transpose = trt_module.Permutation([0, 1, 3, 2, 4])
    return framed.get_output(0)


def _slice_sana_wm_gdn_frame(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    frame_index: int,
    num_heads: int,
    head_dim: int,
    name: str,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    sliced = _add_slice(
        network,
        tensor,
        start=(0, 0, frame_index, 0, 0),
        shape=(shape.batch_size, num_heads, 1, head_dim, spatial_tokens),
        stride=(1, 1, 1, 1, 1),
        name=f"{name}.slice",
    )
    squeezed = network.add_shuffle(sliced)
    squeezed.reshape_dims = (
        shape.batch_size,
        num_heads,
        head_dim,
        spatial_tokens,
    )
    return _set_tensor_name(squeezed.get_output(0), name)


def _slice_sana_wm_gdn_beta(
    network: Any,
    beta: Any,
    shape: SanaWmStage1Shape,
    *,
    frame_index: int,
    num_heads: int,
    name: str,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    return _add_slice(
        network,
        beta,
        start=(0, 0, frame_index, 0),
        shape=(shape.batch_size, num_heads, 1, spatial_tokens),
        stride=(1, 1, 1, 1),
        name=name,
    )


def _slice_sana_wm_gdn_decay(
    network: Any,
    decay: Any,
    shape: SanaWmStage1Shape,
    *,
    frame_index: int,
    num_heads: int,
    name: str,
) -> Any:
    sliced = _add_slice(
        network,
        decay,
        start=(0, 0, frame_index),
        shape=(shape.batch_size, num_heads, 1),
        stride=(1, 1, 1),
        name=f"{name}.slice",
    )
    decay_4d = network.add_shuffle(sliced)
    decay_4d.reshape_dims = (shape.batch_size, num_heads, 1, 1)
    return _set_tensor_name(decay_4d.get_output(0), name)


def _transpose_bhds(network: Any, tensor: Any, trt_module: Any) -> Any:
    transposed = network.add_shuffle(tensor)
    transposed.first_transpose = trt_module.Permutation([0, 1, 3, 2])
    return transposed.get_output(0)


def _add_bhds_loop_concat_output(
    network: Any,
    loop: Any,
    count: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    trt_module: Any,
    name: str,
    reverse_output: bool = False,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    output_kind = (
        trt_module.LoopOutput.REVERSE if reverse_output else trt_module.LoopOutput.CONCATENATE
    )
    output = loop.add_loop_output(tensor, output_kind, 3)
    output.set_input(1, count)
    loop_tensor = output.get_output(0)
    flat = network.add_shuffle(loop_tensor)
    flat_shape = _add_int32_shape(
        network,
        trt_module,
        (
            shape.batch_size,
            num_heads,
            channels,
            shape.latent_frames * spatial_tokens,
        ),
    )
    flat.set_input(1, flat_shape)
    flattened = _set_tensor_name(flat.get_output(0), name)
    return flattened


def _reshape_bhts_to_bht1s(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    name: str,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    reshaped = network.add_shuffle(tensor)
    reshaped.reshape_dims = (
        shape.batch_size,
        num_heads,
        shape.latent_frames,
        1,
        spatial_tokens,
    )
    return _set_tensor_name(reshaped.get_output(0), name)


def _reshape_bht_to_bht11(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    name: str,
) -> Any:
    reshaped = network.add_shuffle(tensor)
    reshaped.reshape_dims = (
        shape.batch_size,
        num_heads,
        shape.latent_frames,
        1,
        1,
    )
    return _set_tensor_name(reshaped.get_output(0), name)


def _zero_like(
    network: Any,
    tensor: Any,
    *,
    rank: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    zero = _add_ranked_scalar(network, trt_module, rank, 0.0, dtype=dtype)
    return network.add_elementwise(
        tensor,
        zero,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)


def _fill_like(
    network: Any,
    tensor: Any,
    value: float,
    *,
    rank: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    zeroed = _zero_like(
        network,
        tensor,
        rank=rank,
        trt_module=trt_module,
        dtype=dtype,
    )
    fill = _add_ranked_scalar(network, trt_module, rank, value, dtype=dtype)
    return network.add_elementwise(
        zeroed,
        fill,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)


def _slice_bhdn_frame(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    frame_index: int,
    name: str,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    return _add_slice(
        network,
        tensor,
        start=(0, 0, 0, frame_index * spatial_tokens),
        shape=(shape.batch_size, num_heads, channels, spatial_tokens),
        stride=(1, 1, 1, 1),
        name=name,
    )


def _reverse_bhdn_frames(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    frames = [
        _slice_bhdn_frame(
            network,
            tensor,
            shape,
            num_heads=num_heads,
            channels=channels,
            frame_index=frame_index,
            name=f"{name}.frame{frame_index}",
        )
        for frame_index in range(shape.latent_frames - 1, -1, -1)
    ]
    del trt_module, dtype
    concat = network.add_concatenation(frames)
    concat.axis = 3
    return _set_tensor_name(concat.get_output(0), name)


def _flip_shift_bhdn_frames(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    channels: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    pad_source = _slice_bhdn_frame(
        network,
        tensor,
        shape,
        num_heads=num_heads,
        channels=channels,
        frame_index=0,
        name=f"{name}.pad_source",
    )
    frames = [
        _zero_like(
            network,
            pad_source,
            rank=4,
            trt_module=trt_module,
            dtype=dtype,
        )
    ]
    frames.extend(
        _slice_bhdn_frame(
            network,
            tensor,
            shape,
            num_heads=num_heads,
            channels=channels,
            frame_index=frame_index,
            name=f"{name}.frame{frame_index}",
        )
        for frame_index in range(shape.latent_frames - 1, 0, -1)
    )
    concat = network.add_concatenation(frames)
    concat.axis = 3
    return _set_tensor_name(concat.get_output(0), name)


def _slice_bhts_frame(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    frame_index: int,
    name: str,
) -> Any:
    spatial_tokens = shape.latent_height * shape.latent_width
    return _add_slice(
        network,
        tensor,
        start=(0, 0, frame_index, 0),
        shape=(shape.batch_size, num_heads, 1, spatial_tokens),
        stride=(1, 1, 1, 1),
        name=name,
    )


def _flip_shift_bhts_frames(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    pad_source = _slice_bhts_frame(
        network,
        tensor,
        shape,
        num_heads=num_heads,
        frame_index=0,
        name=f"{name}.pad_source",
    )
    frames = [
        _zero_like(
            network,
            pad_source,
            rank=4,
            trt_module=trt_module,
            dtype=dtype,
        )
    ]
    frames.extend(
        _slice_bhts_frame(
            network,
            tensor,
            shape,
            num_heads=num_heads,
            frame_index=frame_index,
            name=f"{name}.frame{frame_index}",
        )
        for frame_index in range(shape.latent_frames - 1, 0, -1)
    )
    concat = network.add_concatenation(frames)
    concat.axis = 2
    return _set_tensor_name(concat.get_output(0), name)


def _slice_bht_frame(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    frame_index: int,
    name: str,
) -> Any:
    return _add_slice(
        network,
        tensor,
        start=(0, 0, frame_index),
        shape=(shape.batch_size, num_heads, 1),
        stride=(1, 1, 1),
        name=name,
    )


def _flip_shift_bht_frames(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    trt_module: Any,
    name: str,
) -> Any:
    pad_source = _slice_bht_frame(
        network,
        tensor,
        shape,
        num_heads=num_heads,
        frame_index=0,
        name=f"{name}.pad_source",
    )
    frames = [
        _fill_like(
            network,
            pad_source,
            1.0,
            rank=3,
            trt_module=trt_module,
            dtype=np.float32,
        )
    ]
    frames.extend(
        _slice_bht_frame(
            network,
            tensor,
            shape,
            num_heads=num_heads,
            frame_index=frame_index,
            name=f"{name}.frame{frame_index}",
        )
        for frame_index in range(shape.latent_frames - 1, 0, -1)
    )
    concat = network.add_concatenation(frames)
    concat.axis = 2
    return _set_tensor_name(concat.get_output(0), name)


def _lower_sana_wm_stage1_gdn_forward_components_plugin(
    network: Any,
    framed: dict[str, Any],
    beta: Any,
    decay: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    name: str,
    reverse_output: bool = False,
) -> SanaWmStage1GdnComponents | None:
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_gdn_plugin(
        trt_module,
        mode=0,
        reverse_output=reverse_output,
    )
    beta_5d = _reshape_bhts_to_bht1s(
        network,
        beta,
        shape,
        num_heads=num_heads,
        name=f"{name}.beta_plugin_input",
    )
    decay_5d = _reshape_bht_to_bht11(
        network,
        decay,
        shape,
        num_heads=num_heads,
        name=f"{name}.decay_plugin_input",
    )
    layer = add_plugin(
        [
            framed["q"],
            framed["k"],
            framed["v"],
            framed["q_rot"],
            framed["k_rot"],
            beta_5d,
            decay_5d,
        ],
        plugin,
    )
    return SanaWmStage1GdnComponents(
        num=_set_tensor_name(layer.get_output(0), f"{name}.num"),
        den=_set_tensor_name(layer.get_output(1), f"{name}.den"),
    )


def _lower_sana_wm_stage1_bidirectional_gdn_core_plugin(
    network: Any,
    preamble: SanaWmStage1BlockPreamble,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    raw_config: dict,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> SanaWmStage1GdnCore | None:
    add_plugin = network.add_plugin_v2
    q = _ensure_fp32(network, preamble.q, trt_module, dtype)
    k = _ensure_fp32(network, preamble.k, trt_module, dtype)
    v = _ensure_fp32(network, preamble.v, trt_module, dtype)
    q_rot = _ensure_fp32(network, preamble.q_rot, trt_module, dtype)
    k_rot = _ensure_fp32(network, preamble.k_rot, trt_module, dtype)
    beta = _ensure_fp32(network, preamble.beta, trt_module, dtype)
    decay = _ensure_fp32(network, preamble.decay, trt_module, dtype)
    framed = {
        "q": _reshape_bhdn_to_bhtds(
            network,
            q,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "k": _reshape_bhdn_to_bhtds(
            network,
            k,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "v": _reshape_bhdn_to_bhtds(
            network,
            v,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "q_rot": _reshape_bhdn_to_bhtds(
            network,
            q_rot,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "k_rot": _reshape_bhdn_to_bhtds(
            network,
            k_rot,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
    }
    plugin = _create_sana_wm_gdn_plugin(
        trt_module,
        mode=2,
        reverse_output=False,
        eps=_stage1_attention_eps(raw_config),
    )
    beta_5d = _reshape_bhts_to_bht1s(
        network,
        beta,
        shape,
        num_heads=preamble.num_heads,
        name=f"{name}.beta_plugin_input",
    )
    decay_5d = _reshape_bht_to_bht11(
        network,
        decay,
        shape,
        num_heads=preamble.num_heads,
        name=f"{name}.decay_plugin_input",
    )
    layer = add_plugin(
        [
            framed["q"],
            framed["k"],
            framed["v"],
            framed["q_rot"],
            framed["k_rot"],
            beta_5d,
            decay_5d,
        ],
        plugin,
    )
    out = layer.get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    tokens = network.add_shuffle(out)
    tokens.first_transpose = trt_module.Permutation([0, 3, 1, 2])
    tokens.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        preamble.num_heads * preamble.head_dim,
    )
    token_output = _set_tensor_name(tokens.get_output(0), f"{name}.tokens")
    return SanaWmStage1GdnCore(tokens=token_output, num=out, den=out)


def _lower_sana_wm_stage1_bidirectional_gdn_core_raw_plugin(
    network: Any,
    preamble: SanaWmStage1BlockPreamble,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> SanaWmStage1GdnCore | None:
    """Lower the PR-379 fused main GDN boundary from raw Q/K/V tensors."""
    add_plugin = network.add_plugin_v2
    if (
        add_plugin is None
        or preamble.q_raw is None
        or preamble.k_conv is None
        or preamble.v_raw is None
    ):
        return None
    plugin = _create_sana_wm_gdn_plugin(
        trt_module,
        mode=3,
        reverse_output=False,
        eps=_stage1_attention_eps(raw_config),
        frames=shape.latent_frames,
        head_dim=preamble.head_dim,
        norm_eps=_stage1_norm_eps(raw_config),
    )
    prefix = f"blocks.{block_index}.attn"
    q_raw = _ensure_fp32(network, preamble.q_raw, trt_module, dtype)
    k_raw = _ensure_fp32(network, preamble.k_conv, trt_module, dtype)
    v_raw = _ensure_fp32(network, preamble.v_raw, trt_module, dtype)
    q_norm_weight = _add_constant(
        network,
        trt_module,
        (preamble.num_heads * preamble.head_dim,),
        (
            _round_float32_to_bf16_values(_required_weight(weights, f"{prefix}.q_norm.weight"))
            if _is_bf16_dtype(dtype)
            else np.ascontiguousarray(
                np.asarray(_required_weight(weights, f"{prefix}.q_norm.weight"), dtype=np.float32)
            )
        ),
        dtype=np.float32,
    )
    k_norm_weight = _add_constant(
        network,
        trt_module,
        (preamble.num_heads * preamble.head_dim,),
        (
            _round_float32_to_bf16_values(_required_weight(weights, f"{prefix}.k_norm.weight"))
            if _is_bf16_dtype(dtype)
            else np.ascontiguousarray(
                np.asarray(_required_weight(weights, f"{prefix}.k_norm.weight"), dtype=np.float32)
            )
        ),
        dtype=np.float32,
    )
    rope_cos, rope_sin = _add_wan_rope_constants(
        network,
        shape,
        preamble.head_dim,
        raw_config,
        trt_module=trt_module,
    )
    beta = _ensure_fp32(network, preamble.beta, trt_module, dtype)
    decay = _ensure_fp32(network, preamble.decay, trt_module, dtype)
    layer = add_plugin(
        [
            q_raw,
            k_raw,
            v_raw,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            beta,
            decay,
        ],
        plugin,
    )
    out = layer.get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    token_output = _set_tensor_name(out, f"{name}.tokens")
    return SanaWmStage1GdnCore(tokens=token_output, num=token_output, den=token_output)


def _lower_sana_wm_stage1_camera_single_path_forward_plugin(
    network: Any,
    framed: dict[str, Any],
    beta: Any,
    decay: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    name: str,
    reverse_output: bool = False,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_gdn_plugin(
        trt_module,
        mode=1,
        reverse_output=reverse_output,
    )
    beta_5d = _reshape_bhts_to_bht1s(
        network,
        beta,
        shape,
        num_heads=num_heads,
        name=f"{name}.beta_plugin_input",
    )
    decay_5d = _reshape_bht_to_bht11(
        network,
        decay,
        shape,
        num_heads=num_heads,
        name=f"{name}.decay_plugin_input",
    )
    layer = add_plugin(
        [
            framed["q_rot"],
            framed["k_rot"],
            framed["v"],
            beta_5d,
            decay_5d,
        ],
        plugin,
    )
    return _set_tensor_name(layer.get_output(0), name)


def _lower_sana_wm_stage1_camera_single_path_combined_plugin(
    network: Any,
    camera: SanaWmStage1CameraUcpe,
    decay: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any | None:
    add_plugin = network.add_plugin_v2
    plugin = _create_sana_wm_gdn_plugin(
        trt_module,
        mode=4,
        reverse_output=False,
    )
    q_rot = _ensure_fp32(network, camera.q_rot, trt_module, dtype)
    k_rot = _ensure_fp32(network, camera.k_rot, trt_module, dtype)
    v = _ensure_fp32(network, camera.v, trt_module, dtype)
    framed = {
        "q_rot": _reshape_bhdn_to_bhtds(
            network,
            q_rot,
            shape,
            frontend,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
        ),
        "k_rot": _reshape_bhdn_to_bhtds(
            network,
            k_rot,
            shape,
            frontend,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
        ),
        "v": _reshape_bhdn_to_bhtds(
            network,
            v,
            shape,
            frontend,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
        ),
    }
    beta_5d = _reshape_bhts_to_bht1s(
        network,
        camera.beta,
        shape,
        num_heads=camera.num_heads,
        name=f"{name}.beta_plugin_input",
    )
    decay_5d = _reshape_bht_to_bht11(
        network,
        decay,
        shape,
        num_heads=camera.num_heads,
        name=f"{name}.decay_plugin_input",
    )
    layer = add_plugin(
        [
            framed["q_rot"],
            framed["k_rot"],
            framed["v"],
            beta_5d,
            decay_5d,
        ],
        plugin,
    )
    return _set_tensor_name(layer.get_output(0), name)


def _lower_sana_wm_stage1_gdn_forward_components_loop(
    network: Any,
    framed: dict[str, Any],
    beta: Any,
    decay: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    name: str,
    reverse_output: bool = False,
) -> SanaWmStage1GdnComponents:
    loop, count = _add_trt_count_loop(network, trt_module, shape.latent_frames)
    beta_5d = _reshape_bhts_to_bht1s(
        network,
        beta,
        shape,
        num_heads=num_heads,
        name=f"{name}.beta_loop_input",
    )
    decay_5d = _reshape_bht_to_bht11(
        network,
        decay,
        shape,
        num_heads=num_heads,
        name=f"{name}.decay_loop_input",
    )
    qt = _set_tensor_name(
        loop.add_iterator(framed["q"], 2, False).get_output(0),
        f"{name}.iter.q",
    )
    kt = _set_tensor_name(
        loop.add_iterator(framed["k"], 2, False).get_output(0),
        f"{name}.iter.k",
    )
    vt = _set_tensor_name(
        loop.add_iterator(framed["v"], 2, False).get_output(0),
        f"{name}.iter.v",
    )
    qrt = _set_tensor_name(
        loop.add_iterator(framed["q_rot"], 2, False).get_output(0),
        f"{name}.iter.q_rot",
    )
    krt = _set_tensor_name(
        loop.add_iterator(framed["k_rot"], 2, False).get_output(0),
        f"{name}.iter.k_rot",
    )
    bt = _set_tensor_name(
        loop.add_iterator(beta_5d, 2, False).get_output(0),
        f"{name}.iter.beta",
    )
    gt = _set_tensor_name(
        loop.add_iterator(decay_5d, 2, False).get_output(0),
        f"{name}.iter.decay",
    )

    state_kv_init = _add_constant(
        network,
        trt_module,
        (shape.batch_size, num_heads, head_dim, head_dim),
        np.zeros((shape.batch_size, num_heads, head_dim, head_dim), dtype=np.float32),
        dtype=np.float32,
    )
    state_z_init = _add_constant(
        network,
        trt_module,
        (shape.batch_size, num_heads, head_dim, 1),
        np.zeros((shape.batch_size, num_heads, head_dim, 1), dtype=np.float32),
        dtype=np.float32,
    )
    state_kv_rec = loop.add_recurrence(state_kv_init)
    state_z_rec = loop.add_recurrence(state_z_init)
    state_kv = state_kv_rec.get_output(0)
    state_z = state_z_rec.get_output(0)

    state_kv_decayed = network.add_elementwise(
        state_kv,
        gt,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    state_z_decayed = network.add_elementwise(
        state_z,
        gt,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)

    v_pred = network.add_matrix_multiply(
        state_kv_decayed,
        trt_module.MatrixOperation.NONE,
        krt,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    delta_v = network.add_elementwise(
        vt,
        v_pred,
        trt_module.ElementWiseOperation.SUB,
    ).get_output(0)
    delta_v = network.add_elementwise(
        delta_v,
        bt,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    state_kv_delta = network.add_matrix_multiply(
        delta_v,
        trt_module.MatrixOperation.NONE,
        krt,
        trt_module.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    state_kv_new = network.add_elementwise(
        state_kv_decayed,
        state_kv_delta,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)

    z_pred = network.add_matrix_multiply(
        state_z_decayed,
        trt_module.MatrixOperation.TRANSPOSE,
        kt,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    one = _add_ranked_scalar(network, trt_module, 4, 1.0, dtype=np.float32)
    delta_z = network.add_elementwise(
        one,
        z_pred,
        trt_module.ElementWiseOperation.SUB,
    ).get_output(0)
    delta_z = network.add_elementwise(
        delta_z,
        bt,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    state_z_delta = network.add_matrix_multiply(
        kt,
        trt_module.MatrixOperation.NONE,
        delta_z,
        trt_module.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    state_z_new = network.add_elementwise(
        state_z_decayed,
        state_z_delta,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    state_kv_rec.set_input(1, state_kv_new)
    state_z_rec.set_input(1, state_z_new)

    num_t = network.add_matrix_multiply(
        state_kv_new,
        trt_module.MatrixOperation.NONE,
        qrt,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    den_t = network.add_matrix_multiply(
        state_z_new,
        trt_module.MatrixOperation.TRANSPOSE,
        qt,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)

    return SanaWmStage1GdnComponents(
        num=_add_bhds_loop_concat_output(
            network,
            loop,
            count,
            num_t,
            shape,
            num_heads=num_heads,
            channels=head_dim,
            trt_module=trt_module,
            name=f"{name}.num",
            reverse_output=reverse_output,
        ),
        den=_add_bhds_loop_concat_output(
            network,
            loop,
            count,
            den_t,
            shape,
            num_heads=num_heads,
            channels=1,
            trt_module=trt_module,
            name=f"{name}.den",
            reverse_output=reverse_output,
        ),
    )


def _lower_sana_wm_stage1_camera_single_path_forward_loop(
    network: Any,
    framed: dict[str, Any],
    beta: Any,
    decay: Any,
    shape: SanaWmStage1Shape,
    *,
    num_heads: int,
    head_dim: int,
    trt_module: Any,
    name: str,
    reverse_output: bool = False,
) -> Any:
    loop, count = _add_trt_count_loop(network, trt_module, shape.latent_frames)
    beta_5d = _reshape_bhts_to_bht1s(
        network,
        beta,
        shape,
        num_heads=num_heads,
        name=f"{name}.beta_loop_input",
    )
    decay_5d = _reshape_bht_to_bht11(
        network,
        decay,
        shape,
        num_heads=num_heads,
        name=f"{name}.decay_loop_input",
    )
    qt = _set_tensor_name(
        loop.add_iterator(framed["q_rot"], 2, False).get_output(0),
        f"{name}.iter.q_rot",
    )
    kt = _set_tensor_name(
        loop.add_iterator(framed["k_rot"], 2, False).get_output(0),
        f"{name}.iter.k_rot",
    )
    vt = _set_tensor_name(
        loop.add_iterator(framed["v"], 2, False).get_output(0),
        f"{name}.iter.v",
    )
    bt = _set_tensor_name(
        loop.add_iterator(beta_5d, 2, False).get_output(0),
        f"{name}.iter.beta",
    )
    gt = _set_tensor_name(
        loop.add_iterator(decay_5d, 2, False).get_output(0),
        f"{name}.iter.decay",
    )

    state_kv_init = _add_constant(
        network,
        trt_module,
        (shape.batch_size, num_heads, head_dim, head_dim),
        np.zeros((shape.batch_size, num_heads, head_dim, head_dim), dtype=np.float32),
        dtype=np.float32,
    )
    state_kv_rec = loop.add_recurrence(state_kv_init)
    state_kv = state_kv_rec.get_output(0)

    state_kv_decayed = network.add_elementwise(
        state_kv,
        gt,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    v_pred = network.add_matrix_multiply(
        state_kv_decayed,
        trt_module.MatrixOperation.NONE,
        kt,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    delta_v = network.add_elementwise(
        vt,
        v_pred,
        trt_module.ElementWiseOperation.SUB,
    ).get_output(0)
    delta_v = network.add_elementwise(
        delta_v,
        bt,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    state_kv_delta = network.add_matrix_multiply(
        delta_v,
        trt_module.MatrixOperation.NONE,
        kt,
        trt_module.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    state_kv_new = network.add_elementwise(
        state_kv_decayed,
        state_kv_delta,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    state_kv_rec.set_input(1, state_kv_new)
    out_t = network.add_matrix_multiply(
        state_kv_new,
        trt_module.MatrixOperation.NONE,
        qt,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    return _add_bhds_loop_concat_output(
        network,
        loop,
        count,
        out_t,
        shape,
        num_heads=num_heads,
        channels=head_dim,
        trt_module=trt_module,
        name=name,
        reverse_output=reverse_output,
    )


def lower_sana_wm_stage1_gdn_forward_components(
    network: Any,
    preamble: SanaWmStage1BlockPreamble,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str = "blocks.0.attn.gdn_fwd",
    reverse_output: bool = False,
) -> SanaWmStage1GdnComponents:
    """Lower upstream ``torch_recurrent_sana_gdn`` forward components.

    This implements the inclusive 1..t pass used by bidirectional GDN.  The
    exclusive backward pass is still a separate native-lowering step.
    """
    q = _ensure_fp32(network, preamble.q, trt_module, dtype)
    k = _ensure_fp32(network, preamble.k, trt_module, dtype)
    v = _ensure_fp32(network, preamble.v, trt_module, dtype)
    q_rot = _ensure_fp32(network, preamble.q_rot, trt_module, dtype)
    k_rot = _ensure_fp32(network, preamble.k_rot, trt_module, dtype)
    beta = _ensure_fp32(network, preamble.beta, trt_module, dtype)
    decay = _ensure_fp32(network, preamble.decay, trt_module, dtype)

    framed = {
        "q": _reshape_bhdn_to_bhtds(
            network,
            q,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "k": _reshape_bhdn_to_bhtds(
            network,
            k,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "v": _reshape_bhdn_to_bhtds(
            network,
            v,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "q_rot": _reshape_bhdn_to_bhtds(
            network,
            q_rot,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
        "k_rot": _reshape_bhdn_to_bhtds(
            network,
            k_rot,
            shape,
            frontend,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
        ),
    }

    plugin_components = _lower_sana_wm_stage1_gdn_forward_components_plugin(
        network,
        framed,
        beta,
        decay,
        shape,
        num_heads=preamble.num_heads,
        head_dim=preamble.head_dim,
        trt_module=trt_module,
        name=name,
        reverse_output=reverse_output,
    )
    if plugin_components is not None:
        return plugin_components

    if _can_use_trt_loop(network, trt_module):
        return _lower_sana_wm_stage1_gdn_forward_components_loop(
            network,
            framed,
            beta,
            decay,
            shape,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            trt_module=trt_module,
            name=name,
            reverse_output=reverse_output,
        )

    state_kv = _add_constant(
        network,
        trt_module,
        (1, 1, preamble.head_dim, preamble.head_dim),
        np.zeros((1, 1, preamble.head_dim, preamble.head_dim), dtype=np.float32),
        dtype=np.float32,
    )
    state_z = _add_constant(
        network,
        trt_module,
        (1, 1, preamble.head_dim, 1),
        np.zeros((1, 1, preamble.head_dim, 1), dtype=np.float32),
        dtype=np.float32,
    )
    one = _add_ranked_scalar(network, trt_module, 4, 1.0, dtype=np.float32)
    nums = []
    dens = []
    for frame_index in range(shape.latent_frames):
        suffix = f"{name}.t{frame_index}"
        qt = _slice_sana_wm_gdn_frame(
            network,
            framed["q"],
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            name=f"{suffix}.q",
        )
        kt = _slice_sana_wm_gdn_frame(
            network,
            framed["k"],
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            name=f"{suffix}.k",
        )
        vt = _slice_sana_wm_gdn_frame(
            network,
            framed["v"],
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            name=f"{suffix}.v",
        )
        qrt = _slice_sana_wm_gdn_frame(
            network,
            framed["q_rot"],
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            name=f"{suffix}.q_rot",
        )
        krt = _slice_sana_wm_gdn_frame(
            network,
            framed["k_rot"],
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            head_dim=preamble.head_dim,
            name=f"{suffix}.k_rot",
        )
        bt = _slice_sana_wm_gdn_beta(
            network,
            beta,
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            name=f"{suffix}.beta",
        )
        gt = _slice_sana_wm_gdn_decay(
            network,
            decay,
            shape,
            frame_index=frame_index,
            num_heads=preamble.num_heads,
            name=f"{suffix}.decay",
        )
        state_kv = network.add_elementwise(
            state_kv,
            gt,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)
        state_z = network.add_elementwise(
            state_z,
            gt,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)

        v_pred = network.add_matrix_multiply(
            state_kv,
            trt_module.MatrixOperation.NONE,
            krt,
            trt_module.MatrixOperation.NONE,
        ).get_output(0)
        delta_v = network.add_elementwise(
            vt,
            v_pred,
            trt_module.ElementWiseOperation.SUB,
        ).get_output(0)
        delta_v = network.add_elementwise(
            delta_v,
            bt,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)
        state_kv_delta = network.add_matrix_multiply(
            delta_v,
            trt_module.MatrixOperation.NONE,
            _transpose_bhds(network, krt, trt_module),
            trt_module.MatrixOperation.NONE,
        ).get_output(0)
        state_kv = network.add_elementwise(
            state_kv,
            state_kv_delta,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)

        z_pred = network.add_matrix_multiply(
            _transpose_bhds(network, state_z, trt_module),
            trt_module.MatrixOperation.NONE,
            kt,
            trt_module.MatrixOperation.NONE,
        ).get_output(0)
        delta_z = network.add_elementwise(
            one,
            z_pred,
            trt_module.ElementWiseOperation.SUB,
        ).get_output(0)
        delta_z = network.add_elementwise(
            delta_z,
            bt,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)
        state_z_delta = network.add_matrix_multiply(
            kt,
            trt_module.MatrixOperation.NONE,
            _transpose_bhds(network, delta_z, trt_module),
            trt_module.MatrixOperation.NONE,
        ).get_output(0)
        state_z = network.add_elementwise(
            state_z,
            state_z_delta,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)

        nums.append(
            network.add_matrix_multiply(
                state_kv,
                trt_module.MatrixOperation.NONE,
                qrt,
                trt_module.MatrixOperation.NONE,
            ).get_output(0)
        )
        dens.append(
            network.add_matrix_multiply(
                _transpose_bhds(network, state_z, trt_module),
                trt_module.MatrixOperation.NONE,
                qt,
                trt_module.MatrixOperation.NONE,
            ).get_output(0)
        )

    if reverse_output:
        nums = list(reversed(nums))
        dens = list(reversed(dens))
    num_concat = network.add_concatenation(nums)
    num_concat.axis = 3
    den_concat = network.add_concatenation(dens)
    den_concat.axis = 3
    return SanaWmStage1GdnComponents(
        num=_set_tensor_name(num_concat.get_output(0), f"{name}.num"),
        den=_set_tensor_name(den_concat.get_output(0), f"{name}.den"),
    )


def lower_sana_wm_stage1_camera_single_path_forward(
    network: Any,
    camera: SanaWmStage1CameraUcpe,
    decay: Any,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str = "blocks.0.attn.cam_gdn_fwd",
    reverse_output: bool = False,
) -> Any:
    """Lower upstream numerator-only camera delta-rule recurrence."""
    q_rot = _ensure_fp32(network, camera.q_rot, trt_module, dtype)
    k_rot = _ensure_fp32(network, camera.k_rot, trt_module, dtype)
    v = _ensure_fp32(network, camera.v, trt_module, dtype)
    beta = camera.beta
    decay = _ensure_fp32(network, decay, trt_module, dtype)

    framed = {
        "q_rot": _reshape_bhdn_to_bhtds(
            network,
            q_rot,
            shape,
            frontend,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
        ),
        "k_rot": _reshape_bhdn_to_bhtds(
            network,
            k_rot,
            shape,
            frontend,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
        ),
        "v": _reshape_bhdn_to_bhtds(
            network,
            v,
            shape,
            frontend,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
        ),
    }

    plugin_out = _lower_sana_wm_stage1_camera_single_path_forward_plugin(
        network,
        framed,
        beta,
        decay,
        shape,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        trt_module=trt_module,
        name=name,
        reverse_output=reverse_output,
    )
    if plugin_out is not None:
        return plugin_out

    if _can_use_trt_loop(network, trt_module):
        return _lower_sana_wm_stage1_camera_single_path_forward_loop(
            network,
            framed,
            beta,
            decay,
            shape,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            trt_module=trt_module,
            name=name,
            reverse_output=reverse_output,
        )

    state_kv = _add_constant(
        network,
        trt_module,
        (1, 1, camera.head_dim, camera.head_dim),
        np.zeros((1, 1, camera.head_dim, camera.head_dim), dtype=np.float32),
        dtype=np.float32,
    )
    outs = []
    for frame_index in range(shape.latent_frames):
        suffix = f"{name}.t{frame_index}"
        qt = _slice_sana_wm_gdn_frame(
            network,
            framed["q_rot"],
            shape,
            frame_index=frame_index,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            name=f"{suffix}.q_rot",
        )
        kt = _slice_sana_wm_gdn_frame(
            network,
            framed["k_rot"],
            shape,
            frame_index=frame_index,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            name=f"{suffix}.k_rot",
        )
        vt = _slice_sana_wm_gdn_frame(
            network,
            framed["v"],
            shape,
            frame_index=frame_index,
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
            name=f"{suffix}.v",
        )
        bt = _slice_sana_wm_gdn_beta(
            network,
            beta,
            shape,
            frame_index=frame_index,
            num_heads=camera.num_heads,
            name=f"{suffix}.beta",
        )
        gt = _slice_sana_wm_gdn_decay(
            network,
            decay,
            shape,
            frame_index=frame_index,
            num_heads=camera.num_heads,
            name=f"{suffix}.decay",
        )
        state_kv = network.add_elementwise(
            state_kv,
            gt,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)
        v_pred = network.add_matrix_multiply(
            state_kv,
            trt_module.MatrixOperation.NONE,
            kt,
            trt_module.MatrixOperation.NONE,
        ).get_output(0)
        delta_v = network.add_elementwise(
            vt,
            v_pred,
            trt_module.ElementWiseOperation.SUB,
        ).get_output(0)
        delta_v = network.add_elementwise(
            delta_v,
            bt,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)
        state_kv_delta = network.add_matrix_multiply(
            delta_v,
            trt_module.MatrixOperation.NONE,
            _transpose_bhds(network, kt, trt_module),
            trt_module.MatrixOperation.NONE,
        ).get_output(0)
        state_kv = network.add_elementwise(
            state_kv,
            state_kv_delta,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
        outs.append(
            network.add_matrix_multiply(
                state_kv,
                trt_module.MatrixOperation.NONE,
                qt,
                trt_module.MatrixOperation.NONE,
            ).get_output(0)
        )

    if reverse_output:
        outs = list(reversed(outs))
    concat = network.add_concatenation(outs)
    concat.axis = 3
    return _set_tensor_name(concat.get_output(0), name)


def lower_sana_wm_stage1_camera_single_path_core(
    network: Any,
    camera: SanaWmStage1CameraUcpe,
    preamble: SanaWmStage1BlockPreamble,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    raw_config: dict,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str = "blocks.0.attn.cam_gdn",
    debug_return: str | None = None,
) -> Any:
    """Lower the BothTriton bidirectional single-path camera branch."""
    out = _lower_sana_wm_stage1_camera_single_path_combined_plugin(
        network,
        camera,
        preamble.decay,
        shape,
        frontend,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.combined_scan",
    )
    if out is None:
        fwd = lower_sana_wm_stage1_camera_single_path_forward(
            network,
            camera,
            preamble.decay,
            shape,
            frontend,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.fwd",
        )
        bwd_camera = SanaWmStage1CameraUcpe(
            q_rot=_reverse_bhdn_frames(
                network,
                camera.q_rot,
                shape,
                num_heads=camera.num_heads,
                channels=camera.head_dim,
                trt_module=trt_module,
                dtype=dtype,
                name=f"{name}.bwd.q_rot",
            ),
            k_rot=_flip_shift_bhdn_frames(
                network,
                camera.k_rot,
                shape,
                num_heads=camera.num_heads,
                channels=camera.head_dim,
                trt_module=trt_module,
                dtype=dtype,
                name=f"{name}.bwd.k_rot",
            ),
            v=_flip_shift_bhdn_frames(
                network,
                camera.v,
                shape,
                num_heads=camera.num_heads,
                channels=camera.head_dim,
                trt_module=trt_module,
                dtype=dtype,
                name=f"{name}.bwd.v",
            ),
            beta=_flip_shift_bhts_frames(
                network,
                camera.beta,
                shape,
                num_heads=camera.num_heads,
                trt_module=trt_module,
                dtype=np.float32,
                name=f"{name}.bwd.beta",
            ),
            num_heads=camera.num_heads,
            head_dim=camera.head_dim,
        )
        bwd_decay = _flip_shift_bht_frames(
            network,
            preamble.decay,
            shape,
            num_heads=camera.num_heads,
            trt_module=trt_module,
            name=f"{name}.bwd.decay",
        )
        bwd_flipped = lower_sana_wm_stage1_camera_single_path_forward(
            network,
            bwd_camera,
            bwd_decay,
            shape,
            frontend,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.bwd.scan",
            reverse_output=True,
        )
        out = network.add_elementwise(
            fwd,
            bwd_flipped,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
    if debug_return == "camera_scan":
        return _set_tensor_name(out, f"{name}.scan.debug")
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))

    out_bhnd = _transpose_bhdn_to_bhnd(network, out, trt_module, name=f"{name}.out_bhnd")
    cos, sin = _add_ucpe_cam_rope_constants(
        network,
        shape,
        camera.head_dim,
        raw_config,
        trt_module=trt_module,
    )
    cos_residual, sin_residual = _add_ucpe_cam_rope_residual_constants(
        network,
        shape,
        camera.head_dim,
        raw_config,
        trt_module=trt_module,
    )
    out_ucpe = _apply_ucpe_block_diagonal_to_bhnd(
        network,
        out_bhnd,
        inputs["raymats"],
        cos,
        sin,
        shape,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inverse_rope=True,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.inverse_ucpe",
        rope_cos_residual=cos_residual,
        rope_sin_residual=sin_residual,
    )
    out_bhdn = _transpose_bhnd_to_bhdn(network, out_ucpe, trt_module, name=f"{name}.out_bhdn")
    tokens = network.add_shuffle(out_bhdn)
    tokens.first_transpose = trt_module.Permutation([0, 3, 1, 2])
    tokens.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        camera.num_heads * camera.head_dim,
    )
    return _set_tensor_name(tokens.get_output(0), f"{name}.tokens")


def lower_sana_wm_stage1_bidirectional_gdn_core(
    network: Any,
    preamble: SanaWmStage1BlockPreamble,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    raw_config: dict,
    *,
    weights: WeightDict | None = None,
    block_index: int | None = None,
    trt_module: Any,
    dtype: np.dtype,
    name: str = "blocks.0.attn.gdn",
) -> SanaWmStage1GdnCore:
    """Lower the main bidirectional GDN numerator/denominator core."""
    if weights is not None and block_index is not None:
        raw_plugin_core = _lower_sana_wm_stage1_bidirectional_gdn_core_raw_plugin(
            network,
            preamble,
            shape,
            frontend,
            weights,
            raw_config,
            block_index=block_index,
            trt_module=trt_module,
            dtype=dtype,
            name=name,
        )
        if raw_plugin_core is not None:
            return raw_plugin_core
    plugin_core = _lower_sana_wm_stage1_bidirectional_gdn_core_plugin(
        network,
        preamble,
        shape,
        frontend,
        raw_config,
        trt_module=trt_module,
        dtype=dtype,
        name=name,
    )
    if plugin_core is not None:
        return plugin_core

    scan_dtype = np.float32
    fwd = lower_sana_wm_stage1_gdn_forward_components(
        network,
        preamble,
        shape,
        frontend,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.fwd",
    )
    bwd_preamble = SanaWmStage1BlockPreamble(
        x_msa_in=preamble.x_msa_in,
        qkv=preamble.qkv,
        qkv_heads=preamble.qkv_heads,
        q=_reverse_bhdn_frames(
            network,
            preamble.q,
            shape,
            num_heads=preamble.num_heads,
            channels=preamble.head_dim,
            trt_module=trt_module,
            dtype=scan_dtype,
            name=f"{name}.bwd.q",
        ),
        k=_flip_shift_bhdn_frames(
            network,
            preamble.k,
            shape,
            num_heads=preamble.num_heads,
            channels=preamble.head_dim,
            trt_module=trt_module,
            dtype=scan_dtype,
            name=f"{name}.bwd.k",
        ),
        q_rot=_reverse_bhdn_frames(
            network,
            preamble.q_rot,
            shape,
            num_heads=preamble.num_heads,
            channels=preamble.head_dim,
            trt_module=trt_module,
            dtype=scan_dtype,
            name=f"{name}.bwd.q_rot",
        ),
        k_rot=_flip_shift_bhdn_frames(
            network,
            preamble.k_rot,
            shape,
            num_heads=preamble.num_heads,
            channels=preamble.head_dim,
            trt_module=trt_module,
            dtype=scan_dtype,
            name=f"{name}.bwd.k_rot",
        ),
        v=_flip_shift_bhdn_frames(
            network,
            preamble.v,
            shape,
            num_heads=preamble.num_heads,
            channels=preamble.head_dim,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.bwd.v",
        ),
        beta=_flip_shift_bhts_frames(
            network,
            preamble.beta,
            shape,
            num_heads=preamble.num_heads,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{name}.bwd.beta",
        ),
        decay=_flip_shift_bht_frames(
            network,
            preamble.decay,
            shape,
            num_heads=preamble.num_heads,
            trt_module=trt_module,
            name=f"{name}.bwd.decay",
        ),
        num_heads=preamble.num_heads,
        head_dim=preamble.head_dim,
        modulation=preamble.modulation,
    )
    bwd_flipped = lower_sana_wm_stage1_gdn_forward_components(
        network,
        bwd_preamble,
        shape,
        frontend,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.bwd.scan",
        reverse_output=True,
    )
    total_num = network.add_elementwise(
        fwd.num,
        bwd_flipped.num,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    total_den = network.add_elementwise(
        fwd.den,
        bwd_flipped.den,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    eps = _add_ranked_scalar(
        network,
        trt_module,
        4,
        _stage1_attention_eps(raw_config),
        dtype=np.float32,
    )
    total_den_eps = network.add_elementwise(
        total_den,
        eps,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    out = network.add_elementwise(
        total_num,
        total_den_eps,
        trt_module.ElementWiseOperation.DIV,
    ).get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))

    tokens = network.add_shuffle(out)
    tokens.first_transpose = trt_module.Permutation([0, 3, 1, 2])
    tokens.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        preamble.num_heads * preamble.head_dim,
    )
    return SanaWmStage1GdnCore(
        tokens=_set_tensor_name(tokens.get_output(0), f"{name}.tokens"),
        num=_set_tensor_name(total_num, f"{name}.num"),
        den=_set_tensor_name(total_den, f"{name}.den"),
    )


def lower_sana_wm_stage1_gdn_output_projection(
    network: Any,
    core: SanaWmStage1GdnCore,
    preamble: SanaWmStage1BlockPreamble,
    weights: WeightDict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
    debug_return: str | None = None,
) -> Any:
    """Lower upstream output_gate(out, x) followed by attn.proj."""
    return _lower_sana_wm_stage1_output_gate_projection(
        network,
        core.tokens,
        preamble,
        weights,
        block_index=block_index,
        trt_module=trt_module,
        dtype=dtype,
        debug_return=debug_return,
    )


def _lower_sana_wm_stage1_output_gate_projection(
    network: Any,
    tokens: Any,
    preamble: SanaWmStage1BlockPreamble,
    weights: WeightDict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
    debug_return: str | None = None,
) -> Any:
    """Lower upstream ``_apply_output_gate`` followed by ``attn.proj``."""
    prefix = f"blocks.{block_index}.attn"
    hidden_size = preamble.num_heads * preamble.head_dim
    gated = None
    if debug_return not in _STAGE1_OUTPUT_GATE_DEBUG_RETURNS:
        gated = _add_sana_wm_output_gate_plugin(
            network,
            tokens,
            preamble.x_msa_in,
            weights=weights,
            prefix=f"{prefix}.output_gate",
            hidden_size=hidden_size,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.output_gate.gated",
        )
    if gated is not None:
        gated = _cast_to_dtype(network, gated, _trt_dtype_for_np(trt_module, dtype))
        projected = _add_sana_wm_bf16_linear_plugin(
            network,
            gated,
            weights=weights,
            prefix=f"{prefix}.proj",
            input_dim=hidden_size,
            output_dim=hidden_size,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.proj",
        )
        if projected is None:
            projected = _add_linear(
                network,
                gated,
                weights=weights,
                prefix=f"{prefix}.proj",
                input_dim=hidden_size,
                output_dim=hidden_size,
                batch_prefix_rank=1,
                trt_module=trt_module,
                dtype=dtype,
            )
        return _set_tensor_name(
            projected,
            f"{prefix}.output",
        )
    gate = _add_linear(
        network,
        preamble.x_msa_in,
        weights=weights,
        prefix=f"{prefix}.output_gate",
        input_dim=hidden_size,
        output_dim=hidden_size,
        batch_prefix_rank=1,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "output_gate_linear":
        return _set_tensor_name(gate, f"{prefix}.output_gate.linear.debug")
    if dtype != np.float32:
        gate = _cast_to_dtype(network, gate, trt_module.float32)
    if debug_return == "output_gate_linear_fp32":
        return _set_tensor_name(gate, f"{prefix}.output_gate.linear_fp32.debug")
    gate = _add_silu(network, gate, trt_module=trt_module)
    if debug_return == "output_gate":
        return _set_tensor_name(gate, f"{prefix}.output_gate.silu.debug")
    core_tokens = tokens
    if dtype != np.float32:
        core_tokens = _cast_to_dtype(network, core_tokens, trt_module.float32)
    if debug_return == "output_gate_core":
        return _set_tensor_name(core_tokens, f"{prefix}.output_gate.core_fp32.debug")
    gated = network.add_elementwise(
        core_tokens,
        gate,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    if debug_return == "output_gated":
        return _set_tensor_name(gated, f"{prefix}.output_gate.gated.debug")
    if dtype != np.float32:
        gated = _cast_to_dtype(network, gated, _trt_dtype_for_np(trt_module, dtype))
    if debug_return == "output_gated_cast":
        return _set_tensor_name(gated, f"{prefix}.output_gate.gated_cast.debug")
    projected = _add_sana_wm_bf16_linear_plugin(
        network,
        gated,
        weights=weights,
        prefix=f"{prefix}.proj",
        input_dim=hidden_size,
        output_dim=hidden_size,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.proj",
    )
    if projected is None:
        projected = _add_linear(
            network,
            gated,
            weights=weights,
            prefix=f"{prefix}.proj",
            input_dim=hidden_size,
            output_dim=hidden_size,
            batch_prefix_rank=1,
            trt_module=trt_module,
            dtype=dtype,
        )
    return _set_tensor_name(
        projected,
        f"{prefix}.output",
    )


def _pad_bhnd_head_dim(
    network: Any,
    tensor: Any,
    current_head_dim: int,
    target_head_dim: int,
    *,
    name: str,
) -> Any:
    if current_head_dim == target_head_dim:
        return tensor
    if current_head_dim > target_head_dim:
        raise ValueError(
            f"Cannot pad {name} from head dim {current_head_dim} down to {target_head_dim}"
        )
    layer = network.add_padding_nd(
        tensor,
        pre_padding=(0, 0),
        post_padding=(0, target_head_dim - current_head_dim),
    )
    return _set_tensor_name(layer.get_output(0), name)


def _slice_bhnd_head_dim(
    network: Any,
    tensor: Any,
    head_dim: int,
    *,
    name: str,
) -> Any:
    raw_dims = getattr(tensor, "shape", None)
    if raw_dims is None:
        return _set_tensor_name(tensor, name)
    dims = tuple(int(dim) for dim in raw_dims)
    if len(dims) != 4:
        raise ValueError(f"Expected BHND tensor for {name}, got shape {dims!r}")
    if dims[-1] == head_dim:
        return tensor
    return _add_slice(
        network,
        tensor,
        start=(0, 0, 0, 0),
        shape=(dims[0], dims[1], dims[2], head_dim),
        stride=(1, 1, 1, 1),
        name=name,
    )


def _lower_softmax_attention_bhnd(
    network: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    head_dim: int,
    scale_head_dim: int | None = None,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    """Lower scaled dot-product attention for tensors shaped ``(B, H, N, D)``."""
    effective_scale_head_dim = head_dim if scale_head_dim is None else scale_head_dim
    kernel_head_dim = (
        _stage1_softmax_sdpa_head_dim(head_dim) if _stage1_pad_softmax_head_dim() else head_dim
    )
    if not _stage1_explicit_softmax_attention():
        scale = _add_ranked_scalar(
            network,
            trt_module,
            4,
            effective_scale_head_dim**-0.5,
            dtype=dtype,
        )
        q_scaled = network.add_elementwise(
            q,
            scale,
            trt_module.ElementWiseOperation.PROD,
        ).get_output(0)
        if kernel_head_dim != head_dim:
            q_scaled = _pad_bhnd_head_dim(
                network,
                q_scaled,
                head_dim,
                kernel_head_dim,
                name=f"{name}.q_pad",
            )
            k = _pad_bhnd_head_dim(
                network,
                k,
                head_dim,
                kernel_head_dim,
                name=f"{name}.k_pad",
            )
            v = _pad_bhnd_head_dim(
                network,
                v,
                head_dim,
                kernel_head_dim,
                name=f"{name}.v_pad",
            )
        attention = network.add_attention(
            q_scaled,
            k,
            v,
            trt_module.AttentionNormalizationOp.SOFTMAX,
            False,
        )
        if attention is not None:
            attention.name = f"{name}.attention"
            attention.decomposable = _stage1_decomposable_softmax_attention()
            out = attention.get_output(0)
            if kernel_head_dim != head_dim:
                out = _slice_bhnd_head_dim(
                    network,
                    out,
                    head_dim,
                    name=f"{name}.slice",
                )
            return _set_tensor_name(out, name)

    q_attn = q if dtype == np.float32 else _cast_to_dtype(network, q, trt_module.float32)
    k_attn = k if dtype == np.float32 else _cast_to_dtype(network, k, trt_module.float32)
    v_attn = v if dtype == np.float32 else _cast_to_dtype(network, v, trt_module.float32)
    scores = network.add_matrix_multiply(
        q_attn,
        trt_module.MatrixOperation.NONE,
        _transpose_bhds(network, k_attn, trt_module),
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    scale = _add_ranked_scalar(
        network,
        trt_module,
        4,
        effective_scale_head_dim**-0.5,
        dtype=np.float32,
    )
    scores = network.add_elementwise(
        scores,
        scale,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    out = network.add_matrix_multiply(
        softmax.get_output(0),
        trt_module.MatrixOperation.NONE,
        v_attn,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    return _set_tensor_name(out, name)


def _stage1_softmax_sdpa_head_dim(head_dim: int) -> int:
    """Return HF's effective SDPA head dim after its FlashAttention padding shim."""
    if head_dim in (32, 64, 128, 256) or head_dim >= 256:
        return head_dim
    return 128 if head_dim <= 128 else 256


def _lower_sana_wm_torch_softmax_attention(
    network: Any,
    q_raw: Any,
    k_raw: Any,
    v_raw: Any,
    q_norm_weight: np.ndarray,
    k_norm_weight: np.ndarray,
    shape: SanaWmStage1Shape,
    raw_config: dict,
    *,
    num_heads: int,
    head_dim: int,
    camera: bool,
    raymats: Any | None,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any | None:
    if not _is_bf16_dtype(dtype):
        return None
    if any(value is None for value in (q_raw, k_raw, v_raw, q_norm_weight, k_norm_weight)):
        return None
    plugin = _create_sana_wm_softmax_attention_plugin(
        trt_module,
        frames=shape.latent_frames,
        spatial=shape.latent_height * shape.latent_width,
        heads=num_heads,
        head_dim=head_dim,
        norm_eps=_stage1_norm_eps(raw_config),
        camera=camera,
    )
    channels = num_heads * head_dim
    q_weight = _add_constant(
        network,
        trt_module,
        (channels,),
        np.asarray(q_norm_weight).reshape(channels),
        dtype=dtype,
    )
    k_weight = _add_constant(
        network,
        trt_module,
        (channels,),
        np.asarray(k_norm_weight).reshape(channels),
        dtype=dtype,
    )
    rope = _add_softmax_rope_high_low_constants(
        network,
        shape,
        head_dim,
        raw_config,
        camera=camera,
        trt_module=trt_module,
    )
    plugin_inputs = [q_raw, k_raw, v_raw, q_weight, k_weight]
    if camera:
        if raymats is None:
            raise ValueError("SANA-WM camera softmax plugin requires raymats")
        plugin_inputs.append(raymats)
    plugin_inputs.extend(rope)
    layer = network.add_plugin_v2(plugin_inputs, plugin)
    return _set_tensor_name(layer.get_output(0), name)


def lower_sana_wm_stage1_softmax_main_attention(
    network: Any,
    preamble: SanaWmStage1BlockPreamble,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    """Lower upstream ``_forward_softmax_attn(..., apply_output_gate=False)``."""
    plugin_tokens = _lower_sana_wm_torch_softmax_attention(
        network,
        preamble.q_raw,
        preamble.k_raw,
        preamble.v_raw,
        _required_weight(weights, f"blocks.{block_index}.attn.q_norm.weight"),
        _required_weight(weights, f"blocks.{block_index}.attn.k_norm.weight"),
        shape,
        raw_config,
        num_heads=preamble.num_heads,
        head_dim=preamble.head_dim,
        camera=False,
        raymats=None,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.softmax.tokens.torch_plugin",
    )
    if plugin_tokens is not None:
        return plugin_tokens
    q = _transpose_bhdn_to_bhnd(
        network,
        preamble.q_rot,
        trt_module,
        name=f"blocks.{block_index}.attn.softmax.q_bhnd",
    )
    k = _transpose_bhdn_to_bhnd(
        network,
        preamble.k_rot,
        trt_module,
        name=f"blocks.{block_index}.attn.softmax.k_bhnd",
    )
    v = _transpose_bhdn_to_bhnd(
        network,
        preamble.v,
        trt_module,
        name=f"blocks.{block_index}.attn.softmax.v_bhnd",
    )
    out = _lower_softmax_attention_bhnd(
        network,
        q,
        k,
        v,
        head_dim=preamble.head_dim,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.softmax.out_bhnd",
    )
    out_bnhd = network.add_shuffle(out)
    out_bnhd.first_transpose = trt_module.Permutation([0, 2, 1, 3])
    out_bnhd.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        preamble.num_heads * preamble.head_dim,
    )
    return _set_tensor_name(
        out_bnhd.get_output(0),
        f"blocks.{block_index}.attn.softmax.tokens",
    )


def lower_sana_wm_stage1_camera_softmax_core(
    network: Any,
    camera: SanaWmStage1CameraUcpe,
    camera_preamble: SanaWmStage1CameraPreamble,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    """Lower upstream ``_forward_cam_branch_softmax`` after UCPE QKV setup."""
    plugin_tokens = _lower_sana_wm_torch_softmax_attention(
        network,
        camera_preamble.q_raw,
        camera_preamble.k_raw,
        camera_preamble.v_raw,
        camera_preamble.q_norm_weight,
        camera_preamble.k_norm_weight,
        shape,
        raw_config,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        camera=True,
        raymats=inputs["raymats"],
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.cam_softmax.tokens.torch_plugin",
    )
    if plugin_tokens is not None:
        return plugin_tokens
    q = _transpose_bhdn_to_bhnd(
        network,
        camera.q_rot,
        trt_module,
        name=f"blocks.{block_index}.attn.cam_softmax.q_bhnd",
    )
    k = _transpose_bhdn_to_bhnd(
        network,
        camera.k_rot,
        trt_module,
        name=f"blocks.{block_index}.attn.cam_softmax.k_bhnd",
    )
    v = _transpose_bhdn_to_bhnd(
        network,
        camera.v,
        trt_module,
        name=f"blocks.{block_index}.attn.cam_softmax.v_bhnd",
    )
    out = _lower_softmax_attention_bhnd(
        network,
        q,
        k,
        v,
        head_dim=camera.head_dim,
        scale_head_dim=_stage1_softmax_sdpa_head_dim(camera.head_dim),
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.cam_softmax.out_bhnd",
    )
    cos, sin = _add_ucpe_cam_rope_constants(
        network,
        shape,
        camera.head_dim,
        raw_config,
        trt_module=trt_module,
    )
    out_ucpe = _apply_ucpe_block_diagonal_to_bhnd(
        network,
        out,
        inputs["raymats"],
        cos,
        sin,
        shape,
        num_heads=camera.num_heads,
        head_dim=camera.head_dim,
        inverse_rope=True,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.cam_softmax.inverse_ucpe",
    )
    out_bhdn = _transpose_bhnd_to_bhdn(
        network,
        out_ucpe,
        trt_module,
        name=f"blocks.{block_index}.attn.cam_softmax.out_bhdn",
    )
    tokens = network.add_shuffle(out_bhdn)
    tokens.first_transpose = trt_module.Permutation([0, 3, 1, 2])
    tokens.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        camera.num_heads * camera.head_dim,
    )
    return _set_tensor_name(
        tokens.get_output(0),
        f"blocks.{block_index}.attn.cam_softmax.tokens",
    )


def _reshape_tokens_to_bfsc(
    network: Any,
    tokens: Any,
    shape: SanaWmStage1Shape,
    hidden_size: int,
) -> Any:
    layer = network.add_shuffle(tokens)
    layer.reshape_dims = (
        shape.batch_size,
        shape.latent_frames,
        shape.latent_height * shape.latent_width,
        hidden_size,
    )
    return layer.get_output(0)


def _reshape_bfsc_to_tokens(
    network: Any,
    tensor: Any,
    shape: SanaWmStage1Shape,
    hidden_size: int,
    *,
    name: str,
) -> Any:
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = (
        shape.batch_size,
        shape.latent_frames * shape.latent_height * shape.latent_width,
        hidden_size,
    )
    return _set_tensor_name(layer.get_output(0), name)


def _apply_frame_gate_to_tokens(
    network: Any,
    tokens: Any,
    gate: Any,
    shape: SanaWmStage1Shape,
    hidden_size: int,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    if _is_bf16_dtype(dtype):
        plugin = _create_sana_wm_frame_gate_plugin(trt_module)
        add_plugin = network.add_plugin_v2
        if plugin is not None:
            layer = add_plugin([tokens, gate], plugin)
            return _set_tensor_name(layer.get_output(0), name)
    gated_4d = network.add_elementwise(
        _reshape_tokens_to_bfsc(network, tokens, shape, hidden_size),
        gate,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    return _reshape_bfsc_to_tokens(network, gated_4d, shape, hidden_size, name=name)


def _modulate_tokens_framewise(
    network: Any,
    tokens: Any,
    shift: Any,
    scale: Any,
    shape: SanaWmStage1Shape,
    hidden_size: int,
    *,
    trt_module: Any,
    dtype: np.dtype,
    name: str,
) -> Any:
    modulated = _add_t2i_modulate(
        network,
        _reshape_tokens_to_bfsc(network, tokens, shape, hidden_size),
        shift,
        scale,
        rank=4,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{name}.bfsc",
    )
    return _reshape_bfsc_to_tokens(network, modulated, shape, hidden_size, name=name)


def _add_conv2d(
    network: Any,
    inp: Any,
    *,
    weights: WeightDict,
    prefix: str,
    in_channels: int,
    out_channels: int,
    kernel_shape: tuple[int, int],
    padding: tuple[int, int],
    groups: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    weight = _required_weight(weights, f"{prefix}.weight")
    bias = _optional_weight(weights, f"{prefix}.bias")
    expected_weight_shape = (
        out_channels,
        in_channels // groups,
        kernel_shape[0],
        kernel_shape[1],
    )
    if weight.shape != expected_weight_shape:
        raise ValueError(
            f"{prefix}.weight must have shape {expected_weight_shape}, got {weight.shape}"
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError(f"{prefix}.bias must have shape ({out_channels},), got {bias.shape}")
    conv = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=kernel_shape,
        kernel=_trt_weights(trt_module, weight, dtype),
        bias=_trt_weights(trt_module, bias, dtype),
    )
    conv.stride_nd = (1, 1)
    conv.padding_nd = padding
    conv.num_groups = groups
    return _set_tensor_name(conv.get_output(0), f"{prefix}.output")


def _add_stage1_mlp_conv2d(
    network: Any,
    inp: Any,
    *,
    weights: WeightDict,
    prefix: str,
    in_channels: int,
    out_channels: int,
    kernel_shape: tuple[int, int],
    padding: tuple[int, int],
    groups: int,
    trt_module: Any,
    dtype: np.dtype,
    force_torch_conv: bool = False,
) -> Any:
    use_torch_conv = force_torch_conv
    if use_torch_conv:
        plugin_out = _add_sana_wm_torch_conv2d_plugin(
            network,
            inp,
            weights=weights,
            prefix=prefix,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_shape=kernel_shape,
            padding=padding,
            groups=groups,
            trt_module=trt_module,
            name=f"{prefix}.output",
        )
        if plugin_out is not None:
            return plugin_out
    return _add_conv2d(
        network,
        inp,
        weights=weights,
        prefix=prefix,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_shape=kernel_shape,
        padding=padding,
        groups=groups,
        trt_module=trt_module,
        dtype=dtype,
    )


def _lower_sana_wm_stage1_cross_attention(
    network: Any,
    x: Any,
    conditioning: SanaWmStage1Conditioning,
    shape: SanaWmStage1Shape,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    num_heads: int,
    hidden_size: int,
    trt_module: Any,
    dtype: np.dtype,
    debug_return: str | None = None,
) -> Any:
    prefix = f"blocks.{block_index}.cross_attn"
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"SANA-WM cross attention hidden size {hidden_size} must divide by heads {num_heads}"
        )
    head_dim = hidden_size // num_heads
    internal_debug_returns = {
        "cross_q_linear",
        "cross_k_linear",
        "cross_v_linear",
        "cross_q_norm",
        "cross_k_norm",
        "cross_scores",
        "cross_scores_masked",
        "cross_softmax",
        "cross_pre_proj",
    }
    if debug_return not in internal_debug_returns:
        plugin_output = _apply_sana_wm_cross_attention_plugin(
            network,
            x,
            conditioning,
            weights=weights,
            block_index=block_index,
            num_heads=num_heads,
            hidden_size=hidden_size,
            trt_module=trt_module,
            dtype=dtype,
        )
        if plugin_output is not None:
            return plugin_output
    q = _add_linear(
        network,
        x,
        weights=weights,
        prefix=f"{prefix}.q_linear",
        input_dim=hidden_size,
        output_dim=hidden_size,
        batch_prefix_rank=1,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "cross_q_linear":
        return _set_tensor_name(q, f"{prefix}.q_linear.debug")
    kv = _add_linear(
        network,
        conditioning.y,
        weights=weights,
        prefix=f"{prefix}.kv_linear",
        input_dim=hidden_size,
        output_dim=2 * hidden_size,
        batch_prefix_rank=2,
        trt_module=trt_module,
        dtype=dtype,
    )
    kv_flat = network.add_shuffle(kv)
    kv_flat.reshape_dims = (
        shape.batch_size,
        shape.text_max_length,
        2,
        hidden_size,
    )
    k = _add_slice(
        network,
        kv_flat.get_output(0),
        start=(0, 0, 0, 0),
        shape=(shape.batch_size, shape.text_max_length, 1, hidden_size),
        stride=(1, 1, 1, 1),
        name=f"{prefix}.k.slice",
    )
    v = _add_slice(
        network,
        kv_flat.get_output(0),
        start=(0, 0, 1, 0),
        shape=(shape.batch_size, shape.text_max_length, 1, hidden_size),
        stride=(1, 1, 1, 1),
        name=f"{prefix}.v.slice",
    )
    k_flat = network.add_shuffle(k)
    k_flat.reshape_dims = (shape.batch_size, shape.text_max_length, hidden_size)
    v_flat = network.add_shuffle(v)
    v_flat.reshape_dims = (shape.batch_size, shape.text_max_length, hidden_size)
    k = k_flat.get_output(0)
    v = v_flat.get_output(0)
    if debug_return == "cross_k_linear":
        return _set_tensor_name(k, f"{prefix}.k_linear.debug")
    if debug_return == "cross_v_linear":
        return _set_tensor_name(v, f"{prefix}.v_linear.debug")
    if _stage1_cross_norm(raw_config):
        # PR #379's MultiHeadCrossAttention hardcodes q/k RMSNorm eps=1e-6,
        # independent of the model-wide GDN norm_eps.
        cross_norm_eps = 1.0e-6
        q = _add_rmsnorm(
            network,
            q,
            _required_weight(weights, f"{prefix}.q_norm.weight"),
            rank=3,
            eps=cross_norm_eps,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.q_norm.output",
        )
        k = _add_rmsnorm(
            network,
            k,
            _required_weight(weights, f"{prefix}.k_norm.weight"),
            rank=3,
            eps=cross_norm_eps,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.k_norm.output",
        )
    if debug_return == "cross_q_norm":
        return _set_tensor_name(q, f"{prefix}.q_norm.debug")
    if debug_return == "cross_k_norm":
        return _set_tensor_name(k, f"{prefix}.k_norm.debug")

    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (
        shape.batch_size,
        shape.latent_frames * shape.latent_height * shape.latent_width,
        num_heads,
        head_dim,
    )
    q_bhnd = network.add_shuffle(q_heads.get_output(0))
    q_bhnd.first_transpose = trt_module.Permutation([0, 2, 1, 3])
    k_heads = network.add_shuffle(k)
    k_heads.reshape_dims = (shape.batch_size, shape.text_max_length, num_heads, head_dim)
    k_bhld = network.add_shuffle(k_heads.get_output(0))
    k_bhld.first_transpose = trt_module.Permutation([0, 2, 1, 3])
    v_heads = network.add_shuffle(v)
    v_heads.reshape_dims = (shape.batch_size, shape.text_max_length, num_heads, head_dim)
    v_bhld = network.add_shuffle(v_heads.get_output(0))
    v_bhld.first_transpose = trt_module.Permutation([0, 2, 1, 3])

    # Keep SANA-WM cross attention on the explicit additive-mask softmax path by
    # default.  The source mask uses 1=valid, while TensorRT IAttention's bool
    # mask convention is easy to mis-wire and can turn the uncond branch into
    # all-masked rows.
    if _stage1_use_trt_attention(raw_config):
        attention = network.add_attention(
            q_bhnd.get_output(0),
            k_bhld.get_output(0),
            v_bhld.get_output(0),
            trt_module.AttentionNormalizationOp.SOFTMAX,
            False,
        )
        if attention is not None:
            mask_bool = _cast_to_dtype(network, conditioning.mask, trt_module.bool)
            mask_4d = network.add_shuffle(mask_bool)
            mask_4d.reshape_dims = (shape.batch_size, 1, 1, shape.text_max_length)
            attention.mask = mask_4d.get_output(0)
            attn_bnhd = network.add_shuffle(attention.get_output(0))
            attn_bnhd.first_transpose = trt_module.Permutation([0, 2, 1, 3])
            attn_bnhd.reshape_dims = (
                shape.batch_size,
                shape.latent_frames * shape.latent_height * shape.latent_width,
                hidden_size,
            )
            return _set_tensor_name(
                _add_linear(
                    network,
                    attn_bnhd.get_output(0),
                    weights=weights,
                    prefix=f"{prefix}.proj",
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    batch_prefix_rank=1,
                    trt_module=trt_module,
                    dtype=dtype,
                ),
                f"{prefix}.output",
            )

    scores = network.add_matrix_multiply(
        q_bhnd.get_output(0),
        trt_module.MatrixOperation.NONE,
        _transpose_bhds(network, k_bhld.get_output(0), trt_module),
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    if dtype != np.float32:
        scores = _cast_to_dtype(network, scores, trt_module.float32)
    scale = _add_ranked_scalar(
        network,
        trt_module,
        4,
        head_dim**-0.5,
        dtype=np.float32,
    )
    scores = network.add_elementwise(
        scores,
        scale,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    if debug_return == "cross_scores":
        return _set_tensor_name(scores, f"{prefix}.scores.debug")

    mask_float = _cast_to_dtype(network, conditioning.mask, trt_module.float32)
    mask_4d = network.add_shuffle(mask_float)
    mask_4d.reshape_dims = (shape.batch_size, 1, 1, shape.text_max_length)
    one = _add_ranked_scalar(network, trt_module, 4, 1.0, dtype=np.float32)
    invalid = network.add_elementwise(
        one,
        mask_4d.get_output(0),
        trt_module.ElementWiseOperation.SUB,
    ).get_output(0)
    neg_large = _add_ranked_scalar(network, trt_module, 4, -10000.0, dtype=np.float32)
    attn_mask = network.add_elementwise(
        invalid,
        neg_large,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    scores = network.add_elementwise(
        scores,
        attn_mask,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    if debug_return == "cross_scores_masked":
        return _set_tensor_name(scores, f"{prefix}.scores_masked.debug")
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    if debug_return == "cross_softmax":
        return _set_tensor_name(softmax.get_output(0), f"{prefix}.softmax.debug")
    v_attn = v_bhld.get_output(0)
    if dtype != np.float32:
        v_attn = _cast_to_dtype(network, v_attn, trt_module.float32)
    attn = network.add_matrix_multiply(
        softmax.get_output(0),
        trt_module.MatrixOperation.NONE,
        v_attn,
        trt_module.MatrixOperation.NONE,
    ).get_output(0)
    attn_bnhd = network.add_shuffle(attn)
    attn_bnhd.first_transpose = trt_module.Permutation([0, 2, 1, 3])
    attn_bnhd.reshape_dims = (
        shape.batch_size,
        shape.latent_frames * shape.latent_height * shape.latent_width,
        hidden_size,
    )
    out = attn_bnhd.get_output(0)
    if debug_return == "cross_pre_proj":
        return _set_tensor_name(out, f"{prefix}.pre_proj.debug")
    if dtype != np.float32:
        out = _cast_to_dtype(network, out, _trt_dtype_for_np(trt_module, dtype))
    return _set_tensor_name(
        _add_linear(
            network,
            out,
            weights=weights,
            prefix=f"{prefix}.proj",
            input_dim=hidden_size,
            output_dim=hidden_size,
            batch_prefix_rank=1,
            trt_module=trt_module,
            dtype=dtype,
        ),
        f"{prefix}.output",
    )


def _lower_sana_wm_stage1_glumbconvtemp_mlp(
    network: Any,
    x: Any,
    shape: SanaWmStage1Shape,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    hidden_size: int,
    trt_module: Any,
    dtype: np.dtype,
    debug_return: str | None = None,
) -> Any:
    ffn_type = _stage1_ffn_type(raw_config)
    if ffn_type != "GLUMBConvTemp":
        raise NotImplementedError(
            "SANA-WM native builder currently lowers only ffn_type='GLUMBConvTemp', "
            f"got {ffn_type!r}"
        )
    prefix = f"blocks.{block_index}.mlp"
    hidden_features = int(hidden_size * _stage1_mlp_ratio(raw_config))
    t_kernel = _stage1_t_kernel_size(raw_config)
    if t_kernel % 2 == 0:
        raise ValueError(f"SANA-WM GLUMBConvTemp t_kernel_size must be odd, got {t_kernel}")
    internal_debug_returns = {
        "mlp_inverted_pre_silu",
        "mlp_inverted_pre_silu_sample",
        "mlp_inverted",
        "mlp_inverted_sample",
        "mlp_depth",
        "mlp_depth_sample",
        "mlp_value_sample",
        "mlp_gate_pre_silu",
        "mlp_gate_pre_silu_sample",
        "mlp_gate",
        "mlp_gate_sample",
        "mlp_gated_spatial",
        "mlp_gated_spatial_sample",
        "mlp_point_sample",
        "mlp_point_tokens",
        "mlp_temporal_sample",
        "mlp_temporal_sum_sample",
    }
    if debug_return not in internal_debug_returns:
        plugin_out = _add_sana_wm_glumbconvtemp_plugin(
            network,
            x,
            shape=shape,
            weights=weights,
            prefix=prefix,
            channels=hidden_size,
            hidden=hidden_features,
            t_kernel=t_kernel,
            trt_module=trt_module,
            name=f"{prefix}.output",
        )
        if plugin_out is not None:
            return plugin_out
    spatial_tokens = shape.latent_height * shape.latent_width
    x_btc_hw = network.add_shuffle(x)
    x_btc_hw.reshape_dims = (
        shape.batch_size * shape.latent_frames,
        shape.latent_height,
        shape.latent_width,
        hidden_size,
    )
    x_bchw = network.add_shuffle(x_btc_hw.get_output(0))
    x_bchw.first_transpose = trt_module.Permutation([0, 3, 1, 2])

    inverted_prefix = f"{prefix}.inverted_conv.conv"
    inverted = _add_stage1_mlp_conv2d(
        network,
        x_bchw.get_output(0),
        weights=weights,
        prefix=inverted_prefix,
        in_channels=hidden_size,
        out_channels=hidden_features * 2,
        kernel_shape=(1, 1),
        padding=(0, 0),
        groups=1,
        trt_module=trt_module,
        dtype=dtype,
        force_torch_conv=False,
    )
    if debug_return == "mlp_inverted_pre_silu":
        return _set_tensor_name(inverted, f"{prefix}.inverted_pre_silu.debug")
    if debug_return == "mlp_inverted_pre_silu_sample":
        return _add_slice(
            network,
            inverted,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features * 2),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.inverted_pre_silu_sample.debug",
        )
    inverted = _add_silu_for_stage1_mlp(
        network,
        inverted,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "mlp_inverted":
        return _set_tensor_name(inverted, f"{prefix}.inverted.debug")
    if debug_return == "mlp_inverted_sample":
        return _add_slice(
            network,
            inverted,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features * 2),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.inverted_sample.debug",
        )
    depth = _add_stage1_mlp_conv2d(
        network,
        inverted,
        weights=weights,
        prefix=f"{prefix}.depth_conv.conv",
        in_channels=hidden_features * 2,
        out_channels=hidden_features * 2,
        kernel_shape=(3, 3),
        padding=(1, 1),
        groups=hidden_features * 2,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "mlp_depth":
        return _set_tensor_name(depth, f"{prefix}.depth.debug")
    if debug_return == "mlp_depth_sample":
        return _add_slice(
            network,
            depth,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features * 2),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.depth_sample.debug",
        )
    value = _add_slice(
        network,
        depth,
        start=(0, 0, 0, 0),
        shape=(
            shape.batch_size * shape.latent_frames,
            hidden_features,
            shape.latent_height,
            shape.latent_width,
        ),
        stride=(1, 1, 1, 1),
        name=f"{prefix}.value",
    )
    gate = _add_slice(
        network,
        depth,
        start=(0, hidden_features, 0, 0),
        shape=(
            shape.batch_size * shape.latent_frames,
            hidden_features,
            shape.latent_height,
            shape.latent_width,
        ),
        stride=(1, 1, 1, 1),
        name=f"{prefix}.gate",
    )
    if debug_return == "mlp_value_sample":
        return _add_slice(
            network,
            value,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.value_sample.debug",
        )
    if debug_return == "mlp_gate_pre_silu":
        return _set_tensor_name(gate, f"{prefix}.gate_pre_silu.debug")
    if debug_return == "mlp_gate_pre_silu_sample":
        return _add_slice(
            network,
            gate,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.gate_pre_silu_sample.debug",
        )
    gate = _add_silu_for_stage1_mlp(
        network,
        gate,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "mlp_gate":
        return _set_tensor_name(gate, f"{prefix}.gate.debug")
    if debug_return == "mlp_gate_sample":
        return _add_slice(
            network,
            gate,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.gate_sample.debug",
        )
    gated = network.add_elementwise(
        value,
        gate,
        trt_module.ElementWiseOperation.PROD,
    ).get_output(0)
    if debug_return == "mlp_gated_spatial":
        return _set_tensor_name(gated, f"{prefix}.gated_spatial.debug")
    if debug_return == "mlp_gated_spatial_sample":
        return _add_slice(
            network,
            gated,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_features),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.gated_spatial_sample.debug",
        )
    point = _add_stage1_mlp_conv2d(
        network,
        gated,
        weights=weights,
        prefix=f"{prefix}.point_conv.conv",
        in_channels=hidden_features,
        out_channels=hidden_size,
        kernel_shape=(1, 1),
        padding=(0, 0),
        groups=1,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "mlp_point_sample":
        return _add_slice(
            network,
            point,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size * shape.latent_frames,
                min(64, hidden_size),
                shape.latent_height,
                shape.latent_width,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.point_sample.debug",
        )
    point_btcs = network.add_shuffle(point)
    point_btcs.reshape_dims = (
        shape.batch_size,
        shape.latent_frames,
        hidden_size,
        spatial_tokens,
    )
    point_bcts = network.add_shuffle(point_btcs.get_output(0))
    point_bcts.first_transpose = trt_module.Permutation([0, 2, 1, 3])
    if debug_return == "mlp_point_tokens":
        point_tokens = network.add_shuffle(point_bcts.get_output(0))
        point_tokens.first_transpose = trt_module.Permutation([0, 2, 3, 1])
        point_tokens.reshape_dims = (
            shape.batch_size,
            shape.latent_frames * spatial_tokens,
            hidden_size,
        )
        return _set_tensor_name(point_tokens.get_output(0), f"{prefix}.point_tokens.debug")
    temporal = _add_stage1_mlp_conv2d(
        network,
        point_bcts.get_output(0),
        weights=weights,
        prefix=f"{prefix}.t_conv",
        in_channels=hidden_size,
        out_channels=hidden_size,
        kernel_shape=(t_kernel, 1),
        padding=(t_kernel // 2, 0),
        groups=1,
        trt_module=trt_module,
        dtype=dtype,
    )
    if debug_return == "mlp_temporal_sample":
        return _add_slice(
            network,
            temporal,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size,
                min(64, hidden_size),
                shape.latent_frames,
                spatial_tokens,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.temporal_sample.debug",
        )
    temporal_sum = network.add_elementwise(
        point_bcts.get_output(0),
        temporal,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    if debug_return == "mlp_temporal_sum_sample":
        return _add_slice(
            network,
            temporal_sum,
            start=(0, 0, 0, 0),
            shape=(
                shape.batch_size,
                min(64, hidden_size),
                shape.latent_frames,
                spatial_tokens,
            ),
            stride=(1, 1, 1, 1),
            name=f"{prefix}.temporal_sum_sample.debug",
        )
    out_btsc = network.add_shuffle(temporal_sum)
    out_btsc.first_transpose = trt_module.Permutation([0, 2, 3, 1])
    out_btsc.reshape_dims = (
        shape.batch_size,
        shape.latent_frames * spatial_tokens,
        hidden_size,
    )
    return _set_tensor_name(out_btsc.get_output(0), f"{prefix}.output")


def lower_sana_wm_stage1_block_post_attention(
    network: Any,
    block_input: Any,
    attn_tokens: Any,
    preamble: SanaWmStage1BlockPreamble,
    conditioning: SanaWmStage1Conditioning,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
    debug_return: str | None = None,
) -> Any:
    """Lower the frame-aware residual/cross-attn/MLP body after self-attention."""
    hidden_size = frontend.hidden_size
    gated_attn = _apply_frame_gate_to_tokens(
        network,
        attn_tokens,
        preamble.modulation.gate_msa,
        shape,
        hidden_size,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.gated",
    )
    if debug_return == "attn_gated":
        return _set_tensor_name(gated_attn, f"blocks.{block_index}.attn.gated.debug")
    x = network.add_elementwise(
        block_input,
        gated_attn,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    if debug_return == "post_attn_residual":
        return _set_tensor_name(x, f"blocks.{block_index}.post_attn_residual.debug")
    if frontend.plucker_tokens is not None and _model_bool(
        raw_config,
        "use_chunk_plucker_post_attn",
    ):
        plucker = _add_sana_wm_bf16_linear_plugin(
            network,
            frontend.plucker_tokens,
            weights=weights,
            prefix=f"blocks.{block_index}.plucker_proj",
            input_dim=hidden_size,
            output_dim=hidden_size,
            trt_module=trt_module,
            dtype=dtype,
            name=f"blocks.{block_index}.plucker_proj",
            use_matmul_bias=True,
        )
        if plucker is None:
            plucker = _add_linear(
                network,
                frontend.plucker_tokens,
                weights=weights,
                prefix=f"blocks.{block_index}.plucker_proj",
                input_dim=hidden_size,
                output_dim=hidden_size,
                batch_prefix_rank=1,
                trt_module=trt_module,
                dtype=dtype,
            )
        if debug_return == "plucker":
            return _set_tensor_name(plucker, f"blocks.{block_index}.plucker.debug")
        x = network.add_elementwise(
            x,
            plucker,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
        if debug_return == "post_plucker":
            return _set_tensor_name(x, f"blocks.{block_index}.post_plucker.debug")

    cross = _lower_sana_wm_stage1_cross_attention(
        network,
        x,
        conditioning,
        shape,
        weights,
        raw_config,
        block_index=block_index,
        num_heads=preamble.num_heads,
        hidden_size=hidden_size,
        trt_module=trt_module,
        dtype=dtype,
        debug_return=debug_return,
    )
    if debug_return == "cross" or (debug_return is not None and debug_return.startswith("cross_")):
        return _set_tensor_name(cross, f"blocks.{block_index}.cross.debug")
    x = network.add_elementwise(
        x,
        cross,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    if debug_return == "post_cross":
        return _set_tensor_name(x, f"blocks.{block_index}.post_cross.debug")
    norm2 = _add_layernorm_no_affine(
        network,
        x,
        rank=3,
        eps=1.0e-6,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.norm2.output",
    )
    if debug_return == "norm2":
        return _set_tensor_name(norm2, f"blocks.{block_index}.norm2.debug")
    x_mlp_in = _modulate_tokens_framewise(
        network,
        norm2,
        preamble.modulation.shift_mlp,
        preamble.modulation.scale_mlp,
        shape,
        hidden_size,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.x_mlp_in",
    )
    if _stage1_debug_mlp_input_index() == block_index:
        x_mlp_in = network.add_input(
            "debug_mlp_input",
            _trt_dtype_for_np(trt_module, dtype),
            (shape.batch_size, frontend.token_count, hidden_size),
        )
    if debug_return == "x_mlp_in":
        return _set_tensor_name(x_mlp_in, f"blocks.{block_index}.x_mlp_in.debug")
    mlp = _lower_sana_wm_stage1_glumbconvtemp_mlp(
        network,
        x_mlp_in,
        shape,
        weights,
        raw_config,
        block_index=block_index,
        hidden_size=hidden_size,
        trt_module=trt_module,
        dtype=dtype,
        debug_return=debug_return,
    )
    if debug_return in {
        "mlp_inverted_pre_silu",
        "mlp_inverted_pre_silu_sample",
        "mlp_inverted",
        "mlp_inverted_sample",
        "mlp_depth",
        "mlp_depth_sample",
        "mlp_value_sample",
        "mlp_gate_pre_silu",
        "mlp_gate_pre_silu_sample",
        "mlp_gate",
        "mlp_gate_sample",
        "mlp_gated_spatial",
        "mlp_gated_spatial_sample",
        "mlp_point_sample",
        "mlp_point_tokens",
        "mlp_temporal_sample",
        "mlp_temporal_sum_sample",
    }:
        return mlp
    if debug_return == "mlp":
        return _set_tensor_name(mlp, f"blocks.{block_index}.mlp.debug")
    gated_mlp = _apply_frame_gate_to_tokens(
        network,
        mlp,
        preamble.modulation.gate_mlp,
        shape,
        hidden_size,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.mlp.gated",
    )
    if debug_return == "mlp_gated":
        return _set_tensor_name(gated_mlp, f"blocks.{block_index}.mlp.gated.debug")
    out = network.add_elementwise(
        x,
        gated_mlp,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    if debug_return == "block_output":
        return _set_tensor_name(out, f"blocks.{block_index}.output.debug")
    return _set_tensor_name(out, f"blocks.{block_index}.output")


def _lower_sana_wm_block_modulation(
    network: Any,
    conditioning: SanaWmStage1Conditioning,
    shape: SanaWmStage1Shape,
    hidden_size: int,
    weights: WeightDict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
) -> SanaWmStage1BlockModulation:
    prefix = f"blocks.{block_index}"
    scale_shift_table = _required_weight(weights, f"{prefix}.scale_shift_table")
    if scale_shift_table.shape != (6, hidden_size):
        raise ValueError(
            f"{prefix}.scale_shift_table must have shape (6, {hidden_size}), "
            f"got {scale_shift_table.shape}"
        )

    t0 = network.add_shuffle(conditioning.t0)
    t0.reshape_dims = (shape.batch_size, shape.latent_frames, 6, hidden_size)
    table = _add_constant(
        network,
        trt_module,
        (1, 1, 6, hidden_size),
        scale_shift_table.reshape(1, 1, 6, hidden_size),
        dtype=dtype,
    )
    mod = network.add_elementwise(
        t0.get_output(0),
        table,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)

    slice_shape = (shape.batch_size, shape.latent_frames, 1, hidden_size)
    stride = (1, 1, 1, 1)
    chunks = [
        _add_slice(
            network,
            mod,
            start=(0, 0, index, 0),
            shape=slice_shape,
            stride=stride,
            name=f"{prefix}.modulation.{name}",
        )
        for index, name in enumerate(
            (
                "shift_msa",
                "scale_msa",
                "gate_msa",
                "shift_mlp",
                "scale_mlp",
                "gate_mlp",
            )
        )
    ]
    return SanaWmStage1BlockModulation(*chunks)


def lower_sana_wm_stage1_block_preamble(
    network: Any,
    block_input: Any,
    conditioning: SanaWmStage1Conditioning,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
    softmax_attention: bool = False,
) -> SanaWmStage1BlockPreamble:
    """Lower the frame-aware adaLN and main-branch QKV part of a Sana block."""
    prefix = f"blocks.{block_index}"
    modulation = _lower_sana_wm_block_modulation(
        network,
        conditioning,
        shape,
        frontend.hidden_size,
        weights,
        block_index=block_index,
        trt_module=trt_module,
        dtype=dtype,
    )

    tokens_4d = network.add_shuffle(block_input)
    tokens_4d.reshape_dims = (
        shape.batch_size,
        shape.latent_frames,
        shape.latent_height * shape.latent_width,
        frontend.hidden_size,
    )
    norm1 = _add_layernorm_no_affine(
        network,
        tokens_4d.get_output(0),
        rank=4,
        eps=1.0e-6,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.norm1.output",
    )
    x_msa_4d = _add_t2i_modulate(
        network,
        norm1,
        modulation.shift_msa,
        modulation.scale_msa,
        rank=4,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.x_msa_4d",
    )
    x_msa_in = network.add_shuffle(x_msa_4d)
    x_msa_in.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        frontend.hidden_size,
    )
    x_msa = _set_tensor_name(x_msa_in.get_output(0), f"{prefix}.x_msa_in")

    qkv_weight = _required_weight(weights, f"{prefix}.attn.qkv.weight")
    if qkv_weight.ndim != 2 or qkv_weight.shape[0] != frontend.hidden_size:
        raise ValueError(
            f"{prefix}.attn.qkv.weight must have TRT matmul shape "
            f"({frontend.hidden_size}, 3 * hidden_size), got {qkv_weight.shape}"
        )
    qkv_dim = int(qkv_weight.shape[1])
    if qkv_dim % 3 != 0:
        raise ValueError(f"{prefix}.attn.qkv.weight output dim must divide by 3")
    head_dim = _stage1_linear_head_dim(raw_config, frontend.hidden_size)
    qkv_single_dim = qkv_dim // 3
    if qkv_single_dim % head_dim != 0:
        raise ValueError(
            f"{prefix}.attn.qkv output dim {qkv_single_dim} must divide by "
            f"linear head dim {head_dim}"
        )
    num_heads = qkv_single_dim // head_dim
    qkv = _add_sana_wm_bf16_linear_plugin(
        network,
        x_msa,
        weights=weights,
        prefix=f"{prefix}.attn.qkv",
        input_dim=frontend.hidden_size,
        output_dim=qkv_dim,
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.attn.qkv",
        enabled=False,
    )
    if qkv is None:
        qkv = _add_linear(
            network,
            x_msa,
            weights=weights,
            prefix=f"{prefix}.attn.qkv",
            input_dim=frontend.hidden_size,
            output_dim=qkv_dim,
            batch_prefix_rank=1,
            trt_module=trt_module,
            dtype=dtype,
        )
    qkv = _set_tensor_name(qkv, f"{prefix}.attn.qkv.output")
    qkv_heads = network.add_shuffle(qkv)
    qkv_heads.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        3,
        num_heads,
        head_dim,
    )
    qkv_heads_t = _set_tensor_name(qkv_heads.get_output(0), f"{prefix}.attn.qkv.heads")
    q_flat = _reshape_qkv_component(
        network,
        qkv_heads_t,
        shape,
        frontend,
        component=0,
        num_heads=num_heads,
        head_dim=head_dim,
        name=f"{prefix}.attn.q",
    )
    q_raw = q_flat
    k_flat = _reshape_qkv_component(
        network,
        qkv_heads_t,
        shape,
        frontend,
        component=1,
        num_heads=num_heads,
        head_dim=head_dim,
        name=f"{prefix}.attn.k",
    )
    k_raw = k_flat
    v_flat = _reshape_qkv_component(
        network,
        qkv_heads_t,
        shape,
        frontend,
        component=2,
        num_heads=num_heads,
        head_dim=head_dim,
        name=f"{prefix}.attn.v",
    )
    v_raw = v_flat
    k_conv = None
    if not softmax_attention:
        k_flat = _add_bidirectional_short_conv1d(
            network,
            k_flat,
            _required_weight(weights, f"{prefix}.attn.conv_k.weight"),
            _optional_weight(weights, f"{prefix}.attn.conv_k.bias"),
            shape,
            frontend,
            trt_module=trt_module,
            dtype=dtype,
            name=f"{prefix}.attn.conv_k",
        )
        k_conv = k_flat
    keep_gdn_qk_fp32 = not softmax_attention
    q_flat = _add_rmsnorm(
        network,
        q_flat,
        _required_weight(weights, f"{prefix}.attn.q_norm.weight"),
        rank=3,
        eps=_stage1_norm_eps(raw_config),
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.attn.q_norm.output",
        keep_fp32_output=keep_gdn_qk_fp32,
    )
    k_flat = _add_rmsnorm(
        network,
        k_flat,
        _required_weight(weights, f"{prefix}.attn.k_norm.weight"),
        rank=3,
        eps=_stage1_norm_eps(raw_config),
        trt_module=trt_module,
        dtype=dtype,
        name=f"{prefix}.attn.k_norm.output",
        keep_fp32_output=keep_gdn_qk_fp32,
    )
    if not softmax_attention:
        q_flat = _add_relu(
            network,
            q_flat,
            trt_module=trt_module,
            name=f"{prefix}.attn.q_relu",
        )
        k_flat = _add_relu(
            network,
            k_flat,
            trt_module=trt_module,
            name=f"{prefix}.attn.k_relu",
        )
        k_scale = _add_ranked_scalar(
            network,
            trt_module,
            3,
            (head_dim**-0.5) * ((shape.latent_height * shape.latent_width) ** -0.5),
            dtype=np.float32,
        )
        k_flat = _set_tensor_name(
            network.add_elementwise(
                k_flat,
                k_scale,
                trt_module.ElementWiseOperation.PROD,
            ).get_output(0),
            f"{prefix}.attn.k_scaled",
        )
    q = _permute_bnhd_to_bhdn(
        network,
        q_flat,
        shape,
        frontend,
        num_heads=num_heads,
        head_dim=head_dim,
        trt_module=trt_module,
        name=f"{prefix}.attn.q_bhdn",
    )
    k = _permute_bnhd_to_bhdn(
        network,
        k_flat,
        shape,
        frontend,
        num_heads=num_heads,
        head_dim=head_dim,
        trt_module=trt_module,
        name=f"{prefix}.attn.k_bhdn",
    )
    v = _permute_bnhd_to_bhdn(
        network,
        v_flat,
        shape,
        frontend,
        num_heads=num_heads,
        head_dim=head_dim,
        trt_module=trt_module,
        name=f"{prefix}.attn.v_bhdn",
    )
    q_rot, k_rot = _lower_sana_wm_wan_rope_qk(
        network,
        q,
        k,
        shape,
        frontend,
        raw_config,
        prefix=prefix,
        num_heads=num_heads,
        head_dim=head_dim,
        trt_module=trt_module,
        dtype=np.float32 if keep_gdn_qk_fp32 else dtype,
    )
    beta = decay = x_frame = gate = gate_dt = None
    if not softmax_attention:
        beta, decay, x_frame, gate, gate_dt = _lower_sana_wm_gdn_frame_gates(
            network,
            x_msa,
            shape,
            frontend,
            weights,
            block_index=block_index,
            num_heads=num_heads,
            trt_module=trt_module,
            dtype=dtype,
        )
    return SanaWmStage1BlockPreamble(
        x_msa_in=x_msa,
        qkv=qkv,
        qkv_heads=qkv_heads_t,
        q=q,
        k=k,
        q_rot=q_rot,
        k_rot=k_rot,
        v=v,
        beta=beta,
        decay=decay,
        num_heads=num_heads,
        head_dim=head_dim,
        modulation=modulation,
        norm1=norm1,
        x_msa_4d=x_msa_4d,
        x_frame=x_frame,
        gate=gate,
        gate_dt=gate_dt,
        q_raw=q_raw,
        k_raw=k_raw,
        k_conv=k_conv,
        v_raw=v_raw,
    )


def _final_layer_output_dim(weights: WeightDict, hidden_size: int) -> int:
    weight = _required_weight(weights, "final_layer.linear.weight")
    if weight.ndim != 2:
        raise ValueError(
            "final_layer.linear.weight must be a rank-2 TRT matmul tensor, "
            f"got shape {weight.shape}"
        )
    if weight.shape[0] != hidden_size:
        raise ValueError(
            "final_layer.linear.weight input dim must match SANA-WM hidden size "
            f"{hidden_size}, got {weight.shape[0]}"
        )
    bias = _optional_weight(weights, "final_layer.linear.bias")
    if bias is not None and bias.shape != (weight.shape[1],):
        raise ValueError(
            "final_layer.linear.bias must match final projection output dim "
            f"{weight.shape[1]}, got {bias.shape}"
        )
    return int(weight.shape[1])


def _unpatchify_sana_wm_stage1_tokens(
    network: Any,
    tokens: Any,
    shape: SanaWmStage1Shape,
    raw_config: dict,
    *,
    output_channels: int,
    trt_module: Any,
) -> Any:
    patch_size = _stage1_patch_size(raw_config)
    if patch_size != (1, 1, 1):
        raise NotImplementedError(
            "SANA-WM Stage-1 final unpatchify currently lowers the public "
            "SanaMSVideoCamCtrl_1600M_P1_D20 P1 path only. Non-P1 patch "
            "layouts require the full upstream nfhwopqc->ncfohpwq shuffle."
        )
    tokens_5d = network.add_shuffle(tokens)
    tokens_5d.reshape_dims = (
        shape.batch_size,
        shape.latent_frames,
        shape.latent_height,
        shape.latent_width,
        output_channels,
    )
    tokens_5d.second_transpose = trt_module.Permutation([0, 4, 1, 2, 3])

    latents = network.add_shuffle(tokens_5d.get_output(0))
    latents.reshape_dims = (
        shape.batch_size,
        output_channels,
        shape.latent_frames,
        shape.latent_height,
        shape.latent_width,
    )
    return _set_tensor_name(latents.get_output(0), "final_layer.latents")


def lower_sana_wm_stage1_final_layer(
    network: Any,
    block_output: Any,
    conditioning: SanaWmStage1Conditioning,
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    trt_module: Any,
    dtype: np.dtype,
) -> SanaWmStage1FinalOutput:
    """Lower upstream ``T2IFinalLayer.forward_frame_aware`` and unpatchify."""
    expected_tokens = shape.latent_frames * shape.latent_height * shape.latent_width
    if frontend.token_count != expected_tokens:
        raise ValueError(
            "SANA-WM public P1 final layer expects one token per latent cell: "
            f"got {frontend.token_count}, expected {expected_tokens}"
        )
    scale_shift_table = _required_weight(weights, "final_layer.scale_shift_table")
    if scale_shift_table.shape != (2, frontend.hidden_size):
        raise ValueError(
            "final_layer.scale_shift_table must have shape "
            f"(2, {frontend.hidden_size}), got {scale_shift_table.shape}"
        )

    output_dim = _final_layer_output_dim(weights, frontend.hidden_size)
    patch_size = _stage1_patch_size(raw_config)
    patch_volume = int(np.prod(patch_size))
    if output_dim % patch_volume != 0:
        raise ValueError(
            "final_layer.linear output dim must be divisible by patch volume "
            f"{patch_volume}, got {output_dim}"
        )
    output_channels = output_dim // patch_volume

    tokens_4d = network.add_shuffle(block_output)
    tokens_4d.reshape_dims = (
        shape.batch_size,
        shape.latent_frames,
        shape.latent_height * shape.latent_width,
        frontend.hidden_size,
    )
    normalized = _add_layernorm_no_affine(
        network,
        tokens_4d.get_output(0),
        rank=4,
        eps=1.0e-6,
        trt_module=trt_module,
        dtype=dtype,
        name="final_layer.norm_final.output",
    )

    t_bf1h = network.add_shuffle(conditioning.t)
    t_bf1h.first_transpose = trt_module.Permutation([0, 2, 1, 3])
    t_bf1h_t = t_bf1h.get_output(0)
    shift_table = _add_constant(
        network,
        trt_module,
        (1, 1, 1, frontend.hidden_size),
        scale_shift_table[0].reshape(1, 1, 1, frontend.hidden_size),
        dtype=dtype,
    )
    scale_table = _add_constant(
        network,
        trt_module,
        (1, 1, 1, frontend.hidden_size),
        scale_shift_table[1].reshape(1, 1, 1, frontend.hidden_size),
        dtype=dtype,
    )
    shift = network.add_elementwise(
        t_bf1h_t,
        shift_table,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    scale = network.add_elementwise(
        t_bf1h_t,
        scale_table,
        trt_module.ElementWiseOperation.SUM,
    ).get_output(0)
    modulated = _add_t2i_modulate(
        network,
        normalized,
        shift,
        scale,
        rank=4,
        trt_module=trt_module,
        dtype=dtype,
        name="final_layer.modulated",
    )

    tokens_3d = network.add_shuffle(modulated)
    tokens_3d.reshape_dims = (
        shape.batch_size,
        frontend.token_count,
        frontend.hidden_size,
    )
    projected = _set_tensor_name(
        _add_linear(
            network,
            tokens_3d.get_output(0),
            weights=weights,
            prefix="final_layer.linear",
            input_dim=frontend.hidden_size,
            output_dim=output_dim,
            batch_prefix_rank=1,
            trt_module=trt_module,
            dtype=dtype,
        ),
        "final_layer.tokens",
    )
    latents = _unpatchify_sana_wm_stage1_tokens(
        network,
        projected,
        shape,
        raw_config,
        output_channels=output_channels,
        trt_module=trt_module,
    )
    return SanaWmStage1FinalOutput(tokens=projected, latents=latents)


def lower_sana_wm_stage1_transformer_block(
    network: Any,
    block_input: Any,
    conditioning: SanaWmStage1Conditioning,
    inputs: dict[str, Any],
    shape: SanaWmStage1Shape,
    frontend: SanaWmStage1Frontend,
    weights: WeightDict,
    raw_config: dict,
    *,
    block_index: int,
    trt_module: Any,
    dtype: np.dtype,
) -> Any:
    """Lower one SANA-WM Stage-1 transformer block, including hybrid softmax blocks."""
    use_softmax = _stage1_block_uses_softmax(raw_config, block_index)
    preamble = lower_sana_wm_stage1_block_preamble(
        network,
        block_input,
        conditioning,
        shape,
        frontend,
        weights,
        raw_config,
        block_index=block_index,
        trt_module=trt_module,
        dtype=dtype,
        softmax_attention=use_softmax,
    )
    debug_return = _stage1_debug_block_return(block_index)
    if debug_return == "x_msa_in":
        return _set_tensor_name(preamble.x_msa_in, f"blocks.{block_index}.x_msa_in.debug")
    preamble_debug = {
        "t_freq": conditioning.t_freq,
        "t": conditioning.t,
        "t0": conditioning.t0,
        "norm1": preamble.norm1,
        "x_msa_4d": preamble.x_msa_4d,
        "x_frame": preamble.x_frame,
        "gate": preamble.gate,
        "gate_dt": preamble.gate_dt,
        "shift_msa": preamble.modulation.shift_msa,
        "scale_msa": preamble.modulation.scale_msa,
        "gate_msa": preamble.modulation.gate_msa,
        "qkv": preamble.qkv,
        "qkv_heads": preamble.qkv_heads,
        "q_raw": preamble.q_raw,
        "k_raw": preamble.k_raw,
        "k_conv": preamble.k_conv,
        "v_raw": preamble.v_raw,
        "q": preamble.q,
        "k": preamble.k,
        "v": preamble.v,
        "q_rot": preamble.q_rot,
        "k_rot": preamble.k_rot,
        "beta": preamble.beta,
        "decay": preamble.decay,
    }
    if debug_return in preamble_debug and preamble_debug[debug_return] is not None:
        return _set_tensor_name(
            preamble_debug[debug_return],
            f"blocks.{block_index}.attn.{debug_return}.debug",
        )
    camera = lower_sana_wm_stage1_camera_preamble(
        network,
        preamble.x_msa_in,
        shape,
        frontend,
        weights,
        raw_config,
        block_index=block_index,
        trt_module=trt_module,
        dtype=dtype,
        softmax_attention=use_softmax,
    )
    camera_ucpe = lower_sana_wm_stage1_camera_ucpe(
        network,
        camera,
        preamble,
        inputs,
        shape,
        raw_config,
        trt_module=trt_module,
        dtype=dtype,
        name=f"blocks.{block_index}.attn.cam_ucpe",
        discount_beta=not use_softmax,
        stabilize_transforms=_stage1_stabilizes_camera_ucpe(
            raw_config,
            softmax_attention=use_softmax,
        ),
    )
    camera_debug = {
        "camera_q_raw": camera.q_raw,
        "camera_k_raw": camera.k_raw,
        "camera_v_raw": camera.v_raw,
        "camera_q_pre": camera.q,
        "camera_k_pre": camera.k,
        "camera_v_pre": camera.v,
        "camera_q": camera_ucpe.q_rot,
        "camera_k": camera_ucpe.k_rot,
        "camera_v": camera_ucpe.v,
        "camera_beta": camera_ucpe.beta,
        "camera_inflation": camera_ucpe.inflation_sq,
    }
    if debug_return in camera_debug and camera_debug[debug_return] is not None:
        return _set_tensor_name(
            camera_debug[debug_return],
            f"blocks.{block_index}.attn.{debug_return}.debug",
        )

    if use_softmax:
        main_tokens = lower_sana_wm_stage1_softmax_main_attention(
            network,
            preamble,
            shape,
            frontend,
            weights,
            raw_config,
            block_index=block_index,
            trt_module=trt_module,
            dtype=dtype,
        )
        if debug_return == "main_tokens":
            return _set_tensor_name(main_tokens, f"blocks.{block_index}.attn.main.debug")
        camera_tokens = lower_sana_wm_stage1_camera_softmax_core(
            network,
            camera_ucpe,
            camera,
            inputs,
            shape,
            frontend,
            raw_config,
            block_index=block_index,
            trt_module=trt_module,
            dtype=dtype,
        )
        if debug_return == "camera_tokens":
            return _set_tensor_name(camera_tokens, f"blocks.{block_index}.attn.camera_tokens.debug")
        camera_contrib = _add_sana_wm_bf16_linear_plugin(
            network,
            camera_tokens,
            weights=weights,
            prefix=f"blocks.{block_index}.attn.out_proj_cam",
            input_dim=camera.num_heads * camera.head_dim,
            output_dim=frontend.hidden_size,
            trt_module=trt_module,
            dtype=dtype,
            name=f"blocks.{block_index}.attn.out_proj_cam",
            use_matmul_bias=True,
        )
        if camera_contrib is None:
            camera_contrib = _add_linear(
                network,
                camera_tokens,
                weights=weights,
                prefix=f"blocks.{block_index}.attn.out_proj_cam",
                input_dim=camera.num_heads * camera.head_dim,
                output_dim=frontend.hidden_size,
                batch_prefix_rank=1,
                trt_module=trt_module,
                dtype=dtype,
            )
        if debug_return == "camera_contrib":
            return _set_tensor_name(
                camera_contrib, f"blocks.{block_index}.attn.camera_contrib.debug"
            )
        combined = network.add_elementwise(
            main_tokens,
            camera_contrib,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
        if debug_return == "combined_branches":
            network.mark_output(_set_tensor_name(main_tokens, "output_main"))
            network.mark_output(_set_tensor_name(camera_contrib, "output_camera"))
            return _set_tensor_name(combined, "output0")
        if debug_return == "combined":
            return _set_tensor_name(combined, f"blocks.{block_index}.attn.combined.debug")
        attn_tokens = _lower_sana_wm_stage1_output_gate_projection(
            network,
            combined,
            preamble,
            weights,
            block_index=block_index,
            trt_module=trt_module,
            dtype=dtype,
            debug_return=debug_return,
        )
        if debug_return in _STAGE1_OUTPUT_GATE_DEBUG_RETURNS:
            return attn_tokens
    else:
        main_core = lower_sana_wm_stage1_bidirectional_gdn_core(
            network,
            preamble,
            shape,
            frontend,
            raw_config,
            weights=weights,
            block_index=block_index,
            trt_module=trt_module,
            dtype=dtype,
            name=f"blocks.{block_index}.attn.gdn",
        )
        if debug_return in _STAGE1_GDN_PHASE_DEBUG_RETURNS:
            return _set_tensor_name(
                main_core.tokens, f"blocks.{block_index}.attn.{debug_return}.debug"
            )
        camera_tokens = lower_sana_wm_stage1_camera_single_path_core(
            network,
            camera_ucpe,
            preamble,
            inputs,
            shape,
            frontend,
            raw_config,
            trt_module=trt_module,
            dtype=dtype,
            name=f"blocks.{block_index}.attn.cam_gdn",
            debug_return=debug_return,
        )
        if debug_return == "camera_scan":
            return _set_tensor_name(camera_tokens, f"blocks.{block_index}.attn.camera_scan.debug")
        if debug_return == "camera_tokens":
            return _set_tensor_name(camera_tokens, f"blocks.{block_index}.attn.camera_tokens.debug")
        camera_contrib = _add_sana_wm_bf16_linear_plugin(
            network,
            camera_tokens,
            weights=weights,
            prefix=f"blocks.{block_index}.attn.out_proj_cam",
            input_dim=camera.num_heads * camera.head_dim,
            output_dim=frontend.hidden_size,
            trt_module=trt_module,
            dtype=dtype,
            name=f"blocks.{block_index}.attn.out_proj_cam",
            use_matmul_bias=True,
        )
        if camera_contrib is None:
            camera_contrib = _add_linear(
                network,
                camera_tokens,
                weights=weights,
                prefix=f"blocks.{block_index}.attn.out_proj_cam",
                input_dim=camera.num_heads * camera.head_dim,
                output_dim=frontend.hidden_size,
                batch_prefix_rank=1,
                trt_module=trt_module,
                dtype=dtype,
            )
        if debug_return == "main_tokens":
            return _set_tensor_name(main_core.tokens, f"blocks.{block_index}.attn.main.debug")
        if debug_return == "camera_contrib":
            return _set_tensor_name(
                camera_contrib, f"blocks.{block_index}.attn.camera_contrib.debug"
            )
        combined = network.add_elementwise(
            main_core.tokens,
            camera_contrib,
            trt_module.ElementWiseOperation.SUM,
        ).get_output(0)
        if debug_return == "combined_branches":
            network.mark_output(_set_tensor_name(main_core.tokens, "output_main"))
            network.mark_output(_set_tensor_name(camera_contrib, "output_camera"))
            return _set_tensor_name(combined, "output0")
        if debug_return == "combined":
            return _set_tensor_name(combined, f"blocks.{block_index}.attn.combined.debug")
        attn_tokens = lower_sana_wm_stage1_gdn_output_projection(
            network,
            SanaWmStage1GdnCore(tokens=combined, num=main_core.num, den=main_core.den),
            preamble,
            weights,
            block_index=block_index,
            trt_module=trt_module,
            dtype=dtype,
            debug_return=debug_return,
        )
        if debug_return in _STAGE1_OUTPUT_GATE_DEBUG_RETURNS:
            return attn_tokens
    if debug_return == "attn_tokens":
        return _set_tensor_name(attn_tokens, f"blocks.{block_index}.attn.output.debug")

    post_debug_return = (
        debug_return if debug_return in _STAGE1_POST_ATTENTION_DEBUG_RETURNS else None
    )
    return lower_sana_wm_stage1_block_post_attention(
        network,
        block_input,
        attn_tokens,
        preamble,
        conditioning,
        shape,
        frontend,
        weights,
        raw_config,
        block_index=block_index,
        trt_module=trt_module,
        dtype=dtype,
        debug_return=post_debug_return,
    )


def define_sana_wm_stage1_inputs(
    network: Any,
    shape: SanaWmStage1Shape,
    *,
    trt_module: Any,
    dtype: Any,
) -> dict[str, Any]:
    """Define the native C++ runtime input contract for the Stage-1 denoiser."""
    return {
        "x": network.add_input(
            "x",
            dtype,
            (
                shape.batch_size,
                shape.latent_channels,
                shape.latent_frames,
                shape.latent_height,
                shape.latent_width,
            ),
        ),
        "timestep": network.add_input(
            "timestep",
            trt_module.float32,
            (shape.batch_size, 1, shape.latent_frames),
        ),
        "y": network.add_input(
            "y",
            dtype,
            (shape.batch_size, 1, shape.text_max_length, shape.text_embed_dim),
        ),
        "mask": network.add_input(
            "mask",
            trt_module.int32,
            (shape.batch_size, shape.text_max_length),
        ),
        "camera_conditions": network.add_input(
            "camera_conditions",
            dtype,
            (shape.batch_size, shape.latent_frames, shape.raymap_width),
        ),
        "raymats": network.add_input(
            "raymats",
            trt_module.float32,
            (
                shape.batch_size,
                shape.latent_frames * shape.latent_height * shape.latent_width,
                4,
                4,
            ),
        ),
        "raymats_inv": network.add_input(
            "raymats_inv",
            trt_module.float32,
            (
                shape.batch_size,
                shape.latent_frames * shape.latent_height * shape.latent_width,
                4,
                4,
            ),
        ),
        "chunk_plucker": network.add_input(
            "chunk_plucker",
            dtype,
            (
                shape.batch_size,
                shape.chunk_plucker_channels,
                shape.latent_frames,
                shape.latent_height,
                shape.latent_width,
            ),
        ),
    }


def _unsupported_gdn_lowering_error() -> NotImplementedError:
    return NotImplementedError(
        "SANA-WM Stage-1 DiT raw TensorRT builder could not serialize the "
        "native TensorRT network. "
        "ONNX is not used."
    )


def build_sana_wm_stage1_dit_engine(
    weights: WeightDict,
    raw_config: dict,
    *,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    """Build the SANA-WM Stage-1 denoiser directly with TensorRT APIs.

    The input contract and network shell are constructed with TensorRT APIs.
    No ONNX, tracing, Torch-TensorRT, or Python runtime bridge is used.
    """
    weight_precision = weights.get("_precision")
    if isinstance(weight_precision, str) and weight_precision != precision:
        raise ValueError(
            "SANA-WM Stage-1 weights were loaded with precision "
            f"{weight_precision!r}, but the engine build requested {precision!r}"
        )
    _BF16_WEIGHT_REFS.clear()
    trt_dtype = _target_trt_dtype(trt, precision)
    logger = get_process_trt_logger(trt, verbose=verbose)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    shape = stage1_shape_from_config(raw_config, weights)
    dtype = _target_np_dtype(precision)
    inputs = define_sana_wm_stage1_inputs(
        network,
        shape,
        trt_module=trt,
        dtype=trt_dtype,
    )
    frontend = lower_sana_wm_stage1_frontend(
        network,
        inputs,
        shape,
        weights,
        raw_config,
        trt_module=trt,
        dtype=dtype,
    )
    conditioning = lower_sana_wm_stage1_conditioning(
        network,
        inputs,
        shape,
        weights,
        hidden_size=frontend.hidden_size,
        trt_module=trt,
        dtype=dtype,
    )
    global_debug_return = _stage1_debug_block_return(0)
    if global_debug_return in {"conditioning_y", "plucker_tokens"}:
        output = (
            conditioning.y if global_debug_return == "conditioning_y" else frontend.plucker_tokens
        )
        if output is None:
            raise ValueError(f"SANA-WM debug tensor {global_debug_return!r} is unavailable")
        network.mark_output(_set_tensor_name(output, "output0"))
        plan = builder.build_serialized_network(network, builder_config)
        if plan is None:
            raise _unsupported_gdn_lowering_error()
        return bytes(plan)
    tokens = frontend.x_tokens
    debug_stop_after_block = _stage1_debug_stop_after_block()
    debug_block_input_index = _stage1_debug_block_input_index()
    if debug_stop_after_block == -1:
        output = _set_tensor_name(tokens, "output0")
        network.mark_output(output)
        plan = builder.build_serialized_network(network, builder_config)
        if plan is None:
            raise _unsupported_gdn_lowering_error()
        return bytes(plan)
    for block_index in range(_stage1_depth_from_weights(raw_config, weights)):
        if debug_block_input_index == block_index:
            tokens = network.add_input(
                "debug_block_input",
                trt_dtype,
                (shape.batch_size, frontend.token_count, frontend.hidden_size),
            )
        tokens = lower_sana_wm_stage1_transformer_block(
            network,
            tokens,
            conditioning,
            inputs,
            shape,
            frontend,
            weights,
            raw_config,
            block_index=block_index,
            trt_module=trt,
            dtype=dtype,
        )
        if debug_stop_after_block == block_index:
            output = _set_tensor_name(tokens, "output0")
            network.mark_output(output)
            plan = builder.build_serialized_network(network, builder_config)
            if plan is None:
                raise _unsupported_gdn_lowering_error()
            return bytes(plan)
    final_output = lower_sana_wm_stage1_final_layer(
        network,
        tokens,
        conditioning,
        shape,
        frontend,
        weights,
        raw_config,
        trt_module=trt,
        dtype=dtype,
    )
    output = _set_tensor_name(final_output.latents, "output0")
    network.mark_output(output)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise _unsupported_gdn_lowering_error()
    return bytes(plan)
