# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chroma DiT quantization: which projections are FP8 or NVFP4, their scales and the GEMM graphs.

No ModelOpt FP8 or NVFP4 checkpoint of Chroma1-HD is published, so the original BF16
checkpoint is quantized at build time. Only the MMDiT denoiser is quantized; the T5 text
encoder and the VAE stay as they are.

FP8 W8A8 (``quantization="fp8"``): the 380 projections of ``projection_names`` (query, key,
value and both feed-forward projections of the image and text streams of the 19 double
blocks; query, key, value, ``proj_mlp`` and the MLP half of ``proj_out`` of the 38 single
blocks). Per-tensor e4m3 weights (``weight_scale = amax / 448``, computed at build time) and a
static per-tensor activation scale from ``data/chroma1-hd-scales.json``.

NVFP4 W4A4 (``quantization="nvfp4"``): the projections of ``NVFP4_KINDS`` become packed E2M1
weights with one e4m3 scale per 16 inputs and one ``weight_scale_2 = amax / (6 * 448)`` per
tensor; the activation is block-quantized at run time (``add_dynamic_quantize``, block 16)
with a static global scale equal to the FP8 activation scale divided by 6, so both formats use
one calibration. The other projections of the FP8 recipe stay FP8.

Stay BF16 in both: the attention output projections (``attn.to_out.0``, ``attn.to_add_out``
and the attention half of the single-block ``proj_out``), the input embedders, the
modulation approximator, the final projection, the norms and the attention itself. A
projection whose packaged scale is ``null`` also stays BF16.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FP8_E4M3_MAX = 448.0
E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_MAX = E2M1_MAX * FP8_E4M3_MAX
FORMATS = ("fp8", "nvfp4")
# A calibrated input above this multiple of the median of its kind keeps the projection BF16.
OUTLIER_FACTOR = 20.0
SCALES_PATH = Path(__file__).with_name("data") / "chroma1-hd-scales.json"
# Calibrated and measured image sizes; quantized builds accept sides in this range.
MIN_IMAGE_SIZE = 512
MAX_IMAGE_SIZE = 1024
# Tensors that only a quantized export carries.
_QUANTIZED_SUFFIXES = (
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "scale_weight",
    "scale_input",
)
_QUANTIZED_DTYPES = ("F8_E4M3", "F8_E5M2", "U8", "I8")

# Projection stems inside the blocks. The single-block ``proj_out`` reads [attention output,
# MLP output]; only its MLP half (``proj_out.mlp``, the columns after the attention width) is
# quantized: TensorRT 11.4 cannot compile a Quantize on the output of the fused attention.
DOUBLE_IMAGE_STEMS = ("attn.to_q", "attn.to_k", "attn.to_v", "ff.net.0.proj", "ff.net.2")
DOUBLE_TEXT_STEMS = (
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "ff_context.net.0.proj",
    "ff_context.net.2",
)
SINGLE_STEMS = ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp", "proj_out.mlp")
SPLIT_SUFFIX = ".mlp"
# NVFP4 projections: the MLP input projection of the 38 single blocks. The MLP output half of
# proj_out reads the output of a bias + GELU and stays FP8 (an NVFP4 input there can give NaN
# rows in a partial last tile); NVFP4 on more projections lost more image quality than 2x FP8.
NVFP4_KINDS = ("single_transformer_blocks.proj_mlp",)

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between neighbouring E2M1 values. A value exactly on a midpoint goes to the
# neighbour with an even code: the first, third, fifth and seventh midpoint round down.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MIDPOINT_ROUNDS_UP = (False, True, False, True, False, True, False)


def projection_names(num_layers: int, num_single_layers: int) -> tuple[str, ...]:
    """Every quantized DiT projection (the FP8 recipe), in graph order."""
    names = []
    for index in range(num_layers):
        names += [
            f"transformer_blocks.{index}.{stem}" for stem in DOUBLE_IMAGE_STEMS + DOUBLE_TEXT_STEMS
        ]
    for index in range(num_single_layers):
        names += [f"single_transformer_blocks.{index}.{stem}" for stem in SINGLE_STEMS]
    return tuple(names)


def projection_kind(name: str) -> str:
    """The stem shared by one projection in every block (``transformer_blocks.ff.net.2``)."""
    prefix, index, stem = name.split(".", 2)
    if not index.isdigit():
        raise ValueError(f"not a block projection: {name}")
    return f"{prefix}.{stem}"


def model_key(config) -> str:
    """The model config the scales belong to."""
    return f"chroma-l{config.num_layers}-s{config.num_single_layers}-d{config.inner_dim}"


def weight_sums(weights) -> list[float]:
    """Absolute sums of the input and output projection weights: tell apart checkpoints of one model config."""
    return [
        float(np.abs(np.asarray(weights[name], dtype=np.float64)).sum())
        for name in ("x_embedder.weight", "proj_out.weight")
    ]


def reject_quantized_checkpoint(weights) -> None:
    """The build quantizes BF16 weights itself; an already quantized export is refused."""
    for name in sorted(weights.names()):
        if name.endswith(_QUANTIZED_SUFFIXES) or weights.dtype(name) in _QUANTIZED_DTYPES:
            raise ValueError(
                f"the chroma transformer checkpoint is already quantized ({name}); "
                "--quantization fp8/nvfp4 needs the original BF16 checkpoint"
            )


def validated_scales(scales, config) -> dict[str, float | None]:
    """``{projection: input_scale or None}`` covering exactly the projections of ``config``."""
    if not isinstance(scales, dict):
        raise TypeError("chroma activation scales must be a dictionary")
    expected = projection_names(config.num_layers, config.num_single_layers)
    missing = sorted(set(expected) - set(scales))
    unexpected = sorted(set(scales) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"chroma activation scales do not match the model: missing={missing[:4]}, "
            f"unexpected={unexpected[:4]}"
        )
    result: dict[str, float | None] = {}
    for name in expected:
        value = scales[name]
        if value is None:
            result[name] = None
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"chroma activation scale for {name} must be positive and finite or null"
            )
        result[name] = float(value)
    return result


def load_scales(config, weights, path: Path = SCALES_PATH) -> dict[str, float | None]:
    """The packaged FP8 activation scales for this checkpoint, or a clear error."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key, sums = model_key(config), weight_sums(weights)
    packaged = payload.get("weight_abs_sums")
    same_weights = (
        isinstance(packaged, list)
        and len(packaged) == len(sums)
        and all(
            isinstance(value, (int, float)) and math.isclose(value, actual, rel_tol=1e-9)
            for value, actual in zip(packaged, sums)
        )
    )
    if payload.get("model") != key or not same_weights:
        raise ValueError(
            f"no chroma activation scales for model {key} with weight sums {sums} (the packaged scales are for "
            f"{payload.get('model')} with weight sums {packaged}); run families/chroma/calibrate_scales.py for "
            "this checkpoint"
        )
    return validated_scales(payload.get("scales"), config)


def quantized_projections(quantization: str, config, weights, scales: dict | None = None) -> dict:
    """``{projection: (format, FP8 activation scale)}`` for ``build_transformer_network``."""
    if quantization not in FORMATS:
        raise ValueError(f"chroma quantization must be one of {FORMATS}, not {quantization!r}")
    reject_quantized_checkpoint(weights)
    scales = load_scales(config, weights) if scales is None else validated_scales(scales, config)
    result = {}
    for name, scale in scales.items():
        if scale is None:
            continue
        nvfp4 = quantization == "nvfp4" and projection_kind(name) in NVFP4_KINDS
        result[name] = ("nvfp4" if nvfp4 else "fp8", scale)
    return result


def fp8_weight(weight) -> tuple[np.ndarray, float]:
    """Per-tensor FP8 e4m3 ``[out, in]`` weight and its scale ``amax / 448``."""
    import ml_dtypes

    values = np.asarray(weight, dtype=np.float32)
    amax = float(np.max(np.abs(values)))
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError("FP8 weight must have a positive finite absolute maximum")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(values / np.float32(scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return np.ascontiguousarray(scaled.astype(ml_dtypes.float8_e4m3fn)), scale


def _e2m1_codes(scaled):
    """Sign-magnitude E2M1 codes (0..15) of already block-scaled values."""
    import torch

    magnitude = scaled.abs()
    code = torch.zeros(scaled.shape, dtype=torch.uint8, device=scaled.device)
    for midpoint, up in zip(_MIDPOINTS, _MIDPOINT_ROUNDS_UP, strict=True):
        code += (magnitude >= midpoint if up else magnitude > midpoint).to(torch.uint8)
    return code | ((scaled < 0).to(torch.uint8) << 3)


def quantize_nvfp4_weight(weight) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize one ``[out, in]`` weight to NVFP4.

    Returns the packed bytes ``[out, in / 2]`` (first value in the low nibble), the raw e4m3
    block-scale bytes ``[out, in / 16]`` and ``weight_scale_2``. An all-zero block gets block
    scale 1.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.as_tensor(np.asarray(weight, dtype=np.float32)).to(device)
    if tensor.ndim != 2:
        raise ValueError(f"expected a [out, in] weight, got shape {tuple(tensor.shape)}")
    out_features, in_features = tensor.shape
    if in_features % NVFP4_BLOCK_SIZE:
        raise ValueError(f"in_features={in_features} is not a multiple of {NVFP4_BLOCK_SIZE}")
    amax = float(tensor.abs().max())
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError("NVFP4 weight must have a positive finite absolute maximum")
    weight_scale_2 = float(np.float32(amax) * np.float32(1.0 / NVFP4_MAX))
    blocks = tensor.reshape(out_features, in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_scale = (blocks.abs().amax(-1) / E2M1_MAX / weight_scale_2).clamp_(max=FP8_E4M3_MAX)
    scale_float = block_scale.to(torch.float8_e4m3fn).float()
    scale_float = torch.where(scale_float == 0, torch.ones_like(scale_float), scale_float)
    scaled = blocks / (scale_float.unsqueeze(-1) * weight_scale_2)
    code = _e2m1_codes(scaled).reshape(out_features, in_features)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().cpu().numpy()
    scales = scale_float.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales), weight_scale_2


def dequantize_nvfp4_weight(packed, block_scale, weight_scale_2: float) -> np.ndarray:
    """The float32 ``[out, in]`` values the NVFP4 bytes stand for."""
    import torch

    packed_t = torch.from_numpy(np.ascontiguousarray(packed))
    codes = torch.stack([packed_t & 0xF, packed_t >> 4], dim=-1).reshape(packed_t.shape[0], -1)
    values = torch.tensor(_E2M1_GRID)[(codes & 7).long()] * torch.where(codes >= 8, -1.0, 1.0)
    scale = torch.from_numpy(np.ascontiguousarray(block_scale)).view(torch.float8_e4m3fn).float()
    result = values * scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1) * weight_scale_2
    return result.numpy()


def quantized_linear(graph, x, weight, bias, fmt: str, input_scale: float):
    """``x @ weight.T + bias`` with FP8 or NVFP4 operands; BF16 output and bias as in ``Graph.linear``.

    ``x`` is ``[1, rows, in]`` BF16; ``weight`` is the ``[out, in]`` checkpoint tensor.
    """
    import tensorrt as trt

    if fmt not in FORMATS:
        raise ValueError(f"unknown quantization format {fmt!r}")
    if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("activation scale must be positive and finite")
    network = graph.network
    _, rows, in_features = x.shape
    out_features = int(weight.shape[0])
    if int(weight.shape[1]) != in_features:
        raise ValueError(f"linear input width {in_features} != weight {weight.shape[1]}")
    compute = x.dtype
    flat = graph.reshape(x, (rows, in_features))
    if fmt == "fp8":
        data, weight_scale = fp8_weight(weight)
        graph._owners.append(data)
        scale = graph.constant(np.asarray([input_scale], dtype=np.float32), trt.float32)
        quantized = network.add_quantize(flat, scale, trt.DataType.FP8).get_output(0)
        activation = network.add_dequantize(quantized, scale, compute).get_output(0)
        constant = network.add_constant(
            (out_features, in_features), trt.Weights(trt.DataType.FP8, data.ctypes.data, data.size)
        ).get_output(0)
        w_scale = graph.constant(np.asarray([weight_scale], dtype=np.float32), trt.float32)
        matrix = network.add_dequantize(constant, w_scale, compute).get_output(0)
    else:
        global_scale = input_scale / E2M1_MAX
        packed, block_scale, weight_scale_2 = quantize_nvfp4_weight(weight)
        graph._owners += [packed, block_scale]
        constant = network.add_constant(
            (out_features, in_features),
            trt.Weights(trt.DataType.FP4, packed.ctypes.data, out_features * in_features),
        ).get_output(0)
        block = network.add_constant(
            (out_features, in_features // NVFP4_BLOCK_SIZE),
            trt.Weights(trt.DataType.FP8, block_scale.ctypes.data, block_scale.size),
        ).get_output(0)
        w_scale_2 = graph.constant(np.asarray([weight_scale_2], dtype=np.float32), trt.float32)
        w_scale = network.add_dequantize(block, w_scale_2, trt.float32).get_output(0)
        w_dequant = network.add_dequantize(constant, w_scale, trt.float32)
        w_dequant.axis = 1
        matrix = graph.cast(w_dequant.get_output(0), compute)
        a_scale = graph.constant(np.asarray([global_scale], dtype=np.float32), trt.float32)
        dynamic = network.add_dynamic_quantize(
            flat, 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
        )
        dynamic.set_input(1, a_scale)
        a_block = network.add_dequantize(
            dynamic.get_output(1), graph.cast(a_scale, compute), compute
        ).get_output(0)
        a_dequant = network.add_dequantize(dynamic.get_output(0), a_block, compute)
        a_dequant.axis = 1
        activation = a_dequant.get_output(0)
    product = network.add_matrix_multiply(
        activation, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    product = graph.reshape(product, (1, rows, out_features))
    if bias is None:
        return product
    return graph.add(product, graph.constant(np.asarray(bias).reshape(1, 1, out_features), compute))
