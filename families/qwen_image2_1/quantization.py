# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image 2.1 denoiser quantization: which projections are FP8 or NVFP4, their scales, the GEMMs.

No ModelOpt FP8 or NVFP4 checkpoint of Qwen-Image 2.1 is published, so the original BF16
checkpoint is quantized at build time. Only the denoiser is quantized; the text encoder and the
VAE stay as they are.

FP8 W8A8 (``quantization="fp8"``): the projections of ``projection_names`` (query, key, value and
the three SwiGLU projections of every block). Per-tensor e4m3 weights (``weight_scale =
amax / 448``, computed at build time) and a static per-tensor activation scale from
``data/qwen-image-2.1-scales.json``.

NVFP4 W4A4 (``quantization="nvfp4"``): the projections of ``NVFP4_KINDS`` become packed E2M1
weights with one e4m3 scale per 16 inputs and one ``weight_scale_2 = amax / (6 * 448)`` per
tensor; the activation is block-quantized at run time (``add_dynamic_quantize``, block 16) with a
static global scale equal to the FP8 activation scale divided by 6, so both formats use one
calibration. The other projections of the FP8 recipe stay FP8.

Stay BF16 in both: the attention output projection (TensorRT 11.4 cannot compile an FP8 Quantize
on the output of the fused attention), the text and latent input projections, the timestep
embedder, the shared modulation, the output head, the norms and the attention itself. A
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
SCALES_PATH = Path(__file__).with_name("data") / "qwen-image-2.1-scales.json"
# Tensor names and stored dtypes that only a quantized export carries.
_QUANTIZED_SUFFIXES = (
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "scale_weight",
    "scale_input",
)
_QUANTIZED_DTYPES = ("float8_e4m3fn", "float8_e5m2", "uint8", "int8")

BLOCK_STEMS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)
# NVFP4 projections: the two SwiGLU input projections. The SwiGLU output projection and the
# attention projections stay FP8 (see the family README for the measured alternatives).
NVFP4_KINDS = ("img_mlp.gate_layer", "img_mlp.proj")

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between neighbouring E2M1 values. A value exactly on a midpoint goes to the
# neighbour with an even code: the first, third, fifth and seventh midpoint round down.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MIDPOINT_ROUNDS_UP = (False, True, False, True, False, True, False)


def projection_names(num_layers: int) -> tuple[str, ...]:
    """Every quantized denoiser projection (the FP8 recipe), in graph order."""
    return tuple(
        f"transformer_blocks.{index}.{stem}" for index in range(num_layers) for stem in BLOCK_STEMS
    )


def projection_kind(name: str) -> str:
    """The stem shared by one projection in every block (``img_mlp.out``)."""
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "transformer_blocks" or not parts[1].isdigit():
        raise ValueError(f"not a block projection: {name}")
    return parts[2]


def model_key(config) -> str:
    """The model config the scales belong to."""
    return f"qwen_image2_1-l{config.num_layers}-h{config.hidden}-r{config.mlp_ratio}"


def weight_sums(weights) -> list[float]:
    """Absolute sums of the input and output projection weights: tell apart checkpoints of one model config."""
    return [
        float(np.abs(np.asarray(weights[name], dtype=np.float64)).sum())
        for name in ("img_in.weight", "proj_out.weight")
    ]


def reject_quantized_checkpoint(weights: dict) -> None:
    """The build quantizes BF16 weights itself; an already quantized export is refused."""
    for name in sorted(weights):
        dtype = str(np.asarray(weights[name]).dtype)
        if name.endswith(_QUANTIZED_SUFFIXES) or dtype in _QUANTIZED_DTYPES:
            raise ValueError(
                f"the qwen_image2_1 transformer checkpoint is already quantized ({name}); "
                "--quantization fp8/nvfp4 needs the original BF16 checkpoint"
            )


def validated_scales(scales, config) -> dict[str, float | None]:
    """``{projection: input_scale or None}`` covering exactly the projections of ``config``."""
    if not isinstance(scales, dict):
        raise TypeError("qwen_image2_1 activation scales must be a dictionary")
    expected = projection_names(config.num_layers)
    missing = sorted(set(expected) - set(scales))
    unexpected = sorted(set(scales) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "qwen_image2_1 activation scales do not match the model: "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
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
                f"qwen_image2_1 activation scale for {name} must be positive and finite or null"
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
            f"no qwen_image2_1 activation scales for model {key} with weight sums {sums} (the packaged "
            f"scales are for {payload.get('model')} with weight sums {packaged}); run "
            "families/qwen_image2_1/calibrate_scales.py for this checkpoint"
        )
    return validated_scales(payload.get("scales"), config)


def quantized_projections(quantization: str, config, weights, scales: dict | None = None) -> dict:
    """``{projection: (format, FP8 activation scale)}`` for ``build_denoiser_plan``."""
    if quantization not in FORMATS:
        raise ValueError(
            f"qwen_image2_1 quantization must be one of {FORMATS}, not {quantization!r}"
        )
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


def configure_build(graph) -> None:
    """Keep layer names in quantized plans so the kernels can be inspected."""
    import tensorrt as trt

    graph.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED


def _scalar(graph, value: float):
    import tensorrt as trt

    array = np.asarray([value], dtype=np.float32)
    graph.anchors.append(array)
    return graph.network.add_constant((1,), trt.Weights(array)).get_output(0)


def _activation(graph, x, fmt: str, input_scale: float, cache: dict | None):
    """Quantize-dequantize of ``[1, rows, in]`` as ``[rows, in]``; one per input tensor, format and scale."""
    import tensorrt as trt

    key = (x.name, fmt, input_scale)
    if cache is not None and key in cache:
        return cache[key]
    network = graph.network
    compute = x.dtype
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (int(x.shape[1]), int(x.shape[2]))
    flat = shuffle.get_output(0)
    if fmt == "fp8":
        scale = _scalar(graph, input_scale)
        quantized = network.add_quantize(flat, scale, trt.DataType.FP8).get_output(0)
        activation = network.add_dequantize(quantized, scale, compute).get_output(0)
    else:
        global_scale = _scalar(graph, input_scale / E2M1_MAX)
        dynamic = network.add_dynamic_quantize(
            flat, 1, NVFP4_BLOCK_SIZE, trt.DataType.FP4, trt.DataType.FP8
        )
        dynamic.set_input(1, global_scale)
        block = network.add_dequantize(
            dynamic.get_output(1), network.add_cast(global_scale, compute).get_output(0), compute
        ).get_output(0)
        dequantized = network.add_dequantize(dynamic.get_output(0), block, compute)
        dequantized.axis = 1
        activation = dequantized.get_output(0)
    if cache is not None:
        cache[key] = activation
    return activation


def _weight(graph, weight, fmt: str, compute):
    """The dequantized ``[out, in]`` FP8 or NVFP4 weight in the compute type."""
    import tensorrt as trt

    network = graph.network
    out_features, in_features = (int(dim) for dim in weight.shape)
    if fmt == "fp8":
        data, weight_scale = fp8_weight(weight)
        graph.anchors.append(data)
        constant = network.add_constant(
            (out_features, in_features), trt.Weights(trt.DataType.FP8, data.ctypes.data, data.size)
        ).get_output(0)
        return network.add_dequantize(constant, _scalar(graph, weight_scale), compute).get_output(0)
    packed, block_scale, weight_scale_2 = quantize_nvfp4_weight(weight)
    graph.anchors += [packed, block_scale]
    constant = network.add_constant(
        (out_features, in_features),
        trt.Weights(trt.DataType.FP4, packed.ctypes.data, out_features * in_features),
    ).get_output(0)
    block = network.add_constant(
        (out_features, in_features // NVFP4_BLOCK_SIZE),
        trt.Weights(trt.DataType.FP8, block_scale.ctypes.data, block_scale.size),
    ).get_output(0)
    scale = network.add_dequantize(block, _scalar(graph, weight_scale_2), trt.float32).get_output(0)
    dequantized = network.add_dequantize(constant, scale, trt.float32)
    dequantized.axis = 1
    return network.add_cast(dequantized.get_output(0), compute).get_output(0)


def quantized_linear(graph, x, weight, fmt: str, input_scale: float, cache: dict | None = None):
    """``x @ weight.T`` with FP8 or NVFP4 operands; the output in the compute type like ``trt_ops.linear``.

    ``x`` is ``[1, rows, in]`` in the graph's compute type (BF16); ``weight`` is the ``[out, in]``
    checkpoint tensor. ``cache`` shares one activation quantization between projections that
    read the same tensor.
    """
    import tensorrt as trt

    if fmt not in FORMATS:
        raise ValueError(f"unknown quantization format {fmt!r}")
    if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("activation scale must be positive and finite")
    if x.dtype != trt.bfloat16:
        raise ValueError("quantized projections need a BF16 graph")
    _, rows, in_features = (int(dim) for dim in x.shape)
    out_features = int(weight.shape[0])
    if int(weight.shape[1]) != in_features:
        raise ValueError(f"linear input width {in_features} != weight {weight.shape[1]}")
    network = graph.network
    activation = _activation(graph, x, fmt, input_scale, cache)
    matrix = _weight(graph, weight, fmt, x.dtype)
    product = network.add_matrix_multiply(
        activation, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    shuffle = network.add_shuffle(product)
    shuffle.reshape_dims = (1, rows, out_features)
    return shuffle.get_output(0)
