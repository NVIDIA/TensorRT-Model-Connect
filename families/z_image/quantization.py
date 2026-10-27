# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image-owned FP8 self-quantization of the DiT linear layers.

``quantization="fp8"`` quantizes the original BF16 transformer at build time; no FP8
checkpoint is needed or written. Weights are quantized one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8
bytes are fed to the engine as a constant. The activation scale of each projection input
needs real denoising activations, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``. Users do not run calibration. A projection with no entry
in that file stays in fp16.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize -> MatMul -> cast to fp16
    FP8 weight -> Dequantize --------------^

Quantized projections, for every DiT block ``<block>`` (``noise_refiner.N``,
``context_refiner.N`` and ``main.N``): the SwiGLU gate and up linears ``ff_w1`` and ``ff_w3``.
A block built in fp32 (``fp32_layers``) is not quantized.

Everything else stays in fp16, for these measured reasons (see the PR):

  ``to_out``            TensorRT 11.4 cannot compile an FP8 Quantize on the output of the fused
                        attention (``element type (fp8_e4m3) differs from input 1's element type
                        (half)``), also for a tiny model. TensorRT 11.1 builds it either way.
  ``ff_w2``             Its input, the gated SwiGLU product, has a very heavy tail (the calibrated
                        maximum is about 9000 for the median block and 340000 at most), so one
                        8-bit scale per tensor loses the bulk of the values; it cost the most
                        image quality of all projections.
  ``to_q/to_k/to_v``    Quantizing them also cost more image quality than gate and up.

The AdaLN modulation linears, the patch, caption, timestep and final projections, the text
encoder and the VAE stay as they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"

_SUPPORTED = ("fp8",)
# Build-route knob that turns off TensorRT's fused dual-GEMM kernel.
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"

# Projections inside one DiT block (engine weight names are ``<block>.<stem>``).
BLOCK_PROJECTIONS = ("ff_w1", "ff_w3")


def scale_key(dim: int, num_layers: int, num_refiner_layers: int, ffn_dim: int) -> str:
    """Key of this model's entry in the activation-scale file."""
    return f"z_image-h{int(dim)}-l{int(num_layers)}-r{int(num_refiner_layers)}-i{int(ffn_dim)}"


def block_prefixes(num_layers: int, num_refiner_layers: int) -> list[str]:
    """Engine prefixes of every DiT block."""
    return (
        [f"noise_refiner.{i}" for i in range(num_refiner_layers)]
        + [f"context_refiner.{i}" for i in range(num_refiner_layers)]
        + [f"main.{i}" for i in range(num_layers)]
    )


def quantized_names(num_layers: int, num_refiner_layers: int) -> list[str]:
    """Names (``<block>.<stem>``) of every projection that can be quantized."""
    return [
        f"{block}.{stem}"
        for block in block_prefixes(num_layers, num_refiner_layers)
        for stem in BLOCK_PROJECTIONS
    ]


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is
    the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(
    dim: int, num_layers: int, num_refiner_layers: int, ffn_dim: int
) -> dict[str, float]:
    """Return the committed activation scales for this model shape, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Z-Image FP8 build requires {path}, which was not found") from error
    key = scale_key(dim, num_layers, num_refiner_layers, ffn_dim)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this model shape."
        )
    return table[key]


def _scale_const(network, value: float):
    """One-element fp32 constant holding a Q/DQ scale."""
    arr = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(arr)).get_output(0)


@dataclass
class ZImageQuantContext:
    """Routes the quantized projections through FP8 Q/DQ; others stay in fp16."""

    scales: dict[str, float]
    keep_alive: list = field(default_factory=list)

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection is FP8 (selects the FP8 build options)."""
        return bool(self.scales)

    def should_quantize(self, name: str) -> bool:
        """True when ``name`` (for example ``main.3.to_k``) has an activation scale."""
        return name in self.scales

    def matmul(self, network, x, w_io: np.ndarray, name: str):
        """``x @ w_io`` for ``x`` of shape ``[seq, in]`` with an FP8 weight, in float32.

        ``w_io`` is the engine's ``[in, out]`` weight (``ff_w3`` carries the fp16 pre-scale of
        the SwiGLU branch, which only changes its weight scale). The Dequantize layers and the
        matmul run in float32 and the result is cast back to the activation type. With 16-bit
        Dequantize outputs, TensorRT's FP8 GEMM keeps the raw accumulator in 16 bits and
        applies the scales afterwards, so it overflows once an output exceeds about
        ``65504 * input_scale * weight_scale``.
        """
        in_dim, out_dim = w_io.shape
        packed, weight_scale = quantize_fp8_weight(np.asarray(w_io, dtype=np.float32))
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(packed)
        w_const = network.add_constant(
            (in_dim, out_dim),
            trt.Weights(trt.DataType.FP8, packed.ctypes.data, in_dim * out_dim),
        ).get_output(0)
        w_dequant = network.add_dequantize(
            w_const, _scale_const(network, weight_scale), trt.float32
        )

        act_scale = _scale_const(network, self.scales[name])
        a_quant = network.add_quantize(x, act_scale, trt.DataType.FP8)
        a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

        out = network.add_matrix_multiply(
            a_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            w_dequant.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        return network.add_cast(out, x.dtype).get_output(0)


def configure_build(trt_config) -> None:
    """Apply the FP8 build options and keep layer names for kernel inspection."""
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
    # The route is set only when this TensorRT build lists the knob.
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def calibrate_z_image(
    weights,
    *,
    dim: int,
    num_layers: int,
    num_refiner_layers: int,
    ffn_dim: int,
    quantization: str,
    quantized_checkpoint: bool = False,
) -> ZImageQuantContext:
    """Pair the committed activation scales with the checkpoint, after validating it."""
    if quantization not in _SUPPORTED:
        raise NotImplementedError(
            f"Z-Image supports quantization {list(_SUPPORTED)} or none; got {quantization!r}"
        )
    if quantized_checkpoint:
        raise ValueError(
            "the transformer is already quantized; fp8 self-quantization needs the original checkpoint"
        )
    activation_scales = load_activation_scales(dim, num_layers, num_refiner_layers, ffn_dim)
    scales: dict[str, float] = {}
    for name in quantized_names(num_layers, num_refiner_layers):
        scale = activation_scales.get(name)
        if scale is None:
            continue
        if name not in weights:
            raise ValueError(f"checkpoint has no weight for {name}")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        scales[name] = float(scale)
    if not scales:
        raise RuntimeError("Z-Image FP8 self-quantization matched no projection in the checkpoint")
    return ZImageQuantContext(scales=scales)


def is_quantized_checkpoint(readers) -> bool:
    """True when the transformer checkpoint stores quantization scales."""
    return any(str(name).endswith((".weight_scale", ".input_scale")) for name in readers.tensor_map)
