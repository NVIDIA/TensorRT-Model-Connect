# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot GR00T N1.7 FP8 self-quantization of the action-head DiT projections.

``quantization="fp8"`` quantizes the original checkpoint at build time, because no FP8
GR00T checkpoint exists. Weights are quantized one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and the FP8
bytes are fed to the engine as a constant. FP8 weights are never stored. The activation
scale of each projection input needs real observations, so it was calibrated once offline
with ``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``, max calibration, LIBERO
observations through the LeRobot policy and all its denoising steps) and is read from
``fp8_activation_scales.json``. Users do not run calibration.

Each quantized projection becomes::

    activation (bf16) -> Quantize -> Dequantize (bf16) -> MatMul (bf16)
    FP8 weight -> Dequantize (bf16) ----------------------^

The build runs in bfloat16, so the Dequantize layers output bfloat16: the GEMM keeps the bias
in its epilogue and there is no fp16 overflow to guard against.

Quantized, in every one of the 32 DiT blocks ``action_head.model.transformer_blocks.N``:
``attn1.to_q``, ``attn1.to_k``, ``attn1.to_v`` (in the cross-attention blocks ``to_k`` and
``to_v`` read the vision-language tokens and run once per call in the condition engine),
``attn1.to_out.0``, ``ff.net.0.proj`` and ``ff.net.2``. The attention is built from explicit
matmul and softmax layers, so the attention output projection compiles as FP8 on TensorRT
11.1 and 11.4.

Everything else stays in the build precision, measured (see the README): the Qwen3-VL text
layers and the four VL self-attention blocks (FP8 there adds 1% to 2% to the step speed-up
and doubles the mean action error), the vision tower and DeepStack mergers, the embedding
table, the norms, the attention matmuls, the timestep embedders and the AdaLN modulation
(one-row GEMMs), the DiT output projections and the embodiment-specific state/action
encoders and decoder.

A projection whose calibrated input scale is more than 20 times the median of its kind has
a massive activation: its entry in the scale file is ``null`` and it stays in bfloat16 on
purpose. A projection missing from the file is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ml_dtypes
import numpy as np

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
_SUPPORTED = ("fp8",)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"

BACKBONE_ROOT = "backbone.model.model.language_model.layers"
DIT_ROOT = "action_head.model.transformer_blocks"
# Quantized projections of one DiT block.
QUANTIZED_STEMS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2")


def projection_names(dit_layers: int) -> list[str]:
    """Checkpoint module names of every quantized projection of this model config."""
    return [f"{DIT_ROOT}.{index}.{stem}" for index in range(dit_layers) for stem in QUANTIZED_STEMS]


def projection_kind(name: str, cross_blocks: tuple[int, ...]) -> str:
    """The group a projection's scale is compared within: the stem, and whether the block is a
    cross-attention block (its keys/values read the VL tokens, its queries the action tokens)."""
    if not name.startswith(DIT_ROOT + "."):
        raise ValueError(f"{name} is not a quantized GR00T projection")
    index, stem = name[len(DIT_ROOT) + 1:].split(".", 1)
    if stem not in QUANTIZED_STEMS:
        raise ValueError(f"{name} is not a quantized GR00T projection")
    return f"{stem}.{'cross' if int(index) in cross_blocks else 'self'}"


def _abs_sum(array) -> float:
    return float(np.abs(np.asarray(array, dtype=np.float32)).sum(dtype=np.float64))


def scale_key(weights, *, backbone_layers: int, embodiment_id: int) -> str:
    """Key of this checkpoint's entry in the activation-scale file.

    The architecture is fixed (Cosmos-Reason2-2B backbone, 4 VL blocks, 32 DiT blocks), so the
    key holds what varies between ``lerobot-train`` outputs: the kept backbone layers, the
    embodiment the bundle serves, and numeric weight sums (sum of absolute values, 5
    significant digits) of the first backbone query projection and the first DiT query
    projection, which a fine-tune changes.
    """
    return (
        f"groot-n17-l{int(backbone_layers)}-e{int(embodiment_id)}"
        f"-w{_abs_sum(weights[f'{BACKBONE_ROOT}.0.self_attn.q_proj.weight']):.4e}"
        f"-{_abs_sum(weights[f'{DIT_ROOT}.0.attn1.to_q.weight']):.4e}"
    )


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same scale is the
    dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(key: str) -> dict[str, float | None]:
    """The committed activation scales of ``key``; raises when the checkpoint was never calibrated.

    An entry of ``None`` marks a projection the calibration left unquantized on purpose.
    """
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"lerobot_groot FP8 build requires {path}, which was not found") from error
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: {sorted(table)}. "
            "Run families/lerobot_groot/calibrate_fp8_scales.py for this checkpoint."
        )
    return table[key]


@dataclass(frozen=True)
class FP8Projection:
    """One projection weight as raw fp8_e4m3 bytes ``[out, in]`` with its two static scales."""

    packed: np.ndarray
    weight_scale: float
    input_scale: float
    keep_alive: list = field(default_factory=list, compare=False, repr=False)

    def operands(self, network, activation, rank: int, dtype):
        """The dequantized activation and weight for ``activation @ weight^T`` in ``dtype``."""
        import tensorrt as trt

        out_features, in_features = self.packed.shape
        # trt.Weights holds a raw pointer, so the buffer must outlive the build.
        self.keep_alive.append(self.packed)
        weight = network.add_constant(
            (1,) * (rank - 2) + (out_features, in_features),
            trt.Weights(trt.DataType.FP8, self.packed.ctypes.data, out_features * in_features),
        ).get_output(0)
        weight = network.add_dequantize(weight, _scale(network, self.weight_scale), dtype).get_output(0)
        input_scale = _scale(network, self.input_scale)
        quantized = network.add_quantize(activation, input_scale, trt.DataType.FP8).get_output(0)
        return network.add_dequantize(quantized, input_scale, dtype).get_output(0), weight


def _scale(network, value: float):
    """One-element float32 constant holding a Q/DQ scale."""
    import tensorrt as trt

    array = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(array)).get_output(0)


@dataclass
class GrootQuantContext:
    """The FP8 projections by checkpoint module name; every other projection stays unquantized."""

    projections: dict[str, FP8Projection]

    def get(self, name: str) -> FP8Projection | None:
        """The FP8 projection named ``name`` (for example ``...transformer_blocks.3.ff.net.2``), or None."""
        return self.projections.get(name)


def fp8(quant: GrootQuantContext | None, name: str) -> FP8Projection | None:
    """``quant.get(name)``, or None for an unquantized build."""
    return None if quant is None else quant.get(name)


def configure_fp8_build(trt_config, quant: GrootQuantContext | None) -> None:
    """Keep layer names for kernel inspection and turn off TensorRT's dual-GEMM fusion."""
    if quant is None:
        return
    import tensorrt as trt

    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if _DUAL_GEMM_KNOB in getattr(trt_config, "all_build_routes", ""):
        # The fused gate/up kernel does not compile with scalar-scale FP8 inputs; the route is
        # set only when this TensorRT build lists the knob.
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def refuse_quantized_checkpoint(model_dir: Path) -> None:
    """Raise for a checkpoint that is already quantized: FP8 self-quantizes the original."""
    model_dir = Path(model_dir)
    try:
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        raw = {}
    markers = [name for name in ("hf_quant_config.json", "quantize_config.json") if (model_dir / name).is_file()]
    if (isinstance(raw, dict) and raw.get("quantization_config")) or markers:
        found = markers[0] if markers else "quantization_config in config.json"
        raise ValueError(f"the checkpoint is already quantized ({found}); FP8 self-quantization needs the original")


def calibrate_groot(weights, *, backbone_layers: int, dit_layers: int, embodiment_id: int, quantization: str,
                    precision: str) -> GrootQuantContext:
    """Quantize the loaded ``[out, in]`` weights and pair each with its stored activation scale.

    A projection with no stored scale, or a stored scale for a projection this model config
    does not have, is an error.
    """
    if quantization not in _SUPPORTED:
        raise NotImplementedError(f"lerobot_groot supports quantization {list(_SUPPORTED)} or none; got {quantization!r}")
    if precision != "bf16":
        raise ValueError(f"lerobot_groot FP8 needs precision bf16, got {precision!r}")
    if any(str(name).endswith((".weight_scale", ".input_scale")) for name in weights):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs the original")
    stored = load_activation_scales(scale_key(weights, backbone_layers=backbone_layers, embodiment_id=embodiment_id))
    names = projection_names(dit_layers)
    unexpected = sorted(set(stored) - set(names))
    if unexpected:
        raise ValueError(f"{ACTIVATION_SCALES_FILENAME} has scales for projections this model does not quantize, "
                         f"for example {unexpected[0]}; run calibrate_fp8_scales.py for this checkpoint")
    keep_alive: list = []
    projections: dict[str, FP8Projection] = {}
    for name in names:
        if name not in stored:
            raise ValueError(f"{name} has no activation scale in {ACTIVATION_SCALES_FILENAME}; run calibrate_fp8_scales.py")
        input_scale = stored[name]
        if input_scale is None:
            continue  # massive activation: left in bfloat16 by the calibration
        if not np.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(f"activation scale of {name} must be finite and positive")
        key = f"{name}.weight"
        if key not in weights:
            raise ValueError(f"checkpoint has no {key}")
        packed, weight_scale = quantize_fp8_weight(np.asarray(weights[key], dtype=np.float32))
        projections[name] = FP8Projection(packed, weight_scale, float(input_scale), keep_alive)
    if not projections:
        raise RuntimeError("lerobot_groot FP8 self-quantization matched no projection in the checkpoint")
    return GrootQuantContext(projections)
