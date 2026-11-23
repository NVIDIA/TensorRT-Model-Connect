# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2-owned FP8 and INT4-AWQ self-quantization and TensorRT Q/DQ graph context.

The original BF16 LFM2 checkpoint is used as-is; no FP8 or INT4 checkpoint is
needed. LiquidAI publishes no FP8, NVFP4 or ModelOpt INT4 LFM2 checkpoint.

Weights are quantized on the fly at build time, one projection at a time:
``scale = max(abs(weight)) / 448`` (448 is the largest finite e4m3 value), and
the FP8 bytes are fed to the engine as a constant. The activation scale of each
projection input needs real data, so it was calibrated once offline with
``calibrate_fp8_scales.py`` (ModelOpt ``FP8_DEFAULT_CFG``) and is read from
``fp8_activation_scales.json``, keyed by model shape and checkpoint digest. Users do
not run calibration.

Each quantized projection becomes::

    activation -> Quantize -> Dequantize(fp32) -> MatMul(fp32) -> Cast
    FP8 weight -> Dequantize(fp32) ------------------^

The Dequantize and the matmul run in fp32: a fp16 Dequantize makes the fused
FP8 GEMM overflow to NaN when a projection output is large.

Quantized: the attention q/k/v projections, the SwiGLU w1/w2/w3 projections and
the conv-block in/out projections that surround the depthwise short convolution.
Not quantized: the depthwise short convolution and its state, the attention
output projection (TensorRT 11.4 cannot compile an FP8 Quantize right after a
fused attention), ``lm_head``, the embedding and the norms. A projection with no
scale entry in the JSON file stays unquantized.

INT4-AWQ (W4A16) follows the llama family recipe: ``calibrate_awq_scales.py``
runs ModelOpt ``INT4_AWQ_CFG`` (``awq_lite``, no weight-clipping search) once
per checkpoint and stores only the per-projection activation smoothing
``pre_quant_scale`` under ``awq_scales/``, keyed by float64 sums of checkpoint values.
The build divides each weight by its smoothing scale, takes group-128 block
scales as ``amax / 7`` and rounds to signed INT4, which is the weight ModelOpt's
``awq_lite`` fake quantization evaluates. The engine multiplies the activation
by the smoothing scale and runs INT4 constant -> Dequantize -> MatMul with
16-bit activations, at builder optimization level 3 so the dequantization fuses
into the GEMM. ``AWQ_PROJECTIONS`` lists the quantized projections: the same set
as FP8 (conv in/out, attention q/k/v, SwiGLU w1/w2/w3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np
import tensorrt as trt

FP8_E4M3_MAX = 448.0
ACTIVATION_SCALES_FILENAME = "fp8_activation_scales.json"
INT4_AWQ_BLOCK_SIZE = 128
INT4_MAX = 7
AWQ_SCALES_DIR = Path(__file__).parent / "awq_scales"
# Small tensors whose values identify the checkpoint an AWQ scale file belongs to.
AWQ_IDENTITY_TENSORS = ("model.embedding_norm.weight", "model.layers.0.operator_norm.weight")
# Leading values of every projection's first row, also part of the identity, so a
# checkpoint that changes any projection (a fine-tune with the same norms) never matches.
AWQ_PROJECTION_IDENTITY_VALUES = 64

# HF projection stem -> weight-dict stem used by model.py.
PROJECTIONS = (
    ("conv.in_proj", "conv_in"),
    ("conv.out_proj", "conv_out"),
    ("self_attn.q_proj", "w_q"),
    ("self_attn.k_proj", "w_k"),
    ("self_attn.v_proj", "w_v"),
    ("feed_forward.w1", "w1"),
    ("feed_forward.w3", "w3"),
    ("feed_forward.w2", "w2"),
)
# Every linear projection of an LFM2 block; the AWQ calibration tool stores a smoothing
# scale for each, and the build quantizes the subset in AWQ_PROJECTIONS. INT4 can quantize
# the attention out_proj (no activation Quantize follows the fused attention), but on
# LFM2-1.2B that raises the KL to HF by 9% for no measurable decode gain, so it stays 16 bit
# as in the FP8 recipe.
ALL_PROJECTIONS = PROJECTIONS + (("self_attn.out_proj", "w_o"),)
AWQ_PROJECTIONS = tuple(stem for _, stem in PROJECTIONS)
_DUAL_GEMM_KNOB = "peep:match_dual_gemm"


def scale_key(config, final_norm: np.ndarray) -> str:
    """Key of this checkpoint's entry in the activation-scale file.

    ``config`` is the validated ``DenseLfm2Config``. LFM2 and LFM2.5 releases (and
    their fine-tunes) share model shapes but not activation ranges (scales differ
    by up to 4x between LFM2-1.2B and LFM2.5-1.2B-Instruct), so the key adds the sum
    of the final norm weight, which tells the checkpoints apart.
    """
    norm_sum = float(np.sum(np.asarray(final_norm, dtype=np.float32), dtype=np.float64))
    return (
        f"lfm2-h{int(config.hidden_size)}"
        f"-l{int(config.num_hidden_layers)}"
        f"-i{int(config.intermediate_size)}"
        f"-n{norm_sum:.4f}"
    )


@dataclass(frozen=True)
class _FP8Weight:
    """One projection weight stored as raw fp8_e4m3 bytes with one per-tensor scale."""

    packed: np.ndarray  # [out_features, in_features] raw fp8_e4m3 bytes
    weight_scale: float
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _AWQWeight:
    """INT4 weight quantized at build time from the original checkpoint."""

    packed: np.ndarray  # [out_features, in_features // 2] uint8, two codes per byte along in
    block_scale: np.ndarray  # [out_features, in_features // 128] float16
    pre_quant_scale: np.ndarray  # [in_features] float16, multiplies the activation
    out_features: int
    in_features: int


@dataclass(frozen=True)
class _LayerScales:
    """Activation scale (None for weight-only INT4-AWQ) and weight of one quantized projection."""

    input_scale: float | None
    weight: _FP8Weight | _AWQWeight


def _scale_const(network, value: float):
    """Add a one-element fp32 scale constant to the network."""
    array = np.asarray([value], dtype=np.float32)
    return network.add_constant((1,), trt.Weights(array)).get_output(0)


def _wrap_fp8_matmul(network, activation, scales: _LayerScales, *, keep_alive: list, name: str):
    """Add the FP8 Q/DQ matmul for one projection, with fp32 Dequantize and matmul."""
    w = scales.weight
    n, k = w.out_features, w.in_features

    # trt.Weights holds a raw pointer, so the buffer must outlive the build.
    keep_alive.append(w.packed)
    w_fp8 = trt.Weights(trt.DataType.FP8, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_fp8).get_output(0)
    w_dequant = network.add_dequantize(w_const, _scale_const(network, w.weight_scale), trt.float32)

    act_scale = _scale_const(network, scales.input_scale)
    a_quant = network.add_quantize(activation, act_scale, trt.DataType.FP8)
    a_dequant = network.add_dequantize(a_quant.get_output(0), act_scale, trt.float32)

    layer = network.add_matrix_multiply(
        a_dequant.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    layer.name = name
    output = layer.get_output(0)
    if output.dtype == activation.dtype:
        return output
    return network.add_cast(output, activation.dtype).get_output(0)


def _wrap_awq_matmul(network, activation, scales: _LayerScales, *, keep_alive: list, name: str):
    """Add ``(activation * pre_quant_scale) @ dequantize(int4_weight).T`` for one projection."""
    output_dtype = activation.dtype
    w = scales.weight
    n, k = w.out_features, w.in_features

    def half_const(values: np.ndarray, shape: tuple[int, ...]):
        array = np.ascontiguousarray(values, dtype=np.float16).reshape(shape)
        keep_alive.append(array)
        const = network.add_constant(shape, trt.Weights(array)).get_output(0)
        if const.dtype == output_dtype:
            return const
        return network.add_cast(const, output_dtype).get_output(0)

    keep_alive.append(w.packed)
    w_int4 = trt.Weights(trt.DataType.INT4, w.packed.ctypes.data, n * k)
    w_const = network.add_constant((n, k), w_int4).get_output(0)
    w_dequant = network.add_dequantize(
        w_const, half_const(w.block_scale, w.block_scale.shape), output_dtype
    )
    w_dequant.axis = 1

    rank = len(tuple(activation.shape))
    smooth = half_const(w.pre_quant_scale, (1,) * (rank - 1) + (k,))
    smoothed = network.add_elementwise(activation, smooth, trt.ElementWiseOperation.PROD)
    layer = network.add_matrix_multiply(
        smoothed.get_output(0),
        trt.MatrixOperation.NONE,
        w_dequant.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    layer.name = name
    return layer.get_output(0)


@dataclass(frozen=True)
class Lfm2QuantContext:
    """Routes the quantized projections through Q/DQ; others stay unquantized."""

    scales: dict[str, _LayerScales]
    keep_alive: list = None  # type: ignore[assignment]

    def __post_init__(self):
        """Give each context its own list for buffers that must outlive the build."""
        if self.keep_alive is None:
            object.__setattr__(self, "keep_alive", [])

    def should_quantize(self, weight_name: str) -> bool:
        """Return True when this weight has a calibrated activation scale."""
        return weight_name in self.scales

    @property
    def has_fp8_weights(self) -> bool:
        """True when any projection runs as an FP8 GEMM."""
        return any(isinstance(s.weight, _FP8Weight) for s in self.scales.values())

    @property
    def builder_optimization_level(self) -> int:
        """3 for INT4-AWQ, 1 otherwise. At level 1 TensorRT 11.4 dequantizes every INT4
        weight to 16 bits in a separate kernel on each step instead of fusing the
        dequantization into the GEMM, which makes decode slower than the unquantized engine."""
        return 3 if any(isinstance(s.weight, _AWQWeight) for s in self.scales.values()) else 1

    def quantized_matmul(self, network, lhs, weight_name: str, *, name: str):
        """Add the projection as an FP8 or INT4-AWQ matmul; the caller checks ``should_quantize``."""
        entry = self.scales[weight_name]
        wrap = _wrap_awq_matmul if isinstance(entry.weight, _AWQWeight) else _wrap_fp8_matmul
        return wrap(network, lhs, entry, keep_alive=self.keep_alive, name=name)


def configure_quantized_build(trt_config, quant_ctx) -> None:
    """Apply the FP8 / INT4-AWQ build options; a no-op for an unquantized build."""
    if quant_ctx is None:
        return
    trt_config.builder_optimization_level = quant_ctx.builder_optimization_level
    # Keeps per-layer tactic names in the plan so the quantized GEMMs can be inspected.
    trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # TensorRT's fused dual-GEMM (w1/w3) kernel fails to compile with scalar-scale FP8
    # inputs. The route is set only when this TensorRT build lists the knob.
    if quant_ctx.has_fp8_weights and _DUAL_GEMM_KNOB in getattr(
        trt_config, "all_build_routes", ""
    ):
        trt_config.build_route = f"-{_DUAL_GEMM_KNOB}=off"


def quantize_fp8_weight(weight_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize one weight to FP8 e4m3 with a single scalar scale.

    Returns the raw fp8 bytes and ``scale = max(abs(weight)) / 448``; the same
    scale is the dequantize scale in the engine.
    """
    amax = float(np.abs(weight_fp32).max())
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError("weight tensor has no finite positive amax to scale from")
    scale = amax / FP8_E4M3_MAX
    scaled = np.clip(weight_fp32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    packed = scaled.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.ascontiguousarray(packed), scale


def load_activation_scales(config, final_norm: np.ndarray) -> dict[str, float]:
    """Return the committed activation scales for this checkpoint, or raise when none exist."""
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"LFM2 FP8 build requires {path}, which was not found") from error
    key = scale_key(config, final_norm)
    if key not in table:
        raise NotImplementedError(
            f"no calibrated FP8 activation scales for {key!r}; available: "
            f"{sorted(table)}. Run calibrate_fp8_scales.py for this checkpoint."
        )
    return table[key]


def calibrate_lfm2_fp8(config, weights) -> Lfm2QuantContext:
    """Build the Q/DQ context from the loaded LFM2 weights (``[in, out]`` layout).

    ``config`` is the validated ``DenseLfm2Config``. Quantizes one projection at a
    time, so peak extra memory is one weight tensor.
    """
    if any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights):
        raise ValueError("the checkpoint is already quantized; FP8 self-quantization needs BF16")
    activation_scales = load_activation_scales(config, weights["final_norm"])

    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for _, weight_stem in PROJECTIONS:
            name = f"layer.{layer}.{weight_stem}"
            input_scale = activation_scales.get(name)
            if input_scale is None:
                continue
            if not np.isfinite(input_scale) or input_scale <= 0:
                raise ValueError(f"activation scale of {name} must be finite and positive")
            if name not in weights:
                raise ValueError(f"checkpoint has no {name}")
            transposed = np.asarray(weights[name])
            in_features, out_features = transposed.shape
            packed, weight_scale = quantize_fp8_weight(
                np.ascontiguousarray(transposed.T, dtype=np.float32)
            )
            scales[name] = _LayerScales(
                input_scale=float(input_scale),
                weight=_FP8Weight(packed, weight_scale, out_features, in_features),
            )
    if not scales:
        raise RuntimeError("LFM2 FP8 self-quantization matched no projection in the checkpoint")
    return Lfm2QuantContext(scales=scales)


def quantize_awq_weight(weight: np.ndarray, pre_quant_scale: np.ndarray) -> _AWQWeight:
    """Quantize one ``[out, in]`` weight to group-128 signed INT4 after AWQ smoothing.

    The engine multiplies the activation by ``pre_quant_scale``, so the weight is
    divided by it first. Block scales are ``amax / 7`` rounded to float16, the
    precision the engine dequantizes with, and the codes are rounded with that
    same scale.
    """
    weight = np.asarray(weight, dtype=np.float32)
    out_features, in_features = weight.shape
    if in_features % INT4_AWQ_BLOCK_SIZE:
        raise ValueError(f"INT4-AWQ needs in_features divisible by {INT4_AWQ_BLOCK_SIZE}")
    smooth = np.asarray(pre_quant_scale, dtype=np.float16).reshape(-1)
    if smooth.shape != (in_features,) or not np.isfinite(smooth).all() or np.any(smooth <= 0):
        raise ValueError(f"pre_quant_scale must be {in_features} positive finite values")
    blocks = (weight / smooth.astype(np.float32)).reshape(
        out_features, in_features // INT4_AWQ_BLOCK_SIZE, INT4_AWQ_BLOCK_SIZE
    )
    block_scale = (np.abs(blocks).max(axis=2) / INT4_MAX).astype(np.float16)
    # An all-zero block quantizes to zero codes under any scale.
    block_scale[block_scale == 0] = 1.0
    codes = np.clip(np.rint(blocks / block_scale[..., None].astype(np.float32)), -8, 7)
    nibbles = codes.astype(np.int8).reshape(out_features, in_features).view(np.uint8) & 15
    packed = np.ascontiguousarray(nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
    return _AWQWeight(
        packed,
        np.ascontiguousarray(block_scale),
        np.ascontiguousarray(smooth),
        out_features,
        in_features,
    )


def _summary(values: np.ndarray) -> np.ndarray:
    """Sum and sum of squares of checkpoint values in float64: a checkpoint signature that
    stores no weights. Unchanged bytes always give the same two values."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.array([data.sum(), np.square(data).sum()], dtype=np.float64)


def awq_identity(readers) -> dict[str, np.ndarray]:
    """Signatures that identify an AWQ checkpoint, keyed as in the scale file: the norm
    tensors in full and the leading values of every block projection.

    The scale file stores float64 sums (``_summary``) instead of the values, so it carries
    no checkpoint weights. ``readers`` is the reader collection of ``checkpoint_mapper``.
    """
    tensor_map = readers.tensor_map
    identity = {}
    for key in AWQ_IDENTITY_TENSORS:
        if key not in tensor_map:
            raise ValueError(f"INT4-AWQ checkpoint identity needs {key}")
        identity[f"identity.{key}"] = _summary(tensor_map[key].get_tensor(key))
    suffixes = tuple(f".{stem}.weight" for stem, _ in ALL_PROJECTIONS)
    for key in sorted(k for k in tensor_map if k.startswith("model.layers.") and k.endswith(suffixes)):
        head = tensor_map[key].get_slice(key)[0:1, :AWQ_PROJECTION_IDENTITY_VALUES]
        identity[f"identity.{key}"] = _summary(head)
    return identity


def load_awq_scales(scales_dir: Path, identity: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The pre_quant_scale table calibrated for the checkpoint with this identity."""
    for path in sorted(Path(scales_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as table:
            if all(
                key in table.files and np.array_equal(table[key], value)
                for key, value in identity.items()
            ):
                return {name: table[name] for name in table.files if name.startswith("layer.")}
    raise ValueError(
        f"no INT4-AWQ scales in {scales_dir} match this checkpoint; "
        "run python -m families.lfm2.calibrate_awq_scales"
    )


def calibrate_lfm2_awq(
    config,
    weights,
    model_dir: Path,
    *,
    scales_dir: Path = AWQ_SCALES_DIR,
    projections: tuple[str, ...] | None = None,
) -> Lfm2QuantContext:
    """Quantize the block projections of the original checkpoint to INT4-AWQ.

    ``config`` is the validated ``DenseLfm2Config`` and ``weights`` the loaded
    ``[in, out]`` weights; ``projections`` defaults to ``AWQ_PROJECTIONS``. The
    smoothing scales come from the committed table whose identity matches this
    checkpoint.
    """
    from .checkpoint_mapper import _open_safetensors

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if (
        raw.get("quantization_config")
        or (model_dir / "hf_quant_config.json").is_file()
        or any(str(key).endswith((".weight_scale", ".input_scale")) for key in weights)
    ):
        raise ValueError(
            f"lfm2 int4_awq quantizes the original 16-bit checkpoint; {model_dir} is already "
            "quantized (quantization_config, hf_quant_config.json or scale tensors)"
        )
    selected = AWQ_PROJECTIONS if projections is None else tuple(projections)
    table = load_awq_scales(scales_dir, awq_identity(_open_safetensors(model_dir)))
    scales: dict[str, _LayerScales] = {}
    for layer in range(int(config.num_hidden_layers)):
        for stem in selected:
            name = f"layer.{layer}.{stem}"
            if name not in weights:
                continue  # conv layers have no q/k/v/o, attention layers no conv in/out
            if name not in table:
                raise ValueError(
                    f"INT4-AWQ scales have no {name}; run python -m families.lfm2.calibrate_awq_scales"
                )
            transposed = np.asarray(weights[name], dtype=np.float32)
            scales[name] = _LayerScales(None, quantize_awq_weight(transposed.T, table[name]))
    if not scales:
        raise RuntimeError("LFM2 INT4-AWQ quantization matched no projection in the checkpoint")
    return Lfm2QuantContext(scales=scales)
