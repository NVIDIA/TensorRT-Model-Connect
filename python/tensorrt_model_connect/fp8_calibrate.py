"""Generic FP8 E4M3 calibration and scale extraction.

This module provides the shared infrastructure for FP8 quantization.
Model-specific logic (input generation, layer exclusion) lives in each
family plugin's ``fp8_calibrate()`` method.

Usage:
    # From a family plugin:
    from ..fp8_calibrate import run_fp8_calibration, extract_scales_from_state_dict

    scales = run_fp8_calibration(model, calibration_loop, exclude_pattern)

    # Or from a saved ModelOpt checkpoint:
    scales = extract_scales_from_checkpoint("/tmp/model_fp8.pt", exclude_pattern)
"""

from __future__ import annotations

import re
import sys
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn

# Maxbound for supported quantization formats.
# scale = amax / maxbound maps the calibrated range to the format's full range.
_MAXBOUND = {
    (4, 3): 448.0,     # FP8 E4M3
    (5, 2): 57344.0,   # FP8 E5M2
    (0, 8): 127.0,     # INT8
}
_DEFAULT_MAXBOUND = 448.0  # FP8 E4M3 (used by FP8_DEFAULT_CFG)


def _maxbound_from_config(config: dict) -> float:
    """Derive maxbound from a ModelOpt quantization config."""
    # FP8_DEFAULT_CFG uses num_bits=(4,3) for E4M3
    quant_cfg = config.get("quant_cfg", {})
    for key in ("*weight_quantizer", "*input_quantizer"):
        entry = quant_cfg.get(key, {})
        nb = entry.get("num_bits")
        if isinstance(nb, (tuple, list)) and len(nb) == 2:
            return _MAXBOUND.get(tuple(nb), _DEFAULT_MAXBOUND)
    return _DEFAULT_MAXBOUND


def run_fp8_calibration(
    model: "torch.nn.Module",
    forward_loop: Callable[["torch.nn.Module"], None],
    exclude_pattern: re.Pattern | None = None,
    *,
    config: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Run ModelOpt calibration and extract per-layer scales.

    Args:
        model: PyTorch model (eval mode, on target device).
        forward_loop: ``fn(model)`` that runs calibration data through the model.
            ModelOpt hooks into forward passes to collect activation statistics.
        exclude_pattern: Compiled regex — matching layer names are excluded
            from quantization (kept in BF16/FP32).
        config: ModelOpt quantization config. Defaults to ``FP8_DEFAULT_CFG``
            (FP8 E4M3, per-tensor, max calibration on both weights and activations).

    Returns:
        ``{layer_name: {"input_scale": float, "weight_scale": float}}``
        for every quantized layer.
    """
    import modelopt.torch.quantization as mtq

    if config is None:
        config = mtq.FP8_DEFAULT_CFG

    maxbound = _maxbound_from_config(config)

    print(f"[fp8-calibrate] Starting ModelOpt quantization "
          f"(maxbound={maxbound}) ...", file=sys.stderr)
    model = mtq.quantize(model, config=config, forward_loop=forward_loop)

    if exclude_pattern is not None:
        mtq.disable_quantizer(
            model, lambda name: exclude_pattern.match(name) is not None)

    mtq.print_quant_summary(model)

    return extract_scales_from_state_dict(
        model.state_dict(), exclude_pattern=exclude_pattern,
        maxbound=maxbound)


def extract_scales_from_state_dict(
    state_dict: dict,
    exclude_pattern: re.Pattern | None = None,
    maxbound: float = _DEFAULT_MAXBOUND,
) -> dict[str, dict[str, float]]:
    """Extract quantization scales from a ModelOpt-quantized state dict.

    Converts amax values to TRT Q/DQ scales: ``scale = amax / maxbound``.
    Only returns layers that have **both** input and weight scales and
    do NOT match the exclusion pattern.

    Args:
        state_dict: Model state dict containing ``*_quantizer._amax`` entries.
        exclude_pattern: Compiled regex — matching layer names are dropped.
        maxbound: Max representable value for the quantization format
            (448.0 for FP8 E4M3, 127.0 for INT8, 57344.0 for FP8 E5M2).
    """
    scales: dict[str, dict[str, float]] = {}

    for key, value in state_dict.items():
        if "_amax" not in key:
            continue
        m = re.match(
            r"(.+)\.(input_quantizer|weight_quantizer)\._amax", key)
        if m is None:
            continue

        prefix = m.group(1)
        qtype = m.group(2)
        amax = value.item() if hasattr(value, "item") else float(value)
        scale = amax / maxbound

        if prefix not in scales:
            scales[prefix] = {}
        if "input" in qtype:
            scales[prefix]["input_scale"] = scale
        else:
            scales[prefix]["weight_scale"] = scale

    # Keep only complete entries (both input + weight) that aren't excluded
    result: dict[str, dict[str, float]] = {}
    for name, entry in scales.items():
        if "input_scale" not in entry or "weight_scale" not in entry:
            continue
        if exclude_pattern is not None and exclude_pattern.match(name):
            continue
        result[name] = entry

    print(f"[fp8-calibrate] Extracted {len(result)} layer scales "
          f"(excluded {len(scales) - len(result)})", file=sys.stderr)
    return result


def extract_scales_from_checkpoint(
    checkpoint_path: str,
    exclude_pattern: re.Pattern | None = None,
    maxbound: float = _DEFAULT_MAXBOUND,
) -> dict[str, dict[str, float]]:
    """Extract quantization scales from a saved ModelOpt checkpoint.

    The checkpoint must contain ``model_state_dict`` (from ``mto.save()``)
    or be a plain state dict.

    Args:
        checkpoint_path: Path to saved checkpoint (``.pt``).
        exclude_pattern: Compiled regex — matching layer names are dropped.
        maxbound: Max representable value for the quantization format.
    """
    import torch

    print(f"[fp8-calibrate] Loading checkpoint: {checkpoint_path}",
          file=sys.stderr)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    return extract_scales_from_state_dict(
        sd, exclude_pattern=exclude_pattern, maxbound=maxbound)
