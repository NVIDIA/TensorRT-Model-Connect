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


FP8_MHA_CONFIG = {
    "quant_cfg": {
        "*": {"enable": False},
        "*weight_quantizer": {"num_bits": [4, 3], "axis": None},
        "*input_quantizer": {"num_bits": [4, 3], "axis": None},
        "*q_bmm_quantizer": {"num_bits": [4, 3], "axis": None},
        "*k_bmm_quantizer": {"num_bits": [4, 3], "axis": None},
        "*v_bmm_quantizer": {"num_bits": [4, 3], "axis": None},
        "*softmax_quantizer": {"num_bits": [4, 3], "axis": None},
        "*bmm2_output_quantizer": {"num_bits": [4, 3], "axis": None},
    },
    "algorithm": "max",
}

_QUANTIZER_SCALE_FIELDS = {
    "input_quantizer": "input_scale",
    "weight_quantizer": "weight_scale",
    "q_bmm_quantizer": "q_bmm_scale",
    "k_bmm_quantizer": "k_bmm_scale",
    "v_bmm_quantizer": "v_bmm_scale",
    "softmax_quantizer": "softmax_scale",
    "bmm2_output_quantizer": "bmm2_output_scale",
}
_LINEAR_REQUIRED_SCALE_FIELDS = {"input_scale", "weight_scale"}
_ATTENTION_REQUIRED_SCALE_FIELDS = {
    "q_bmm_scale",
    "k_bmm_scale",
    "v_bmm_scale",
    "softmax_scale",
}


def _config_requests_mha_quantizers(config: dict) -> bool:
    quant_cfg = config.get("quant_cfg", {})
    return any("bmm_quantizer" in str(pattern) for pattern in quant_cfg)


def _register_diffusers_flux2_attention_quantizers() -> None:
    """Register FLUX.2 Diffusers attention modules with ModelOpt's MHA wrapper.

    ModelOpt has the FP8 attention quantizer implementation, but some releases
    only register it for FLUX.1's ``FluxAttention`` class. FLUX.2 uses separate
    ``Flux2Attention`` and ``Flux2ParallelSelfAttention`` classes, so register
    those classes before ModelOpt recursively replaces modules.
    """
    try:
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2Attention,
            Flux2ParallelSelfAttention,
        )
        from modelopt.torch.quantization.nn import QuantModuleRegistry
    except Exception as exc:  # pragma: no cover - optional Diffusers/ModelOpt deps
        print(
            "[fp8-calibrate] Skipping FLUX.2 MHA quantizer registration: "
            f"{exc}",
            file=sys.stderr,
        )
        return

    try:
        try:
            from modelopt.torch.quantization.plugins.diffusion.diffusers import (
                _QuantAttention,
            )
        except ModuleNotFoundError:
            from modelopt.torch.quantization.plugins.diffusers import _QuantAttention
    except Exception as exc:  # pragma: no cover - private ModelOpt API drift
        print(
            "[fp8-calibrate] ModelOpt Diffusers MHA quantizer is unavailable: "
            f"{exc}",
            file=sys.stderr,
        )
        return

    try:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            attention_backend,
        )
    except Exception:  # pragma: no cover - older Diffusers fallback
        AttentionBackendName = None
        attention_backend = None

    class _QuantFlux2Attention(_QuantAttention):
        def forward(self, *args, **kwargs):
            if attention_backend is None or AttentionBackendName is None:
                return super().forward(*args, **kwargs)
            with attention_backend(AttentionBackendName.NATIVE):
                return super().forward(*args, **kwargs)

    for module_cls, key in (
        (Flux2Attention, "Flux2Attention"),
        (Flux2ParallelSelfAttention, "Flux2ParallelSelfAttention"),
    ):
        if module_cls not in QuantModuleRegistry:
            QuantModuleRegistry.register({module_cls: key})(_QuantFlux2Attention)


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
        for every quantized linear layer, plus attention entries containing
        ``q_bmm_scale``, ``k_bmm_scale``, ``v_bmm_scale``,
        ``softmax_scale``, and optionally ``bmm2_output_scale`` when the
        ModelOpt config includes MHA BMM quantizers.
    """
    import modelopt.torch.quantization as mtq

    if config is None:
        config = mtq.FP8_DEFAULT_CFG

    if _config_requests_mha_quantizers(config):
        _register_diffusers_flux2_attention_quantizers()

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
    Returns complete linear entries that have both input and weight scales,
    and complete attention entries that have Q/K/V BMM scales plus a softmax
    normalization scale. Entries matching the exclusion pattern are dropped.

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
        m = re.match(r"(.+)\.([A-Za-z0-9_]+_quantizer)\._amax", key)
        if m is None:
            continue

        prefix = m.group(1)
        qtype = m.group(2)
        scale_field = _QUANTIZER_SCALE_FIELDS.get(qtype)
        if scale_field is None:
            continue
        amax = value.item() if hasattr(value, "item") else float(value)
        scale = amax / maxbound

        if prefix not in scales:
            scales[prefix] = {}
        scales[prefix][scale_field] = scale

    # Keep only complete linear or attention entries that aren't excluded.
    result: dict[str, dict[str, float]] = {}
    for name, entry in scales.items():
        fields = set(entry)
        complete_linear = _LINEAR_REQUIRED_SCALE_FIELDS.issubset(fields)
        if {
            "q_bmm_scale",
            "k_bmm_scale",
            "v_bmm_scale",
        }.issubset(fields) and "softmax_scale" not in entry:
            # ModelOpt's SDPA FP8 exporter hard-codes the softmax output amax
            # to 1.0 instead of storing a softmax_quantizer._amax state.
            entry["softmax_scale"] = 1.0 / maxbound
            fields.add("softmax_scale")
        complete_attention = _ATTENTION_REQUIRED_SCALE_FIELDS.issubset(fields)
        if not complete_linear and not complete_attention:
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
