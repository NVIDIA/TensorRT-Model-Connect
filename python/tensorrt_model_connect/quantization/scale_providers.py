# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scale acquisition strategies.

Each ScaleProvider knows how to obtain quantization scales for a model.
Implementations range from running ModelOpt calibration to loading
pre-computed JSON files or extracting from pre-quantized checkpoints.
"""

from __future__ import annotations

import logging
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ..fp8_calibrate import extract_scales_from_state_dict
from .adapters import AutoCausalLMCalibrationAdapter, CalibrationAdapter
from .formats import QuantFormat
from .scales import LayerScales, QuantScaleMap

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)


def _matches_exclude_pattern(weight_name: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(weight_name, pattern):
            return True
        if "/" in weight_name:
            _, suffix = weight_name.split("/", 1)
            if fnmatch.fnmatch(suffix, pattern):
                return True
    return False


class ScaleProvider(Protocol):
    """Strategy for obtaining quantization scales."""

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
        adapter: CalibrationAdapter | None = None,
    ) -> QuantScaleMap:
        """Return per-layer scales for the given format."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class PrecomputedJsonProvider:
    """Load pre-computed scales from a JSON file."""

    def __init__(self, json_path: str) -> None:
        self.json_path = json_path

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
        adapter: CalibrationAdapter | None = None,
    ) -> QuantScaleMap:
        logger.info("Loading pre-computed scales from %s", self.json_path)
        return QuantScaleMap.load(self.json_path)


class ModelOptCalibrationProvider:
    """Run ModelOpt PTQ calibration to compute scales.

    ModelOpt format configs are shared, but model loading and calibration
    batches are delegated through a family-specific calibration adapter.
    """

    # Map our format names to ModelOpt config names
    _MTQ_CONFIG_NAMES: dict[str, str] = {
        "fp8": "FP8_DEFAULT_CFG",
        "int8_sq": "INT8_SMOOTHQUANT_CFG",
        "int4_awq": "INT4_AWQ_CFG",
        "nvfp4": "NVFP4_DEFAULT_CFG",
        "w4a8": "W4A8_AWQ_BETA_CFG",
    }

    # Maxbound values for scale computation: scale = amax / maxbound
    _MAXBOUND: dict[str, float] = {
        "fp8": 448.0,       # FP8 E4M3
        "int8_sq": 127.0,   # INT8
        "int4_awq": 7.0,    # INT4
        "nvfp4": 6.0,       # NVFP4
        "w4a8": 7.0,        # W4A8 weight
    }

    def __init__(
        self,
        num_samples: int = 512,
        calibration_prompts: list[str] | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.calibration_prompts = calibration_prompts

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
        adapter: CalibrationAdapter | None = None,
    ) -> QuantScaleMap:
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            raise RuntimeError(
                "nvidia-modelopt is required for auto-calibration. "
                "Install with: pip install nvidia-modelopt"
            )

        mtq_config_name = self._MTQ_CONFIG_NAMES.get(quant_format.name)
        if mtq_config_name is None:
            raise ValueError(
                f"No ModelOpt config for format {quant_format.name!r}")
        mtq_cfg = getattr(mtq, mtq_config_name)

        logger.info(
            "Running ModelOpt calibration (%s, %d samples) ...",
            mtq_config_name, self.num_samples)

        import re

        adapter = adapter or AutoCausalLMCalibrationAdapter()
        model, aux = adapter.load_calibration_model(model_dir, config)

        def forward_loop(m):
            for batch in adapter.iter_calibration_batches(
                m,
                aux,
                model_dir=model_dir,
                config=config,
                num_samples=self.num_samples,
                calibration_prompts=self.calibration_prompts,
            ):
                adapter.run_calibration_batch(m, batch)

        # Disable quantization on excluded layers
        exclude_re = re.compile("|".join(
            f"({p.replace('*', '.*')})" for p in exclude_patterns
        )) if exclude_patterns else None

        quantized = mtq.quantize(model, mtq_cfg, forward_loop)

        if exclude_re:
            for name, module in quantized.named_modules():
                if exclude_re.search(name):
                    mtq.disable_quantizer(module, "input_quantizer")
                    mtq.disable_quantizer(module, "weight_quantizer")

        # Extract scales from state dict
        maxbound = self._MAXBOUND.get(quant_format.name, 448.0)
        return self._build_scale_map(
            quantized.state_dict(),
            adapter=adapter,
            exclude_re=exclude_re,
            exclude_patterns=exclude_patterns,
            maxbound=maxbound,
        )

    def _build_scale_map(
        self,
        state_dict: dict,
        *,
        adapter: CalibrationAdapter,
        exclude_re,
        exclude_patterns: list[str],
        maxbound: float,
    ) -> QuantScaleMap:
        raw_scales = extract_scales_from_state_dict(
            state_dict,
            exclude_pattern=exclude_re,
            maxbound=maxbound,
        )
        scales: dict[str, LayerScales] = {}
        for layer_name, scale_dict in raw_scales.items():
            if "input_scale" not in scale_dict or "weight_scale" not in scale_dict:
                continue
            mapped = adapter.map_layer_name(layer_name)
            if mapped is None:
                continue
            if _matches_exclude_pattern(mapped, exclude_patterns):
                continue
            scales[mapped] = LayerScales(
                input_scale=scale_dict["input_scale"],
                weight_scale=scale_dict["weight_scale"],
            )

        logger.info("Extracted scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)


class DynamicQuantizationProvider:
    """No calibration — scales computed at runtime by TRT.

    Used for NVFP4 and MXFP8 where TRT's IDynamicQuantizeLayer computes
    per-block scales during inference.
    """

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
        adapter: CalibrationAdapter | None = None,
    ) -> QuantScaleMap:
        logger.info("Using dynamic quantization (runtime scales)")
        return QuantScaleMap(scales={}, dynamic=True)


class PreQuantizedCheckpointProvider:
    """Extract scales from pre-quantized HF checkpoints (GPTQ, AWQ).

    Detects quantization format from the model's config.json
    quantization_config field.
    """

    def acquire_scales(
        self,
        model_dir: str,
        config: ModelConfig,
        quant_format: QuantFormat,
        exclude_patterns: list[str],
        adapter: CalibrationAdapter | None = None,
    ) -> QuantScaleMap:
        quant_config = config.raw.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "")

        if quant_method == "gptq":
            return self._extract_gptq(model_dir, config, exclude_patterns)
        elif quant_method == "awq":
            return self._extract_awq(model_dir, config, exclude_patterns)
        elif quant_method in ("compressed-tensors", "compressed_tensors"):
            if quant_format.name != "fp8":
                raise ValueError(
                    "Compressed-Tensors float-quantized checkpoints require "
                    f"the fp8 format, got {quant_format.name!r}")
            if adapter is None:
                raise ValueError(
                    "Compressed-Tensors scale extraction requires a family "
                    "quantization adapter")
            return self._extract_compressed_tensors(
                model_dir, exclude_patterns, adapter)
        else:
            raise ValueError(
                f"Unsupported pre-quantized format: {quant_method!r}. "
                "Expected 'gptq', 'awq', or 'compressed-tensors'.")

    def _extract_compressed_tensors(
        self,
        model_dir: str,
        exclude_patterns: list[str],
        adapter: CalibrationAdapter,
    ) -> QuantScaleMap:
        """Extract per-tensor FP8 scales from Compressed-Tensors weights."""
        from safetensors import safe_open

        scales: dict[str, LayerScales] = {}
        for sf_path in Path(model_dir).glob("*.safetensors"):
            with safe_open(str(sf_path), framework="pt", device="cpu") as reader:
                keys = set(reader.keys())
                for key in keys:
                    if not key.endswith(".weight_scale"):
                        continue
                    layer_name = key.removesuffix(".weight_scale")
                    input_key = f"{layer_name}.input_scale"
                    if input_key not in keys:
                        raise ValueError(
                            f"Compressed-Tensors layer {layer_name} is missing "
                            f"{input_key}")
                    mapped = adapter.map_layer_name(layer_name)
                    if mapped is None or _matches_exclude_pattern(
                            mapped, exclude_patterns):
                        continue
                    weight_scale = reader.get_tensor(key).float().cpu().numpy()
                    input_scale = reader.get_tensor(input_key).float().cpu().numpy()
                    if weight_scale.size != 1 or input_scale.size != 1:
                        raise NotImplementedError(
                            "Compressed-Tensors FP8 loading currently supports "
                            f"per-tensor scales only; {layer_name} has "
                            f"weight_scale={weight_scale.shape}, "
                            f"input_scale={input_scale.shape}")
                    scales[mapped] = LayerScales(
                        input_scale=float(input_scale.reshape(-1)[0]),
                        weight_scale=float(weight_scale.reshape(-1)[0]),
                    )

        logger.info(
            "Extracted Compressed-Tensors FP8 scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)

    def _extract_gptq(self, model_dir, config, exclude_patterns):
        """Extract scales from GPTQ checkpoint."""
        import re
        from safetensors import safe_open

        exclude_re = re.compile("|".join(
            f"({p.replace('*', '.*')})" for p in exclude_patterns
        )) if exclude_patterns else None

        scales: dict[str, LayerScales] = {}
        model_path = Path(model_dir)
        for sf_path in model_path.glob("*.safetensors"):
            with safe_open(str(sf_path), framework="numpy") as f:
                for key in f.keys():
                    if key.endswith(".g_idx") or key.endswith(".qzeros"):
                        continue
                    if key.endswith(".scales"):
                        layer_name = key.rsplit(".scales", 1)[0]
                        if exclude_re and exclude_re.search(layer_name):
                            continue
                        scale_array = f.get_tensor(key)
                        scales[layer_name] = LayerScales(
                            weight_scale=scale_array,
                            input_scale=1.0,
                            block_size=config.raw.get(
                                "quantization_config", {}).get(
                                "group_size", 128),
                        )

        logger.info("Extracted GPTQ scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)

    def _extract_awq(self, model_dir, config, exclude_patterns):
        """Extract scales from AWQ checkpoint."""
        import re
        from safetensors import safe_open

        exclude_re = re.compile("|".join(
            f"({p.replace('*', '.*')})" for p in exclude_patterns
        )) if exclude_patterns else None

        quant_config = config.raw.get("quantization_config", {})
        group_size = quant_config.get("group_size", 128)

        scales: dict[str, LayerScales] = {}
        model_path = Path(model_dir)
        for sf_path in model_path.glob("*.safetensors"):
            with safe_open(str(sf_path), framework="numpy") as f:
                for key in f.keys():
                    if not key.endswith(".scales"):
                        continue
                    layer_name = key.rsplit(".scales", 1)[0]
                    if exclude_re and exclude_re.search(layer_name):
                        continue
                    scale_array = f.get_tensor(key)
                    scales[layer_name] = LayerScales(
                        weight_scale=scale_array,
                        input_scale=1.0,
                        block_size=group_size,
                    )

        logger.info("Extracted AWQ scales for %d layers", len(scales))
        return QuantScaleMap(scales=scales)
