# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured quantization plan used by build and E2E entrypoints."""

from __future__ import annotations

from dataclasses import dataclass

_FORMAT_ALIASES = {
    "int8": "int8_sq",
    "int4": "int4_awq",
}


def canonicalize_quant_format(format_name: str | None) -> str | None:
    """Normalize CLI aliases to canonical internal format names."""
    if format_name is None:
        return None
    return _FORMAT_ALIASES.get(format_name, format_name)


@dataclass(frozen=True)
class QuantPlan:
    """Canonical description of a single quantized build attempt."""

    base_precision: str = "fp32"
    quant_format: str | None = None
    scale_source: str = "none"
    scale_artifact: str | None = None
    calibration_samples: int = 512

    @property
    def enabled(self) -> bool:
        return self.quant_format is not None

    @classmethod
    def from_build_args(
        cls,
        *,
        precision: str,
        quantize: str | None,
        quant_scales: str | None = None,
        quant_calibration_samples: int = 512,
    ) -> "QuantPlan":
        quant_format = canonicalize_quant_format(quantize)
        if quant_format is None:
            scale_source = "none"
        elif quant_scales:
            scale_source = "precomputed"
        elif quant_format == "nvfp4":
            scale_source = "dynamic"
        else:
            scale_source = "modelopt"
        return cls(
            base_precision=precision,
            quant_format=quant_format,
            scale_source=scale_source,
            scale_artifact=quant_scales,
            calibration_samples=quant_calibration_samples,
        )

    def as_config_dict(self) -> dict[str, str]:
        """Return the minimal bundle/config representation."""
        if not self.enabled or self.quant_format is None:
            return {"format": "none", "scale_source": self.scale_source}
        result = {
            "format": self.quant_format,
            "scale_source": self.scale_source,
        }
        if self.scale_artifact:
            result["scale_artifact"] = self.scale_artifact
        return result
