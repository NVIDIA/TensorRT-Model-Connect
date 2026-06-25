"""Retired shared time-series TensorRT helper compatibility stub."""

from __future__ import annotations


class RetiredSharedFamilyHelperError(RuntimeError):
    """Raised when legacy shared family helper entrypoints are used."""


def __getattr__(name: str):
    raise RetiredSharedFamilyHelperError(
        "tensorrt_model_connect.families._time_series_trt is retired; "
        "use the owning family's time_series_trt module instead."
    )
