"""Compatibility shim for the shared default TP dual-profile decoder builder."""

from ...builders.default_dual_profile_decoder_tp import (
    _const_in_work_dtype,
    _make_matmul_fn,
    _norm_multi,
    _validate_tp_quantization,
    build_dual_profile_tp_decoder_engine,
)

__all__ = [
    "_const_in_work_dtype",
    "_make_matmul_fn",
    "_norm_multi",
    "_validate_tp_quantization",
    "build_dual_profile_tp_decoder_engine",
]
