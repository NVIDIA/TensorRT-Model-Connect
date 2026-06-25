"""Retired shared dual-profile decoder builder compatibility stub.

Concrete decoder builder implementations are owned by
``tensorrt_model_connect.families.<family>`` modules.
"""

from __future__ import annotations


class RetiredSharedBuilderError(RuntimeError):
    """Raised when legacy shared builder entrypoints are used."""


def build_dual_profile_decoder_engine(*_args, **_kwargs):
    raise RetiredSharedBuilderError(
        "tensorrt_model_connect.builders.default_dual_profile_decoder is retired; "
        "use the owning family builder module instead."
    )
