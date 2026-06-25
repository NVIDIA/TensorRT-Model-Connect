"""Retired shared decoder builder compatibility stub.

Concrete decoder builder implementations are owned by
``tensorrt_model_connect.families.<family>`` modules. Import the owning
family's ``default_decoder`` or ``standard_decoder_builder`` module instead.
"""

from __future__ import annotations


class RetiredSharedBuilderError(RuntimeError):
    """Raised when legacy shared builder entrypoints are used."""


def build_standard_decoder_engine(*_args, **_kwargs):
    raise RetiredSharedBuilderError(
        "tensorrt_model_connect.builders.default_decoder is retired; "
        "use the owning family builder module instead."
    )
