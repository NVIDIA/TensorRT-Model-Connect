"""Retired shared builder utility compatibility stub.

Builder utilities are copied into each family package so model builder changes
do not couple sibling families through this package.
"""

from __future__ import annotations


class RetiredSharedBuilderError(RuntimeError):
    """Raised when legacy shared builder utilities are used."""


def create_builder_context(*_args, **_kwargs):
    raise RetiredSharedBuilderError(
        "tensorrt_model_connect.builders.utils is retired; "
        "use the owning family builder utilities instead."
    )


def const_in_work_dtype(*_args, **_kwargs):
    raise RetiredSharedBuilderError(
        "tensorrt_model_connect.builders.utils is retired; "
        "use the owning family builder utilities instead."
    )


def add_norm(*_args, **_kwargs):
    raise RetiredSharedBuilderError(
        "tensorrt_model_connect.builders.utils is retired; "
        "use the owning family builder utilities instead."
    )
