"""Schema for the ``parallel`` namespace.

Tensor-parallel builds are opt-in and build-time only by default. Runtime
selection comes from the bundle metadata and mpirun environment variables, so
single-device sessions keep the same default behavior.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_BUILD_AND_SESSION = frozenset(
    {
        Layer.BUILD_TIME,
        Layer.BUNDLE_DEFAULT,
        Layer.PLATFORM_PROFILE,
        Layer.SESSION_REQUEST,
    }
)


def _valid_mode(value: object) -> bool:
    return str(value) in {"single", "tensor_parallel"}


def _valid_tp_size(value: object) -> bool:
    try:
        return int(value) in {1, 2, 4, 8}
    except (TypeError, ValueError):
        return False


def _valid_rank(value: object) -> bool:
    try:
        return int(value) >= -1
    except (TypeError, ValueError):
        return False


SCHEMA = Schema(
    namespace="parallel",
    fields=(
        ConfigField(
            name="mode",
            type_tag="string",
            default="single",
            allowed_layers=_BUILD_AND_SESSION,
            validator=_valid_mode,
        ),
        ConfigField(
            name="tp_size",
            type_tag="int32",
            default=1,
            allowed_layers=_BUILD_AND_SESSION,
            validator=_valid_tp_size,
        ),
        ConfigField(
            name="rank",
            type_tag="int32",
            default=-1,
            allowed_layers=_BUILD_AND_SESSION,
            validator=_valid_rank,
        ),
        ConfigField(
            name="require_mpirun",
            type_tag="bool",
            default=True,
            allowed_layers=_BUILD_AND_SESSION,
        ),
    ),
)


register_schema(SCHEMA)
