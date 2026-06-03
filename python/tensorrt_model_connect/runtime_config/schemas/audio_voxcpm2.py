"""Schema for the ``audio_voxcpm2`` namespace.

Runtime generation knobs for VoxCPM2 text-to-audio inference. These mirror the
model-card TTS example so E2E runs can pass the same parameters to the TRT path
that the official ``voxcpm`` reference backend uses.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})


def _nonnegative_float(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0.0


def _nonnegative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


SCHEMA = Schema(
    namespace="audio_voxcpm2",
    fields=(
        ConfigField(
            name="cfg_value",
            type_tag="float",
            default=2.0,
            allowed_layers=_SESSION,
            validator=_nonnegative_float,
        ),
        ConfigField(
            name="inference_timesteps",
            type_tag="int32",
            default=10,
            allowed_layers=_SESSION,
            validator=_nonnegative_int,
        ),
        ConfigField(
            name="normalize",
            type_tag="bool",
            default=True,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="denoise",
            type_tag="bool",
            default=True,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="retry_badcase",
            type_tag="bool",
            default=True,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="retry_badcase_max_times",
            type_tag="int32",
            default=3,
            allowed_layers=_SESSION,
            validator=_nonnegative_int,
        ),
        ConfigField(
            name="retry_badcase_ratio_threshold",
            type_tag="float",
            default=6.0,
            allowed_layers=_SESSION,
            validator=_nonnegative_float,
        ),
        ConfigField(
            name="seed",
            type_tag="int64",
            default=-1,
            allowed_layers=_SESSION,
        ),
    ),
)


register_schema(SCHEMA)
