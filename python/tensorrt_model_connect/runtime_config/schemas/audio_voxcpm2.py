"""Schema for the ``audio_voxcpm2`` namespace.

Session knobs mirror the upstream ``voxcpm`` generation API. The current
family plugin is detection-only; these fields reserve the runtime surface for a
future TensorRT implementation without overloading Bark or Magpie settings.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})


SCHEMA = Schema(
    namespace="audio_voxcpm2",
    fields=(
        ConfigField(
            name="cfg_value",
            type_tag="float",
            default=2.0,
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, (int, float)) and v >= 0.0,
        ),
        ConfigField(
            name="inference_timesteps",
            type_tag="int32",
            default=10,
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, int) and v > 0,
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
            validator=lambda v: isinstance(v, int) and v >= 0,
        ),
        ConfigField(
            name="retry_badcase_ratio_threshold",
            type_tag="float",
            default=6.0,
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, (int, float)) and v > 0.0,
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
