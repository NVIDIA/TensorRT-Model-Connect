"""Schema for the ``audio_magpie`` namespace.

Sampling / CFG / debug knobs for the Magpie TTS pipeline plus the
build-time ``max_source_positions`` truncation. Replaces the six
``TRTMC_MAGPIE_*`` env vars.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})
_BUILD_AND_BUNDLE = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT})


SCHEMA = Schema(
    namespace="audio_magpie",
    fields=(
        # Runtime sampling — session/platform layers.
        ConfigField(
            name="greedy",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="cfg_scale",
            type_tag="float",
            default=0.0,  # 0 = use bundle default
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, (int, float)) and v >= 0.0,
        ),
        ConfigField(
            name="temperature",
            type_tag="float",
            default=0.0,  # 0 = use bundle default
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, (int, float)) and v >= 0.0,
        ),
        ConfigField(
            name="finished_limit",
            type_tag="int32",
            default=-1,  # -1 = leave default; >=0 = cap
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="seed",
            type_tag="int64",
            default=-1,  # -1 = leave RNG unchanged
            allowed_layers=_SESSION,
        ),
        # Build-time — baked into the encoder position embedding.
        ConfigField(
            name="max_source_positions",
            type_tag="int32",
            default=0,  # 0 = keep model default
            allowed_layers=_BUILD_AND_BUNDLE,
            validator=lambda v: isinstance(v, int) and v >= 0,
        ),
    ),
)


register_schema(SCHEMA)
