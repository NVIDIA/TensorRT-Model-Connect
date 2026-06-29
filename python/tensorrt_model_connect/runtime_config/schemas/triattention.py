"""Schema for the ``triattention`` namespace.

Replaces the former ``TRTMC_TRIATTN_*`` env vars and the fragment of bundle
``config.json`` previously parsed by
family-local ``src/runtime/models/<family>/triattention_kv_cache.cpp``
parsers.

Each field declares which config layers may supply a value. The former
``TRTMC_TRIATTN_OVERRIDE_*`` env vars (they baked the "override" frame into
their names) become layer-gated fields here — the same value shape, but
now ``SESSION_REQUEST`` is just a legitimate source, not an "override".
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_BUNDLE_AND_SESSION = frozenset({
    Layer.BUILD_TIME,
    Layer.BUNDLE_DEFAULT,
    Layer.PLATFORM_PROFILE,
    Layer.SESSION_REQUEST,
})
_SESSION_ONLY = frozenset({Layer.SESSION_REQUEST})


def _agg_validator(v) -> bool:
    return isinstance(v, str) and v in {"mean", "max"}


SCHEMA = Schema(
    namespace="triattention",
    fields=(
        # --- Core runtime config (formerly bundle config.json + overrides) ---
        ConfigField(
            name="enabled",
            type_tag="bool",
            default=False,
            allowed_layers=_BUNDLE_AND_SESSION,
        ),
        ConfigField(
            name="kv_budget",
            type_tag="int32",
            default=4096,
            allowed_layers=_BUNDLE_AND_SESSION,
            validator=lambda v: isinstance(v, int) and v > 0,
        ),
        ConfigField(
            name="divide_length",
            type_tag="int32",
            default=128,
            allowed_layers=_BUNDLE_AND_SESSION,
            validator=lambda v: isinstance(v, int) and v > 0,
        ),
        ConfigField(
            name="recent_window",
            type_tag="int32",
            default=128,
            allowed_layers=_BUNDLE_AND_SESSION,
            validator=lambda v: isinstance(v, int) and v >= 0,
        ),
        ConfigField(
            name="score_aggregation",
            type_tag="string",
            default="mean",
            allowed_layers=_BUNDLE_AND_SESSION,
            validator=_agg_validator,
        ),
        ConfigField(
            name="per_layer_aggregation",
            type_tag="string",
            default="mean",
            allowed_layers=_BUNDLE_AND_SESSION,
            validator=_agg_validator,
        ),
        ConfigField(
            name="count_prompt_tokens",
            type_tag="bool",
            default=True,
            allowed_layers=_BUNDLE_AND_SESSION,
        ),
        ConfigField(
            name="protect_prefill",
            type_tag="bool",
            default=True,
            allowed_layers=_BUNDLE_AND_SESSION,
        ),
        ConfigField(
            name="disable_mlr",
            type_tag="bool",
            default=False,
            allowed_layers=_BUNDLE_AND_SESSION,
        ),
        ConfigField(
            name="disable_trig",
            type_tag="bool",
            default=False,
            allowed_layers=_BUNDLE_AND_SESSION,
        ),
        ConfigField(
            name="offset_max_length",
            type_tag="int32",
            default=65536,
            allowed_layers=_BUNDLE_AND_SESSION,
            validator=lambda v: isinstance(v, int) and v > 0,
        ),
        # Build-time only: points to a bundle section with calibration stats.
        ConfigField(
            name="stats_section",
            type_tag="string",
            default="triattention_stats.json",
            allowed_layers=frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT}),
        ),
        # --- Debug / profiling knobs (session-only) ---
        ConfigField(
            name="debug",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="profile",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="runtime_bucket_rows",
            type_tag="int32",
            default=32,
            allowed_layers=_SESSION_ONLY,
            validator=lambda v: isinstance(v, int) and v >= 1,
        ),
        ConfigField(
            name="disable_gpu_selection",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="disable_gpu_compaction",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="disable_gpu_state",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="zero_tail",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="dump_keep_path",
            type_tag="string",
            default="",
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="dump_compaction_index",
            type_tag="int32",
            default=0,
            allowed_layers=_SESSION_ONLY,
            validator=lambda v: isinstance(v, int) and v >= 0,
        ),
        ConfigField(
            name="abort_after_dump",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="dump_score_cache",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
        ConfigField(
            name="dump_score_values",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION_ONLY,
        ),
    ),
)


register_schema(SCHEMA)
