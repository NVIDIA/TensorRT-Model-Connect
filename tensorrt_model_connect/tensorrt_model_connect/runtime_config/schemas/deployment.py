"""Schema for the ``deployment`` namespace.

Deployment specialization is selected through the existing two-flag config
surface instead of adding one flag per provider.  The build command consumes
these fields to choose a build provider or kernel specialization; the runtime
command can use the same namespace to force a bundled variant for debugging.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_BUILD_AND_SESSION = frozenset({
    Layer.BUILD_TIME,
    Layer.BUNDLE_DEFAULT,
    Layer.PLATFORM_PROFILE,
    Layer.SESSION_REQUEST,
})


SCHEMA = Schema(
    namespace="deployment",
    fields=(
        ConfigField(
            name="target",
            type_tag="string",
            default="generic",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="provider",
            type_tag="string",
            default="native_trt",
            allowed_layers=_BUILD_AND_SESSION,
            validator=lambda v: v in {
                "native_trt",
                "tvm_ffi",
                "tensorrt-edge-llm",
            },
        ),
        ConfigField(
            name="variant",
            type_tag="string",
            default="",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="force_fallback",
            type_tag="bool",
            default=False,
            allowed_layers=frozenset({
                Layer.PLATFORM_PROFILE,
                Layer.SESSION_REQUEST,
            }),
        ),
        ConfigField(
            name="enable_ffi_attention",
            type_tag="bool",
            default=False,
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="ffi_kernel_artifacts",
            type_tag="string",
            default="",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="edge_llm_workspace",
            type_tag="string",
            default="",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="edge_llm_engine_dir",
            type_tag="string",
            default="",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="edge_llm_export_tool",
            type_tag="string",
            default="tensorrt-edgellm-export-llm",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="edge_llm_build_tool",
            type_tag="string",
            default="llm_build",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="edge_llm_export_device",
            type_tag="string",
            default="cuda",
            allowed_layers=_BUILD_AND_SESSION,
        ),
        ConfigField(
            name="edge_llm_max_input_len",
            type_tag="int32",
            default=1024,
            allowed_layers=_BUILD_AND_SESSION,
            validator=lambda v: isinstance(v, int) and v > 0,
        ),
        ConfigField(
            name="edge_llm_max_batch_size",
            type_tag="int32",
            default=4,
            allowed_layers=_BUILD_AND_SESSION,
            validator=lambda v: isinstance(v, int) and v > 0,
        ),
    ),
)


register_schema(SCHEMA)
