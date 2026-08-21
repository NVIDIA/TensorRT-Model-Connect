# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time schemas for Qwen-VL engines."""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import ConfigField, Layer, Schema, register_schema


# The build CLI currently resolves ``--config`` / ``--set`` contributions at
# SESSION_REQUEST priority before forwarding the values to family builders.
_BUILD = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT, Layer.SESSION_REQUEST})


DECODER_SCHEMA = Schema(
    namespace="qwen_vl_decoder",
    fields=(
        ConfigField(
            name="decode_attention",
            type_tag="string",
            default="native",
            allowed_layers=_BUILD,
            validator=lambda value: value in {"native", "decomposed"},
        ),
        ConfigField(
            name="max_prefill_length",
            type_tag="int32",
            default=0,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value >= 0,
        ),
        ConfigField(
            name="opt_prefill_length",
            type_tag="int32",
            default=64,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value > 0,
        ),
        ConfigField(
            name="builder_workspace_gib",
            type_tag="int32",
            default=0,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value >= 0,
        ),
    ),
)

LORA_SCHEMA = Schema(
    namespace="qwen_vl_lora",
    fields=(
        ConfigField(
            name="enabled",
            type_tag="bool",
            default=False,
            allowed_layers=_BUILD,
        ),
        ConfigField(
            name="max_rank",
            type_tag="int32",
            default=0,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and 0 <= value <= 256,
        ),
        ConfigField(
            name="target_modules",
            type_tag="string",
            default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            allowed_layers=_BUILD,
        ),
    ),
)

VISION_SCHEMA = Schema(
    namespace="qwen_vl_vision",
    fields=(
        ConfigField(
            name="image_height",
            type_tag="int32",
            default=448,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value > 0,
        ),
        ConfigField(
            name="image_width",
            type_tag="int32",
            default=448,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value > 0,
        ),
        ConfigField(
            name="dynamic_resolution",
            type_tag="bool",
            default=False,
            allowed_layers=_BUILD,
        ),
        ConfigField(
            name="min_pixels",
            type_tag="int32",
            default=0,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value >= 0,
        ),
        ConfigField(
            name="opt_pixels",
            type_tag="int32",
            default=200704,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value > 0,
        ),
        ConfigField(
            name="max_pixels",
            type_tag="int32",
            default=0,
            allowed_layers=_BUILD,
            validator=lambda value: isinstance(value, int) and value >= 0,
        ),
    ),
)


register_schema(DECODER_SCHEMA)
register_schema(LORA_SCHEMA)
register_schema(VISION_SCHEMA)
