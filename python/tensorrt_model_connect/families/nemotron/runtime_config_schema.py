# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time configuration for Nemotron decoder engines."""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import ConfigField, Layer, Schema, register_schema


_BUILD = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT, Layer.SESSION_REQUEST})


register_schema(
    Schema(
        namespace="nemotron_decoder",
        fields=(
            ConfigField(
                name="builder_workspace_gib",
                type_tag="int32",
                default=0,
                allowed_layers=_BUILD,
                validator=lambda value: type(value) is int and value >= 0,
            ),
        ),
    )
)
