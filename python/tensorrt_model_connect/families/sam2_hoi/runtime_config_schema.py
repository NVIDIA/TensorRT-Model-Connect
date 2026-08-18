# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time switch for the opt-in SAM2-HOI Phase-A bundle."""

from tensorrt_model_connect.runtime_config import ConfigField, Layer, Schema, register_schema


_BUILD = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT, Layer.SESSION_REQUEST})


SCHEMA = Schema(
    namespace="sam2_hoi",
    fields=(
        ConfigField(
            name="phase_a_pafpn",
            type_tag="bool",
            default=False,
            allowed_layers=_BUILD,
        ),
    ),
)


register_schema(SCHEMA)
