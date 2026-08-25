# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time settings for the MiniMax-H3 native TensorRT bundle."""

from __future__ import annotations

import math

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)
from tensorrt_model_connect.families.minimax_h3.config import MINIMAX_H3_WORKFLOWS


# The build CLI currently contributes ``--config`` / ``--set`` values at
# SESSION_REQUEST priority before forwarding them as opaque family options.
_BUILD = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT, Layer.SESSION_REQUEST})


SCHEMA = Schema(
    namespace="minimax_h3",
    fields=(
        ConfigField(
            name="workflow",
            type_tag="string",
            default="t2va",
            allowed_layers=_BUILD,
            validator=lambda value: value in MINIMAX_H3_WORKFLOWS,
        ),
        ConfigField(
            name="first_block_cache",
            type_tag="bool",
            default=False,
            allowed_layers=_BUILD,
        ),
        ConfigField(
            name="first_block_cache_threshold",
            type_tag="double",
            default=0.025,
            allowed_layers=_BUILD,
            validator=lambda value: (
                isinstance(value, float) and math.isfinite(value) and value > 0.0
            ),
        ),
    ),
)


register_schema(SCHEMA)
