# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time and runtime settings for the MiniMax-H3 native TensorRT path."""

from __future__ import annotations

import math

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


# The build CLI currently contributes ``--config`` / ``--set`` values at
# SESSION_REQUEST priority before forwarding them as opaque family options.
_BUILD = frozenset({Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT, Layer.SESSION_REQUEST})
_BUILD_PATH = frozenset({Layer.BUILD_TIME, Layer.SESSION_REQUEST})
_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})


def _positive_budget_gib(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= ((2**63 - 1) >> 30)
    )


SCHEMA = Schema(
    namespace="minimax_h3",
    fields=(
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
        ConfigField(
            name="fast_h3_adapter",
            type_tag="string",
            default="",
            allowed_layers=_BUILD_PATH,
        ),
        ConfigField(
            name="transformer_ref",
            type_tag="string",
            default="",
            allowed_layers=_BUILD_PATH,
        ),
        ConfigField(
            name="retain_engines",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="retained_tail_weight_budget_gib",
            type_tag="int64",
            default=24,
            allowed_layers=_SESSION,
            validator=_positive_budget_gib,
        ),
    ),
)


register_schema(SCHEMA)
