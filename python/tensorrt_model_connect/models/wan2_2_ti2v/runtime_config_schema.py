# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative runtime settings for the ``wan2_2_ti2v`` pipeline."""

from __future__ import annotations

import math

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})
_INT32_MAX = 2**31 - 1


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and value > 0.0
    )


def _nonnegative_int32(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _INT32_MAX
    )


def _positive_int32(value: object) -> bool:
    return _nonnegative_int32(value) and value > 0


SCHEMA = Schema(
    namespace="wan2_2_ti2v",
    fields=(
        ConfigField(
            name="easycache_enabled",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="easycache_threshold",
            type_tag="double",
            default=0.02,
            allowed_layers=_SESSION,
            validator=_positive_finite,
        ),
        ConfigField(
            name="easycache_first_exact_steps",
            type_tag="int64",
            default=7,
            allowed_layers=_SESSION,
            validator=_nonnegative_int32,
        ),
        ConfigField(
            name="easycache_last_exact_steps",
            type_tag="int64",
            default=2,
            allowed_layers=_SESSION,
            validator=_nonnegative_int32,
        ),
        ConfigField(
            name="easycache_max_consecutive_reuse",
            type_tag="int64",
            default=1,
            allowed_layers=_SESSION,
            validator=_positive_int32,
        ),
        ConfigField(
            name="late_cfg_enabled",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
    ),
)


register_schema(SCHEMA)
