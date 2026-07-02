# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema for the ``text_trace`` namespace.

Covers the per-step debug tracing of text-generation decodes that used to
live behind the ``TRTMC_TEXT_STEP_TRACE_*`` environment variables. Session-
only — these knobs are purely debug plumbing; no builder path touches them.
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
    namespace="text_trace",
    fields=(
        ConfigField(
            name="step_trace_path",
            type_tag="string",
            default="",  # empty → tracing disabled
            allowed_layers=_SESSION,
        ),
        ConfigField(
            name="step_trace_start_pos",
            type_tag="int32",
            default=0,
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, int) and v >= 0,
        ),
        # Large enough to act as "unbounded" in practice without spilling the
        # uint32 range; matches the previous INT32_MAX sentinel in the C++.
        ConfigField(
            name="step_trace_end_pos",
            type_tag="int32",
            default=2_000_000_000,
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, int) and v >= 0,
        ),
        ConfigField(
            name="step_trace_topk",
            type_tag="int32",
            default=8,
            allowed_layers=_SESSION,
            validator=lambda v: isinstance(v, int) and v >= 1,
        ),
    ),
)


register_schema(SCHEMA)
