# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated MiniMax-H3 family options and runtime defaults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable


class Layer(str, Enum):
    """Where a family option may be supplied."""

    BUILD_TIME = "build_time"
    BUNDLE_DEFAULT = "bundle_default"
    SESSION_REQUEST = "session_request"
    PLATFORM_PROFILE = "platform_profile"


@dataclass(frozen=True)
class ConfigField:
    name: str
    type_tag: str
    default: object
    allowed_layers: frozenset[Layer]
    validator: Callable[[object], bool] | None = None

    def validate(self, value: object) -> object:
        valid_type = {
            "bool": lambda item: isinstance(item, bool),
            "double": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "int64": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "string": lambda item: isinstance(item, str),
        }[self.type_tag](value)
        if not valid_type or (self.validator is not None and not self.validator(value)):
            raise ValueError(f"invalid MiniMax-H3 option {self.name}={value!r}")
        return value


@dataclass(frozen=True)
class Schema:
    namespace: str
    fields: tuple[ConfigField, ...]

    def normalize(self, values: dict[str, object]) -> dict[str, object]:
        fields = {field.name: field for field in self.fields}
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ValueError(f"unknown MiniMax-H3 option(s): {', '.join(unknown)}")
        return {
            name: fields[name].validate(value)
            for name, value in values.items()
        }


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
        ConfigField("first_block_cache", "bool", True, _BUILD),
        ConfigField(
            "first_block_cache_threshold",
            "double",
            0.08,
            _BUILD,
            lambda value: math.isfinite(float(value)) and float(value) > 0.0,
        ),
        ConfigField("transformer_ref", "string", "", _BUILD_PATH),
        ConfigField("quantized_transformer", "string", "", _BUILD_PATH),
        ConfigField("super_resolution_model", "string", "", _BUILD_PATH),
        ConfigField("super_resolution_weak_model", "string", "", _BUILD_PATH),
        ConfigField("retain_engines", "bool", False, _SESSION),
        ConfigField(
            "retained_tail_weight_budget_gib",
            "int64",
            24,
            _SESSION,
            _positive_budget_gib,
        ),
    ),
)


def normalize_build_options(values: dict[str, object]) -> dict[str, object]:
    """Validate family options accepted by the generic ``--set`` bridge."""

    normalized = SCHEMA.normalize(values)
    disallowed = [
        name
        for name in normalized
        if Layer.BUILD_TIME
        not in next(field for field in SCHEMA.fields if field.name == name).allowed_layers
    ]
    if disallowed:
        raise ValueError(
            f"MiniMax-H3 option(s) are runtime-only: {', '.join(sorted(disallowed))}"
        )
    return normalized
