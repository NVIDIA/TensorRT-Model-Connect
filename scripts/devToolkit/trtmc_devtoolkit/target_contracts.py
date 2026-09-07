# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral execution-target lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .models import DevToolkitError
from .resolution import ExecutionTarget, ProviderDescriptor


def _scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise DevToolkitError("Target identity float values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise DevToolkitError(
        f"Target identity values must be JSON-compatible: {type(value).__name__}"
    )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return _scalar(value)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise DevToolkitError("Target identity mapping keys must be strings")
        return MappingProxyType({name: _freeze(value[name]) for name in sorted(value)})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return _scalar(value)


def _digest(namespace: bytes, value: object) -> str:
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()


@dataclass(frozen=True)
class TargetPlan:
    provider: ProviderDescriptor
    plan_id: str
    intent: Mapping[str, object]
    request: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.plan_id) is None:
            raise DevToolkitError("Target plans require a lowercase SHA-256 plan ID")
        frozen = _freeze(self.intent)
        if not isinstance(frozen, Mapping):
            raise DevToolkitError("Target plan intent must be a mapping")
        object.__setattr__(self, "intent", frozen)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "provider": {
                "name": self.provider.name,
                "implementation": self.provider.implementation,
                "lock_schema": self.provider.lock_schema,
            },
            "intent": _plain(self.intent),
        }


@dataclass(frozen=True)
class TargetHandle:
    provider: ProviderDescriptor
    plan_id: str
    target_id: str
    action: str
    policy: str
    identity: Mapping[str, object]
    observation: Mapping[str, object]
    execution_target: ExecutionTarget
    request: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.plan_id) is None
            or re.fullmatch(r"[0-9a-f]{64}", self.target_id) is None
        ):
            raise DevToolkitError("Target handles require lowercase SHA-256 identities")
        if not self.action or not self.policy:
            raise DevToolkitError("Target handles require non-empty action and policy names")
        identity = _freeze(self.identity)
        observation = _freeze(self.observation)
        if not isinstance(identity, Mapping) or not isinstance(observation, Mapping):
            raise DevToolkitError("Target handle identity and observation must be mappings")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "observation", observation)


@dataclass(frozen=True)
class ProvisionedTarget:
    provider: ProviderDescriptor
    plan_id: str
    target_id: str
    action: str
    policy: str
    identity: Mapping[str, object]
    observation: Mapping[str, object]
    execution_target: ExecutionTarget
    receipt: Path = field(compare=False)
    request: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.action or not self.policy:
            raise DevToolkitError("Provisioned targets require non-empty action and policy names")
        identity = _freeze(self.identity)
        observation = _freeze(self.observation)
        if not isinstance(identity, Mapping) or not isinstance(observation, Mapping):
            raise DevToolkitError("Provisioned target evidence must be mappings")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "observation", observation)
