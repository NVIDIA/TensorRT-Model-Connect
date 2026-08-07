# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External-system adapter boundary.

Phase 0 intentionally installs only disabled adapters. Future adapters must keep
the same preview/approve/publish/read-back boundary and use Store's persistent
idempotency reservation before performing a remote mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .domain import PermissionDeniedError


@dataclass(frozen=True)
class AdapterCapabilities:
    system: str
    mode: str
    can_read: bool
    can_publish: bool
    message: str


@dataclass(frozen=True)
class PublishEnvelope:
    operation: str
    idempotency_key: str
    actor: str
    payload: Mapping[str, Any]


class ExternalAdapter(Protocol):
    capabilities: AdapterCapabilities

    def read(self, external_id: str) -> Mapping[str, Any]: ...

    def publish(self, envelope: PublishEnvelope) -> Mapping[str, Any]: ...


class DisabledAdapter:
    def __init__(self, capabilities: AdapterCapabilities):
        self.capabilities = capabilities

    def read(self, external_id: str) -> Mapping[str, Any]:
        raise PermissionDeniedError(f"{self.capabilities.system} read adapter is disabled")

    def publish(self, envelope: PublishEnvelope) -> Mapping[str, Any]:
        raise PermissionDeniedError(f"{self.capabilities.system} write adapter is disabled")


class IntegrationRegistry:
    def __init__(self, adapters: Mapping[str, ExternalAdapter]):
        self._adapters = dict(adapters)

    @classmethod
    def phase_zero(cls) -> "IntegrationRegistry":
        specifications = (
            AdapterCapabilities(
                "github",
                "manual_link",
                False,
                False,
                "Issue and pull request links are stored locally.",
            ),
            AdapterCapabilities(
                "devtest",
                "local_draft",
                False,
                False,
                "Test plan drafts remain in Hub until an approved adapter is configured.",
            ),
            AdapterCapabilities(
                "nvbug",
                "local_draft",
                False,
                False,
                "Defect drafts remain in Hub; manual IDs can be linked after filing.",
            ),
        )
        return cls({item.system: DisabledAdapter(item) for item in specifications})

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "system": adapter.capabilities.system,
                "mode": adapter.capabilities.mode,
                "read": adapter.capabilities.can_read,
                "write": adapter.capabilities.can_publish,
                "message": adapter.capabilities.message,
            }
            for adapter in self._adapters.values()
        ]

    def get(self, system: str) -> ExternalAdapter:
        return self._adapters[system]
