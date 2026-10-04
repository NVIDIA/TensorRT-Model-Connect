# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend contract shared by the TRTMC worker and the Python references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


class BackendError(RuntimeError):
    """A request the backend rejected; the backend remains usable."""


class BackendUnavailable(RuntimeError):
    """The backend can no longer serve (crashed, timed out, or failed to load)."""


@dataclass(frozen=True)
class Invocation:
    """One operation call.

    ``model_call_ms`` is the backend's timed boundary, the whole Task call on both sides: the public
    Task call for TRTMC (``public_task_call_wall``) and the adapter call including input
    preparation and output decoding for references (``task-call-wall``).
    """

    observation: Mapping[str, Any]
    model_call_ms: float
    extra: Mapping[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    operation: str

    def describe(self) -> Mapping[str, Any]:
        """Identity reported by ``/v1/serving/info`` and every request record."""

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        """Run one operation request; artifacts use ``artifact_base`` as prefix."""

    def close(self) -> None:
        """Release the model and any worker process."""
