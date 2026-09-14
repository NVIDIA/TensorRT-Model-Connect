# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned weight stripping for Qwen decoder builds.

Large GEMM operands are handed to TensorRT as null weight placeholders
(``Weights{dtype, nullptr, count}``) instead of real values. With
``kSTRIP_PLAN`` + ``kREFIT_INDIVIDUAL`` the builder then produces a plan that
carries no weight bytes, and the omitted values are supplied at load time
through ``IRefitter``.

The omitted arrays are recorded here, at the same chokepoint that substitutes
the placeholder, so what a refit sidecar carries is exactly the array that
would otherwise have been baked into the plan -- byte-identical by
construction rather than by audit.

Small constants (norms, biases, eps, ALiBi slopes) stay real and bake into the
plan: they are not worth a refit entry and keep the plan self-contained.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import numpy as np

# Only GEMM-scale operands are worth stripping. On Qwen3-0.6B this selects the
# 198 projection/embedding tensors (1433.5 MiB) and leaves 113 small tensors
# (564 KiB) baked.
DEFAULT_MIN_STRIP_BYTES = 1 << 20


class StripSession:
    """Collects the weights omitted from one engine build."""

    __slots__ = ("min_bytes", "_entries")

    def __init__(self, min_bytes: int = DEFAULT_MIN_STRIP_BYTES) -> None:
        self.min_bytes = int(min_bytes)
        self._entries: dict[str, np.ndarray] = {}

    def should_strip(self, name: str | None, values: np.ndarray) -> bool:
        return bool(name) and values.nbytes >= self.min_bytes

    def record(self, name: str, values: np.ndarray) -> None:
        """Record the exact array handed to TensorRT under *name*."""
        previous = self._entries.get(name)
        if previous is None:
            self._entries[name] = values
            return
        # The same weight legitimately appears once per graph; it must not
        # appear twice with different content.
        if previous.shape != values.shape or previous.dtype != values.dtype:
            raise ValueError(
                f"weight {name!r} recorded twice with different shape/dtype: "
                f"{previous.shape}/{previous.dtype} vs {values.shape}/{values.dtype}")

    @property
    def entries(self) -> dict[str, np.ndarray]:
        return self._entries

    @property
    def total_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self._entries.values())

    def manifest(self) -> dict:
        """Describe the stripped set without materializing the values."""
        return {
            "schema_version": 1,
            "min_strip_bytes": self.min_bytes,
            "count": len(self._entries),
            "total_bytes": self.total_bytes,
            "weights": {
                name: {
                    "shape": [int(dim) for dim in value.shape],
                    "dtype": str(value.dtype),
                    "bytes": int(value.nbytes),
                }
                for name, value in sorted(self._entries.items())
            },
        }


_ACTIVE: ContextVar["StripSession | None"] = ContextVar(
    "qwen_weight_strip_session", default=None)


def active() -> "StripSession | None":
    """Return the strip session for the engine build on this context, if any."""
    return _ACTIVE.get()


@contextmanager
def stripping(session: StripSession) -> Iterator[StripSession]:
    """Make *session* the active strip session for the enclosed build."""
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)
