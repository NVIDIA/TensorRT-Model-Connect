# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-backed configuration for the story-scene application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


def _bounded_integer(
    value: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if not value.isascii() or not value.isdigit():
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _plain_value(value: str, *, name: str) -> str:
    if not value or "\x00" in value:
        raise ValueError(f"{name} must not be empty")
    return value


def _context_parallel_size(value: str) -> int:
    result = _bounded_integer(
        value,
        name="COSMOS3_CP_SIZE",
        minimum=1,
        maximum=8,
    )
    if result not in {1, 2, 4, 8}:
        raise ValueError("COSMOS3_CP_SIZE must be one of 1, 2, 4, or 8")
    return result


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration whose public values map one-to-one to supported env vars."""

    trtmc_bin: str = "trtmc"
    cosmos3_bundle: Path = Path("/models/cosmos3.trtfb")
    cosmos3_cp_size: int = 1
    output_root: Path = Path("/outputs")
    host: str = "0.0.0.0"
    port: int = 8080

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AppConfig:
        values = os.environ if environ is None else environ
        trtmc_bin = _plain_value(
            values.get("TRTMC_BIN", "trtmc"),
            name="TRTMC_BIN",
        )
        bundle = _plain_value(
            values.get("COSMOS3_BUNDLE", "/models/cosmos3.trtfb"),
            name="COSMOS3_BUNDLE",
        )
        output_root = _plain_value(
            values.get("OUTPUT_ROOT", "/outputs"),
            name="OUTPUT_ROOT",
        )
        host = _plain_value(values.get("HOST", "0.0.0.0"), name="HOST")
        return cls(
            trtmc_bin=trtmc_bin,
            cosmos3_bundle=Path(bundle),
            cosmos3_cp_size=_context_parallel_size(
                values.get("COSMOS3_CP_SIZE", "1")
            ),
            output_root=Path(output_root),
            host=host,
            port=_bounded_integer(
                values.get("PORT", "8080"),
                name="PORT",
                minimum=1,
                maximum=65535,
            ),
        )

    @property
    def static_root(self) -> Path:
        return Path(__file__).resolve().parent.parent / "static"
