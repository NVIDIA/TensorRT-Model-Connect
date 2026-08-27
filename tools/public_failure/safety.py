# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defense-in-depth scans for data that a public report must never contain."""

from __future__ import annotations

import re
from typing import Mapping

from .contract import serialize_public_failure


class PublicFailureSafetyError(ValueError):
    """Raised when an otherwise valid payload resembles protected data."""


REGISTRY_IMAGE_REFERENCE_PATTERN = re.compile(
    r"\b[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?::[0-9]+)?/[A-Za-z0-9._/@:+-]+"
)


SENSITIVE_PATTERNS = (
    (
        "credential-like token",
        re.compile(
            r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|"
            r"glpat-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{16,})"
        ),
    ),
    ("authorization header", re.compile(r"\b(?:authorization|bearer)\s*[: ]", re.I)),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("URL", re.compile(r"\bhttps?://", re.I)),
    (
        "internal hostname",
        re.compile(
            r"\b[A-Za-z0-9.-]+\.(?:internal|local|corp|lan|cluster|nvidia\.com)\b",
            re.I,
        ),
    ),
    ("IP address", re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")),
    ("internal filesystem path", re.compile(r"/(?:home|workspace|mnt|var|tmp|builds|opt)/")),
    ("internal CI name", re.compile(r"\b(?:jenkins|slurm)\b", re.I)),
    ("long base64-like value", re.compile(r"\b[A-Za-z0-9+/]{80,}={0,2}\b")),
)


def assert_public_payload_safe(report: Mapping[str, object], document: bytes) -> None:
    """Scan canonical JSON and rendered text without echoing a matched secret."""
    candidates = (
        serialize_public_failure(report).decode("utf-8"),
        document.decode("utf-8"),
    )
    for candidate in candidates:
        for label, pattern in SENSITIVE_PATTERNS:
            if pattern.search(candidate):
                raise PublicFailureSafetyError(f"public failure payload contains a {label}")
    for failure in report.get("failures", ()):
        if not isinstance(failure, Mapping):
            continue
        excerpt = failure.get("excerpt", ())
        if not isinstance(excerpt, list):
            continue
        if any(
            isinstance(line, str) and REGISTRY_IMAGE_REFERENCE_PATTERN.search(line)
            for line in excerpt
        ):
            raise PublicFailureSafetyError(
                "public failure payload contains a registry image reference"
            )
