# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic bundle construction support for model-owned debug-runner tests."""

from __future__ import annotations

import json
import struct


def make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    vision_plan: bytes | None = None,
    extra_sections: dict[str, bytes] | None = None,
) -> bytes:
    """Build a minimal ``.trtfb`` bundle in memory."""
    magic = b"TRTFB\x00\x01\x00"
    sections: dict[str, dict[str, int]] = {}
    body = b""

    sections["engine_plan"] = {"offset": len(body), "size": len(engine_plan)}
    body += engine_plan

    if vision_plan is not None:
        sections["vision_engine_plan"] = {
            "offset": len(body),
            "size": len(vision_plan),
        }
        body += vision_plan

    if extra_sections:
        for name, data in extra_sections.items():
            sections[name] = {"offset": len(body), "size": len(data)}
            body += data

    header["sections"] = sections
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    return magic + header_len + header_json + body
