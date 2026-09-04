# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-hash operational gates for the public SAM2 L0 case."""

from __future__ import annotations

import json
import struct
from pathlib import Path

_MAGIC = b"BUNDLE\x01\x00"
_PLANS = {
    "engine.plan",
    "prompt.plan",
    "recurrent.1.plan",
    "recurrent.2.plan",
    "recurrent.3.plan",
    "recurrent.4.plan",
}
_VARIANT = "public_sam2_1_small_with_synthetic_bbox_v1"
_FRAME_PIXELS = 1280 * 1088


def assert_bundle_contract(bundle: Path) -> None:
    with bundle.open("rb") as stream:
        assert stream.read(8) == _MAGIC
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
        sections = header["sections"]
        assert set(sections) == _PLANS | {"runtime.json"}
        for name in _PLANS:
            assert int(sections[name]["length"]) > 0
        runtime = sections["runtime.json"]
        stream.seek(16 + header_length + int(runtime["offset"]))
        config = json.loads(stream.read(int(runtime["length"])))
    assert config == {"sam2_checkpoint_variant": _VARIANT}


def assert_operational_receipt(receipt: dict) -> None:
    assert receipt["metadata_exact"] is True
    assert receipt["same_session_repeat_exact"] is True
    assert receipt["bbox_xyxy"] == [136.0, 160.0, 952.0, 1120.0]
    assert receipt["detector_score"] == 1.0
    assert receipt["label"] == 1
    assert receipt["binary_masks"] is True
    counts = receipt["mask_foreground_pixels"]
    minimum = _FRAME_PIXELS // 1000
    assert len(counts) == 5
    assert all(
        type(value) is int and minimum <= value <= _FRAME_PIXELS - minimum for value in counts
    )
    assert receipt["temporally_distinct_masks"] is True
