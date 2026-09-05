# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

import pytest

from families.sam2.tests.operational_oracle import (
    assert_bundle_contract,
    assert_operational_receipt,
)


def _bundle(path, runtime: dict) -> None:
    sections = {}
    data = []
    offset = 0
    for name in (
        "engine.plan",
        "prompt.plan",
        "recurrent.1.plan",
        "recurrent.2.plan",
        "recurrent.3.plan",
        "recurrent.4.plan",
    ):
        sections[name] = {"offset": offset, "length": 1}
        data.append(b"P")
        offset += 1
    encoded_runtime = json.dumps(runtime).encode()
    sections["runtime.json"] = {"offset": offset, "length": len(encoded_runtime)}
    data.append(encoded_runtime)
    header = json.dumps(
        {
            "format": 1,
            "family": "sam2",
            "task": "video_segmentation",
            "backend": "trt",
            "sections": sections,
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(b"BUNDLE\x01\x00" + struct.pack("<Q", len(header)) + header + b"".join(data))


def test_bundle_requires_exact_six_plan_public_variant(tmp_path) -> None:
    bundle = tmp_path / "sam2.bundle"
    _bundle(bundle, {"sam2_checkpoint_variant": "public_sam2_1_small_with_synthetic_bbox_v1"})
    assert_bundle_contract(bundle)


def test_operational_gate_rejects_non_distinct_masks() -> None:
    receipt = {
        "metadata_exact": True,
        "same_session_repeat_exact": True,
        "bbox_xyxy": [136.0, 160.0, 952.0, 1120.0],
        "detector_score": 1.0,
        "label": 1,
        "binary_masks": True,
        "mask_foreground_pixels": [2000] * 5,
        "temporally_distinct_masks": False,
        "device_mask_ordinal": 0,
        "device_metadata_exact": True,
        "device_masks_match_host": True,
    }
    with pytest.raises(AssertionError):
        assert_operational_receipt(receipt)


def test_operational_gate_requires_device_masks_to_match_host() -> None:
    receipt = {
        "metadata_exact": True,
        "same_session_repeat_exact": True,
        "bbox_xyxy": [136.0, 160.0, 952.0, 1120.0],
        "detector_score": 1.0,
        "label": 1,
        "binary_masks": True,
        "mask_foreground_pixels": [2000] * 5,
        "temporally_distinct_masks": True,
        "device_mask_ordinal": 0,
        "device_metadata_exact": True,
        "device_masks_match_host": False,
    }
    with pytest.raises(AssertionError):
        assert_operational_receipt(receipt)
