# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each displayed Cosmos request keeps only its own input media."""

import copy

import numpy as np

from families.cosmos3_edge.tests.reporting import record_request_inputs
from tools import e2e_evidence


def test_image_then_text_request_does_not_reuse_image_or_output(tmp_path):
    image = tmp_path / "image.rgb"
    np.arange(12, dtype=np.uint8).tofile(image)
    request = {"prompt": "Describe this image", "image": {"path": str(image), "height": 2, "width": 2}}
    original = copy.deepcopy(request)
    recorder = e2e_evidence.Evidence(tmp_path / "evidence", family="cosmos3_edge", case="report",
                                    source_revision="a" * 40, roots=(tmp_path,))
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        record_request_inputs(request)
        assert recorder.data["inputs"]["media"]
        e2e_evidence.record_evidence("native", {"text": "image output"})
        e2e_evidence.record_evidence("reference", {"text": "image reference"})
        record_request_inputs({"prompt": "What is 7 times 8?"})
    finally:
        e2e_evidence._ACTIVE.reset(token)
    assert request == original
    assert recorder.data["inputs"]["prompt"] == "What is 7 times 8?"
    assert recorder.data["inputs"]["media"] == []
    assert recorder.data["native"] is recorder.data["reference"] is None
    assert recorder.data["views"] == []
    assert any(row["name"] == "native" and row["value"] for row in recorder.data["observations"])
