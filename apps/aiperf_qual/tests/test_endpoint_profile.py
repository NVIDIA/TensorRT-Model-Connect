# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest

from trtmc_aiperf_qual.endpoint_profile import check_output, input_key


def test_singleton_embedding_wire_forms_keep_the_same_input_identity():
    assert input_key("embeddings", {"input": "hello"}) == input_key("embeddings", {"input": ["hello"]})
    with pytest.raises(ValueError):
        input_key("embeddings", {"input": ["hello", "second input"]})


def test_media_identity_uses_original_bytes():
    def image(data):
        return {"input": [{"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}]}

    assert input_key("detection", image(b"original")) != input_key("detection", image(b"recompressed"))


def test_embedding_check_rejects_truncation_and_nonfinite_output():
    native = {"values": [0.1, 0.2]}
    assert check_output("embeddings", {"data": [{"embedding": [0.1, 0.2]}]}, native) == 2
    for vector in ([0.1], [0.1, float("nan")], [0.1, 0.3]):
        with pytest.raises(ValueError):
            check_output("embeddings", {"data": [{"embedding": vector}]}, native)


def test_reranking_checks_original_indices_and_scores():
    native = {"scores": [0.2, 0.8]}
    correct = {"rankings": [{"index": 1, "logit": 0.8}, {"index": 0, "logit": 0.2}]}
    assert check_output("reranking", correct, native) == 2
    with pytest.raises(ValueError):
        check_output("reranking", {"rankings": correct["rankings"][:1]}, native)


def test_detection_checks_all_boxes_and_preserves_labels_and_outside_coordinates():
    native = {"boxes": [-10, 20, 110, 80], "scores": [0.75], "class_ids": [90]}
    box = {"x_min": -0.1, "y_min": 0.1, "x_max": 1.1, "y_max": 0.4, "confidence": 0.75}
    correct = {"data": [{"index": 0, "bounding_boxes": {"90": [box]}}]}
    assert check_output("detection", correct, native, (100, 200)) == 5
    with pytest.raises(ValueError):
        check_output("detection", {"data": [{"index": 0, "bounding_boxes": {"0": [box]}}]}, native, (100, 200))
    with pytest.raises(ValueError):
        check_output("detection", {"data": [{"index": 0, "bounding_boxes": {"90": []}}]}, native, (100, 200))


def test_transcription_check_requires_the_complete_native_text():
    assert check_output("transcription", {"text": "Full transcript."}, {"text": "Full transcript."}) == 1
    with pytest.raises(ValueError):
        check_output("transcription", {"text": "Full"}, {"text": "Full transcript."})
