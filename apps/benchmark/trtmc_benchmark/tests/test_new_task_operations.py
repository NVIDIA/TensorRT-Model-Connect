# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from trtmc_benchmark.operations import operation_for_name
from trtmc_benchmark.task_adapters import resolve_task_case, supported_tasks
from trtmc_benchmark.types import BenchmarkError


@pytest.mark.parametrize("task,operation,observed", [
    ("audio_to_class_scores", "classify_audio", "classified_clips"),
    ("text_image_to_relevance", "relevance", "scored_pairs"),
    ("masked_text_to_token_scores", "token_scores", "scored_sequences"),
    ("monocular_depth", "depth", "depth_images"),
])
def test_new_tasks_have_an_operation_and_its_rate(task, operation, observed):
    assert task in supported_tasks()
    assert operation_for_name(operation).rate_metrics[0].observation_field == observed


def test_relevance_scores_every_query_of_the_case_against_its_image(tmp_path):
    (tmp_path / "car.jpeg").write_bytes(b"fixture")
    case = {"test_image": "car.jpeg", "query": "a photo of a car", "negative_query": "a photo of a cat"}
    resolved = resolve_task_case("text_image_to_relevance", case, tmp_path)
    assert resolved.operation == "relevance"
    assert resolved.request == {"image_path": str(tmp_path / "car.jpeg"),
                                "queries": ["a photo of a car", "a photo of a cat"]}
    listed = resolve_task_case("text_image_to_relevance", {"test_image": "car.jpeg", "queries": ["a", "b", "c"]},
                               tmp_path)
    assert listed.request["queries"] == ["a", "b", "c"]
    with pytest.raises(BenchmarkError):
        resolve_task_case("text_image_to_relevance", {"test_image": "car.jpeg"}, tmp_path)


def test_audio_tokens_and_depth_requests_name_their_one_input(tmp_path):
    for name in ("speech.wav", "scene.jpeg"):
        (tmp_path / name).write_bytes(b"fixture")
    audio = resolve_task_case("audio_to_class_scores", {"test_input_audio": "speech.wav", "top_k": 3}, tmp_path)
    assert audio.request == {"audio_path": str(tmp_path / "speech.wav")}
    depth = resolve_task_case("monocular_depth", {"test_image": "scene.jpeg"}, tmp_path)
    assert depth.request == {"image_path": str(tmp_path / "scene.jpeg")}
    tokens = resolve_task_case("masked_text_to_token_scores", {"prompt": "MKT<mask>V"}, tmp_path)
    assert tokens.request == {"prompt": "MKT<mask>V"}
