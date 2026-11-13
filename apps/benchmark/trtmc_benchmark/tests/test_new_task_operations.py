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
    ("points_to_semantic_segmentation", "segment_points", "segmented_points"),
    ("images_state_text_to_action_chunk", "control_chunk", "action_values"),
    ("initial_image_text_to_video", "generate_image", "generated_images"),
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


def test_point_cloud_requests_carry_the_points_file_and_its_shape(tmp_path):
    (tmp_path / "block.f32").write_bytes(b"\0" * 4 * 9 * 2)
    points = resolve_task_case("points_to_semantic_segmentation", {"test_points": "block.f32", "num_points": 2},
                               tmp_path)
    assert points.operation == "segment_points"
    assert points.request == {"points_path": str(tmp_path / "block.f32"), "num_points": 2, "input_dim": 9}
    with pytest.raises(BenchmarkError):
        resolve_task_case("points_to_semantic_segmentation", {"test_points": "block.f32"}, tmp_path)


def test_recorded_frames_send_named_cameras_state_instruction_and_noise(tmp_path):
    import json
    from array import array

    frame = tmp_path / "data/recorded_observation/ep0001_f002"
    frame.mkdir(parents=True)
    for name in ("image.png", "wrist_image.png", "image.processed.png"):
        (frame / name).write_bytes(b"fixture")
    (frame / "observation.json").write_text(json.dumps({"state": [0.5, -1.0], "task": "pick up the bowl"}))
    array("f", [0.25, -2.0]).tofile(open(tmp_path / "noise.f32", "wb"))
    case = {"inputs": {"frame": "data/recorded_observation/ep0001_f002", "noise": "noise.f32"}}
    resolved = resolve_task_case("images_state_text_to_action_chunk", case, tmp_path)
    assert resolved.operation == "control_chunk"
    assert resolved.request == {
        "images": [{"name": "observation.images.image", "image_path": str(frame / "image.png")},
                   {"name": "observation.images.wrist_image", "image_path": str(frame / "wrist_image.png")}],
        "state": [0.5, -1.0], "prompt": "pick up the bowl", "noise": [0.25, -2.0]}
    case["inputs"]["cameras"] = {"observation.images.image2": "wrist_image.png"}
    named = resolve_task_case("images_state_text_to_action_chunk", case, tmp_path)
    assert named.request["images"] == [{"name": "observation.images.image2",
                                        "image_path": str(frame / "wrist_image.png")}]
    case["inputs"]["cameras"] = {"observation.images.image": "missing.png"}
    with pytest.raises(BenchmarkError):
        resolve_task_case("images_state_text_to_action_chunk", case, tmp_path)


def test_image_to_video_requests_send_only_the_stated_controls_and_an_optional_image(tmp_path):
    (tmp_path / "start.png").write_bytes(b"fixture")
    case = {"test_prompt": "waves", "initial_image": True, "num_inference_steps": 50, "guidance_scale": 5.0,
            "seed": 42}
    flagged = resolve_task_case("initial_image_text_to_video", case, tmp_path)
    assert flagged.operation == "generate_image"
    assert flagged.request == {"prompt": "waves", "seed": 42, "num_steps": 50, "guidance_scale": 5.0}
    pictured = resolve_task_case("initial_image_text_to_video", {**case, "image": "start.png"}, tmp_path)
    assert pictured.request["image_path"] == str(tmp_path / "start.png")
