# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import trtmc_benchmark.builder as benchmark_builder
from trtmc_benchmark.builder import BundleBuilder, _build_command
from trtmc_benchmark.catalog import (
    ManifestCatalog,
    default_manifest_root,
    resolve_case,
)
from trtmc_benchmark.cli import main
from trtmc_benchmark.metrics import reduce_metrics
from trtmc_benchmark.report import generate_collection_report
from trtmc_benchmark.service import BenchmarkService
from trtmc_benchmark.types import BenchmarkError
from trtmc_benchmark.worker import find_worker
from trtmc_benchmark.task_adapters import resolve_task_case


REPO = Path(__file__).resolve().parents[4]


def test_catalog_reads_family_owned_manifests_without_a_registry() -> None:
    entries = ManifestCatalog(REPO / "families").entries()
    assert {entry.family for entry in entries} == {
        path.name
        for path in (REPO / "families").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    distilgpt2 = next(entry for entry in entries if entry.name == "distilgpt2")
    assert distilgpt2.operation == "generate"
    assert distilgpt2.status == "ready"


def test_case_resolves_current_task_and_manifest_fields(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert case.operation == "generate"
    assert case.request["prompt"] == "Hello, I'm a language model"
    assert case.request["max_new_tokens"] == 12
    assert case.measurement.timing_scope == "public_task_call_wall"


def test_forecast_case_uses_public_forecast_request(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("chronos-bolt-tiny-official")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert case.operation == "solve"
    assert case.request["past_values"][:2] == [100.1, 100.15]


@pytest.mark.parametrize("task", [
    "text_continuation", "conditional_text_generation", "corrupted_text_reconstruction",
    "text_summarization",
])
def test_semantic_text_benchmark_keeps_family_config_and_presence(tmp_path: Path, task: str) -> None:
    default = resolve_task_case(task, {"prompt": "Hello"}, tmp_path)
    assert default.operation == "generate"
    assert default.request == {"prompt": "Hello"}
    explicit = resolve_task_case(task, {
        "prompt": "Hello", "max_new_tokens": 0, "top_p": "wrong type", "use_chat_template": False,
        "config": {"suffix": "", "stop_ids": [0, 2], "top_p": 0.8},
    }, tmp_path)
    assert explicit.request["max_new_tokens"] == 0
    assert explicit.request["use_chat_template"] is False
    assert explicit.request["top_p"] == "wrong type"
    assert explicit.request["config"] == {"suffix": "", "stop_ids": [0, 2], "top_p": 0.8}
    # A repeated flat/nested key is preserved for native family rejection.
    special = resolve_task_case(task, {"prompt": "Hello", "inputs": {
        "generation_mode": 17, "block_length": 2.75, "threshold": "0.8",
    }}, tmp_path)
    assert special.request["text_generation_mode"] == 17
    assert special.request["block_length"] == 2.75
    assert special.request["confidence_threshold"] == "0.8"


@pytest.mark.parametrize("task", [
    "series_to_point_forecast", "series_to_quantile_forecast", "series_to_point_and_quantile_forecast",
])
def test_semantic_forecast_does_not_override_family_frequency(tmp_path: Path, task: str) -> None:
    resolved = resolve_task_case(task, {"inputs": {"past_values": [1, 2, 3]}}, tmp_path)
    assert resolved.operation == "solve"
    assert resolved.request == {"past_values": [1.0, 2.0, 3.0]}
    explicit = resolve_task_case(task, {"inputs": {"past_values": [1, 2], "frequency": 0}}, tmp_path)
    assert explicit.request["frequency"] == 0


@pytest.mark.parametrize("task,operation", [
    ("text_to_audio", "generate_audio"), ("text_to_speech", "generate_audio"),
    ("streaming_text_to_speech", "generate_audio"), ("speech_to_speech_response", "speak"),
    ("speech_transcription", "transcribe"), ("speech_translation", "transcribe"),
    ("streaming_speech_transcription", "transcribe"),
])
def test_semantic_audio_preserves_types_and_family_defaults(
    tmp_path: Path, task: str, operation: str
) -> None:
    (tmp_path / "input.wav").write_bytes(b"fixture")
    base = {"prompt": "Hello", "inputs": {"audio": "input.wav"}}
    resolved = resolve_task_case(task, base, tmp_path)
    assert resolved.operation == operation
    assert resolved.request == (
        {"prompt": "Hello"} if operation == "generate_audio"
        else {"audio_path": str(tmp_path / "input.wav")}
    )
    explicit = resolve_task_case(task, {
        **base, "max_new_tokens": 0, "seed": 2**40, "speaker": 3, "language": "",
        "streaming": "false", "chunk_ms": 0.75, "target_language": None,
        "config": {"normalize": False, "suffix": "", "sampling_steps": [0.0, 0.5]},
    }, tmp_path)
    assert explicit.request["max_new_tokens"] == 0
    assert explicit.request["seed"] == 2**40
    assert explicit.request["speaker"] == 3
    assert explicit.request["language"] == ""
    assert explicit.request["streaming"] == "false"
    assert explicit.request["chunk_ms"] == 0.75
    assert explicit.request["target_language"] is None
    assert explicit.request["config"] == {
        "normalize": False, "suffix": "", "sampling_steps": [0.0, 0.5],
    }


def test_semantic_speech_limit_alias_preserves_value_and_rejects_duplicate(tmp_path: Path) -> None:
    (tmp_path / "input.wav").write_bytes(b"fixture")
    case = {"inputs": {"audio": "input.wav", "tail_frames": 0}, "speech_test_max_frames": 2.5}
    result = resolve_task_case("speech_to_speech_response", case, tmp_path)
    assert result.request["max_new_tokens"] == 2.5
    assert result.request["tail_frames"] == 0
    with pytest.raises(BenchmarkError, match="duplicate limits"):
        resolve_task_case("speech_to_speech_response", {**case, "max_new_tokens": 5}, tmp_path)


def test_semantic_vlm_benchmark_keeps_image_and_rejects_bad_config(tmp_path: Path) -> None:
    image = tmp_path / "image.ppm"
    image.write_bytes(b"P6\n1 1\n255\nabc")
    case = resolve_task_case("images_text_to_text", {"prompt": "Describe", "image": str(image)}, tmp_path)
    assert case.request == {"prompt": "Describe", "image_path": str(image)}
    with pytest.raises(BenchmarkError, match="config must be an object"):
        resolve_task_case("text_continuation", {"prompt": "Hello", "config": []}, tmp_path)
    with pytest.raises(BenchmarkError, match="requires a semantic Task"):
        resolve_task_case("text_generation", {"prompt": "Hello", "config": {}}, tmp_path)


@pytest.fixture
def sdk_assets(tmp_path: Path) -> Path:
    for name in ("image.ppm", "second.ppm"):
        (tmp_path / name).write_bytes(b"P6\n1 1\n255\nabc")
    for name in ("state.f32", "latents.f32"):
        (tmp_path / name).write_bytes(b"\0" * 16)
    return tmp_path


@pytest.mark.parametrize("task,operation,case,expected", [
    ("text_to_image", "generate_image", {"prompt": ""}, {"prompt": "", "media_type": "image"}),
    ("images_text_to_image_edit", "generate_image", {"prompt": "edit", "image": "image.ppm"},
     {"prompt": "edit", "image_path": "image.ppm", "media_type": "image"}),
    ("batch_text_to_image", "generate_image", {"inputs": {"batch_prompts": ["a", "b"]}},
     {"prompt": ["a", "b"], "media_type": "image"}),
    ("text_to_video", "generate_image", {"prompt": "video"}, {"prompt": "video", "media_type": "video"}),
    ("image_text_action_to_video", "generate_image", {"prompt": "", "image": "image.ppm",
     "action": "", "camera_intrinsics": [10, 20.5, 0, 0]},
     {"prompt": "", "image_path": "image.ppm", "action": "", "camera_intrinsics": [10, 20.5, 0, 0], "media_type": "video"}),
    ("image_to_class_scores", "classify", {"test_image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_to_token_and_pooled_features", "extract_features", {"image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_to_token_features", "extract_features", {"image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_to_pooled_features", "extract_features", {"image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_to_spatial_features", "extract_features", {"image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_to_semantic_segmentation", "segment", {"image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_points_to_masks", "segment", {"image": "image.ppm"}, {"image_path": "image.ppm"}),
    ("image_text_to_instance_masks", "segment_prompted", {"image": "image.ppm", "prompt": "cup"},
     {"image_path": "image.ppm", "prompt": "cup"}),
    ("stereo_images_to_disparity", "disparity", {"inputs": {"left_image": "image.ppm", "right_image": "second.ppm"}},
     {"left_image_path": "image.ppm", "right_image_path": "second.ppm"}),
    ("text_to_pooled_features", "encode", {"prompt": "text"}, {"prompt": "text"}),
    ("text_to_token_features", "encode", {"prompt": "text"}, {"prompt": "text"}),
    ("text_to_embedding", "embed", {"prompt": "text"}, {"prompt": "text"}),
    ("text_query_documents_to_relevance", "rerank", {"inputs": {"query": "q", "documents": ["b", "a", ""]}},
     {"query": "q", "documents": ["b", "a", ""]}),
    ("image_state_to_action_chunk", "control", {"inputs": {"image": "image.ppm", "state": "state.f32"}},
     {"image_path": "image.ppm", "state_path": "state.f32"}),
])
def test_remaining_semantic_routes_have_only_required_inputs(
    sdk_assets: Path, task: str, operation: str, case: dict, expected: dict
) -> None:
    expected = {
        key: str(sdk_assets / value) if key.endswith("_path") else value
        for key, value in expected.items()
    }
    resolved = resolve_task_case(task, case, sdk_assets)
    assert resolved.operation == operation
    assert resolved.request == expected
    if "media_type" in expected:
        assert resolved.sources["media_type"] == "task default"


def test_semantic_media_preserves_explicit_controls_and_order(sdk_assets: Path) -> None:
    case = {
        "prompt": "edit", "inputs": {"image_paths": ["second.ppm", "image.ppm"], "num_sampling_steps": 2.75},
        "seed": -1, "negative_prompt": "", "height": 0, "width": "wrong",
        "guidance_scale": False, "cfg_scale": 0.0, "batch_size": 1,
        "initial_latents_path": "latents.f32", "media_type": "image",
        "config": {"family_switch": False, "family_zero": 0},
    }
    value = resolve_task_case("images_text_to_image_edit", case, sdk_assets)
    assert value.request["image_paths"] == [str(sdk_assets / "second.ppm"), str(sdk_assets / "image.ppm")]
    assert value.request["initial_latents_path"] == str(sdk_assets / "latents.f32")
    assert value.request["num_steps"] == 2.75
    assert value.request["guidance_scale"] is False
    assert value.request["width"] == "wrong"
    assert value.request["negative_prompt"] == ""
    assert value.request["seed"] == -1
    assert value.request["config"] == {"family_switch": False, "family_zero": 0}
    assert value.sources["media_type"] == "family manifest"


def test_semantic_world_does_not_omit_explicit_frame_count(sdk_assets: Path) -> None:
    base = {"prompt": "move", "image": "image.ppm", "action": "w-2",
            "camera_intrinsics": [1, 0, 0, 0, 2.5, 0, 0, 0, 1]}
    absent = resolve_task_case("image_text_action_to_video", base, sdk_assets).request
    assert "num_frames" not in absent and "num_steps" not in absent and "fps" not in absent
    value = resolve_task_case("image_text_action_to_video", {
        **base, "inputs": {"num_frames": 0, "num_inference_steps": "bad", "fps": 2.5},
        "translation_speed": 0.0, "rotation_speed_deg": 1.5, "no_action_overlay": False,
        "initial_latents_path": "latents.f32",
    }, sdk_assets).request
    assert value["num_frames"] == 0 and value["num_steps"] == "bad" and value["fps"] == 2.5
    assert value["no_action_overlay"] is False
    assert value["camera_intrinsics"] == base["camera_intrinsics"]


def test_semantic_image_batch_keeps_signed_seeds_and_item_config(sdk_assets: Path) -> None:
    config = [{"quality": 0, "normalize": False}, {"quality": 2.75, "empty": ""}]
    value = resolve_task_case("batch_text_to_image", {
        "inputs": {"batch_prompts": ["a", ""], "batch_seeds": [-1, 2**40], "item_configs": config},
        "batch_size": 2, "guidance_scale": 0.0, "config": {"steps": 4},
    }, sdk_assets).request
    assert value == {"prompt": ["a", ""], "media_type": "image", "batch_size": 2,
                     "seeds": [-1, 2**40], "item_configs": config, "guidance_scale": 0.0,
                     "config": {"steps": 4}}
    default = resolve_task_case("batch_text_to_image", {"prompt": ["a", "b"]}, sdk_assets).request
    assert "seeds" not in default and "seed" not in default and "item_configs" not in default


@pytest.mark.parametrize("change", [
    {"seed": 1, "seeds": [1, 2]},
    {"seeds": [1, 2], "config": {"seed": 1}},
    {"seeds": [1, 2], "item_configs": [{"seed": 1}, {}]},
    {"guidance_scale": 1.0, "item_configs": [{}, {"guidance_scale": 1.0}]},
    {"config": {"x": False}, "item_configs": [{"x": False}, {}]},
    {"num_steps": 3, "config": {"num_steps": 3}},
    {"num_steps": 3, "inputs": {"num_inference_steps": 3}},
    {"guidance_scale": 1.0, "inputs": {"guidance_scale": 1.0}},
])
def test_semantic_duplicate_controls_are_errors_not_overrides(sdk_assets: Path, change: dict) -> None:
    with pytest.raises(BenchmarkError, match="duplicate"):
        resolve_task_case("batch_text_to_image", {"prompt": ["a", "b"], **change}, sdk_assets)


@pytest.mark.parametrize("seeds", [[1], [1, 2, 3], [True, 2], [1.5, 2], ["1", 2], [2**63, 2], [-(2**63)-1, 2], None])
def test_semantic_batch_seed_shape_and_integer_domain(sdk_assets: Path, seeds) -> None:
    with pytest.raises(BenchmarkError, match="signed 64-bit integer"):
        resolve_task_case("batch_text_to_image", {"prompt": ["a", "b"], "seeds": seeds}, sdk_assets)


@pytest.mark.parametrize("configs", [[], [{}], [{}, {}, {}], [None, {}], [[], {}], {}])
def test_semantic_item_config_requires_one_object_per_prompt(sdk_assets: Path, configs) -> None:
    with pytest.raises(BenchmarkError, match="one object per prompt"):
        resolve_task_case("batch_text_to_image", {"prompt": ["a", "b"], "item_configs": configs}, sdk_assets)


@pytest.mark.parametrize("task,case,count", [
    ("text_to_image", {"prompt": "x"}, 2),
    ("text_to_video", {"prompt": "x"}, 0),
    ("image_to_class_scores", {"image": "image.ppm"}, True),
    ("stereo_images_to_disparity", {"left_image": "image.ppm", "right_image": "second.ppm"}, 1.0),
    ("images_text_to_image_edit", {"prompt": "x", "images": ["image.ppm", "second.ppm"]}, 2),
    ("text_query_documents_to_relevance", {"query": "x", "documents": ["a", "b"]}, 2),
    ("batch_text_to_image", {"prompt": ["a", "b"]}, 1),
])
def test_semantic_batch_count_is_request_count_not_frames_or_documents(
    sdk_assets: Path, task: str, case: dict, count
) -> None:
    with pytest.raises(BenchmarkError, match="actual request count"):
        resolve_task_case(task, {**case, "batch_size": count}, sdk_assets)


@pytest.mark.parametrize("role", ["default", "query", "document"])
def test_semantic_embedding_role_is_typed(sdk_assets: Path, role: str) -> None:
    value = resolve_task_case("text_to_embedding", {"prompt": "x", "inputs": {"role": role}}, sdk_assets)
    assert value.request == {"prompt": "x", "role": role}


@pytest.mark.parametrize("task,role", [("text_to_embedding", 1), ("text_to_embedding", "title"), ("text_to_token_features", "query")])
def test_semantic_embedding_role_cannot_change_encoder_semantics(sdk_assets: Path, task: str, role) -> None:
    with pytest.raises(BenchmarkError, match="embedding role"):
        resolve_task_case(task, {"prompt": "x", "role": role}, sdk_assets)


def test_semantic_point_input_keeps_types_and_requires_prompted_operation(sdk_assets: Path) -> None:
    case = {"image": "image.ppm", "inputs": {"point_x": 0, "point_y": 1.0, "is_foreground": False}}
    value = resolve_task_case("image_points_to_masks", case, sdk_assets, operation="segment_prompted")
    assert value.operation == "segment_prompted"
    assert value.request == {"image_path": str(sdk_assets / "image.ppm"), "point_x": 0, "point_y": 1.0, "is_foreground": False}
    with pytest.raises(BenchmarkError, match="center helper"):
        resolve_task_case("image_points_to_masks", case, sdk_assets)


@pytest.mark.parametrize("controls", [
    {"point_x": "0.5"}, {"point_y": True}, {"point_x": float("nan")},
    {"point_y": float("inf")}, {"is_foreground": "false"}, {"is_foreground": 1}, {"prompt": "text"},
])
def test_semantic_point_input_rejects_coercion_and_text_conflict(sdk_assets: Path, controls: dict) -> None:
    with pytest.raises(BenchmarkError):
        resolve_task_case("image_points_to_masks", {"image": "image.ppm", **controls}, sdk_assets,
                          operation="segment_prompted")


def test_semantic_points_preserve_finite_outside_image_fractions(sdk_assets: Path) -> None:
    value = resolve_task_case("image_points_to_masks", {
        "image": "image.ppm", "point_x": -0.25, "point_y": 1.1, "is_foreground": False,
    }, sdk_assets, operation="segment_prompted")
    assert value.request["point_x"] == -0.25
    assert value.request["point_y"] == 1.1
    assert value.request["is_foreground"] is False


def test_semantic_rerank_accepts_empty_document_list_without_native_batch_claim(sdk_assets: Path) -> None:
    value = resolve_task_case("text_query_documents_to_relevance", {
        "inputs": {"query": "q", "documents": []}, "batch_size": 1,
    }, sdk_assets)
    assert value.operation == "rerank"
    assert value.request == {"query": "q", "documents": [], "batch_size": 1}
    # This is one list-scoring request, including when that list is empty.
    with pytest.raises(BenchmarkError, match="actual request count 1"):
        resolve_task_case("text_query_documents_to_relevance", {
            "query": "q", "documents": [], "batch_size": 0,
        }, sdk_assets)
    # The existing legacy helper's narrower historical behavior is unchanged.
    with pytest.raises(BenchmarkError, match="non-empty inputs.documents"):
        resolve_task_case("reranking", {"inputs": {"query": "q", "documents": []}}, sdk_assets)


@pytest.mark.parametrize("task,case", [
    ("text_to_image", {"prompt": "x", "media_type": "video"}),
    ("text_to_video", {"prompt": "x", "media_type": "image"}),
    ("text_to_image", {"prompt": "x", "image": "image.ppm"}),
    ("text_to_image", {"prompt": "x", "action": "w"}),
    ("text_to_image", {"prompt": "x", "batch_prompts": ["a"]}),
    ("images_text_to_image_edit", {"prompt": "x", "image": "image.ppm", "images": ["second.ppm"]}),
    ("batch_text_to_image", {"prompt": ["x"], "initial_latents_path": "latents.f32"}),
    ("batch_text_to_image", {"prompt": ["x"], "prompt_repeat": {"text": "a", "count": 1}}),
    ("image_text_to_instance_masks", {"prompt": "x", "image": "image.ppm", "point_x": 0.5}),
    ("text_query_documents_to_relevance", {"query": "x", "documents": ["a", 7]}),
    ("image_state_to_action_chunk", {"image": "image.ppm", "state": "missing.f32"}),
])
def test_semantic_routes_reject_conflicting_or_missing_operands(sdk_assets: Path, task: str, case: dict) -> None:
    with pytest.raises(BenchmarkError):
        resolve_task_case(task, case, sdk_assets)


@pytest.mark.parametrize("camera", [[True, 1, 2, 3], ["1", 2, 3, 4], [float("inf"), 2, 3, 4], "1,2,3,4", None])
def test_semantic_world_intrinsics_are_numbers_not_coerced_values(sdk_assets: Path, camera) -> None:
    with pytest.raises(BenchmarkError, match="numeric array"):
        resolve_task_case("image_text_action_to_video", {
            "prompt": "x", "image": "image.ppm", "action": "w", "camera_intrinsics": camera,
        }, sdk_assets)


def test_semantic_prompt_file_and_repeat_preserve_string_contract(sdk_assets: Path) -> None:
    (sdk_assets / "prompt.json").write_text(json.dumps({"prompt": "a\u0000b"}))
    assert resolve_task_case("text_to_image", {"prompt_file": "prompt.json"}, sdk_assets).request["prompt"] == "a\0b"
    repeat = {"text": "a", "separator": ":", "suffix": "!", "count": 2}
    assert resolve_task_case("text_to_embedding", {"prompt_repeat": repeat}, sdk_assets).request["prompt"] == "a:a!"
    for invalid in ({**repeat, "count": 2.5}, {**repeat, "count": True}, {**repeat, "text": 17}):
        with pytest.raises(BenchmarkError):
            resolve_task_case("text_to_embedding", {"prompt_repeat": invalid}, sdk_assets)
    with pytest.raises(BenchmarkError, match="duplicate"):
        resolve_task_case("text_to_image", {"prompt": "x", "inputs": {"prompt": "x"}}, sdk_assets)
    (sdk_assets / "prompt.json").write_text(json.dumps({"prompt": 17}))
    with pytest.raises(BenchmarkError, match="string prompt"):
        resolve_task_case("text_to_image", {"prompt_file": "prompt.json"}, sdk_assets)


def test_stereo_benchmark_uses_family_owned_images(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("fast-foundation-stereo")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert Path(case.request["left_image_path"]).is_file()
    assert Path(case.request["right_image_path"]).is_file()


def test_robot_control_benchmark_uses_family_owned_observation(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("act-aloha-sim-transfer-cube")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert case.operation == "control"
    assert Path(case.request["image_path"]).is_file()
    assert Path(case.request["state_path"]).is_file()


def test_build_command_is_the_current_closed_build_request(tmp_path: Path) -> None:
    manifest_path = tmp_path / "model.json"
    manifest_path.write_text(json.dumps({
        "name": "example-model", "bundle": "model.bundle", "family": "example_owner",
        "task": "text_continuation", "precision": "fp32",
        "testcases": [{"name": "example", "prompt": "Hello"}],
    }))
    model = ManifestCatalog(tmp_path).resolve(str(manifest_path))
    case = resolve_case(model, tmp_path / "model.bundle")
    command = _build_command(model, tmp_path / "checkpoint", tmp_path / "model.bundle", (case,))
    assert command[:4] == (
        sys.executable,
        "-m",
        "tensorrt_model_connect",
        "build",
    )
    assert "--family" not in command
    assert command[command.index("--task") + 1] == "text_continuation"
    joined = " ".join(command).lower()
    assert "profile" not in joined
    assert "source-revision" not in joined


def test_build_command_passes_manifest_backend_and_dynamic_kv_cache(tmp_path: Path) -> None:
    manifest = json.loads(
        (REPO / "families/llama/tests/manifests/minitron-4b-width-l0.json").read_text()
    )
    manifest["backend"] = "trt_rtx"
    manifest_path = tmp_path / "model.json"
    manifest_path.write_text(json.dumps(manifest))
    model = ManifestCatalog(tmp_path).resolve(str(manifest_path))

    assert model.build_settings["backend"] == "trt_rtx"
    assert model.build_settings["dynamic_kv_cache"] is True
    case = resolve_case(model, tmp_path / "model.bundle")

    command = _build_command(model, tmp_path / "checkpoint", tmp_path / "model.bundle", (case,))

    assert command[command.index("--backend") + 1] == "trt_rtx"
    assert command.count("--dynamic-kv-cache") == 1


def test_bundle_builder_uses_core_model_resolution(tmp_path: Path, monkeypatch) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "model.bundle")
    checkpoint = tmp_path / "resolved-checkpoint"
    checkpoint.mkdir()
    calls = []

    def resolve_model(value: str, revision: str | None) -> Path:
        calls.append((value, revision))
        return checkpoint

    monkeypatch.setattr(benchmark_builder, "_resolve_model", resolve_model)
    plan = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    assert calls == [(model.hf_id, model.hf_revision or None)]
    assert plan.model_dir == checkpoint


@pytest.mark.parametrize("selector", ["model", "family", "default"])
def test_bundle_builder_keeps_explicit_model_dir_cli_behavior(
    tmp_path: Path, monkeypatch, selector: str
) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "model.bundle")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls = []

    def resolve_model(value: str, revision: str | None) -> Path:
        calls.append((value, revision))
        return Path(value)

    monkeypatch.setattr(benchmark_builder, "_resolve_model", resolve_model)
    key = {"model": model.name, "family": model.family, "default": ""}[selector]
    plan = BundleBuilder(tmp_path / "cache", model_dirs={key: checkpoint})._plan(model, (case,))

    assert calls == [(str(checkpoint.resolve()), None)]
    assert plan.model_dir == checkpoint.resolve()


def test_bundle_builder_has_no_second_model_resolver() -> None:
    source = (REPO / "apps/benchmark/trtmc_benchmark/builder.py").read_text()
    assert "snapshot_download" not in source
    assert "_MODEL_DIR" not in source
    assert "repository / model.hf_id" not in source


def _worker(tmp_path: Path) -> Path:
    path = tmp_path / "worker"
    path.write_text(
        """#!/usr/bin/env python3
import json, sys
request = json.load(open(sys.argv[sys.argv.index('--request') + 1]))
output = sys.argv[sys.argv.index('--output') + 1]
count = request['measurement']['iterations']
result = {
  'schema_version': 'trtmc.benchmark-worker-result/v2',
  'status': 'completed',
  'case_name': request['case_name'],
  'operation': request['operation'],
  'timing_scope': 'public_task_call_wall',
  'asset_loading_included': request['measurement']['asset_loading_included'],
  'load_ms': 1.0,
  'observations': [
    {'runtime_e2e_wall_ms': 2.0, 'output_tokens': 3, 'prefill_ms': 0.5, 'decode_ms': 1.0}
    for _ in range(count)
  ],
  'output_summary': {'text': 'ok'},
}
json.dump(result, open(output, 'w'))
"""
    )
    path.chmod(0o755)
    return path


def test_service_runs_worker_and_writes_reports(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    case = resolve_case(
        model,
        bundle,
        overrides={"measurement.warmup": 0, "measurement.iterations": 2, "telemetry.gpu": "off"},
    ).with_values(runtime_root=runtime)
    output = tmp_path / "results"
    result = BenchmarkService(_worker(tmp_path)).run((case,), output)
    assert result["status"] == "completed"
    assert result["cells"][0]["metrics"]["latency_ms"]["p50"] == 2.0
    assert result["cells"][0]["samples_ms"] == [2.0, 2.0]
    assert (output / "result.json").is_file()
    assert (output / "report.html").is_file()


def test_case_resolution_rejects_unknown_telemetry_override(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")

    with pytest.raises(BenchmarkError, match="unknown telemetry field"):
        resolve_case(
            model,
            tmp_path / "model.bundle",
            overrides={"telemetry.typo": 100},
        )


def test_metrics_keep_task_specific_rates() -> None:
    metrics = reduce_metrics(
        "generate_audio",
        [
            {
                "runtime_e2e_wall_ms": 100.0,
                "output_audio_seconds": 0.2,
                "output_samples": 4800,
            }
        ],
    )
    assert metrics["audio_seconds_per_s"] == pytest.approx(2.0)
    assert metrics["realtime_factor"] == pytest.approx(0.5)


def test_collection_rejects_duplicate_run_id_without_content_fingerprints(
    tmp_path: Path,
) -> None:
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        (root / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "trtmc.benchmark-run/v2",
                    "run_id": "same",
                    "status": "completed",
                    "cells": [],
                }
            )
        )
    with pytest.raises(BenchmarkError, match="duplicate run_id"):
        generate_collection_report((tmp_path,), tmp_path / "report")


def test_cli_dry_run_uses_explicit_bundle_without_runtime(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle")
    assert (
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--manifest-root",
                str(REPO / "families"),
                "--bundle",
                str(bundle),
                "--dry-run",
                "--no-build",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["operation"] == "generate"


def test_native_examples_depend_only_on_public_headers() -> None:
    for name in ("benchmark_worker.cpp", "dataset_benchmark.cpp"):
        source = (REPO / "apps/benchmark/native" / name).read_text(encoding="utf-8")
        assert '#include "trtmc/task.h"' in source
        assert '#include "trtmc/runtime/family_loader.h"' in source
        assert '#include "src/' not in source


def test_worker_and_catalog_resolution_have_one_explicit_path(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    worker.chmod(0o755)
    assert find_worker(worker) == worker.resolve()

    with pytest.raises(BenchmarkError, match="use --worker"):
        find_worker()
    with pytest.raises(BenchmarkError, match="use --manifest-root"):
        default_manifest_root()

    worker_source = (REPO / "apps/benchmark/trtmc_benchmark/worker.py").read_text()
    catalog_source = (REPO / "apps/benchmark/trtmc_benchmark/catalog.py").read_text()
    assert "shutil.which" not in worker_source
    assert 'os.environ.get("TRTMC_BENCH_WORKER")' not in worker_source
    assert "TRTMC_BENCH_MANIFEST_ROOT" not in catalog_source
