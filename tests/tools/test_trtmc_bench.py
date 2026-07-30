# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

import pytest

from tensorrt_model_connect.benchmark import builder as benchmark_builder
from tensorrt_model_connect.benchmark import catalog as benchmark_catalog
from tensorrt_model_connect.benchmark.builder import BundleBuilder
from tensorrt_model_connect.benchmark.catalog import (
    ManifestCatalog,
    expand_sweeps,
    resolve_case,
)
from tensorrt_model_connect.benchmark.cli import main
from tensorrt_model_connect.benchmark.metrics import reduce_metrics
from tensorrt_model_connect.benchmark.operations import registered_operations
from tensorrt_model_connect.benchmark.task_adapters import registered_task_adapters
from tensorrt_model_connect.benchmark.types import BenchmarkError
from tensorrt_model_connect.benchmark.worker import worker_backend_abi, worker_metadata


pytestmark = pytest.mark.unit
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _bundle(tmp_path: Path, name: str = "model.trtfb") -> Path:
    path = tmp_path / name
    path.write_bytes(b"bundle")
    return path


def _worker(tmp_path: Path) -> Path:
    worker = tmp_path / "trtmc_benchmark_worker"
    worker.write_text(
        """#!/usr/bin/env python3
import argparse, json, sys
if sys.argv[1:] == ['--metadata']:
    print(json.dumps({
      'schema_version': 'trtmc.benchmark-worker-metadata/v1',
      'build': {'configuration': 'Release', 'source_revision': 'test-revision'},
    }))
    raise SystemExit(0)
p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
r = json.load(open(a.request, encoding='utf-8'))
n = r['measurement']['iterations']
o = [{
  'iteration': i,
  'runtime_e2e_wall_ms': float(i + 1),
  'output_tokens': 4,
  'prefill_ms': 0.25,
  'decode_ms': 0.75,
} for i in range(n)]
json.dump({
  'schema_version': 'trtmc.benchmark-worker-result/v1',
  'status': 'completed',
  'case_digest': r['case_digest'],
  'timing_scope': r['measurement']['timing_scope'],
  'asset_loading_included': r['measurement']['asset_loading_included'],
  'pipeline_type': 'fake',
  'load_ms': 10.0,
  'observations': o,
  'output_summary': {'text': 'ok', 'token_ids': [1, 2, 3, 4]},
}, open(a.output, 'w', encoding='utf-8'))
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return worker


def _failing_worker(tmp_path: Path) -> Path:
    worker = tmp_path / "failing_trtmc_benchmark_worker"
    worker.write_text(
        """#!/usr/bin/env python3
import argparse, json, sys
if sys.argv[1:] == ['--metadata']:
    print(json.dumps({
      'schema_version': 'trtmc.benchmark-worker-metadata/v1',
      'build': {'configuration': 'Release', 'source_revision': 'test-revision'},
    }))
    raise SystemExit(0)
p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
r = json.load(open(a.request, encoding='utf-8'))
print('intentional worker failure', file=sys.stderr)
json.dump({
  'schema_version': 'trtmc.benchmark-worker-result/v1',
  'status': 'failed',
  'case_digest': r['case_digest'],
  'error': 'intentional failure',
}, open(a.output, 'w', encoding='utf-8'))
raise SystemExit(1)
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return worker


def _write_benchmark_result(
    output: Path,
    *,
    run_id: str,
    model: str,
    status: str = "completed",
    started_at: str = "2026-07-21T09:00:00+00:00",
) -> None:
    artifact = output / f"001-{model}-default"
    artifact.mkdir(parents=True)
    cell: dict[str, object] = {
        "status": status,
        "name": "default",
        "model": model,
        "operation": "generate",
        "case_digest": f"digest-{model}",
        "artifact_dir": artifact.name,
    }
    if status == "completed":
        cell["metrics"] = {
            "latency_ms": {"p50": 2.5, "p95": 4.0},
            "output_tokens_per_s": 100.0,
        }
    else:
        cell["error"] = "worker failed"
    (output / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "trtmc.benchmark-run/v1",
                "run_id": run_id,
                "status": status,
                "started_at": started_at,
                "finished_at": "2026-07-21T09:01:00+00:00",
                "measurement_policy": {"timing_scope": "public_pipeline_call_wall"},
                "environment": {"hostname": "test-host"},
                "cells": [cell],
            }
        ),
        encoding="utf-8",
    )


def test_catalog_reuses_existing_model_manifests_for_different_tasks(tmp_path: Path) -> None:
    catalog = ManifestCatalog()
    expectations = {
        "distilgpt2": ("generate", 5, 50),
        "flux-schnell-l0": ("generate_image", 1, 5),
        "chronos-bolt-tiny-official": ("solve", 50, 500),
        "nemotron-embed-vl-1b-v2": ("embed", 50, 500),
        "whisper-tiny-fp16": ("transcribe", 1, 10),
    }
    for model_name, expected in expectations.items():
        case = resolve_case(catalog.resolve(model_name), _bundle(tmp_path, model_name))
        assert (case.operation, case.measurement.warmup, case.measurement.iterations) == expected


def test_operation_registry_declares_supported_task_semantics() -> None:
    operations = {operation.name: operation for operation in registered_operations()}
    adapters = {adapter.task_strategy: adapter for adapter in registered_task_adapters()}

    assert set(operations) == {
        "generate",
        "generate_image",
        "generate_audio",
        "speak",
        "segment",
        "segment_prompted",
        "classify",
        "detect",
        "rerank",
        "encode",
        "embed",
        "solve",
        "transcribe",
    }
    assert {name: adapter.operation for name, adapter in adapters.items()} == {
        "text_generation_causal": "generate",
        "vision_language_generation": "generate",
        "diffusion_text_generation": "generate",
        "diffusion_media_generation": "generate_image",
        "text_to_audio": "generate_audio",
        "omni_multimodal": "generate_audio",
        "speech_to_speech": "speak",
        "prompted_segmentation": "segment_prompted",
        "segmentation": "segment",
        "image_classification": "classify",
        "object_detection": "detect",
        "reranking": "rerank",
        "encoder_only_nlp": "encode",
        "embedding": "embed",
        "neural_operator": "solve",
        "speech_to_text": "transcribe",
    }
    assert operations["generate_image"].supports_batch is True
    assert operations["generate_image"].per_item_latency.result_name == "seconds_per_image_p50"
    assert [metric.result_name for metric in operations["solve"].rate_metrics] == [
        "windows_per_s",
        "forecast_elements_per_s",
    ]
    assert operations["transcribe"].rate_metrics[0].inverse_result_name == "realtime_factor"


def test_native_worker_has_a_runner_for_every_advertised_operation() -> None:
    worker_source = (REPOSITORY_ROOT / "examples/trtmc_benchmark_worker.cpp").read_text(
        encoding="utf-8"
    )

    for operation in registered_operations():
        assert f'{{"{operation.name}", run_' in worker_source, operation.name

    for replay_input in (
        "initial_latents_path",
        "sampling_steps_path",
        "condition_latents_path",
        "condition_mask_path",
        "sde_noises_path",
    ):
        assert replay_input in worker_source


def test_default_catalog_falls_back_to_installed_package_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site-packages/tensorrt_model_connect/benchmark"
    installed_catalog = package / "_catalog" / "installed" / "manifests"
    installed_catalog.mkdir(parents=True)
    (installed_catalog.parent / "MODEL.toml").write_text(
        'id = "installed"\ntest_manifests = ["manifests/installed.json"]\n',
        encoding="utf-8",
    )
    (installed_catalog / "installed.json").write_text(
        json.dumps(
            {
                "name": "installed-model",
                "hf_id": "example/installed-model",
                "bundle": "installed-model.trtfb",
                "family": "installed",
                "task_strategy": "text_generation_causal",
                "runtime_strategy": "installed_runtime",
                "precision": "fp16",
                "testcases": [{"name": "default", "prompt": "hello"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TRTMC_BENCH_MANIFEST_ROOT", raising=False)
    monkeypatch.setattr(benchmark_catalog, "__file__", str(package / "catalog.py"))

    model = ManifestCatalog().resolve("installed-model")

    assert model.manifest_path == installed_catalog / "installed.json"


def test_cli_lists_supported_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "models"]) == 0

    output = capsys.readouterr().out
    assert "MODEL" in output
    assert "OPERATION" in output
    assert "distilgpt2" in output
    assert "flux-schnell-l0" in output
    assert "generate_image" in output
    assert "STATUS" in output
    assert "chronos-bolt-tiny-official-tp4" in output
    assert "distributed" in output
    assert "sana-wm-bidirectional" in output
    assert "bark-small" in output
    assert "qwen3-vl-2b" in output
    assert "sam3" in output


def test_catalog_exposes_every_declared_profile_and_family() -> None:
    entries = ManifestCatalog().entries()
    expected_manifests: list[Path] = []
    descriptors = sorted((REPOSITORY_ROOT / "tests/e2e/models").glob("*/MODEL.toml"))
    for descriptor in descriptors:
        with descriptor.open("rb") as stream:
            declared = tomllib.load(stream)["test_manifests"]
        expected_manifests.extend(descriptor.parent / path for path in declared)
    expected_distributed = 0
    for manifest in expected_manifests:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        expected_distributed += bool(raw.get("distributed_runtime", {}).get("enabled"))

    assert len(entries) == len(expected_manifests)
    assert len({entry.family for entry in entries}) == len(descriptors)
    assert sum(entry.status == "ready" for entry in entries) == (
        len(expected_manifests) - expected_distributed
    )
    assert sum(entry.status == "distributed" for entry in entries) == expected_distributed
    assert not [entry for entry in entries if entry.status in {"invalid", "unsupported"}]


@pytest.mark.parametrize(
    ("model_name", "operation"),
    [
        ("bark-small", "generate_audio"),
        ("deepseek-ocr-l0", "generate"),
        ("elf-b-owt-l0", "generate"),
        ("internvl3-2b", "generate"),
        ("lance-3b-x2t-image", "generate"),
        ("locateanything-3b", "generate"),
        ("magpie-tts-357m", "generate_audio"),
        ("personaplex-7b-l0", "speak"),
        ("phi4-multimodal", "generate"),
        ("qwen3-omni-30b-a3b-instruct", "generate_audio"),
        ("qwen3-vl-2b", "generate"),
        ("sam-vit-base", "segment_prompted"),
        ("sam3", "segment_prompted"),
        ("sana-wm-bidirectional", "generate_image"),
        ("segformer-b0-ade", "segment"),
        ("timm-vit-base-p16-224-augreg-in21k-ft-in1k", "classify"),
    ],
)
def test_previously_filtered_families_resolve_without_family_registration(
    tmp_path: Path, model_name: str, operation: str
) -> None:
    model = ManifestCatalog().resolve(model_name)
    case = resolve_case(model, tmp_path / model.bundle_name)

    assert case.operation == operation
    assert case.worker_request()["operation"] == operation


def test_future_family_reuses_existing_task_adapter_without_benchmark_changes(
    tmp_path: Path,
) -> None:
    family = tmp_path / "catalog/wan2_2_ti2v"
    manifests = family / "manifests"
    manifests.mkdir(parents=True)
    (family / "MODEL.toml").write_text(
        'id = "wan2_2_ti2v"\n'
        'plugin = "wan2_2_ti2v"\n'
        'test_manifests = ["manifests/wan2.2-ti2v-5b.json"]\n'
        "[e2e_defaults.diffusion_media_generation]\n"
        "build_cli_args = [\n"
        '  { flag = "--video-height", input = "video_height" },\n'
        '  { flag = "--video-width", input = "video_width" },\n'
        '  { flag = "--video-num-frames", input = "video_num_frames" },\n'
        "]\n",
        encoding="utf-8",
    )
    manifest = manifests / "wan2.2-ti2v-5b.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "wan2.2-ti2v-5b",
                "hf_id": "Wan-AI/Wan2.2-TI2V-5B",
                "bundle": "wan2.2-ti2v-5b.trtfb",
                "family": "wan2_2_ti2v",
                "task_strategy": "diffusion_media_generation",
                "runtime_strategy": "diffusion_wan2_2_ti2v",
                "precision": "fp16",
                "testcases": [
                    {
                        "name": "default",
                        "test_prompt": "A boat sailing at sunset.",
                        "video_num_frames": 17,
                        "video_height": 256,
                        "video_width": 448,
                        "num_inference_steps": 4,
                        "guidance_scale": 5.0,
                        "flow_shift": 5.0,
                        "fps": 16,
                        "seed": 42,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = ManifestCatalog(tmp_path / "catalog")
    case = resolve_case(catalog.resolve("wan2.2-ti2v-5b"), tmp_path / "pending.trtfb")

    assert case.operation == "generate_image"
    assert case.request["media_type"] == "video"
    assert case.request["video_num_frames"] == 17
    assert case.request["flow_shift"] == 5.0
    assert case.request["fps"] == 16
    command = BundleBuilder(tmp_path / "cache")._plan(case.model, (case,)).command
    assert command[command.index("--video-height") + 1] == "256"
    assert command[command.index("--video-width") + 1] == "448"
    assert command[command.index("--video-num-frames") + 1] == "17"


def test_future_object_detection_family_uses_existing_public_capability(tmp_path: Path) -> None:
    family = tmp_path / "yolox"
    manifest = family / "manifests/yolox-tiny.json"
    image = family / "data/test_img.jpeg"
    manifest.parent.mkdir(parents=True)
    image.parent.mkdir()
    image.write_bytes(b"synthetic image for resolution only")
    manifest.write_text(
        json.dumps(
            {
                "name": "yolox-tiny",
                "hf_id": "example/yolox-tiny",
                "bundle": "yolox-tiny.trtfb",
                "family": "yolox",
                "task_strategy": "object_detection",
                "runtime_strategy": "yolox_object_detection",
                "precision": "fp16",
                "testcases": [
                    {
                        "name": "default",
                        "test_image": "data/test_img.jpeg",
                        "score_threshold": 0.3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    case = resolve_case(ManifestCatalog().resolve(str(manifest)), tmp_path / "pending.trtfb")

    assert case.operation == "detect"
    assert case.request["score_threshold"] == 0.3
    assert Path(case.worker_request()["request"]["image_path"]).is_file()


def test_catalog_rejects_distributed_profiles_not_supported_by_worker() -> None:
    with pytest.raises(
        BenchmarkError,
        match=r"requires distributed execution \(mpirun, world_size=4\).+single-process",
    ):
        ManifestCatalog().resolve("chronos-bolt-tiny-official-tp4")


def test_model_identity_does_not_depend_on_catalog_install_path(tmp_path: Path) -> None:
    source = Path("tests/e2e/models/gpt2/manifests/distilgpt2.json")
    models = []
    for layout in ("source-checkout", "installed-wheel"):
        manifest = tmp_path / layout / "gpt2/manifests/distilgpt2.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(source.read_bytes())
        models.append(ManifestCatalog._load(manifest))

    bundle = _bundle(tmp_path)
    source_case = resolve_case(models[0], bundle)
    installed_case = resolve_case(models[1], bundle)

    assert models[0].identity() == models[1].identity()
    assert models[0].summary()["manifest_path"] != models[1].summary()["manifest_path"]
    assert source_case.digest == installed_case.digest


def test_named_cases_are_literal_while_sweep_is_cartesian(tmp_path: Path, capsys) -> None:
    config = tmp_path / "bench.yaml"
    bundle = _bundle(tmp_path, "flux-schnell-l0.trtfb")
    config.write_text(
        f"""
models:
  - model: flux-schnell-l0
    bundle: {bundle}
    cases:
      - name: fast
        set:
          request.num_inference_steps: 4
      - name: quality
        set:
          request.num_inference_steps: 20
""",
        encoding="utf-8",
    )
    assert main(["run", str(config), "--dry-run"]) == 0
    literal = json.loads(capsys.readouterr().out)
    assert [(case["name"], case["request"]["num_inference_steps"]) for case in literal] == [
        ("fast", 4),
        ("quality", 20),
    ]

    assert (
        main(
            [
                "run",
                str(config),
                "--case",
                "fast",
                "--sweep",
                "request.seed=1,2",
                "--dry-run",
            ]
        )
        == 0
    )
    swept = json.loads(capsys.readouterr().out)
    assert [case["request"]["seed"] for case in swept] == [1, 2]


def test_batch_size_fails_closed_instead_of_being_clamped(tmp_path: Path) -> None:
    catalog = ManifestCatalog()
    case = resolve_case(catalog.resolve("distilgpt2"), _bundle(tmp_path))
    with pytest.raises(BenchmarkError, match="supports request.batch_size=1 only"):
        expand_sweeps(case, {"request.batch_size": [1, 2]})

    image_case = resolve_case(catalog.resolve("flux-schnell-l0"), _bundle(tmp_path, "flux"))
    batches = expand_sweeps(image_case, {"request.batch_size": [1, 2]})
    assert [item.request["batch_size"] for item in batches] == [1, 2]


def test_auto_build_reuses_model_defaults_and_largest_diffusion_shape(tmp_path: Path) -> None:
    catalog = ManifestCatalog()
    model = catalog.resolve("flux-schnell-l0")
    base = resolve_case(model, tmp_path / "pending.trtfb")
    cases = expand_sweeps(base, {"request.batch_size": [1, 2]})
    plan = BundleBuilder(tmp_path / "cache")._plan(model, cases)

    command = list(plan.command)
    assert command[command.index("--image-height") + 1] == "384"
    assert command[command.index("--image-width") + 1] == "384"
    assert command[command.index("--num-inference-steps") + 1] == "20"
    assert command[command.index("--max-batch-size") + 1] == "2"
    assert command.count("--max-batch-size") == 1


def test_pinned_hf_revision_is_auditable_and_forwarded_to_builder(tmp_path: Path) -> None:
    manifest = (
        REPOSITORY_ROOT
        / "tests/e2e/models/magpie_tts/manifests/magpie-tts-357m.json"
    )
    model = ManifestCatalog._load(manifest)
    plan = BundleBuilder(tmp_path / "cache")._plan(model, ())

    expected = "34d7e40da85cabc97f92198889b65cea27bc7fd1"
    assert model.hf_revision == expected
    assert model.identity()["hf_revision"] == expected
    command = list(plan.command)
    assert command[command.index("--model-revision") + 1] == expected


def test_batch_two_profile_preserves_manifest_batch_inputs_and_build_shape(
    tmp_path: Path,
) -> None:
    model = ManifestCatalog().resolve("flux-schnell-l0-batch2")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    plan = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    assert case.request["batch_size"] == 2
    assert case.request["prompts"] == [
        "A red cube on a white table",
        "A blue sphere on a white table",
    ]
    assert case.request["seeds"] == [42, 42]
    command = list(plan.command)
    assert command[command.index("--max-batch-size") + 1] == "2"


def test_image_edit_profile_resolves_built_in_condition_image(tmp_path: Path) -> None:
    model = ManifestCatalog().resolve("qwen-image-edit-2511")
    case = resolve_case(model, tmp_path / "pending.trtfb")

    assert case.request["image_path"] == "data/test_img.jpeg"
    assert len(case.request["image_sha256"]) == 64
    assert Path(case.worker_request()["request"]["image_path"]).is_file()


@pytest.mark.parametrize("catalog_layout", ["source", "installed"])
def test_auto_build_requires_and_passes_manifest_fp8_scales(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    catalog_layout: str,
) -> None:
    if catalog_layout == "installed":
        package = tmp_path / "site-packages/tensorrt_model_connect/benchmark"
        family = package / "_catalog/flux"
        manifest = family / "manifests/flux-2-dev-fp8.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(
            (
                REPOSITORY_ROOT
                / "tests/e2e/models/flux/manifests/flux-2-dev-fp8.json"
            ).read_bytes()
        )
        scales = family / "data/flux2-fp8-scales.json"
        scales.parent.mkdir()
        scales.write_bytes(
            (
                REPOSITORY_ROOT
                / "tests/e2e/models/flux/data/flux2-fp8-scales.json"
            ).read_bytes()
        )
        monkeypatch.delenv("TRTMC_BENCH_MANIFEST_ROOT", raising=False)
        monkeypatch.setattr(benchmark_catalog, "__file__", str(package / "catalog.py"))

    catalog = ManifestCatalog()
    model = catalog.resolve("flux-2-dev-fp8")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    plan = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    assert model.build_settings["fp8_scales"] == "data/flux2-fp8-scales.json"
    expected_scales = (
        model.manifest_path.parent.parent / str(model.build_settings["fp8_scales"])
    ).resolve()
    assert expected_scales.is_file()
    command = list(plan.command)
    assert command[command.index("--fp8-scales") + 1] == str(expected_scales)


def test_manifest_declared_fp8_scales_fail_closed_when_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "flux/manifests/missing-fp8-scales.json"
    manifest.parent.mkdir(parents=True)
    raw = json.loads(
        (REPOSITORY_ROOT / "tests/e2e/models/flux/manifests/flux-2-dev-fp8.json").read_text(
            encoding="utf-8"
        )
    )
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="fp8_scales file.*is missing"):
        ManifestCatalog().resolve(str(manifest))


def test_fp8_scale_contents_participate_in_bundle_cache_identity(tmp_path: Path) -> None:
    family = tmp_path / "catalog/flux"
    manifest = family / "manifests/flux-2-dev-fp8.json"
    manifest.parent.mkdir(parents=True)
    scales = family / "data/flux2-fp8-scales.json"
    scales.parent.mkdir()
    scales.write_text('{"version": 1}\n', encoding="utf-8")
    source = REPOSITORY_ROOT / "tests/e2e/models/flux/manifests/flux-2-dev-fp8.json"
    manifest.write_bytes(source.read_bytes())

    catalog = ManifestCatalog(tmp_path / "catalog")
    model = catalog.resolve("flux-2-dev-fp8")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    first = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    scales.write_text('{"version": 2}\n', encoding="utf-8")
    updated_model = catalog.resolve("flux-2-dev-fp8")
    updated_case = resolve_case(updated_model, tmp_path / "pending.trtfb")
    second = BundleBuilder(tmp_path / "cache")._plan(updated_model, (updated_case,))

    assert first.cache_key != second.cache_key


def test_build_environment_asset_contents_participate_in_bundle_cache_identity(
    tmp_path: Path,
) -> None:
    family = tmp_path / "catalog/qwen_image"
    manifest = family / "manifests/qwen-image-edit-2511.json"
    manifest.parent.mkdir(parents=True)
    image = family / "data/test_img.jpeg"
    image.parent.mkdir()
    image.write_bytes(b"first-image")
    source = REPOSITORY_ROOT / "tests/e2e/models/qwen_image/manifests/qwen-image-edit-2511.json"
    manifest.write_bytes(source.read_bytes())

    catalog = ManifestCatalog(tmp_path / "catalog")
    model = catalog.resolve("qwen-image-edit-2511")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    first = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    image.write_bytes(b"second-image")
    updated_model = catalog.resolve("qwen-image-edit-2511")
    updated_case = resolve_case(updated_model, tmp_path / "pending.trtfb")
    second = BundleBuilder(tmp_path / "cache")._plan(updated_model, (updated_case,))

    assert first.environment["TRTMC_QWEN_IMAGE_EDIT_CONDITION_IMAGE"] == str(image)
    assert first.cache_key != second.cache_key


def test_required_build_environment_asset_participates_in_bundle_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = tmp_path / "catalog/qwen_image"
    manifest = family / "manifests/qwen-image-edit-2511.json"
    manifest.parent.mkdir(parents=True)
    source = REPOSITORY_ROOT / "tests/e2e/models/qwen_image/manifests/qwen-image-edit-2511.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["build_env"]["TRTMC_BARK_TIMING_CACHE_PATH"] = {
        "required_from_env": True,
        "path_like": True,
    }
    raw["build_env"]["TRTMC_BARK_TIMING_CACHE_SHA256"] = {
        "required_from_env": True,
    }
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    condition_image = family / "data/test_img.jpeg"
    condition_image.parent.mkdir()
    condition_image.write_bytes(b"condition-image")
    image = tmp_path / "injected-private-asset"
    image.write_bytes(b"first-image")
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_PATH", str(image))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", "first-digest")

    catalog = ManifestCatalog(tmp_path / "catalog")
    model = catalog.resolve("qwen-image-edit-2511")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    first = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    image.write_bytes(b"second-image")
    second = BundleBuilder(tmp_path / "cache")._plan(model, (case,))
    monkeypatch.setenv("TRTMC_BARK_TIMING_CACHE_SHA256", "second-digest")
    third = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    assert first.environment["TRTMC_BARK_TIMING_CACHE_PATH"] == str(image)
    assert first.cache_key != second.cache_key
    assert second.cache_key != third.cache_key

    monkeypatch.delenv("TRTMC_BARK_TIMING_CACHE_PATH")
    with pytest.raises(
        BenchmarkError,
        match=(
            "required build environment variable "
            "TRTMC_BARK_TIMING_CACHE_PATH"
        ),
    ):
        BundleBuilder(tmp_path / "cache")._plan(model, (case,))


def test_builder_source_digest_participates_in_bundle_cache_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ManifestCatalog().resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    builder = BundleBuilder(tmp_path / "cache")

    monkeypatch.setattr(benchmark_builder, "_builder_source_digest", lambda _family: "first")
    first = builder._plan(model, (case,))
    monkeypatch.setattr(benchmark_builder, "_builder_source_digest", lambda _family: "second")
    second = builder._plan(model, (case,))

    assert first.cache_key != second.cache_key


def test_image_rate_and_seconds_per_image_account_for_batch_size() -> None:
    metrics = reduce_metrics(
        "generate_image",
        [
            {
                "runtime_e2e_wall_ms": 400.0,
                "generated_images": 2,
                "generated_frames": 2,
                "generated_pixels": 20,
            }
        ],
        request={"media_type": "image"},
    )
    assert metrics["images_per_s"] == 5.0
    assert "frames_per_s" not in metrics
    assert metrics["request_throughput_per_s"] == 2.5
    assert metrics["seconds_per_image_p50"] == 0.2


def test_audio_and_rerank_operations_report_task_specific_rates() -> None:
    audio = reduce_metrics(
        "generate_audio",
        [
            {
                "runtime_e2e_wall_ms": 500.0,
                "output_audio_seconds": 2.0,
                "output_samples": 48000,
            }
        ],
    )
    assert audio["audio_seconds_per_s"] == 4.0
    assert audio["realtime_factor"] == 0.25
    assert audio["audio_samples_per_s"] == 96000.0

    rerank = reduce_metrics(
        "rerank",
        [{"runtime_e2e_wall_ms": 20.0, "documents": 4}],
    )
    assert rerank["documents_per_s"] == 200.0


def test_metric_contract_rejects_missing_native_observations() -> None:
    with pytest.raises(BenchmarkError, match="output_audio_seconds in every observation"):
        reduce_metrics(
            "generate_audio",
            [{"runtime_e2e_wall_ms": 500.0, "output_samples": 48000}],
        )


def test_sana_runtime_config_resolves_manifest_assets_for_native_worker(tmp_path: Path) -> None:
    case = resolve_case(
        ManifestCatalog().resolve("sana-wm-bidirectional"),
        tmp_path / "pending.trtfb",
    )

    assert case.operation == "generate_image"
    assert case.request["media_type"] == "video"
    assert case.runtime["config"]["sana_wm.image_path"] == "assets/demo_0.png"
    worker_config = case.worker_request()["runtime"]["config"]
    assert Path(worker_config["sana_wm.image_path"]).is_file()


def test_multimodal_and_speech_cases_preserve_required_runtime_inputs(tmp_path: Path) -> None:
    vlm = resolve_case(
        ManifestCatalog().resolve("deepseek-ocr-l0"),
        tmp_path / "deepseek-ocr.trtfb",
    )
    assert Path(vlm.worker_request()["request"]["image_path"]).is_file()

    speech = resolve_case(
        ManifestCatalog().resolve("personaplex-7b-l0"),
        tmp_path / "personaplex.trtfb",
    )
    assert speech.operation == "speak"
    assert speech.request["max_new_tokens"] == 100
    assert Path(speech.worker_request()["request"]["audio_path"]).is_file()
    assert Path(speech.runtime["hf_python"]).is_file()

    magpie = resolve_case(
        ManifestCatalog().resolve("magpie-tts-357m"),
        tmp_path / "magpie.trtfb",
    )
    assert magpie.request["seed"] == 42
    assert magpie.runtime["config"] == {
        "audio_magpie.cfg_scale": 2.5,
        "audio_magpie.temperature": 0.6,
        "audio_magpie.seed": 42,
    }

    bark = resolve_case(
        ManifestCatalog().resolve("bark-small"),
        tmp_path / "bark.trtfb",
    )
    assert bark.request["seed"] == 0
    assert bark.sources["request.seed"] == "model testcase"

    qwen3_omni = resolve_case(
        ManifestCatalog().resolve("qwen3-omni-30b-a3b-instruct"),
        tmp_path / "qwen3-omni.trtfb",
    )
    assert qwen3_omni.request["seed"] == 42
    assert qwen3_omni.sources["request.seed"] == "model testcase"


def test_seeded_ready_profiles_preserve_seed_in_public_request(tmp_path: Path) -> None:
    def declares_seed(value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        return any(
            name in {"seed", "seeds"} or declares_seed(item)
            for name, item in value.items()
        )

    missing: list[str] = []
    for entry in ManifestCatalog().entries():
        if entry.status != "ready" or entry.model is None:
            continue
        testcase = entry.model.testcases[0]
        seed_contract = {
            name: testcase[name]
            for name in ("seed", "seeds", "inputs", "determinism", "runtime_config")
            if name in testcase
        }
        if not declares_seed(seed_contract):
            continue
        case = resolve_case(entry.model, tmp_path / f"{entry.name}.trtfb")
        if not {"seed", "seeds"} & set(case.request):
            missing.append(entry.name)

    assert missing == []


def test_text_generation_preserves_sampling_contract(tmp_path: Path) -> None:
    case = resolve_case(
        ManifestCatalog().resolve("qwen3-0.6b-topp"),
        tmp_path / "pending.trtfb",
    )

    assert case.request["temperature"] == 0.7
    assert case.request["top_p"] == 0.9
    assert case.request["top_k"] == 50
    assert case.request["seed"] == 42

    for field in ("temperature", "top_p", "top_k", "seed"):
        assert case.sources[f"request.{field}"] == "model testcase"


def test_text_generation_distinguishes_testcase_and_operation_defaults(
    tmp_path: Path,
) -> None:
    case = resolve_case(
        ManifestCatalog().resolve("glm-4-9b"),
        tmp_path / "pending.trtfb",
    )

    assert case.sources["request.prompt"] == "model testcase"
    assert case.sources["request.max_new_tokens"] == "model testcase"
    for field in (
        "batch_size",
        "enable_thinking",
        "min_p",
        "seed",
        "temperature",
        "top_k",
        "top_p",
        "use_chat_template",
    ):
        assert case.sources[f"request.{field}"] == "operation default"

    overridden = benchmark_catalog.apply_overrides(
        case,
        {
            "request.temperature": 0.5,
            "measurement.iterations": 7,
            "measurement.timing_scope": "model_call_wall",
            "measurement.asset_loading_included": False,
        },
    )
    assert overridden.sources["request.temperature"] == "user override"
    assert overridden.sources["measurement.iterations"] == "user override"
    assert overridden.measurement.timing_scope == "model_call_wall"
    assert overridden.measurement.asset_loading_included is False


def test_all_advertised_defaults_explain_every_request_field(tmp_path: Path) -> None:
    for model in ManifestCatalog().models():
        case = resolve_case(model, tmp_path / f"{model.name}.trtfb")
        request_sources = {
            name.removeprefix("request."): source
            for name, source in case.sources.items()
            if name.startswith("request.")
        }
        assert request_sources.keys() == case.request.keys(), model.name


def test_text_generation_preserves_mode_and_chat_contract(tmp_path: Path) -> None:
    case = resolve_case(
        ManifestCatalog().resolve("nemotron-labs-diffusion-8b-l0"),
        tmp_path / "pending.trtfb",
    )

    assert case.request["generation_mode"] == "ar"
    assert case.request["temperature"] == 0.0
    assert case.request["seed"] == -1
    assert case.request["use_chat_template"] is True
    assert case.request["enable_thinking"] is False


def test_video_profile_preserves_video_build_shape_and_frame_rate(tmp_path: Path) -> None:
    model = ManifestCatalog().resolve("ltx-video-l0")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    plan = BundleBuilder(tmp_path / "cache")._plan(model, (case,))

    assert case.request["media_type"] == "video"
    assert case.request["height"] == 256
    assert case.request["width"] == 256
    assert case.request["video_num_frames"] == 9
    command = list(plan.command)
    assert command[command.index("--video-height") + 1] == "256"
    assert command[command.index("--video-width") + 1] == "256"
    assert command[command.index("--video-num-frames") + 1] == "9"
    assert "--image-height" not in command

    metrics = reduce_metrics(
        "generate_image",
        [
            {
                "runtime_e2e_wall_ms": 1000.0,
                "generated_images": 1,
                "generated_frames": 9,
                "generated_pixels": 100,
            }
        ],
        request={"media_type": "video"},
    )
    assert metrics["videos_per_s"] == 1.0
    assert metrics["frames_per_s"] == 9.0
    assert metrics["seconds_per_video_p50"] == 1.0
    assert "images_per_s" not in metrics
    assert "seconds_per_image_p50" not in metrics


def test_transcription_resolves_audio_artifact_and_reports_realtime_factor(
    tmp_path: Path,
) -> None:
    case = resolve_case(
        ManifestCatalog().resolve("whisper-tiny-fp16"),
        _bundle(tmp_path, "whisper.trtfb"),
    )

    assert case.request["audio_path"] == "data/Recording.wav"
    assert len(case.request["audio_sha256"]) == 64
    assert case.sources["request.audio_path"] == "model testcase"
    assert case.sources["request.audio_sha256"] == "derived from model testcase"
    assert Path(case.worker_request()["request"]["audio_path"]).is_file()

    replacement_audio = tmp_path / "replacement.wav"
    replacement_audio.write_bytes(b"replacement audio")
    overridden = benchmark_catalog.apply_overrides(
        case, {"request.audio_path": str(replacement_audio)}
    )
    assert overridden.sources["request.audio_path"] == "user override"
    assert overridden.sources["request.audio_sha256"] == "derived from user override"
    assert overridden.request["audio_sha256"] != case.request["audio_sha256"]

    metrics = reduce_metrics(
        "transcribe",
        [
            {
                "runtime_e2e_wall_ms": 1000.0,
                "input_audio_seconds": 10.0,
                "output_tokens": 5,
                "first_partial_ms": 250.0,
            },
            {
                "runtime_e2e_wall_ms": 1000.0,
                "input_audio_seconds": 10.0,
                "output_tokens": 5,
                "first_partial_ms": 300.0,
            },
        ],
    )
    assert metrics["audio_seconds_per_s"] == 10.0
    assert metrics["realtime_factor"] == 0.1
    assert metrics["output_tokens_per_s"] == 5.0
    assert metrics["reported_stages_ms"]["first_partial_ms"]["p50"] == 275.0


def test_cli_runs_fake_native_worker_and_writes_evidence(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    bundle = _bundle(tmp_path)
    output = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--bundle",
                str(bundle),
                "--iterations",
                "4",
                "--warmup",
                "0",
                "--telemetry",
                "off",
                "--worker",
                str(worker),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert isinstance(result["run_id"], str)
    assert result["run_id"]
    assert result["status"] == "completed"
    assert result["measurement_policy"]["task_quality_evaluated"] is False
    assert result["cells"][0]["metrics"]["latency_ms"]["p50"] == 2.5
    assert (output / "report.html").is_file()
    artifact_dir = output / result["cells"][0]["artifact_dir"]
    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "observations.jsonl",
        "resolved-case.json",
        "worker.log",
    ]
    observations = [
        json.loads(line)
        for line in (artifact_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(observations) == 4
    assert observations[0]["iteration"] == 0


def test_failed_worker_keeps_protocol_evidence_without_cell_duplicate(tmp_path: Path) -> None:
    worker = _failing_worker(tmp_path)
    output = tmp_path / "failed-run"

    assert (
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--bundle",
                str(_bundle(tmp_path)),
                "--telemetry",
                "off",
                "--worker",
                str(worker),
                "-o",
                str(output),
            ]
        )
        == 1
    )

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    artifact_dir = output / result["cells"][0]["artifact_dir"]
    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "resolved-case.json",
        "worker-request.json",
        "worker-result.json",
        "worker.log",
    ]
    assert "intentional worker failure" in (artifact_dir / "worker.log").read_text(encoding="utf-8")


def test_cli_combines_model_result_subdirectories_into_one_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    collection = tmp_path / "result-20260721"
    _write_benchmark_result(collection / "distilgpt2", run_id="run-text", model="distilgpt2")
    _write_benchmark_result(
        collection / "flux-schnell",
        run_id="run-image",
        model="flux-schnell-l0",
        status="failed",
        started_at="2026-07-21T10:00:00+00:00",
    )

    assert main(["report", str(collection)]) == 0

    report = json.loads((collection / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "trtmc.benchmark-report/v1"
    assert report["status"] == "failed"
    assert report["summary"] == {
        "cases": 2,
        "completed_runs": 1,
        "failed_runs": 1,
        "incomplete_runs": 0,
        "models": 2,
        "runs": 2,
    }
    assert report["models"] == ["distilgpt2", "flux-schnell-l0"]
    assert [run["result_path"] for run in report["runs"]] == [
        "distilgpt2/result.json",
        "flux-schnell/result.json",
    ]
    html = (collection / "report.html").read_text(encoding="utf-8")
    assert "distilgpt2" in html
    assert "flux-schnell-l0" in html
    assert "distilgpt2/001-distilgpt2-default" in html
    assert "failed: 2 run(s), 2 model(s), 2 case(s)" in capsys.readouterr().out


def test_cli_report_rescans_collection_and_atomically_replaces_summary(tmp_path: Path) -> None:
    collection = tmp_path / "result-20260721"
    _write_benchmark_result(collection / "first", run_id="run-1", model="distilgpt2")
    assert main(["report", str(collection)]) == 0

    _write_benchmark_result(collection / "second", run_id="run-2", model="bart-base")
    assert main(["report", str(collection)]) == 0

    report = json.loads((collection / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["runs"] == 2
    assert report["models"] == ["bart-base", "distilgpt2"]
    assert not list(collection.glob(".report.*.trtmc-bench-*.tmp"))


def test_cli_report_supports_multiple_roots_and_deduplicates_run_id(tmp_path: Path) -> None:
    first = tmp_path / "gb300"
    second = tmp_path / "h100"
    combined = tmp_path / "combined"
    _write_benchmark_result(first / "model", run_id="same-run", model="distilgpt2")
    _write_benchmark_result(second / "copied-model", run_id="same-run", model="distilgpt2")

    assert main(["report", str(first), str(second), "-o", str(combined)]) == 0

    report = json.loads((combined / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["runs"] == 1
    assert report["runs"][0]["result_path"].endswith("gb300/model/result.json")


def test_cli_report_rejects_run_id_collision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    collection = tmp_path / "results"
    _write_benchmark_result(collection / "first", run_id="collision", model="distilgpt2")
    _write_benchmark_result(collection / "second", run_id="collision", model="bart-base")

    with pytest.raises(SystemExit, match="2"):
        main(["report", str(collection)])

    assert "run_id collision" in capsys.readouterr().err
    assert not (collection / "report.json").exists()


def test_cli_report_refuses_to_replace_single_run_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "single-run"
    _write_benchmark_result(run, run_id="run-1", model="distilgpt2")
    original = run / "report.html"
    original.write_text("single run report", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(["report", str(run)])

    assert "single-run report" in capsys.readouterr().err
    assert original.read_text(encoding="utf-8") == "single run report"


def test_cli_replaces_explicit_existing_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worker = _worker(tmp_path)
    bundle = _bundle(tmp_path)
    output = tmp_path / "run"
    arguments = [
        "run",
        "--model",
        "distilgpt2",
        "--bundle",
        str(bundle),
        "--iterations",
        "1",
        "--warmup",
        "0",
        "--telemetry",
        "off",
        "--worker",
        str(worker),
        "-o",
        str(output),
    ]

    assert main(arguments) == 0
    (output / "obsolete-from-previous-run.txt").write_text("old", encoding="utf-8")
    capsys.readouterr()

    assert main(arguments) == 0

    assert not (output / "obsolete-from-previous-run.txt").exists()
    assert (output / "result.json").is_file()
    assert f"Replacing existing output after run completes: {output}" in capsys.readouterr().err
    assert not list(tmp_path.glob(".run.trtmc-bench-staging-*"))
    assert not list(tmp_path.glob(".run.trtmc-bench-backup-*"))


def test_cli_refuses_to_replace_symlink_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worker = _worker(tmp_path)
    bundle = _bundle(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output = tmp_path / "linked-output"
    output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--bundle",
                str(bundle),
                "--worker",
                str(worker),
                "-o",
                str(output),
            ]
        )

    assert "refusing to overwrite symlink output directory" in capsys.readouterr().err
    assert output.is_symlink()


def test_cli_refuses_to_replace_non_benchmark_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worker = _worker(tmp_path)
    bundle = _bundle(tmp_path)
    output = tmp_path / "unrelated-data"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--bundle",
                str(bundle),
                "--worker",
                str(worker),
                "-o",
                str(output),
            ]
        )

    assert "refusing to overwrite a non-benchmark directory" in capsys.readouterr().err
    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_cli_auto_builds_missing_bundle_then_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path)
    cache = tmp_path / "cache"
    commands: list[list[str]] = []

    def fake_build(
        command: list[str], _environment: dict[str, str], _timeout_s: int
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"built bundle")
        return subprocess.CompletedProcess(command, 0, "builder stdout\n", "")

    monkeypatch.setenv("TRTMC_BENCH_BUILD_PLATFORM", "test-sm80")
    monkeypatch.setattr(BundleBuilder, "_execute", staticmethod(fake_build))

    common = [
        "run",
        "--model",
        "distilgpt2",
        "--bundle-cache",
        str(cache),
        "--iterations",
        "1",
        "--warmup",
        "0",
        "--telemetry",
        "off",
        "--worker",
        str(worker),
    ]
    first_output = tmp_path / "first"
    assert main([*common, "-o", str(first_output)]) == 0
    assert len(commands) == 1
    assert "distilbert/distilgpt2" in commands[0]
    assert commands[0][commands[0].index("--precision") + 1] == "fp16"
    assert commands[0][commands[0].index("--max-cache-length") + 1] == "256"
    first_result = json.loads((first_output / "result.json").read_text(encoding="utf-8"))
    build = first_result["preparation"]["bundles"][0]
    assert build["status"] == "built"
    assert build["included_in_performance_metrics"] is False
    assert Path(build["bundle"]).is_file()

    second_output = tmp_path / "second"
    assert main([*common, "-o", str(second_output)]) == 0
    assert len(commands) == 1
    second_result = json.loads((second_output / "result.json").read_text(encoding="utf-8"))
    cached = second_result["preparation"]["bundles"][0]
    assert cached["status"] == "cache_hit"
    assert cached["builder_tensorrt_version"]


def test_cli_rebuilds_stale_managed_bundle_found_by_bundle_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path)
    cache = tmp_path / "cache"
    stale_bundle = cache / "distilgpt2" / "stale-cache-key" / "distilgpt2.trtfb"
    stale_bundle.parent.mkdir(parents=True)
    stale_bundle.write_bytes(b"stale bundle")
    commands: list[list[str]] = []

    def fake_build(
        command: list[str], _environment: dict[str, str], _timeout_s: int
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"current bundle")
        return subprocess.CompletedProcess(command, 0, "builder stdout\n", "")

    monkeypatch.setenv("TRTMC_BENCH_BUILD_PLATFORM", "test-sm80")
    monkeypatch.setattr(BundleBuilder, "_execute", staticmethod(fake_build))

    output = tmp_path / "result"
    assert (
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--bundle-cache",
                str(cache),
                "--bundle-root",
                str(cache),
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--telemetry",
                "off",
                "--worker",
                str(worker),
                "-o",
                str(output),
            ]
        )
        == 0
    )

    assert len(commands) == 1
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    preparation = result["preparation"]["bundles"][0]
    assert preparation["status"] == "built"
    assert Path(preparation["bundle"]) != stale_bundle
    assert stale_bundle.read_bytes() == b"stale bundle"


def test_builder_source_digest_tracks_generic_and_family_build_inputs(tmp_path: Path) -> None:
    package_root = tmp_path / "tensorrt_model_connect"
    family_root = package_root / "families" / "distilbert"
    family_root.mkdir(parents=True)
    generic_builder = package_root / "engine_builder.py"
    family_builder = family_root / "plugin.py"
    ignored_test = family_root / "tests" / "test_family.py"
    ignored_test.parent.mkdir()
    generic_builder.write_text("GENERIC = 1\n", encoding="utf-8")
    family_builder.write_text("FAMILY = 1\n", encoding="utf-8")
    ignored_test.write_text("TEST_ONLY = 1\n", encoding="utf-8")

    initial = benchmark_builder._builder_source_digest(
        "distilbert", package_root=package_root
    )
    ignored_test.write_text("TEST_ONLY = 2\n", encoding="utf-8")
    assert (
        benchmark_builder._builder_source_digest(
            "distilbert", package_root=package_root
        )
        == initial
    )

    family_builder.write_text("FAMILY = 2\n", encoding="utf-8")
    family_changed = benchmark_builder._builder_source_digest(
        "distilbert", package_root=package_root
    )
    assert family_changed != initial

    generic_builder.write_text("GENERIC = 2\n", encoding="utf-8")
    assert (
        benchmark_builder._builder_source_digest(
            "distilbert", package_root=package_root
        )
        != family_changed
    )


def test_cli_no_build_fails_closed_when_bundle_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--bundle-cache",
                str(tmp_path / "cache"),
                "--no-build",
                "--dry-run",
            ]
        )
    assert "bundle for distilgpt2 is unavailable and --no-build was set" in capsys.readouterr().err


def test_auto_build_selects_python_tensorrt_matching_runtime_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_packages = tmp_path / "system-packages"
    tensorrt_package = system_packages / "tensorrt"
    tensorrt_package.mkdir(parents=True)
    (tensorrt_package / "__init__.py").write_text('__version__ = "10.15.2.7"\n', encoding="utf-8")
    monkeypatch.setenv("TRTMC_BENCH_TRT_PYTHON_ROOT", str(system_packages))
    monkeypatch.setenv("TRTMC_BENCH_BUILD_PLATFORM", "test-sm80")
    monkeypatch.setattr(benchmark_builder.metadata, "version", lambda _name: "10.16.1.11")

    catalog = ManifestCatalog()
    model = catalog.resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "pending.trtfb")
    plan = BundleBuilder(tmp_path / "cache", backend_abi="10.15")._plan(model, (case,))

    assert plan.runtime.version == "10.15.2.7"
    assert plan.runtime.abi == "10.15"
    assert plan.runtime.backend_abi == "10.15"
    assert plan.command[2] == "tensorrt_model_connect.benchmark._build_entry"
    assert plan.environment["TRTMC_BENCH_TRT_PYTHON_ROOT"] == str(system_packages)
    assert plan.environment["TRTMC_BENCH_BLOCK_TRT_LIBS_WHEEL"] == "1"


def test_worker_backend_abi_uses_packaged_alias(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    (tmp_path / "libtrtmc_backend_trt_10_15.so").touch()
    assert worker_backend_abi(worker) == "10.15"


def test_worker_metadata_reports_build_provenance(tmp_path: Path) -> None:
    assert worker_metadata(_worker(tmp_path)) == {
        "schema_version": "trtmc.benchmark-worker-metadata/v1",
        "build": {
            "configuration": "Release",
            "source_revision": "test-revision",
        },
    }
