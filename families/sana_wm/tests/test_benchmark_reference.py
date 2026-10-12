# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from families.sana_wm.tests.benchmark import reference as sana_family_reference
from families.sana_wm.tests.benchmark import upstream_reference as sana_wm_reference
from qualification_tests.benchmark_qualification.performance import matrix as perf
from qualification_tests.benchmark_qualification.performance import reference_harness
from qualification_tests.benchmark_qualification.performance.references import generic_reference


REPO = Path(__file__).resolve().parents[3]
SUITE = REPO / "qualification_tests/benchmark_qualification/performance/config/release.yaml"


def _environment(tmp_path: Path) -> tuple[Path, perf.Environment]:
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("bench", "worker", "hf.py", "task.py"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in (
        "libtrtmc_runtime.so",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_gpt2.so",
        "libtrtmc_model_lance.so",
    ):
        (runtime / name).write_bytes(b"")
    _, entries, _ = perf._load_suite_file(SUITE)
    reference_fields = {
        str(declaration["environment"])
        for entry in entries
        for declaration in entry.get("baseline", {}).get("reference_inputs", {}).values()
    }
    references = {}
    for name in reference_fields:
        path = tmp_path / name
        path.mkdir()
        references[name] = str(path)
    value = {
        "schema_version": perf.ENVIRONMENT_SCHEMA,
        "name": "test",
        "tools": {
            "trtmc_bench": str(tools / "bench"),
            "trtmc_worker": str(tools / "worker"),
            "hf_transformers_runner": str(tools / "hf.py"),
            "task_reference_runner": str(tools / "task.py"),
        },
        "references": references,
        "storage": {
            "results_root": str(tmp_path / "results"),
            "scratch_root": str(tmp_path / "scratch"),
            "bundle_cache": str(tmp_path / "bundles"),
            "bundle_roots": [],
            "runtime_root": str(runtime),
            "bundle_retention": "retain",
        },
        "execution": {"local_files_only": True, "timeout_seconds": 10},
    }
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path, perf.load_environment(path)


def test_sana_reference_reports_materialized_video_shape() -> None:
    video = np.stack(
        [
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.ones((24, 32, 3), dtype=np.uint8),
        ]
    )
    summary = sana_wm_reference.media_summary(video)
    assert summary == {
        "media_type": "video",
        "media_count": 2,
        "num_frames": 2,
        "height": 24,
        "width": 32,
        "channels": 3,
    }
    assert perf._media_shape(summary) == (2, 24, 32, 3)
    source = (REPO / "families/sana_wm/tests/benchmark/reference.py").read_text(
        encoding="utf-8"
    )
    assert 'request.get("action"' in source


def test_sana_world_request_preserves_official_camera_controls(tmp_path: Path) -> None:
    model = replace(
        perf.ManifestCatalog(REPO / "families").resolve("sana-wm-bidirectional"),
        task="world_model_generation",
    )
    request = perf.resolve_case(
        model, tmp_path / "model.bundle", selected_task="world_model_generation",
    ).request
    assert request["translation_speed"] == 0.055
    assert request["rotation_speed_deg"] == 1.2
    assert request["fps"] == 16
    assert request["flow_shift"] == 9.8
    assert request["no_action_overlay"] is True


@pytest.mark.parametrize("route", ["builtin", "script", "hf"])
def test_reference_testcase_name_is_only_sent_to_builtin_runner(monkeypatch, tmp_path, route):
    _, environment = _environment(tmp_path)
    # Exercise explicit runner fixtures, not a family's active owned replacement.
    _, entries, _ = perf._load_suite_file(SUITE)
    entry = perf.resolve_entries(
        [value for value in entries if value["id"] == "sana_wm.generate_image"], environment,
    )[0]
    baseline = dict(entry.spec["baseline"])
    if route == "builtin":
        baseline.pop("script")
        baseline["adapter"] = "hf-diffusers"
    elif route == "script":
        baseline["script"] = "tests/reference.py"
        monkeypatch.setattr(perf, "_family_script", lambda _: tmp_path / "reference.py")
    elif route == "hf":
        baseline["runner"] = "hf-transformers"
    entry = replace(entry, spec={**entry.spec, "baseline": baseline})
    command = perf.baseline_command(entry, environment, tmp_path / "reference.json")
    assert command[command.index("--case-name") + 1] == entry.spec["id"]
    if route == "builtin":
        assert command[command.index("--testcase-name") + 1] == entry.case.testcase_name
        parser = generic_reference.build_parser()
        parsed = parser.parse_args(command[2:])
        assert parsed.testcase_name == entry.case.testcase_name
        index = command.index("--testcase-name")
        direct = command[2:index] + command[index + 2:]
        assert parser.parse_args(direct).testcase_name is None
    else:
        assert "--testcase-name" not in command


@pytest.mark.parametrize("selection", ["selected", "missing", "duplicate", ""])
def test_sana_reference_resolves_only_exact_testcase_before_execution(
    monkeypatch, tmp_path, selection,
):
    manifest = tmp_path / "manifest.json"
    controls = {"translation_speed": 0.055, "rotation_speed_deg": 1.2,
                "fps": 16, "flow_shift": 9.8, "no_action_overlay": True}
    chosen = {"name": "selected", **controls}
    manifest.write_text(json.dumps({"video_num_frames": 321, "testcases": [
        {"name": "performance-entry", **dict.fromkeys(controls, "wrong testcase")},
        chosen, *([chosen] if selection == "duplicate" else []),
    ]}))
    arguments = SimpleNamespace(
        manifest=manifest, testcase_name="selected" if selection == "duplicate" else selection,
        case_name="performance-entry", warmup=1, iterations=2,
    )
    assets = REPO / "families/sana_wm/tests/assets"
    request = {"prompt": "drive forward", "image_path": str(assets / "demo_0.png"),
               "action": "w-320", "num_steps": 60, "cfg_scale": 5.0, "seed": 42}
    options = {"reference_repo": str(tmp_path), "model_dir": str(tmp_path / "model"),
               "intrinsics": str(assets / "demo_0_intrinsics.npy")}
    before = deepcopy(request), deepcopy(options), manifest.read_bytes()

    def run(command, **kwargs):
        assert selection == "selected", "invalid selection must fail before reference execution"
        for name in ("translation_speed", "rotation_speed_deg", "fps", "flow_shift"):
            assert command[command.index("--" + name) + 1] == str(controls[name])
        assert "--no_action_overlay" in command
        assert command[command.index("--num_frames") + 1] == "321"
        raise RuntimeError("captured selected reference command")

    monkeypatch.setattr(sana_family_reference.subprocess, "run", run)
    if selection == "selected":
        with pytest.raises(RuntimeError, match="captured selected reference command"):
            sana_family_reference._run_compat(arguments, request, options)
    else:
        with pytest.raises(ValueError, match="exactly one testcase"):
            sana_family_reference._run_compat(arguments, request, options)
    assert (request, options, manifest.read_bytes()) == before


def test_sana_reference_options_use_resolved_testcase_and_explicit_options(
    monkeypatch, tmp_path: Path,
) -> None:
    _, environment = _environment(tmp_path)
    # This tests the existing builtin adapter even after an owned script takes over.
    _, entries, _ = perf._load_suite_file(SUITE)
    selected = [entry for entry in entries if entry["id"] == "sana_wm.generate_image"]
    entry = perf.resolve_entries(selected, environment)[0]
    original = {"translation_speed": 0.055, "rotation_speed_deg": 1.2,
                "fps": 16, "flow_shift": 9.8, "no_action_overlay": True}
    assert entry.case.testcase_name != entry.spec["id"]
    testcase = next(value for value in entry.manifest["testcases"]
                    if value["name"] == entry.case.testcase_name)
    other = {**testcase, "name": entry.spec["id"], **dict.fromkeys(original, "wrong testcase")}
    entry = replace(entry, manifest={**entry.manifest, "testcases": [other, testcase]})
    semantic = perf.resolve_case(entry.model, tmp_path / "model.bundle", selected_task="image_text_action_to_video")
    request = generic_reference.flatten_config(semantic.request)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(entry.manifest))
    arguments = SimpleNamespace(
        manifest=manifest_path, testcase_name=entry.case.testcase_name, warmup=1, iterations=2,
    )
    before = json.dumps(entry.manifest, sort_keys=True)
    options = perf._adapter_options(entry, environment)
    assert {name: options[name] for name in original} == original
    assert "action" not in options and "prompt" not in options
    assert options["reference_repo"] == str(Path(environment.references["sana_repo"]).resolve())
    assert options["model_dir"] == str(Path(environment.references["sana_model"]).resolve())
    options["intrinsics"] = str(entry.model.manifest_path.parent.parent / "assets/demo_0_intrinsics.npy")
    expected = original

    def run(command, **kwargs):
        for name in ("translation_speed", "rotation_speed_deg", "fps", "flow_shift"):
            assert command[command.index("--" + name) + 1] == str(expected[name])
        assert ("--no_action_overlay" in command) is expected["no_action_overlay"]
        assert command[command.index("--action") + 1] == request["action"]
        assert Path(command[command.index("--prompt") + 1]).read_text() == request["prompt"]
        raise RuntimeError("captured reference options")

    monkeypatch.setattr(sana_family_reference.subprocess, "run", run)
    original_options = deepcopy(options)
    with pytest.raises(RuntimeError, match="captured reference options"):
        sana_family_reference._run_compat(arguments, request, options)
    assert options == original_options
    explicit = {"translation_speed": 0.0, "rotation_speed_deg": 0.0,
                "fps": 0, "flow_shift": 0.0, "no_action_overlay": False}
    configured = {**entry.spec["baseline"].get("adapter_options", {}), **explicit}
    entry = replace(entry, spec={**entry.spec, "baseline": {**entry.spec["baseline"], "adapter_options": configured}})
    options = perf._adapter_options(entry, environment)
    assert {name: options[name] for name in original} == explicit
    assert options["no_action_overlay"] is False
    options["intrinsics"] = original_options["intrinsics"]
    explicit_options = deepcopy(options)
    expected = explicit
    with pytest.raises(RuntimeError, match="captured reference options"):
        sana_family_reference._run_compat(arguments, request, options)
    assert options == explicit_options
    assert request == generic_reference.flatten_config(semantic.request)
    assert json.loads(manifest_path.read_text()) == entry.manifest
    assert json.dumps(entry.manifest, sort_keys=True) == before
    assert entry.spec["baseline"]["adapter_options"] == configured


def test_sana_semantic_request_metadata_moves_only_to_reference_options(
    monkeypatch, tmp_path: Path,
) -> None:
    _, environment = _environment(tmp_path)
    # Keep the builtin reference contract independent of canonical script selection.
    _, entries, _ = perf._load_suite_file(SUITE)
    selected = [entry for entry in entries if entry["id"] == "sana_wm.generate_image"]
    entry = perf.resolve_entries(selected, environment)[0]
    semantic = perf.resolve_case(entry.model, tmp_path / "model.bundle", selected_task="image_text_action_to_video")
    entry = replace(entry, case=semantic)
    options = perf._adapter_options(entry, environment)
    expected = {"translation_speed": 0.055, "rotation_speed_deg": 1.2,
                "fps": 16, "flow_shift": 9.8, "no_action_overlay": True}
    assert not expected.keys() & semantic.request.keys()
    assert {name: options[name] for name in expected} == expected
    command = perf.baseline_command(entry, environment, tmp_path / "reference.json")
    arguments = reference_harness.parser("test").parse_args(command[2:])

    def run(command, **kwargs):
        for name in ("translation_speed", "rotation_speed_deg", "fps", "flow_shift"):
            assert command[command.index("--" + name) + 1] == str(expected[name])
        assert "--no_action_overlay" in command
        raise RuntimeError("captured reference metadata")

    monkeypatch.setattr(sana_family_reference.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="captured reference metadata"):
        sana_family_reference._run_compat(
            arguments, generic_reference.flatten_config(semantic.request), options
        )
    assert semantic.request["action"] == entry.manifest["testcases"][0]["action"]
    assert semantic.request["seed"] == 42 and semantic.request["num_steps"] == 60
    assert semantic.request["cfg_scale"] == 5.0


def test_sana_reference_calls_official_pipeline_with_exact_workload(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {"generate": []}
    reference_repo = tmp_path / "Sana"
    reference_repo.mkdir()
    model_dir = tmp_path / "model"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    stage1_text_encoder = model_dir / "stage1_text_encoder"
    stage1_text_encoder.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (stage1_text_encoder / name).write_text("{}\n", encoding="utf-8")
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("drive forward", encoding="utf-8")
    intrinsics = tmp_path / "intrinsics.npy"
    intrinsics.write_bytes(b"intrinsics")
    output = tmp_path / "result.json"

    class FakeImage:
        def convert(self, mode):
            captured["image_mode"] = mode
            return self

        def save(self, path):
            Path(path).write_bytes(b"png")
            return None

    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace(open=lambda path: FakeImage(), fromarray=lambda value: FakeImage())
    monkeypatch.setitem(sys.modules, "PIL", pil)

    synchronize_calls = []
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: synchronize_calls.append(True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    pyrallis = ModuleType("pyrallis")

    def parse_config(**kwargs):
        captured["parse"] = kwargs
        return "config"

    pyrallis.parse = parse_config
    monkeypatch.setitem(sys.modules, "pyrallis", pyrallis)

    class RefinerSettings:
        def __init__(self, **kwargs):
            captured["refiner"] = kwargs

    class GenerationParams:
        def __init__(self, **kwargs):
            captured["generation"] = kwargs

    class Pipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def generate(self, *args):
            captured["generate"].append(args)
            return {
                "video": np.zeros((321, 24, 32, 3), dtype=np.uint8),
                "c2w": "camera",
            }

    trajectory = np.zeros((321, 4, 4), dtype=np.float32)
    official = SimpleNamespace(
        InferenceConfig=object,
        RefinerSettings=RefinerSettings,
        GenerationParams=GenerationParams,
        SanaWMPipeline=Pipeline,
        action_string_to_c2w=lambda action, **kwargs: (
            captured.update(action=(action, kwargs)) or trajectory
        ),
        _snap_num_frames=lambda value, **kwargs: value,
        resize_and_center_crop=lambda value: ("cropped", (1, 1), (2, 2), (0, 0)),
        load_intrinsics=lambda path, frames: (
            captured.update(load_intrinsics=(path, frames)) or "raw-intrinsics"
        ),
        transform_intrinsics_for_crop=lambda value, *sizes: (
            captured.update(transform_intrinsics=(value, sizes)) or "intrinsics"
        ),
        apply_overlay=lambda *_: (_ for _ in ()).throw(
            AssertionError("no-action-overlay must skip overlay")
        ),
    )
    monkeypatch.setattr(sana_wm_reference, "_official_module", lambda path: official)
    installed = {}
    monkeypatch.setattr(
        sana_wm_reference,
        "install_local_stage1_text_encoder",
        lambda module, path: installed.update(module=module, path=path),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sana_wm_reference.py",
            "--reference-repo",
            str(reference_repo),
            "--image",
            str(image),
            "--model-dir",
            str(model_dir),
            "--prompt",
            str(prompt),
            "--action",
            "w-320",
            "--intrinsics",
            str(intrinsics),
            "--num_frames",
            "321",
            "--fps",
            "16",
            "--step",
            "60",
            "--cfg_scale",
            "5.0",
            "--flow_shift",
            "9.8",
            "--seed",
            "42",
            "--refiner_seed",
            "42",
            "--translation_speed",
            "0.055",
            "--rotation_speed_deg",
            "1.2",
            "--no_action_overlay",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(output),
        ],
    )

    assert sana_wm_reference.main() == 0
    assert captured["action"] == (
        "w-320",
        {"translation_speed": 0.055, "rotation_speed_deg": 1.2},
    )
    assert captured["refiner"] == {
        "root": model_dir / "refiner",
        "gemma_root": model_dir / "refiner/text_encoder",
        "seed": 42,
    }
    assert installed == {"module": official, "path": stage1_text_encoder.resolve()}
    assert captured["generation"] == {
        "num_frames": 321,
        "fps": 16,
        "step": 60,
        "cfg_scale": 5.0,
        "flow_shift": 9.8,
        "seed": 42,
    }
    assert len(captured["generate"]) == 3
    assert len(synchronize_calls) == 5
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["output_summary"]["num_frames"] == 321


def test_sana_reference_requires_action_from_current_request(tmp_path: Path) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json"
    )
    with pytest.raises(ValueError, match="non-empty action"):
        sana_family_reference._run_compat(
            arguments,
            {"prompt": "drive", "image_path": "assets/demo_0.png"},
            {"reference_repo": str(checkout)},
        )


@pytest.mark.parametrize("frame_controls", [{"num_frames": 321}, {}])
def test_sana_generic_reference_uses_one_explicit_official_command(
    monkeypatch, tmp_path: Path, frame_controls: dict,
) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples_ms": [1.0, 2.0],
                    "output_summary": {
                        "media_type": "video",
                        "media_count": 321,
                        "num_frames": 321,
                        "height": 704,
                        "width": 1280,
                        "channels": 3,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sana_family_reference.subprocess, "run", run)
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json",
        warmup=1,
        iterations=2,
    )
    result = sana_family_reference._run_compat(
        arguments,
        {
            "prompt": "drive forward",
            "image_path": "assets/demo_0.png",
            "action": "w-80,jw-40,w-40,lw-60,w-100",
            "translation_speed": 0.055,
            "rotation_speed_deg": 1.2,
            **frame_controls,
            "fps": 16,
            "num_steps": 60,
            "cfg_scale": 5.0,
            "flow_shift": 9.8,
            "seed": 42,
            "no_action_overlay": True,
        },
        {
            "reference_repo": str(checkout),
            "model_dir": str(model_dir),
            "intrinsics": "assets/demo_0_intrinsics.npy",
        },
    )

    command = captured["command"]
    assert command[command.index("--reference-repo") + 1] == str(checkout.resolve())
    assert command[command.index("--translation_speed") + 1] == "0.055"
    assert command[command.index("--rotation_speed_deg") + 1] == "1.2"
    assert command[command.index("--num_frames") + 1] == "321"
    assert command[command.index("--fps") + 1] == "16"
    assert command[command.index("--flow_shift") + 1] == "9.8"
    assert command[command.index("--refiner_seed") + 1] == "42"
    assert command[command.index("--warmup") + 1] == "1"
    assert command[command.index("--iterations") + 1] == "2"
    assert "--no_action_overlay" in command
    assert "env" not in captured["kwargs"]
    assert result[0] == [1.0, 2.0]
    assert result[1]["num_frames"] == 321


@pytest.mark.parametrize("controls,manifest_frames,expected", [
    ({}, 321, "321"),
    ({"num_frames": 7}, 321, "7"),
    ({"num_frames": 0}, 321, "0"),
    ({"config": {"num_frames": 0}}, 321, "0"),
    ({"num_frames": 7}, None, "7"),
    ({}, None, None),
])
def test_sana_semantic_reference_uses_manifest_frames_only_when_absent(
    monkeypatch, tmp_path: Path, controls: dict, manifest_frames: int | None, expected: str | None,
) -> None:
    model = perf.ManifestCatalog(REPO / "families").resolve("sana-wm-bidirectional")
    model = replace(model, testcases=({**model.testcases[0], **controls},))
    case = perf.resolve_case(model, tmp_path / "model.bundle", selected_task="image_text_action_to_video")
    request = generic_reference.flatten_config(case.request)
    before = dict(request)
    manifest = json.loads(model.manifest_path.read_text())
    if manifest_frames is None:
        del manifest["video_num_frames"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    options = {
        "reference_repo": str(checkout), "model_dir": str(tmp_path / "model"),
        "intrinsics": str(model.manifest_path.parent.parent / "assets/demo_0_intrinsics.npy"),
    }
    testcase = next(value for value in manifest["testcases"] if value["name"] == case.testcase_name)
    for name in ("translation_speed", "rotation_speed_deg", "fps", "flow_shift", "no_action_overlay"):
        options[name] = testcase[name]
        assert name not in request

    def run(command, **kwargs):
        assert command[command.index("--num_frames") + 1] == expected
        assert command[command.index("--action") + 1] == request["action"]
        assert command[command.index("--translation_speed") + 1] == "0.055"
        assert command[command.index("--rotation_speed_deg") + 1] == "1.2"
        assert command[command.index("--fps") + 1] == "16"
        assert command[command.index("--flow_shift") + 1] == "9.8"
        assert "--no_action_overlay" in command
        raise RuntimeError("captured reference command")

    monkeypatch.setattr(sana_family_reference.subprocess, "run", run)
    arguments = SimpleNamespace(manifest=manifest_path, warmup=1, iterations=2)
    if expected is None:
        with pytest.raises(KeyError, match="video_num_frames"):
            sana_family_reference._run_compat(arguments, request, options)
    else:
        with pytest.raises(RuntimeError, match="captured reference command"):
            sana_family_reference._run_compat(arguments, request, options)
    assert request == before


@pytest.mark.parametrize("source", ["request", "config", "options"])
@pytest.mark.parametrize("controls", [
    {"translation_speed": 0.055, "rotation_speed_deg": 1.2,
     "fps": 16, "flow_shift": 9.8, "no_action_overlay": True},
    {"translation_speed": 0.0, "rotation_speed_deg": 0.0,
     "fps": 0, "flow_shift": 0.0, "no_action_overlay": False},
])
def test_sana_reference_control_presence_and_request_precedence(
    monkeypatch, tmp_path: Path, source: str, controls: dict,
) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json",
        warmup=1, iterations=2,
    )
    request = {"prompt": "drive forward", "image_path": "assets/demo_0.png",
               "action": "w-80,jw-40,w-40,lw-60,w-100", "num_frames": 321,
               "num_steps": 60, "cfg_scale": 5.0, "seed": 42}
    options = {"reference_repo": str(checkout), "model_dir": str(tmp_path / "model"),
               "translation_speed": 0.055, "rotation_speed_deg": 1.2,
               "fps": 16, "flow_shift": 9.8, "no_action_overlay": True}
    if source == "options":
        options.update(controls)
    elif source == "config":
        request = generic_reference.flatten_config({**request, "config": controls})
    else:
        request.update(controls)
    before = dict(request), dict(options)

    def run(command, **kwargs):
        for name in ("translation_speed", "rotation_speed_deg", "fps", "flow_shift"):
            assert command[command.index("--" + name) + 1] == str(controls[name])
        assert ("--no_action_overlay" in command) is controls["no_action_overlay"]
        assert command[command.index("--action") + 1] == request["action"]
        assert command[command.index("--refiner_seed") + 1] == "42"
        assert "env" not in kwargs
        raise RuntimeError("captured reference controls")

    monkeypatch.setattr(sana_family_reference.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="captured reference controls"):
        sana_family_reference._run_compat(arguments, request, options)
    assert (request, options) == before


@pytest.mark.parametrize("missing", [
    "translation_speed", "rotation_speed_deg", "fps", "flow_shift", "no_action_overlay",
])
def test_sana_reference_control_missing_from_request_and_options_still_errors(
    monkeypatch, tmp_path: Path, missing: str,
) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json",
        warmup=1, iterations=2,
    )
    request = {"prompt": "drive forward", "image_path": "assets/demo_0.png", "action": "w-320",
               "num_frames": 321, "num_steps": 60, "cfg_scale": 5.0, "seed": 42}
    options = {"reference_repo": str(checkout), "model_dir": str(tmp_path / "model"),
               "translation_speed": 0.055, "rotation_speed_deg": 1.2,
               "fps": 16, "flow_shift": 9.8, "no_action_overlay": True}
    del options[missing]
    monkeypatch.setattr(
        sana_family_reference.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "missing control must fail before reference execution"
        ),
    )
    with pytest.raises(KeyError, match=missing):
        sana_family_reference._run_compat(arguments, request, options)
