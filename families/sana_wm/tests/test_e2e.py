# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for sana_wm."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "sana_wm"
TASKS = frozenset({"world_model_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    required_gpus = int(manifest["tensor_parallel_size"])
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
    return (binary, runtime_root)


def _prepared_model_dir(model_dir: Path, manifest: dict, tmp_path: Path) -> Path:
    if (model_dir / "stage1_text_encoder/config.json").is_file():
        return model_dir
    from huggingface_hub import snapshot_download

    dependencies = manifest["hf_dependencies"]
    assert isinstance(dependencies, list) and len(dependencies) == 1
    stage1_repo = dependencies[0]["repo_id"]
    stage1_text_encoder = Path(
        snapshot_download(
            repo_id=stage1_repo,
            local_files_only=True,
            allow_patterns=["config.json"],
        )
    )
    prepared = tmp_path / "prepared-model"
    prepared.mkdir()
    for source in model_dir.iterdir():
        (prepared / source.name).symlink_to(source, target_is_directory=source.is_dir())
    (prepared / "stage1_text_encoder").symlink_to(stage1_text_encoder, target_is_directory=True)
    return prepared


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _case_text(case: dict) -> str:
    path = _asset(case["prompt_file"])
    if path.suffix == ".json":
        value = str(json.loads(path.read_text(encoding="utf-8"))["prompt"])
    else:
        value = path.read_text(encoding="utf-8").strip()
    assert value, f"selected {FAMILY} E2E prompt file is empty"
    return value


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    output = tmp_path / "native-frames"
    intrinsics = np.asarray(case["camera_intrinsics"], dtype=np.float32)
    intrinsics_path = tmp_path / "intrinsics.f32"
    intrinsics.tofile(intrinsics_path)
    arguments = [
        "--prompt",
        _case_text(case),
        "--image",
        str(_asset(case["test_image"])),
        "--output",
        str(output),
        "--intrinsics",
        str(intrinsics_path),
        "--num-frames",
        str(int(manifest["video_num_frames"])),
        "--height",
        str(int(manifest["image_height"])),
        "--width",
        str(int(manifest["image_width"])),
        "--num-steps",
        str(int(case["num_inference_steps"])),
        "--seed",
        str(int(case["seed"])),
    ]
    if case.get("action"):
        arguments.extend(("--action", str(case["action"])))
    for key, option in (("guidance_scale", "--guidance-scale"), ("cfg_scale", "--cfg-scale")):
        if key in case:
            arguments.extend((option, str(float(case[key]))))
    payload = _run_json(binary, runtime_root, bundle, manifest, case, "generate-world", *arguments)
    payload["artifact"] = str(output)
    return payload


def _decode_reference_video(video_path: Path, frames_dir: Path) -> list[Path]:
    import imageio.v3 as iio
    from PIL import Image

    assert video_path.is_file(), f"official Sana reference did not write {video_path}"
    frames_dir.mkdir()
    frame_paths = []
    for index, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        path = frames_dir / f"frame_{index:04d}.png"
        Image.fromarray(np.asarray(frame)).convert("RGB").save(path)
        frame_paths.append(path)
    return frame_paths


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    assert case["no_action_overlay"] is True
    video_dir = tmp_path / "reference-video"
    source_root = _required_path(
        os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"), "TRTMC_REFERENCE_SOURCE_DIR"
    ).resolve()
    model_dir = model_dir.resolve()
    entrypoint = source_root / "inference_video_scripts/wm/inference_sana_wm.py"
    assert entrypoint.is_file(), f"declared Sana reference entrypoint is missing: {entrypoint}"
    config = model_dir / "config.yaml"
    model_path = model_dir / "dit/sana_wm_1600m_720p.safetensors"
    refiner_root = model_dir / "refiner"
    refiner_gemma_root = refiner_root / "text_encoder"
    assert config.is_file(), f"Sana reference config is missing: {config}"
    assert model_path.is_file(), f"Sana reference weights are missing: {model_path}"
    assert refiner_root.is_dir(), f"Sana reference refiner is missing: {refiner_root}"
    assert refiner_gemma_root.is_dir(), (
        f"Sana reference refiner text encoder is missing: {refiner_gemma_root}"
    )
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONPATH"] = str(source_root)
    command = [
        sys.executable,
        str(entrypoint),
        "--image",
        str(_asset(case["test_image"])),
        "--prompt",
        str(_asset(case["prompt_file"])),
        "--intrinsics",
        str(_asset(case["camera_intrinsics_file"])),
        "--action",
        str(case["action"]),
        "--translation_speed",
        str(float(case["translation_speed"])),
        "--rotation_speed_deg",
        str(float(case["rotation_speed_deg"])),
        "--num_frames",
        str(int(manifest["video_num_frames"])),
        "--fps",
        str(int(case["fps"])),
        "--step",
        str(int(case["num_inference_steps"])),
        "--cfg_scale",
        str(float(case["cfg_scale"])),
        "--flow_shift",
        str(float(case["flow_shift"])),
        "--seed",
        str(int(case["seed"])),
        "--refiner_seed",
        str(int(case["seed"])),
        "--no_action_overlay",
        "--config",
        str(config),
        "--model_path",
        str(model_path),
        "--refiner_root",
        str(refiner_root),
        "--refiner_gemma_root",
        str(refiner_gemma_root),
        "--output_dir",
        str(video_dir),
        "--name",
        "reference",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=source_root,
        env=environment,
        timeout=int(case.get("runtime_timeout_s", 7200)),
    )
    frame_paths = _decode_reference_video(
        video_dir / "reference_generated.mp4", tmp_path / "reference-frames"
    )
    expected_names = [
        f"frame_{index:04d}.png" for index in range(int(manifest["video_num_frames"]))
    ]
    assert [path.name for path in frame_paths] == expected_names
    return {"frame_paths": frame_paths}


def _write_semantic_artifacts(
    case: dict, actual_paths: list[Path], expected_paths: list[Path]
) -> None:
    if not case.get("semantic_assessment"):
        return
    output_root = os.environ.get("TRTMC_E2E_ARTIFACT_DIR")
    if not output_root:
        return
    output = Path(output_root) / str(case["name"])
    output.mkdir(parents=True, exist_ok=False)
    paired_count = min(len(actual_paths), len(expected_paths))
    assert paired_count > 0
    sample_count = int(case.get("semantic_samples", 1))
    assert 1 <= sample_count <= min(6, paired_count)
    if sample_count == 1:
        sample_indices = [(paired_count - 1) // 2]
    else:
        sample_indices = [
            round(index * (paired_count - 1) / (sample_count - 1)) for index in range(sample_count)
        ]
    for source_index in sample_indices:
        for label, source in (
            ("trt", actual_paths[source_index]),
            ("reference", expected_paths[source_index]),
        ):
            shutil.copyfile(source, output / f"{label}-{source_index:03d}.png")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "family": "sana_wm",
                "case": case["name"],
                "frame_count": len(actual_paths),
                "prompt": _case_text(case),
                "sample_count": sample_count,
                "sampled_frame_indices": sample_indices,
                "schema_version": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _frame_stats(paths: list[Path]) -> tuple[float, float]:
    assert paths
    count = 0
    total = 0.0
    squared_total = 0.0
    for path in paths:
        pixels = _load_rgb(path).astype(np.float64)
        count += pixels.size
        total += float(pixels.sum())
        squared_total += float(np.square(pixels).sum())
    mean = total / count
    variance = max(0.0, squared_total / count - mean**2)
    return mean, float(np.sqrt(variance))


def _assert_contract(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    del manifest
    artifact = Path(actual["artifact"])
    actual_paths = sorted(artifact.glob("*.png")) if artifact.is_dir() else [artifact]
    expected_paths = [Path(path) for path in expected["frame_paths"]]
    minimum_frames = int(thresholds["contract_min_frame_count"])
    assert len(actual_paths) >= minimum_frames
    assert len(expected_paths) >= minimum_frames
    pixel_mean, pixel_std = _frame_stats(actual_paths)
    assert pixel_mean >= float(thresholds["min_pixel_mean"])
    assert pixel_mean <= float(thresholds["max_pixel_mean"])
    assert pixel_std >= float(thresholds["min_pixel_std"])
    _write_semantic_artifacts(case, actual_paths, expected_paths)


def test_semantic_artifacts_are_paired(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(tmp_path))
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a test image\n", encoding="utf-8")
    case = {
        "name": "semantic-probe",
        "prompt": "a test image",
        "prompt_file": str(prompt_file),
        "semantic_assessment": True,
    }
    actual = tmp_path / "actual.png"
    reference = tmp_path / "reference.png"
    actual.write_bytes(b"actual")
    reference.write_bytes(b"reference")
    _write_semantic_artifacts(case, [actual], [reference])
    output = tmp_path / "semantic-probe"
    assert sorted(path.name for path in output.iterdir()) == [
        "metadata.json",
        "reference-000.png",
        "trt-000.png",
    ]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "case": "semantic-probe",
        "family": "sana_wm",
        "frame_count": 1,
        "prompt": "a test image",
        "sample_count": 1,
        "sampled_frame_indices": [0],
        "schema_version": 1,
    }


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(_prepared_model_dir(model_dir, manifest, tmp_path), bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_contract(actual, expected, manifest, case, _thresholds(case_name))
