# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for wan2_2_ti2v."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from .frame_accuracy import compare_png_sequences
from .official_reference import generate as generate_official_reference

FAMILY = "wan2_2_ti2v"
TASKS = frozenset({"image_generation"})
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
    name = f"TRTMC_{FAMILY.upper()}_MODEL_DIR"
    explicit = os.environ.get(name)
    if explicit:
        return _required_path(explicit, name)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=True,
        )
    )


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
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert denominator > 0.0
    return float(np.dot(a, b) / denominator)


def _centered_correlation(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    centered_a = a - float(a.mean())
    centered_b = b - float(b.mean())
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(a, b, atol=1.0e-8, rtol=1.0e-6) else 0.0
    return float(np.clip(np.dot(centered_a, centered_b) / denominator, -1.0, 1.0))


def test_centered_correlation_does_not_treat_a_shared_offset_as_structure() -> None:
    left = np.asarray([100.0, 101.0, 100.0, 101.0])
    right = np.asarray([100.0, 101.0, 101.0, 100.0])
    assert _cosine(left, right) > 0.99
    assert _centered_correlation(left, right) == pytest.approx(0.0)


def test_l0_reference_is_invariant_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    _, manifest, case = CASES["wan22-ti2v-5b-l0"]

    assert _official_reference(tmp_path, manifest, case, tmp_path) == {"_invariant_only": True}


def test_model_dir_resolves_the_complete_materialized_snapshot(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    monkeypatch.delenv(f"TRTMC_{FAMILY.upper()}_MODEL_DIR", raising=False)
    calls = []

    def snapshot_download(**options):
        calls.append(options)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    manifest = {"hf_id": "Wan-AI/Wan2.2-TI2V-5B", "hf_revision": "revision"}

    assert _model_dir(manifest) == tmp_path
    assert calls == [
        {
            "repo_id": manifest["hf_id"],
            "revision": manifest["hf_revision"],
            "local_files_only": True,
        }
    ]


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
    frames = int(manifest["video_num_frames"]) if "video_num_frames" in manifest else 1
    is_video = frames > 1 or "video" in str(case.get("test_type", ""))
    command = "generate-video" if is_video else "generate-image"
    output = tmp_path / ("native-frames" if is_video else "native.png")
    arguments = [
        "--prompt",
        _case_text(case),
        "--output",
        str(output),
        "--height",
        str(int(manifest["image_height"])),
        "--width",
        str(int(manifest["image_width"])),
        "--num-steps",
        str(int(case["num_inference_steps"])),
        "--seed",
        str(int(case["seed"])),
    ]
    if case.get("negative_prompt"):
        arguments.extend(("--negative-prompt", str(case["negative_prompt"])))
    for key, option in (("guidance_scale", "--guidance-scale"), ("cfg_scale", "--cfg-scale")):
        if key in case:
            arguments.extend((option, str(float(case[key]))))
    payload = _run_json(binary, runtime_root, bundle, manifest, case, command, *arguments)
    payload["artifact"] = str(output)
    return payload


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    backend = case.get("reference_backend")
    if backend == "invariant_only":
        return {"_invariant_only": True}
    assert backend == "wan_official", f"unsupported Wan2.2 reference backend: {backend!r}"
    return generate_official_reference(
        model_dir,
        tmp_path,
        prompt=_case_text(case),
        height=int(manifest["image_height"]),
        width=int(manifest["image_width"]),
        num_frames=int(manifest["video_num_frames"]),
        num_steps=int(case["num_inference_steps"]),
        guidance_scale=float(case["guidance_scale"]),
        flow_shift=float(case.get("flow_shift", 5.0)),
        seed=int(case["seed"]),
        timeout_s=int(case.get("runtime_timeout_s", 14400)),
        **(
            {"negative_prompt": str(case["negative_prompt"])} if case.get("negative_prompt") else {}
        ),
    )


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
    assert len(actual_paths) == len(expected_paths) and actual_paths
    sample_count = int(case.get("semantic_samples", 1))
    assert 1 <= sample_count <= min(6, len(actual_paths))
    if sample_count == 1:
        sample_indices = [(len(actual_paths) - 1) // 2]
    else:
        sample_indices = [
            round(index * (len(actual_paths) - 1) / (sample_count - 1))
            for index in range(sample_count)
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
                "family": "wan2_2_ti2v",
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


def _frame_stats(paths: list[Path]) -> dict[str, float | int | bool]:
    from PIL import Image

    total = 0.0
    total_squared = 0.0
    element_count = 0
    expected_size: tuple[int, int] | None = None
    dimensions_consistent = True
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if expected_size is None:
                expected_size = rgb.size
            dimensions_consistent = dimensions_consistent and rgb.size == expected_size
            pixels = np.asarray(rgb, dtype=np.uint8)
        total += float(pixels.sum(dtype=np.float64))
        total_squared += float(np.square(pixels, dtype=np.float64).sum(dtype=np.float64))
        element_count += int(pixels.size)
    assert element_count and expected_size is not None
    mean_u8 = total / element_count
    variance_u8 = max(total_squared / element_count - mean_u8 * mean_u8, 0.0)
    return {
        "mean": mean_u8 / 255.0,
        "std": variance_u8**0.5 / 255.0,
        "width": expected_size[0],
        "height": expected_size[1],
        "dimensions_consistent": dimensions_consistent,
    }


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    assert manifest["task"] == "image_generation"
    artifact = Path(actual["artifact"])
    if artifact.is_dir():
        actual_paths = sorted(artifact.glob("frame_*.png"))
    else:
        actual_paths = [artifact]
    assert actual_paths
    stats = _frame_stats(actual_paths)
    if "exact_num_frames" in thresholds:
        assert len(actual_paths) == int(thresholds["exact_num_frames"])
    if "exact_video_height" in thresholds:
        assert stats["height"] == int(thresholds["exact_video_height"])
    if "exact_video_width" in thresholds:
        assert stats["width"] == int(thresholds["exact_video_width"])
    assert stats["dimensions_consistent"] is True
    if "min_pixel_mean" in thresholds:
        assert float(stats["mean"]) >= float(thresholds["min_pixel_mean"])
    if "max_pixel_mean" in thresholds:
        assert float(stats["mean"]) <= float(thresholds["max_pixel_mean"])
    if "min_pixel_std" in thresholds:
        assert float(stats["std"]) >= float(thresholds["min_pixel_std"])

    if expected.get("_invariant_only"):
        return

    expected_paths = [Path(path) for path in expected["frame_paths"]]
    accuracy = compare_png_sequences(
        [str(path) for path in expected_paths],
        [str(path) for path in actual_paths],
    )
    assert accuracy["frame_count"] == float(len(actual_paths))

    # The existing Nightly semantic gate owns visual acceptance. Keep raw pixel
    # parity diagnostic because the native and official diffusion backends do
    # not promise pixel identity.
    if not case.get("semantic_assessment"):
        if "min_cosine_uint8" in thresholds:
            assert accuracy["cosine_uint8"] >= float(thresholds["min_cosine_uint8"])
        if "min_frame_cosine_uint8" in thresholds:
            assert accuracy["minimum_frame_cosine_uint8"] >= float(
                thresholds["min_frame_cosine_uint8"]
            )
        if "max_rmse_uint8" in thresholds:
            assert accuracy["maximum_frame_rmse_uint8"] <= float(thresholds["max_rmse_uint8"])
    if "min_temporal_motion_ratio" in thresholds:
        assert accuracy["temporal_motion_ratio"] >= float(thresholds["min_temporal_motion_ratio"])
        assert accuracy["temporal_motion_ratio"] <= float(thresholds["max_temporal_motion_ratio"])
        assert accuracy["temporal_profile_correlation"] >= float(
            thresholds["min_temporal_profile_correlation"]
        )
        assert accuracy["trt_active_transition_fraction"] >= float(
            thresholds["min_active_transition_fraction"]
        )
        assert accuracy["reference_active_transition_fraction"] >= float(
            thresholds["min_active_transition_fraction"]
        )
    _write_semantic_artifacts(case, actual_paths, expected_paths)


def test_semantic_artifacts_are_paired(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(tmp_path))
    actual = tmp_path / "actual.png"
    reference = tmp_path / "reference.png"
    actual.write_bytes(b"actual")
    reference.write_bytes(b"reference")
    case = {
        "name": "semantic-probe",
        "prompt": "a test image",
        "semantic_assessment": True,
    }
    _write_semantic_artifacts(
        case,
        [actual],
        [reference],
    )
    output = tmp_path / "semantic-probe"
    assert sorted(path.name for path in output.iterdir()) == [
        "metadata.json",
        "reference-000.png",
        "trt-000.png",
    ]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "case": "semantic-probe",
        "family": "wan2_2_ti2v",
        "frame_count": 1,
        "prompt": "a test image",
        "sample_count": 1,
        "sampled_frame_indices": [0],
        "schema_version": 1,
    }


def test_semantic_artifacts_reject_unpaired_frames(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRTMC_E2E_ARTIFACT_DIR", str(tmp_path))
    actual = tmp_path / "actual.png"
    actual.write_bytes(b"actual")
    case = {
        "name": "unpaired-semantic-probe",
        "prompt": "a test image",
        "semantic_assessment": True,
    }

    with pytest.raises(AssertionError):
        _write_semantic_artifacts(case, [actual], [])


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
