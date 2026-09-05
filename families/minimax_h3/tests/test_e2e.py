# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for minimax_h3."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "minimax_h3"
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
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=True,
            allow_patterns=["model_index.json"],
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


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


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
    import torch
    from diffusers import ModularPipeline

    pipeline = ModularPipeline.from_pretrained(model_dir, workflow="t2va", local_files_only=True)
    pipeline.load_components(
        dtype=_torch_dtype(case["reference_precision"]),
        pretrained_model_name_or_path=model_dir,
        local_files_only=True,
    )
    pipeline = pipeline.to("cuda")
    prompts = _case_text(case)
    generator = torch.Generator().manual_seed(int(case["seed"]))
    kwargs = {
        "prompt": prompts,
        "height": int(manifest["image_height"]),
        "width": int(manifest["image_width"]),
        "num_inference_steps": int(case["num_inference_steps"]),
        "generator": generator,
    }
    if int(manifest.get("video_num_frames", 1)) > 1:
        kwargs["num_frames"] = int(manifest["video_num_frames"])
    if case.get("negative_prompt"):
        kwargs["negative_prompt"] = case["negative_prompt"]
    if "guidance_scale" in case:
        kwargs["guidance_scale"] = float(case["guidance_scale"])
    output = pipeline(output="videos", output_type="np", **kwargs)
    videos = output.get("videos")
    if isinstance(videos, torch.Tensor):
        frames = videos.detach().float().cpu().numpy()
    else:
        frames = np.asarray(videos[0])
    assert frames.shape == (
        int(manifest["video_num_frames"]),
        int(manifest["image_height"]),
        int(manifest["image_width"]),
        3,
    )
    frames_path = tmp_path / "reference-frames.npy"
    np.save(frames_path, frames)
    return {"frames_path": frames_path}


def _write_semantic_artifacts(
    case: dict, actual_paths: list[Path], expected_frames_path: Path
) -> None:
    if not case.get("semantic_assessment"):
        return
    output_root = os.environ.get("TRTMC_E2E_ARTIFACT_DIR")
    if not output_root:
        return
    output = Path(output_root) / str(case["name"])
    output.mkdir(parents=True, exist_ok=False)
    expected_frames = np.load(expected_frames_path, mmap_mode="r", allow_pickle=False)
    from PIL import Image

    assert len(actual_paths) == len(expected_frames) and actual_paths
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
        shutil.copyfile(actual_paths[source_index], output / f"trt-{source_index:03d}.png")
        pixels = np.rint(np.clip(expected_frames[source_index], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(pixels).save(output / f"reference-{source_index:03d}.png")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "family": "minimax_h3",
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


def _block_means(frame: np.ndarray, block: int) -> np.ndarray:
    height, width, channels = frame.shape
    assert height % block == 0 and width % block == 0 and channels == 3
    return frame.reshape(height // block, block, width // block, block, channels).mean(
        axis=(1, 3), dtype=np.float64
    )


def _streaming_visual_metrics(
    actual_paths: list[Path], expected_frames_path: Path, block: int
) -> dict[str, float | tuple[int, int, int]]:
    expected_frames = np.load(expected_frames_path, mmap_mode="r", allow_pickle=False)
    assert len(actual_paths) == len(expected_frames) and actual_paths
    correlations = []
    actual_brightness = []
    expected_brightness = []
    std_ratios = []
    actual_activity = []
    expected_activity = []
    previous_actual = None
    previous_expected = None
    frame_shape = None
    for index, actual_path in enumerate(actual_paths):
        actual = _load_rgb(actual_path)
        expected = np.asarray(expected_frames[index], dtype=np.float32)
        assert actual.shape == expected.shape and actual.ndim == 3 and actual.shape[2] == 3
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        assert 0.0 <= float(actual.min()) <= float(actual.max()) <= 1.0
        assert 0.0 <= float(expected.min()) <= float(expected.max()) <= 1.0
        if frame_shape is None:
            frame_shape = actual.shape
        else:
            assert actual.shape == frame_shape
        actual_blocks = _block_means(actual, block)
        expected_blocks = _block_means(expected, block)
        correlations.append(_centered_correlation(actual_blocks, expected_blocks))
        actual_brightness.append(float(actual_blocks.mean()))
        expected_brightness.append(float(expected_blocks.mean()))
        expected_std = float(expected.std(dtype=np.float64))
        actual_std = float(actual.std(dtype=np.float64))
        std_ratios.append(
            actual_std / expected_std
            if expected_std > np.finfo(np.float64).eps
            else (1.0 if actual_std <= np.finfo(np.float64).eps else float("inf"))
        )
        if previous_actual is not None and previous_expected is not None:
            actual_activity.append(float(np.mean(np.abs(actual_blocks - previous_actual))))
            expected_activity.append(float(np.mean(np.abs(expected_blocks - previous_expected))))
        previous_actual = actual_blocks
        previous_expected = expected_blocks
    assert frame_shape is not None and actual_activity and expected_activity
    actual_activity_array = np.asarray(actual_activity)
    expected_activity_array = np.asarray(expected_activity)
    expected_activity_sum = float(expected_activity_array.sum())
    assert expected_activity_sum > np.finfo(np.float64).eps
    return {
        "frame_shape": frame_shape,
        "minimum_frame_low_frequency_correlation": float(min(correlations)),
        "mean_low_frequency_correlation": float(np.mean(correlations)),
        "brightness_profile_correlation": _centered_correlation(
            actual_brightness, expected_brightness
        ),
        "maximum_frame_brightness_absolute_error": float(
            np.max(np.abs(np.asarray(actual_brightness) - np.asarray(expected_brightness)))
        ),
        "minimum_frame_std_ratio": float(min(std_ratios)),
        "maximum_frame_std_ratio": float(max(std_ratios)),
        "temporal_motion_ratio": float(actual_activity_array.sum() / expected_activity_sum),
        "temporal_profile_correlation": _centered_correlation(
            actual_activity_array, expected_activity_array
        ),
        "maximum_temporal_activity_absolute_error": float(
            np.max(np.abs(actual_activity_array - expected_activity_array))
        ),
    }


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    artifact = Path(actual["artifact"])
    actual_paths = sorted(artifact.glob("*.png")) if artifact.is_dir() else [artifact]
    expected_frames_path = Path(expected["frames_path"])
    assert actual_paths
    assert len(actual_paths) == int(thresholds["exact_num_frames"])
    metrics = _streaming_visual_metrics(
        actual_paths, expected_frames_path, int(thresholds["low_frequency_block_size"])
    )
    assert metrics["frame_shape"] == (
        int(thresholds["exact_video_height"]),
        int(thresholds["exact_video_width"]),
        3,
    )
    assert metrics["minimum_frame_low_frequency_correlation"] >= float(
        thresholds["minimum_frame_low_frequency_correlation"]
    )
    assert metrics["mean_low_frequency_correlation"] >= float(
        thresholds["minimum_mean_low_frequency_correlation"]
    )
    assert metrics["brightness_profile_correlation"] >= float(
        thresholds["minimum_brightness_profile_correlation"]
    )
    assert metrics["maximum_frame_brightness_absolute_error"] <= float(
        thresholds["maximum_frame_brightness_absolute_error"]
    )
    assert metrics["minimum_frame_std_ratio"] >= float(thresholds["minimum_frame_std_ratio"])
    assert metrics["maximum_frame_std_ratio"] <= float(thresholds["maximum_frame_std_ratio"])
    assert metrics["temporal_motion_ratio"] >= float(thresholds["min_temporal_motion_ratio"])
    assert metrics["temporal_motion_ratio"] <= float(thresholds["max_temporal_motion_ratio"])
    assert metrics["temporal_profile_correlation"] >= float(
        thresholds["min_temporal_profile_correlation"]
    )
    assert metrics["maximum_temporal_activity_absolute_error"] <= float(
        thresholds["maximum_temporal_activity_absolute_error"]
    )
    _write_semantic_artifacts(case, actual_paths, expected_frames_path)


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
    actual.write_bytes(b"actual")
    reference = tmp_path / "reference.npy"
    np.save(reference, np.ones((1, 2, 2, 3), dtype=np.float32))
    _write_semantic_artifacts(case, [actual], reference)
    output = tmp_path / "semantic-probe"
    assert sorted(path.name for path in output.iterdir()) == [
        "metadata.json",
        "reference-000.png",
        "trt-000.png",
    ]
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "case": "semantic-probe",
        "family": "minimax_h3",
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
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
