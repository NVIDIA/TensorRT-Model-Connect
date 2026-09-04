# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for qwen_image."""

from __future__ import annotations
import json
import os
import subprocess
from functools import cache
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "qwen_image"
TASKS = frozenset({"image_generation", "image_edit"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
LATENT_CHANNELS = 16
VAE_SCALE_FACTOR = 8
LATENT_PATCH_SIZE = 2


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


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def _initial_latents(manifest: dict, case: dict) -> np.ndarray:
    height = int(manifest["image_height"])
    width = int(manifest["image_width"])
    assert height % VAE_SCALE_FACTOR == 0 and width % VAE_SCALE_FACTOR == 0
    generator = np.random.default_rng(int(case["seed"]))
    return generator.standard_normal(
        (LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR),
        dtype=np.float32,
    )


def _pack_reference_latents(initial_latents: np.ndarray):
    import torch

    channels, height, width = initial_latents.shape
    assert height % LATENT_PATCH_SIZE == 0 and width % LATENT_PATCH_SIZE == 0
    packed_height = height // LATENT_PATCH_SIZE
    packed_width = width // LATENT_PATCH_SIZE
    return (
        torch.from_numpy(initial_latents.copy())
        .reshape(
            1,
            channels,
            packed_height,
            LATENT_PATCH_SIZE,
            packed_width,
            LATENT_PATCH_SIZE,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(
            1,
            packed_height * packed_width,
            channels * LATENT_PATCH_SIZE * LATENT_PATCH_SIZE,
        )
    )


def _assert_pixel_statistics(actual_images: list, expected_images: list, thresholds: dict) -> dict:
    actual_pixels = np.asarray(actual_images)
    expected_pixels = np.asarray(expected_images)
    metrics = {
        "actual_mean": float(actual_pixels.mean()),
        "expected_mean": float(expected_pixels.mean()),
        "actual_std": float(actual_pixels.std()),
        "expected_std": float(expected_pixels.std()),
    }
    min_mean = float(thresholds["min_pixel_mean"])
    max_mean = float(thresholds["max_pixel_mean"])
    min_std = float(thresholds["min_pixel_std"])
    assert min_mean <= metrics["actual_mean"] <= max_mean
    assert min_mean <= metrics["expected_mean"] <= max_mean
    assert metrics["actual_std"] >= min_std
    assert metrics["expected_std"] >= min_std
    metrics["std_ratio"] = metrics["actual_std"] / metrics["expected_std"]
    if metrics["expected_std"] >= float(thresholds["reference_min_pixel_std_for_ratio"]):
        assert metrics["std_ratio"] >= float(thresholds["min_reference_std_ratio"])
    metrics["temporal_consistency"] = (
        1.0
        if len(actual_images) == 1
        else float(
            np.mean([_cosine(left, right) for left, right in zip(actual_images, actual_images[1:])])
        )
    )
    assert metrics["temporal_consistency"] >= float(thresholds["temporal_consistency"])
    return metrics


@cache
def _native_harness() -> Path:
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            "8",
            "--target",
            "qwen_image_e2e_harness",
        ],
        check=True,
        timeout=600,
    )
    executable = build_dir / "families" / FAMILY / "qwen_image_e2e_harness"
    assert executable.is_file(), f"selected {FAMILY} E2E native harness is missing: {executable}"
    return executable


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
    initial_latents: np.ndarray,
):
    del binary, model_dir
    task = manifest["task"]
    assert task in TASKS
    latents_path = tmp_path / "initial-latents.raw"
    np.ascontiguousarray(initial_latents).tofile(latents_path)
    output = tmp_path / "native.ppm"
    invocation = [
        str(_native_harness()),
        str(bundle),
        str(runtime_root),
        str(output),
        task,
        _case_text(case),
        str(latents_path),
        str(int(manifest["image_height"])),
        str(int(manifest["image_width"])),
        str(int(case["num_inference_steps"])),
        str(int(case["seed"])),
        str(float(case.get("guidance_scale", -1.0))),
        str(float(case.get("cfg_scale", -1.0))),
        str(case.get("negative_prompt", "")),
    ]
    if task == "image_edit":
        from PIL import Image

        image = np.asarray(
            Image.open(_asset(case["test_image"])).convert("RGB"), dtype=np.float32
        ) / np.float32(255.0)
        image_path = tmp_path / "edit-image.raw"
        np.ascontiguousarray(image).tofile(image_path)
        invocation.extend((str(image_path), str(image.shape[0]), str(image.shape[1])))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    assert output.is_file(), f"native {task} returned no image"
    return {"artifact": str(output)}


def _official_reference(
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
    initial_latents: np.ndarray,
):
    task = manifest["task"]
    from diffusers import DiffusionPipeline
    from PIL import Image

    pipeline = DiffusionPipeline.from_pretrained(
        model_dir, torch_dtype=_torch_dtype(case["reference_precision"]), local_files_only=True
    ).to("cuda")
    prompts = _case_text(case)
    kwargs = {
        "prompt": prompts,
        "height": int(manifest["image_height"]),
        "width": int(manifest["image_width"]),
        "num_inference_steps": int(case["num_inference_steps"]),
        "latents": _pack_reference_latents(initial_latents),
    }
    if int(manifest.get("video_num_frames", 1)) > 1:
        kwargs["num_frames"] = int(manifest["video_num_frames"])
    if case.get("negative_prompt"):
        kwargs["negative_prompt"] = case["negative_prompt"]
    if "guidance_scale" in case:
        kwargs["guidance_scale"] = float(case["guidance_scale"])
    if "cfg_scale" in case:
        kwargs["true_cfg_scale"] = float(case["cfg_scale"])
    if task == "image_edit":
        kwargs["image"] = Image.open(_asset(case["test_image"])).convert("RGB")
    output = pipeline(**kwargs)
    if task == "world_model_generation" or int(manifest.get("video_num_frames", 1)) > 1:
        images = output.frames[0]
    else:
        images = output.images
    return {
        "images": [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
    }


def _write_semantic_artifacts(case: dict, actual_images: list, expected_images: list) -> None:
    if not case.get("semantic_assessment"):
        return
    output_root = os.environ.get("TRTMC_E2E_ARTIFACT_DIR")
    if not output_root:
        return
    from PIL import Image

    output = Path(output_root) / str(case["name"])
    output.mkdir(parents=True, exist_ok=False)
    assert len(actual_images) == len(expected_images) and actual_images
    sample_count = int(case.get("semantic_samples", 1))
    assert 1 <= sample_count <= min(6, len(actual_images))
    if sample_count == 1:
        sample_indices = [(len(actual_images) - 1) // 2]
    else:
        sample_indices = [
            round(index * (len(actual_images) - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    for source_index in sample_indices:
        actual = actual_images[source_index]
        expected = expected_images[source_index]
        for label, image in (("trt", actual), ("reference", expected)):
            pixels = np.rint(np.clip(np.asarray(image), 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(pixels).save(output / f"{label}-{source_index:03d}.png")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "family": "qwen_image",
                "case": case["name"],
                "frame_count": len(actual_images),
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


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    from PIL import Image

    artifact = Path(actual["artifact"])
    if artifact.is_dir():
        actual_paths = sorted(artifact.glob("*.png"))
    else:
        actual_paths = [artifact]
    assert actual_paths
    actual_images = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        for path in actual_paths
    ]
    expected_images = expected["images"]
    assert len(actual_images) == len(expected_images)
    statistics = _assert_pixel_statistics(actual_images, expected_images, thresholds)
    if "exact_num_frames" in thresholds:
        assert len(actual_images) == int(thresholds["exact_num_frames"])
    for image in actual_images:
        if "exact_video_height" in thresholds:
            assert image.shape[0] == int(thresholds["exact_video_height"])
        if "exact_video_width" in thresholds:
            assert image.shape[1] == int(thresholds["exact_video_width"])
    cosine = min((_cosine(a, b) for a, b in zip(actual_images, expected_images)))
    rmse = max(
        (float(np.sqrt(np.mean((a - b) ** 2))) for a, b in zip(actual_images, expected_images))
    )
    print(f"{FAMILY} parity cosine={cosine:.9f}")
    print(f"{FAMILY} parity rmse_uint8={rmse * 255.0:.6f}")
    print(
        f"{FAMILY} pixel statistics: "
        f"native_mean={statistics['actual_mean']:.9f} "
        f"reference_mean={statistics['expected_mean']:.9f} "
        f"native_std={statistics['actual_std']:.9f} "
        f"reference_std={statistics['expected_std']:.9f} "
        f"std_ratio={statistics['std_ratio']:.9f} "
        f"temporal={statistics['temporal_consistency']:.9f}"
    )
    if "min_frame_cosine_uint8" in thresholds:
        assert cosine >= float(thresholds["min_frame_cosine_uint8"])
    elif "min_cosine_uint8" in thresholds:
        assert cosine >= float(thresholds["min_cosine_uint8"])
    if "max_rmse_uint8" in thresholds:
        assert rmse * 255.0 <= float(thresholds["max_rmse_uint8"])
    psnr = float("inf") if rmse == 0 else 20.0 * np.log10(1.0 / rmse)
    print(f"{FAMILY} parity psnr={psnr:.9f}")
    if "contract_psnr_threshold" in thresholds:
        assert psnr >= float(thresholds["contract_psnr_threshold"])
    elif "psnr" in thresholds:
        assert psnr >= float(thresholds["psnr"])
    if "ssim" in thresholds or "contract_ssim_threshold" in thresholds:
        scores = []
        for left, right in zip(actual_images, expected_images):
            mean_left = float(left.mean())
            mean_right = float(right.mean())
            variance_left = float(left.var())
            variance_right = float(right.var())
            covariance = float(np.mean((left - mean_left) * (right - mean_right)))
            scores.append(
                (2 * mean_left * mean_right + 0.01**2)
                * (2 * covariance + 0.03**2)
                / (
                    (mean_left**2 + mean_right**2 + 0.01**2)
                    * (variance_left + variance_right + 0.03**2)
                )
            )
        limit = (
            float(thresholds["contract_ssim_threshold"])
            if "contract_ssim_threshold" in thresholds
            else float(thresholds["ssim"])
        )
        ssim = min(scores)
        print(f"{FAMILY} parity ssim={ssim:.9f}")
        assert ssim >= limit
    if "minimum_frame_low_frequency_correlation" in thresholds:
        block = int(thresholds["low_frequency_block_size"])
        correlations = []
        for left, right in zip(actual_images, expected_images):
            height = left.shape[0] // block * block
            width = left.shape[1] // block * block
            low_left = (
                left[:height, :width]
                .reshape(height // block, block, width // block, block, 3)
                .mean(axis=(1, 3))
            )
            low_right = (
                right[:height, :width]
                .reshape(height // block, block, width // block, block, 3)
                .mean(axis=(1, 3))
            )
            correlations.append(_cosine(low_left, low_right))
        assert min(correlations) >= float(thresholds["minimum_frame_low_frequency_correlation"])
        assert float(np.mean(correlations)) >= float(
            thresholds["minimum_mean_low_frequency_correlation"]
        )
        brightness_left = np.asarray([image.mean() for image in actual_images])
        brightness_right = np.asarray([image.mean() for image in expected_images])
        assert _cosine(brightness_left, brightness_right) >= float(
            thresholds["minimum_brightness_profile_correlation"]
        )
        assert float(np.max(np.abs(brightness_left - brightness_right))) <= float(
            thresholds["maximum_frame_brightness_absolute_error"]
        )
    if len(actual_images) > 1 and "min_temporal_motion_ratio" in thresholds:
        actual_motion = np.asarray(
            [np.mean(np.abs(b - a)) for a, b in zip(actual_images, actual_images[1:])]
        )
        expected_motion = np.asarray(
            [np.mean(np.abs(b - a)) for a, b in zip(expected_images, expected_images[1:])]
        )
        ratio = float(actual_motion.mean() / max(expected_motion.mean(), 1e-12))
        assert ratio >= float(thresholds["min_temporal_motion_ratio"])
        assert ratio <= float(thresholds["max_temporal_motion_ratio"])
        assert _cosine(actual_motion, expected_motion) >= float(
            thresholds["min_temporal_profile_correlation"]
        )
    _write_semantic_artifacts(case, actual_images, expected_images)
    return


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
    _write_semantic_artifacts(
        case,
        [np.zeros((2, 2, 3), dtype=np.float32)],
        [np.ones((2, 2, 3), dtype=np.float32)],
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
        "family": "qwen_image",
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
    initial_latents = _initial_latents(manifest, case)
    actual = _native(
        binary,
        runtime_root,
        bundle,
        model_dir,
        manifest,
        case,
        tmp_path,
        initial_latents,
    )
    expected = _official_reference(model_dir, manifest, case, tmp_path, initial_latents)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
