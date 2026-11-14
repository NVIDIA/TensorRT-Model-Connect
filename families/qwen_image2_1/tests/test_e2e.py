# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for qwen_image2_1.

One case builds a bundle from the pinned checkpoint, generates an image through the native CLI and
through the public C and C++ SDK consumers, and compares each against the official
``QwenImage21Pipeline`` started from the same initial latents. The same bundle then serves the CLI
paths the comparison does not take: generation from a seed, and the requests the runtime rejects.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "qwen_image2_1"
TASKS = frozenset({"text_to_image"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
LATENT_CHANNELS = 64
VAE_SCALE_FACTOR = 16
IMAGE_CHANNELS = 4
SDK_LANGUAGES = ("c", "cpp")

# The sampler amplifies rounding differences, so two BF16 implementations of this model do not
# produce the same pixels. At 512 x 512 and 20 steps, over twelve prompt / seed pairs, the official
# pipeline's own BF16 and FP32 runs are 25-38 dB PSNR apart without guidance and 23-33 dB apart at
# guidance scale 4; the native image is 20-39 dB and 17-29 dB from the official BF16 run. A wrong
# graph or sampler lands far below both ranges: ignoring the guidance scale, for instance, scores
# 10 dB, a windowed SSIM of 0.68 and a layout correlation of 0.70.
#
# The limits therefore bound drift, not bit equality: same composition (block means), same local
# structure (windowed SSIM), same pixels within the PSNR floor. Exactness of the graphs is
# asserted in FP32 by test_engine_parity.py. Every case uses a prompt and seed on which the
# official BF16 and FP32 runs agree to at least 32 dB, which keeps the native image well inside
# the limits; the family README lists the values measured per case.
DEFAULT_THRESHOLDS = {
    "min_psnr_db": 20.0,
    "min_ssim": 0.90,
    "ssim_window": 8,
    "min_layout_correlation": 0.95,
    "max_alpha_mean_abs_error": 0.02,
    "layout_block_size": 32,
}


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
    assert torch.cuda.device_count() >= required_gpus
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
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _thresholds(case_name: str) -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if path.is_file():
        thresholds.update(json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"])
    return thresholds


def _initial_latents(manifest: dict, case: dict) -> np.ndarray:
    """Standard-normal latents ``[channels, height / 16, width / 16]`` shared by both sides."""
    height = int(manifest["image_height"])
    width = int(manifest["image_width"])
    assert height % VAE_SCALE_FACTOR == 0 and width % VAE_SCALE_FACTOR == 0
    generator = np.random.default_rng(int(case["seed"]))
    return generator.standard_normal(
        (LATENT_CHANNELS, height // VAE_SCALE_FACTOR, width // VAE_SCALE_FACTOR), dtype=np.float32
    )


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    return env


def _guidance(case: dict) -> tuple[str, str] | None:
    """The case's classifier-free guidance scale and negative prompt, when it declares them."""
    if "cfg_scale" not in case:
        return None
    return (repr(float(case["cfg_scale"])), str(case["negative_prompt"]))


def _generate(
    binary: Path, runtime_root: Path, bundle: Path, prompt: str, output: Path, *options: str
) -> subprocess.CompletedProcess:
    """One native ``generate-image`` run; the caller decides whether it has to succeed."""
    return subprocess.run(
        [
            str(binary),
            "generate-image",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--prompt",
            prompt,
            "--output",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
        env=_environment(runtime_root),
        timeout=3600,
    )


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    latents_path: Path,
    output: Path,
) -> np.ndarray:
    """Generate through the native CLI; returns the PNG as float ``[H, W, 4]`` in [0, 1]."""
    from PIL import Image

    guidance = _guidance(case)
    completed = _generate(
        binary,
        runtime_root,
        bundle,
        str(case["prompt"]),
        output,
        "--initial-latents-raw",
        str(latents_path),
        "--num-steps",
        str(int(case["num_inference_steps"])),
        "--height",
        str(int(manifest["image_height"])),
        "--width",
        str(int(manifest["image_width"])),
        *(("--cfg-scale", guidance[0], "--negative-prompt", guidance[1]) if guidance else ()),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    record_evidence("native_log", completed.stderr[-4000:])
    payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 1, f"native generate-image returned no result: {completed.stdout[-500:]}"
    assert payloads[0]["height"] == int(manifest["image_height"])
    assert payloads[0]["width"] == int(manifest["image_width"])
    with Image.open(output) as image:
        assert image.mode == "RGBA", f"native image must be RGBA, got {image.mode}"
        return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def _assert_native_contract(
    binary: Path, runtime_root: Path, bundle: Path, manifest: dict, case: dict, directory: Path
) -> None:
    """Generation from a seed, and the requests the runtime must reject, on the built bundle."""
    from PIL import Image

    directory.mkdir()
    prompt = str(case["prompt"])
    steps = ("--num-steps", str(int(case["num_inference_steps"])))
    size = (int(manifest["image_width"]), int(manifest["image_height"]))

    def seeded(name: str, seed: int) -> bytes:
        output = directory / f"{name}.png"
        completed = _generate(
            binary, runtime_root, bundle, prompt, output, "--seed", str(seed), *steps
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        with Image.open(output) as image:
            assert image.mode == "RGBA" and image.size == size
            pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        assert pixels[..., :3].std() > 0.02, "an image generated from a seed must not be flat"
        return output.read_bytes()

    first = seeded("seed-7", 7)
    assert seeded("seed-7-again", 7) == first, "the same seed must reproduce the image"
    assert seeded("seed-8", 8) != first, "another seed must give another image"

    short_latents = directory / "short-latents.raw"
    np.zeros(LATENT_CHANNELS, dtype=np.float32).tofile(short_latents)
    rejections = {
        "one step": (prompt, ("--seed", "7", "--num-steps", "1"), "num_steps"),
        "another size": (
            prompt,
            ("--seed", "7", *steps, "--height", str(size[1] + 32)),
            "rebuild it for another size",
        ),
        "a prompt over the text budget": (
            "word " * (2 * int(manifest["max_sequence_length"])),
            ("--seed", "7", *steps),
            "text tokens",
        ),
        "initial latents of the wrong size": (
            prompt,
            (*steps, "--initial-latents-raw", str(short_latents)),
            "initial latents",
        ),
    }
    for label, (text, options, message) in rejections.items():
        output = directory / "rejected.png"
        completed = _generate(binary, runtime_root, bundle, text, output, *options)
        assert completed.returncode != 0, f"{label} must be rejected"
        assert message in completed.stderr, f"{label}: {completed.stderr[-500:]}"
        assert not output.exists(), f"{label} must not write an image"
    record_evidence("native_contract", {"seeded_runs": 3, "rejected": sorted(rejections)})


def _sdk_images(
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    latents_path: Path,
    tmp_path: Path,
) -> dict[str, np.ndarray]:
    """Generate through the public C and C++ SDK consumers; float ``[H, W, 4]`` per language."""
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    images = {}
    for language in SDK_LANGUAGES:
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        pixels = tmp_path / f"sdk-{language}.f32"
        completed = subprocess.run(
            [
                str(consumer),
                str(bundle),
                str(runtime_root),
                str(case["prompt"]),
                str(latents_path),
                str(int(case["num_inference_steps"])),
                str(pixels),
                *(_guidance(case) or ()),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=_environment(runtime_root),
            timeout=int(case.get("runtime_timeout_s", 3600)),
        )
        payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(payloads) == 1, f"the {language} consumer must print one result"
        shape = json.loads(payloads[0])
        assert shape == {
            "task": "text_to_image",
            "height": int(manifest["image_height"]),
            "width": int(manifest["image_width"]),
            "channels": IMAGE_CHANNELS,
        }
        images[language] = np.fromfile(pixels, dtype=np.float32).reshape(
            shape["height"], shape["width"], shape["channels"]
        )
    return images


def _official_reference(
    model_dir: Path, manifest: dict, case: dict, initial_latents: np.ndarray
) -> np.ndarray:
    """Run the official pipeline from the same latents; float ``[H, W, 4]`` in [0, 1]."""
    import torch
    from diffusers import QwenImage21Pipeline

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[case["reference_precision"]]
    pipeline = QwenImage21Pipeline.from_pretrained(
        model_dir, torch_dtype=dtype, local_files_only=True
    ).to("cuda")
    channels, height, width = initial_latents.shape
    packed = torch.from_numpy(initial_latents.copy()).reshape(1, channels, height * width)
    guidance = {}
    if "cfg_scale" in case:
        guidance = {
            "true_cfg_scale": float(case["cfg_scale"]),
            "negative_prompt": str(case["negative_prompt"]),
        }
    output = pipeline(
        prompt=str(case["prompt"]),
        **guidance,
        height=int(manifest["image_height"]),
        width=int(manifest["image_width"]),
        num_inference_steps=int(case["num_inference_steps"]),
        latents=packed.transpose(1, 2),
        # The native denoiser recomputes the text rows every step, which is the pipeline's
        # uncached attention path; its cached path is a different, equally valid BF16 trajectory.
        use_kv_cache=False,
        output_type="np",
    )
    image = np.asarray(output.images[0], dtype=np.float32)
    del pipeline
    torch.cuda.empty_cache()
    return image


def _block_means(image: np.ndarray, block: int) -> np.ndarray:
    height = image.shape[0] // block * block
    width = image.shape[1] // block * block
    return (
        image[:height, :width]
        .reshape(height // block, block, width // block, block, image.shape[2])
        .mean(axis=(1, 3))
    )


def _windowed_ssim(actual: np.ndarray, expected: np.ndarray, window: int) -> float:
    """Mean SSIM of the luminance over non-overlapping ``window`` x ``window`` tiles."""

    def tiles(image: np.ndarray) -> np.ndarray:
        luminance = image.mean(axis=2)
        height = luminance.shape[0] // window * window
        width = luminance.shape[1] // window * window
        return (
            luminance[:height, :width]
            .reshape(height // window, window, width // window, window)
            .transpose(0, 2, 1, 3)
            .reshape(-1, window * window)
        )

    left, right = tiles(actual), tiles(expected)
    mean_left, mean_right = left.mean(axis=1), right.mean(axis=1)
    covariance = ((left - mean_left[:, None]) * (right - mean_right[:, None])).mean(axis=1)
    luminance_term = (2 * mean_left * mean_right + 0.01**2) / (
        mean_left**2 + mean_right**2 + 0.01**2
    )
    structure_term = (2 * covariance + 0.03**2) / (left.var(axis=1) + right.var(axis=1) + 0.03**2)
    return float((luminance_term * structure_term).mean())


def _metrics(actual: np.ndarray, expected: np.ndarray, block: int, window: int) -> dict:
    """Image-level agreement between two float ``[H, W, 4]`` RGBA images in [0, 1]."""
    rgb_actual = actual[..., :3].astype(np.float64)
    rgb_expected = expected[..., :3].astype(np.float64)
    mse = float(np.mean((rgb_actual - rgb_expected) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    ssim = _windowed_ssim(rgb_actual, rgb_expected, window)
    low_actual = _block_means(rgb_actual, block)
    low_expected = _block_means(rgb_expected, block)
    low_actual = (low_actual - low_actual.mean()).reshape(-1)
    low_expected = (low_expected - low_expected.mean()).reshape(-1)
    denominator = float(np.linalg.norm(low_actual) * np.linalg.norm(low_expected))
    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "layout_correlation": float(np.dot(low_actual, low_expected) / denominator)
        if denominator > 0
        else 0.0,
        "alpha_mean_abs_error": float(np.mean(np.abs(actual[..., 3] - expected[..., 3]))),
        "actual_std": float(rgb_actual.std()),
        "expected_std": float(rgb_expected.std()),
    }


def _assert_parity(label: str, actual: np.ndarray, expected: np.ndarray, thresholds: dict) -> None:
    assert actual.shape == expected.shape, f"{label}: {actual.shape} vs {expected.shape}"
    assert actual.shape[2] == IMAGE_CHANNELS
    assert np.isfinite(actual).all() and actual.min() >= 0.0 and actual.max() <= 1.0
    metrics = _metrics(
        actual, expected, int(thresholds["layout_block_size"]), int(thresholds["ssim_window"])
    )
    record_evidence(f"metrics_{label}", metrics)
    print(f"{FAMILY} {label} parity: " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))
    assert metrics["expected_std"] > 0.02, "the reference image is degenerate"
    assert metrics["psnr_db"] >= float(thresholds["min_psnr_db"])
    assert metrics["ssim"] >= float(thresholds["min_ssim"])
    assert metrics["layout_correlation"] >= float(thresholds["min_layout_correlation"])
    assert metrics["alpha_mean_abs_error"] <= float(thresholds["max_alpha_mean_abs_error"])


def _record_views(images: dict[str, np.ndarray], directory: Path) -> None:
    """Archive downscaled previews of the compared images; previews never enter the comparison."""
    if not evidence_enabled():
        return
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    views = []
    for label, image in images.items():
        path = directory / f"{label}.png"
        preview = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), "RGBA")
        preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
        preview.save(path, format="PNG")
        views.append({"title": label, "image": path, "caption": f"{label} output, downscaled"})
    record_evidence("views", views)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    record_evidence("bundle", {"bytes": bundle.stat().st_size})

    initial_latents = _initial_latents(manifest, case)
    latents_path = tmp_path / "initial-latents.raw"
    np.ascontiguousarray(initial_latents).tofile(latents_path)
    with evidence_stage("native"):
        images = {
            "cli": _native(
                binary, runtime_root, bundle, manifest, case, latents_path, tmp_path / "native.png"
            )
        }
    with evidence_stage("sdk"):
        for language, image in _sdk_images(
            runtime_root, bundle, manifest, case, latents_path, tmp_path
        ).items():
            images[f"sdk_{language}"] = image
    with evidence_stage("contract"):
        _assert_native_contract(
            binary, runtime_root, bundle, manifest, case, tmp_path / "contract"
        )
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, manifest, case, initial_latents)
    _record_views({**images, "reference": expected}, tmp_path / "report-views")

    thresholds = record_evidence("thresholds", _thresholds(case_name))
    with evidence_stage("compare"):
        for label, image in images.items():
            _assert_parity(label, image, expected, thresholds)
