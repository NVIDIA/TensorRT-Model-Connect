# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and diffusers-reference E2E for stable_diffusion."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "stable_diffusion"
TASKS = frozenset({"image_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


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


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return (binary, runtime_root)


def _thresholds(case_name: str) -> dict:
    path = TEST_ROOT / "thresholds" / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            image_height=manifest["image_height"],
            image_width=manifest["image_width"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        )
    )


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, latents: Path,
            destination: Path):
    import numpy as np
    from PIL import Image

    invocation = [
        str(binary), "generate-image", str(bundle), "--runtime-root", str(runtime_root),
        "--prompt", str(case["prompt"]), "--num-steps", str(case["num_steps"]),
        "--guidance-scale", str(case["guidance_scale"]),
        "--initial-latents-raw", str(latents), "--output", str(destination),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    assert destination.is_file(), f"native {FAMILY} wrote no image: {completed.stdout[-500:]}"
    return np.asarray(Image.open(destination).convert("RGB"), dtype=np.float32) / 255.0


def _official_reference(model_dir: Path, manifest: dict, case: dict, latents):
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipeline = StableDiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False,
    )
    # The family samples with DDIM; the checkpoint ships PNDM, so the reference
    # is switched to match rather than comparing two different samplers.
    pipeline.scheduler = DDIMScheduler.from_pretrained(str(model_dir / "scheduler"))
    pipeline.set_progress_bar_config(disable=True)
    image = pipeline(
        str(case["prompt"]), height=manifest["image_height"], width=manifest["image_width"],
        num_inference_steps=int(case["num_steps"]),
        guidance_scale=float(case["guidance_scale"]), latents=latents, output_type="np",
    ).images[0]
    return np.asarray(image, dtype=np.float32)


def _psnr(left, right) -> float:
    import numpy as np

    mse = float(((left - right) ** 2).mean())
    return 99.0 if mse == 0.0 else 10.0 * float(np.log10(1.0 / mse))


def _ssim(left, right) -> float:

    left, right = left.mean(axis=2), right.mean(axis=2)
    mean_left, mean_right = float(left.mean()), float(right.mean())
    covariance = float(((left - mean_left) * (right - mean_right)).mean())
    stabilizer_one, stabilizer_two = 0.01 ** 2, 0.03 ** 2
    return float(
        ((2 * mean_left * mean_right + stabilizer_one) * (2 * covariance + stabilizer_two))
        / ((mean_left ** 2 + mean_right ** 2 + stabilizer_one)
           * (float(left.var()) + float(right.var()) + stabilizer_two))
    )


def _assert_parity(native, reference, thresholds: dict) -> None:
    import numpy as np

    assert np.isfinite(native).all(), f"native {FAMILY} image is not finite"
    assert native.shape == reference.shape, (
        f"image shape mismatch: native={native.shape} reference={reference.shape}"
    )
    mean = float(native.mean())
    assert float(thresholds["min_pixel_mean"]) <= mean <= float(thresholds["max_pixel_mean"]), (
        f"native image mean {mean:.4f} is outside the expected range"
    )
    assert float(native.std()) >= float(thresholds["min_pixel_std"]), (
        f"native image is too flat: std {float(native.std()):.4f}"
    )
    # An iterative sampler amplifies per-step differences, so parity is scored
    # statistically rather than exactly. A single denoise step matches the
    # reference at cosine 0.9999998; ten steps do not, and are not meant to.
    psnr = _psnr(native, reference)
    ssim = _ssim(native, reference)
    assert psnr >= float(thresholds["psnr"]), f"PSNR {psnr:.2f} is below {thresholds['psnr']}"
    assert ssim >= float(thresholds["ssim"]), f"SSIM {ssim:.4f} is below {thresholds['ssim']}"


def test_e2e(case_name: str, tmp_path: Path) -> None:
    import numpy as np
    import torch

    _, manifest, case = CASES[case_name]
    assert manifest["trust_remote_code"] is False
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]

    # The native pipeline seeds with the C++ standard library and the reference
    # with torch, so a shared seed is not a shared starting point. Both are given
    # the same latents instead.
    generator = torch.Generator().manual_seed(int(case["seed"]))
    latent_side = int(manifest["image_height"]) // 8
    latents = torch.randn((1, 4, latent_side, latent_side), generator=generator,
                          dtype=torch.float32)
    raw = tmp_path / "latents.f32"
    raw.write_bytes(np.ascontiguousarray(latents.numpy(), dtype=np.float32).tobytes())

    _build(model_dir, bundle, manifest)
    native = _native(binary, runtime_root, bundle, case, raw, tmp_path / "native.png")
    reference = _official_reference(model_dir, manifest, case, latents)
    _assert_parity(native, reference, _thresholds(case_name))
