# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI, public SDK and official-pipeline E2E for stable_video_diffusion.

The native runtime draws its noise from the same torch CPU generator stream the
official pipeline uses with ``generator=torch.Generator("cpu").manual_seed(seed)``,
so one seed is one starting point on both sides. The oracle is the official
``StableVideoDiffusionPipeline`` in FP32 (TF32 off). An FP16 bundle is also
held against the official FP16 pipeline's own distance to that FP32 run: the
Euler sampler amplifies rounding, so FP16 parity is a bounded drift, not equality.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "stable_video_diffusion"
TASKS = frozenset({"initial_image_text_to_video"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
# Config fields a testcase may override; the rest use the pipeline defaults.
CONFIG_OVERRIDES = ("min_guidance_scale", "max_guidance_scale", "fps", "motion_bucket_id",
                    "noise_aug_strength")
DEFAULT_FPS = 7


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
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(name, marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"))
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
        snapshot = snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"],
                                     local_files_only=True)
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"),
                                  "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, native_build


def _thresholds(case_name: str) -> dict:
    path = TEST_ROOT / "thresholds" / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    return env


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(BuildRequest(
        model_dir=model_dir, output_path=bundle, family=FAMILY, task=manifest["task"],
        precision=manifest["precision"], image_height=manifest["image_height"],
        image_width=manifest["image_width"], video_num_frames=manifest["video_num_frames"],
        tensor_parallel_size=manifest["tensor_parallel_size"]))


def _initial_image(manifest: dict, case: dict, tmp_path: Path) -> tuple[Path, np.ndarray]:
    """The case's photograph center-cropped to the video aspect and resized to its size."""
    from PIL import Image

    height, width = int(manifest["image_height"]), int(manifest["image_width"])
    source = Image.open(TEST_ROOT / "data" / case["image"]).convert("RGB")
    crop_width = min(source.width, round(source.height * width / height))
    crop_height = min(source.height, round(crop_width * height / width))
    left = (source.width - crop_width) // 2
    top = (source.height - crop_height) // 2
    image = source.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    path = tmp_path / "initial.png"
    image.save(path)
    return path, np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _config_arguments(case: dict) -> list[str]:
    arguments = ["--seed", str(case["seed"]), "--num-steps", str(case["num_steps"])]
    for name in CONFIG_OVERRIDES:
        if name in case:
            arguments += ["--set", f"{name}={case[name]}"]
    return arguments


def _native(binary: Path, runtime_root: Path, bundle: Path, image: Path, case: dict,
            directory: Path) -> tuple[np.ndarray, dict]:
    from PIL import Image

    completed = subprocess.run(
        [str(binary), "generate-video", str(bundle), "--runtime-root", str(runtime_root),
         "--image", str(image), "--prompt", "", "--output", str(directory),
         *_config_arguments(case)],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=3600)
    payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 1, f"generate-video printed no single result: {completed.stdout[-500:]}"
    payload = json.loads(payloads[0])
    frames = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
                       for path in payload["frames"]])
    return frames, payload


def _sdk_frames(native_build: Path, runtime_root: Path, bundle: Path, pixels: np.ndarray,
                case: dict, tmp_path: Path) -> dict[str, np.ndarray]:
    """Run the public C (uint8 input) and C++ (float32 input) consumers with the case's seed."""
    raw = tmp_path / "sdk-input.rgb.u8"
    np.ascontiguousarray(pixels).tofile(raw)
    height, width = pixels.shape[:2]
    frames = {}
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        output = tmp_path / f"sdk-{language}.f32"
        completed = subprocess.run(
            [str(consumer), str(bundle), str(runtime_root), str(raw), str(height), str(width),
             str(case["seed"]), str(case["num_steps"]), str(output)],
            check=True, capture_output=True, text=True, env=_environment(runtime_root),
            timeout=3600)
        payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(payloads) == 1, f"the {language} consumer must print one result"
        payload = json.loads(payloads[0])
        assert payload["task"] == "initial_image_text_to_video"
        values = np.fromfile(output, dtype=np.float32)
        frames[language] = values.reshape(payload["frames"], height, width, 3)
    return frames


def _official(model_dir: Path, manifest: dict, case: dict, image: Path,
              precision: str) -> np.ndarray:
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    dtype = torch.float32 if precision == "fp32" else torch.float16
    pipeline = StableVideoDiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=dtype)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    height, width = int(manifest["image_height"]), int(manifest["image_width"])
    frames = int(manifest["video_num_frames"])
    overrides = {name: case[name] for name in CONFIG_OVERRIDES if name in case}
    # The pipeline's first draw is the float32 conditioning-image noise and its second
    # the latents, in the pipeline dtype. Both precisions are given the float32 second
    # draw the native runtime makes, cast to the pipeline dtype, so their starting
    # point is the same by construction.
    generator = torch.Generator("cpu").manual_seed(int(case["seed"]))
    torch.randn((1, 3, height, width), generator=generator)
    factor = pipeline.vae_scale_factor
    latents = torch.randn((1, frames, pipeline.unet.config.in_channels // 2, height // factor,
                           width // factor), generator=generator)
    with torch.no_grad():
        output = pipeline(
            Image.open(image).convert("RGB"), height=height, width=width, num_frames=frames,
            num_inference_steps=int(case["num_steps"]), decode_chunk_size=frames,
            generator=torch.Generator("cpu").manual_seed(int(case["seed"])),
            latents=latents.to(dtype), output_type="np", **overrides).frames[0]
    del pipeline
    torch.cuda.empty_cache()
    return np.asarray(output, dtype=np.float32)


def _quantize(frames: np.ndarray) -> np.ndarray:
    """The CLI's PNG rounding: clamp to [0, 1], scale by 255 in float32, add 0.5, truncate."""
    values = np.clip(frames.astype(np.float32), np.float32(0.0), np.float32(1.0))
    return (values * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    return 99.0 if mse == 0.0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def _frame_psnrs(native: np.ndarray, reference: np.ndarray) -> list[float]:
    return [_psnr(left, right) for left, right in zip(native, reference)]


def test_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    assert manifest["trust_remote_code"] is False
    thresholds = _thresholds(case_name)
    binary, runtime_root, native_build = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]
    image, pixels = _initial_image(manifest, case, tmp_path)
    frames = int(manifest["video_num_frames"])
    shape = (frames, int(manifest["image_height"]), int(manifest["image_width"]), 3)

    _build(model_dir, bundle, manifest)
    native, payload = _native(binary, runtime_root, bundle, image, case, tmp_path / "native")
    fps = int(case.get("fps", DEFAULT_FPS))
    print(f"{FAMILY} {case_name}: native generation {payload['inference_ms']:.0f} ms")
    assert native.shape == shape
    assert payload["conditioned_prefix_frames"] == 0, "every frame is generated"
    assert payload["timestamps_seconds"] == pytest.approx([index / fps for index in range(frames)])

    # The public C and C++ SDKs run the same Task on the same pixels.
    for language, values in _sdk_frames(native_build, runtime_root, bundle, pixels, case,
                                        tmp_path).items():
        assert values.shape == shape
        sdk_psnr = min(_frame_psnrs(_quantize(values), native))
        print(f"{FAMILY} {case_name} {language} SDK vs CLI: min frame PSNR {sdk_psnr:.2f} dB")
        assert sdk_psnr >= float(thresholds["min_sdk_frame_psnr"])

    # Task invariants: the clip starts at the conditioning image and moves.
    first_psnr = _psnr(native[0], pixels)
    motion = float(np.mean(np.abs(native[-1].astype(np.float64) - native[0].astype(np.float64))))
    print(f"{FAMILY} {case_name}: first frame vs image {first_psnr:.2f} dB, "
          f"mean |last - first| {motion:.2f} / 255")
    assert first_psnr >= float(thresholds["min_first_frame_psnr_vs_image"])
    assert motion >= float(thresholds["min_mean_abs_motion"])

    references = {precision: _quantize(_official(model_dir, manifest, case, image, precision))
                  for precision in case["reference_precisions"]}
    exact = references["fp32"]
    native_psnrs = _frame_psnrs(native, exact)
    print(f"{FAMILY} {case_name} native {manifest['precision']} vs official fp32: "
          f"min {min(native_psnrs):.2f} mean {np.mean(native_psnrs):.2f} dB")
    assert min(native_psnrs) >= float(thresholds["min_frame_psnr_vs_official_fp32"])
    assert float(np.mean(native_psnrs)) >= float(thresholds["min_mean_psnr_vs_official_fp32"])
    if "fp16" in references:
        official_psnrs = _frame_psnrs(references["fp16"], exact)
        print(f"{FAMILY} {case_name} official fp16 vs official fp32: "
              f"min {min(official_psnrs):.2f} mean {np.mean(official_psnrs):.2f} dB")
        shortfall = float(np.mean(official_psnrs)) - float(np.mean(native_psnrs))
        assert shortfall <= float(thresholds["max_mean_psnr_shortfall_vs_official_fp16_db"])
