# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, C and C++ SDK, and official-reference E2E for videomae.

The shared CLI has no command for masked_video_text_to_video, so the native
path is the family's public SDK consumers (tests/sdk_consumer.c and .cpp).
Each case reconstructs the masked tubes of a 16-frame clip, a camera pan over
the test photograph, and checks:

- the C consumer (8-bit frames) and the C++ consumer (the same pixels as
  float32, with a timeline) return identical pixels;
- every conditioning tube keeps its source pixels exactly;
- every reconstructed pixel agrees with the reference: transformers'
  VideoMAEForPreTraining in float32 with eager attention on the CPU, its
  predictions turned into pixels the way the authors' visualization does
  (``run_videomae_vis.py``: times the tube's source standard deviation plus
  1e-6, plus its source mean), clamped to [0, 1];
- the reconstruction beats filling each masked tube with its mean, by as much
  as the reference does.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

import pytest

from families.videomae import config as config_module

FAMILY = "videomae"
TASKS = frozenset({"masked_video_text_to_video"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

# Largest |native - reference| of a reconstructed pixel ([0, 1] scale). Measured
# on 1x B300, both cases, the same over four runs: 6.6e-6 (box) and 2.1e-5 (tube).
# Single defects injected into the reference move pixels by 8.9e-4 (biased variance
# in the restoration) to 1.0 (tanh GELU 1.7e-3 / 4.9e-3; q and v biases swapped,
# final encoder norm missing, tubes reordered: 0.86 or more); a decoder-norm epsilon
# of 1e-12 or a missing +1e-6 on the std (1.1e-5 to 1.9e-5) are below what any
# float32 pixel bound can resolve.
PIXEL_TOLERANCE = 1e-4
# Reconstruction error over the masked tubes relative to filling each masked
# tube with its source mean (1.0 = no better than the mean). Measured: 0.33 (tube)
# and 0.20 (box), native and reference equal to 2e-7.
QUALITY_CEILING = 0.8
QUALITY_AGREEMENT = 1e-3

# The clip: crops of the model's frame size from the test photograph, the
# window moving 2 pixels down and 8 right per frame.
PAN_ORIGIN = (60, 100)
PAN_STEP = (2, 8)
FRAME_RATE = 30.0  # the timeline the C++ consumer attaches


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
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    # Pytest skips an empty parametrization silently. A selection that names this
    # family, one of its manifests or one of its cases and still matches nothing
    # is an error; a selection of another family skips this file.
    owned = {FAMILY} | set(CASES) | {manifest["name"] for _, manifest, _ in CASES.values()}
    if not selected and model_filters & owned:
        raise pytest.UsageError(
            f"the E2E selection names {FAMILY} but matches none of its cases "
            f"({', '.join(sorted(CASES))}); --e2e-testcase: {', '.join(sorted(testcase_filters))}")
    return sorted(selected), True


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
        # The family reads only these files; the checkpoint's pytorch_model.bin is never needed.
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=True,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime(manifest: dict) -> tuple[Path, Path]:
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    assert int(manifest["tensor_parallel_size"]) == 1, "videomae runs on one GPU"
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return runtime_root, native_build


_BUNDLES: dict[str, Path] = {}


def _bundle(model_dir: Path, manifest: dict, tmp_path_factory) -> Path:
    """Build each manifest's bundle once per session; its cases share it."""
    key = manifest["name"]
    if key not in _BUNDLES:
        bundle = tmp_path_factory.mktemp("bundle") / manifest["bundle"]
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            )
        )
        _BUNDLES[key] = bundle
    return _BUNDLES[key]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    record_evidence("inputs", {"asset": path})
    return path


def _clip(case: dict, cfg: dict) -> np.ndarray:
    """``[frames, size, size, 3]`` uint8: a pan over the case's photograph."""
    from PIL import Image

    image = np.asarray(Image.open(_asset(case["clip_image"])).convert("RGB"))
    size = cfg["image_size"]
    frames = []
    for index in range(cfg["num_frames"]):
        top = PAN_ORIGIN[0] + PAN_STEP[0] * index
        left = PAN_ORIGIN[1] + PAN_STEP[1] * index
        frame = image[top: top + size, left: left + size]
        assert frame.shape == (size, size, 3), "the pan must stay inside the photograph"
        frames.append(frame)
    return np.ascontiguousarray(np.stack(frames))


def _tube_mask(case: dict, cfg: dict) -> np.ndarray:
    """One bool per tube, True where the tube is reconstructed."""
    spec = case["mask"]
    grid, patch = cfg["grid"], cfg["patch_size"]
    tubelets = cfg["num_frames"] // cfg["tubelet_size"]
    if spec["kind"] == "tube":
        # VideoMAE's tube masking: one random spatial pattern for every tubelet.
        pattern = np.zeros(grid * grid, dtype=bool)
        pattern[: int(spec["ratio"] * grid * grid)] = True
        np.random.default_rng(int(spec["seed"])).shuffle(pattern)
        return np.tile(pattern, tubelets)
    assert spec["kind"] == "box", f"unknown mask kind {spec['kind']!r}"
    # A box of whole patches in every frame.
    box = [int(spec[key]) for key in ("top", "left", "height", "width")]
    assert all(value % patch == 0 for value in box), "box edges must lie on the patch grid"
    top, left, height, width = (value // patch for value in box)
    pattern = np.zeros((grid, grid), dtype=bool)
    pattern[top: top + height, left: left + width] = True
    return np.tile(pattern.reshape(-1), tubelets)


def _pixel_mask(tubes: np.ndarray, cfg: dict) -> np.ndarray:
    """``[frames, size, size]`` float32: the tube mask in the Task's FHW layout."""
    grid, patch, tubelet = cfg["grid"], cfg["patch_size"], cfg["tubelet_size"]
    plane = tubes.reshape(-1, grid, grid)
    plane = np.repeat(np.repeat(np.repeat(plane, tubelet, axis=0), patch, axis=1), patch, axis=2)
    return np.ascontiguousarray(plane.astype(np.float32))


def _rescaled(clip: np.ndarray) -> np.ndarray:
    """The runtime's float view of 8-bit frames: double product with 1/255, then float32."""
    return (clip.astype(np.float64) * (1.0 / 255.0)).astype(np.float32)


def _sdk(native_build: Path, runtime_root: Path, bundle: Path, language: str, clip: np.ndarray,
         pixel_mask: np.ndarray, case: dict, directory: Path) -> tuple[dict, np.ndarray]:
    consumer = native_build / f"test_{FAMILY}_sdk_{language}"
    assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
    pixel_format = "u8" if language == "c" else "f32"
    frames_path = directory / f"frames-{language}.{pixel_format}"
    (clip if pixel_format == "u8" else _rescaled(clip)).tofile(frames_path)
    mask_path = directory / "mask.f32"
    pixel_mask.tofile(mask_path)
    output = directory / f"result-{language}.f32"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    frames, size = clip.shape[0], clip.shape[1]
    completed = subprocess.run(
        [str(consumer), str(bundle), str(runtime_root), str(frames_path), pixel_format,
         str(frames), str(size), str(size), str(mask_path), str(output)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    record_evidence(f"sdk_{language}_process",
                    {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr})
    payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 1, f"SDK {language} consumer must print one result"
    pixels = np.fromfile(output, dtype=np.float32)
    assert pixels.size == clip.size, f"SDK {language} consumer wrote another clip size"
    return payloads[0], pixels.reshape(clip.shape)


def _official_reference(model_dir: Path, clip: np.ndarray, tubes: np.ndarray,
                        cfg: dict) -> np.ndarray:
    """Reference pixels: transformers' prediction, restored as the authors' visualization does."""
    import torch
    from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor
    from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    processor = VideoMAEImageProcessor.from_pretrained(model_dir)
    pixel_values = processor(list(clip), return_tensors="pt")["pixel_values"]
    model = VideoMAEForPreTraining.from_pretrained(
        model_dir, attn_implementation="eager", dtype=torch.float32).eval()
    masked = torch.from_numpy(tubes)
    with torch.no_grad():
        logits = model(pixel_values, bool_masked_pos=masked[None]).logits[0]

    # The modeling code and the visualization undo the input normalization with
    # these constants; the checkpoint's processor uses the same statistics.
    assert list(processor.image_mean) == list(IMAGENET_DEFAULT_MEAN)
    assert list(processor.image_std) == list(IMAGENET_DEFAULT_STD)
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 1, 3, 1, 1)
    frames = pixel_values * std + mean
    count, size = cfg["num_frames"], cfg["image_size"]
    tubelet, patch, grid = cfg["tubelet_size"], cfg["patch_size"], cfg["grid"]
    patches = frames.view(1, count // tubelet, tubelet, 3, grid, patch, grid, patch)
    patches = patches.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(cfg["tubes"], -1, 3)
    center = patches.mean(dim=-2, keepdim=True)
    scale = patches.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
    restored = patches.clone()
    restored[masked] = logits.view(int(masked.sum()), -1, 3) * scale[masked] + center[masked]
    restored = restored.clamp(0.0, 1.0).view(1, count // tubelet, grid, grid, tubelet, patch, patch, 3)
    return restored.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(count, size, size, 3).numpy()


def _tube_means(source: np.ndarray, cfg: dict) -> np.ndarray:
    """Each tube's per-channel source mean, broadcast over the tube's pixels."""
    count, size = cfg["num_frames"], cfg["image_size"]
    tubelet, patch, grid = cfg["tubelet_size"], cfg["patch_size"], cfg["grid"]
    tubes = source.astype(np.float64).reshape(count // tubelet, tubelet, grid, patch, grid, patch, 3)
    means = tubes.mean(axis=(1, 3, 5), keepdims=True)
    return np.broadcast_to(means, tubes.shape).reshape(count, size, size, 3)


def _quality(result: np.ndarray, source: np.ndarray, baseline: np.ndarray,
             selected: np.ndarray) -> float:
    error = np.mean((result.astype(np.float64)[selected] - source[selected]) ** 2)
    return float(error / np.mean((baseline[selected] - source[selected]) ** 2))


def _bytes(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames
    return np.round(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)


def _sheet(frames: np.ndarray, path: Path) -> Path:
    """The clip as one PNG, four frames per row, for the evidence report."""
    from PIL import Image

    pixels = _bytes(frames)
    count, height, width, _ = pixels.shape
    columns = 4
    sheet = np.zeros((-(-count // columns) * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(pixels):
        row, column = divmod(index, columns)
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = frame
    Image.fromarray(sheet).save(path)
    return path


def _animation(frames: np.ndarray, path: Path) -> Path:
    """The clip as an animated GIF, for the evidence report."""
    from PIL import Image

    images = [Image.fromarray(frame) for frame in _bytes(frames)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=125, loop=0)
    return path


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path, tmp_path_factory) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence("thresholds", {"pixel_tolerance": PIXEL_TOLERANCE,
                                   "quality_ceiling": QUALITY_CEILING,
                                   "quality_agreement": QUALITY_AGREEMENT})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"),
                                   "hf_revision": manifest.get("hf_revision")})
    runtime_root, native_build = _runtime(manifest)
    cfg = config_module.resolve(model_dir)
    with evidence_stage("build"):
        bundle = _bundle(model_dir, manifest, tmp_path_factory)

    clip = _clip(case, cfg)
    tubes = _tube_mask(case, cfg)
    pixel_mask = _pixel_mask(tubes, cfg)
    source = _rescaled(clip)
    # What the model sees: the reconstructed tubes are shown gray.
    shown = np.where(pixel_mask[..., None] > 0, np.float32(0.5), source)
    record_evidence("inputs", {"clip": _animation(clip, tmp_path / "input.gif"),
                               "frames": _sheet(clip, tmp_path / "input.png"),
                               "conditioning_tubes": _sheet(shown, tmp_path / "conditioning.png"),
                               "reconstructed_tubes": int(tubes.sum()), "tubes": int(tubes.size)})
    with evidence_stage("native"):
        c_payload, c_pixels = _sdk(native_build, runtime_root, bundle, "c", clip, pixel_mask,
                                   case, tmp_path)
        cpp_payload, cpp_pixels = _sdk(native_build, runtime_root, bundle, "cpp", clip,
                                       pixel_mask, case, tmp_path)
    native_frames = _sheet(c_pixels, tmp_path / "native.png")
    record_evidence("native", {"video": _animation(c_pixels, tmp_path / "native.gif"),
                               "frames": native_frames, "c": c_payload, "cpp": cpp_payload})
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, clip, tubes, cfg)
    reference_frames = _sheet(expected, tmp_path / "reference.png")
    record_evidence("reference", {
        "video": _animation(expected, tmp_path / "reference.gif"), "frames": reference_frames,
        "implementation": "transformers VideoMAEForPreTraining, float32, eager attention, CPU; "
                          "masked tubes restored with their source mean and standard deviation"})

    with evidence_stage("compare"):
        count, size = cfg["num_frames"], cfg["image_size"]
        for payload, timeline in ((c_payload, []),
                                  (cpp_payload, [index / FRAME_RATE for index in range(count)])):
            assert payload["task"] == "masked_video_text_to_video"
            assert (payload["frames"], payload["height"], payload["width"], payload["channels"]) == (
                count, size, size, 3)
            assert payload["conditioned_prefix_frames"] == 0
            assert payload["timestamps_seconds"] == timeline
            assert payload["refused"] == ["prompt", "mixed_tube", "empty_mask", "frame_count"]
        assert c_pixels.shape == cpp_pixels.shape == expected.shape == source.shape

        reconstructed = pixel_mask.astype(bool)[..., None].repeat(3, axis=-1)
        consumers_agree = bool(np.array_equal(c_pixels, cpp_pixels))
        conditioning_kept = bool(np.array_equal(c_pixels[~reconstructed], source[~reconstructed]))
        error = float(np.max(np.abs(c_pixels[reconstructed] - expected[reconstructed])))
        baseline = _tube_means(source, cfg)
        native_quality = _quality(c_pixels, source, baseline, reconstructed)
        reference_quality = _quality(expected, source, baseline, reconstructed)
        quality_gap = abs(native_quality - reference_quality)
        record_evidence("reference_comparison", {
            "label": f"{case_name}: reconstructed clip",
            "scope": "independent_reference",
            "enforced": True,
            "native": native_frames,
            "reference": reference_frames,
            "checks": [
                {"name": "max_abs_pixel_error", "label": "Largest |native - reference| pixel",
                 "scope": "independent_reference", "actual": error, "operator": "<=",
                 "expected": PIXEL_TOLERANCE, "passed": error <= PIXEL_TOLERANCE},
                {"name": "quality_gap", "label": "|native - reference| error relative to tube means",
                 "scope": "independent_reference", "actual": quality_gap, "operator": "<=",
                 "expected": QUALITY_AGREEMENT, "passed": quality_gap <= QUALITY_AGREEMENT},
                {"name": "quality", "label": "Native error relative to filling tubes with their mean",
                 "scope": "contract", "actual": native_quality, "operator": "<=",
                 "expected": QUALITY_CEILING, "passed": native_quality <= QUALITY_CEILING},
                {"name": "conditioning_kept", "label": "Conditioning tubes keep their source pixels",
                 "scope": "contract", "actual": conditioning_kept, "operator": "==",
                 "expected": True, "passed": conditioning_kept},
                {"name": "consumers_agree", "label": "C and C++ consumers return identical pixels",
                 "scope": "contract", "actual": consumers_agree, "operator": "==",
                 "expected": True, "passed": consumers_agree},
            ],
        })
        assert consumers_agree, "C and C++ consumers disagree"
        assert conditioning_kept, "conditioning tubes must keep their source pixels"
        assert error <= PIXEL_TOLERANCE, error
        assert native_quality <= QUALITY_CEILING, native_quality
        assert quality_gap <= QUALITY_AGREEMENT, quality_gap
