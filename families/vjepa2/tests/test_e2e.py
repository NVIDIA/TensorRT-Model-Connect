# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned V-JEPA2 native video preprocessing, patch features and frame-span parity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "vjepa2"
TASKS = frozenset({"video_to_token_features"})
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
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
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


def test_official_checkpoint_e2e(case_name, tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as vision
    from torchvision.transforms import InterpolationMode
    from transformers import AutoVideoProcessor, VJEPA2Model

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    processor = AutoVideoProcessor.from_pretrained(directory)
    image = Image.open(TEST_ROOT / "data" / case["image"]).convert("RGB")
    geometry = None
    for index, sample in enumerate(
        [image, image.transpose(Image.Transpose.ROTATE_90), image.resize((302, 300))]
    ):
        frame = np.asarray(sample, dtype=np.float32) / 255
        source = tmp_path / f"frame-{index}.f32"
        target = tmp_path / f"pixels-{index}.f32"
        source.write_bytes(frame.astype("<f4").tobytes())
        completed = subprocess.run(
            [str(runtime_root / "test_vjepa2_preprocess")],
            input=json.dumps(
                dict(
                    source=str(source), output=str(target), height=sample.height, width=sample.width
                )
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        native_geometry = json.loads(completed.stdout)
        inputs = processor(
            videos=[frame[None, ...] * 255],
            input_data_format="channels_last",
            do_sample_frames=False,
            return_tensors="pt",
        )
        expected = inputs.pixel_values_videos[0, 0].numpy()
        actual = np.fromfile(target, dtype="<f4").reshape(3, 256, 256)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        resized = vision.resize(
            torch.from_numpy(frame * 255).permute(2, 0, 1),
            [292],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        rh, rw = resized.shape[-2:]
        assert native_geometry == dict(
            height=rh, width=rw, top=round((rh - 256) / 2), left=round((rw - 256) / 2)
        )
        if index == 0:
            geometry = native_geometry
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        VJEPA2Model.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
        .eval()
        .to("cuda")
    )
    base = np.asarray(image, dtype=np.float32) / 255
    clip = np.stack([np.roll(base, 2 * index, axis=1) for index in range(64)])
    for label, video in [("forward", clip), ("reversed", clip[::-1])]:
        video = np.ascontiguousarray(video)
        source = tmp_path / f"video-{label}.f32"
        target = tmp_path / f"features-{label}.f32"
        source.write_bytes(video.astype("<f4").tobytes())
        inputs = processor(
            videos=[video * 255],
            input_data_format="channels_last",
            do_sample_frames=False,
            return_tensors="pt",
        ).to("cuda")
        assert inputs.pixel_values_videos.shape == (1, 64, 3, 256, 256)
        with torch.no_grad():
            expected = reference(**inputs, skip_predictor=True).last_hidden_state[0].cpu().numpy()
        completed = subprocess.run(
            [
                str(runtime_root / "test_vjepa2_runtime"),
                str(bundle),
                str(runtime_root),
                str(source),
                str(target),
                "64",
                str(image.height),
                str(image.width),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        metadata = json.loads(completed.stdout.splitlines()[-1])
        assert metadata["rows"] == 8192 and metadata["columns"] == 1024
        assert metadata["grid"] == [32, 16, 16] and metadata["source_frames"] == 64
        assert metadata["first"]["role"] == metadata["last"]["role"] == 1
        assert metadata["first"]["grid"] == [0, 0, 0] and metadata["last"]["grid"] == [31, 15, 15]
        assert metadata["first"]["span"] == [0, 2] and metadata["last"]["span"] == [62, 64]
        top, left, rh, rw = [geometry[key] for key in ("top", "left", "height", "width")]
        np.testing.assert_allclose(
            metadata["first"]["box"],
            [left / rw, top / rh, (left + 16) / rw, (top + 16) / rh],
            atol=1e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            metadata["last"]["box"],
            [(left + 240) / rw, (top + 240) / rh, (left + 256) / rw, (top + 256) / rh],
            atol=1e-7,
            rtol=0,
        )
        actual = np.fromfile(target, dtype="<f4").reshape(8192, 1024)
        assert expected.shape == actual.shape and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=case["max_abs_error"], rtol=0)
        cosine = np.sum(actual * expected, axis=1) / (
            np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
        )
        assert float(cosine.min()) >= case["min_cosine"]
        print(
            f"V-JEPA2 {label}: max_abs={np.max(np.abs(actual - expected)):.7g}, min_cosine={cosine.min():.8f}"
        )
