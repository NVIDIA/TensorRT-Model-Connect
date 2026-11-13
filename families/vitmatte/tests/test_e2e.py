# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned ViTMatte image and categorical trimap to continuous alpha parity."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "vitmatte"
TASKS = frozenset({"image_trimap_to_alpha"})
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
    from transformers import VitMatteImageProcessor, VitMatteForImageMatting

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    image = (
        Image.open(TEST_ROOT / "data" / case["image"])
        .convert("RGB")
        .resize((512, 512), Image.Resampling.BICUBIC)
    )
    rgb = tmp_path / "image.rgb"
    rgb.write_bytes(np.asarray(image, dtype=np.uint8).tobytes())
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        VitMatteForImageMatting.from_pretrained(directory, dtype=torch.float32).eval().to("cuda")
    )
    processor = VitMatteImageProcessor.from_pretrained(directory)
    for index in range(2):
        states = np.ones((512, 512), dtype=np.uint8)
        states[: 64 + index * 32, :] = 0
        states[200:360, 160 + index * 64 : 300 + index * 64] = 2
        trimap = Image.fromarray(np.array([0, 128, 255], dtype=np.uint8)[states])
        inputs = processor(images=image, trimaps=trimap, return_tensors="pt").to("cuda")
        assert inputs.pixel_values.shape == (1, 4, 512, 512)
        with torch.no_grad():
            expected = reference(**inputs).alphas[0, 0].cpu().numpy()
        source = tmp_path / f"trimap-{index}.u8"
        source.write_bytes(states.tobytes())
        target = tmp_path / f"alpha-{index}.f32"
        completed = subprocess.run(
            [
                str(runtime_root / "test_vitmatte_runtime"),
                str(bundle),
                str(runtime_root),
                str(rgb),
                str(source),
                str(target),
                "512",
                "512",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert json.loads(completed.stdout.splitlines()[-1]) == dict(
            rows=512, columns=512, count=512 * 512
        )
        actual = np.fromfile(target, dtype="<f4").reshape(512, 512)
        assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
        error = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = float(np.max(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        assert maximum <= case["max_abs_error"]
        assert rmse <= case["max_rmse"]
        assert np.count_nonzero((actual > 0.05) & (actual < 0.95)) > 100
        print(f"ViTMatte trimap={index}: max_abs={maximum:.8g}, RMSE={rmse:.8g}")
