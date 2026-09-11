# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and ultralytics reference proof for YOLO11."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "yolo11"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "object_detection"
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selection(config) -> set[str]:
    selected = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option, default=[]) or []:
            selected.update(value.strip() for value in str(raw).split(",") if value.strip())
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selection(metafunc.config)
    names = [
        name
        for name, (manifest, _) in CASES.items()
        if not selected or selected & {FAMILY, name, manifest["name"]}
    ]
    if not selected:
        names = [
            pytest.param(
                name,
                marks=pytest.mark.skip(reason="real YOLO11 E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected YOLO11 E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected YOLO11 E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_YOLO11_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_YOLO11_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    )


def _asset(case: dict) -> Path:
    path = TEST_ROOT / str(case["test_image"])
    assert path.is_file(), f"selected YOLO11 E2E image is missing: {path}"
    return path


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )
    image = _asset(case)
    completed = subprocess.run(
        [
            str(binary),
            "detect",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    actual = json.loads(completed.stdout)

    import torch
    from PIL import Image
    from ultralytics.utils.ops import non_max_suppression

    # The reference must run the archive under test, and must see the same
    # letterboxed pixels the runtime seam produced.
    blob = torch.load(str(model_dir / "yolo11n.pt"), map_location="cpu", weights_only=False)
    reference = blob["model"].float().eval()

    size = 640
    source = Image.open(image).convert("RGB")
    scale = min(size / source.height, size / source.width)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - resized.width) // 2
    pad_y = (size - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

    with torch.no_grad():
        raw = reference(torch.from_numpy(pixels.astype(np.float32)))
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    kept = non_max_suppression(raw, conf_thres=0.25, iou_thres=0.7, max_det=300)[0].numpy()

    assert len(kept), "the reference found nothing above the threshold to compare against"
    assert len(actual["scores"]) == len(kept), (
        f"engine returned {len(actual['scores'])} detections, reference {len(kept)}"
    )
    for index, row in enumerate(kept):
        x0, y0, x1, y1, score, label = row
        assert int(actual["classes"][index]) == int(label)
        assert abs(float(actual["scores"][index]) - float(score)) < 5e-3
        # The engine reports source-image pixels, so the reference box has to
        # come back out of the letterbox before they can be compared.
        expected = (
            (float(x0) - pad_x) / scale,
            (float(y0) - pad_y) / scale,
            (float(x1) - pad_x) / scale,
            (float(y1) - pad_y) / scale,
        )
        for axis, want in enumerate(expected):
            # The CLI writes the boxes as one flat array, four per detection.
            assert abs(float(actual["boxes"][index * 4 + axis]) - want) < 1.0
