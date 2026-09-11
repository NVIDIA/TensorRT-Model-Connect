# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and ultralytics reference proof for YOLOv10."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence


FAMILY = "yolov10"
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
                marks=pytest.mark.skip(reason="real YOLOv10 E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected YOLOv10 E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected YOLOv10 E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_YOLOV10_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_YOLOV10_MODEL_DIR")
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
    assert path.is_file(), f"selected YOLOv10 E2E image is missing: {path}"
    record_evidence("inputs", {"image": path})
    return path


def _reference_yaml(model_dir: Path) -> str:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, str) or model not in {"yolov10n.yaml", "yolov10s.yaml", "yolov10x.yaml"}:
        raise ValueError(f"unsupported YOLOv10 reference architecture: {model!r}")
    record_evidence("checkpoint", {"configuration": config})
    return model


def _reference_tensors(model_dir: Path) -> dict:
    from families.yolov10.checkpoint import Checkpoint

    checkpoint = Checkpoint.open(model_dir, framework="pt")
    return {name: reader.get_tensor(name) for name, reader in checkpoint.tensor_map.items()}


def _record_reference_boxes(rows, scale: float, pad_x: float, pad_y: float) -> None:
    if not evidence_enabled():
        return
    try:
        record_evidence(
            "reference",
            {
                "boxes": [
                    (float(row[axis]) - (pad_x if axis % 2 == 0 else pad_y)) / scale
                    for row in rows
                    for axis in range(4)
                ],
                "scores": [float(row[4]) for row in rows],
                "classes": [int(row[5]) for row in rows],
            },
        )
    except Exception as error:
        record_evidence("reference_preview", {"error": f"{type(error).__name__}: {error}"})


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "detection_format",
        {"boxes": "flattened XYXY in original-image pixels", "native_boxes": "clamped to image bounds by the runtime", "reference_boxes": "unclamped inverse-letterbox pixels used by the original comparison", "classes": "recorded numeric class IDs"},
    )
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]})
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
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
    with evidence_stage("native"):
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
        record_evidence("native_process", {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr})
        actual = json.loads(completed.stdout)

    record_evidence("native", actual)

    with evidence_stage("reference"):
        from ultralytics import YOLO

        # The reference must run the checkpoint under test. YOLO(<directory>)
        # quietly downloads its own weights instead, which would compare the engine
        # against a different model.
        reference = YOLO(_reference_yaml(model_dir), task="detect").model
        raw = _reference_tensors(model_dir)
        state = {key[len("model."):]: value for key, value in raw.items() if key.startswith("model.")}
        missing, unexpected = reference.load_state_dict(state, strict=False)
        assert not missing, f"reference is missing checkpoint tensors: {missing[:5]}"
        reference = reference.eval()

        import torch
        from PIL import Image

        source = Image.open(image).convert("RGB")
        letterboxed, scale, pad_x, pad_y = _letterbox(source, 640)
        with torch.no_grad():
            expected = reference(torch.from_numpy(letterboxed))
        expected = (expected[0] if isinstance(expected, (list, tuple)) else expected)[0].numpy()

    record_evidence("reference_raw", {"detections": expected, "coordinate_space": "letterboxed-image pixels", "scale": scale, "pad_x": pad_x, "pad_y": pad_y})
    threshold = 0.25
    record_evidence("thresholds", {"reference_score_min": threshold, "detection_count_equal": True, "class_ids_equal": True, "score_absolute_error_lt": 5e-3, "box_coordinate_error_pixels_lt": 1.0})
    kept = [row for row in expected if float(row[4]) >= threshold]
    _record_reference_boxes(kept, scale, pad_x, pad_y)
    with evidence_stage("compare"):
        assert kept, "the reference found nothing above the threshold to compare against"
        assert len(actual["scores"]) == len(kept), (
            f"engine returned {len(actual['scores'])} detections, reference {len(kept)}"
        )
        for index, row in enumerate(kept):
            assert int(actual["classes"][index]) == int(row[5])
            assert abs(float(actual["scores"][index]) - float(row[4])) < 5e-3
            for axis in range(4):
                # The engine reports original-image pixels, so the reference box has
                # to come back out of the letterbox before they can be compared.
                reference_pixel = (float(row[axis]) - (pad_x if axis % 2 == 0 else pad_y)) / scale
                assert abs(float(actual["boxes"][index * 4 + axis]) - reference_pixel) < 1.0


def _letterbox(image, size: int):
    """Ultralytics preprocessing: keep aspect, pad with 114, scale to [0, 1]."""
    from PIL import Image

    scale = min(size / image.height, size / image.width)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return array[None].copy(), scale, float(left), float(top)
