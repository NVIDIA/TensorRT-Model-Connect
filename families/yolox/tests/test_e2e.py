# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and official Megvii reference proof for YOLOX."""

from __future__ import annotations

import json
import struct
import sys
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "yolox"
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
                marks=pytest.mark.skip(reason="real YOLOX E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected YOLOX E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected YOLOX E2E {label} does not exist: {path}"
    return path


def _reference_root(manifest):
    root = _required_path(
        os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"), "TRTMC_REFERENCE_SOURCE_DIR"
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == manifest["reference_revision"], (
        "reference checkout must match the pinned revision"
    )
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True
    )
    assert not dirty.strip(), "the official reference must be unmodified"
    metadata = json.loads((TEST_ROOT / "reference-source.json").read_text())
    assert metadata["revision"] == manifest["reference_revision"]
    sys.path.insert(0, str(root))
    return root


def _native_pixels(image, binary, tmp_path):
    height, width = image.shape[:2]
    rgb = np.ascontiguousarray(image[..., ::-1], dtype=np.float32) / 255.0
    input_path, output_path = tmp_path / "rgb.f32", tmp_path / "preprocessed.f32"
    rgb.tofile(input_path)
    subprocess.run(
        [str(binary), str(input_path), str(height), str(width), str(output_path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return np.fromfile(output_path, dtype=np.float32).reshape(3, 640, 640)


def _engine_outputs(bundle, pixels):
    import tensorrt as trt
    import torch

    # Read the public bundle container to replay the exact engine built above.
    with bundle.open("rb") as stream:
        assert stream.read(8) == b"BUNDLE\x01\x00"
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        section = header["sections"]["engine.plan"]
        stream.seek(16 + size + section["offset"])
        plan = stream.read(section["length"])
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        context = engine.create_execution_context()
        buffers = {"pixel_values": torch.from_numpy(pixels[None]).cuda().contiguous()}
        for name, dtype in (
            ("boxes", torch.float32),
            ("scores", torch.float32),
            ("classes", torch.int32),
        ):
            buffers[name] = torch.empty(
                tuple(engine.get_tensor_shape(name)), device="cuda", dtype=dtype
            )
        for name, tensor in buffers.items():
            assert context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return {
            name: tensor.cpu().numpy() for name, tensor in buffers.items() if name != "pixel_values"
        }


def _compare_detections(actual, reference, ratio):
    boxes = np.asarray(actual["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(actual["scores"], dtype=np.float32)
    classes = np.asarray(actual["classes"], dtype=np.int32)
    expected = reference.cpu().numpy()
    assert len(expected) > 0, "fixture must exercise real detections"
    assert len(boxes) == len(expected), (actual, expected.tolist())
    np.testing.assert_array_equal(classes, expected[:, 6].astype(np.int32))
    np.testing.assert_allclose(scores, expected[:, 4] * expected[:, 5], rtol=0, atol=0.01)
    target = expected[:, :4] / ratio
    # Two pixels in network coordinates, independent of original image size.
    np.testing.assert_allclose(boxes * ratio, target * ratio, rtol=0, atol=2.0)
    intersection = np.maximum(
        0, np.minimum(boxes[:, 2:], target[:, 2:]) - np.maximum(boxes[:, :2], target[:, :2])
    ).prod(axis=1)
    union = (
        (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        + (target[:, 2:] - target[:, :2]).prod(axis=1)
        - intersection
    )
    assert np.all(intersection / union >= 0.98), intersection / union


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    reference_root = _reference_root(manifest)
    import cv2
    import torch
    import yolox
    from yolox.exp import get_exp
    from yolox.data.data_augment import preproc
    from yolox.utils import postprocess

    assert Path(yolox.__file__).resolve().is_relative_to(reference_root.resolve())
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", str(runtime_root)))
    seam = native_build / "families/yolox/test_yolox_image_preprocess"
    assert seam.is_file(), f"selected YOLOX E2E requires the native seam test: {seam}"
    model_dir = _required_path(os.environ.get("TRTMC_YOLOX_MODEL_DIR"), "TRTMC_YOLOX_MODEL_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_yolox.so").is_file()
    checkpoint_path = model_dir / manifest["external_files"][0]["path"]
    assert torch.cuda.is_available(), "selected YOLOX E2E requires a CUDA GPU"
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
        )
    )
    reference = (
        get_exp(str(reference_root / "exps/default/yolox_s.py"), None).get_model().eval().cuda()
    )
    reference.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"], strict=True
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = cv2.imread(str(TEST_ROOT / case["test_image"]))
    assert original is not None
    height, width = original.shape[:2]
    portrait = np.full((width * 2, width, 3), 114, dtype=np.uint8)
    portrait[:height] = original
    for label, image in (("landscape", original), ("portrait", portrait)):
        official_pixels, ratio = preproc(image, (640, 640))
        native_pixels = _native_pixels(image, seam, tmp_path)
        # OpenCV's byte resize uses fixed-point rounding. No channel, padding,
        # scale or normalization error can fit within this one-byte bound.
        np.testing.assert_allclose(native_pixels, official_pixels, rtol=0, atol=1.0)
        actual_raw = _engine_outputs(bundle, native_pixels)
        with torch.no_grad():
            same_input = reference(torch.from_numpy(native_pixels[None]).cuda())[0].cpu().numpy()
            official = reference(torch.from_numpy(official_pixels[None]).cuda())
            expected = postprocess(
                official.clone(), 80, conf_thre=0.25, nms_thre=0.45, class_agnostic=False
            )[0]
        expected_scores = same_input[:, 4] * same_input[:, 5:].max(axis=1)
        expected_classes = same_input[:, 5:].argmax(axis=1)
        expected_boxes = np.concatenate(
            (
                same_input[:, :2] - same_input[:, 2:4] * 0.5,
                same_input[:, :2] + same_input[:, 2:4] * 0.5,
            ),
            axis=1,
        )
        assert actual_raw["boxes"].shape == (8400, 4)
        assert all(np.isfinite(value).all() for value in actual_raw.values())
        np.testing.assert_allclose(actual_raw["scores"], expected_scores, rtol=0, atol=0.01)
        foreground = (expected_scores >= 0.1) | (actual_raw["scores"] >= 0.1)
        assert foreground.any()
        np.testing.assert_array_equal(
            actual_raw["classes"][foreground], expected_classes[foreground]
        )
        np.testing.assert_allclose(
            actual_raw["boxes"][foreground], expected_boxes[foreground], rtol=0, atol=2.0
        )
        image_path = tmp_path / f"{label}.png"
        assert cv2.imwrite(str(image_path), image)
        completed = subprocess.run(
            [
                str(binary),
                "detect",
                str(bundle),
                "--runtime-root",
                str(runtime_root),
                "--image",
                str(image_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        actual = json.loads(completed.stdout)
        assert expected is not None
        _compare_detections(actual, expected, ratio)
        print(f"{case_name} {label}: {len(expected)} detections match the official reference")
