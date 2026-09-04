# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for sam2."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "sam2"
TASKS = frozenset({"video_segmentation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


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


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
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
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
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


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _case_text(case: dict) -> str:
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    del binary, model_dir
    from PIL import Image

    frame_paths = []
    for index in range(int(manifest["video_num_frames"])):
        height = int(manifest["image_height"])
        width = int(manifest["image_width"])
        rows = np.arange(height, dtype=np.uint16)[:, None]
        columns = np.arange(width, dtype=np.uint16)[None, :]
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = columns + 17 * index & 255
        frame[..., 1] = rows + 31 * index & 255
        frame[..., 2] = (columns // 8 ^ rows // 8 ^ 53 * index) & 255
        top = 300 + 20 * index
        left = 250 + 30 * index
        frame[top : top + 500, left : left + 400] = (235, 45 + 20 * index, 25)
        path = tmp_path / f"frame-{index:03}.png"
        Image.fromarray(frame).save(path)
        frame_paths.append(path)
    masks_path = tmp_path / "native-masks.u8"
    receipt = _operational_receipt(bundle, runtime_root, masks_path)
    masks = np.fromfile(masks_path, dtype=np.uint8)
    expected = (
        int(manifest["video_num_frames"])
        * int(manifest["image_height"])
        * int(manifest["image_width"])
    )
    assert masks.size == expected
    return {
        "bundle": str(bundle),
        "input_frames": [str(path) for path in frame_paths],
        "masks": masks,
        "operational_receipt": receipt,
    }


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    from PIL import Image
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    frame_paths = sorted(tmp_path.glob("frame-*.png"))
    assert frame_paths
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    processor = Sam2VideoProcessor.from_pretrained(model_dir)
    model = Sam2VideoModel.from_pretrained(model_dir).to("cuda").eval()
    encoded = processor(videos=frames, text=_case_text(case), return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**encoded)
    return {"masks": outputs.pred_masks.float().cpu().numpy()}


@cache
def _operational_probe_binary() -> Path:
    build_value = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert build_value, "selected SAM2 L0 E2E requires TRTMC_NATIVE_BUILD_DIR"
    build_dir = Path(build_value)
    assert build_dir.is_dir()
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            "8",
            "--target",
            "sam2_operational_probe",
        ],
        check=True,
        timeout=600,
    )
    probe = build_dir / "sam2_operational_probe"
    assert probe.is_file()
    return probe


def _operational_receipt(bundle: Path, runtime_root: Path, masks_path: Path) -> dict:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [str(_operational_probe_binary()), str(bundle), str(runtime_root), str(masks_path)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    return json.loads(completed.stdout)


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    task = manifest["task"]
    left = np.asarray(actual["masks"]).astype(bool).reshape(-1)
    right = np.asarray(expected["masks"]).astype(bool).reshape(-1)
    assert left.shape == right.shape
    union = np.logical_or(left, right).sum()
    iou = np.logical_and(left, right).sum() / max(int(union), 1)
    threshold_name = "mean_mask_iou" if task == "video_segmentation" else "iou_per_prompt"
    assert iou >= float(thresholds[threshold_name])
    from families.sam2.tests.operational_oracle import (
        assert_bundle_contract,
        assert_operational_receipt,
    )

    assert_bundle_contract(Path(actual["bundle"]))
    assert_operational_receipt(actual["operational_receipt"])
    return


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
