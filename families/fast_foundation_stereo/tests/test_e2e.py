# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for fast_foundation_stereo."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from families.fast_foundation_stereo.tests.accuracy import (
    aggregate_task_accuracy,
    scene_statistics,
)
from families.fast_foundation_stereo.tests.input_contract import ground_truth
from tensorrt_model_connect import BuildRequest, build

FAMILY = "fast_foundation_stereo"
TASKS = frozenset({"stereo_disparity"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
MIDDLEBURY_CASE = "fast-foundation-stereo-middlebury-q"
MIDDLEBURY_DATASET_ENV = "TRTMC_FAST_FOUNDATION_STEREO_MIDDLEBURY_Q_DATASET"


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


def _selection_filters(config) -> tuple[set[str], set[str]]:
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
    return model_filters, testcase_filters


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters, testcase_filters = _selection_filters(config)
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


def _middlebury_selected(config) -> bool:
    model_filters, testcase_filters = _selection_filters(config)
    if not model_filters and not testcase_filters:
        return False
    model_match = not model_filters or bool(
        {FAMILY, "fast-foundation-stereo", MIDDLEBURY_CASE} & model_filters
    )
    testcase_match = not testcase_filters or MIDDLEBURY_CASE in testcase_filters
    return model_match and testcase_match


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        if metafunc.function.__name__ == "test_middlebury_q_task_accuracy_e2e":
            metafunc.parametrize("case_name", [MIDDLEBURY_CASE], ids=[MIDDLEBURY_CASE])
            return
        names, enabled = _selected_cases(metafunc.config)
        names = [name for name in names if name != MIDDLEBURY_CASE]
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


def _model_dir(manifest: dict, tmp_path: Path) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")

    source = _required_path(
        os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"),
        "TRTMC_REFERENCE_SOURCE_DIR",
    )
    from huggingface_hub import snapshot_download

    checkpoint = (
        Path(
            snapshot_download(
                repo_id=manifest["hf_id"],
                revision=manifest.get("hf_revision"),
                local_files_only=True,
                allow_patterns=["model_best_bp2_serialize.pth"],
            )
        )
        / "model_best_bp2_serialize.pth"
    )
    assert checkpoint.is_file()

    prepared = tmp_path / "fast-foundation-stereo-model"
    prepared.mkdir()
    for path in source.iterdir():
        (prepared / path.name).symlink_to(path, target_is_directory=path.is_dir())
    weights = prepared / "weights/23-36-37"
    weights.mkdir(parents=True)
    (weights / "model_best_bp2_serialize.pth").symlink_to(checkpoint)
    return prepared


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


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert denominator > 0.0
    return float(np.dot(a, b) / denominator)


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
):
    manifest["task"]
    inputs = case["inputs"]
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "disparity",
        "--left",
        str(_asset(inputs["left_image"])),
        "--right",
        str(_asset(inputs["right_image"])),
    )


def _load_official_reference(model_dir: Path):
    import sys

    import torch

    from families.fast_foundation_stereo.prepare_model import configure_official_model_args

    previous_cwd = Path.cwd()
    try:
        os.chdir(model_dir)
        sys.path.insert(0, str(model_dir))
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        model = torch.load(
            model_dir / "weights/23-36-37/model_best_bp2_serialize.pth",
            map_location="cpu",
            weights_only=False,
        )
    finally:
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(model_dir):
            sys.path.pop(0)
    configure_official_model_args(model, max_disparity=192, valid_iters=8)
    return model.to("cuda").eval(), InputPadder, AMP_DTYPE


def _run_official_reference(loaded_reference, case: dict) -> dict:
    import torch
    from PIL import Image

    model, input_padder, amp_dtype = loaded_reference
    inputs = case["inputs"]
    images = [
        Image.open(_asset(inputs["left_image"])).convert("RGB"),
        Image.open(_asset(inputs["right_image"])).convert("RGB"),
    ]
    tensors = [
        torch.as_tensor(np.asarray(image), device="cuda").float()[None].permute(0, 3, 1, 2)
        for image in images
    ]
    padder = input_padder(tensors[0].shape, divis_by=32, force_square=False)
    left, right = padder.pad(*tensors)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
        disparity = model.forward(
            left, right, iters=8, test_mode=True, optimize_build_volume="pytorch1"
        )
    value = padder.unpad(disparity.float()).cpu().numpy()
    return {"disparity": np.clip(value, 0, None)}


def _disparity(payload: dict) -> np.ndarray:
    values = np.asarray(payload["disparity"], dtype=np.float32)
    assert values.size == 700 * 700
    return values.reshape(700, 700)


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    candidate = _disparity(actual)
    reference = _disparity(expected)
    assert np.isfinite(candidate).all()
    assert np.all(candidate >= 0.0)
    absolute_error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    assert _cosine(candidate, reference) >= float(thresholds["global_cosine"])
    assert float(absolute_error.mean()) <= float(thresholds["mean_abs_error"])
    assert float(np.mean(absolute_error > 2.0)) <= float(thresholds["bad_2px_fraction"])


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest, tmp_path)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, manifest, case)
    expected = _run_official_reference(_load_official_reference(model_dir), case)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))


def test_middlebury_q_task_accuracy_e2e(case_name: str, request, tmp_path: Path) -> None:
    assert case_name == MIDDLEBURY_CASE
    if not _middlebury_selected(request.config):
        pytest.skip("Middlebury-Q E2E requires an explicit matching E2E selector")
    configured = os.environ.get(MIDDLEBURY_DATASET_ENV)
    if not configured:
        output = tmp_path / "middlebury-q"
        subprocess.run(
            [
                sys.executable,
                str(TEST_ROOT / "prepare_middlebury_q.py"),
                "--output",
                str(output),
            ],
            check=True,
        )
        configured = str(output / "dataset.json")
    dataset_path = _required_path(configured, MIDDLEBURY_DATASET_ENV)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = dataset.get("requests")
    assert isinstance(requests, list) and len(requests) == 15
    scenes = [str(item["inputs"]["scene"]) for item in requests]
    assert len(set(scenes)) == 15

    _, manifest, _ = CASES["fast-foundation-stereo"]
    model_dir = _model_dir(manifest, tmp_path)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    reference = _load_official_reference(model_dir)
    thresholds = _thresholds(case_name)
    statistics = []
    for item in requests:
        case = {"name": str(item["sample_id"]), "inputs": dict(item["inputs"])}
        actual = _native(binary, runtime_root, bundle, manifest, case)
        expected = _run_official_reference(reference, case)
        _assert_parity(actual, expected, manifest, case, thresholds)
        truth, valid = ground_truth(case["inputs"])
        statistics.append(scene_statistics(_disparity(actual), _disparity(expected), truth, valid))

    aggregate = aggregate_task_accuracy(
        statistics,
        epe_allowance_px=float(thresholds["candidate_nonocc_epe_max_reference_plus_px"]),
        bp2_allowance_fraction=float(
            thresholds["candidate_nonocc_bp2_max_reference_plus_fraction"]
        ),
    )
    assert aggregate["candidate_nonocc_epe_passed"] is True
    assert aggregate["candidate_nonocc_bp2_passed"] is True
