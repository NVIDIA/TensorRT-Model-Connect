# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for timesfm."""

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

FAMILY = "timesfm"
TASKS = frozenset({"time_series_forecast"})
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


def _relative_l2(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


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
    inputs = case["inputs"]
    values = np.asarray(inputs["past_values"], dtype=np.float32)
    mask = np.ones_like(values)
    input_path = tmp_path / "values.f32"
    mask_path = tmp_path / "mask.f32"
    values.tofile(input_path)
    mask.tofile(mask_path)
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "forecast",
        "--input",
        str(input_path),
        "--mask",
        str(mask_path),
        "--frequency",
        str(inputs["frequency"]),
    )
    payload["input_values"] = values.tolist()
    payload["input_mask"] = mask.tolist()
    return payload


@cache
def _reference_model(model_dir: Path):
    import torch
    import transformers

    return transformers.TimesFmModelForPrediction.from_pretrained(
        model_dir, torch_dtype=torch.float32
    ).eval()


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch

    inputs = case["inputs"]
    model = _reference_model(model_dir)
    context_length = int(model.config.context_length)
    raw = list(inputs["past_values"])
    series = [0.0] * context_length
    series[-min(len(raw), context_length) :] = raw[-context_length:]
    padding = [1] * context_length
    padding[-min(len(raw), context_length) :] = [0] * min(len(raw), context_length)
    freq = int(inputs["frequency"])
    with torch.no_grad():
        decoded = model.decoder(
            past_values=torch.tensor(series).reshape(1, context_length),
            past_values_padding=torch.tensor(padding, dtype=torch.int32).reshape(1, context_length),
            freq=torch.tensor([[freq]], dtype=torch.long),
            output_attentions=False,
            output_hidden_states=False,
        )
        output = model._postprocess_output(decoded.last_hidden_state, (decoded.loc, decoded.scale))
    return {"values": output[:, -1, : model.config.horizon_length, 0].float().cpu().numpy()}


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    left = np.asarray(actual["values"])
    shape = tuple(int(value) for value in actual["shape"])
    assert shape and all(value > 0 for value in shape)
    assert left.size == int(np.prod(shape))
    left = left.reshape(shape)
    right = np.asarray(expected["values"])
    assert left.shape == right.shape and left.size > 0
    assert right.ndim in (2, 3)
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert _relative_l2(left, right) <= float(thresholds["relative_l2"])
    assert np.max(np.abs(left.reshape(-1) - right.reshape(-1))) <= float(
        thresholds["max_pointwise_error"]
    )


def test_forecast_contract_requires_matching_shape() -> None:
    expected = {"values": np.asarray([0.1, 0.2], dtype=np.float32)}
    with pytest.raises(AssertionError):
        _assert_parity(
            {"values": np.asarray([0.1, 0.2], dtype=np.float32), "shape": [2]},
            expected,
            {"task": "time_series_forecast"},
            {},
            {"relative_l2": 1.0, "max_pointwise_error": 1.0},
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    from families.timesfm.tests.etth1 import CASE as ETTH1_CASE

    if case_name == ETTH1_CASE:
        from families.timesfm.tests.etth1 import GATES, windows

        for index, inputs in enumerate(windows()):
            window_root = tmp_path / f"etth1-{index:02}"
            window_root.mkdir()
            window_case = {**case, "inputs": inputs}
            actual = _native(
                binary, runtime_root, bundle, model_dir, manifest, window_case, window_root
            )
            expected = _official_reference(model_dir, manifest, window_case, window_root)
            _assert_parity(actual, expected, manifest, window_case, GATES)
        return
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
