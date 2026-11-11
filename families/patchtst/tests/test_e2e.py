# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for patchtst."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence
import json
import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "patchtst"
TASKS = frozenset(
    {"series_to_point_forecast", "series_to_regression_values", "series_to_regression_distribution"}
)
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
            sys.executable,
            str(TEST_ROOT / "rank_exec.py"),
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
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence(
        "native",
        {
            "stdout": getattr(completed, "stdout", None),
            "stderr": getattr(completed, "stderr", None),
        },
    )
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert len(payloads) == int(manifest["tensor_parallel_size"]), (
        f"native {command} must return one result per rank"
    )
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def _to_reference_layout(payload: dict, manifest: dict, config: dict) -> dict:
    """Validate complete Task metadata, then restore the official model's batch layout."""
    task = manifest["task"]
    assert payload["task"] == task
    if task == "series_to_point_forecast":
        horizon = int(config["prediction_length"])
        channels = int(config["num_input_channels"])
        assert payload["kind"] == "point" and payload["shape"] == [horizon, channels]
        assert payload["axes"] == ["horizon", "channel"]
        assert payload["horizon_steps"] == list(range(1, horizon + 1))
        assert payload["channel_names"] == payload["channel_units"] == []
        data = np.asarray(payload["values"])
        assert data.ndim == 1 and data.size == horizon * channels
        data = data.reshape(1, horizon, channels)
    else:
        targets = int(config.get("num_targets", 1))
        assert payload["target_count"] == targets
        assert payload["target_names"] == payload["target_units"] == []
        if task == "series_to_regression_values":
            assert payload["kind"] == "regression_values"
            data = np.asarray(payload["values"])
            assert data.shape == (targets,)
            data = data.reshape(1, targets)
        else:
            assert payload["kind"] == "regression_distribution"
            distribution = str(config["distribution_output"]).lower()
            assert payload["distribution"] == distribution
            names = {
                "normal": ("location", "scale"),
                "student_t": ("degrees_of_freedom", "location", "scale"),
                "negative_binomial": ("total_count", "logits"),
            }[distribution]
            assert set(payload["parameters"]) == set(names)
            vectors = [np.asarray(payload["parameters"][name]) for name in names]
            assert all(value.shape == (targets,) for value in vectors)
            data = np.stack(vectors, axis=-1)
            # DistributionOutput squeezes the final target axis when num_targets=1.
            data = data.reshape((1, len(names)) if targets == 1 else (1, targets, len(names)))
    return {"task_output": payload, "values": data, "shape": list(data.shape)}


def _assert_sdk_consumers(
    runtime_root, bundle, model_dir, manifest, case, tmp_path, expected, thresholds
):
    build_root = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    config = json.loads((model_dir / "config.json").read_text())
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(runtime_root) + ":" + env.get("LD_LIBRARY_PATH", "")
    for language in ("c", "cpp"):
        executable = build_root / f"test_patchtst_sdk_{language}"
        assert executable.is_file(), f"build the {language} SDK consumer first"
        invocation = [
            str(executable),
            str(bundle),
            str(runtime_root),
            str(tmp_path / "values.f32"),
            str(tmp_path / "mask.f32"),
            manifest["task"],
            str(config["num_input_channels"]),
        ]
        if int(manifest["tensor_parallel_size"]) > 1:
            mpirun = shutil.which("mpirun")
            assert mpirun, "multi-GPU SDK tests require mpirun"
            invocation = [
                mpirun,
                "--tag-output",
                "-x",
                "LD_LIBRARY_PATH",
                "-np",
                str(manifest["tensor_parallel_size"]),
                sys.executable,
                str(TEST_ROOT / "rank_exec.py"),
                *invocation,
            ]
        completed = subprocess.run(
            invocation,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=int(case.get("runtime_timeout_s", 3600)),
        )
        record_evidence("commands", {"argv": invocation})
        payloads = []
        for line in completed.stdout.splitlines():
            offset = line.find("{")
            if offset >= 0:
                payloads.append(json.loads(line[offset:]))
        assert len(payloads) == int(manifest["tensor_parallel_size"])
        assert all(payload == payloads[0] for payload in payloads)
        actual = _to_reference_layout(payloads[0], manifest, config)
        record_evidence("native", {"sdk_language": language, **actual})
        _assert_parity(actual, expected, manifest, case, thresholds)


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
    mask = np.asarray(inputs.get("observed_mask", np.ones_like(values)), dtype=np.float32)
    input_path = tmp_path / "values.f32"
    mask_path = tmp_path / "mask.f32"
    values.tofile(input_path)
    record_evidence("inputs", {"raw_file": input_path})
    mask.tofile(mask_path)
    record_evidence("inputs", {"raw_file": mask_path})
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
        str(inputs.get("frequency", 0)),
    )
    payload = _to_reference_layout(
        payload, manifest, json.loads((model_dir / "config.json").read_text())
    )
    payload["input_values"] = values.tolist()
    payload["input_mask"] = mask.tolist()
    return payload


@cache
def _reference_model(model_dir: Path):
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_dir)
    architecture = " ".join(getattr(config, "architectures", ()) or ()).lower()
    if "regression" in architecture:
        model_class = transformers.PatchTSTForRegression
        output_name = "regression_outputs"
    else:
        model_class = transformers.PatchTSTForPrediction
        output_name = "prediction_outputs"
    model = model_class.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    channels = int(getattr(config, "num_input_channels", 1))
    context = int(getattr(config, "context_length", 1))
    return model, output_name, channels, context


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch

    inputs = case["inputs"]
    model, output_name, channels, context = _reference_model(model_dir)
    raw = list(inputs["past_values"])
    values = np.zeros(context * channels, dtype=np.float32)
    values[-min(len(raw), values.size) :] = raw[-values.size :]
    mask = np.zeros_like(values)
    observed = list(inputs.get("observed_mask", [1] * len(raw)))
    assert len(observed) == len(raw) and all(value in (0, 1) for value in observed)
    mask[-min(len(raw), values.size) :] = observed[-values.size :]
    kwargs = {
        "past_values": torch.tensor(values).reshape(1, context, channels),
        "return_dict": True,
    }
    kwargs["past_observed_mask"] = torch.tensor(mask).reshape(1, context, channels).gt(0.5)
    with torch.no_grad():
        output = getattr(model(**kwargs), output_name)
    if isinstance(output, tuple):
        output = torch.stack(output, dim=-1)
    return {"values": output.float().cpu().numpy()}


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    left = np.asarray(actual["values"])
    shape = tuple(int(value) for value in actual["shape"])
    assert shape and all(value > 0 for value in shape)
    assert left.size == int(np.prod(shape))
    left = left.reshape(shape)
    right = np.asarray(expected["values"])
    assert left.shape == right.shape and left.size > 0
    assert np.isfinite(left).all() and np.isfinite(right).all()
    if "regression" in str(case["name"]):
        assert right.ndim in (1, 2, 3)
        assert np.array_equal(np.sign(left), np.sign(right))
    else:
        assert right.ndim in (2, 3)
    assert _relative_l2(left, right) <= float(thresholds["relative_l2"])
    assert np.max(np.abs(left.reshape(-1) - right.reshape(-1))) <= float(
        thresholds["max_pointwise_error"]
    )


def test_regression_contract_requires_shape_and_sign_match() -> None:
    manifest = {"task": "series_to_regression_distribution"}
    case = {"name": "patchtst-regression"}
    thresholds = {"relative_l2": 1.0, "max_pointwise_error": 1.0}
    expected = {"values": np.asarray([[0.1, -0.1]], dtype=np.float32)}
    _assert_parity({**expected, "shape": [1, 2]}, expected, manifest, case, thresholds)
    with pytest.raises(AssertionError):
        _assert_parity(
            {"values": np.asarray([[-0.1, 0.1]], dtype=np.float32), "shape": [1, 2]},
            expected,
            manifest,
            case,
            thresholds,
        )
    with pytest.raises(AssertionError):
        _assert_parity(
            {"values": np.asarray([0.1, -0.1], dtype=np.float32), "shape": [2]},
            expected,
            manifest,
            case,
            thresholds,
        )
    with pytest.raises(AssertionError):
        _assert_parity(
            {"values": np.asarray([0.1, -0.1], dtype=np.float32), "shape": [2]},
            {"values": np.asarray([0.1, -0.1], dtype=np.float32)},
            manifest,
            {"name": "patchtst-point-forecast"},
            thresholds,
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": CASES[case_name][-1]})
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    from families.patchtst.tests.etth1 import CASES as ETTH1_CASES

    if case_name in ETTH1_CASES:
        from families.patchtst.tests.etth1 import gates, windows

        for index, inputs in enumerate(windows(case_name)):
            window_root = tmp_path / f"etth1-{index:02}"
            window_root.mkdir()
            window_case = {**case, "inputs": inputs}
            record_evidence("inputs", {"window_index": index, "case": window_case})
            with evidence_stage("native"):
                actual = _native(
                    binary, runtime_root, bundle, model_dir, manifest, window_case, window_root
                )
            record_evidence("native", actual)
            with evidence_stage("reference"):
                expected = _official_reference(model_dir, manifest, window_case, window_root)
            record_evidence("reference", expected)
            with evidence_stage("compare"):
                _assert_parity(
                    actual,
                    expected,
                    manifest,
                    window_case,
                    record_evidence("thresholds", gates(case_name)),
                )
                if index == 0:
                    _assert_sdk_consumers(
                        runtime_root,
                        bundle,
                        model_dir,
                        manifest,
                        window_case,
                        window_root,
                        expected,
                        gates(case_name),
                    )
        masked_inputs = dict(windows(case_name)[0])
        masked_inputs["observed_mask"] = [
            int(index % 11 != 0) for index in range(len(masked_inputs["past_values"]))
        ]
        masked_case = {**case, "inputs": masked_inputs}
        masked_root = tmp_path / "masked"
        masked_root.mkdir()
        actual = _native(
            binary, runtime_root, bundle, model_dir, manifest, masked_case, masked_root
        )
        expected = _official_reference(model_dir, manifest, masked_case, masked_root)
        record_evidence("native", actual)
        record_evidence("reference", expected)
        _assert_parity(actual, expected, manifest, masked_case, gates(case_name))
        _assert_sdk_consumers(
            runtime_root,
            bundle,
            model_dir,
            manifest,
            masked_case,
            masked_root,
            expected,
            gates(case_name),
        )
        if manifest["task"] == "series_to_point_forecast":
            _check_regression_values(binary, runtime_root, tmp_path)
        return
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    record_evidence("native", actual)
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, manifest, case, tmp_path)
    record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_parity(
            actual, expected, manifest, case, record_evidence("thresholds", _thresholds(case_name))
        )
        _assert_sdk_consumers(
            runtime_root,
            bundle,
            model_dir,
            manifest,
            case,
            tmp_path,
            expected,
            _thresholds(case_name),
        )


def _check_regression_values(binary: Path, runtime_root: Path, tmp_path: Path) -> None:
    """Exercise the already-supported deterministic regression head using a fixed-seed official model."""
    import torch
    from transformers import PatchTSTConfig, PatchTSTForRegression

    model_root = tmp_path / "generated-regression-values"
    config = PatchTSTConfig(
        context_length=32,
        num_input_channels=2,
        num_targets=3,
        patch_length=8,
        patch_stride=8,
        d_model=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        ffn_dim=16,
        norm_type="layernorm",
        loss="mse",
        dropout=0,
        attention_dropout=0,
        ff_dropout=0,
        head_dropout=0,
        positional_dropout=0,
        path_dropout=0,
    )
    torch.manual_seed(1495)
    model = PatchTSTForRegression(config).eval()
    with torch.no_grad():
        model.head.projection.bias.copy_(torch.tensor([1.0, -1.0, 0.25]))
    model.save_pretrained(model_root)
    for precision in ("fp32", "fp16"):
        manifest = {
            "family": FAMILY,
            "task": "series_to_regression_values",
            "precision": precision,
            "tensor_parallel_size": 1,
        }
        case = {
            "name": "patchtst-regression-values",
            "inputs": {"past_values": np.sin(np.arange(64, dtype=np.float32) / 7).tolist()},
        }
        root = tmp_path / ("regression-values-" + precision)
        root.mkdir()
        bundle = root / "regression-values.bundle"
        _build(model_root, bundle, manifest)
        actual = _native(binary, runtime_root, bundle, model_root, manifest, case, root)
        expected = _official_reference(model_root, manifest, case, root)
        thresholds = {
            "relative_l2": 1e-5 if precision == "fp32" else 1e-3,
            "max_pointwise_error": 1e-5 if precision == "fp32" else 1e-3,
        }
        record_evidence(
            "inputs",
            {
                "generated_model_config": config.to_dict(),
                "seed": 1495,
                "precision": precision,
                "case": case,
            },
        )
        record_evidence("native", actual)
        record_evidence("reference", expected)
        record_evidence("thresholds", thresholds)
        _assert_parity(actual, expected, manifest, case, thresholds)
        _assert_sdk_consumers(
            runtime_root, bundle, model_root, manifest, case, root, expected, thresholds
        )
