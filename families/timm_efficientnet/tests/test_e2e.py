# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for timm_efficientnet."""

from __future__ import annotations

from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence
import json
import os
import subprocess
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "timm_efficientnet"
TASKS = frozenset({"classification"})
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


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


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
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    record_evidence(
        "native_process",
        {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
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


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    record_evidence("inputs", {"asset": path})
    return path


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
):
    return _run_json(
        binary,
        runtime_root,
        bundle,
        case,
        "classify",
        "--image",
        str(_asset(case["test_image"])),
    )


def _official_reference(model_dir: Path, case: dict):
    import torch
    import timm
    from PIL import Image
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_model_data_config

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = timm.create_model(
        config["architecture"],
        pretrained=False,
        num_classes=int(config["num_classes"]),
    )
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.pretrained_cfg = config["pretrained_cfg"]
    model = model.to("cuda").eval()
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    pixels = transform(image).unsqueeze(0).to("cuda")
    with torch.no_grad():
        logits = model(pixels).float().cpu().numpy().reshape(-1)
    order = np.argsort(logits)
    return {
        "top_class": int(order[-1]),
        "second_class": int(order[-2]),
        "top1_margin": float(logits[order[-1]] - logits[order[-2]]),
        "top_score": float(logits[order[-1]]),
    }


def _assert_parity(actual, expected, thresholds: dict) -> None:
    if int(actual["top_class"]) == int(expected["top_class"]):
        return
    margin = thresholds.get("top1_margin_atol")
    assert margin is not None
    assert float(expected["top1_margin"]) <= float(margin)
    assert int(actual["top_class"]) == int(expected["second_class"])


def _record_exact_class_match(actual, expected, tmp_path: Path) -> None:
    """Observe the original successful early return without adding another gate."""
    if not evidence_enabled():
        return
    try:
        native_class = int(actual["top_class"])
        reference_class = int(expected["top_class"])
        if native_class != reference_class:
            return
        native = tmp_path / "classification-native.json"
        reference = tmp_path / "classification-reference.json"
        native.write_text(json.dumps({"top_class": native_class}) + "\n", encoding="utf-8")
        reference.write_text(json.dumps({"top_class": reference_class}) + "\n", encoding="utf-8")
        record_evidence(
            "reference_comparison",
            {
                "label": "Exact top class match",
                "scope": "independent_reference",
                "enforced": True,
                "native": native,
                "reference": reference,
                "checks": [
                    {
                        "name": "top_class",
                        "label": "Predicted class ID",
                        "scope": "independent_reference",
                        "actual": native_class,
                        "operator": "==",
                        "expected": reference_class,
                        "passed": native_class == reference_class,
                    }
                ],
            },
        )
    except Exception as error:
        record_evidence("comparison_preview", {"error": f"{type(error).__name__}: {error}"})


def test_top1_margin_contract_accepts_only_the_reference_runner_up() -> None:
    thresholds = {"top1_margin_atol": 0.12}
    _assert_parity(
        {"top_class": 2},
        {"top_class": 1, "second_class": 2, "top1_margin": 0.1},
        thresholds,
    )
    with pytest.raises(AssertionError):
        _assert_parity(
            {"top_class": 2},
            {"top_class": 1, "second_class": 2, "top1_margin": 0.2},
            thresholds,
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": CASES[case_name][-1]})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")})
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, case)
    record_evidence("native", actual)
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, case)
    record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_parity(actual, expected, record_evidence("thresholds", _thresholds(case_name)))
    _record_exact_class_match(actual, expected, tmp_path)


@pytest.fixture
def classification_reporting_run(tmp_path: Path, monkeypatch):
    from tools import e2e_evidence

    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence", family=FAMILY, case="classification-report-control",
        source_revision="a" * 40, roots=(tmp_path,),
    )
    original_parity = _assert_parity
    original_write = Path.write_text
    events = []
    errors = []

    def run(actual, expected, thresholds, *, fail_file=None, capture=True):
        monkeypatch.setitem(globals(), "_model_dir", lambda manifest: tmp_path)
        monkeypatch.setitem(globals(), "_runtime", lambda: (tmp_path / "trtmc", tmp_path))
        monkeypatch.setitem(globals(), "_build", lambda *args: events.append("build"))
        monkeypatch.setitem(globals(), "_native", lambda *args: actual)
        monkeypatch.setitem(globals(), "_official_reference", lambda *args: expected)
        monkeypatch.setitem(globals(), "_thresholds", lambda name: thresholds)

        def parity(*args):
            events.append("parity")
            try:
                result = original_parity(*args)
            except AssertionError as error:
                errors.append(error)
                raise
            events.append("parity-passed")
            return result

        def write(path, *args, **kwargs):
            if path.name in {"classification-native.json", "classification-reference.json"}:
                events.append(path.name)
                if path.name == fail_file:
                    raise OSError("report file unavailable")
            return original_write(path, *args, **kwargs)

        monkeypatch.setitem(globals(), "_assert_parity", parity)
        monkeypatch.setattr(Path, "write_text", write)
        token = e2e_evidence._ACTIVE.set(recorder if capture else None)
        caught = None
        try:
            test_official_checkpoint_e2e("efficientnet-b0-ra-in1k", tmp_path)
        except AssertionError as error:
            caught = error
        finally:
            e2e_evidence._ACTIVE.reset(token)
        if capture:
            recorder.finish("failed" if caught else "passed", failure=str(caught or ""))
        return recorder, events, errors, caught

    return run


@pytest.mark.parametrize("native_class,reference_class", [(0, 0), (656, 656), ("0", np.int64(0))])
def test_classification_reporting_exact_match_has_real_distinct_artifacts(
    classification_reporting_run, native_class, reference_class
):
    import copy
    from tools.e2e_report import _assessment, render_case

    actual = {"top_class": native_class, "extra": [1, 2]}
    expected = {"top_class": reference_class, "second_class": 817, "top1_margin": 0.1}
    original = copy.deepcopy((actual, expected))
    recorder, events, errors, caught = classification_reporting_run(actual, expected, {"top1_margin_atol": 0.12})
    assert caught is None and errors == [] and (actual, expected) == original
    assert events == ["build", "parity", "parity-passed", "classification-native.json", "classification-reference.json"]
    comparison = recorder.data["reference_comparison"]
    native, reference = comparison["native"], comparison["reference"]
    assert native["artifact"] != reference["artifact"]
    assert native["size_bytes"] > 0 and reference["size_bytes"] > 0
    files = [recorder.directory / value["artifact"] for value in (native, reference)]
    assert files[0].read_bytes() == files[1].read_bytes()
    assert all(json.loads(path.read_text()) == {"top_class": int(reference_class)} for path in files)
    assert comparison["scope"] == "independent_reference" and comparison["enforced"] is True
    assert comparison["checks"] == [{"name": "top_class", "label": "Predicted class ID", "scope": "independent_reference", "actual": int(native_class), "operator": "==", "expected": int(reference_class), "passed": True}]
    assert recorder.data["checks"] == []
    assert _assessment(recorder.data)["kind"] == "reference"
    assert "Exact top class match" in render_case(recorder.data, recorder.directory)


@pytest.mark.parametrize(
    "native_class,margin,limit,accepted",
    [(817, 0.1, 0.12, True), (817, 0.12, 0.12, True), (817, 0.2, 0.12, False),
     (99, 0.1, 0.12, False), (817, 0.1, None, False)],
)
def test_classification_reporting_keeps_original_runner_up_route(
    classification_reporting_run, native_class, margin, limit, accepted
):
    import copy
    from tools.e2e_report import _assessment

    actual = {"top_class": native_class}
    expected = {"top_class": 656, "second_class": 817, "top1_margin": margin}
    original = copy.deepcopy((actual, expected))
    thresholds = {} if limit is None else {"top1_margin_atol": limit}
    recorder, events, errors, caught = classification_reporting_run(actual, expected, thresholds)
    assert (actual, expected) == original
    assert "reference_comparison" not in recorder.data
    assert not list(recorder.directory.parent.glob("classification-*.json"))
    if accepted:
        assert caught is None and errors == []
        assert events == ["build", "parity", "parity-passed"]
        assert len(recorder.data["checks"]) == 3
        assert "runner-up" in _assessment(recorder.data)["summary"]
    else:
        assert errors == [caught] and isinstance(caught, AssertionError)
        assert events == ["build", "parity"]
        assert recorder.data["failure_stage"] == "compare"
        assert _assessment(recorder.data)["kind"] == "failed"


@pytest.mark.parametrize("fail_file", ["classification-native.json", "classification-reference.json"])
def test_classification_reporting_io_failure_keeps_original_success(
    classification_reporting_run, fail_file
):
    from tools.e2e_report import _assessment

    recorder, events, errors, caught = classification_reporting_run(
        {"top_class": 0}, {"top_class": 0}, {}, fail_file=fail_file,
    )
    assert caught is None and errors == [] and "parity-passed" in events
    assert recorder.data["status"] == "passed"
    assert "reference_comparison" not in recorder.data
    assert "OSError: report file unavailable" in recorder.data["comparison_preview"]["error"]
    assert _assessment(recorder.data)["kind"] == "unverified"


def test_classification_reporting_disabled_does_not_write_files(classification_reporting_run):
    recorder, events, errors, caught = classification_reporting_run(
        {"top_class": 0}, {"top_class": 0}, {}, capture=False,
    )
    assert caught is None and errors == []
    assert events == ["build", "parity", "parity-passed"]
    assert not list(recorder.directory.parent.glob("classification-*.json"))
