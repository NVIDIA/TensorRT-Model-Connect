# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for dinov2."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence
import json
import os
import subprocess
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "dinov2"
TASKS = frozenset({"image_to_token_and_pooled_features"})
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


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    return binary, runtime_root


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
    record_evidence("inputs", {"asset": path})
    return path


def _cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert denominator > 0.0
    return float(np.dot(a, b) / denominator)


def _native(binary: Path, runtime_root: Path, bundle: Path, manifest: dict, case: dict):
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "extract-features",
        "--image",
        str(_asset(case["test_image"])),
    )


def _official_reference(model_dir: Path, case: dict) -> dict:
    import torch
    import transformers
    from PIL import Image

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_class = {
        "dinov2": transformers.Dinov2Model,
        "dinov2_with_registers": transformers.Dinov2WithRegistersModel,
    }[config["model_type"]]
    # Pin the Pillow processor: AutoImageProcessor selects the torchvision
    # variant whenever torchvision happens to be installed.
    processor = transformers.BitImageProcessor.from_pretrained(model_dir)
    model = model_class.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        outputs = model(pixel_values=pixels)
    return {
        "last_hidden_state": outputs.last_hidden_state.float().numpy(),
        "pooler_output": outputs.pooler_output.float().numpy(),
        "num_register_tokens": int(getattr(model.config, "num_register_tokens", 0)),
        "patch_grid": [dim // model.config.patch_size for dim in pixels.shape[-2:]],
    }


def _assert_token_contract(actual: dict, expected: dict, case: dict) -> None:
    registers = int(case["num_register_tokens"])
    assert int(expected["num_register_tokens"]) == registers
    assert actual["grid_shape"] == expected["patch_grid"]
    rows, columns = actual["grid_shape"]
    roles = [token["role"] for token in actual["tokens"]]
    assert roles == ["class"] + ["register"] * registers + ["patch"] * (rows * columns)
    patches = actual["tokens"][1 + registers :]
    assert [(token["grid_row"], token["grid_column"]) for token in patches] == [
        (row, column) for row in range(rows) for column in range(columns)
    ]
    for token in patches:
        x_min, y_min, x_max, y_max = token["source_normalized_box"]
        assert 0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0
    assert actual["pooling"] == "cls" and actual["normalization"] == "none"


def _assert_parity(actual: dict, expected: dict, case: dict, thresholds: dict) -> None:
    actual_hidden = np.asarray(actual["last_hidden_state"], dtype=np.float64).reshape(
        actual["last_hidden_state_shape"]
    )
    expected_hidden = np.asarray(expected["last_hidden_state"], dtype=np.float64)
    actual_pooler = np.asarray(actual["pooler_output"], dtype=np.float64).reshape(
        actual["pooler_output_shape"]
    )
    expected_pooler = np.asarray(expected["pooler_output"], dtype=np.float64)
    assert actual_hidden.shape == expected_hidden.shape
    assert actual_pooler.shape == expected_pooler.shape
    assert np.isfinite(actual_hidden).all() and np.isfinite(actual_pooler).all()
    assert np.isfinite(expected_hidden).all() and np.isfinite(expected_pooler).all()
    assert np.array_equal(actual_pooler, actual_hidden[:, 0, :])
    assert np.array_equal(expected_pooler, expected_hidden[:, 0, :])
    assert _cosine(actual_hidden, expected_hidden) >= float(thresholds["full_cosine"])
    assert _cosine(actual_hidden[:, 0, :], expected_hidden[:, 0, :]) >= float(
        thresholds["cls_cosine"]
    )
    assert _cosine(actual_pooler, expected_pooler) >= float(thresholds["pooler_cosine"])
    relative_frobenius = np.linalg.norm(actual_hidden - expected_hidden) / max(
        float(np.linalg.norm(expected_hidden)), 1e-12
    )
    assert float(relative_frobenius) <= float(thresholds["relative_frobenius"])
    register_count = int(case["num_register_tokens"])
    patch_start = 1 + register_count
    if register_count:
        assert _cosine(
            actual_hidden[:, 1:patch_start, :], expected_hidden[:, 1:patch_start, :]
        ) >= float(thresholds["register_cosine"])
    actual_patches = actual_hidden[:, patch_start:, :].reshape(-1, actual_hidden.shape[-1])
    expected_patches = expected_hidden[:, patch_start:, :].reshape(-1, expected_hidden.shape[-1])
    assert actual_patches.shape[0] > 0
    denominators = np.linalg.norm(actual_patches, axis=1) * np.linalg.norm(expected_patches, axis=1)
    patch_cosines = np.divide(
        np.sum(actual_patches * expected_patches, axis=1),
        denominators,
        out=np.zeros_like(denominators),
        where=denominators != 0.0,
    )
    assert float(np.mean(patch_cosines)) >= float(thresholds["mean_patch_cosine"])
    assert float(np.percentile(patch_cosines, 1.0)) >= float(thresholds["p01_patch_cosine"])


def test_register_token_roles_must_match_case_contract() -> None:
    actual = {
        "grid_shape": [1, 1],
        "tokens": [
            {"role": "class"},
            {
                "role": "patch",
                "grid_row": 0,
                "grid_column": 0,
                "source_normalized_box": [0, 0, 1, 1],
            },
        ],
        "pooling": "cls",
        "normalization": "none",
    }
    expected = {"num_register_tokens": 0, "patch_grid": [1, 1]}
    _assert_token_contract(actual, expected, {"num_register_tokens": 0})
    with pytest.raises(AssertionError):
        _assert_token_contract(
            actual, {**expected, "num_register_tokens": 1}, {"num_register_tokens": 1}
        )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, manifest, case)
    record_evidence("native", actual)
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, case)
    record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_token_contract(actual, expected, case)
        _assert_parity(
            actual, expected, case, record_evidence("thresholds", _thresholds(case_name))
        )
