# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded checkpoint -> public builder -> native C++ runtime -> NVIDIA HSTU oracle.

These cases qualify inference semantics and numerical fidelity. They make no
claim about recommendation quality from a pretrained production checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from safetensors.numpy import load_file

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

from families.hstu.tests.fixtures import (
    ITEM_IDS,
    SEED,
    context_length,
    make_checkpoint,
    sample_request,
    token_count,
)


FAMILY = "hstu"
TEST_ROOT = Path(__file__).resolve().parent


def _case_index() -> dict[str, tuple[dict, dict]]:
    cases = {}
    for path in sorted((TEST_ROOT / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "recommendation"
        for case in manifest["testcases"]:
            assert case["name"] not in cases
            cases[case["name"]] = (manifest, case)
    return cases


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    models = {
        item.strip()
        for raw in config.getoption("--e2e-model") or []
        for item in str(raw).split(",")
        if item.strip()
    }
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcases = {
        item.strip()
        for raw in config.getoption("--e2e-testcase") or []
        for item in str(raw).split(",")
        if item.strip()
    }
    enabled = bool(models or testcases)
    names = [
        name
        for name, (manifest, _) in CASES.items()
        if (not models or models.intersection({FAMILY, name, manifest["name"]}))
        and (not testcases or name in testcases)
    ]
    return sorted(names), enabled


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = (
        names
        if enabled
        else [
            pytest.param(
                name,
                marks=pytest.mark.skip(reason="HSTU E2E requires an explicit E2E selector"),
            )
            for name in names
        ]
    )
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected HSTU E2E requires {label}"
    path = Path(value).resolve()
    assert path.exists(), f"selected HSTU E2E requires existing {label}: {path}"
    return path


def _runtime() -> tuple[Path, Path, Path]:
    root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    explicit = os.environ.get("TRTMC_HSTU_BINARY")
    if not explicit:
        generic = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
        explicit = str(generic.with_name("trtmc-hstu"))
    binary = _required_path(explicit, "TRTMC_HSTU_BINARY or sibling trtmc-hstu")
    assert binary.is_file() and os.access(binary, os.X_OK)
    assert (root / "libtrtmc_backend_trt.so").is_file()
    assert (root / "libtrtmc_model_hstu.so").is_file()
    from families.hstu.tests.environment import reference_source

    upstream = reference_source()
    import torch

    assert torch.cuda.is_available(), "selected HSTU E2E requires CUDA"
    return binary, root, upstream


def _thresholds(name: str) -> dict:
    path = TEST_ROOT / "thresholds" / f"{name}.json"
    assert path.is_file(), f"selected HSTU E2E requires exact thresholds: {path}"
    values = json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]
    assert values["rtol"] > 0 and values["atol"] > 0
    return record_evidence("thresholds", values)


def _native(binary: Path, runtime_root: Path, bundle: Path, request: dict, path: Path) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    input_path, output_path = path / "request.json", path / "result.json"
    input_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    command = [
        str(binary),
        "--bundle",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--input-json",
        str(input_path),
        "--output-json",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        filter(None, (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")))
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=120, check=False
    )
    record_evidence("commands", {"argv": command, "returncode": completed.returncode})
    record_evidence("inputs", {"raw_file": input_path})
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    assert completed.returncode == 0, (
        f"native HSTU failed ({completed.returncode})\n{completed.stdout}\n{completed.stderr}"
    )
    assert output_path.is_file(), "native HSTU did not write its requested result JSON"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    record_evidence("native", output)
    return output


def _array(sequence: dict, field: str, width: int) -> np.ndarray:
    values = np.asarray(sequence[field], dtype=np.float64)
    assert values.size > 0 and values.ndim == 1 and np.isfinite(values).all()
    assert values.size % width == 0
    return values.reshape(-1, width)


def _assert_close(left, right, thresholds: dict) -> None:
    actual, expected = np.asarray(left), np.asarray(right)
    assert actual.shape == expected.shape and actual.size > 0
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = np.abs(actual - expected)
    record_evidence("metrics", {
        "shape": list(actual.shape),
        "max_absolute_error": float(difference.max()),
        "max_relative_error": float((difference / np.maximum(np.abs(expected), 1e-12)).max()),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12)),
    })
    np.testing.assert_allclose(actual, expected, rtol=thresholds["rtol"], atol=thresholds["atol"])


def _assert_parity(
    actual: dict, expected: dict, request: dict, config: dict, thresholds: dict
) -> None:
    assert len(actual["sequences"]) == len(expected["sequences"]) == len(request["sequences"])
    for native, reference, inputs in zip(
        actual["sequences"], expected["sequences"], request["sequences"]
    ):
        candidates = inputs["candidate_item_ids"]
        assert native["candidate_item_ids"] == candidates
        assert native["num_candidates"] == len(candidates)
        assert native["embedding_dim"] == config["hidden_size"]
        sequence_length = (
            token_count(inputs) - len(candidates)
            if config["mode"] == "retrieval"
            else token_count(inputs)
        )
        assert native["sequence_length"] == sequence_length
        embeddings = _array(native, "embeddings", config["hidden_size"])
        sequence_embeddings = _array(native, "sequence_embeddings", config["hidden_size"])
        assert embeddings.shape[0] == len(candidates)
        assert sequence_embeddings.shape[0] == sequence_length
        _assert_close(
            embeddings, _array(reference, "embeddings", config["hidden_size"]), thresholds
        )
        _assert_close(
            sequence_embeddings,
            _array(reference, "sequence_embeddings", config["hidden_size"]),
            thresholds,
        )
        _assert_close(np.linalg.norm(embeddings, axis=-1), np.ones(len(candidates)), thresholds)
        if config["mode"] == "ranking":
            assert native["output_dim"] == config["prediction_head"][-1]
            _assert_close(
                _array(native, "logits", native["output_dim"]),
                _array(reference, "logits", native["output_dim"]),
                thresholds,
            )
        else:
            assert native["output_dim"] == 1
            _assert_close(native["scores"], reference["scores"], thresholds)
            assert len(native["scores"]) == len(candidates)
            assert np.max(np.abs(native["scores"])) <= 1 + thresholds["atol"]


def _metamorphic_checks(
    binary, runtime_root, bundle, request, config, actual, thresholds, tmp_path
):
    """Observe native masks, temporal causality and variable-length batch behavior."""
    first = request["sequences"][0]
    width = config["hidden_size"]
    if config["is_causal"]:
        changed = deepcopy(request)
        changed["sequences"][0]["candidate_item_ids"][-1] = ITEM_IDS[-1]
        result = _native(binary, runtime_root, bundle, changed, tmp_path / "candidate-change")
        group = config["target_group_size"]
        untouched = ((len(first["candidate_item_ids"]) - 1) // group) * group
        assert untouched > 0
        _assert_close(
            _array(actual["sequences"][0], "embeddings", width)[:untouched],
            _array(result["sequences"][0], "embeddings", width)[:untouched],
            thresholds,
        )
        # Changing the final history item cannot affect earlier history queries.
        changed = deepcopy(request)
        changed["sequences"][0]["history_item_ids"][-1] = ITEM_IDS[-2]
        result = _native(binary, runtime_root, bundle, changed, tmp_path / "history-change")
        start = context_length(first)
        stride = 2 if "history_action_ids" in first else 1
        stop = start + stride * (len(first["history_item_ids"]) - 1)
        _assert_close(
            _array(actual["sequences"][0], "sequence_embeddings", width)[start:stop],
            _array(result["sequences"][0], "sequence_embeddings", width)[start:stop],
            thresholds,
        )
    if config["scaling_seqlen"] > 0:
        # The shorter user has padding in a batch; alone it has no padding.
        single = {"sequences": [deepcopy(request["sequences"][1])]}
        result = _native(binary, runtime_root, bundle, single, tmp_path / "single-user")
        _assert_close(
            actual["sequences"][1]["sequence_embeddings"],
            result["sequences"][0]["sequence_embeddings"],
            thresholds,
        )
        field = "logits" if config["mode"] == "ranking" else "scores"
        _assert_close(actual["sequences"][1][field], result["sequences"][0][field], thresholds)


def test_seeded_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    binary, runtime_root, upstream = _runtime()
    thresholds = _thresholds(case_name)
    model_dir = tmp_path / "checkpoint"
    config = make_checkpoint(model_dir, **case["config_overrides"])
    request = sample_request(config, **case.get("fixture", {}))
    # Preserve every canonical tensor through the generic evidence file format.
    # The recorder intentionally does not copy checkpoint-specific extensions.
    arrays_path = model_dir / "checkpoint-arrays.npz"
    np.savez_compressed(arrays_path, **load_file(str(model_dir / "model.safetensors")))
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "seed": SEED,
            "weights": model_dir / "model.safetensors",
            "canonical_arrays": arrays_path,
            "config": model_dir / "config.json",
            "qualification": "seeded semantic parity; not pretrained recommendation accuracy",
        },
    )
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_batch_size=manifest["max_batch_size"],
                max_sequence_length=config["max_sequence_length"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, request, tmp_path / "baseline")
    with evidence_stage("reference"):
        from families.hstu.tests.reference import reference_receipt, run_reference

        record_evidence("reference_source", reference_receipt(upstream))
        expected = run_reference(
            model_dir, request, upstream_root=upstream, precision=manifest["precision"]
        )
        record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_parity(actual, expected, request, config, thresholds)
        _metamorphic_checks(
            binary, runtime_root, bundle, request, config, actual, thresholds, tmp_path
        )
