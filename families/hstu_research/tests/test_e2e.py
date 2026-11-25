# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected real-checkpoint Task, cache, and alternate-profile qualification."""

import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from tools.e2e_evidence import mark_loaded, record_evidence
from tools.variant_coverage import covers_variant
from families.hstu_research.config import Config
from families.hstu_research.checkpoint import load_weights
from families.hstu_research.reference import Reference, example_inputs
from families.hstu_research.triton.native_client import NativeClient

ROOT = Path(__file__).parent
CASES = {
    case["name"]: (manifest, case)
    for path in (ROOT / "manifests").glob("*.json")
    for manifest in [json.loads(path.read_text())]
    for case in manifest["testcases"]
}


@pytest.fixture(scope="session")
def research_root(tmp_path_factory):
    supplied = os.environ.get("HSTU_RESEARCH_E2E_ROOT")
    if supplied:
        return Path(supplied)
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    assert runtime, "selected HSTU research E2E requires TRTMC_RUNTIME_ROOT"
    temporary = tmp_path_factory.mktemp("trained-hstu")
    source = os.environ.get("HSTU_RESEARCH_CHECKPOINT")
    if source is None:
        from families.hstu_research.checkpoint import download_default

        source = temporary / "checkpoint"
        download_default(source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "families.hstu_research.tests.prepare",
            "--checkpoint",
            str(source),
            "--root",
            str(temporary / "prepared"),
            "--runtime-root",
            runtime,
        ],
        check=True,
    )
    return temporary / "prepared/output"


def pytest_generate_tests(metafunc):
    if "research_case" not in metafunc.fixturenames:
        return
    selected = set(",".join(metafunc.config.getoption("--e2e-model")).split(",")) - {""}
    cases = set(",".join(metafunc.config.getoption("--e2e-testcase")).split(",")) - {""}
    filename = metafunc.config.getoption("--e2e-models-file")
    if filename:
        selected.update(
            value.strip()
            for value in Path(filename).read_text().splitlines()
            if value.strip() and not value.startswith("#")
        )
    names = [
        name
        for name, (m, _) in CASES.items()
        if (
            not selected or "hstu_research" in selected or m["name"] in selected or name in selected
        )
        and (not cases or name in cases)
    ]
    enabled = bool(selected or cases or os.environ.get("HSTU_RESEARCH_E2E_ROOT"))
    metafunc.parametrize(
        "research_case",
        [
            name
            if enabled
            else pytest.param(
                name, marks=pytest.mark.skip(reason="select HSTU research E2E explicitly")
            )
            for name in sorted(names)
        ],
    )


@pytest.mark.gpu
@covers_variant(precision="fp32")
@covers_variant(precision="bf16")
def test_trained_hstu_task_cache_and_profile(research_case, research_root):
    manifest, case = CASES[research_case]
    root = research_root
    directory = root / f"research-{case['profile']}-{manifest['precision']}"
    runtime = os.environ["TRTMC_RUNTIME_ROOT"]
    c = Config.load(directory)
    assert manifest["family"] == "hstu_research"
    assert manifest["task"] == "structured_ranking"
    assert manifest["max_sequence_length"] == c.architecture.max_history_length
    assert manifest["max_batch_size"] == 4
    assert manifest["tensor_parallel_size"] == manifest["context_parallel_size"] == 1
    assert case["reference_precision"] == "fp32"
    record_evidence("artifact", dict(bundle=manifest["bundle"], checkpoint=manifest["checkpoint"]))
    oracle = Reference(c, load_weights(directory, c))
    record_evidence("inputs", dict(manifest=manifest, case=case, checkpoint_dir=str(directory)))
    record_evidence(
        "checkpoint", json.loads((directory / "checkpoint_provenance.json").read_text())
    )
    b = case["batch"]
    a, s = c.architecture, c.serving
    per_history = (
        a.num_layers * a.num_heads * s.history_length * (a.attention_dim + a.value_dim) * 4
    )
    maximum = c.runtime(4, manifest["precision"])["max_batch_size"]
    with NativeClient(
        directory / "model.bundle",
        runtime,
        hydration=True,
        kv_cache_size_bytes=134217728,
        max_entries=8,
        lease_seconds=0,
    ) as task:
        mark_loaded()
        errors, agreement, overlap = [], [], []
        for seed in range(20):
            inputs = example_inputs(c, b, seed=42 + seed)
            expected, actual = oracle.score(inputs), task.score(inputs)
            gate = json.loads((ROOT / "thresholds" / f"{research_case}.json").read_text())[
                "threshold_overrides"
            ]
            np.testing.assert_allclose(
                actual["scores"],
                expected["scores"],
                atol=gate["score_atol"],
                rtol=gate["score_rtol"],
            )
            errors.append(float(np.max(np.abs(actual["scores"] - expected["scores"]))))
            agreement.append(float(np.mean(actual["picked_indices"] == expected["picked_indices"])))
            k = min(10, s.candidates)
            expected_top = np.argsort(-expected["scores"], axis=1, kind="stable")[:, :k]
            actual_top = np.argsort(-actual["scores"], axis=1, kind="stable")[:, :k]
            overlap.extend(
                len(set(left) & set(right)) / k for left, right in zip(expected_top, actual_top)
            )
        assert np.mean(agreement) >= gate["greedy_position_agreement"]
        assert np.mean(overlap) >= gate.get("top10_overlap", 1.0)
        history = {
            spec["name"]: inputs[spec["name"]] for spec in task.metadata["stages"][0]["inputs"]
        }
        request = {
            spec["name"]: inputs[spec["name"]] for spec in task.metadata["stages"][1]["inputs"]
        }
        snapshot = task.export_history(history, ["v1"] * b)
        handles = task.restore_history(snapshot)
        before = task.stats()
        restored = task.score_with_history(handles, ["v1"] * b, request)
        for key in actual:
            np.testing.assert_array_equal(actual[key], restored[key])
        after = task.stats()
        assert after["history_prepare_calls_total"] == before["history_prepare_calls_total"]
        assert after["decode_step_calls_total"] == before["decode_step_calls_total"] + 1
        assert after["resident_bytes"] == per_history * b
        assert after["reserved_bytes"] == per_history * (maximum + b)
        assert after["active_sequences"] == after["used_bytes"] == 0
        corrupt = bytearray(snapshot)
        corrupt[-1] ^= 1
        with pytest.raises(RuntimeError, match="checksum"):
            task.restore_history(bytes(corrupt))
        if s.batching == "fixed":
            with pytest.raises(RuntimeError):
                task.score(example_inputs(c, 1))
    record_evidence(
        "correctness",
        dict(
            case=research_case,
            max_abs_score_error=max(errors),
            greedy_position_agreement=float(np.mean(agreement)),
            top10_overlap=float(np.mean(overlap)),
            cold_vs_restored="bit identical",
            stats=after,
        ),
    )
