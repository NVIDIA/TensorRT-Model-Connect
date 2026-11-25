# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected real-checkpoint Task, cache, and alternate-profile qualification."""

import json
import os
import struct
import subprocess
from pathlib import Path
import numpy as np
import pytest
from tools.e2e_evidence import mark_loaded, record_evidence
from tools.variant_coverage import covers_variant
from families.hstu.config import Config
from families.hstu.checkpoint import load_weights
from families.hstu.reference import Reference, example_inputs
from families.hstu.triton.native_client import NativeClient

ROOT = Path(__file__).parent
CASES = {
    case["name"]: (manifest, case)
    for path in (ROOT / "manifests").glob("*.json")
    for manifest in [json.loads(path.read_text())]
    for case in manifest["testcases"]
}


def pytest_generate_tests(metafunc):
    if "dlrm_case" not in metafunc.fixturenames:
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
        if (not selected or "hstu" in selected or m["name"] in selected or name in selected)
        and (not cases or name in cases)
    ]
    enabled = bool(selected or cases or os.environ.get("HSTU_E2E_ROOT"))
    metafunc.parametrize(
        "dlrm_case",
        [
            name
            if enabled
            else pytest.param(name, marks=pytest.mark.skip(reason="select DLRMv3 E2E explicitly"))
            for name in sorted(names)
        ],
    )


@pytest.mark.gpu
@covers_variant(precision="fp32")
@covers_variant(precision="bf16")
def test_dlrmv3_e2e(dlrm_case, dlrm_root, dlrm_bridge_root, tmp_path):
    manifest, case = CASES[dlrm_case]
    root = dlrm_root
    directory = root / f"dlrmv3-{case['profile']}-{manifest['precision']}"
    runtime = os.environ["TRTMC_RUNTIME_ROOT"]
    c = Config.load(directory)
    assert manifest["family"] == "hstu"
    assert manifest["task"] == "structured_ranking"
    assert manifest["max_sequence_length"] == 512
    assert case.get("premerge", False) in (True, False)
    assert manifest["max_batch_size"] == 4
    assert manifest["tensor_parallel_size"] == manifest["context_parallel_size"] == 1
    assert case["reference_precision"] == "fp64"
    record_evidence("external_files", manifest["external_files"])
    record_evidence("artifact", dict(bundle=manifest["bundle"], checkpoint=manifest["checkpoint"]))
    oracle = Reference(c, load_weights(directory, c))
    record_evidence("inputs", dict(manifest=manifest, case=case, checkpoint_dir=str(directory)))
    record_evidence(
        "checkpoint", json.loads((directory / "checkpoint_provenance.json").read_text())
    )
    b = case["batch"]
    s = c.serving
    per_history = 5 * 4 * (s.history_length + 1) * 256 * 4
    maximum = c.runtime(4, manifest["precision"])["max_batch_size"]
    with NativeClient(
        directory / "model.bundle",
        runtime,
        bridge_root=dlrm_bridge_root,
        hydration=True,
        kv_cache_size_bytes=134217728,
        max_entries=8,
        lease_seconds=0,
    ) as task:
        mark_loaded()
        errors, agreement, overlap = [], [], []
        seeds = [*range(42, 50), *range(142, 158), *range(242, 258)]
        for seed in seeds:
            inputs = example_inputs(c, b, seed=seed)
            expected, actual = oracle.score(inputs), task.score(inputs)
            gate = json.loads((ROOT / "thresholds" / f"{dlrm_case}.json").read_text())[
                "threshold_overrides"
            ]
            np.testing.assert_allclose(
                actual["scores"],
                expected["scores"],
                atol=gate["score_atol"],
                rtol=gate["score_rtol"],
            )
            errors.append(float(np.max(np.abs(actual["scores"] - expected["scores"]))))
            agreement.append(
                float(
                    np.mean(
                        np.argmax(actual["scores"], axis=1) == np.argmax(expected["scores"], axis=1)
                    )
                )
            )
            k = min(10, s.candidates)
            expected_top = np.argsort(-expected["scores"], axis=1, kind="stable")[:, :k]
            actual_top = np.argsort(-actual["scores"], axis=1, kind="stable")[:, :k]
            overlap.extend(
                len(set(left) & set(right)) / k for left, right in zip(expected_top, actual_top)
            )
        groups = []
        for begin, end in ((0, 8), (8, 24), (24, 40)):
            top1 = float(np.mean(agreement[begin:end]))
            top10 = float(np.mean(overlap[begin * b : end * b]))
            assert top1 >= gate["top1_agreement"]
            assert top10 >= gate.get("top10_overlap", 1.0)
            groups.append(
                dict(seeds=seeds[begin:end], top1=top1, top10=top10, max_abs=max(errors[begin:end]))
            )
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
    fixture = tmp_path / "sdk-inputs.bin"
    with fixture.open("wb") as stream:
        stream.write(struct.pack("=QQQ", b, s.history_length, s.candidates))
        for spec in c.runtime(4, manifest["precision"])["inputs"]:
            stream.write(np.ascontiguousarray(inputs[spec["name"]]).tobytes())
        stream.write(np.ascontiguousarray(expected["scores"]).tobytes())
    build = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", runtime))
    for language in ("c", "cpp"):
        subprocess.run(
            [
                str(build / f"test_hstu_sdk_{language}"),
                str(directory / "model.bundle"),
                runtime,
                str(fixture),
                str(gate["score_atol"]),
                str(gate["score_rtol"]),
            ],
            check=True,
        )
    record_evidence("sdk_consumers", {"c": "passed", "cpp": "passed"})
    record_evidence(
        "correctness",
        dict(
            case=dlrm_case,
            max_abs_score_error=max(errors),
            top1_agreement=float(np.mean(agreement)),
            top10_overlap=float(np.mean(overlap)),
            cold_vs_restored="bit identical",
            stats=after,
            seeds=seeds,
            seed_groups=groups,
        ),
    )
