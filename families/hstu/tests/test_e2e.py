# SPDX-License-Identifier: Apache-2.0
"""Public HSTU direct Task tests; selected tests require real built bundles."""

import json
import os
from pathlib import Path
import numpy as np
import pytest
from families.hstu.checkpoint import load_weights
from families.hstu.config import Config
from families.hstu.reference import Reference, example_inputs
from families.hstu.triton.native_client import NativeClient
from tools.e2e_evidence import mark_loaded, record_evidence
from tools.variant_coverage import covers_variant

ROOT = Path(__file__).parent
CASES = {
    case["name"]: (manifest, case)
    for path in (ROOT / "manifests").glob("*.json")
    for manifest in [json.loads(path.read_text())]
    for case in manifest["testcases"]
}


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    model = set(",".join(metafunc.config.getoption("--e2e-model")).split(",")) - {""}
    cases = set(",".join(metafunc.config.getoption("--e2e-testcase")).split(",")) - {""}
    filename = metafunc.config.getoption("--e2e-models-file")
    if filename:
        model.update(
            v.strip()
            for v in Path(filename).read_text().splitlines()
            if v.strip() and not v.startswith("#")
        )
    selected = [
        n
        for n, (m, _) in CASES.items()
        if (not model or "hstu" in model or m["name"] in model or n in model)
        and (not cases or n in cases)
    ]
    enabled = bool(model or cases)
    metafunc.parametrize(
        "case_name",
        [
            n
            if enabled
            else pytest.param(n, marks=pytest.mark.skip(reason="select HSTU E2E explicitly"))
            for n in sorted(selected)
        ],
    )


@pytest.fixture(scope="session")
def e2e_root(tmp_path_factory):
    supplied = os.environ.get("HSTU_E2E_ROOT")
    return Path(supplied) if supplied else tmp_path_factory.mktemp("public-hstu")


@pytest.mark.gpu
@covers_variant(precision="fp32")
@covers_variant(precision="bf16")
def test_public_hstu_e2e(case_name, e2e_root):
    m, case = CASES[case_name]
    root = e2e_root / f"hstu{m['candidates']}-{m['precision']}"
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    assert runtime, "selected E2E requires TRTMC_RUNTIME_ROOT"
    if not os.environ.get("HSTU_E2E_ROOT") and not (root / "model.bundle").exists():
        from tensorrt_model_connect import BuildRequest, build
        from families.hstu.checkpoint import create_fixture

        create_fixture(root, m["candidates"])
        build(
            BuildRequest(
                model_dir=root,
                output_path=root / "model.bundle",
                family="hstu",
                task="structured_ranking",
                precision=m["precision"],
                max_batch_size=4,
            )
        )
    record_evidence(
        "inputs", dict(manifest=m, case=case, checkpoint_dir=str(root), runtime_root=runtime)
    )
    record_evidence("checkpoint", json.loads((root / "checkpoint_provenance.json").read_text()))
    c = Config.load(root)
    x = example_inputs(c, case["batch"])
    with NativeClient(root / "model.bundle", runtime) as task:
        mark_loaded()
        expected = Reference(c, load_weights(root, c)).score(x)
        actual = task.score(x)
        for name in actual:
            assert (
                actual[name].shape == expected[name].shape
                and actual[name].dtype == expected[name].dtype
            )
        gate = json.loads((ROOT / "thresholds/correctness.json").read_text())[m["precision"]]
        np.testing.assert_allclose(
            actual["scores"], expected["scores"], atol=gate["score_atol"], rtol=gate["score_rtol"]
        )
        agreement = float(np.mean(actual["picked_indices"] == expected["picked_indices"]))
        assert agreement >= gate["greedy_position_agreement"]
        overlap = float(
            np.mean(
                [
                    len(set(a[:10]) & set(b[:10])) / 10
                    for a, b in zip(actual["picked_indices"], expected["picked_indices"])
                ]
            )
        )
        if "top10_overlap" in gate:
            assert overlap >= gate["top10_overlap"]
        stats = task.stats()
        assert stats["reserved_bytes"] == 4 * 262144
        assert (
            stats["peak_bytes"] == case["batch"] * 262144
            and stats["used_bytes"] == stats["active_sequences"] == 0
        )
    record_evidence(
        "correctness",
        dict(
            case=case_name,
            max_abs_score_error=float(np.max(np.abs(actual["scores"] - expected["scores"]))),
            greedy_position_agreement=agreement,
            top10_overlap=overlap,
        ),
    )
