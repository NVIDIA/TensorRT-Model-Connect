# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and reference-parity E2E for hstu.

Each testcase generates its synthetic checkpoint (or takes
``TRTMC_HSTU_MODEL_DIR`` / ``TRTMC_HSTU_SMALL_MODEL_DIR``), builds the bundle,
ranks the fixture record twice through the public Task API
(``hstu_task_probe``) and judges the logits against the golden under the
tolerance declared before any TensorRT output existed (``data/README.md``).
It then runs the same record through the family's native CLI (``trtmc hstu
decide``) and holds its document to the probe's. Needs ``TRTMC_RUNTIME_ROOT``
(the backend, family and CLI DSOs), ``TRTMC_NATIVE_BUILD_DIR`` (the probe; it
defaults to the runtime root) and ``TRTMC_BINARY`` (the ``trtmc`` CLI). No
torch: ``families.hstu.tests.golden`` wrote the goldens.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

FAMILY = "hstu"
TASK = "structured_decision"
TEST_ROOT = Path(__file__).resolve().parent


def _case_index() -> dict[str, tuple[dict, dict]]:
    cases = {}
    for path in sorted((TEST_ROOT / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK
        for case in manifest["testcases"]:
            assert case["name"] not in cases
            cases[case["name"]] = (manifest, case)
    return cases


CASES = _case_index()


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = set()
    for raw in metafunc.config.getoption("--e2e-model") or []:
        selected.update(item.strip() for item in str(raw).split(",") if item.strip())
    for raw in metafunc.config.getoption("--e2e-testcase") or []:
        selected.update(item.strip() for item in str(raw).split(",") if item.strip())
    names = sorted(CASES)
    chosen = {
        name for name, (manifest, _) in CASES.items() if selected & {FAMILY, name, manifest["name"]}
    }
    metafunc.parametrize(
        "case_name",
        [
            name
            if name in chosen
            else pytest.param(
                name,
                marks=pytest.mark.skip(
                    reason="direct E2E requires --e2e-model hstu (or a case name)"
                ),
            )
            for name in names
        ],
        ids=names,
    )


def _required_dir(name: str, default: str | None = None) -> Path:
    value = os.environ.get(name) or default
    assert value, f"selected {FAMILY} E2E requires {name}"
    path = Path(value)
    assert path.is_dir(), f"selected {FAMILY} E2E {name} is not a directory: {path}"
    return path


def _run(argv: list[str]) -> str:
    completed = subprocess.run(argv, capture_output=True, text=True)
    assert completed.returncode == 0, f"{argv[0]} failed:\n{completed.stderr}"
    return completed.stdout


def test_e2e(case_name: str, tmp_path: Path) -> None:
    from families.hstu.tests.parity import checkpoint_of, parity_record
    from families.hstu.weights import generate
    from tensorrt_model_connect import BuildRequest, build

    manifest, case = CASES[case_name]
    small = manifest["small"]
    golden = json.loads((TEST_ROOT / case["golden"]).read_text(encoding="utf-8"))
    overrides = json.loads(
        (TEST_ROOT / "thresholds" / f"{case_name}.json").read_text(encoding="utf-8")
    )["threshold_overrides"]
    limits = {"max_abs": overrides["max_abs"], "min_task_cosine": overrides["min_task_cosine"]}
    runtime_root = _required_dir("TRTMC_RUNTIME_ROOT")
    probe = _required_dir("TRTMC_NATIVE_BUILD_DIR", str(runtime_root)) / "hstu_task_probe"
    assert probe.is_file(), f"missing {probe}: build the hstu_task_probe target"
    trtmc = Path(os.environ.get("TRTMC_BINARY", ""))
    assert trtmc.is_file(), "selected hstu E2E requires TRTMC_BINARY (the trtmc CLI)"

    variable = "TRTMC_HSTU_SMALL_MODEL_DIR" if small else "TRTMC_HSTU_MODEL_DIR"
    model_dir = (
        Path(os.environ[variable])
        if os.environ.get(variable)
        else generate(tmp_path / "checkpoint", small=small)
    )
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=TASK,
            precision=manifest["precision"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        )
    )

    record_path = TEST_ROOT / case["record"]
    native = json.loads(_run([str(probe), str(bundle), str(runtime_root), str(record_path), "2"]))
    first, second = native["results"]
    assert first["scores"] == second["scores"], "two runs of one request disagree"
    record = parity_record(native, golden, limits, checkpoint=checkpoint_of(model_dir))
    assert record["passed"], json.dumps(record["repeats"], indent=1)

    document = json.loads(
        _run(
            [
                str(trtmc),
                FAMILY,
                "decide",
                str(bundle),
                "--record",
                str(record_path),
                "--runtime-root",
                str(runtime_root),
            ]
        )
    )
    assert document == first["response"], "the CLI's decision is not the Task API's"
