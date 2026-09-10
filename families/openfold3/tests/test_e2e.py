# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build and native invariant qualification for OpenFold3."""

from __future__ import annotations

import errno
import json
import math
import os
import shlex
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "openfold3"
TASK = "structure_prediction"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == TASK
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selection(config) -> set[str]:
    selected = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option, default=[]) or []:
            selected.update(value.strip() for value in str(raw).split(",") if value.strip())
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selection(metafunc.config)
    names = sorted(
        name
        for name, (manifest, _) in CASES.items()
        if not selected or selected & {FAMILY, manifest["name"], name}
    )
    parameters = (
        names
        if selected
        else [
            pytest.param(name, marks=pytest.mark.skip(reason="real OpenFold3 E2E was not selected"))
            for name in names
        ]
    )
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    root = _required_path(
        os.environ.get("TRTMC_OPENFOLD3_MODEL_DIR"),
        "TRTMC_OPENFOLD3_MODEL_DIR",
    )
    missing = [
        dependency["path"]
        for dependency in manifest["external_files"]
        if not (root / dependency["path"]).is_file()
    ]
    assert not missing, f"OpenFold3 model directory is missing declared files: {missing}"
    return root


def _prepared_package(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for name in ("of3-ob-2025-06-30-174k.pt", "components.bcif"):
        path = source / name
        assert path.is_file() and not path.is_symlink(), path
        try:
            os.link(path, destination / name)
        except OSError as error:
            if error.errno not in (
                errno.EXDEV,
                errno.EPERM,
                errno.EMLINK,
                errno.EOPNOTSUPP,
            ):
                raise
            shutil.copyfile(path, destination / name)
    shutil.copyfile(
        TEST_ROOT.parents[2] / "examples/models/openfold3/query_ubiquitin.json",
        destination / "query.json",
    )
    for name in ("openfold3_features.npz", "openfold3_structure.json"):
        shutil.copyfile(TEST_ROOT / "data" / name, destination / name)
    return destination


def _atom_coordinates(cif: str) -> list[float]:
    lines = cif.splitlines()
    for loop_index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers = []
        cursor = loop_index + 1
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_atom_site."):
            headers.append(lines[cursor].strip())
            cursor += 1
        if not headers:
            continue
        required = ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z")
        if any(name not in headers for name in required):
            raise AssertionError("OpenFold3 mmCIF atom loop is missing coordinate columns")
        indices = tuple(headers.index(name) for name in required)
        coordinates = []
        while cursor < len(lines):
            row = lines[cursor].strip()
            cursor += 1
            if not row or row.startswith("#") or row == "loop_" or row.startswith("_"):
                break
            fields = shlex.split(row)
            if len(fields) != len(headers):
                raise AssertionError("OpenFold3 mmCIF atom row has the wrong column count")
            coordinates.extend(float(fields[index]) for index in indices)
        if coordinates:
            return coordinates
    raise AssertionError("OpenFold3 result has no atom-site loop")


def _run_native(
    qualification: Path,
    runtime_root: Path,
    bundle: Path,
    request: Path,
    output_root: Path,
    index: int,
    timeout: int,
) -> tuple[str, dict]:
    structure = output_root / f"prediction-{index}.cif"
    metadata = output_root / f"prediction-{index}.json"
    subprocess.run(
        [
            str(qualification),
            str(bundle),
            str(runtime_root),
            str(request),
            str(structure),
            str(metadata),
        ],
        check=True,
        timeout=timeout,
    )
    return structure.read_text(encoding="utf-8"), json.loads(metadata.read_text(encoding="utf-8"))


@cache
def _qualification_binary() -> Path:
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(native_build),
            "--parallel",
            "8",
            "--target",
            "openfold3_qualification",
        ],
        check=True,
        timeout=600,
    )
    qualification = native_build / "families/openfold3/openfold3_qualification"
    assert qualification.is_file(), qualification
    return qualification


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    source = _model_dir(manifest)
    package = _prepared_package(source, tmp_path / "package")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    qualification = _qualification_binary()
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=package,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )
    request = package / "query.json"
    receipts = [
        _run_native(
            qualification,
            runtime_root,
            bundle,
            request,
            tmp_path,
            index,
            int(case["runtime_timeout_s"]),
        )
        for index in range(2)
    ]
    assert receipts[0] == receipts[1]
    cif, confidence = receipts[0]
    coordinates = _atom_coordinates(cif)
    plddt = confidence["plddt"]
    pae = confidence["pae"]
    pde = confidence["pde"]
    assert confidence["precision"] == f"{manifest['precision']}-mixed"
    assert confidence["token_count"] == case["expected_token_count"]
    assert confidence["atom_count"] == case["expected_atom_count"]
    assert len(coordinates) == 3 * case["expected_atom_count"]
    assert len(plddt) == case["expected_atom_count"]
    assert len(pae) == len(pde) == case["expected_token_count"] ** 2
    assert all(math.isfinite(float(value)) for value in coordinates + plddt + pae + pde)
    assert all(0.0 <= float(value) <= 100.0 for value in plddt)
    assert all(0.0 <= float(value) <= 32.0 for value in (*pae, *pde))
    assert 0.0 <= float(confidence["average_plddt"]) <= 100.0
    assert 0.0 <= float(confidence["gpde"]) <= 32.0
    assert 0.0 <= float(confidence["ptm"]) <= 1.0
    assert confidence["sample_rank"] == 0
    assert confidence["sample_ranking_score"] is None
    assert confidence["sample_ranking_score_applicable"] is False
