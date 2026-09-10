# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build and native Task qualification for Boltz-2."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from tensorrt_model_connect import BuildRequest, build


FAMILY = "boltz2"
TASK = "structure_prediction"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in MANIFEST_ROOT.glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK
        for case in manifest["testcases"]:
            result[case["name"]] = (manifest, case)
    return result


CASES = _cases()


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for value in metafunc.config.getoption(option) or []:
            selected.update(item.strip() for item in value.split(",") if item.strip())
    models_file = metafunc.config.getoption("--e2e-models-file")
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    enabled = bool(selected)
    names = [
        name
        for name, (manifest, _) in CASES.items()
        if not selected or selected & {FAMILY, name, manifest["name"]}
    ]
    metafunc.parametrize(
        "case_name",
        names
        if enabled
        else [pytest.param(name, marks=pytest.mark.skip(reason="select Boltz-2 E2E explicitly")) for name in names],
        ids=names,
    )


def _required_environment(name: str) -> Path:
    value = os.environ.get(name)
    assert value, f"selected Boltz-2 E2E requires {name}"
    path = Path(value)
    assert path.exists(), f"{name} does not exist: {path}"
    return path


def _model_dir(manifest: dict, tmp_path: Path) -> Path:
    explicit = os.environ.get("TRTMC_BOLTZ2_MODEL_DIR")
    if explicit:
        return _required_environment("TRTMC_BOLTZ2_MODEL_DIR")

    from boltz.main import process_inputs
    from huggingface_hub import snapshot_download

    try:
        snapshot = Path(
            snapshot_download(
                repo_id=manifest["hf_id"],
                revision=manifest["hf_revision"],
                local_files_only=True,
                allow_patterns=["boltz2_conf.ckpt", "mols.tar"],
            )
        )
    except Exception as error:
        raise AssertionError(
            "selected Boltz-2 E2E requires the exact cached public checkpoint"
        ) from error

    model_dir = tmp_path / "boltz2-model"
    model_dir.mkdir()
    for name in ("boltz2_conf.ckpt", "mols.tar"):
        source = snapshot / name
        assert source.is_file(), f"cached Boltz-2 snapshot is missing {name}"
        (model_dir / name).symlink_to(source.resolve(strict=True))
    shutil.copy2(TEST_ROOT / "data/protein_monomer.yaml", model_dir)
    shutil.copy2(TEST_ROOT / "data/protein_monomer.a3m", model_dir)
    with tarfile.open(model_dir / "mols.tar", "r") as archive:
        archive.extractall(model_dir, filter="data")
    mols = model_dir / "mols"
    assert mols.is_dir(), "cached Boltz-2 molecule archive has no mols directory"

    previous = Path.cwd()
    try:
        os.chdir(model_dir)
        process_inputs(
            data=[model_dir / "protein_monomer.yaml"],
            out_dir=model_dir,
            ccd_path=mols / "unused.pkl",
            mol_dir=mols,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            max_msa_seqs=1024,
            use_msa_server=False,
            boltz2=True,
            preprocessing_threads=1,
        )
    finally:
        os.chdir(previous)
    return model_dir


def _last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AssertionError("native prediction emitted no JSON summary")


def _assert_live_reference_parity(
    model_dir: Path,
    processed_dir: Path,
    structure: Path,
    metadata: Path,
    output_dir: Path,
    *,
    atom_count: int,
    token_count: int,
) -> None:
    import torch

    from families.boltz2.reference import (
        compare_native,
        load_reference_model,
        predict_reference,
        save_reference_output,
    )
    from families.boltz2.request_preparation import load_profile_features

    batch = load_profile_features(processed_dir, model_dir / "mols")
    model = load_reference_model(model_dir / "boltz2_conf.ckpt")
    prediction = predict_reference(model, batch)
    reference = output_dir / "eager-reference.npz"
    save_reference_output(reference, prediction)
    del prediction, model, batch
    torch.cuda.empty_cache()
    compare_native(
        reference,
        structure,
        metadata,
        output_dir / "accuracy.json",
        expected_atom_count=atom_count,
        expected_token_count=token_count,
    )


def test_model_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest, tmp_path)
    binary = _required_environment("TRTMC_BINARY")
    runtime_root = _required_environment("TRTMC_RUNTIME_ROOT")
    assert manifest["hf_id"] and manifest["hf_revision"]

    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        )
    )
    structure = tmp_path / "prediction.cif"
    metadata = tmp_path / "prediction.json"
    request = TEST_ROOT / case["request"]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH")) if value
    )
    completed = subprocess.run(
        [
            str(binary),
            "predict-structure",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--input",
            str(request),
            "--output",
            str(structure),
            "--output-json",
            str(metadata),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    _last_json(completed.stdout)
    details = json.loads(metadata.read_text(encoding="utf-8"))
    thresholds = json.loads(
        (THRESHOLD_ROOT / f"{case_name}.json").read_text(encoding="utf-8")
    )["threshold_overrides"]
    atom_count = sum(line.startswith("ATOM ") for line in structure.read_text().splitlines())
    token_count = len(details["plddt"])
    assert atom_count == thresholds["atom_count"]
    assert token_count == thresholds["token_count"]
    assert structure.read_text(encoding="utf-8").startswith("data_boltz2\n#\nloop_\n")
    _assert_live_reference_parity(
        model_dir,
        model_dir / "processed",
        structure,
        metadata,
        tmp_path,
        atom_count=thresholds["atom_count"],
        token_count=thresholds["token_count"],
    )

    from families.boltz2.request_preparation import prepare_structure_request

    bundle_stat = bundle.stat()
    bundle_identity = (
        bundle_stat.st_dev,
        bundle_stat.st_ino,
        bundle_stat.st_size,
        bundle_stat.st_mtime_ns,
        bundle_stat.st_ctime_ns,
    )
    variant_request = TEST_ROOT / "data/protein_monomer_variant/protein_monomer_variant.yaml"
    prepared = tmp_path / "protein_monomer_variant.b2rq"
    request_cache = tmp_path / "request-cache"
    preparation = prepare_structure_request(
        model_dir,
        variant_request,
        prepared,
        cache_dir=request_cache,
    )
    variant_structure = tmp_path / "variant.cif"
    variant_metadata = tmp_path / "variant.json"
    completed = subprocess.run(
        [
            str(binary),
            "predict-structure",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--input",
            str(prepared),
            "--output",
            str(variant_structure),
            "--output-json",
            str(variant_metadata),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
    )
    _last_json(completed.stdout)
    bundle_stat = bundle.stat()
    assert (
        bundle_stat.st_dev,
        bundle_stat.st_ino,
        bundle_stat.st_size,
        bundle_stat.st_mtime_ns,
        bundle_stat.st_ctime_ns,
    ) == bundle_identity
    assert json.loads(variant_metadata.read_text(encoding="utf-8"))["profile"] == (
        "tokens_117_atoms_928"
    )
    _assert_live_reference_parity(
        model_dir,
        request_cache
        / str(preparation["cache_key"])[:2]
        / str(preparation["cache_key"])
        / "work/processed",
        variant_structure,
        variant_metadata,
        tmp_path / "variant-reference",
        atom_count=thresholds["atom_count"],
        token_count=thresholds["token_count"],
    )
    cached = prepare_structure_request(
        model_dir,
        variant_request,
        tmp_path / "protein_monomer_variant-cached.b2rq",
        cache_dir=request_cache,
    )
    assert cached["cache_hit"] is True
