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
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence


FAMILY = "boltz2"
TASK = "structure_prediction"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


def _record_request_evidence(request: Path, prepared: Path | None = None) -> None:
    if not evidence_enabled():
        return
    try:
        from families.boltz2.contracts import parse_request_yaml

        record_evidence("inputs", {"request_file": request, "prepared_request": prepared})
        if request.stat().st_size > 65536:
            record_evidence("input_preview", {"omitted": "request exceeds the text preview bound"})
            return
        parsed = parse_request_yaml(request.read_text(encoding="utf-8"))
        record_evidence(
            "inputs",
            {
                "text": "\n\n".join(
                    "Protein " + ", ".join(sequence.chain_ids) + "\n" + sequence.sequence
                    for sequence in parsed.sequences
                ),
                "sequences": [
                    {
                        "chain_ids": list(sequence.chain_ids),
                        "sequence": sequence.sequence,
                        "msa": request.parent / sequence.msa_path,
                    }
                    for sequence in parsed.sequences
                ],
            },
        )
    except Exception as error:
        record_evidence("input_preview", {"error": f"{type(error).__name__}: {error}"})


def _record_native_evidence(structure: Path, metadata: Path) -> None:
    if not evidence_enabled():
        return
    try:
        record_evidence("native_artifacts", {"structure": structure, "metadata": metadata})
        if metadata.stat().st_size <= 8 * 1024 * 1024:
            record_evidence("native", json.loads(metadata.read_text(encoding="utf-8")))
        else:
            record_evidence("native", {"metadata": metadata, "omitted": "metadata exceeds the JSON preview bound"})
    except Exception as error:
        record_evidence("native_preview", {"error": f"{type(error).__name__}: {error}"})


def _record_reference_comparison(structure: Path, reference: Path, accuracy: Path) -> None:
    if not evidence_enabled():
        return
    try:
        record_evidence("comparison_artifacts", {"accuracy": accuracy})
        if accuracy.stat().st_size > 8 * 1024 * 1024:
            record_evidence("comparison_preview", {"omitted": "accuracy exceeds the JSON preview bound"})
            return
        metrics = json.loads(accuracy.read_text(encoding="utf-8"))
        record_evidence("metrics", metrics)
        qualification = metrics["qualification"]
        thresholds, outcomes = qualification["thresholds"], qualification["checks"]
        checks = [
            {
                "name": name,
                "label": label,
                "scope": "contract",
                "actual": metrics[name],
                "operator": "==",
                "expected": expected,
                "passed": outcomes[name],
            }
            for name, label, expected in (
                ("all_outputs_finite", "Finite outputs", True),
                ("atom_count", "Atom count", thresholds["atom_count"]),
                ("token_count", "Token count", thresholds["token_count"]),
            )
        ]
        checks.extend(
            {
                "name": name,
                "label": label,
                "scope": "independent_reference",
                "actual": metrics[name],
                "operator": operator,
                "expected": thresholds[threshold],
                "passed": outcomes[name],
            }
            for name, label, operator, threshold in (
                ("lddt", "lDDT", ">=", "lddt_min"),
                ("kabsch_rmsd_angstrom", "Aligned RMSD (Å)", "<=", "kabsch_rmsd_angstrom_max"),
                ("plddt_mean_abs", "Mean pLDDT difference", "<=", "plddt_mean_abs_max"),
                ("confidence_score_abs", "Confidence score difference", "<=", "confidence_score_abs_max"),
                ("complex_plddt_abs", "Complex pLDDT difference", "<=", "complex_plddt_abs_max"),
                ("ptm_abs", "pTM difference", "<=", "ptm_abs_max"),
            )
        )
        record_evidence(
            "reference_comparison",
            {
                "label": structure.stem,
                "scope": "independent_reference",
                "enforced": True,
                "native": structure,
                "reference": reference,
                "checks": checks,
            },
        )
    except Exception as error:
        record_evidence("comparison_preview", {"error": f"{type(error).__name__}: {error}"})


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in MANIFEST_ROOT.glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK
        for case in manifest["testcases"]:
            result[case["name"]] = (manifest, case)
    return result


CASES = _cases()


def test_multichain_protein_request_contract() -> None:
    from families.boltz2.contracts import parse_request_yaml

    request = parse_request_yaml(
        """version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: ACDE
      msa: shared.a3m
  - protein:
      id: C
      sequence: FGHIK
      msa: chain-c.a3m
"""
    )
    assert request.token_count == 13
    assert request.sequences[0].chain_ids == ("A", "B")
    assert request.sequences[1].chain_ids == ("C",)


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
    with evidence_stage("reference"):
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
        record_evidence("reference", {"artifact": reference, "kind": "seeded eager model output"})
        del prediction, model, batch
        torch.cuda.empty_cache()
    with evidence_stage("compare"):
        try:
            compare_native(
                reference,
                structure,
                metadata,
                output_dir / "accuracy.json",
                expected_atom_count=atom_count,
                expected_token_count=token_count,
            )
        finally:
            _record_reference_comparison(structure, reference, output_dir / "accuracy.json")


def test_model_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    _record_request_evidence(TEST_ROOT / case["request"])
    model_dir = _model_dir(manifest, tmp_path)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]})
    binary = _required_environment("TRTMC_BINARY")
    runtime_root = _required_environment("TRTMC_RUNTIME_ROOT")
    assert manifest["hf_id"] and manifest["hf_revision"]

    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
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
    with evidence_stage("native"):
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
        record_evidence("native_process", {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr})
        record_evidence("native_summary", _last_json(completed.stdout))
        details = json.loads(metadata.read_text(encoding="utf-8"))
        record_evidence("native", {**details, "structure": structure, "metadata": metadata})
    thresholds = json.loads(
        (THRESHOLD_ROOT / f"{case_name}.json").read_text(encoding="utf-8")
    )["threshold_overrides"]
    record_evidence("thresholds", thresholds)
    atom_count = sum(line.startswith("ATOM ") for line in structure.read_text().splitlines())
    token_count = len(details["plddt"])
    with evidence_stage("compare"):
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
    from families.boltz2.contracts import parse_request_yaml

    variant_request = TEST_ROOT / "data/protein_monomer_variant/protein_monomer_variant.yaml"
    variant = parse_request_yaml(variant_request.read_text(encoding="utf-8"))
    complex_sequence = variant.sequences[0].sequence[:50]
    complex_root = tmp_path / "protein-complex"
    complex_root.mkdir()
    complex_a3m = complex_root / "protein_complex.a3m"
    complex_a3m.write_text(f">query\n{complex_sequence}\n", encoding="utf-8")
    complex_request = complex_root / "protein_complex.yaml"
    complex_request.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: [A, B]\n"
        f"      sequence: {complex_sequence}\n"
        f"      msa: {complex_a3m.name}\n",
        encoding="utf-8",
    )
    prepared = tmp_path / "protein_complex.b2rq"
    request_cache = tmp_path / "request-cache"
    preparation = prepare_structure_request(
        model_dir,
        complex_request,
        prepared,
        cache_dir=request_cache,
    )
    record_evidence("request_preparation", preparation)
    _record_request_evidence(complex_request, prepared)
    complex_structure = tmp_path / "complex.cif"
    complex_metadata = tmp_path / "complex.json"
    with evidence_stage("native"):
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
                str(complex_structure),
                "--output-json",
                str(complex_metadata),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=1800,
        )
        record_evidence("native_process", {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr})
        record_evidence("native_summary", _last_json(completed.stdout))
    _record_native_evidence(complex_structure, complex_metadata)
    bundle_stat = bundle.stat()
    with evidence_stage("compare"):
        assert (
            bundle_stat.st_dev,
            bundle_stat.st_ino,
            bundle_stat.st_size,
            bundle_stat.st_mtime_ns,
            bundle_stat.st_ctime_ns,
        ) == bundle_identity
        complex_details = json.loads(complex_metadata.read_text(encoding="utf-8"))
        assert complex_details["profile"] == "tokens_117_atoms_928"
        assert complex_details["active_token_count"] == 100
        assert complex_details["active_atom_count"] == 794
        assert complex_details["chain_pair_confidence"] == []
    _assert_live_reference_parity(
        model_dir,
        request_cache
        / str(preparation["cache_key"])[:2]
        / str(preparation["cache_key"])
        / "work/processed",
        complex_structure,
        complex_metadata,
        tmp_path / "complex-reference",
        atom_count=794,
        token_count=100,
    )
    cached = prepare_structure_request(
        model_dir,
        complex_request,
        tmp_path / "protein_complex-cached.b2rq",
        cache_dir=request_cache,
    )
    record_evidence("request_preparation", cached)
    with evidence_stage("compare"):
        assert cached["cache_hit"] is True
