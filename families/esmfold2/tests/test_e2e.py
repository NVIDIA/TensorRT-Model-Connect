# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and reference E2E for ESMFold2.

The native CLI folds the case's protein; the pinned transformers port folds it again with the same random draws
(``reference.py``), so every sample is compared with its reference twin: coordinates in the same frame, aligned
C-alpha RMSD, C-alpha lDDT, per-residue pLDDT and pTM, against the case's calibrated thresholds.
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, mark_loaded, record_evidence

from .reference import load_reference, reference_fold
from .structure_metrics import kabsch_rmsd, lddt, read_mmcif_atoms, rmsd

FAMILY = "esmfold2"
TASK = "structure_prediction"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
# What the builder and the reference read; the chemical component pickle and the images are not needed.
CHECKPOINT_PATTERNS = ("config.json", "model.safetensors.index.json", "model-*-of-*.safetensors")
_CASE_KEYS = {"name", "premerge", "request", "seed", "reference_precision"}
# The CLI sends the Task defaults: recycling_steps 3, sampling_steps and diffusion_samples 0 (the checkpoint's).
TASK_RECYCLING_STEPS = 3


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK, path
        assert manifest["tensor_parallel_size"] == 1, path
        for case in manifest["testcases"]:
            assert set(case) == _CASE_KEYS, (path, sorted(set(case) ^ _CASE_KEYS))
            assert isinstance(case["seed"], int) and not isinstance(case["seed"], bool) and case["seed"] >= 0
            assert case["reference_precision"] in ("fp32", "bf16"), case
            assert case["name"] not in cases, case["name"]
            cases[case["name"]] = (manifest, case)
    assert cases, f"{FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> tuple[set[str], set[str]]:
    models = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        models |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return models, _csv_values(config.getoption("--e2e-testcase", default=[]) or [])


def _collected_cases(items) -> set[str]:
    cases = set()
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        if re.fullmatch(r"test_.*e2e", getattr(item, "originalname", "") or "") and "case_name" in parameters:
            cases.add(str(parameters["case_name"]))
    return cases


def _require_selected(case_name: str, manifest: dict, config, collected: set[str]) -> None:
    models, testcases = _selection(config)
    if os.environ.get("TRTMC_E2E") != "1" and not (models or testcases):
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if models and not ({FAMILY, manifest["name"], case_name} & models):
        pytest.skip(f"{case_name} was not selected")
    if testcases and case_name not in testcases:
        selected = {name for name, (other, _) in _CASES.items() if {FAMILY, other["name"], name} & models}
        if models and not selected & testcases:
            pytest.fail(f"--e2e-testcase {sorted(testcases)} matches none of the selected {FAMILY} cases")
        if not models and not testcases & collected:
            pytest.fail(f"--e2e-testcase {sorted(testcases)} matches no E2E case collected in this session")
        pytest.skip(f"{case_name} was not selected")


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _model_dir(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
            allow_patterns=list(CHECKPOINT_PATTERNS),
        )
    except Exception as error:
        raise AssertionError(f"selected {FAMILY} E2E requires the cached checkpoint {manifest['hf_id']}") from error
    return Path(snapshot)


def _sequence(request: Path) -> str:
    lines = [line.strip() for line in request.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines and lines[0].startswith(">") and len([line for line in lines if line.startswith(">")]) == 1
    return "".join(lines[1:])


def _expected_controls(model_dir: Path) -> tuple[int, int, int]:
    """``(loops, configured sampling steps, samples)`` a default CLI fold must run, read from the checkpoint."""

    head = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["structure_head"]
    return TASK_RECYCLING_STEPS, int(head["inference_num_steps"]), int(head["num_diffusion_samples"])


def _native(binary: Path, runtime_root: Path, bundle: Path, request: Path, seed: int, workdir: Path) -> list[dict]:
    structure = workdir / "prediction.cif"
    metadata = workdir / "prediction.json"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    completed = subprocess.run(
        [str(binary), "predict-structure", str(bundle), "--runtime-root", str(runtime_root), "--input", str(request),
         "--output", str(structure), "--output-json", str(metadata), "--seed", str(seed)],
        check=True, capture_output=True, text=True, env=env, timeout=3600,
    )
    record_evidence("native_process", {"argv": completed.args, "stderr": completed.stderr[-4000:]})
    payloads = [json.loads(line[line.find("{"):]) for line in completed.stdout.splitlines() if "{" in line]
    assert len(payloads) == 1, completed.stdout[-1000:]
    summary = payloads[0]
    entries = summary["samples"] if "samples" in summary else [summary]
    samples = []
    for entry in entries:
        details = json.loads(Path(entry["metadata_path"]).read_text(encoding="utf-8"))
        samples.append({**details, "structure_path": Path(entry["structure_path"])})
    return samples


def _calpha(atom_ids: list[tuple[str, str, int]], coords: np.ndarray) -> np.ndarray:
    return np.asarray([coords[index] for index, (name, _, _) in enumerate(atom_ids) if name == "CA"])


def _reference_calpha(reference_pdb: str) -> np.ndarray:
    rows = [line for line in reference_pdb.splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA"]
    return np.asarray([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in rows])


def _compare(native: list[dict], reference, sequence: str, thresholds: dict, workdir: Path) -> None:
    samples = len(reference.ptm)
    assert len(native) == samples, f"native returned {len(native)} samples, reference {samples}"
    tokens = len(sequence)
    assert reference.coords.shape[0] == reference.plddt.shape[0] == reference.complex_plddt.shape[0] == samples
    assert reference.plddt.shape[1] == tokens
    for name in ("coords", "plddt", "ptm", "complex_plddt"):
        assert np.isfinite(getattr(reference, name)).all(), f"the reference {name} is not finite"
    worst: dict[str, float] = {
        "aligned_rmsd": 0.0, "lddt": 1.0, "plddt_mae": 0.0, "ptm_abs": 0.0, "complex_plddt_abs": 0.0,
    }
    frames: list[float] = []
    native_all: list[np.ndarray] = []
    reference_all: list[np.ndarray] = []
    for entry in native:
        index = int(entry["sample_index"])
        atom_ids, coords, bfactors = read_mmcif_atoms(entry["structure_path"])
        assert len(atom_ids) == reference.real_atoms, "native structure has another atom count"
        assert len({seq for _, _, seq in atom_ids}) == tokens and np.isfinite(coords).all()
        native_ca = _calpha(atom_ids, coords)
        reference_ca = _reference_calpha(reference.pdb[index])
        assert native_ca.shape == reference_ca.shape == (tokens, 3)
        native_plddt = np.asarray(entry["plddt"], dtype=np.float64)
        assert native_plddt.shape == (tokens,) and ((native_plddt >= 0) & (native_plddt <= 1)).all()
        assert all(np.isfinite(float(entry[key])) for key in ("ptm", "complex_plddt")), entry
        # The B-factor column carries each atom's residue pLDDT (written with four decimals).
        residues = np.asarray([seq - 1 for _, _, seq in atom_ids])
        assert np.abs(bfactors - native_plddt[residues]).max() <= 5.1e-5, "B-factors are not the residue pLDDT"
        metrics = {
            "frame_rmsd": rmsd(coords, reference.coords[index][: reference.real_atoms]),
            "aligned_rmsd": kabsch_rmsd(native_ca, reference_ca),
            "lddt": lddt(native_ca, reference_ca),
            "plddt_mae": float(np.abs(native_plddt - reference.plddt[index]).mean()),
            "ptm_abs": abs(float(entry["ptm"]) - float(reference.ptm[index])),
            "complex_plddt_abs": abs(float(entry["complex_plddt"]) - float(reference.complex_plddt[index])),
        }
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        frames.append(metrics["frame_rmsd"])
        native_all.append(native_ca)
        reference_all.append(reference_ca)
        for key in ("aligned_rmsd", "plddt_mae", "ptm_abs", "complex_plddt_abs"):
            worst[key] = max(worst[key], metrics[key])
        worst["lddt"] = min(worst["lddt"], metrics["lddt"])
        native_path = workdir / f"native_ca_{index}.npy"
        reference_path = workdir / f"reference_ca_{index}.npy"
        np.save(native_path, native_ca.astype(np.float32))
        np.save(reference_path, reference_ca.astype(np.float32))
        checks = [
            {"name": "aligned_rmsd", "label": "C-alpha RMSD after superposition (A)", "operator": "<=",
             "actual": metrics["aligned_rmsd"], "expected": thresholds["aligned_rmsd_max"]},
            {"name": "lddt", "label": "C-alpha lDDT", "operator": ">=",
             "actual": metrics["lddt"], "expected": thresholds["lddt_min"]},
            {"name": "plddt_mae", "label": "Per-residue pLDDT mean absolute difference", "operator": "<=",
             "actual": metrics["plddt_mae"], "expected": thresholds["plddt_mae_max"]},
            {"name": "ptm_abs", "label": "pTM absolute difference", "operator": "<=",
             "actual": metrics["ptm_abs"], "expected": thresholds["ptm_abs_max"]},
            {"name": "complex_plddt_abs", "label": "Mean pLDDT absolute difference", "operator": "<=",
             "actual": metrics["complex_plddt_abs"], "expected": thresholds["plddt_mae_max"]},
        ]
        for check in checks:
            check["scope"] = "independent_reference"
            check["passed"] = (
                check["actual"] <= check["expected"] if check["operator"] == "<=" else check["actual"] >= check["expected"]
            )
        record_evidence(
            "reference_comparison",
            {"label": f"sample {index}", "scope": "independent_reference", "enforced": True,
             "native": native_path, "reference": reference_path, "checks": checks},
        )
    # The same draws put every sample in the reference's frame; bf16 rounding can still turn a few samples' global
    # pose, so the frame check takes the median over samples (a fold from other noise is tens of A off).
    worst["frame_rmsd_median"] = float(np.median(frames))
    record_evidence("worst_sample", worst)
    native_stack, reference_stack = workdir / "native_ca_all.npy", workdir / "reference_ca_all.npy"
    np.save(native_stack, np.stack(native_all).astype(np.float32))
    np.save(reference_stack, np.stack(reference_all).astype(np.float32))
    record_evidence("reference_comparison", {
        "label": "all samples", "scope": "independent_reference", "enforced": True,
        "native": native_stack, "reference": reference_stack,
        "checks": [{"name": "frame_rmsd_median", "label": "Median all-atom RMSD in the shared frame (A)",
                    "scope": "independent_reference", "operator": "<=", "actual": worst["frame_rmsd_median"],
                    "expected": thresholds["frame_rmsd_median_max"],
                    "passed": worst["frame_rmsd_median"] <= thresholds["frame_rmsd_median_max"]}],
    })
    print(f"{FAMILY} parity over {samples} samples: {json.dumps(worst, sort_keys=True)}")
    # Calibration (B300, TensorRT 11.4; 4 proteins of 20-76 residues x 2 seeds x 32 samples against the fp32
    # reference): the worst healthy sample of the native bf16 build and of the reference's own bf16 run reached, for
    # ESMFold2, aligned RMSD 0.28 A, lDDT 0.991, pLDDT MAE 0.014, pTM 0.020 and a median frame RMSD of 0.38 A; for
    # ESMFold2-Fast 0.67 A, 0.976, 0.023, 0.028 and 0.46 A. The thresholds keep about 2x. A fold from another seed has
    # a median frame RMSD of 15-18 A; a trunk without the language-model pair 4.5-7.8 A median aligned RMSD and
    # lDDT 0.40-0.46.
    assert worst["frame_rmsd_median"] <= thresholds["frame_rmsd_median_max"], worst
    assert worst["aligned_rmsd"] <= thresholds["aligned_rmsd_max"], worst
    assert worst["lddt"] >= thresholds["lddt_min"], worst
    assert worst["plddt_mae"] <= thresholds["plddt_mae_max"], worst
    assert worst["ptm_abs"] <= thresholds["ptm_abs_max"], worst
    assert worst["complex_plddt_abs"] <= thresholds["plddt_mae_max"], worst


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config, _collected_cases(request.session.items))
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    fasta = TEST_ROOT / case["request"]
    sequence = _sequence(fasta)
    record_evidence("inputs", {"manifest": manifest, "case": case, "prompt": sequence, "request_file": fasta})
    record_evidence("checkpoint", {"hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]})
    thresholds = json.loads((THRESHOLD_ROOT / f"{case_name}.json").read_text(encoding="utf-8"))["threshold_overrides"]
    record_evidence("thresholds", thresholds)

    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=model_dir, output_path=bundle, family=FAMILY, task=TASK,
                           precision=manifest["precision"], max_sequence_length=manifest["max_sequence_length"],
                           tensor_parallel_size=manifest["tensor_parallel_size"]))
    with evidence_stage("native"):
        native = _native(binary, runtime_root, bundle, fasta, case["seed"], tmp_path)
    mark_loaded()
    loops, sampling_steps, samples = _expected_controls(model_dir)
    for entry in native:
        assert (entry["recycling_steps"], entry["configured_sampling_steps"], entry["diffusion_samples"]) == (
            loops, sampling_steps, samples), entry
        assert entry["seed"] == case["seed"], entry
    assert sorted(int(entry["sample_index"]) for entry in native) == list(range(samples))
    assert [int(entry["rank"]) for entry in native] == list(range(samples))
    record_evidence("native", {"samples": len(native), "best": {key: native[0][key] for key in ("ptm", "complex_plddt")},
                               "structure": native[0]["structure_path"]})

    with evidence_stage("reference"):
        import torch

        model = load_reference(model_dir, case["reference_precision"])
        reference = reference_fold(
            model, sequence, seed=case["seed"], loops=loops, sampling_steps=sampling_steps, samples=samples
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    record_evidence("reference", {"reference_ptm": reference.ptm.tolist(),
                                  "reference_complex_plddt": reference.complex_plddt.tolist()})
    with evidence_stage("compare"):
        _compare(native, reference, sequence, thresholds, tmp_path)
