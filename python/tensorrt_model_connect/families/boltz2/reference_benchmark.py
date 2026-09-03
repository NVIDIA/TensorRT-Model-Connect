# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned eager/``torch.compile`` Boltz-2 benchmark and parity harness.

Run this module in the isolated Boltz v2.2.1 Python profile. Model loading,
feature construction, compilation, and warm-up are deliberately outside the
reported steady-state interval.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat

from .provenance import PINNED_BOLTZ2


NATIVE_QUALIFICATION_THRESHOLDS = {
    "atom_count": 899,
    "token_count": 117,
    "lddt_min": 0.55,
    "kabsch_rmsd_angstrom_max": 9.0,
    "plddt_mean_abs_max": 0.10,
    "confidence_score_abs_max": 0.10,
    "complex_plddt_abs_max": 0.10,
    "ptm_abs_max": 0.10,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision(source_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _load_batch(processed_dir: Path, mol_dir: Path):
    import torch
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest

    manifest = Manifest.load(processed_dir / "manifest.json")
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=(
            processed_dir / "constraints"
            if (processed_dir / "constraints").is_dir()
            else None
        ),
        template_dir=(
            processed_dir / "templates" if (processed_dir / "templates").is_dir() else None
        ),
        extra_mols_dir=(
            processed_dir / "mols" if (processed_dir / "mols").is_dir() else None
        ),
    )
    batch = next(iter(data_module.predict_dataloader()))
    return data_module.transfer_batch_to_device(batch, torch.device("cuda"), 0)


def _load_seeded_batch(processed_dir: Path, mol_dir: Path):
    """Assemble one static fixture after pinning preprocessing randomness."""

    _seed()
    return _load_batch(processed_dir, mol_dir)


def _load_model(checkpoint: Path, *, compiled: bool):
    import torch

    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    config = PINNED_BOLTZ2.reference_configuration
    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        # The digest and source revision are validated before this trusted
        # upstream Lightning checkpoint is deserialized. PyTorch 2.6+ defaults
        # to weights-only loading, which cannot decode its DictConfig metadata.
        weights_only=False,
        predict_args={
            "recycling_steps": config.recycling_steps,
            "sampling_steps": config.sampling_steps,
            "diffusion_samples": config.diffusion_samples,
            "max_parallel_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(
            MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=config.max_msa_sequences,
                use_paired_feature=True,
            )
        ),
        steering_args=asdict(BoltzSteeringParams()),
        # Compile only after strict checkpoint restoration. Constructing the
        # wrappers here prefixes state-dict keys with ``_orig_mod`` before
        # Lightning loads the unwrapped upstream checkpoint.
        compile_msa=False,
        compile_pairformer=False,
        compile_structure=False,
        compile_confidence=False,
    )
    if compiled:
        compile_options = {"dynamic": False, "fullgraph": False}
        model.msa_module = torch.compile(model.msa_module, **compile_options)
        model.pairformer_module = torch.compile(
            model.pairformer_module, **compile_options
        )
        model.structure_module.score_model = torch.compile(
            model.structure_module.score_model, **compile_options
        )
        model.confidence_module = torch.compile(
            model.confidence_module, **compile_options
        )
        # Boltz v2.2.1 deliberately selects ``_orig_mod`` during eval if these
        # selectors are true. Leave them false so inference executes the
        # wrappers installed above.
        model.is_msa_compiled = False
        model.is_pairformer_compiled = False
    return model.eval().cuda()


def _seed() -> None:
    import torch

    seed = PINNED_BOLTZ2.reference_configuration.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _predict(model, batch):
    import torch

    _seed()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.predict_step(batch, 0)


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _save_output(path: Path, prediction: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=_as_numpy(prediction["coords"]),
        atom_mask=_as_numpy(prediction["masks"]).astype(bool),
        token_mask=_as_numpy(prediction["token_masks"]).astype(bool),
        plddt=_as_numpy(prediction["plddt"]),
        confidence_score=_as_numpy(prediction["confidence_score"]),
        complex_plddt=_as_numpy(prediction["complex_plddt"]),
        complex_iplddt=_as_numpy(prediction["complex_iplddt"]),
        ptm=_as_numpy(prediction["ptm"]),
        iptm=_as_numpy(prediction["iptm"]),
        ligand_iptm=_as_numpy(prediction["ligand_iptm"]),
        protein_iptm=_as_numpy(prediction["protein_iptm"]),
    )


def _prediction_is_finite(prediction: dict[str, Any]) -> bool:
    import torch

    required = (
        "coords",
        "plddt",
        "confidence_score",
        "complex_plddt",
        "complex_iplddt",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
    )
    return all(bool(torch.isfinite(prediction[name]).all()) for name in required)


def _qualification_fixture_name(request_sha256: str, msa_sha256: str) -> str:
    pinned_inputs = {
        (
            PINNED_BOLTZ2.qualification_request_sha256,
            PINNED_BOLTZ2.qualification_msa_sha256,
        ): "protein_monomer",
        (
            PINNED_BOLTZ2.reusable_profile_fixture.request_sha256,
            PINNED_BOLTZ2.reusable_profile_fixture.msa_sha256,
        ): PINNED_BOLTZ2.reusable_profile_fixture.name,
    }
    name = pinned_inputs.get((request_sha256, msa_sha256))
    if name is None:
        raise RuntimeError("qualification request/MSA digest pair mismatch")
    return name


def run(args: argparse.Namespace) -> None:
    import torch

    compiled = args.mode == "compile"
    source_revision = _source_revision(args.source_dir)
    if source_revision != PINNED_BOLTZ2.source_revision:
        raise RuntimeError(
            f"Boltz source revision mismatch: {source_revision} != "
            f"{PINNED_BOLTZ2.source_revision}"
        )
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    if checkpoint_sha256 != PINNED_BOLTZ2.structure_checkpoint.sha256:
        raise RuntimeError(
            f"checkpoint digest mismatch: {checkpoint_sha256} != "
            f"{PINNED_BOLTZ2.structure_checkpoint.sha256}"
        )
    request_sha256 = _sha256_file(args.request)
    msa_sha256 = _sha256_file(args.msa)
    fixture_name = _qualification_fixture_name(request_sha256, msa_sha256)

    # Boltz applies a random rigid transform while assembling ``ref_pos``.
    # Seed before preprocessing so the oracle and the native bundle consume
    # the same model input, not merely the same diffusion stream.
    batch = _load_seeded_batch(args.processed_dir, args.mol_dir)
    model = _load_model(args.checkpoint, compiled=compiled)
    torch.cuda.synchronize()

    compile_and_warmup_start = time.perf_counter()
    prediction = None
    for _ in range(args.warmups):
        prediction = _predict(model, batch)
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - compile_and_warmup_start

    # Keep the steady-state memory baseline free of warm-up outputs. Also drop
    # each prior measured output before producing the next one so peak memory
    # represents a single prediction rather than two live result dictionaries.
    del prediction
    gc.collect()
    torch.cuda.empty_cache()

    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    latencies_ms: list[float] = []
    valid_outputs = 0
    prediction = None
    for _ in range(args.iterations):
        if prediction is not None:
            del prediction
        started = time.perf_counter()
        prediction = _predict(model, batch)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        valid_outputs += int(_prediction_is_finite(prediction))
    peak_bytes = torch.cuda.max_memory_allocated()
    if prediction is None:
        raise RuntimeError("benchmark produced no prediction")

    _save_output(args.output_npz, prediction)
    output_sha256 = _sha256_file(args.output_npz)
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "precision": "bf16-mixed",
        "source_revision": source_revision,
        "checkpoint_revision": PINNED_BOLTZ2.checkpoint_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "request_sha256": request_sha256,
        "msa_sha256": msa_sha256,
        "qualification_fixture": fixture_name,
        "output_sha256": output_sha256,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "compile_and_warmup_seconds": warmup_seconds,
        "latency_ms": latencies_ms,
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "throughput_samples_per_second": 1000.0 / statistics.mean(latencies_ms),
        "valid_outputs": valid_outputs,
        "valid_output_rate": valid_outputs / args.iterations,
        "atom_count": int(prediction["masks"].sum()),
        "token_count": int(prediction["token_masks"].sum()),
        "baseline_memory_bytes": baseline_bytes,
        "peak_memory_bytes": peak_bytes,
        "incremental_peak_memory_bytes": max(peak_bytes - baseline_bytes, 0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tensorrt_version": trt_compat.tensorrt_version(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

    del prediction, model, batch
    gc.collect()
    torch.cuda.empty_cache()


def _masked_coords(data: np.lib.npyio.NpzFile) -> np.ndarray:
    coords = np.asarray(data["coords"], dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(data["atom_mask"], dtype=bool).reshape(-1)
    if coords.shape[0] != mask.shape[0]:
        raise ValueError("coordinate and atom-mask shapes do not match")
    result = coords[mask]
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("coordinates are empty or non-finite")
    return result


def _kabsch_rmsd(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=0, keepdims=True)
    covariance = candidate_centered.T @ reference_centered
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    aligned = candidate_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))


def _lddt(reference: np.ndarray, candidate: np.ndarray, cutoff: float = 15.0) -> float:
    reference_distances = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
    candidate_distances = np.linalg.norm(candidate[:, None] - candidate[None, :], axis=-1)
    pairs = (reference_distances < cutoff) & (reference_distances > 0.0)
    if not pairs.any():
        raise ValueError("no atom pairs are eligible for lDDT")
    delta = np.abs(candidate_distances[pairs] - reference_distances[pairs])
    return float(np.mean([(delta < threshold).mean() for threshold in (0.5, 1.0, 2.0, 4.0)]))


def _native_mmcif_coords(path: Path) -> np.ndarray:
    coordinates: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM "):
            continue
        fields = line.split()
        if len(fields) != 13:
            raise ValueError("native Boltz-2 mmCIF atom row differs from its contract")
        coordinates.append([float(value) for value in fields[7:10]])
    result = np.asarray(coordinates, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or not np.isfinite(result).all():
        raise ValueError("native Boltz-2 coordinates are empty or non-finite")
    return result


def _native_qualification(
    metrics: dict[str, Any],
    *,
    expected_atom_count: int = NATIVE_QUALIFICATION_THRESHOLDS["atom_count"],
    expected_token_count: int = NATIVE_QUALIFICATION_THRESHOLDS["token_count"],
) -> dict[str, Any]:
    """Apply the frozen native quality gate at one static request shape."""

    thresholds = {
        **NATIVE_QUALIFICATION_THRESHOLDS,
        "atom_count": expected_atom_count,
        "token_count": expected_token_count,
    }
    checks = {
        "all_outputs_finite": bool(metrics["all_outputs_finite"]),
        "atom_count": int(metrics["atom_count"]) == thresholds["atom_count"],
        "token_count": int(metrics["token_count"]) == thresholds["token_count"],
        "lddt": float(metrics["lddt"]) >= thresholds["lddt_min"],
        "kabsch_rmsd_angstrom": (
            float(metrics["kabsch_rmsd_angstrom"])
            <= thresholds["kabsch_rmsd_angstrom_max"]
        ),
        "plddt_mean_abs": (
            float(metrics["plddt_mean_abs"])
            <= thresholds["plddt_mean_abs_max"]
        ),
        "confidence_score_abs": (
            float(metrics["confidence_score_abs"])
            <= thresholds["confidence_score_abs_max"]
        ),
        "complex_plddt_abs": (
            float(metrics["complex_plddt_abs"])
            <= thresholds["complex_plddt_abs_max"]
        ),
        "ptm_abs": float(metrics["ptm_abs"]) <= thresholds["ptm_abs_max"],
    }
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_native(args: argparse.Namespace) -> None:
    expected_atom_count = getattr(
        args, "expected_atom_count", NATIVE_QUALIFICATION_THRESHOLDS["atom_count"]
    )
    expected_token_count = getattr(
        args, "expected_token_count", NATIVE_QUALIFICATION_THRESHOLDS["token_count"]
    )
    if expected_atom_count <= 0 or expected_token_count <= 0:
        raise ValueError("native qualification expected counts must be positive")
    with np.load(args.reference_npz) as reference:
        reference_coords = _masked_coords(reference)
        reference_plddt = np.asarray(reference["plddt"], dtype=np.float64).reshape(-1)
        reference_confidence = float(np.asarray(reference["confidence_score"]).reshape(-1)[0])
        reference_complex_plddt = float(np.asarray(reference["complex_plddt"]).reshape(-1)[0])
        reference_ptm = float(np.asarray(reference["ptm"]).reshape(-1)[0])
    candidate_coords = _native_mmcif_coords(args.candidate_mmcif)
    metadata = json.loads(args.candidate_metadata.read_text(encoding="utf-8"))
    candidate_plddt = np.asarray(metadata.get("plddt", []), dtype=np.float64)
    if reference_coords.shape != candidate_coords.shape:
        raise ValueError("reference and native coordinate shapes do not match")
    if reference_plddt.shape != candidate_plddt.shape:
        raise ValueError("reference and native pLDDT shapes do not match")
    result = {
        "schema_version": 1,
        "atom_count": int(candidate_coords.shape[0]),
        "token_count": int(candidate_plddt.size),
        "all_outputs_finite": bool(
            np.isfinite(reference_coords).all()
            and np.isfinite(candidate_coords).all()
            and np.isfinite(reference_plddt).all()
            and np.isfinite(candidate_plddt).all()
        ),
        "lddt": _lddt(reference_coords, candidate_coords),
        "kabsch_rmsd_angstrom": _kabsch_rmsd(reference_coords, candidate_coords),
        "plddt_max_abs": float(np.max(np.abs(reference_plddt - candidate_plddt))),
        "plddt_mean_abs": float(np.mean(np.abs(reference_plddt - candidate_plddt))),
        "confidence_score_abs": abs(
            reference_confidence - float(metadata["confidence_score"])
        ),
        "complex_plddt_abs": abs(
            reference_complex_plddt - float(metadata["complex_plddt"])
        ),
        "ptm_abs": abs(reference_ptm - float(metadata["ptm"])),
    }
    if not result["all_outputs_finite"] or not all(
        math.isfinite(float(value))
        for key, value in result.items()
        if key not in {"schema_version", "atom_count", "token_count", "all_outputs_finite"}
    ):
        raise ValueError("native Boltz-2 qualification metrics are non-finite")
    result["qualification"] = _native_qualification(
        result,
        expected_atom_count=expected_atom_count,
        expected_token_count=expected_token_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if getattr(args, "enforce", False) and not result["qualification"]["passed"]:
        failed = ", ".join(
            name
            for name, passed in result["qualification"]["checks"].items()
            if not passed
        )
        raise RuntimeError(f"Boltz-2 native qualification failed: {failed}")


def compare(args: argparse.Namespace) -> None:
    with (
        np.load(args.reference_npz) as reference,
        np.load(args.candidate_npz) as candidate,
    ):
        reference_coords = _masked_coords(reference)
        candidate_coords = _masked_coords(candidate)
        if reference_coords.shape != candidate_coords.shape:
            raise ValueError("reference and candidate coordinate shapes do not match")
        reference_plddt = np.asarray(reference["plddt"], dtype=np.float64)
        candidate_plddt = np.asarray(candidate["plddt"], dtype=np.float64)
        if reference_plddt.shape != candidate_plddt.shape:
            raise ValueError("reference and candidate pLDDT shapes do not match")
        for name in ("confidence_score", "complex_plddt"):
            if np.shape(reference[name]) != np.shape(candidate[name]):
                raise ValueError(f"reference and candidate {name} shapes do not match")
        result = {
            "schema_version": 1,
            "atom_count": reference_coords.shape[0],
            "all_outputs_finite": bool(
                np.isfinite(reference_plddt).all()
                and np.isfinite(candidate_plddt).all()
                and np.isfinite(reference_coords).all()
                and np.isfinite(candidate_coords).all()
            ),
            "kabsch_rmsd_angstrom": _kabsch_rmsd(
                reference_coords, candidate_coords
            ),
            "lddt": _lddt(reference_coords, candidate_coords),
            "plddt_max_abs": float(
                np.max(np.abs(reference_plddt - candidate_plddt))
            ),
            "plddt_mean_abs": float(
                np.mean(np.abs(reference_plddt - candidate_plddt))
            ),
            "confidence_score_abs": float(
                np.max(
                    np.abs(
                        reference["confidence_score"] - candidate["confidence_score"]
                    )
                )
            ),
            "complex_plddt_abs": float(
                np.max(
                    np.abs(reference["complex_plddt"] - candidate["complex_plddt"])
                )
            ),
        }
    if not math.isfinite(result["kabsch_rmsd_angstrom"]):
        raise ValueError("Kabsch RMSD is non-finite")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def repeatability(args: argparse.Namespace) -> None:
    if len(args.prediction_npz) < 2:
        raise ValueError("repeatability requires at least two predictions")
    predictions: list[dict[str, Any]] = []
    for path in args.prediction_npz:
        with np.load(path) as archive:
            predictions.append(
                {
                    "path": str(path),
                    "coords": _masked_coords(archive),
                    "plddt": np.asarray(archive["plddt"], dtype=np.float64),
                    "confidence_score": np.asarray(
                        archive["confidence_score"], dtype=np.float64
                    ),
                    "complex_plddt": np.asarray(
                        archive["complex_plddt"], dtype=np.float64
                    ),
                }
            )
    comparisons: list[dict[str, Any]] = []
    reference = predictions[0]
    for candidate in predictions[1:]:
        if reference["coords"].shape != candidate["coords"].shape:
            raise ValueError("repeatability coordinate shapes do not match")
        for name in ("plddt", "confidence_score", "complex_plddt"):
            if reference[name].shape != candidate[name].shape:
                raise ValueError(f"repeatability {name} shapes do not match")
        comparison = {
            "reference": reference["path"],
            "candidate": candidate["path"],
            "lddt": _lddt(reference["coords"], candidate["coords"]),
            "kabsch_rmsd_angstrom": _kabsch_rmsd(
                reference["coords"], candidate["coords"]
            ),
            "plddt_mean_abs": float(
                np.mean(np.abs(reference["plddt"] - candidate["plddt"]))
            ),
            "confidence_score_abs": float(
                np.max(
                    np.abs(reference["confidence_score"] - candidate["confidence_score"])
                )
            ),
            "complex_plddt_abs": float(
                np.max(np.abs(reference["complex_plddt"] - candidate["complex_plddt"]))
            ),
        }
        comparisons.append(comparison)
    result = {
        "schema_version": 1,
        "prediction_count": len(predictions),
        "comparisons_to_primary": comparisons,
        "observed_envelope": {
            "lddt_min": min(value["lddt"] for value in comparisons),
            "kabsch_rmsd_angstrom_max": max(
                value["kabsch_rmsd_angstrom"] for value in comparisons
            ),
            "plddt_mean_abs_max": max(value["plddt_mean_abs"] for value in comparisons),
            "confidence_score_abs_max": max(
                value["confidence_score_abs"] for value in comparisons
            ),
            "complex_plddt_abs_max": max(
                value["complex_plddt_abs"] for value in comparisons
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    run_parser.add_argument("--source-dir", type=Path, required=True)
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--mol-dir", type=Path, required=True)
    run_parser.add_argument("--processed-dir", type=Path, required=True)
    run_parser.add_argument("--request", type=Path, required=True)
    run_parser.add_argument("--msa", type=Path, required=True)
    run_parser.add_argument("--output-json", type=Path, required=True)
    run_parser.add_argument("--output-npz", type=Path, required=True)
    run_parser.add_argument("--warmups", type=int, default=1)
    run_parser.add_argument("--iterations", type=int, default=3)
    run_parser.set_defaults(handler=run)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference-npz", type=Path, required=True)
    compare_parser.add_argument("--candidate-npz", type=Path, required=True)
    compare_parser.add_argument("--output-json", type=Path, required=True)
    compare_parser.set_defaults(handler=compare)

    native_parser = subparsers.add_parser("compare-native")
    native_parser.add_argument("--reference-npz", type=Path, required=True)
    native_parser.add_argument("--candidate-mmcif", type=Path, required=True)
    native_parser.add_argument("--candidate-metadata", type=Path, required=True)
    native_parser.add_argument("--output-json", type=Path, required=True)
    native_parser.add_argument("--expected-atom-count", type=int, default=899)
    native_parser.add_argument("--expected-token-count", type=int, default=117)
    native_parser.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero when the frozen native quality gate fails",
    )
    native_parser.set_defaults(handler=compare_native)

    repeatability_parser = subparsers.add_parser("repeatability")
    repeatability_parser.add_argument(
        "--prediction-npz", type=Path, action="append", required=True
    )
    repeatability_parser.add_argument("--output-json", type=Path, required=True)
    repeatability_parser.set_defaults(handler=repeatability)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "warmups", 1) < 1 or getattr(args, "iterations", 1) < 1:
        raise ValueError("warmups and iterations must both be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
