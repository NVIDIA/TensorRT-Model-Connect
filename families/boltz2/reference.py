# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded eager reference and native accuracy gate for Boltz-2 E2E tests."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import (
    checkpoint_safe_globals,
    validate_affinity_checkpoint,
    validate_structure_checkpoint,
)
from .provenance import PINNED_BOLTZ2
from .random_samples import AFFINITY_DIFFUSION_RNG_OFFSET


NATIVE_QUALIFICATION_THRESHOLDS = {
    "atom_count": 899,
    "token_count": 117,
    "lddt_min": 0.55,
    "kabsch_rmsd_angstrom_max": 9.0,
    "plddt_mean_abs_max": 0.10,
    "confidence_score_abs_max": 0.10,
    "complex_plddt_abs_max": 0.10,
    "complex_iplddt_abs_max": 0.10,
    "ptm_abs_max": 0.10,
    "iptm_abs_max": 0.10,
    "protein_iptm_abs_max": 0.10,
    "chains_ptm_max_abs_max": 0.10,
    "pair_chains_iptm_max_abs_max": 0.10,
}


def load_reference_model(checkpoint: Path):
    """Load the pinned public checkpoint without enabling pickle code execution."""

    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    validate_structure_checkpoint(checkpoint)
    config = PINNED_BOLTZ2.reference_configuration
    with checkpoint_safe_globals():
        model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            weights_only=True,
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
            compile_msa=False,
            compile_pairformer=False,
            compile_structure=False,
            compile_confidence=False,
        )
    return model.eval().cuda()


def load_affinity_reference_model(checkpoint: Path):
    """Load the pinned public affinity ensemble without unsafe pickle."""

    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    validate_affinity_checkpoint(checkpoint)
    with checkpoint_safe_globals():
        model = Boltz2.load_from_checkpoint(
            checkpoint,
            strict=True,
            weights_only=True,
            predict_args={
                "recycling_steps": 5,
                "sampling_steps": 200,
                "diffusion_samples": 5,
                "max_parallel_samples": 1,
                "write_confidence_summary": False,
                "write_full_pae": False,
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
                    num_subsampled_msa=8,
                    use_paired_feature=True,
                )
            ),
            steering_args=asdict(BoltzSteeringParams()),
            affinity_mw_correction=False,
            compile_msa=False,
            compile_pairformer=False,
            compile_structure=False,
            compile_confidence=False,
        )
    return model.eval().cuda()


def _seed() -> None:
    import torch

    seed = PINNED_BOLTZ2.reference_configuration.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_reference(model, batch):
    import torch

    _seed()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.predict_step(batch, 0)


def predict_affinity_reference(model, batch):
    """Run the affinity pass at the official post-structure RNG boundary."""

    import torch

    _seed()
    generator = torch.cuda.default_generators[torch.cuda.current_device()]
    generator.set_offset(AFFINITY_DIFFUSION_RNG_OFFSET)
    affinity_batch = dict(batch)
    affinity_batch["method_feature"] = (affinity_batch["token_pad_mask"] * 4).to(torch.int64)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.predict_step(affinity_batch, 0)


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _pair_chain_arrays(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pairs = prediction["pair_chains_iptm"]
    chain_ids = sorted(int(chain_id) for chain_id in pairs)
    if any(set(int(value) for value in pairs[chain_id]) != set(chain_ids) for chain_id in pairs):
        raise ValueError("reference pair-chain ipTM map is not square")
    values = np.empty((len(chain_ids), len(chain_ids)), dtype=np.float32)
    for first_index, first in enumerate(chain_ids):
        for second_index, second in enumerate(chain_ids):
            score = _as_numpy(pairs[first][second]).reshape(-1)
            if score.size != 1:
                raise ValueError("reference pair-chain ipTM score is not scalar")
            values[first_index, second_index] = score[0]
    return np.asarray(chain_ids, dtype=np.int32), values


def save_reference_output(path: Path, prediction: dict[str, Any]) -> None:
    pair_chain_ids, pair_chains_iptm = _pair_chain_arrays(prediction)
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
        protein_iptm=_as_numpy(prediction["protein_iptm"]),
        pair_chain_ids=pair_chain_ids,
        pair_chains_iptm=pair_chains_iptm,
    )


def _masked_coords(data: np.lib.npyio.NpzFile) -> np.ndarray:
    coords = np.asarray(data["coords"], dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(data["atom_mask"], dtype=bool).reshape(-1)
    if coords.shape[0] != mask.shape[0]:
        raise ValueError("coordinate and atom-mask shapes do not match")
    result = coords[mask]
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("coordinates are empty or non-finite")
    return result


def _masked_tokens(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    values = np.asarray(data[name], dtype=np.float64).reshape(-1)
    mask = np.asarray(data["token_mask"], dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError(f"{name} and token-mask shapes do not match")
    result = values[mask]
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"active {name} values are empty or non-finite")
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


def _qualification(
    metrics: dict[str, Any],
    *,
    expected_atom_count: int,
    expected_token_count: int,
) -> dict[str, Any]:
    thresholds = {
        **NATIVE_QUALIFICATION_THRESHOLDS,
        "atom_count": expected_atom_count,
        "token_count": expected_token_count,
    }
    checks = {
        "all_outputs_finite": bool(metrics["all_outputs_finite"]),
        "atom_count": int(metrics["atom_count"]) == expected_atom_count,
        "token_count": int(metrics["token_count"]) == expected_token_count,
        "lddt": float(metrics["lddt"]) >= thresholds["lddt_min"],
        "kabsch_rmsd_angstrom": (
            float(metrics["kabsch_rmsd_angstrom"]) <= thresholds["kabsch_rmsd_angstrom_max"]
        ),
        "plddt_mean_abs": (float(metrics["plddt_mean_abs"]) <= thresholds["plddt_mean_abs_max"]),
        "confidence_score_abs": (
            float(metrics["confidence_score_abs"]) <= thresholds["confidence_score_abs_max"]
        ),
        "complex_plddt_abs": (
            float(metrics["complex_plddt_abs"]) <= thresholds["complex_plddt_abs_max"]
        ),
        "complex_iplddt_abs": (
            float(metrics["complex_iplddt_abs"]) <= thresholds["complex_iplddt_abs_max"]
        ),
        "ptm_abs": float(metrics["ptm_abs"]) <= thresholds["ptm_abs_max"],
        "iptm_abs": float(metrics["iptm_abs"]) <= thresholds["iptm_abs_max"],
        "protein_iptm_abs": (
            float(metrics["protein_iptm_abs"]) <= thresholds["protein_iptm_abs_max"]
        ),
        "chains_ptm_max_abs": (
            float(metrics["chains_ptm_max_abs"]) <= thresholds["chains_ptm_max_abs_max"]
        ),
        "pair_chains_iptm_max_abs": (
            float(metrics["pair_chains_iptm_max_abs"]) <= thresholds["pair_chains_iptm_max_abs_max"]
        ),
    }
    return {"thresholds": thresholds, "checks": checks, "passed": all(checks.values())}


def compare_native(
    reference_npz: Path,
    candidate_mmcif: Path,
    candidate_metadata: Path,
    output_json: Path,
    *,
    expected_atom_count: int,
    expected_token_count: int,
    enforce: bool = True,
) -> dict[str, Any]:
    """Compare native output with seeded eager output and enforce accuracy gates."""

    if expected_atom_count <= 0 or expected_token_count <= 0:
        raise ValueError("native qualification expected counts must be positive")
    with np.load(reference_npz) as reference:
        reference_coords = _masked_coords(reference)
        reference_plddt = _masked_tokens(reference, "plddt")
        reference_confidence = float(np.asarray(reference["confidence_score"]).reshape(-1)[0])
        reference_complex_plddt = float(np.asarray(reference["complex_plddt"]).reshape(-1)[0])
        reference_complex_iplddt = float(np.asarray(reference["complex_iplddt"]).reshape(-1)[0])
        reference_ptm = float(np.asarray(reference["ptm"]).reshape(-1)[0])
        reference_iptm = float(np.asarray(reference["iptm"]).reshape(-1)[0])
        reference_protein_iptm = float(np.asarray(reference["protein_iptm"]).reshape(-1)[0])
        reference_chain_ids = np.asarray(reference["pair_chain_ids"], dtype=np.int32)
        reference_pair_chains_iptm = np.asarray(reference["pair_chains_iptm"], dtype=np.float64)
    candidate_coords = _native_mmcif_coords(candidate_mmcif)
    metadata = json.loads(candidate_metadata.read_text(encoding="utf-8"))
    candidate_plddt = np.asarray(metadata.get("plddt", []), dtype=np.float64)
    if reference_coords.shape != candidate_coords.shape:
        raise ValueError("reference and native coordinate shapes do not match")
    if reference_plddt.shape != candidate_plddt.shape:
        raise ValueError("reference and native pLDDT shapes do not match")
    if reference_pair_chains_iptm.shape != (reference_chain_ids.size,) * 2:
        raise ValueError("reference pair-chain ipTM shape does not match its chain IDs")
    chain_keys = {str(int(chain_id)) for chain_id in reference_chain_ids}
    candidate_chains = metadata.get("chains_ptm")
    candidate_pairs = metadata.get("pair_chains_iptm")
    if not isinstance(candidate_chains, dict) or set(candidate_chains) != chain_keys:
        raise ValueError("native per-chain pTM keys do not match the reference")
    if not isinstance(candidate_pairs, dict) or set(candidate_pairs) != chain_keys:
        raise ValueError("native pair-chain ipTM keys do not match the reference")
    if metadata.get("chain_pair_confidence") != []:
        raise ValueError("native legacy chain-pair confidence field changed shape")
    candidate_pair_chains_iptm = np.empty_like(reference_pair_chains_iptm)
    candidate_chains_ptm = np.empty(reference_chain_ids.size, dtype=np.float64)
    for first_index, first in enumerate(reference_chain_ids):
        first_key = str(int(first))
        row = candidate_pairs[first_key]
        if not isinstance(row, dict) or set(row) != chain_keys:
            raise ValueError("native pair-chain ipTM map is not square")
        candidate_chains_ptm[first_index] = float(candidate_chains[first_key])
        for second_index, second in enumerate(reference_chain_ids):
            candidate_pair_chains_iptm[first_index, second_index] = float(row[str(int(second))])
    result = {
        "schema_version": 1,
        "atom_count": int(candidate_coords.shape[0]),
        "token_count": int(candidate_plddt.size),
        "all_outputs_finite": bool(
            np.isfinite(reference_coords).all()
            and np.isfinite(candidate_coords).all()
            and np.isfinite(reference_plddt).all()
            and np.isfinite(candidate_plddt).all()
            and np.isfinite(reference_pair_chains_iptm).all()
            and np.isfinite(candidate_pair_chains_iptm).all()
            and np.isfinite(candidate_chains_ptm).all()
        ),
        "lddt": _lddt(reference_coords, candidate_coords),
        "kabsch_rmsd_angstrom": _kabsch_rmsd(reference_coords, candidate_coords),
        "plddt_max_abs": float(np.max(np.abs(reference_plddt - candidate_plddt))),
        "plddt_mean_abs": float(np.mean(np.abs(reference_plddt - candidate_plddt))),
        "confidence_score_abs": abs(reference_confidence - float(metadata["confidence_score"])),
        "complex_plddt_abs": abs(reference_complex_plddt - float(metadata["complex_plddt"])),
        "complex_iplddt_abs": abs(reference_complex_iplddt - float(metadata["complex_iplddt"])),
        "ptm_abs": abs(reference_ptm - float(metadata["ptm"])),
        "iptm_abs": abs(reference_iptm - float(metadata["iptm"])),
        "protein_iptm_abs": abs(reference_protein_iptm - float(metadata["protein_iptm"])),
        "chains_ptm_max_abs": float(
            np.max(np.abs(np.diag(reference_pair_chains_iptm) - candidate_chains_ptm))
        ),
        "pair_chains_iptm_max_abs": float(
            np.max(np.abs(reference_pair_chains_iptm - candidate_pair_chains_iptm))
        ),
    }
    numeric = (
        value
        for key, value in result.items()
        if key not in {"schema_version", "atom_count", "token_count", "all_outputs_finite"}
    )
    if not result["all_outputs_finite"] or not all(
        math.isfinite(float(value)) for value in numeric
    ):
        raise ValueError("native Boltz-2 qualification metrics are non-finite")
    result["qualification"] = _qualification(
        result,
        expected_atom_count=expected_atom_count,
        expected_token_count=expected_token_count,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if enforce and not result["qualification"]["passed"]:
        failed = ", ".join(
            name for name, passed in result["qualification"]["checks"].items() if not passed
        )
        raise RuntimeError(f"Boltz-2 native qualification failed: {failed}")
    return result
