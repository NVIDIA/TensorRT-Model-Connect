# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native host arithmetic against the pinned reference, on the CPU.

``test_esmfold2_host --dump`` (built with the family's CTest) evaluates featurization, the random streams, the noise
schedule, the rigid augmentation and alignment, the pTM weights and the PDB writer; each result is compared with the
transformers ESMFold2 port or with the NumPy twin of the streams that the reference E2E uses.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from . import family_rng

SEQUENCES = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "ACDEFGHIKLMNPQRSTVWYX",
    "GW",
)
SAMPLER = {
    "sigma_data": 16.0, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "step_scale": 1.5,
    "sigma_max_ratio": 160.0, "sigma_min_ratio": 0.0004, "exponent": 7.0, "sigma_cap": 256.0,
}


@pytest.fixture(scope="module")
def host_binary() -> Path:
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert build, "the ESMFold2 host comparison requires TRTMC_NATIVE_BUILD_DIR"
    binary = Path(build) / "test_esmfold2_host"
    assert binary.is_file(), f"build the family CTest target test_esmfold2_host in {build}"
    return binary


def _dump(binary: Path, jobs: list[dict], tmp_path: Path) -> list[dict]:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(jobs), encoding="utf-8")
    completed = subprocess.run([str(binary), "--dump", str(path)], check=True, capture_output=True, text=True,
                               timeout=600)
    results = json.loads(completed.stdout)
    assert len(results) == len(jobs)
    for job, result in zip(jobs, results):
        assert result["ok"], (job["kind"], result.get("error"))
    return [result["result"] for result in results]


def _mixin():
    from transformers.models.esmfold2.configuration_esmfold2 import EsmFold2Config
    from transformers.models.esmfold2.generation_esmfold2 import EsmFold2FoldingMixin

    config = EsmFold2Config(
        structure_head={
            "gamma_0": SAMPLER["gamma_0"], "gamma_min": SAMPLER["gamma_min"], "noise_scale": SAMPLER["noise_scale"],
            "step_scale": SAMPLER["step_scale"], "inference_sigma_max_ratio": SAMPLER["sigma_max_ratio"],
            "inference_sigma_min_ratio": SAMPLER["sigma_min_ratio"], "inference_exponent": SAMPLER["exponent"],
            "inference_num_steps": 14, "inference_sigma_cap": SAMPLER["sigma_cap"],
            "diffusion_module": {"sigma_data": SAMPLER["sigma_data"]},
        }
    )

    class Sampler(EsmFold2FoldingMixin):
        def __init__(self) -> None:
            self.config = config

    return Sampler()


def test_featurization_matches_the_reference(host_binary: Path, tmp_path: Path) -> None:
    from transformers.models.esmfold2.protein_utils import prepare_protein_features

    jobs = [{"kind": "features", "document": f">case\n{sequence}\n"} for sequence in SEQUENCES]
    for sequence, native in zip(SEQUENCES, _dump(host_binary, jobs, tmp_path)):
        reference = {key: value[0].numpy() for key, value in prepare_protein_features(sequence).items()}
        atoms = reference["ref_pos"].shape[0]
        assert native["atoms"] == atoms and native["sequence"] == sequence
        ids = np.asarray(native["input_ids"])
        assert ids[0] == 0 and ids[-1] == 2 and np.array_equal(ids[1:-1], reference["input_ids"])
        for name, ref_name in (("res_type", "res_type"), ("residue_index", "residue_index"),
                               ("rep_atom", "distogram_atom_idx"), ("ref_element", "ref_element"),
                               ("ref_space_uid", "ref_space_uid"), ("atom_to_token", "atom_to_token")):
            assert np.array_equal(np.asarray(native[name]), reference[ref_name]), name
        assert np.array_equal(np.asarray(native["ref_pos"], np.float32).reshape(-1, 3), reference["ref_pos"])
        assert np.array_equal(np.asarray(native["ref_charge"]), reference["ref_charge"].astype(np.float32))
        assert np.array_equal(np.asarray(native["atom_mask"]) > 0.5, reference["atom_attention_mask"])
        chars = np.asarray(native["ref_name_chars"]).reshape(-1, 4)
        assert np.array_equal(chars, reference["ref_atom_name_chars"])
        # The confidence head's slot index: a running count that restarts at each token (padding is never read).
        real = int(native["real_atoms"])
        owner = reference["atom_to_token"][:real]
        expected = np.concatenate([np.arange((owner == token).sum()) for token in np.unique(owner)])
        assert np.array_equal(np.asarray(native["intra_index"])[:real], expected)


def test_random_streams_match_the_python_twin(host_binary: Path, tmp_path: Path) -> None:
    jobs, expected = [], []
    for seed, stream, first in ((0, family_rng.COORDINATES, 0), (7, family_rng.CHURN, 123456789), (2**40, 1, 5)):
        indices = np.arange(first, first + 64, dtype=np.uint64)
        base = {"kind": "random", "seed": seed, "stream": stream, "first": first, "count": 64}
        jobs += [{**base, "draw": "bits"}, {**base, "draw": "normal"},
                 {**base, "draw": "truncated", "std": 0.0395}, {**base, "draw": "keep", "dropout": 0.25}]
        expected += [family_rng.bits(seed, stream, indices), family_rng.normal(seed, stream, indices),
                     family_rng.truncated_normal(seed, stream, indices, 0.0395),
                     family_rng.keep(seed, stream, indices, 0.25).astype(np.int64)]
    for job, native, twin in zip(jobs, _dump(host_binary, jobs, tmp_path), expected):
        values = native["values"]
        if job["draw"] == "bits":
            assert [int(value) for value in values] == [int(value) for value in twin]
        elif job["draw"] == "keep":
            assert values == twin.tolist()
        else:
            # Box-Muller in double precision on both sides; libm and NumPy may differ in the last double bit.
            np.testing.assert_allclose(np.asarray(values, np.float32), twin, rtol=2.0e-7, atol=1.0e-7)


def test_noise_schedule_matches_the_reference(host_binary: Path, tmp_path: Path) -> None:
    sampler = _mixin()
    for steps in (1, 2, 14, 50):
        native = _dump(host_binary, [{"kind": "schedule", **SAMPLER, "steps": steps}], tmp_path)[0]
        schedule, gammas = sampler._build_noise_schedule(steps, "cpu")
        # torch's float32 operations in torch's order: only pow may differ (SLEEF vectors against libm), by an
        # ulp. A base rounded otherwise (a product fused into the addition) moves a sigma by 3 ulps or more,
        # since pow(base, 7) multiplies its relative error by 7.
        sigmas = np.asarray(native["sigmas"], dtype=np.float32)
        reference = schedule.numpy()
        assert sigmas.shape == reference.shape
        ulps = np.abs(sigmas.view(np.int32).astype(np.int64) - reference.view(np.int32).astype(np.int64))
        assert ulps.max() <= 2, (steps, ulps.tolist())
        assert native["gammas"] == pytest.approx(gammas.numpy().tolist())
        # The reference's step formulas on the schedule the native sampler walks: mixing in the reference's
        # sigma would turn its last ulp into churn at the steps without any (a difference of near-equal squares).
        assert len(native["levels"]) == len(sigmas) - 1
        for index, (noise_level, churn, sigma_next) in enumerate(native["levels"]):
            sigma_prev = float(sigmas[index])
            gamma = gammas[index + 1].item()
            assert noise_level == sigma_prev * (1.0 + gamma)
            if gamma == 0.0:
                assert churn == 0.0, (steps, index)
            assert churn == pytest.approx(SAMPLER["noise_scale"] * max(noise_level**2 - sigma_prev**2, 0.0) ** 0.5,
                                          rel=1.0e-5, abs=1.0e-6)
            assert sigma_next == float(sigmas[index + 1])
    assert len(sampler._build_noise_schedule(14, "cpu")[0]) == 11


def test_rigid_steps_match_the_reference(host_binary: Path, tmp_path: Path) -> None:
    import torch
    from transformers.models.esmfold2 import generation_esmfold2 as generation

    generator = np.random.default_rng(3)
    samples, atoms = 3, 64
    coords = generator.normal(scale=10.0, size=(samples, atoms, 3)).astype(np.float32)
    mask = np.ones(atoms, np.float32)
    mask[-5:] = 0.0
    quaternions = generator.normal(size=(samples, 4)).astype(np.float32)
    translations = generator.normal(size=(samples, 1, 3)).astype(np.float32)
    native = _dump(host_binary, [{"kind": "augment", "coords": coords.ravel().tolist(), "samples": samples,
                                  "atoms": atoms, "mask": mask.tolist(), "quaternions": quaternions.tolist(),
                                  "translations": translations.ravel().tolist()}], tmp_path)[0]
    original_randn, original_like = torch.randn, torch.randn_like
    torch.randn = lambda *a, **k: torch.from_numpy(quaternions.copy())
    torch.randn_like = lambda *a, **k: torch.from_numpy(translations.copy())
    try:
        augmented, _ = generation._center_random_augmentation(
            torch.from_numpy(coords.copy()), torch.from_numpy(mask).expand(samples, atoms)
        )
    finally:
        torch.randn, torch.randn_like = original_randn, original_like
    np.testing.assert_allclose(np.asarray(native["coords"]).reshape(samples, atoms, 3), augmented.numpy(),
                               rtol=0, atol=2.0e-5)

    target = coords + generator.normal(scale=0.3, size=coords.shape).astype(np.float32)
    aligned = _dump(host_binary, [{"kind": "align", "coords": coords.ravel().tolist(),
                                   "target": target.ravel().tolist(), "samples": samples, "atoms": atoms,
                                   "mask": mask.tolist()}], tmp_path)[0]
    weights = torch.from_numpy(mask).expand(samples, atoms)
    reference = generation._weighted_rigid_align(torch.from_numpy(coords), torch.from_numpy(target), weights,
                                                 weights)
    real = mask > 0.5
    native_aligned = np.asarray(aligned["coords"]).reshape(samples, atoms, 3)
    np.testing.assert_allclose(native_aligned[:, real], reference.numpy()[:, real], rtol=0, atol=5.0e-5)


def test_ptm_weights_match_the_reference(host_binary: Path, tmp_path: Path) -> None:
    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2ConfidenceHead

    for tokens in (2, 19, 76, 300):
        native = np.asarray(_dump(host_binary, [{"kind": "tm", "tokens": tokens, "bins": 64}], tmp_path)[0]["weights"])
        # One-hot PAE distributions make the reference's pTM equal to one bin's TM weight.
        for bin_index in (0, 13, 63):
            logits = torch.full((1, tokens, tokens, 64), -1.0e4)
            logits[..., bin_index] = 1.0e4
            ptm, _, _ = EsmFold2ConfidenceHead._compute_ptm_iptm(
                type("Head", (), {"eps": 1.0e-6})(), logits, torch.ones(1, tokens, dtype=torch.bool),
                torch.zeros(1, tokens, dtype=torch.long), 1,
            )
            assert float(ptm[0]) == pytest.approx(native[bin_index] * tokens / (tokens + 1.0e-6), rel=1.0e-6)


def test_pdb_matches_the_reference_writer(host_binary: Path, tmp_path: Path) -> None:
    import torch
    from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

    generator = np.random.default_rng(5)
    for sequence in SEQUENCES:
        features = prepare_protein_features(sequence)
        atoms = int(features["ref_pos"].shape[1])
        coords = (generator.normal(scale=20.0, size=(atoms, 3))).astype(np.float32)
        plddt = generator.uniform(size=len(sequence)).astype(np.float32)
        native = _dump(host_binary, [{"kind": "pdb", "sequence": sequence, "coords": coords.ravel().tolist(),
                                      "plddt": plddt.tolist()}], tmp_path)[0]["text"]
        output = {"sample_atom_coords": torch.from_numpy(coords)[None], "plddt": torch.from_numpy(plddt)[None]}
        assert native == output_to_pdb(output, features, sample_idx=0)


def test_reference_numerics_are_restored_after_a_failure() -> None:
    import torch

    from .reference import reference_numerics

    def state() -> tuple:
        return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.cudnn_sdp_enabled())

    before = state()
    with pytest.raises(RuntimeError, match="reference failed"):
        with reference_numerics():
            assert state() == ("highest", False, False, False)
            raise RuntimeError("reference failed")
    assert state() == before


def test_the_featurizer_bounds_match_the_reference_tables() -> None:
    from transformers.models.esmfold2 import protein_utils

    from ..config import (
        FEATURIZER_MAX_ABS_COORDINATE,
        FEATURIZER_MAX_ELEMENT,
        FEATURIZER_MAX_NAME_CHAR,
        FEATURIZER_MAX_RES_TYPE,
        FEATURIZER_MAX_TOKEN_ID,
        FEATURIZER_NAME_CHARS,
    )

    names = [name for atoms in protein_utils.PROTEIN_HEAVY_ATOMS.values() for name in atoms]
    res_types = [*protein_utils.PROTEIN_RESIDUE_TO_RES_TYPE.values(), protein_utils.PROTEIN_UNK_RES_TYPE]
    assert max(res_types) == FEATURIZER_MAX_RES_TYPE
    assert max(len(name) for name in names) <= FEATURIZER_NAME_CHARS
    assert max(ord(char) - 32 for name in names for char in name) == FEATURIZER_MAX_NAME_CHAR
    atomic_numbers = {"C": 6, "N": 7, "O": 8, "S": 16}
    assert {name[0] for name in names} <= set(atomic_numbers)
    assert max(atomic_numbers[name[0]] for name in names) == FEATURIZER_MAX_ELEMENT
    assert max(protein_utils.ESM_PROTEIN_VOCAB.values()) == FEATURIZER_MAX_TOKEN_ID
    coordinates = [abs(float(np.float32(value))) for atoms in protein_utils.PROTEIN_REF_POS.values()
                   for position in (atoms.values() if isinstance(atoms, dict) else atoms) for value in position]
    assert max(coordinates) == FEATURIZER_MAX_ABS_COORDINATE
