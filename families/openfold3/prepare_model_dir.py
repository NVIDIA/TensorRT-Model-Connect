# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the external, pinned OpenFold3 preprocessing artifacts for a bundle."""

from __future__ import annotations

import argparse
import io
import json
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .atom_attention_builder import padded_atom_count
from .checkpoint import validate_artifact
from .contracts import parse_query_json
from .feature_bundle import FEATURE_NAMES, profile_feature_shapes
from .provenance import PINNED_OPENFOLD3


def _numpy(value, *, dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _pad_atoms(value, padded_atoms: int) -> np.ndarray:
    array = np.asarray(value)
    padding = [(0, 0)] * array.ndim
    padding[1] = (0, padded_atoms - array.shape[1])
    return np.pad(array, padding, mode="constant")


def _representative_indices(batch, atom_array) -> np.ndarray:
    starts = _numpy(batch["start_atom_index"], dtype=np.int32)[0]
    counts = _numpy(batch["num_atoms_per_token"], dtype=np.int32)[0]
    restype = _numpy(batch["restype"], dtype=np.float32)[0]
    names = np.asarray(atom_array.atom_name)
    glycine_index = 7
    result = []
    for token, (start, count) in enumerate(zip(starts, counts, strict=True)):
        desired = "CA" if restype[token, glycine_index] else "CB"
        local = np.flatnonzero(names[start : start + count] == desired)
        if local.size != 1:
            raise ValueError(f"OpenFold3 token {token} has no unique {desired} atom")
        result.append(int(start + local[0]))
    return np.asarray(result, dtype=np.int32)


def _structure_metadata(atom_array) -> bytes:
    document = {
        "schema_version": 1,
        "atom_count": len(atom_array),
        "atoms": [
            {
                "name": str(atom_array.atom_name[index]),
                "element": str(atom_array.element[index]),
                "residue_name": str(atom_array.res_name[index]),
                "residue_index": int(atom_array.res_id[index]),
                "chain_id": str(atom_array.chain_id[index]),
                "hetero": bool(atom_array.hetero[index]),
            }
            for index in range(len(atom_array))
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _validate_disabled_template_features(batch, token_count: int) -> None:
    """Lock the v0.5.0 dummy-template convention embedded by the TRT graph."""

    restype = _numpy(batch["template_restype"], dtype=np.int32)
    expected_shape = (1, 4, token_count, 32)
    if restype.shape != expected_shape:
        raise ValueError(
            f"OpenFold3 disabled-template shape {restype.shape} differs from {expected_shape}"
        )
    if not np.all(restype[..., 31] == 1) or int(restype.sum()) != 4 * token_count:
        raise ValueError("OpenFold3 disabled-template residue convention changed upstream")
    for name in (
        "template_backbone_frame_mask",
        "template_distogram",
        "template_pseudo_beta_mask",
        "template_unit_vector",
    ):
        if np.any(_numpy(batch[name], dtype=np.float32)):
            raise ValueError(f"OpenFold3 disabled-template feature {name!r} is not zero")


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write pickle-free NumPy members with stable order and ZIP metadata."""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in FEATURE_NAMES:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def _normalized_preprocessing_query(request_bytes: bytes, directory: Path) -> Path:
    """Make the pinned upstream shorthand explicit for OpenFold3 featurization."""

    document = json.loads(request_bytes)
    document["seeds"] = [42]
    raw_query = next(iter(document["queries"].values()))
    raw_query["use_msas"] = False
    raw_query["use_main_msas"] = False
    raw_query["use_paired_msas"] = False
    for chain in raw_query["chains"]:
        chain["molecule_type"] = str(chain["molecule_type"]).upper()
    path = directory / "query.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def prepare(query_path: Path, components_path: Path, output_dir: Path) -> None:
    """Run v0.5.0 featurization and emit a pickle-free request package."""

    validate_artifact(components_path, PINNED_OPENFOLD3.chemical_components)
    request_bytes = query_path.read_bytes()
    request = parse_query_json(request_bytes.decode("utf-8"))

    from biotite.structure.info.ccd import set_ccd_path

    # The upstream dataset wrapper only accepts text CIF through ccd_file_path;
    # point Biotite's binary CCD reader at the pinned BCIF artifact instead.
    set_ccd_path(components_path)

    from openfold3.core.data.framework.data_module import (
        DataModuleConfig,
        InferenceDataModule,
    )
    from openfold3.core.data.pipelines.preprocessing.template import (
        TemplatePreprocessorSettings,
    )
    from openfold3.core.data.tools.colabfold_msa_server import MsaComputationSettings
    from openfold3.core.utils.relpos import relpos_complex
    from openfold3.projects.of3_all_atom.config.dataset_configs import (
        InferenceDatasetSpec,
        InferenceJobConfig,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory(prefix="trtmc-openfold3-query-") as directory:
        normalized_query = _normalized_preprocessing_query(request_bytes, Path(directory))
        query_set = InferenceQuerySet.from_json(normalized_query)
    job = InferenceJobConfig(
        query_set=query_set,
        seeds=[42],
        template_preprocessor_settings=TemplatePreprocessorSettings(mode="predict"),
    )
    module = InferenceDataModule(
        DataModuleConfig(
            datasets=[InferenceDatasetSpec(config=job)],
            batch_size=1,
            epoch_len=1,
            num_workers=0,
        ),
        use_msa_server=False,
        use_templates=False,
        msa_computation_settings=MsaComputationSettings(),
    )
    module.prepare_data()
    module.setup()
    batch = next(iter(module.predict_dataloader()))
    if not bool(batch["valid_sample"].item()):
        raise RuntimeError("OpenFold3 rejected the request during preprocessing")
    atom_array = batch["atom_array"][0]
    atom_count = int(batch["atom_mask"].shape[1])
    token_count = int(batch["token_mask"].shape[1])
    msa_depth = int(batch["msa"].shape[1])
    padded_atoms = padded_atom_count(atom_count)
    if token_count != request.token_count:
        raise ValueError("OpenFold3 preprocessing changed the request token count")
    _validate_disabled_template_features(batch, token_count)

    features = {
        name: _pad_atoms(batch[name], padded_atoms)
        for name in (
            "ref_pos",
            "ref_mask",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
            "atom_mask",
            "atom_to_token_index",
        )
    }
    for name in (
        "token_mask",
        "restype",
        "profile",
        "deletion_mean",
        "token_bonds",
        "msa",
        "has_deletion",
        "deletion_value",
        "msa_mask",
    ):
        features[name] = batch[name]
    features["relpos"] = relpos_complex(batch, 32, 2)
    representative = _representative_indices(batch, atom_array)
    representative_map = np.zeros((1, token_count, atom_count), np.float32)
    representative_map[0, np.arange(token_count), representative] = 1.0
    features["representative_atom_map"] = representative_map
    token_indices = _numpy(batch["atom_to_token_index"], dtype=np.int32)[0]
    slots = np.zeros(atom_count, np.int32)
    counts = np.zeros(token_count, np.int32)
    for atom, token in enumerate(token_indices):
        slots[atom] = counts[token]
        counts[token] += 1
    if int(slots.max()) >= 23:
        raise ValueError("OpenFold3 request exceeds 23 atoms per token")
    features["atom_head_index"] = token_indices * 23 + slots

    int_names = {"ref_space_uid", "atom_to_token_index", "atom_head_index"}
    serializable = {
        name: _numpy(features[name], dtype=np.int32 if name in int_names else np.float32)
        for name in FEATURE_NAMES
    }
    expected = profile_feature_shapes(token_count, atom_count, padded_atoms, msa_depth)
    for name, shape in expected.items():
        if serializable[name].shape != shape:
            raise ValueError(
                f"OpenFold3 feature {name!r} has shape {serializable[name].shape}, expected {shape}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(output_dir / "openfold3_features.npz", serializable)
    (output_dir / "query.json").write_bytes(request_bytes)
    (output_dir / "openfold3_structure.json").write_bytes(_structure_metadata(atom_array))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.query, arguments.components, arguments.output_dir)


if __name__ == "__main__":
    main()
