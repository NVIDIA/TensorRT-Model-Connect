# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable native feature and structure-metadata sections for Boltz-2 bundles."""

from __future__ import annotations

import io
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np


MAGIC: Final = b"B2FT"
VERSION: Final = 1
DTYPE_TO_CODE: Final = {
    np.dtype("float32"): 1,
    np.dtype("int32"): 2,
    np.dtype("bool"): 3,
}
CODE_TO_DTYPE: Final = {value: key for key, value in DTYPE_TO_CODE.items()}
INT32_FEATURE_NAMES: Final = frozenset(
    {
        "ref_space_uid",
        "ref_element",
        "ref_atom_name_chars",
        "atom_to_token",
        "res_type",
        "method_feature",
        "modified",
        "mol_type",
        "asym_id",
        "residue_index",
        "entity_id",
        "token_index",
        "sym_id",
        "type_bonds",
        "contact_conditioning",
        "msa",
        "has_deletion",
        "msa_mask",
        "token_to_rep_atom",
        "frames_idx",
    }
)
FEATURE_NAMES: Final = (
    "ref_pos",
    "ref_space_uid",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "atom_to_token",
    "atom_pad_mask",
    "res_type",
    "profile",
    "deletion_mean",
    "method_feature",
    "modified",
    "cyclic_period",
    "mol_type",
    "asym_id",
    "residue_index",
    "entity_id",
    "token_index",
    "sym_id",
    "token_bonds",
    "type_bonds",
    "contact_conditioning",
    "contact_threshold",
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_paired",
    "msa_mask",
    "token_pad_mask",
    "token_to_rep_atom",
    "frames_idx",
)


def profile_feature_shapes(
    token_count: int, atom_count: int, msa_depth: int
) -> dict[str, tuple[int, ...]]:
    """Return the complete runtime feature contract for one reusable profile."""

    if token_count <= 0 or atom_count <= 0 or msa_depth <= 0:
        raise ValueError("Boltz-2 profile dimensions must be positive")
    return {
        "ref_pos": (1, atom_count, 3),
        "ref_space_uid": (1, atom_count),
        "ref_charge": (1, atom_count),
        "ref_element": (1, atom_count, 128),
        "ref_atom_name_chars": (1, atom_count, 4, 64),
        "atom_to_token": (1, atom_count, token_count),
        "atom_pad_mask": (1, atom_count),
        "res_type": (1, token_count, 33),
        "profile": (1, token_count, 33),
        "deletion_mean": (1, token_count),
        "method_feature": (1, token_count),
        "modified": (1, token_count),
        "cyclic_period": (1, token_count),
        "mol_type": (1, token_count),
        "asym_id": (1, token_count),
        "residue_index": (1, token_count),
        "entity_id": (1, token_count),
        "token_index": (1, token_count),
        "sym_id": (1, token_count),
        "token_bonds": (1, token_count, token_count, 1),
        "type_bonds": (1, token_count, token_count),
        "contact_conditioning": (1, token_count, token_count, 5),
        "contact_threshold": (1, token_count, token_count),
        "msa": (1, msa_depth, token_count),
        "has_deletion": (1, msa_depth, token_count),
        "deletion_value": (1, msa_depth, token_count),
        "msa_paired": (1, msa_depth, token_count),
        "msa_mask": (1, msa_depth, token_count),
        "token_pad_mask": (1, token_count),
        "token_to_rep_atom": (1, token_count, atom_count),
        "frames_idx": (1, 1, token_count, 3),
    }


def _numpy(name: str, value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if name in INT32_FEATURE_NAMES:
        if not np.issubdtype(array.dtype, np.integer) and array.dtype != np.dtype("bool"):
            raise ValueError(f"Boltz-2 categorical feature {name!r} must be integer-valued")
        if array.size:
            info = np.iinfo(np.int32)
            minimum = int(array.min())
            maximum = int(array.max())
            if minimum < info.min or maximum > info.max:
                raise ValueError(
                    "Boltz-2 categorical feature value is outside the INT32 range: "
                    f"[{minimum}, {maximum}]"
                )
        array = array.astype(np.int32)
    elif array.dtype != np.dtype("float32"):
        raise ValueError(f"Boltz-2 continuous feature {name!r} must use float32")
    dtype = np.dtype(array.dtype)
    if dtype not in DTYPE_TO_CODE:
        raise ValueError(f"unsupported Boltz-2 feature dtype: {dtype}")
    return np.ascontiguousarray(array)


def serialize_features(features: Mapping[str, Any]) -> bytes:
    """Serialize exactly the tensors consumed by one native static profile."""

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Boltz-2 feature set is missing: {', '.join(missing)}")
    output = io.BytesIO()
    output.write(MAGIC)
    output.write(struct.pack("<II", VERSION, len(FEATURE_NAMES)))
    for name in FEATURE_NAMES:
        encoded_name = name.encode("utf-8")
        if len(encoded_name) > 0xFFFF:
            raise ValueError("Boltz-2 feature name is too long")
        array = _numpy(name, features[name])
        output.write(struct.pack("<H", len(encoded_name)))
        output.write(encoded_name)
        output.write(struct.pack("<BB", DTYPE_TO_CODE[array.dtype], array.ndim))
        output.write(struct.pack(f"<{array.ndim}I", *array.shape))
        output.write(struct.pack("<Q", array.nbytes))
        output.write(array.tobytes(order="C"))
    return output.getvalue()


def deserialize_features(data: bytes) -> dict[str, np.ndarray]:
    """Reference decoder used by tests; the native runtime owns its C++ peer."""

    source = io.BytesIO(data)
    if source.read(4) != MAGIC:
        raise ValueError("invalid Boltz-2 feature section magic")
    version, count = struct.unpack("<II", source.read(8))
    if version != VERSION:
        raise ValueError(f"unsupported Boltz-2 feature section version: {version}")
    result: dict[str, np.ndarray] = {}
    for _ in range(count):
        (name_size,) = struct.unpack("<H", source.read(2))
        name = source.read(name_size).decode("utf-8")
        dtype_code, rank = struct.unpack("<BB", source.read(2))
        if dtype_code not in CODE_TO_DTYPE:
            raise ValueError(f"unsupported Boltz-2 feature dtype code: {dtype_code}")
        shape = struct.unpack(f"<{rank}I", source.read(4 * rank))
        (size,) = struct.unpack("<Q", source.read(8))
        payload = source.read(size)
        if len(payload) != size:
            raise ValueError(f"truncated Boltz-2 feature payload: {name}")
        result[name] = np.frombuffer(payload, dtype=CODE_TO_DTYPE[dtype_code]).reshape(shape)
    if source.read(1):
        raise ValueError("Boltz-2 feature section has trailing bytes")
    if tuple(result) != FEATURE_NAMES:
        raise ValueError("Boltz-2 feature section tensor order differs from its contract")
    return result


def structure_metadata_json(structure_path: Path) -> bytes:
    """Serialize atom/residue rows needed for native mmCIF/PDB writing."""

    with np.load(structure_path, allow_pickle=False) as archive:
        atoms = archive["atoms"]
        residues = archive["residues"]
        chains = archive["chains"]
    document = {
        "schema_version": 1,
        "atoms": [str(value) for value in atoms["name"]],
        "residues": [
            {
                "name": str(row["name"]),
                "index": int(row["res_idx"]) + 1,
                "atom_index": int(row["atom_idx"]),
                "atom_count": int(row["atom_num"]),
            }
            for row in residues
        ],
        "chains": [
            {
                "name": str(row["name"]),
                "atom_index": int(row["atom_idx"]),
                "atom_count": int(row["atom_num"]),
                "residue_index": int(row["res_idx"]),
                "residue_count": int(row["res_num"]),
            }
            for row in chains
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
