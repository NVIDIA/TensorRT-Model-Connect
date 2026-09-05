# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable feature section shared by OpenFold3 bundle construction and runtime."""

from __future__ import annotations

import io
import struct
from collections.abc import Mapping
from typing import Any, Final

import numpy as np


MAGIC: Final = b"OF3F"
VERSION: Final = 1
DTYPE_TO_CODE: Final = {np.dtype("float32"): 1, np.dtype("int32"): 2}
CODE_TO_DTYPE: Final = {value: key for key, value in DTYPE_TO_CODE.items()}
INT32_FEATURE_NAMES: Final = frozenset({"ref_space_uid", "atom_to_token_index", "atom_head_index"})
FEATURE_NAMES: Final = (
    "ref_pos",
    "ref_mask",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_mask",
    "atom_to_token_index",
    "token_mask",
    "restype",
    "profile",
    "deletion_mean",
    "relpos",
    "token_bonds",
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_mask",
    "representative_atom_map",
    "atom_head_index",
)


def profile_feature_shapes(
    token_count: int, atom_count: int, padded_atom_count: int, msa_depth: int
) -> dict[str, tuple[int, ...]]:
    """Return the complete static request profile consumed by native inference."""

    if min(token_count, atom_count, padded_atom_count, msa_depth) <= 0:
        raise ValueError("OpenFold3 profile dimensions must be positive")
    if padded_atom_count < atom_count or padded_atom_count % 32:
        raise ValueError("OpenFold3 padded atom count must cover atoms in 32-atom blocks")
    return {
        "ref_pos": (1, padded_atom_count, 3),
        "ref_mask": (1, padded_atom_count),
        "ref_element": (1, padded_atom_count, 119),
        "ref_charge": (1, padded_atom_count),
        "ref_atom_name_chars": (1, padded_atom_count, 4, 64),
        "ref_space_uid": (1, padded_atom_count),
        "atom_mask": (1, padded_atom_count),
        "atom_to_token_index": (1, padded_atom_count),
        "token_mask": (1, token_count),
        "restype": (1, token_count, 32),
        "profile": (1, token_count, 32),
        "deletion_mean": (1, token_count),
        "relpos": (1, token_count, token_count, 139),
        "token_bonds": (1, token_count, token_count),
        "msa": (1, msa_depth, token_count, 32),
        "has_deletion": (1, msa_depth, token_count),
        "deletion_value": (1, msa_depth, token_count),
        "msa_mask": (1, msa_depth, token_count),
        "representative_atom_map": (1, token_count, atom_count),
        "atom_head_index": (atom_count,),
    }


def _numpy(name: str, value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if name in INT32_FEATURE_NAMES:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"OpenFold3 categorical feature {name!r} must be integer")
        if array.size and (
            int(array.min()) < np.iinfo(np.int32).min or int(array.max()) > np.iinfo(np.int32).max
        ):
            raise ValueError(f"OpenFold3 categorical feature {name!r} exceeds INT32")
        array = array.astype(np.int32)
    else:
        array = array.astype(np.float32)
    if array.dtype not in DTYPE_TO_CODE:
        raise ValueError(f"unsupported OpenFold3 feature dtype: {array.dtype}")
    return np.ascontiguousarray(array)


def serialize_features(features: Mapping[str, Any]) -> bytes:
    """Serialize exactly the tensors required after external preprocessing."""

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"OpenFold3 feature set is missing: {', '.join(missing)}")
    output = io.BytesIO()
    output.write(MAGIC)
    output.write(struct.pack("<II", VERSION, len(FEATURE_NAMES)))
    for name in FEATURE_NAMES:
        encoded = name.encode("utf-8")
        array = _numpy(name, features[name])
        output.write(struct.pack("<H", len(encoded)))
        output.write(encoded)
        output.write(struct.pack("<BB", DTYPE_TO_CODE[array.dtype], array.ndim))
        output.write(struct.pack(f"<{array.ndim}I", *array.shape))
        output.write(struct.pack("<Q", array.nbytes))
        output.write(array.tobytes(order="C"))
    return output.getvalue()


def deserialize_features(payload: bytes) -> dict[str, np.ndarray]:
    """Decode a feature section for validation tests and bundle inspection."""

    source = io.BytesIO(payload)
    if source.read(4) != MAGIC:
        raise ValueError("invalid OpenFold3 feature section magic")
    version, count = struct.unpack("<II", source.read(8))
    if version != VERSION or count != len(FEATURE_NAMES):
        raise ValueError("unsupported OpenFold3 feature section header")
    result: dict[str, np.ndarray] = {}
    for _ in range(count):
        (name_size,) = struct.unpack("<H", source.read(2))
        name = source.read(name_size).decode("utf-8")
        code, rank = struct.unpack("<BB", source.read(2))
        if code not in CODE_TO_DTYPE:
            raise ValueError("unsupported OpenFold3 feature dtype code")
        shape = struct.unpack(f"<{rank}I", source.read(4 * rank))
        (size,) = struct.unpack("<Q", source.read(8))
        data = source.read(size)
        if len(data) != size:
            raise ValueError(f"truncated OpenFold3 feature payload: {name}")
        result[name] = np.frombuffer(data, CODE_TO_DTYPE[code]).reshape(shape)
    if source.read(1) or tuple(result) != FEATURE_NAMES:
        raise ValueError("OpenFold3 feature section order or extent is invalid")
    return result


def load_npz_features(path) -> dict[str, np.ndarray]:
    """Load a closed, pickle-free prepared feature archive."""

    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in FEATURE_NAMES if name not in archive]
        if missing:
            raise ValueError(f"OpenFold3 feature archive is missing: {', '.join(missing)}")
        return {name: _numpy(name, archive[name]) for name in FEATURE_NAMES}
