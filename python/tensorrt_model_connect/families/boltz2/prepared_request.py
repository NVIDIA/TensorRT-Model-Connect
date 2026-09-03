# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable request-time feature envelope for reusable Boltz-2 bundles."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Final

from .feature_bundle import deserialize_features, serialize_features


MAGIC: Final = b"B2RQ"
VERSION: Final = 2
_HEADER: Final = struct.Struct("<4sIQQQQ")


@dataclass(frozen=True)
class PreparedRequest:
    request: bytes
    features: dict
    random_samples: bytes
    structure_metadata: bytes


def serialize_prepared_request(
    request: bytes, features: dict, random_samples: bytes, structure_metadata: bytes
) -> bytes:
    """Serialize request-specific artifacts independently of TensorRT plans."""

    if not request:
        raise ValueError("Boltz-2 prepared request document must not be empty")
    if not structure_metadata:
        raise ValueError("Boltz-2 prepared structure metadata must not be empty")
    if not random_samples.startswith(b"B2RN"):
        raise ValueError("Boltz-2 prepared random samples are invalid")
    feature_payload = serialize_features(features)
    return b"".join(
        (
            _HEADER.pack(
                MAGIC,
                VERSION,
                len(request),
                len(feature_payload),
                len(random_samples),
                len(structure_metadata),
            ),
            request,
            feature_payload,
            random_samples,
            structure_metadata,
        )
    )


def deserialize_prepared_request(payload: bytes) -> PreparedRequest:
    """Reference decoder mirrored by the native Boltz-2 runtime."""

    source = io.BytesIO(payload)
    header = source.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise ValueError("truncated Boltz-2 prepared-request header")
    magic, version, request_size, feature_size, random_size, metadata_size = _HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError("invalid Boltz-2 prepared-request magic")
    if version != VERSION:
        raise ValueError(f"unsupported Boltz-2 prepared-request version: {version}")

    def read_exact(size: int, label: str) -> bytes:
        value = source.read(size)
        if len(value) != size:
            raise ValueError(f"truncated Boltz-2 prepared-request {label}")
        return value

    request = read_exact(request_size, "document")
    feature_payload = read_exact(feature_size, "features")
    random_samples = read_exact(random_size, "random samples")
    metadata = read_exact(metadata_size, "structure metadata")
    if source.read(1):
        raise ValueError("Boltz-2 prepared request has trailing bytes")
    if not request or not random_samples.startswith(b"B2RN") or not metadata:
        raise ValueError("Boltz-2 prepared request contains an empty required section")
    return PreparedRequest(
        request, deserialize_features(feature_payload), random_samples, metadata
    )
