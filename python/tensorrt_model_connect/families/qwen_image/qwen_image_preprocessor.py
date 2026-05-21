"""Qwen-Image preprocessor weights blob.

Packs the small runtime-resident tensors that the C++ pipeline reads
from the bundle (NOT baked into a TRT engine):

  - ``latents_mean``: [16] float32 — per-channel VAE latent mean.
  - ``latents_std``:  [16] float32 — per-channel VAE latent std.

The C++ pipeline uses these to un-normalize the denoiser output before
VAE decode (``z = z * latents_std + latents_mean``), matching the
diffusers normalization contract.

Most other tensors (timestep MLP, text projection, etc.) are baked
into the denoiser engine by :mod:`qwen_image_dit_builder`, so unlike
Z-Image / Wan T2V this blob is intentionally tiny.

Format (matches Z-Image / Wan T2V so
``src/runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h``
can parse it unchanged):

    <u32 little-endian index_len>
    <UTF-8 JSON dict mapping name -> {"offset": int, "shape": [int, ...]}>
    <contiguous raw float32 data, all entries in declaration order>

Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
"""

from __future__ import annotations

import json
import struct
from typing import Mapping

import numpy as np


# Names the C++ pipeline will look up in the index.
_PREPROCESSOR_KEYS: tuple[str, ...] = (
    "latents_mean",
    "latents_std",
)


def pack_qwen_image_preprocessor_weights(
    named_tensors: Mapping[str, np.ndarray],
) -> bytes:
    """Serialize named tensors into the diffusion preprocessor blob format.

    Entries are written in :data:`_PREPROCESSOR_KEYS` order; any extra
    keys in ``named_tensors`` are appended in dict-iteration order so the
    helper is also useful for unit tests that exercise round-trip with
    synthetic tensors. Missing canonical keys raise ``ValueError``.

    All arrays are coerced to contiguous float32. Shapes are recorded as
    Python ``int`` lists in the JSON index.
    """
    for key in _PREPROCESSOR_KEYS:
        if key not in named_tensors:
            raise ValueError(
                f"pack_qwen_image_preprocessor_weights: missing required "
                f"key {key!r}"
            )

    # Preserve canonical-first ordering, then any extras for test flexibility.
    ordered_keys: list[str] = list(_PREPROCESSOR_KEYS)
    for key in named_tensors:
        if key not in ordered_keys:
            ordered_keys.append(key)

    index: dict[str, dict] = {}
    data_parts: list[bytes] = []
    offset = 0
    for key in ordered_keys:
        arr = np.ascontiguousarray(np.asarray(named_tensors[key], dtype=np.float32))
        index[key] = {"offset": offset, "shape": [int(d) for d in arr.shape]}
        data_parts.append(arr.tobytes())
        offset += arr.nbytes

    index_json = json.dumps(index).encode("utf-8")
    header = struct.pack("<I", len(index_json))
    return header + index_json + b"".join(data_parts)


def load_qwen_image_preprocessor_weights(
    blob: bytes,
) -> dict[str, np.ndarray]:
    """Inverse of :func:`pack_qwen_image_preprocessor_weights`.

    Used by unit tests to assert lossless round-trip; the production C++
    runtime has its own parser in
    ``diffusion_preprocessor_weights_helpers.h``.
    """
    if len(blob) < 4:
        raise ValueError(
            f"qwen_image preprocessor blob too small: {len(blob)} bytes"
        )
    (index_len,) = struct.unpack("<I", blob[:4])
    if 4 + index_len > len(blob):
        raise ValueError(
            f"qwen_image preprocessor index length {index_len} overflows "
            f"blob of {len(blob)} bytes"
        )

    index = json.loads(blob[4:4 + index_len].decode("utf-8"))
    data = blob[4 + index_len:]

    out: dict[str, np.ndarray] = {}
    for name, entry in index.items():
        shape = tuple(int(d) for d in entry["shape"])
        offset = int(entry["offset"])
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * 4  # float32
        if offset + nbytes > len(data):
            raise ValueError(
                f"qwen_image preprocessor weight {name!r} overflows blob "
                f"(offset={offset}, nbytes={nbytes}, blob_data={len(data)})"
            )
        flat = np.frombuffer(data, dtype=np.float32, count=count, offset=offset)
        out[name] = np.ascontiguousarray(flat.reshape(shape))
    return out


def extract_preprocessor_source(vae_config) -> dict[str, np.ndarray]:
    """Pull the runtime-resident tensors out of a ``QwenImageVAEConfig``.

    Only ``latents_mean`` / ``latents_std`` need to ship outside an engine
    for Qwen-Image T2I; the timestep MLP, text projection, etc. are baked
    into the denoiser engine by ``build_qwen_image_dit_engine``.

    Returned arrays are contiguous float32 vectors. The caller passes the
    result to :func:`pack_qwen_image_preprocessor_weights`.
    """
    return {
        "latents_mean": np.ascontiguousarray(
            np.asarray(vae_config.latents_mean, dtype=np.float32)
        ),
        "latents_std": np.ascontiguousarray(
            np.asarray(vae_config.latents_std, dtype=np.float32)
        ),
    }


__all__ = [
    "pack_qwen_image_preprocessor_weights",
    "load_qwen_image_preprocessor_weights",
    "extract_preprocessor_source",
]
