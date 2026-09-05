# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned CUDA random stream for reproducible OpenFold3 native diffusion."""

from __future__ import annotations

import io
import struct
from typing import Final

import numpy as np


MAGIC: Final = b"OF3R"
VERSION: Final = 1


def serialize_random_arrays(
    initial: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise: np.ndarray,
    *,
    seed: int,
) -> bytes:
    arrays = tuple(
        np.ascontiguousarray(value, dtype=np.float32)
        for value in (initial, rotations, translations, noise)
    )
    initial, rotations, translations, noise = arrays
    if initial.ndim != 2 or initial.shape[1:] != (3,):
        raise ValueError("OpenFold3 initial noise must have shape [atoms, 3]")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("OpenFold3 rotations must have shape [steps, 3, 3]")
    steps, atoms = rotations.shape[0], initial.shape[0]
    if translations.shape != (steps, 3) or noise.shape != (steps, atoms, 3):
        raise ValueError("OpenFold3 random stream shapes are inconsistent")
    if not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("OpenFold3 seed is outside the INT32 range")
    return b"".join(
        (
            MAGIC,
            struct.pack("<IIII", VERSION, seed, steps, atoms),
            *(value.tobytes(order="C") for value in arrays),
        )
    )


def serialize_pinned_random_samples(
    *,
    atom_mask: np.ndarray,
    seed: int = 42,
    sampling_steps: int = 200,
) -> bytes:
    """Resolve Algorithm 18/19 random tensors with the upstream CUDA RNG."""

    import torch

    from openfold3.core.model.structure.augmentation import centre_random_augmentation

    if not torch.cuda.is_available():
        raise RuntimeError("OpenFold3 bundle construction requires CUDA")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    atom_mask = np.asarray(atom_mask, dtype=np.float32)
    if atom_mask.ndim != 2 or atom_mask.shape[0] != 1:
        raise ValueError("OpenFold3 atom mask must have shape [1, padded_atoms]")
    padded_atom_count = atom_mask.shape[1]
    if not np.array_equal(atom_mask, atom_mask.astype(bool)):
        raise ValueError("OpenFold3 atom mask must be binary")
    atom_count = int(atom_mask.sum())
    if atom_count <= 0 or not np.array_equal(
        atom_mask[0], np.arange(padded_atom_count) < atom_count
    ):
        raise ValueError("OpenFold3 atom mask must contain contiguous real atoms then padding")
    # Upstream diffusion state is unpadded; sequence-local attention pads only
    # inside the atom modules. Capture the exact real-atom CUDA RNG draws, then
    # append zero slots required by the static TensorRT atom-window profile.
    shape = (1, 1, atom_count, 3)
    initial = torch.randn(shape, device=device, dtype=torch.float32)
    atom_mask_tensor = torch.ones((1, atom_count), device=device, dtype=torch.float32)
    rotations = []
    translations = []
    noises = []
    # Capture Algorithm 19's resolved transform by augmenting a basis. This
    # avoids duplicating upstream quaternion conventions in the native runtime.
    basis = torch.zeros(shape, device=device)
    basis[0, 0, :3] = torch.eye(3, device=device)
    for _ in range(sampling_steps):
        before = torch.cuda.get_rng_state()
        augmented = centre_random_augmentation(basis, atom_mask_tensor)
        torch.cuda.set_rng_state(before)
        from openfold3.core.model.structure.augmentation import sample_rotations

        rotation = sample_rotations((1, 1), torch.float32, device)
        translation = torch.randn((1, 1, 3), device=device, dtype=torch.float32)
        # The basis call above is solely a convention assertion.
        mean = (basis * atom_mask_tensor[..., None]).sum(dim=-2, keepdim=True) / (
            atom_mask_tensor[..., None].sum(dim=-2, keepdim=True).clamp(min=1)
        )
        expected = (
            (basis - mean) @ rotation.transpose(-1, -2) + translation[..., None, :]
        ) * atom_mask_tensor[..., None]
        if not torch.allclose(augmented, expected, atol=1.0e-6, rtol=1.0e-6):
            raise RuntimeError("OpenFold3 augmentation convention changed upstream")
        rotations.append(rotation[0, 0])
        translations.append(translation[0, 0])
        noises.append(torch.randn(shape, device=device, dtype=torch.float32)[0, 0])
    initial_padded = np.zeros((padded_atom_count, 3), dtype=np.float32)
    initial_padded[:atom_count] = initial[0, 0].cpu().numpy()
    noise_padded = np.zeros((sampling_steps, padded_atom_count, 3), dtype=np.float32)
    noise_padded[:, :atom_count] = torch.stack(noises).cpu().numpy()
    payload = serialize_random_arrays(
        initial_padded,
        torch.stack(rotations).cpu().numpy(),
        torch.stack(translations).cpu().numpy(),
        noise_padded,
        seed=seed,
    )
    torch.cuda.empty_cache()
    return payload


def deserialize_random_samples(payload: bytes) -> tuple[int, tuple[np.ndarray, ...]]:
    """Decode the stream for deterministic contract tests."""

    source = io.BytesIO(payload)
    if source.read(4) != MAGIC:
        raise ValueError("invalid OpenFold3 random-sample magic")
    version, seed, steps, atoms = struct.unpack("<IIII", source.read(16))
    if version != VERSION or steps != 200 or atoms <= 0:
        raise ValueError("unsupported OpenFold3 random-sample header")

    def array(shape):
        count = int(np.prod(shape))
        data = source.read(count * 4)
        if len(data) != count * 4:
            raise ValueError("truncated OpenFold3 random-sample payload")
        return np.frombuffer(data, np.float32).reshape(shape)

    arrays = (
        array((atoms, 3)),
        array((steps, 3, 3)),
        array((steps, 3)),
        array((steps, atoms, 3)),
    )
    if source.read(1):
        raise ValueError("OpenFold3 random-sample payload has trailing bytes")
    return seed, arrays
