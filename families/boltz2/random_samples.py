# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned build-time random stream for deterministic native diffusion parity."""

from __future__ import annotations

import struct
from typing import Final

import numpy as np


MAGIC: Final = b"B2RN"
VERSION: Final = 2
# The pinned model/profile consumes this request-invariant Philox offset before
# entering structure_module.sample. E2E parity gates the resulting stream.
_DIFFUSION_RNG_OFFSET: Final = 4544
# The pinned 928-atom, 200-step structure sample advances Philox to this
# offset before the official affinity pass draws its five samples.
AFFINITY_DIFFUSION_RNG_OFFSET: Final = 6948


def serialize_random_arrays(
    initial: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise: np.ndarray,
    *,
    seed: int,
) -> bytes:
    """Serialize an already-resolved standard-normal/augmentation stream."""

    arrays = tuple(
        np.ascontiguousarray(value, dtype=np.float32)
        for value in (initial, rotations, translations, noise)
    )
    initial, rotations, translations, noise = arrays
    if initial.ndim == 2:
        initial, rotations, translations, noise = (
            value[None] for value in (initial, rotations, translations, noise)
        )
    if initial.ndim != 3 or initial.shape[2:] != (3,):
        raise ValueError("Boltz-2 initial random coordinates must have shape [samples, atoms, 3]")
    if rotations.ndim != 4 or rotations.shape[2:] != (3, 3):
        raise ValueError("Boltz-2 random rotations must have shape [samples, steps, 3, 3]")
    samples, atoms = initial.shape[:2]
    steps = rotations.shape[1]
    if (
        rotations.shape[0] != samples
        or translations.shape != (samples, steps, 3)
        or noise.shape != (samples, steps, atoms, 3)
    ):
        raise ValueError("Boltz-2 random stream shapes are inconsistent")
    if seed < 0 or seed > np.iinfo(np.int32).max:
        raise ValueError("Boltz-2 random stream seed is outside the INT32 range")
    header = MAGIC + struct.pack("<IIIII", VERSION, seed, steps, atoms, samples)
    return header + b"".join(value.tobytes(order="C") for value in arrays)


def _resolve_current_cuda_stream(
    *,
    sampling_steps: int = 200,
    atom_count: int = 928,
    sample_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one upstream diffusion call from the current CUDA RNG state."""

    import torch

    from boltz.model.modules.utils import compute_random_augmentation

    if not torch.cuda.is_available():
        raise RuntimeError("Boltz-2 bundle construction requires CUDA random-stream resolution")
    device = torch.device("cuda")
    if sample_count <= 0:
        raise ValueError("Boltz-2 random sample count must be positive")
    shape = (sample_count, atom_count, 3)
    initial = torch.randn(shape, device=device, dtype=torch.float32)
    rotations, translations, noise = [], [], []
    for _ in range(sampling_steps):
        rotation, translation = compute_random_augmentation(
            sample_count,
            device=device,
            dtype=torch.float32,
        )
        rotations.append(rotation)
        translations.append(translation[:, 0])
        noise.append(torch.randn(shape, device=device, dtype=torch.float32))
    return tuple(
        value.cpu().numpy()
        for value in (
            initial,
            torch.stack(rotations, dim=1),
            torch.stack(translations, dim=1),
            torch.stack(noise, dim=1),
        )
    )


def _serialize_current_cuda_stream(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
    sample_count: int = 1,
) -> bytes:
    """Serialize one upstream diffusion call from the current CUDA RNG state."""

    arrays = _resolve_current_cuda_stream(
        sampling_steps=sampling_steps,
        atom_count=atom_count,
        sample_count=sample_count,
    )
    return serialize_random_arrays(
        *arrays,
        seed=seed,
    )


def serialize_pinned_random_samples(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
    sample_count: int = 1,
) -> bytes:
    """Resolve a direct seed-42 CUDA stream for low-level sampler tests."""

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    structure = _resolve_current_cuda_stream(
        sampling_steps=sampling_steps,
        atom_count=atom_count,
        sample_count=1,
    )
    if sample_count == 1:
        arrays = structure
    else:
        affinity = _resolve_current_cuda_stream(
            sampling_steps=sampling_steps,
            atom_count=atom_count,
            sample_count=sample_count - 1,
        )
        arrays = tuple(
            np.concatenate((structure_array, affinity_array), axis=0)
            for structure_array, affinity_array in zip(structure, affinity, strict=True)
        )
    return serialize_random_arrays(
        *arrays,
        seed=seed,
    )


def serialize_profile_random_samples(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
    sample_count: int = 6,
) -> bytes:
    """Resolve the pinned stream at Boltz v2.2.1's diffusion boundary."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Boltz-2 bundle construction requires CUDA random-stream resolution")
    torch.cuda.init()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.cuda.default_generators[torch.cuda.current_device()]
    generator.set_offset(_DIFFUSION_RNG_OFFSET)
    structure = _resolve_current_cuda_stream(
        sampling_steps=sampling_steps,
        atom_count=atom_count,
        sample_count=1,
    )
    if sample_count == 1:
        arrays = structure
    else:
        generator.set_offset(AFFINITY_DIFFUSION_RNG_OFFSET)
        affinity = _resolve_current_cuda_stream(
            sampling_steps=sampling_steps,
            atom_count=atom_count,
            sample_count=sample_count - 1,
        )
        arrays = tuple(
            np.concatenate((structure_array, affinity_array), axis=0)
            for structure_array, affinity_array in zip(structure, affinity, strict=True)
        )
    return serialize_random_arrays(
        *arrays,
        seed=seed,
    )
