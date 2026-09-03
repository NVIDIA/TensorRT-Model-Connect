# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned build-time random stream for deterministic native diffusion parity."""

from __future__ import annotations

import struct
from typing import Final

import numpy as np


MAGIC: Final = b"B2RN"
VERSION: Final = 1
# The pinned model/profile consumes this request-invariant Philox offset before
# entering structure_module.sample. E2E parity gates the resulting stream.
_DIFFUSION_RNG_OFFSET: Final = 4544


def serialize_random_arrays(
    initial: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise: np.ndarray,
    *,
    seed: int,
) -> bytes:
    """Serialize an already-resolved standard-normal/augmentation stream."""

    arrays = tuple(np.ascontiguousarray(value, dtype=np.float32) for value in (
        initial,
        rotations,
        translations,
        noise,
    ))
    initial, rotations, translations, noise = arrays
    if initial.ndim != 2 or initial.shape[1:] != (3,):
        raise ValueError("Boltz-2 initial random coordinates must have shape [atoms, 3]")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("Boltz-2 random rotations must have shape [steps, 3, 3]")
    steps = rotations.shape[0]
    atoms = initial.shape[0]
    if translations.shape != (steps, 3) or noise.shape != (steps, atoms, 3):
        raise ValueError("Boltz-2 random stream shapes are inconsistent")
    if seed < 0 or seed > np.iinfo(np.int32).max:
        raise ValueError("Boltz-2 random stream seed is outside the INT32 range")
    header = MAGIC + struct.pack("<IIII", VERSION, seed, steps, atoms)
    return header + b"".join(value.tobytes(order="C") for value in arrays)


def _serialize_current_cuda_stream(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
) -> bytes:
    """Resolve the upstream random arrays from the current CUDA RNG state."""

    import torch

    from boltz.model.modules.utils import compute_random_augmentation

    if not torch.cuda.is_available():
        raise RuntimeError("Boltz-2 bundle construction requires CUDA random-stream resolution")
    device = torch.device("cuda")
    shape = (1, atom_count, 3)
    initial = torch.randn(shape, device=device, dtype=torch.float32)
    rotations = []
    translations = []
    noise = []
    for _ in range(sampling_steps):
        rotation, translation = compute_random_augmentation(
            1,
            device=device,
            dtype=torch.float32,
        )
        rotations.append(rotation)
        translations.append(translation[:, 0])
        noise.append(torch.randn(shape, device=device, dtype=torch.float32))
    return serialize_random_arrays(
        initial[0].cpu().numpy(),
        torch.cat(rotations).cpu().numpy(),
        torch.cat(translations).cpu().numpy(),
        torch.cat(noise).cpu().numpy(),
        seed=seed,
    )


def serialize_pinned_random_samples(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
) -> bytes:
    """Resolve a direct seed-42 CUDA stream for low-level sampler tests."""

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return _serialize_current_cuda_stream(
        seed=seed,
        sampling_steps=sampling_steps,
        atom_count=atom_count,
    )


def serialize_profile_random_samples(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
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
    return _serialize_current_cuda_stream(
        seed=seed,
        sampling_steps=sampling_steps,
        atom_count=atom_count,
    )
