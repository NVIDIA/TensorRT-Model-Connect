# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-owned random streams for deterministic native diffusion parity."""

from __future__ import annotations

import struct
from typing import Final

import numpy as np


MAGIC: Final = b"B2RN"
VERSION: Final = 3
MAX_SAMPLING_STEPS: Final = 1000
MAX_STRUCTURE_SAMPLES: Final = 25
MAX_AFFINITY_SAMPLES: Final = 5
# The pinned model/profile consumes this request-invariant Philox offset before
# entering structure_module.sample. E2E parity gates the resulting stream.
_DIFFUSION_RNG_OFFSET: Final = 4544
# The structure pass leaves the official affinity model at this offset. Its
# trunk advances Philox further even in inference mode.
AFFINITY_MODEL_RNG_OFFSET: Final = 6948
# Native execution resolves random arrays directly at the official affinity
# diffusion boundary, after the affinity trunk has advanced Philox.
_AFFINITY_DIFFUSION_RNG_OFFSET: Final = 13572


def _validated_arrays(
    initial: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate one resolved standard-normal/augmentation stream."""

    arrays = tuple(
        np.ascontiguousarray(value, dtype=np.float32)
        for value in (initial, rotations, translations, noise)
    )
    initial, rotations, translations, noise = arrays
    if initial.ndim == 2:
        initial, rotations, translations, noise = (
            value[None] for value in (initial, rotations, translations, noise)
        )
        arrays = tuple(
            np.ascontiguousarray(value) for value in (initial, rotations, translations, noise)
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
    if not 10 <= steps <= MAX_SAMPLING_STEPS:
        raise ValueError(f"Boltz-2 random stream steps must be in [10, {MAX_SAMPLING_STEPS}]")
    return arrays


def _serialize_arrays(arrays: tuple[np.ndarray, ...]) -> bytes:
    return b"".join(value.tobytes(order="C") for value in arrays)


def serialize_random_arrays(
    initial: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    noise: np.ndarray,
    *,
    seed: int,
) -> bytes:
    """Serialize one structure-only request-owned random stream."""

    arrays = _validated_arrays(initial, rotations, translations, noise)
    samples, atoms = arrays[0].shape[:2]
    steps = arrays[1].shape[1]
    if not 1 <= samples <= MAX_STRUCTURE_SAMPLES:
        raise ValueError(f"Boltz-2 structure samples must be in [1, {MAX_STRUCTURE_SAMPLES}]")
    if seed < 0 or seed > np.iinfo(np.int32).max:
        raise ValueError("Boltz-2 random stream seed is outside the INT32 range")
    header = MAGIC + struct.pack("<IIIIIII", VERSION, seed, atoms, steps, samples, 0, 0)
    return header + _serialize_arrays(arrays)


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
    if not 10 <= sampling_steps <= MAX_SAMPLING_STEPS:
        raise ValueError(f"Boltz-2 random stream steps must be in [10, {MAX_SAMPLING_STEPS}]")
    if sample_count > MAX_STRUCTURE_SAMPLES:
        raise ValueError(f"Boltz-2 random sample count must be at most {MAX_STRUCTURE_SAMPLES}")
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


def serialize_pinned_random_samples(
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
    sample_count: int = 1,
) -> bytes:
    """Resolve a direct seed-42 CUDA stream for low-level sampler tests."""

    import torch

    if seed < 0 or seed > np.iinfo(np.int32).max:
        raise ValueError("Boltz-2 random stream seed is outside the INT32 range")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    structure = _resolve_current_cuda_stream(
        sampling_steps=sampling_steps,
        atom_count=atom_count,
        sample_count=sample_count,
    )
    return serialize_random_arrays(
        *structure,
        seed=seed,
    )


def serialize_profile_random_samples(
    *,
    seed: int = 42,
    atom_count: int = 928,
    structure_sampling_steps: int = 200,
    structure_sample_count: int = 1,
    affinity_sampling_steps: int = 200,
    affinity_sample_count: int = 0,
) -> bytes:
    """Resolve exact request-owned streams at Boltz v2.2.1 diffusion boundaries."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Boltz-2 bundle construction requires CUDA random-stream resolution")
    if seed < 0 or seed > np.iinfo(np.int32).max:
        raise ValueError("Boltz-2 random stream seed is outside the INT32 range")
    if not 1 <= structure_sample_count <= MAX_STRUCTURE_SAMPLES:
        raise ValueError(f"Boltz-2 structure samples must be in [1, {MAX_STRUCTURE_SAMPLES}]")
    if not 0 <= affinity_sample_count <= MAX_AFFINITY_SAMPLES:
        raise ValueError(f"Boltz-2 affinity samples must be in [0, {MAX_AFFINITY_SAMPLES}]")
    torch.cuda.init()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.cuda.default_generators[torch.cuda.current_device()]
    generator.set_offset(_DIFFUSION_RNG_OFFSET)
    structure = _validated_arrays(
        *_resolve_current_cuda_stream(
            sampling_steps=structure_sampling_steps,
            atom_count=atom_count,
            sample_count=structure_sample_count,
        )
    )
    affinity: tuple[np.ndarray, ...] = ()
    if affinity_sample_count:
        generator.set_offset(_AFFINITY_DIFFUSION_RNG_OFFSET)
        affinity = _validated_arrays(
            *_resolve_current_cuda_stream(
                sampling_steps=affinity_sampling_steps,
                atom_count=atom_count,
                sample_count=affinity_sample_count,
            )
        )
    header = MAGIC + struct.pack(
        "<IIIIIII",
        VERSION,
        seed,
        atom_count,
        structure_sampling_steps,
        structure_sample_count,
        affinity_sampling_steps if affinity_sample_count else 0,
        affinity_sample_count,
    )
    return header + _serialize_arrays(structure) + _serialize_arrays(affinity)
