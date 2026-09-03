# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned build-time random stream for deterministic native diffusion parity."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Final

import numpy as np


MAGIC: Final = b"B2RN"
VERSION: Final = 1


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


def serialize_predict_random_samples(
    checkpoint_path: Path,
    features,
    *,
    seed: int = 42,
    sampling_steps: int = 200,
    atom_count: int = 928,
) -> bytes:
    """Capture the CUDA stream at Boltz's real diffusion-sampling boundary."""

    import torch

    from .reference_benchmark import _load_model, _predict

    class _DiffusionBoundary(RuntimeError):
        pass

    model = _load_model(checkpoint_path, compiled=False)
    captured_state = None
    original_sample = model.structure_module.sample

    def capture_state(*_args, **_kwargs):
        nonlocal captured_state
        captured_state = torch.cuda.get_rng_state()
        raise _DiffusionBoundary

    model.structure_module.sample = capture_state
    try:
        _predict(model, features)
    except _DiffusionBoundary:
        pass
    finally:
        model.structure_module.sample = original_sample
    if captured_state is None:
        raise RuntimeError("Boltz-2 did not reach its diffusion-sampling boundary")
    del model
    torch.cuda.set_rng_state(captured_state)
    payload = _serialize_current_cuda_stream(
        seed=seed,
        sampling_steps=sampling_steps,
        atom_count=atom_count,
    )
    torch.cuda.empty_cache()
    return payload
