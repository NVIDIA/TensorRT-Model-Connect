# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pinned reference: the transformers ESMFold2 port, folding with the family's random draws.

Only the reference's random sources are replaced: the trunk's initial pair state, the per-loop LM dropout, and the
diffusion sampler's initial coordinates, rotations, translations and churn noise take the values of the family's
streams (``family_rng``), so the native runtime and the reference fold from identical noise. Each replacement checks
the call it expects, so a change in the upstream sampling code fails instead of silently drawing other noise.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import family_rng

REFERENCE_TRANSFORMERS = "5.24.1"


@dataclass
class ReferenceFold:
    coords: np.ndarray  # [samples, atoms, 3] float32, padded atoms included
    plddt: np.ndarray  # [samples, tokens] 0-1
    ptm: np.ndarray  # [samples]
    complex_plddt: np.ndarray  # [samples]
    pdb: list[str]  # per sample, the reference's output_to_pdb
    real_atoms: int


class _Draws:
    def __init__(self, torch: Any, seed: int, samples: int, atoms: int, loops: int, dropout: float) -> None:
        self.torch = torch
        self.seed = seed
        self.samples = samples
        self.atoms = atoms
        self.loops = loops
        self.lm_dropout = dropout
        self.dropout_calls = 0
        self.coordinates_drawn = False
        self.rotation_steps = 0
        self.translation_steps = 0
        self.churn_steps = 0

    def _tensor(self, values: np.ndarray, shape: tuple[int, ...], device: Any, dtype: Any) -> Any:
        tensor = self.torch.from_numpy(values.reshape(shape).astype(np.float32))
        return tensor.to(device=device, dtype=dtype or self.torch.float32)

    def initial_pair_state(self, reference: Any) -> Any:
        count = int(np.prod(reference.shape))
        std = float(np.sqrt(2.0 / (5.0 * reference.shape[-1])))
        values = family_rng.truncated_normal(self.seed, family_rng.TRUNK_INIT, np.arange(count, dtype=np.uint64), std)
        return self._tensor(values, tuple(reference.shape), reference.device, self.torch.float32).to(reference.dtype)

    def dropout(self, original: Any, tensor: Any, p: float = 0.5, training: bool = True, inplace: bool = False) -> Any:
        if not training or p != self.lm_dropout or tensor.dim() != 4:
            return original(tensor, p, training, inplace)
        if self.dropout_calls > self.loops:
            raise AssertionError("the reference drew more LM dropout masks than trunk iterations")
        count = int(tensor.numel())
        indices = np.arange(count, dtype=np.uint64) + np.uint64(self.dropout_calls * count)
        keep = family_rng.keep(self.seed, family_rng.LM_DROPOUT, indices, p).reshape(tuple(tensor.shape))
        self.dropout_calls += 1
        mask = self.torch.from_numpy(keep.astype(np.float32)).to(tensor.device)
        return (tensor.float() * mask * (1.0 / (1.0 - p))).to(tensor.dtype)

    def randn(self, original: Any, *size: Any, **kwargs: Any) -> Any:
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        shape = tuple(int(value) for value in shape)
        device, dtype = kwargs.get("device"), kwargs.get("dtype")
        if shape == (self.samples, self.atoms, 3) and not self.coordinates_drawn:
            self.coordinates_drawn = True
            values = family_rng.normal(self.seed, family_rng.COORDINATES, np.arange(np.prod(shape), dtype=np.uint64))
            return self._tensor(values, shape, device, dtype)
        if shape == (self.samples, 4):
            step = self.rotation_steps
            self.rotation_steps += 1
            first = np.uint64(step * self.samples * 4)
            values = family_rng.normal(
                self.seed, family_rng.ROTATION, first + np.arange(self.samples * 4, dtype=np.uint64)
            )
            return self._tensor(values, shape, device, dtype)
        raise AssertionError(f"unexpected torch.randn{shape} in the reference sampler")

    def randn_like(self, original: Any, tensor: Any, **kwargs: Any) -> Any:
        shape = tuple(int(value) for value in tensor.shape)
        if shape == (self.samples, 1, 3):
            step = self.translation_steps
            self.translation_steps += 1
            if step != self.rotation_steps - 1:
                raise AssertionError("translation drawn out of step order")
            first = np.uint64(step * self.samples * 3)
            values = family_rng.normal(
                self.seed, family_rng.TRANSLATION, first + np.arange(self.samples * 3, dtype=np.uint64)
            )
            return self._tensor(values, shape, tensor.device, tensor.dtype)
        if shape == (self.samples, self.atoms, 3):
            step = self.churn_steps
            self.churn_steps += 1
            if step != self.rotation_steps - 1:
                raise AssertionError("churn noise drawn out of step order")
            count = self.samples * self.atoms * 3
            values = family_rng.normal(
                self.seed, family_rng.CHURN, np.uint64(step * count) + np.arange(count, dtype=np.uint64)
            )
            return self._tensor(values, shape, tensor.device, tensor.dtype)
        raise AssertionError(f"unexpected torch.randn_like{shape} in the reference sampler")


@contextlib.contextmanager
def family_draws(model: Any, draws: _Draws) -> Iterator[None]:
    import torch
    from transformers.models.esmfold2 import generation_esmfold2, modeling_esmfold2

    original_dropout = torch.nn.functional.dropout
    original_randn = torch.randn
    original_randn_like = torch.randn_like
    if generation_esmfold2.torch is not torch or modeling_esmfold2.F is not torch.nn.functional:
        raise AssertionError("the reference no longer draws through torch's module functions")
    saved_init = model._init_pair_state
    model._init_pair_state = draws.initial_pair_state
    torch.nn.functional.dropout = lambda *a, **k: draws.dropout(original_dropout, *a, **k)
    torch.randn = lambda *a, **k: draws.randn(original_randn, *a, **k)
    torch.randn_like = lambda *a, **k: draws.randn_like(original_randn_like, *a, **k)
    try:
        yield
    finally:
        model._init_pair_state = saved_init
        torch.nn.functional.dropout = original_dropout
        torch.randn = original_randn
        torch.randn_like = original_randn_like


@contextlib.contextmanager
def reference_numerics() -> Iterator[None]:
    """float32 matmuls without TF32 and attention without cuDNN's fused kernel while the reference runs.

    The previous settings come back afterwards, also when the reference fails.
    """

    import torch

    saved_precision = torch.get_float32_matmul_precision()
    saved_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    saved_cudnn_sdp = torch.backends.cuda.cudnn_sdp_enabled()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # cuDNN's fused SDPA is not used, so the reference attention does not depend on that backend's tactics.
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(saved_precision)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved_tf32
        torch.backends.cuda.enable_cudnn_sdp(saved_cudnn_sdp)


def load_reference(model_dir: Path, precision: str, device: str = "cuda") -> Any:
    """The transformers ESMFold2 port in float32 or bfloat16, on ``device`` (``reference_fold`` sets its numerics)."""

    import torch
    import transformers
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model

    if transformers.__version__ != REFERENCE_TRANSFORMERS:
        raise AssertionError(f"the reference is pinned to transformers {REFERENCE_TRANSFORMERS}")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    model = EsmFold2Model.from_pretrained(str(model_dir), dtype=dtype, local_files_only=True)
    return model.to(device).eval()


def reference_fold(
    model: Any,
    sequence: str,
    *,
    seed: int,
    loops: int,
    sampling_steps: int,
    samples: int,
) -> ReferenceFold:
    """Fold ``sequence`` with the reference model and the family's draws for ``seed``."""

    import torch
    from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

    device = next(model.parameters()).device
    features = prepare_protein_features(sequence, device=device)
    atoms = int(features["atom_to_token"].shape[1])
    dropout = float(model.config.lm_encoder.lm_dropout)
    draws = _Draws(torch, seed, samples, atoms, loops, dropout)
    with torch.no_grad(), reference_numerics(), family_draws(model, draws):
        output = model.fold(
            **features, num_loops=loops, num_sampling_steps=sampling_steps, num_diffusion_samples=samples
        )
    if draws.dropout_calls != loops + 1 or not draws.coordinates_drawn:
        raise AssertionError("the reference did not draw the expected trunk and sampler noise")
    if not (draws.rotation_steps == draws.translation_steps == draws.churn_steps):
        raise AssertionError("the reference sampler drew an unexpected number of step draws")
    pdb = [output_to_pdb(output, features, sample_idx=index) for index in range(samples)]
    return ReferenceFold(
        coords=output["sample_atom_coords"].float().cpu().numpy(),
        plddt=output["plddt"].float().cpu().numpy(),
        ptm=output["ptm"].float().cpu().numpy(),
        complex_plddt=output["complex_plddt"].float().cpu().numpy(),
        pdb=pdb,
        real_atoms=int(features["atom_attention_mask"].sum().item()),
    )
