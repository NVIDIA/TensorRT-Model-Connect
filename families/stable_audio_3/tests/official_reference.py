# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official ``stable-audio-3`` pipeline as the oracle of this family.

* The library comes from a checkout of ``Stability-AI/stable-audio-3`` at the
  revision in ``reference-source.json`` (``TRTMC_REFERENCE_SOURCE_DIR``).
* Stable Audio 3 Medium requires Flash Attention 2 (the library's README). Its
  CUDA kernels do not cover every GPU this repository validates on, so the
  ``flash_attn`` module is provided here with the same semantics on top of
  PyTorch: FP32 accumulation of FP16/BF16 inputs, ``window_size`` bands and
  variable-length (packed) sequences.
* ``StableAudioModel.generate`` runs unmodified. Its two random draws -- the
  initial latent noise and the per-step noise of the sampler -- are taken from
  the family's own noise stream (``family_sampling.gaussian_noise``), the same
  values the native runtime uses for the same seed.
* The noise the official decoder adds at inference (the bottleneck's
  ``noise_regularize`` and the decoder's ``mask_noise``) is switched off, as the
  family does and as Stability AI's own TensorRT and MLX ports of the SAME
  decoder do.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

from families.stable_audio_3.tests import family_sampling

SOURCE_ENTRYPOINT = "stable_audio_3/model.py"


def source_dir() -> Path:
    value = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    assert value, "selected stable_audio_3 reference requires TRTMC_REFERENCE_SOURCE_DIR"
    source = Path(value).resolve()
    assert (source / SOURCE_ENTRYPOINT).is_file(), f"stable-audio-3 checkout is missing {SOURCE_ENTRYPOINT}"
    return source


# -- flash_attn with PyTorch ----------------------------------------------------
_CHUNK = 1024


def _attention_bhnd(q, k, v, causal: bool, window, scale):
    """Exact attention on [B, H, N, D]; ``window`` = (left, right), -1 = unbounded."""
    import torch
    import torch.nn.functional as F

    left, right = (int(window[0]), int(window[1])) if window is not None else (-1, -1)
    n_q, n_k = q.shape[-2], k.shape[-2]
    if left < 0 and right < 0 and not causal:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    outputs = []
    for start in range(0, n_q, _CHUNK):
        stop = min(start + _CHUNK, n_q)
        k_start = 0 if left < 0 else max(0, start - left)
        k_stop = n_k if (right < 0 and not causal) else min(n_k, stop + (0 if causal else right))
        rows = torch.arange(start, stop, device=q.device)[:, None]
        cols = torch.arange(k_start, k_stop, device=q.device)[None, :]
        keep = torch.ones((stop - start, k_stop - k_start), dtype=torch.bool, device=q.device)
        if left >= 0:
            keep &= cols >= rows - left
        if right >= 0:
            keep &= cols <= rows + right
        if causal:
            keep &= cols <= rows
        outputs.append(F.scaled_dot_product_attention(
            q[..., start:stop, :], k[..., k_start:k_stop, :], v[..., k_start:k_stop, :], attn_mask=keep,
            scale=scale))
    return torch.cat(outputs, dim=-2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **_):
    """flash_attn.flash_attn_func on [B, N, H, D] FP16/BF16 inputs."""
    assert dropout_p == 0.0
    dtype = q.dtype
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    out = _attention_bhnd(q, k, v, causal, window_size, softmax_scale)
    return out.transpose(1, 2).to(dtype)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                           softmax_scale=None, causal=False, window_size=(-1, -1), **_):
    """flash_attn.flash_attn_varlen_func on packed [total, H, D] sequences."""
    import torch

    assert dropout_p == 0.0
    del max_seqlen_q, max_seqlen_k
    dtype = q.dtype
    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        qs, qe = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        ks, ke = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        seq = [t.float().transpose(0, 1).unsqueeze(0) for t in (q[qs:qe], k[ks:ke], v[ks:ke])]
        out = _attention_bhnd(*seq, causal, window_size, softmax_scale)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0).to(dtype)


def flash_attn_kvpacked_func(*_args, **_kwargs):
    raise NotImplementedError("flash_attn_kvpacked_func is not used by stable-audio-3")


def index_first_axis(x, indices):
    return x.reshape(-1, *x.shape[1:])[indices] if x.dim() > 1 else x[indices]


def pad_input(hidden, indices, batch, seqlen):
    import torch

    out = torch.zeros((batch * seqlen, *hidden.shape[1:]), dtype=hidden.dtype, device=hidden.device)
    out[indices] = hidden
    return out.reshape(batch, seqlen, *hidden.shape[1:])


def unpad_input(hidden, attention_mask):
    import torch
    import torch.nn.functional as F

    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden.reshape(-1, *hidden.shape[2:])
    return flat[indices], indices, cu_seqlens, int(seqlens.max())


def install_flash_attn() -> None:
    """Register the ``flash_attn`` module before the library imports it."""
    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    module.__version__ = "2.6.3+trtmc-sdpa"
    module.flash_attn_func = flash_attn_func
    module.flash_attn_kvpacked_func = flash_attn_kvpacked_func
    module.flash_attn_varlen_func = flash_attn_varlen_func
    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    padding.pad_input = pad_input
    padding.unpad_input = unpad_input
    padding.index_first_axis = index_first_axis
    module.bert_padding = padding
    sys.modules["flash_attn"] = module
    sys.modules["flash_attn.bert_padding"] = padding


# -- official model -------------------------------------------------------------
def load_official(model_dir: Path, *, half: bool, device: str = "cuda"):
    """``StableAudioModel.from_pretrained`` for a local snapshot, decoder noise off."""
    install_flash_attn()
    source = str(source_dir())
    if source not in sys.path:
        sys.path.insert(0, source)
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.model import StableAudioModel

    config = json.loads((Path(model_dir) / "model_config.json").read_text(encoding="utf-8"))
    for entry in config["model"]["conditioning"]["configs"]:
        if entry["type"] == "t5gemma":
            entry["config"]["model_path"] = str(model_dir)
    model = load_diffusion_cond(config, str(Path(model_dir) / "model.safetensors"), device=device,
                                model_half=half)
    model.use_lora = False
    model.lora_names = []
    autoencoder = model.pretransform.model
    autoencoder.bottleneck.noise_regularize = False
    for layer in autoencoder.decoder.layers:
        if hasattr(layer, "mask_noise"):
            layer.mask_noise = 0
    return StableAudioModel(model, config, device, half)


@contextlib.contextmanager
def family_noise(seed: int):
    """Serve ``torch.randn`` / ``torch.randn_like`` from the family noise stream.

    Draw 0 is generate()'s initial noise, draw i + 1 the noise of sampler step i.
    """
    import torch

    original_randn, original_randn_like = torch.randn, torch.randn_like
    draws = {"next": 0}

    def take(shape, dtype, device):
        count = int(np.prod(shape))
        values = family_sampling.gaussian_noise(seed, draws["next"], count).reshape(shape)
        draws["next"] += 1
        return torch.from_numpy(values).to(device=device, dtype=dtype)

    def randn(*size, dtype=None, device=None, generator=None, **_):
        assert generator is None
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (list, tuple)) else tuple(size)
        return take(shape, dtype or torch.float32, device or "cpu")

    def randn_like(tensor, dtype=None, **_):
        return take(tuple(tensor.shape), dtype or tensor.dtype, tensor.device)

    torch.randn, torch.randn_like = randn, randn_like
    try:
        yield draws
    finally:
        torch.randn, torch.randn_like = original_randn, original_randn_like


def generate(official, prompt: str, *, duration: float, steps: int, seed: int) -> np.ndarray:
    """Official ``generate`` with the family noise; returns [frames, channels] float32."""
    import torch

    with family_noise(seed) as draws:
        audio = official.generate(prompt=prompt, duration=duration, steps=steps, cfg_scale=1.0, seed=seed)
    # The initial noise, plus one draw per step for the pingpong sampler.
    expected = 1 + steps
    assert draws["next"] == expected, f"expected {expected} noise draws, saw {draws['next']}"
    torch.cuda.synchronize()
    return audio[0].float().cpu().numpy().T.copy()
