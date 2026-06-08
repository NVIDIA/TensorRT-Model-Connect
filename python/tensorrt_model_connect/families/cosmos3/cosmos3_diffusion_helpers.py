"""Cosmos 3 diffusion-side numerical helpers.

These are pure-Python (numpy) building blocks used by the DM generator
builder (Phase 4). Each helper computes weights or schedule values that the
TRT graph references as constants, so they can be unit-tested without
touching the TRT runtime.

Helpers provided:
  - ``cosmos3_sinusoidal_timestep_embedding``: log-spaced sinusoidal
    embedding of a (batch,) timestep vector → (batch, hidden) tensor.
    Mirrors the diffusers ``get_timestep_embedding`` with ``flip_sin_to_cos``
    and applies the ``timestep_scale=0.001`` factor documented in
    ``transformer/config.json``.
  - ``cosmos3_rectified_flow_sigmas``: rectified-flow sigma schedule for a
    given number of inference steps and resolution. Uses the
    ``shift_by_resolution`` mapping {256:3, 480:5, 720:10} from
    ``config.json``.
  - ``cosmos3_patch_shape``: derives the (T_lat, H_lat, W_lat) latent grid
    and the post-patchify DM token count for an output video shape.
  - ``cosmos3_unpatchify_shape``: the inverse — given DM token count and the
    (T_lat, H_lat, W_lat), returns the final pixel-space (T, 3, H, W).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


COSMOS3_TIMESTEP_SCALE = 0.001
COSMOS3_BASE_FPS = 24
COSMOS3_SHIFT_BY_RESOLUTION = {256: 3, 480: 5, 720: 10}
COSMOS3_VAE_SPATIAL_SCALE = 16
COSMOS3_VAE_TEMPORAL_SCALE = 4
COSMOS3_LATENT_PATCH_SIZE = 2
COSMOS3_LATENT_CHANNEL = 48


def cosmos3_sinusoidal_timestep_embedding(
    timesteps: np.ndarray,
    hidden_size: int,
    *,
    timestep_scale: float = COSMOS3_TIMESTEP_SCALE,
    max_period: float = 10_000.0,
) -> np.ndarray:
    """Compute the log-spaced sinusoidal timestep embedding.

    Args:
      timesteps: 1-D array of shape ``(batch,)`` containing diffusion
        timesteps (typically integers in [0, num_train_timesteps)).
      hidden_size: target embedding dimension (5120 for Super, 4096 for Nano).
      timestep_scale: multiplier applied to ``timesteps`` before the
        sinusoidal frequencies are computed. Cosmos 3 uses 0.001.
      max_period: log-base for the frequencies. Diffusers default = 10_000.

    Returns:
      Embedding of shape ``(batch, hidden_size)``.
    """
    half = hidden_size // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(half, dtype=np.float32) / half
    )
    args = (timesteps.astype(np.float32) * timestep_scale)[:, None] * freqs[None, :]
    emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
    if hidden_size % 2 == 1:  # pad odd hidden_size
        emb = np.concatenate([emb, np.zeros_like(emb[:, :1])], axis=-1)
    return emb.astype(np.float32)


def cosmos3_rectified_flow_sigmas(
    num_inference_steps: int,
    resolution: int,
    *,
    num_train_timesteps: int = 1000,
    use_dynamic_shifting: bool = False,
) -> np.ndarray:
    """Build the rectified-flow sigma schedule.

    Cosmos 3 uses ``UniPCMultistepScheduler`` with a resolution-dependent
    shift (256→3, 480→5, 720→10). For a given resolution, the schedule is::

        t = linspace(num_train, 1, num_inference)
        sigma = shift * t / (1 + (shift - 1) * t)

    Args:
      num_inference_steps: number of denoising steps (30 for L0, 50 for the
        full lane per the HF model card).
      resolution: the *target frame height*. Must be one of {256, 480, 720}
        (the table from config.json). Other resolutions use the closest
        documented shift.
      num_train_timesteps: 1000 (Cosmos 3 trained at this).
      use_dynamic_shifting: false in config.json — kept as a flag for
        forward compatibility.

    Returns:
      Sigma schedule of shape ``(num_inference_steps,)``, descending.
    """
    if use_dynamic_shifting:
        raise NotImplementedError("dynamic shifting not implemented")

    # Pick the closest documented resolution shift.
    closest = min(COSMOS3_SHIFT_BY_RESOLUTION.keys(),
                  key=lambda r: abs(r - resolution))
    shift = float(COSMOS3_SHIFT_BY_RESOLUTION[closest])

    t = np.linspace(num_train_timesteps, 1, num_inference_steps, dtype=np.float32)
    t = t / num_train_timesteps
    sigma = shift * t / (1.0 + (shift - 1.0) * t)
    return sigma


def cosmos3_patch_shape(
    num_frames: int, height: int, width: int,
) -> Tuple[int, int, int, int]:
    """Compute the latent grid and DM token count for an output video shape.

    Args:
      num_frames: output frame count (pixel-space).
      height: output frame height in pixels.
      width: output frame width in pixels.

    Returns:
      Tuple ``(t_lat, h_lat, w_lat, num_dm_tokens)``:
        - ``t_lat`` = number of latent frames after VAE temporal compression.
        - ``h_lat`` = ``height // 16`` (VAE spatial scale).
        - ``w_lat`` = ``width // 16``.
        - ``num_dm_tokens`` = ``t_lat * (h_lat // 2) * (w_lat // 2)``.
    """
    t_lat = max(1, num_frames // COSMOS3_VAE_TEMPORAL_SCALE)
    h_lat = height // COSMOS3_VAE_SPATIAL_SCALE
    w_lat = width // COSMOS3_VAE_SPATIAL_SCALE
    p = COSMOS3_LATENT_PATCH_SIZE
    num_dm_tokens = t_lat * (h_lat // p) * (w_lat // p)
    return (t_lat, h_lat, w_lat, num_dm_tokens)


def cosmos3_unpatchify_shape(
    t_lat: int, h_lat: int, w_lat: int,
) -> Tuple[int, int, int, int]:
    """The inverse of patch_shape — return the pixel-space output shape.

    Args:
      t_lat: latent frame count (from ``cosmos3_patch_shape``).
      h_lat: latent height in patches × patch_size.
      w_lat: latent width in patches × patch_size.

    Returns:
      Tuple ``(num_frames, 3, height, width)`` in pixel space.
    """
    num_frames = t_lat * COSMOS3_VAE_TEMPORAL_SCALE
    height = h_lat * COSMOS3_VAE_SPATIAL_SCALE
    width = w_lat * COSMOS3_VAE_SPATIAL_SCALE
    return (num_frames, 3, height, width)
