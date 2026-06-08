"""Cosmos 3 video VAE TRT builder.

Cosmos 3 reuses the Wan 2.2 TI2V-5B VAE (per ``vae/config.json``: same
``AutoencoderKLWan`` class, ``_name_or_path: Wan-AI/Wan2.2-TI2V-5B-Diffusers``).
The encoder/decoder are wider than Wan 2.1's (base_dim=160 / z_dim=48 vs Wan
2.1's base_dim=96 / z_dim=16) but architecturally identical, so this builder
just dispatches to ``families/wan_t2v/causal_vae_3d_builder.py`` with the
Wan 2.2 dimensions baked in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..wan_t2v.causal_vae_3d_builder import build_causal_vae_3d_engine

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# Cosmos3-Super video VAE constants (locked from vae/config.json).
COSMOS3_SUPER_VAE_BASE_DIM = 160
COSMOS3_SUPER_VAE_DECODER_BASE_DIM = 256
COSMOS3_SUPER_VAE_Z_DIM = 48
COSMOS3_SUPER_VAE_DIM_MULT = (1, 2, 4, 4)
COSMOS3_SUPER_VAE_NUM_RES_BLOCKS = 2
COSMOS3_SUPER_VAE_IN_CHANNELS = 12
COSMOS3_SUPER_VAE_OUT_CHANNELS = 12
COSMOS3_SUPER_VAE_PATCH_SIZE = 2
COSMOS3_SUPER_VAE_SCALE_FACTOR_SPATIAL = 16
COSMOS3_SUPER_VAE_SCALE_FACTOR_TEMPORAL = 4
COSMOS3_SUPER_VAE_TEMPORAL_DOWNSAMPLE = (False, True, True)
COSMOS3_SUPER_VAE_IS_RESIDUAL = True


def build_cosmos3_vae_engine(
    weights: "WeightDict",
    *,
    h_lat: int,
    w_lat: int,
    verbose: bool = False,
) -> bytes:
    """Build the Cosmos 3 video VAE decoder TRT engine.

    Args:
      weights: weights dict containing the VAE keys (under ``vae.*``).
      h_lat: latent height (post-spatial-compression). For a 720p × 1280p
        target frame, ``h_lat = 720 // scale_factor_spatial = 720 // 16 = 45``.
      w_lat: latent width. For 1280p target, ``w_lat = 1280 // 16 = 80``.
      verbose: verbose engine build logging.

    Returns:
      Serialized TRT engine bytes for the VAE decoder.
    """
    return build_causal_vae_3d_engine(
        weights,
        z_dim=COSMOS3_SUPER_VAE_Z_DIM,
        base_dim=COSMOS3_SUPER_VAE_DECODER_BASE_DIM,
        dim_mult=COSMOS3_SUPER_VAE_DIM_MULT,
        num_res_blocks=COSMOS3_SUPER_VAE_NUM_RES_BLOCKS,
        temporal_upsample=COSMOS3_SUPER_VAE_TEMPORAL_DOWNSAMPLE,
        h_lat=h_lat,
        w_lat=w_lat,
        out_channels=3,
        verbose=verbose)
