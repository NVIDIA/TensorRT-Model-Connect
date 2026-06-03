"""LTX-2 video VAE decoder TRT builder (stub).

Builds ``AutoencoderKLLTX2Video`` decode path. The compression ratios
match LTX-Video (8x temporal, 32x spatial) but the channel layout and
norm types may differ; do not blindly reuse the ltx_video builder
without verifying weight keys.

Not yet implemented.
"""

from __future__ import annotations


def load_ltx2_video_vae_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_video_vae_weights: implement once the LTX-2 video VAE "
        "checkpoint key map is known"
    )


def build_ltx2_video_vae_decoder_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_video_vae_decoder_engine: not implemented"
    )
