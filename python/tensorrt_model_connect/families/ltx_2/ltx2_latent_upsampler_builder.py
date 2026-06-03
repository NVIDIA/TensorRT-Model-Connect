"""LTX-2 latent upsampler TRT builder (stub).

Builds ``LTX2LatentUpsamplerModel`` — used by the two-stage spatial
pipeline (``LTX2LatentUpsamplePipeline``) to bilinearly + neural-net
upsample low-resolution latents before final VAE decode.

Not yet implemented.
"""

from __future__ import annotations


def load_ltx2_latent_upsampler_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_latent_upsampler_weights: implement once the LTX-2 "
        "latent_upsampler checkpoint key map is known"
    )


def build_ltx2_latent_upsampler_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_latent_upsampler_engine: not implemented"
    )
