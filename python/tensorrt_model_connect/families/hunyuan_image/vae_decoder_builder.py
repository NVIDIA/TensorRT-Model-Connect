"""HunyuanImage-2.1 VAE decoder builder — scaffold.

HunyuanImage-2.1 ships a custom VAE (``AutoencoderKLHunyuanImage`` once it
lands in diffusers, or ``hyimage.vae.AutoencoderKL`` from Tencent's repo).
Architectural highlights from public write-ups:

  * **32x spatial compression** in a single autoencoder (one stage),
    vs the standard 8x in SD/FLUX/Qwen-Image. A 2048x2048 image maps
    to a 64x64 latent.
  * **64 latent channels** (vs 16 in FLUX.1, 32 in FLUX.2).
  * **DINOv2 alignment**: the VAE feature space is regularized to align
    with Meta's DINOv2 features so semantic information survives the
    aggressive compression. This is an analysis-time regularization
    only; the runtime decoder is a plain conv-transpose stack.
  * 4 to 6 upsampling stages (depending on whether 32x is achieved as
    2^5 with one identity-stride stage, or 2^5 = 32 with five stages).

----------------------------------------------------------------------------
GAPS — fill in on a GPU host:

  1. Confirm latent_channels / down_block_types / up_block_types /
     block_out_channels / layers_per_block from ``vae/config.json``.

  2. Confirm activation (silu vs gelu) and norm (group_norm vs RMSNorm).
     Tencent VAEs sometimes swap to RMSNorm for memory reasons.

  3. Confirm scaling_factor / shift_factor (latents are commonly
     normalized to N(0,1) via (latent - shift) * scale).

  4. Decide whether to reuse ``flux.flux_vae_builder`` (which is a thin
     wrapper around ``diffusers.AutoencoderKL.decode()`` via TRT) or
     write a dedicated builder. Reuse is the right starting point
     because diffusers' AutoencoderKLHunyuanImage will be loadable via
     ``AutoencoderKL.from_pretrained`` once the diffusers PR lands.

This scaffold raises NotImplementedError but pins the expected I/O
contract so the plugin can be wired end-to-end.
"""
from __future__ import annotations


DEFAULT_VAE_LATENT_CHANNELS = 64
DEFAULT_VAE_SCALE_FACTOR_SPATIAL = 32
DEFAULT_VAE_SCALING_FACTOR = 1.0
DEFAULT_VAE_SHIFT_FACTOR = 0.0


def build_hunyuan_image_vae_decoder_engine(
    vae_dir: str,
    *,
    latent_channels: int = DEFAULT_VAE_LATENT_CHANNELS,
    h_lat: int,
    w_lat: int,
    scaling_factor: float = DEFAULT_VAE_SCALING_FACTOR,
    shift_factor: float = DEFAULT_VAE_SHIFT_FACTOR,
    verbose: bool = False,
    build_timing: dict | None = None,
    timing_component: str = "vae_decoder",
) -> bytes:
    """Build the HunyuanImage VAE decoder TRT engine plan.

    Expected engine I/O:
        Input  : latents [B, latent_channels, h_lat, w_lat] float32
        Output : image   [B, 3, h_lat * scale, w_lat * scale] float32

    NOT YET IMPLEMENTED. To implement:
      1. Try delegating to ``flux.flux_vae_builder.build_flux_vae_decoder_engine``
         first; if AutoencoderKLHunyuanImage loads via diffusers
         ``AutoencoderKL.from_pretrained``, the FLUX path's ONNX-export
         pipeline will work transparently.
      2. If diffusers does not yet support the model class natively,
         port the relevant blocks directly into a TRT graph. The pattern
         to follow is ``qwen_image.qwen_image_vae_builder`` -- it builds
         a single-stage decoder manually because Qwen-Image also has a
         non-standard VAE.
      3. Cache the ONNX intermediate under
         ``$TRTMC_CACHE_DIR/hunyuan_image/vae_decoder.onnx`` so
         repeated builds are fast.

    GAP: the actual spatial scale factor (32x assumed) must be verified
    against the VAE config. If the VAE is 8x like FLUX/SD3, then the
    DiT operates on a 256x256 grid for 2K outputs and num_img_tokens
    must be recomputed.
    """
    raise NotImplementedError(
        "HunyuanImage VAE decoder builder is a scaffold. Start by "
        "delegating to families.flux.flux_vae_builder once "
        "AutoencoderKLHunyuanImage is loadable through diffusers."
    )
