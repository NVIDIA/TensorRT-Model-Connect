"""VAE (Cosmos-Tokenizer CV8x8x8) decoder builder for Cosmos-Transfer1-7B.

Cosmos-Transfer1 uses NVIDIA's Cosmos-Tokenizer family for the latent-space
autoencoder. The 7B model is paired with CV8x8x8: a continuous-latent video
tokenizer with 8x spatial compression and 8x temporal compression. Key
architectural details (from the Cosmos-Tokenizer repo / model card):

  * Encoder/decoder start/end with a 2-level Haar wavelet transform (4x
    downsample / upsample on both spatial and temporal axes).
  * Continuous AE formulation (not FSQ); latent dim ``z_dim = 16``.
  * Spatial compression factor: 8        (=> H_lat = H / 8)
  * Temporal compression factor: 8       (=> T_lat = (T - 1) / 8 + 1)
  * Total compression: 8 * 8 * 8 = 512x

For the trtmc engine we only need the *decoder* (the runtime takes
denoised latents from the DiT and decodes them into pixel-space frames).

This builder is currently a *signature-only* stub — the Cosmos-Tokenizer
decoder graph (Haar inverse + residual blocks + attention) is substantial
and needs its own dedicated PR. See the open questions at the bottom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# Cosmos-Tokenizer CV8x8x8 defaults.
Z_DIM = 16
SPATIAL_COMPRESSION = 8
TEMPORAL_COMPRESSION = 8


def load_vae_weights(pt_or_dir: str) -> "WeightDict":
    """Load Cosmos-Tokenizer decoder weights.

    Handles two on-disk layouts: a single ``cosmos_tokenizer.pt`` flat file,
    or a ``cosmos_tokenizer/`` sub-dir with sharded ``*.pt`` files. The
    decoder-only keys are kept; encoder keys are dropped to save memory.
    """
    from pathlib import Path

    from ...checkpoint_mapper import WeightDict
    from .pt_loader import load_pt_state_dict

    p = Path(pt_or_dir)
    raw: dict = {}
    if p.is_file():
        raw = load_pt_state_dict(p)
    elif p.is_dir():
        # Merge all .pt shards in directory order.
        for shard in sorted(p.glob("*.pt")):
            raw.update(load_pt_state_dict(shard))
    else:
        raise FileNotFoundError(f"Cosmos VAE path is neither file nor dir: {p}")

    w = WeightDict()
    w["_role"] = "vae_decoder"
    w["_source_pt"] = str(p)
    # Keep only decoder-side parameters.
    for k, v in raw.items():
        if k.startswith("decoder.") or k.startswith("post_quant_conv."):
            w[k] = v
    return w


def build_cosmos_vae_decoder_engine(
    weights: "WeightDict",
    *,
    z_dim: int = Z_DIM,
    h_lat: int,
    w_lat: int,
    t_lat: int,
    verbose: bool = False,
) -> bytes:
    """Build the Cosmos-Tokenizer CV8x8x8 decoder TRT engine.

    Engine I/O:
        Input:
            latent      [1, z_dim, t_lat, h_lat, w_lat] float32
        Output:
            video       [1, 3, t_pix, h_pix, w_pix] float32
                t_pix = (t_lat - 1) * 8 + 1
                h_pix = h_lat * 8
                w_pix = w_lat * 8

    Open questions:
      (1) Inverse Haar wavelet — TRT has no built-in op for this; needs to
          be expressed as a custom layer or unrolled into a sequence of
          conv + slice ops. Cosmos-Tokenizer reference uses
          ``torch.nn.functional.conv3d`` with hand-crafted wavelet kernels.
      (2) Spatial vs spatio-temporal attention blocks — Cosmos uses both.
          Need to confirm block ordering against the released checkpoint.
      (3) Causal padding policy on the temporal dim — affects the (T - 1)
          offset in T_lat computation.
      (4) BFloat16 dtype: the tokenizer ships fp32 weights in some releases
          and bf16 in others. pt_loader promotes bf16 -> fp32 already, but
          we should confirm fp32 build path is numerically stable.
    """
    raise NotImplementedError(
        "cosmos_vae_builder is a stub. The Cosmos-Tokenizer CV8x8x8 "
        "decoder needs its own builder file in a follow-up PR; the "
        "structural layout (Haar inverse + residual + attention) is "
        "documented in github.com/NVIDIA/Cosmos-Tokenizer. Implementing "
        "this in TRT requires either: a custom plugin for the inverse "
        "Haar wavelet, or expressing it as a fixed conv3d (preferred — "
        "the Haar kernels are constants)."
    )
