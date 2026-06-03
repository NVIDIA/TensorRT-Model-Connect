"""HunyuanImage-2.1 DiT (Diffusion Transformer) builder — scaffold.

HunyuanImage-2.1 is a 17B-parameter text-to-image diffusion model that
generates up to 2K (2048x2048) images. Its DiT is described publicly as
a hybrid **dual-stream + single-stream MMDiT** (combined / unified
attention over image + text tokens), similar in spirit to FLUX.1 and
Stable Diffusion 3. Tencent's reference implementation lives in
``HunyuanImage-2.1/hyimage/diffusion/transformer/``.

Released DiT variants in the HuggingFace repo:
  * ``dit/`` — base 17B BF16 DiT.
  * ``dit/refiner`` — refiner head (denoises near-final timesteps).
  * ``dit/fp8`` — FP8 quantized version of the base.
  * ``dit/distilled`` — guidance-distilled student (fewer steps, smaller
    classifier-free-guidance scale).

This scaffold targets the **base** DiT first. Refiner / FP8 / distilled
variants are stretch goals (see open questions at bottom of this file).

----------------------------------------------------------------------------
Architecture summary (per Tencent's tech notes / community VAE analysis):

  * Patch embedding: conv-based, patch_size = 2 (over the 8x-compressed
    VAE latent).
  * Joint attention: image and text tokens attend to each other in
    **dual-stream** blocks for early layers, then a **single-stream**
    stack continues over the concatenated sequence (FLUX.1 pattern).
  * Cross-attention: text conditioning comes from BOTH byT5 and Qwen2.5-VL,
    typically concatenated along the sequence axis before joint attention.
  * Position encoding: 2D RoPE on image tokens (axes_dims_rope split over
    height / width); text tokens use either zero positional offset or a
    learned positional embedding (TBD on GPU).
  * Modulation: timestep + (optional) guidance scalar -> AdaLN-style
    scale/shift/gate per block (FLUX-style).
  * Latent z_dim: 64 (the HunyuanImage VAE uses 32x spatial compression
    with 64 latent channels -- 2K image -> 64x64 latent -- per the
    "Qwen-Image vs HunyuanImage VAE" comparison in the public lilting.ch
    write-up). Confirm against ``vae/config.json`` on GPU.

NOTE on z_dim/compression: HunyuanImage-2.1 ships an unusually heavy VAE
that performs 32x spatial compression in a single stage (vs the standard
8x in FLUX/SD3). Its latent shape is ``[64, H/32, W/32]`` rather than
``[16, H/8, W/8]``. The DiT therefore operates on a much smaller token
grid for 2K outputs.

----------------------------------------------------------------------------
GAPS — must be confirmed by reading the released config on a GPU host:

  1. Exact transformer depth: published reports give 17B parameters but
     don't pin ``num_layers`` / ``num_single_layers``. Read
     ``dit/config.json`` for ``num_layers``, ``num_single_layers``,
     ``num_attention_heads``, ``attention_head_dim``, ``mlp_ratio``,
     ``in_channels``, ``axes_dims_rope``, ``guidance_embeds``,
     ``joint_attention_dim``, ``pooled_projection_dim``.

  2. Exact text encoder fusion strategy. Three candidates:
       a) Concatenate (byT5_seq, qwen_vl_seq) along the seq axis before
          feeding to context_embedder, with separate type embeddings.
       b) Two separate context_embedders, fused additively.
       c) byT5 -> token embeddings, qwen_vl -> pooled global modulation.
     Tencent's reference code in ``HunyuanImage-2.1/hyimage/`` is the
     source of truth.

  3. 2D vs 3D RoPE table layout: FLUX.1 uses ``axes_dims_rope=(16,56,56)``
     where the first axis is for text/conditioning, and (56,56) for image
     (H,W). HunyuanImage may use only (H,W) for image, with text getting
     a separate positional treatment.

  4. Modulation block structure: AdaLN-Zero (FLUX.1) or AdaLN-Single (FLUX.2 /
     SD3 with global modulation tables)? The fp8 variant being a flat
     suffix of the base implies AdaLN-Zero, but verify.

  5. Whether the DiT itself accepts a "negative byT5" or whether
     classifier-free guidance happens externally (host-side).

Until those are confirmed, this module raises NotImplementedError but
defines a stable signature so the plugin can be wired end-to-end.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# Placeholder architectural defaults — DO NOT trust these without
# verifying against the released ``dit/config.json``.
DEFAULT_DIT_DIM = 3072
DEFAULT_DIT_NUM_HEADS = 24
DEFAULT_DIT_HEAD_DIM = 128
DEFAULT_DIT_NUM_LAYERS = 19         # dual-stream blocks
DEFAULT_DIT_NUM_SINGLE_LAYERS = 38  # single-stream blocks
DEFAULT_DIT_MLP_RATIO = 4.0
DEFAULT_DIT_IN_CHANNELS = 64        # VAE z_dim (TBD; see GAP #1)
DEFAULT_DIT_PATCH_SIZE = 2
DEFAULT_DIT_AXES_DIMS_ROPE = (16, 56, 56)


def load_hunyuan_image_dit_weights(
    transformer_dir: str,
    *,
    dim: int = DEFAULT_DIT_DIM,
    num_heads: int = DEFAULT_DIT_NUM_HEADS,
    num_layers: int = DEFAULT_DIT_NUM_LAYERS,
    num_single_layers: int = DEFAULT_DIT_NUM_SINGLE_LAYERS,
) -> "WeightDict":
    """Load HunyuanImage DiT weights from a diffusers-format checkpoint.

    NOT YET IMPLEMENTED. To implement:
      1. Decide on the MMDiT key prefix used in the safetensors shards
         (likely ``transformer_blocks.{i}.*`` for joint blocks and
         ``single_transformer_blocks.{i}.*`` for single-stream, matching
         the diffusers HunyuanImageTransformer2DModel).
      2. Mirror ``flux.flux_dit_builder.load_flux_dit_weights`` for the
         dual+single stream pattern.
      3. Capture preprocessor weights (patch_embedding, time embedder,
         guidance embedder, context_embedder, byT5/Qwen-VL projections)
         separately so the C++ runtime can apply them outside the engine
         (same pattern as ``wan_t2v`` and ``flux``).
    """
    raise NotImplementedError(
        "HunyuanImage DiT weight loader is a scaffold. Implement once "
        "transformer/config.json and safetensors key prefixes have been "
        "inspected on a GPU host. See module docstring GAP #1, #4."
    )


def build_hunyuan_image_dit_engine(
    weights: "WeightDict",
    *,
    dim: int = DEFAULT_DIT_DIM,
    num_heads: int = DEFAULT_DIT_NUM_HEADS,
    num_layers: int = DEFAULT_DIT_NUM_LAYERS,
    num_single_layers: int = DEFAULT_DIT_NUM_SINGLE_LAYERS,
    num_img_tokens: int = 4096,
    byt5_seq_len: int = 128,
    qwen_vl_seq_len: int = 256,
    mlp_ratio: float = DEFAULT_DIT_MLP_RATIO,
    in_channels: int = DEFAULT_DIT_IN_CHANNELS,
    axes_dims_rope: tuple[int, ...] = DEFAULT_DIT_AXES_DIMS_ROPE,
    guidance_embeds: bool = True,
    cast_dtype: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build the HunyuanImage DiT TRT engine plan.

    NOT YET IMPLEMENTED. Once the gaps in the module docstring are
    closed, implementation should:

      1. Build a static-shape TRT network keyed on
         (num_img_tokens, byt5_seq_len, qwen_vl_seq_len) so the C++
         runtime can bind fixed buffers (same approach as Qwen-Image
         and FLUX).

      2. Reuse ``flux.flux_dit_builder`` as a template for the dual-
         stream blocks and ``flux.flux_dit_builder`` (single block
         path) for the single-stream tail. The major divergences will
         be:
           - Dual text-encoder fusion (concat byT5 + Qwen-VL on the
             sequence axis, with optional separate context projections).
           - Different RoPE schedule for 32x-compressed latents.
           - Possibly different modulation (AdaLN-Zero vs global).

      3. Mark outputs ``denoised_latents [B, num_img_tokens,
         in_channels * patch_size^2]`` so the existing
         "flux_2d"-style ``diffusion_backend_type`` runtime can patch
         it back to ``[B, in_channels, h_lat, w_lat]``.
    """
    raise NotImplementedError(
        "HunyuanImage DiT engine builder is a scaffold. See module "
        "docstring GAP #2, #3, #4 for what needs confirmation. The "
        "intended template is families.flux.flux_dit_builder + a custom "
        "dual-text-encoder fusion preamble."
    )


def serialize_hunyuan_image_preprocessor(
    dit_weights: "WeightDict", guidance_embeds: bool = True,
) -> bytes:
    """Serialize DiT preprocessor weights (patch embed, timestep MLP,
    context embedders, guidance embedder, text projections) into the
    Wan-compatible binary format consumed by the C++ runtime.

    NOT YET IMPLEMENTED. See ``flux._serialize_flux_preprocessor`` and
    ``wan_t2v._serialize_preprocessor_weights`` for the binary layout
    contract (length-prefixed JSON index + contiguous float32 data).
    HunyuanImage will add two extra entries for the dual text encoders'
    context projections (``byt5_context_embedder.*`` and
    ``qwen_vl_context_embedder.*``).
    """
    raise NotImplementedError(
        "HunyuanImage preprocessor serializer is a scaffold. The format "
        "follows the existing Wan/FLUX contract; this stub will be "
        "filled in once the dual context_embedder keys are confirmed."
    )
