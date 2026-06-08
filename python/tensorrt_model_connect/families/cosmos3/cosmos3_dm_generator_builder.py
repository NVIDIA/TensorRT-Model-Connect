"""Cosmos 3 diffusion (DM) generator TRT builder.

The DM generator is the diffusion-side subsequence of the Cosmos 3
Mixture-of-Transformers. It shares the same transformer body shape as the
AR reasoner (64 layers, hidden=5120, 64 heads with 8 KV heads / GQA,
head_dim=128, FFN=25600 SwiGLU) but adds:

  - **Diffusion-side input projections**: video latents
    (T × H/16 × W/16 × z_dim=48) are unfolded with latent_patch_size=2 into
    DM tokens of patch_latent_dim = 48 × 2 × 2 = 192, then linearly
    projected to hidden=5120.
  - **Modality tokens**: action trajectories (32 embodiments × 64-D padded),
    audio latents (sound_dim=64 at 25 Hz FPS), and image/video latents share
    the DM token stream alongside text tokens from the AR lane.
  - **Time conditioning**: a sinusoidal timestep embedding with
    timestep_scale=0.001 feeds into adaLN-Zero modulators on every block.
  - **3-axis unified-mrope RoPE**: rope_theta=5e6 with
    mrope_section=[24, 20, 20] across (T, H, W) axes —
    unified_3d_mrope_reset_spatial_ids=True applies sequence-level resets
    when crossing the temporal_modality_margin=15000 boundary.
  - **Per-modality FFN experts** (``use_moe=True``): the LoRA target list
    (``q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen``)
    indicates the generator pathway carries dedicated QKV-O expert
    projections separate from the reasoner pathway.
  - **Two-way joint attention**: each block's attention pool sees the
    concatenated [AR_tokens | DM_tokens] sequence, with AR causal among
    itself, DM bidirectional among itself, and cross-direction free.
    Inside this builder we expose the attention as standard self-attention
    over the DM-token sub-range; the C++ runtime (Phase 6) wires the AR
    tokens in via the unified KV cache.

This file currently provides:
  - All architectural constants (locked from transformer/config.json)
  - The public entry point ``build_cosmos3_dm_generator_engine``
  - A precise documented layout of the per-layer forward pass

The actual TRT graph construction in ``build_cosmos3_dm_generator_engine``
raises ``NotImplementedError`` listing the specific TRT graph layers that
remain to be written. The shape of each layer is fully specified below so
the next iteration can fill in the graph construction without re-deriving
dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig
    from ...parallel_config import ParallelConfig


# === Backbone (shared shape with AR reasoner — see ARCH.md) ===

COSMOS3_SUPER_DM_HIDDEN_SIZE = 5120
COSMOS3_SUPER_DM_NUM_LAYERS = 64
COSMOS3_SUPER_DM_NUM_HEADS = 64
COSMOS3_SUPER_DM_NUM_KV_HEADS = 8
COSMOS3_SUPER_DM_HEAD_DIM = 128
COSMOS3_SUPER_DM_INTERMEDIATE_SIZE = 25600
COSMOS3_SUPER_DM_RMS_NORM_EPS = 1e-6
COSMOS3_SUPER_DM_ROPE_THETA = 5_000_000
COSMOS3_SUPER_DM_MROPE_SECTION: Tuple[int, int, int] = (24, 20, 20)
COSMOS3_SUPER_DM_QK_NORM = True
COSMOS3_SUPER_DM_HIDDEN_ACT = "silu"

# === Diffusion-side projections (from transformer/config.json) ===

COSMOS3_SUPER_DM_LATENT_CHANNEL = 48          # matches VAE z_dim
COSMOS3_SUPER_DM_LATENT_PATCH_SIZE = 2
COSMOS3_SUPER_DM_PATCH_LATENT_DIM = 192       # 48 * 2 * 2
COSMOS3_SUPER_DM_TIMESTEP_SCALE = 0.001
COSMOS3_SUPER_DM_BASE_FPS = 24
COSMOS3_SUPER_DM_ENABLE_FPS_MODULATION = True
COSMOS3_SUPER_DM_POSITION_EMBEDDING_TYPE = "unified_3d_mrope"
COSMOS3_SUPER_DM_TEMPORAL_MODALITY_MARGIN = 15_000
COSMOS3_SUPER_DM_RESET_SPATIAL_IDS = True

# === Modality embeddings ===

COSMOS3_SUPER_DM_MAX_ACTION_DIM = 64
COSMOS3_SUPER_DM_NUM_EMBODIMENT_DOMAINS = 32
COSMOS3_SUPER_DM_SOUND_DIM = 64
COSMOS3_SUPER_DM_SOUND_LATENT_FPS = 25

# === MoE / per-modality expert configuration ===

# Per ``transformer/config.json``: ``use_moe=True`` with LoRA targets
# ``q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen``. The
# generator pathway has its own QKV-O expert projections distinct from the
# reasoner pathway. The exact number of FFN experts and routing strategy
# need verification against the safetensors weight key list — these
# constants are placeholders pending that inspection.
COSMOS3_SUPER_DM_NUM_EXPERTS = None  # TODO: confirm from safetensors
COSMOS3_SUPER_DM_LORA_RANK = 16      # from config.json; only used if LoRA enabled
COSMOS3_SUPER_DM_LORA_ALPHA = 32

# === Sampling (rectified-flow / UniPC scheduler) ===

COSMOS3_SUPER_DM_NUM_TRAIN_TIMESTEPS = 1000
COSMOS3_SUPER_DM_SHIFT_BY_RESOLUTION = {256: 3, 480: 5, 720: 10}


def cosmos3_dm_num_patches(num_frames: int, height: int, width: int) -> int:
    """Return the number of DM tokens for a given (T, H, W) latent shape.

    Cosmos 3 patchifies (T_latent, H_latent / 16, W_latent / 16) latents at
    a fixed spatial patch_size = 2 (no temporal patching — temporal
    compression happens entirely inside the VAE at scale_factor_temporal=4).

    Args:
      num_frames: number of *output* frames (5-400 per the HF model card).
        Internally this is ceil(num_frames / scale_factor_temporal) latent
        frames, with one extra "I-frame" for image conditioning.
      height: output frame height in pixels (256/480/720).
      width: output frame width in pixels.

    Returns:
      The DM token count after patchify.
    """
    # VAE: spatial 16x, temporal 4x.
    h_lat = height // 16
    w_lat = width // 16
    t_lat = max(1, num_frames // 4)
    # Patch size 2 on spatial axes only.
    return t_lat * (h_lat // COSMOS3_SUPER_DM_LATENT_PATCH_SIZE) * (
        w_lat // COSMOS3_SUPER_DM_LATENT_PATCH_SIZE)


def build_cosmos3_dm_generator_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    video_num_frames: int,
    video_height: int,
    video_width: int,
    num_inference_steps: int = 30,
    precision: str = "bf16",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config: Optional["ParallelConfig"] = None,
) -> bytes:
    """Build the Cosmos 3 DM generator TRT engine (single-lane diffusion).

    Args:
      config: ``ModelConfig`` carrying the transformer/config.json fields.
      weights: weights from ``Cosmos3Plugin.load_weights``.
      video_num_frames: target output frame count (5-400).
      video_height: target frame height (256/480/720).
      video_width: target frame width.
      num_inference_steps: number of denoising steps the scheduler will
        request from the engine. Defaults to 30.
      precision: ``bf16`` recommended (Cosmos 3 is bf16-only).
      quant_ctx: optional quantization context.
      verbose: verbose engine build logging.
      parallel_config: optional ``ParallelConfig`` for SP modes
        (``sp_ulysses`` / ``sp_ring`` / ``sp_allgather_kv``). Single-lane
        operation lives in ``single`` / ``tensor_parallel`` modes;
        sequence-parallel modes (Phase-2 SP infra from PR #205) require
        adaptations to the patch-token sharding and joint-attention layer
        and are not implemented yet.

    Returns:
      Serialized TRT engine bytes for the DM generator (denoiser).

    Layer plan (per per-layer block i, i ∈ [0, 64)):
      1. input_norm = RMSNorm(eps=1e-6)(dm_tokens)             [N, H=5120]
      2. q/k/v MoE projection:
            q = Linear(q_proj_moe_gen)(input_norm) [N, num_heads * head_dim]
            k = Linear(k_proj_moe_gen)(input_norm) [N, num_kv * head_dim]
            v = Linear(v_proj_moe_gen)(input_norm) [N, num_kv * head_dim]
      3. q,k = RMSNorm(qk_norm_for_diffusion=True)(q),(k)
      4. apply 3-axis RoPE (mrope_interleaved=[24,20,20]) to q,k
      5. attn = scaled_dot_product_attention(q, k, v)
            mask handles the AR/DM two_way concatenation at runtime
      6. o = Linear(o_proj_moe_gen)(attn) [N, H]
      7. dm_tokens = dm_tokens + adaLN-Zero-scale * o      # post-attn residual
      8. ffn_norm = RMSNorm(eps=1e-6)(dm_tokens)
      9. gate, up = Linear(gate_proj), Linear(up_proj)(ffn_norm) [N, FFN=25600]
     10. mid = silu(gate) * up
     11. ffn_out = Linear(down_proj)(mid)
     12. dm_tokens = dm_tokens + adaLN-Zero-scale * ffn_out  # post-FFN residual

    Time / modality conditioning enters through adaLN-Zero scale/shift
    parameters: t_emb is broadcast across DM tokens, sliced into
    (scale_attn, shift_attn, scale_ffn, shift_ffn) per block.

    Status:
      Not yet implemented at the TRT graph level. The dimensions documented
      above are sufficient to construct the layer-by-layer graph in a
      follow-up iteration; this builder serves as the architectural source
      of truth for the next coder.
    """
    raise NotImplementedError(
        "Cosmos 3 DM generator TRT graph construction not yet implemented.\n"
        "Required follow-up work (in this file):\n"
        "  - Input patchify: Conv3d-style unfold of [B, T_lat, 48, H_lat, "
        "W_lat] → DM tokens [N, 192] → Linear(192 → 5120)\n"
        "  - Time embed: sinusoidal(t * 0.001) → MLP → 5120-d t_emb; per-layer"
        " split into adaLN-Zero scale/shift\n"
        "  - Action embed: Linear(64 → 5120) + Embedding(32 → 5120) for "
        "embodiment domain\n"
        "  - Sound embed: project 64-d sound latents into 5120 (Phase 5+ work)"
        "\n"
        "  - 64 backbone blocks per the layer plan in this docstring\n"
        "  - Output unpatchify: Linear(5120 → 192) → unfold → [B, T_lat, 48, "
        "H_lat, W_lat]\n"
        "Tensor-parallel sharding follows the same column-/row-shard pattern"
        " as the AR reasoner; sequence-parallel (sp_*) modes shard along the"
        " DM token dimension and need re-tiling at the joint-attention layer."
        "\nThe two-way AR↔DM joint attention is exposed as plain self-attn at"
        " engine-build time; the C++ runtime (Phase 6) is responsible for "
        "concatenating AR tokens into the KV stream at decoding time.")
