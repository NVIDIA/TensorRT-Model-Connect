# Cosmos 3 — locked architecture (sourced from `nvidia/Cosmos3-Super` config.json)

Fetched 2026-06-05. The `config.json` reveals that the "Mixture of Transformers"
labelling in the marketing material is **a single transformer body** with both
AR (text) and DM (continuous) tokens flowing through the same layers, separated
by a `joint_attn_implementation: "two_way"` mechanism. The MoE flag
(`use_moe: true`) refers to per-modality experts inside the FFN, not separate
transformer stacks.

## Reusable pieces

| Component        | Reuses                              | Notes                                                |
|------------------|-------------------------------------|------------------------------------------------------|
| Reasoner / DM backbone | Qwen3-VL 32B Instruct (text)  | Single transformer body — see "Backbone" below       |
| ViT visual encoder | Qwen3-VL Vision (`Qwen3VLVisionModel`) | Identical to qwen_vl family's vision tower      |
| VAE              | `AutoencoderKLWan` (Wan 2.2 TI2V-5B) | z_dim=48 (Wan 2.2 variant, not Wan 2.1's 16)        |
| Scheduler        | `UniPCMultistepScheduler`           | Already wired in our diffusion runtime               |
| Tokenizer (text) | `Qwen2TokenizerFast`                | Same as qwen_vl                                      |
| Audio tokenizer  | `Cosmos3AVAEAudioTokenizer`         | **NEW** — 48 kHz noncausal AVAE, 25 Hz, 64 channels  |

## Backbone (single transformer, ~32B params for Super)

From `transformer/config.json`:

| Field                       | Value                          |
|-----------------------------|--------------------------------|
| hidden_size                 | 5120                           |
| num_hidden_layers           | 64                             |
| num_attention_heads         | 64                             |
| num_key_value_heads         | 8 (8:1 GQA)                    |
| head_dim                    | 128                            |
| intermediate_size           | 25600                          |
| hidden_act                  | silu (SwiGLU)                  |
| qk_norm_for_text            | true                           |
| qk_norm_for_diffusion       | true                           |
| rms_norm_eps                | 1e-6                           |
| rope_theta                  | 5,000,000                      |
| rope_scaling                | mrope_interleaved              |
| mrope_section               | [24, 20, 20]                   |
| max_position_embeddings     | 262144 (256K context)          |
| vocab_size                  | 151936                         |
| dtype                       | bfloat16                       |
| use_moe                     | true (per-modality FFN experts)|
| joint_attn_implementation   | "two_way"                      |

**Joint attention "two_way"**: AR and DM tokens share the same attention pool
in each layer — bidirectional self-attention over the concatenated
`[AR_tokens, DM_tokens]` sequence. AR tokens use causal masking among
themselves; DM tokens are non-causal among themselves and can attend to all
AR tokens; AR tokens can also attend to DM tokens (hence "two-way"). The exact
mask layout still needs verification from the reference forward pass.

**Per-modality experts (`use_moe: true`)**: From the LoRA target list
(`q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen`) it appears the
generation pathway has dedicated QKV-O projection experts. Possibly an MoE in
the FFN too — verify by inspecting safetensors weight key list.

## Diffusion-specific projections

From `transformer/config.json`:

| Field                       | Value           |
|-----------------------------|-----------------|
| patch_latent_dim            | 192             |
| latent_patch_size           | 2               |
| latent_channel              | 48 (matches VAE z_dim) |
| max_action_dim              | 64              |
| num_embodiment_domains      | 32              |
| sound_dim                   | 64              |
| sound_latent_fps            | 25              |
| timestep_scale              | 0.001           |
| base_fps                    | 24              |
| position_embedding_type     | unified_3d_mrope |
| unified_3d_mrope_reset_spatial_ids | true     |
| unified_3d_mrope_temporal_modality_margin | 15000 |

## Vision encoder (Qwen3-VL Vision)

From `vision_encoder/config.json`:

| Field               | Value             |
|---------------------|-------------------|
| depth               | 27                |
| hidden_size         | 1152              |
| num_heads           | 16                |
| head_dim            | 72 (1152/16)      |
| intermediate_size   | 4304              |
| hidden_act          | gelu_pytorch_tanh |
| out_hidden_size     | 5120              |
| patch_size          | 16                |
| temporal_patch_size | 2                 |
| spatial_merge_size  | 2                 |
| in_channels         | 3                 |
| num_position_embeddings | 2304          |
| deepstack_visual_indexes | [8, 16, 24]  |

`deepstack_visual_indexes` indicates the ViT exports intermediate features
from layers 8, 16, 24 in addition to the final layer — feature pyramid input
to the LLM head, identical to Qwen3-VL.

## Video VAE (`AutoencoderKLWan` — Wan 2.2 variant)

From `vae/config.json`:

| Field                       | Value                          |
|-----------------------------|--------------------------------|
| base_dim                    | 160                            |
| decoder_base_dim            | 256                            |
| dim_mult                    | [1, 2, 4, 4]                   |
| num_res_blocks              | 2                              |
| in_channels / out_channels  | 12                             |
| z_dim                       | 48                             |
| patch_size                  | 2                              |
| scale_factor_spatial        | 16                             |
| scale_factor_temporal       | 4                              |
| temperal_downsample         | [false, true, true]            |
| is_residual                 | true                           |
| _name_or_path               | Wan-AI/Wan2.2-TI2V-5B-Diffusers |

The existing `families/wan_t2v/causal_vae_3d_builder.py` targets Wan 2.1
(base_dim=96, z_dim=16). The Wan 2.2 VAE used here is the same architecture
shape but wider (base_dim=160, z_dim=48) — the builder needs new size
constants, not a structural rewrite.

## Audio tokenizer (new component)

From the `sound_tokenizer` block in `config.json`:

| Field          | Value                                     |
|----------------|-------------------------------------------|
| sample_rate    | 48000                                     |
| hop_size       | 1920 (→ 25 Hz latent FPS)                 |
| io_channels    | 64                                        |
| audio_channels | 2 (stereo)                                |
| name           | avae_48k_noncausal_25hz_64ch              |
| tanh_clamp     | 0.995                                     |
| tanh_input_scale | 1.5                                     |
| tanh_output_scale | 3.5                                    |

`Cosmos3AVAEAudioTokenizer` is a new diffusers class — needs its own builder.
Out of scope for the initial video-generation lane.

## Sampling (rectified flow / unipc)

From `scheduler/scheduler_config.json` and the rectified-flow inference block
in `config.json`:

- scheduler_type: unipc
- num_train_timesteps: 1000
- shift schedule by resolution: {256: 3, 480: 5, 720: 10}
- shift: 1 (inference default)
- use_dynamic_shifting: false

## Variants

- **Cosmos3-Super** (this config): 64 layers, 5120 hidden, 64 heads, 25600 FFN → ~32B reasoner + ~32B generator
- **Cosmos3-Nano**: not yet fetched; expected to share architecture shape at smaller scale (~16 layers / 2048 hidden / 16 heads, matching the 8B reasoner + 8B generator split). Pull and verify before scaffolding the Nano lane.

## Bring-up order (revised after architecture lock)

1. **AR reasoner builder** — clone the qwen_vl text TP builder, swap configs:
   `hidden=5120, layers=64, heads=64/8, ffn=25600, head_dim=128, rope_theta=5e6,
   mrope_section=[24,20,20]`. Test against the reasoner-only path
   (text→text) using the Qwen3-VL reasoner backbone weights.
2. **DM generator builder** — same backbone shape, add the diffusion-side
   projections (patch_latent_dim=192, latent_patch_size=2, latent_channel=48)
   and the per-modality FFN experts. SP-ready via the parallel_config.sp_*
   modes from #205.
3. **ViT** — reuse `qwen_vl` vision tower; verify weight key mapping for
   `Qwen3VLVisionModel` keys (deepstack indices 8/16/24).
4. **Wan 2.2 VAE** — parametrize `wan_t2v.causal_vae_3d_builder` for base_dim=160
   and z_dim=48; or fork a small `wan_t2v_2_2_vae_builder.py`.
5. **Joint attention runtime** — extend the C++ diffusion pipeline to maintain
   `[AR_tokens | DM_tokens]` concatenated sequence with the two-way mask
   pattern; AR token generation interleaves with DM denoising steps.
6. **Action / audio encoders + scheduler glue** — last (only needed for
   action-output and audio-output capabilities; text→video doesn't need them).
