#pragma once
// diffusion_types.h — Shared types for diffusion pipelines.
// Extracted from old diffusion_backend.h during TrtModule migration.

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

/// Configuration for the diffusion pipeline.
struct DiffusionConfig {
    std::string scheduler{"flow_match_euler"};
    int32_t num_inference_steps{50};
    float guidance_scale{5.0F};
    float flow_shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{1.15F};
    int32_t base_image_seq_len{256};
    int32_t max_image_seq_len{4096};
    float shift_terminal{0.0F};

    // Per-call requested batch size. Defaults to 1 to preserve existing
    // single-image behavior for old call sites.
    int32_t batch_size{1};

    struct {
        int32_t dit{1};
        int32_t text_encoder{1};
        int32_t vae{1};
    } max_batch_size;

    int32_t video_height{480};
    int32_t video_width{832};
    int32_t video_num_frames{81};

    int32_t z_dim{16};
    int32_t scale_factor_temporal{4};
    int32_t scale_factor_spatial{8};
    int32_t dit_dim{1536};
    int32_t dit_num_heads{12};
    int32_t freq_dim{256};
    int32_t text_seq_len{512};
    int32_t text_encoder_dim{4096};

    int32_t num_vae_caches{0};
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    std::vector<int32_t> patch_size;     // [pt, ph, pw]
    std::vector<int32_t> axes_dims_rope; // RoPE axis dimensions
    float rope_theta{10000.0F};
    std::string vae_model_id;

    bool guidance_embeds{false};
    bool use_rope{true};
    float vae_scaling_factor{0.0F};

    std::string diffusion_backend_type{"wan_3d"};
};

/// Preprocessor weights for the DiT (external to the TRT engine graph).
struct PreprocessorWeights {
    // Patch embedding (Conv3D weights, used as matmul)
    std::vector<float> patch_embed_weight; // [patch_dim, dit_dim]
    std::vector<float> patch_embed_bias;   // [dit_dim]
    int32_t patch_dim{0};

    // TimestepEmbedding MLP
    std::vector<float> time_emb_0_weight; // [freq_dim, dim]
    std::vector<float> time_emb_0_bias;   // [dim]
    std::vector<float> time_emb_2_weight; // [dim, dim]
    std::vector<float> time_emb_2_bias;   // [dim]

    // time_proj
    std::vector<float> time_proj_weight; // [dim, 6*dim]
    std::vector<float> time_proj_bias;   // [6*dim]

    // Text projection MLP
    std::vector<float> text_proj_weight;   // [text_dim, dim]
    std::vector<float> text_proj_bias;     // [dim]
    std::vector<float> text_proj_2_weight; // [dim, dim]
    std::vector<float> text_proj_2_bias;   // [dim]

    // Context embedder (FLUX)
    std::vector<float> context_embed_weight; // [text_encoder_dim, dit_dim]
    std::vector<float> context_embed_bias;   // [dit_dim]

    // Guidance embedding MLP (FLUX)
    std::vector<float> guidance_emb_0_weight;
    std::vector<float> guidance_emb_0_bias;
    std::vector<float> guidance_emb_2_weight;
    std::vector<float> guidance_emb_2_bias;

    // VAE BN denormalization (FLUX.2)
    std::vector<float> vae_bn_mean;
    std::vector<float> vae_bn_var;

    bool valid{false};
};

/// Result of video generation.
struct VideoResult {
    std::vector<float> frames; // [T, H, W, 3] float in [0,1]
    int32_t num_frames{0};
    int32_t height{0};
    int32_t width{0};
};

/// Parse preprocessor weights from bundle section bytes.
PreprocessorWeights parse_preprocessor_weights(const std::vector<char>& data);

} // namespace trtmc
