#pragma once

// ZImagePipeline: Z-Image diffusion pipeline with Qwen3 text encoder,
// denoiser, and VAE. Uses TrtModule::forward() for all GPU work.

#include "runtime/domains/diffusion/diffusion_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

/// Internal layout descriptor for Z-Image (latent grid, patches, dims).
/// Declared here so that ``ZImagePipeline`` private helpers can refer to it
/// without dragging the full pipeline body into the header.
struct ZImageLayout {
    int32_t dit_dim{0};
    int32_t text_seq{0};
    int32_t z_dim{0};
    int32_t h_lat{0};
    int32_t w_lat{0};
    int32_t ph{2};
    int32_t pw{2};
    int32_t nh{0};
    int32_t nw{0};
    int32_t num_patches{0};
    int32_t patch_dim{0};
    int32_t head_dim{0};
};

/// Z-Image-specific preprocessor weights (separate from standard PreprocessorWeights)
struct ZImagePreprocessorWeights {
    std::vector<float> t_embedder_mlp_0_weight;
    std::vector<float> t_embedder_mlp_0_bias;
    std::vector<float> t_embedder_mlp_2_weight;
    std::vector<float> t_embedder_mlp_2_bias;
    std::vector<float> cap_proj_weight;
    std::vector<float> cap_proj_bias;
    std::vector<float> cap_norm_weight;
    std::vector<float> cap_pad_token;
    std::vector<float> x_embed_weight;
    std::vector<float> x_embed_bias;
    int32_t cap_dim{0};
    int32_t dit_dim{0};
    int32_t freq_dim{0};
    bool valid{false};
};

class ZImagePipeline final : public IPipeline {
  public:
    ZImagePipeline(std::unique_ptr<TrtModule> text_encoder, std::unique_ptr<TrtModule> denoiser,
                   std::unique_ptr<TrtModule> vae, DiffusionConfig config,
                   PreprocessorWeights weights, ZImagePreprocessorWeights z_weights,
                   std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                   std::string bundle_path, std::shared_ptr<void> distributed_owner = {},
                   int32_t tensor_parallel_rank = 0, int32_t tensor_parallel_size = 1);

    ~ZImagePipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    std::vector<ImageResult>
    generate_image_batch(const std::vector<std::string>& prompts,
                         const std::vector<std::uint32_t>& per_sample_seeds,
                         const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "ZImagePipeline"; }

  private:
    bool run_text_encoder(const std::vector<int32_t>& input_ids,
                          std::vector<float>& text_embeddings);
    bool run_denoiser(const std::vector<float>& hidden, const std::vector<float>& encoder_hidden,
                      const std::vector<float>& temb, const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals, std::vector<float>& output);

    /// Batched DiT forward. Every input carries a leading batch dim of size
    /// ``batch_size`` and the output is contiguous ``[B, num_patches,
    /// patch_dim]``. See ``generate_image_batch`` for the chunking contract.
    bool run_denoiser_batched(const std::vector<float>& hidden,
                              const std::vector<float>& encoder_hidden,
                              const std::vector<float>& temb, const std::vector<float>& cos_vals,
                              const std::vector<float>& sin_vals, int32_t batch_size,
                              int32_t num_patches, int32_t dit_dim, int32_t text_seq,
                              int32_t freq_dim, int32_t total_seq, int32_t head_dim,
                              int32_t patch_dim, std::vector<float>& output);

    void project_caption(const std::vector<float>& text_emb, int32_t actual_len, int32_t padded_len,
                         std::vector<float>& projected) const;
    void compute_3d_rope(int32_t cap_padded_len, int32_t num_patches, int32_t nh, int32_t nw,
                         std::vector<float>& cos_out, std::vector<float>& sin_out) const;
    void patchify_2d(const std::vector<float>& latents, int32_t c, int32_t h, int32_t w,
                     std::vector<float>& patches) const;
    void unpatchify_2d(const std::vector<float>& patches, int32_t c, int32_t h, int32_t w,
                       std::vector<float>& output) const;
    ImageResult decode_z_image_result(int32_t z_dim, int32_t h_lat, int32_t w_lat,
                                      std::vector<float>& latents, ImageResult result);

    // ---- generate_image_batch helpers (PR2 refactor) ---------------------
    // These split the per-chunk body of ``generate_image_batch`` into smaller
    // pieces so the outer function stays under the team's CCN gate. They are
    // pure implementation details and never observable outside this class.

    /// Resolve the per-chunk batch cap for the DiT engine. Combines the
    /// configured ``max_batch_size.dit`` with the engine profile's kMax for
    /// the ``hidden_states`` input. Returns 1 when the engine is the legacy
    /// static (rank-2) build.
    int32_t resolve_batch_cap(bool engine_is_batched) const;

    /// Generate one chunk of images. ``prompt_offset`` is the absolute index
    /// of ``prompts[0]`` in the original batch (used for logging only). On
    /// any internal failure the returned vector is empty, mirroring the
    /// caller's existing fail-fast semantics.
    std::vector<ImageResult> generate_image_batch_chunk(
        const std::vector<std::string>& prompts, const std::vector<std::uint32_t>& resolved_seeds,
        std::size_t prompt_offset, int32_t batch, const ZImageLayout& layout,
        int32_t num_inference_steps, int32_t freq_dim, bool engine_is_batched, int32_t cap);

    /// Run the Qwen3 text encoder + caption projection + RoPE + latent
    /// initialization for every sample in a chunk. Fills the four batched
    /// buffers in-place. Returns false on encoder failure.
    bool run_qwen3_encoder_for_chunk(const std::vector<std::string>& prompts,
                                     const std::vector<std::uint32_t>& resolved_seeds,
                                     std::size_t prompt_offset, int32_t batch,
                                     const ZImageLayout& layout,
                                     std::vector<float>& caption_projected_b,
                                     std::vector<float>& rope_cos_b, std::vector<float>& rope_sin_b,
                                     std::vector<float>& latents);

    /// Run the FlowMatchEuler denoise loop for one chunk. ``latents`` is
    /// updated in-place. Returns false if any DiT invocation fails.
    bool run_denoise_loop_for_chunk(int32_t batch, int32_t num_inference_steps, int32_t freq_dim,
                                    bool engine_is_batched, std::size_t prompt_offset,
                                    const ZImageLayout& layout,
                                    const std::vector<float>& caption_projected_b,
                                    const std::vector<float>& rope_cos_b,
                                    const std::vector<float>& rope_sin_b,
                                    std::vector<float>& latents);

    /// Per-step fallback when the DiT engine is the legacy static (rank-2)
    /// build: invoke ``run_denoiser`` once per sample in the chunk and pack
    /// the outputs contiguously. Extracted so ``run_denoise_loop_for_chunk``
    /// itself stays under the CCN gate.
    bool run_denoiser_unbatched_step(
        int32_t batch, int32_t step, std::size_t prompt_offset, std::size_t hidden_size,
        std::size_t caption_size, std::size_t rope_size, std::size_t patch_size,
        const std::vector<float>& hidden_b, const std::vector<float>& caption_projected_b,
        const std::vector<float>& temb_one, const std::vector<float>& rope_cos_b,
        const std::vector<float>& rope_sin_b, std::vector<float>& denoiser_output);

    /// VAE-decode each sample in the chunk at B=1, routing through
    /// ``decode_z_image_result`` so non-rank-0 TP ranks return empty
    /// ``ImageResult``s. Appends results in-order to ``out``.
    void decode_chunk_vae_per_sample(int32_t batch, const ZImageLayout& layout,
                                     const std::vector<float>& latents,
                                     std::vector<ImageResult>& out);

    // Keep TP communicator ownership until after TRT modules are destroyed.
    std::shared_ptr<void> distributed_owner_;
    int32_t tensor_parallel_rank_{0};
    int32_t tensor_parallel_size_{1};
    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    DiffusionConfig config_;
    PreprocessorWeights weights_;
    ZImagePreprocessorWeights z_weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::string bundle_path_;
};

} // namespace trtmc
