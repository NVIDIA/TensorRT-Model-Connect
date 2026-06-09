#pragma once

// FluxPipeline: FLUX diffusion pipeline with T5 + CLIP text encoders,
// denoiser, and VAE. Uses TrtModule::forward() for all GPU work.

#include "runtime/domains/diffusion/diffusion_generation_plan.h"
#include "runtime/domains/diffusion/diffusion_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class FluxPipeline final : public IPipeline {
  public:
    FluxPipeline(std::vector<std::unique_ptr<TrtModule>> text_encoders,
                 std::unique_ptr<TrtModule> denoiser, std::unique_ptr<TrtModule> vae,
                 DiffusionConfig config, PreprocessorWeights weights,
                 std::shared_ptr<ITokenizer> tokenizer, std::unique_ptr<ITokenizer> clip_tokenizer,
                 std::string model_id_str, std::shared_ptr<void> distributed_owner = nullptr,
                 int32_t tensor_parallel_rank = 0, int32_t tensor_parallel_size = 1);

    ~FluxPipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    // Batched generation override: per-sample seeds, internal chunking against
    // ``config_.max_batch_size.dit``. The single-prompt ``generate_image`` is a
    // thin wrapper around this — see pipeline.cpp.
    std::vector<ImageResult>
    generate_image_batch(const std::vector<std::string>& prompts,
                         const std::vector<std::uint32_t>& per_sample_seeds,
                         const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FluxPipeline"; }

  private:
    bool run_clip_encoder(const std::vector<int32_t>& input_ids, std::vector<float>& pooled_output);
    bool run_t5_encoder(int32_t encoder_idx, const std::vector<int32_t>& input_ids,
                        std::vector<float>& text_embeddings);
    // Batched T5: returns ``[B, seq_len, te_dim]`` packed contiguous (row-major).
    bool run_t5_encoder_batch(int32_t encoder_idx,
                              const std::vector<std::vector<int32_t>>& batch_input_ids,
                              std::vector<float>& text_embeddings_batch);
    bool run_flux_denoiser(const std::vector<float>& hidden,
                           const std::vector<float>& encoder_hidden, const std::vector<float>& temb,
                           const std::vector<float>& cos_vals, const std::vector<float>& sin_vals,
                           std::vector<float>& output);
    // Batched FLUX.1 denoiser: leading dim ``B`` threaded through all five
    // inputs (hidden, encoder_hidden, temb, rotary_cos, rotary_sin). RoPE
    // tables are position-shared across the batch but still bound as
    // ``[B, total_seq, head_dim]`` because the dynamic engine input declares
    // a leading batch dim — see flux_dit_builder._build_flux_dit_dynamic.
    bool run_flux_denoiser_batch(int32_t batch, const std::vector<float>& hidden,
                                 const std::vector<float>& encoder_hidden,
                                 const std::vector<float>& temb, const std::vector<float>& cos_vals,
                                 const std::vector<float>& sin_vals, std::vector<float>& output);
    // FLUX.2: denoiser with baked temb MLP + context embedder
    bool run_flux2_denoiser(const std::vector<float>& hidden,
                            const std::vector<float>& encoder_hidden, float timestep,
                            float guidance, const std::vector<float>& cos_vals,
                            const std::vector<float>& sin_vals, std::vector<float>& output);
    bool run_flux2_denoiser_batch(int32_t batch, const std::vector<float>& hidden,
                                  const std::vector<float>& encoder_hidden, float timestep,
                                  float guidance, const std::vector<float>& cos_vals,
                                  const std::vector<float>& sin_vals, std::vector<float>& output);

    void compute_flux_timestep_embedding(float timestep, float guidance,
                                         const std::vector<float>& pooled_text,
                                         std::vector<float>& temb) const;
    void compute_flux_rope(int32_t h_patches, int32_t w_patches, int32_t text_seq_len,
                           std::vector<float>& cos_out, std::vector<float>& sin_out) const;

    bool prepare_conditioning(const std::string& prompt, const GenerateConfig& cfg,
                              diffusion::FluxGenerationPlan& plan,
                              std::vector<float>& pooled_output,
                              std::vector<float>& text_embeddings);
    void prepare_denoising_state(const diffusion::FluxGenerationPlan& plan,
                                 const std::vector<float>& text_embeddings,
                                 std::vector<float>& encoder_hidden, std::vector<float>& cos_vals,
                                 std::vector<float>& sin_vals, std::vector<float>& latents);
    bool run_denoising(const diffusion::FluxGenerationPlan& plan,
                       const std::vector<float>& pooled_output, std::vector<float>& encoder_hidden,
                       std::vector<float>& cos_vals, std::vector<float>& sin_vals,
                       std::vector<float>& latents);
    bool decode_and_convert(const diffusion::FluxGenerationPlan& plan, std::vector<float>& latents,
                            ImageResult& result);

    // Legacy single-sample path. Same body as the original generate_image() but
    // accepts an explicit per-sample seed so the batched path can drive RNG.
    // Used both by the public generate_image() wrapper and the chunk-size-1
    // branch inside generate_image_batch().
    ImageResult generate_one_for_batch(const std::string& prompt, std::uint32_t per_sample_seed,
                                       const GenerateConfig& cfg);

    // Keep TP communicator ownership until after TRT modules are destroyed.
    std::shared_ptr<void> distributed_owner_;
    int32_t tensor_parallel_rank_{0};
    int32_t tensor_parallel_size_{1};
    std::vector<std::unique_ptr<TrtModule>> text_encoders_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    DiffusionConfig config_;
    PreprocessorWeights weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITokenizer> clip_tokenizer_;
    std::string model_id_;
    std::string raw_prompt_;

    int32_t h_latent_{0};
    int32_t w_latent_{0};
    int32_t num_img_tokens_{0};
};

} // namespace trtmc
