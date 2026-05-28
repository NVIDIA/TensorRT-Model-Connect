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

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "ZImagePipeline"; }

  private:
    bool run_text_encoder(const std::vector<int32_t>& input_ids,
                          std::vector<float>& text_embeddings);
    bool run_denoiser(const std::vector<float>& hidden, const std::vector<float>& encoder_hidden,
                      const std::vector<float>& temb, const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals, std::vector<float>& output);

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
