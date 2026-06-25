#pragma once

// PixArtPipeline: TRT API PixArt diffusion pipeline.
// Uses preprocessor weights, a T5 text encoder, DiT denoiser, and VAE decoder.

#include "runtime/models/pixart/pixart_diffusion_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class PixArtPipeline final : public IPipeline {
  public:
    PixArtPipeline(std::unique_ptr<TrtModule> text_encoder, std::unique_ptr<TrtModule> denoiser,
                   std::unique_ptr<TrtModule> vae, PixArtDiffusionConfig config,
                   PixArtPreprocessorWeights weights, std::shared_ptr<ITokenizer> tokenizer,
                   std::string model_id_str, std::shared_ptr<void> distributed_owner = {},
                   int32_t tensor_parallel_rank = 0, int32_t tensor_parallel_size = 1);

    ~PixArtPipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "PixArtPipeline"; }

  private:
    bool run_t5_encoder(const std::vector<int32_t>& input_ids, std::vector<float>& text_embeddings);
    bool run_denoiser(const std::vector<float>& hidden, const std::vector<float>& temb_6d,
                      const std::vector<float>& time_embed,
                      const std::vector<float>& encoder_hidden, const std::vector<float>& cos_vals,
                      const std::vector<float>& sin_vals, std::vector<float>& output,
                      const std::vector<float>& encoder_attn_mask = {});

    void compute_timestep_embedding(float timestep, std::vector<float>& temb_6d,
                                    std::vector<float>& time_embed) const;
    void project_text(const std::vector<float>& in, int32_t seq_len, std::vector<float>& out) const;
    void patchify(const std::vector<float>& latents, int32_t c, int32_t t, int32_t h, int32_t w,
                  std::vector<float>& patches) const;
    void unpatchify(const std::vector<float>& patches, int32_t c, int32_t t, int32_t h, int32_t w,
                    std::vector<float>& output) const;
    void compute_3d_rope(int32_t nt, int32_t nh, int32_t nw, std::vector<float>& cos_out,
                         std::vector<float>& sin_out) const;
    bool decode_vae_2d(const std::vector<float>& latents, int32_t c, int32_t h, int32_t w,
                       PixArtVideoResult& result);
    bool decode_vae_3d(const std::vector<float>& latents, int32_t c, int32_t t, int32_t h,
                       int32_t w, PixArtVideoResult& result);

    int32_t query_vae_output_temporal_dim() const;
    void init_vae_caches();
    void zero_vae_caches();
    void decode_vae_single_frame(const std::vector<float>& latents, int32_t c, int32_t t_lat,
                                 int32_t h_lat, int32_t w_lat, int32_t t,
                                 std::size_t out_frame_floats, std::vector<float>& all_raw_frames);

    bool run_pixart_text_conditioning(const std::vector<int32_t>& input_ids, int32_t seq_len,
                                      std::vector<float>& text_projected,
                                      std::vector<float>& null_text, std::string& error);
    bool run_pixart_vae_decode(int32_t z_dim, int32_t t_lat, int32_t h_lat, int32_t w_lat,
                               std::vector<float>& latents, PixArtVideoResult& result);
    ImageResult finish_pixart_generation(int32_t z_dim, int32_t t_lat, int32_t h_lat, int32_t w_lat,
                                         std::vector<float>& latents, PixArtVideoResult& result);

    // Keep TP communicator ownership until after TRT modules are destroyed.
    std::shared_ptr<void> distributed_owner_;
    int32_t tensor_parallel_rank_{0};
    int32_t tensor_parallel_size_{1};
    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    PixArtDiffusionConfig config_;
    PixArtPreprocessorWeights weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    std::vector<DeviceTensor> vae_cache_in_;
    std::vector<DeviceTensor> vae_cache_out_;
    bool vae_caches_initialized_{false};
};

} // namespace trtmc
