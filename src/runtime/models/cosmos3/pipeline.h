#pragma once

// Cosmos3Pipeline: omni-model pipeline for NVIDIA Cosmos 3 (Nano/Super).
//
// Cosmos 3 is a Mixture-of-Transformers omni-model: a single transformer
// body that interleaves text (AR) and image/video/audio/action (DM) tokens
// through joint two-way attention. This pipeline composes the four TRT
// engines that make up a generation run:
//
//   - Reasoner engine (AR text decoder):   trt module + KvCache
//   - DM generator engine (denoiser):      trt module
//   - ViT engine (vision encoder):         trt module (optional)
//   - VAE decoder engine (video tokens →   trt module
//     pixels): wraps families/wan_t2v
//
// The two_way joint attention is realized at this pipeline level: each
// denoising step runs the DM generator with the AR KV cache concatenated
// into the attention pool. For a pure text→video lane the AR side runs in
// "prefill" mode only (encode the prompt once, hold its KV cache fixed
// across denoising steps); for the omni reasoner-action loop a full
// AR/DM interleave is needed.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Cosmos3Config {
    int32_t num_reasoner_layers;   // 64 for Super, ~24 for Nano
    int32_t reasoner_hidden_size;  // 5120 for Super, 2048 for Nano
    int32_t num_reasoner_heads;    // 64 / 16
    int32_t num_reasoner_kv_heads; // 8  / 2
    int32_t reasoner_head_dim;     // 128
    int32_t max_cache_length;
    int32_t num_dm_layers;         // shares reasoner depth in Super
    int32_t dm_hidden_size;
    int32_t latent_channel;        // 48
    int32_t latent_patch_size;     // 2
    int32_t patch_latent_dim;      // 192 = 48 * 2 * 2
    int32_t vae_spatial_scale;     // 16
    int32_t vae_temporal_scale;    // 4
    int32_t num_inference_steps;   // 30 (L0) / 50 (full)
    int32_t video_num_frames;
    int32_t video_height;
    int32_t video_width;
    float   timestep_scale;        // 0.001
    int32_t mrope_section[3];      // {24, 20, 20}
    int64_t rope_theta;            // 5e6
    bool    qk_norm;
    bool    fps_modulation;
    int32_t base_fps;              // 24
    std::string precision;         // "bf16"
};

class Cosmos3Pipeline final : public IPipeline {
  public:
    Cosmos3Pipeline(std::unique_ptr<TrtModule> reasoner,
                    std::unique_ptr<IInferenceState> reasoner_state,
                    std::unique_ptr<TrtModule> dm_generator,
                    std::unique_ptr<TrtModule> vae_decoder,
                    std::unique_ptr<TrtModule> vit_encoder,
                    Cosmos3Config config, cudaStream_t stream,
                    std::shared_ptr<ITokenizer> tokenizer = nullptr,
                    std::string model_id_str = "");

    ~Cosmos3Pipeline() override;

    // Primary text→video lane.
    DiffusionResult generate_diffusion(const std::string& prompt,
                                       const DiffusionConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Cosmos3Pipeline"; }

  private:
    // Encode prompt through the AR reasoner; populate the shared KV cache
    // that the DM generator will read from during joint attention.
    void prefill_reasoner(const std::vector<int32_t>& prompt_ids);

    // Run one denoising step. Reads the latent noise tensor, the timestep,
    // and (via reasoner_state_) the prefilled AR KV cache. Writes the
    // denoised latent prediction back into latent_out.
    void denoise_step(const float* latent_in, float timestep,
                      float* latent_out);

    // VAE: decode the final latent (T_lat, 48, H_lat, W_lat) into pixels
    // (T, 3, H, W) suitable for MP4 muxing.
    std::vector<uint8_t> decode_video(const float* latent_final,
                                      int32_t latent_t, int32_t latent_h,
                                      int32_t latent_w);

    std::unique_ptr<TrtModule>       reasoner_;
    std::unique_ptr<IInferenceState> reasoner_state_;
    std::unique_ptr<TrtModule>       dm_generator_;
    std::unique_ptr<TrtModule>       vae_decoder_;
    std::unique_ptr<TrtModule>       vit_encoder_;
    Cosmos3Config                    config_;
    cudaStream_t                     stream_;
    std::shared_ptr<ITokenizer>      tokenizer_;
    std::string                      model_id_;
};

} // namespace trtmc
