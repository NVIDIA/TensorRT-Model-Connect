/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// WanPipeline: Wan2.1 diffusion pipeline with T5 text encoder, denoiser, and VAE.
// Uses TrtModule::forward() for all GPU work.

#include "runtime/models/wan/wan_diffusion_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class WanGpuMatmul;

class WanPipeline final : public IPipeline {
  public:
    WanPipeline(std::unique_ptr<TrtModule> text_encoder, std::unique_ptr<TrtModule> denoiser,
                std::unique_ptr<TrtModule> vae, WanDiffusionConfig config,
                WanPreprocessorWeights weights, std::shared_ptr<ITokenizer> tokenizer,
                std::string model_id_str, std::shared_ptr<void> distributed_owner = {},
                int32_t distributed_rank = 0, int32_t distributed_world_size = 1,
                std::unique_ptr<TrtModule> vae_first_frame = nullptr);

    ~WanPipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "WanPipeline"; }

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
    void matmul_bias(const float* lhs, const float* rhs, const float* bias, float* output,
                     int32_t rows, int32_t inner, int32_t columns) const;
    void patchify(const std::vector<float>& latents, int32_t c, int32_t t, int32_t h, int32_t w,
                  std::vector<float>& patches) const;
    void unpatchify(const std::vector<float>& patches, int32_t c, int32_t t, int32_t h, int32_t w,
                    std::vector<float>& output) const;
    void compute_3d_rope(int32_t nt, int32_t nh, int32_t nw, std::vector<float>& cos_out,
                         std::vector<float>& sin_out) const;
    bool decode_vae_2d(const std::vector<float>& latents, int32_t c, int32_t h, int32_t w,
                       WanVideoResult& result);
    bool decode_vae_3d(const std::vector<float>& latents, int32_t c, int32_t t, int32_t h,
                       int32_t w, WanVideoResult& result);

    static int32_t query_vae_output_temporal_dim(const TrtModule& module);
    bool has_first_frame_vae() const;
    void init_vae_caches();
    void zero_vae_caches();
    void decode_vae_single_frame(const std::vector<float>& latents, int32_t c, int32_t t_lat,
                                 int32_t h_lat, int32_t w_lat, int32_t t,
                                 std::size_t out_frame_floats, TrtModule& module,
                                 std::vector<float>& all_raw_frames);

    bool run_wan_text_conditioning(const std::vector<int32_t>& input_ids, int32_t seq_len,
                                   std::vector<float>& text_projected,
                                   std::vector<float>& null_text, std::string& error);
    std::vector<int32_t> tokenize_wan_prompt(const std::string& prompt) const;
    bool run_wan_vae_decode(int32_t z_dim, int32_t t_lat, int32_t h_lat, int32_t w_lat,
                            std::vector<float>& latents, WanVideoResult& result);
    ImageResult finish_wan_generation(int32_t z_dim, int32_t t_lat, int32_t h_lat, int32_t w_lat,
                                      std::vector<float>& latents, WanVideoResult& result);

    // Keep distributed communicator ownership until after TRT modules are destroyed.
    std::shared_ptr<void> distributed_owner_;
    int32_t distributed_rank_{0};
    int32_t distributed_world_size_{1};
    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    std::unique_ptr<TrtModule> vae_first_frame_;
    WanDiffusionConfig config_;
    WanPreprocessorWeights weights_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<WanGpuMatmul> gpu_matmul_;

    std::vector<DeviceTensor> vae_cache_in_;
    std::vector<DeviceTensor> vae_cache_out_;
    bool vae_caches_initialized_{false};
};

} // namespace trtmc
