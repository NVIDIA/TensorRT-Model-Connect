/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// LTXVideoPipeline: native C++ runtime for Lightricks LTX-Video.
// All model execution goes through TensorRT component engines.

#include "runtime/models/ltx_video/ltx_video_diffusion_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct LTXVideoOptions {
    std::string negative_prompt{"worst quality, inconsistent motion, blurry, jittery, distorted"};
    int32_t frame_rate{25};
    float guidance_rescale{0.0F};
};

LTXVideoOptions parse_ltx_video_options(const std::string& config_json);

class LTXVideoPipeline final : public IPipeline {
  public:
    LTXVideoPipeline(std::unique_ptr<TrtModule> text_encoder, std::unique_ptr<TrtModule> denoiser,
                     std::unique_ptr<TrtModule> vae, LTXVideoDiffusionConfig config,
                     LTXVideoOptions options, std::shared_ptr<ITokenizer> tokenizer,
                     std::string model_id_str);

    ~LTXVideoPipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "LTXVideoPipeline"; }

  private:
    bool run_t5_encoder(const std::vector<int32_t>& input_ids, std::vector<float>& text_embeddings,
                        int32_t& real_tokens);
    bool encode_prompt(const std::string& prompt, std::vector<float>& prompt_embeddings,
                       int32_t& prompt_tokens, std::vector<float>& negative_embeddings,
                       int32_t& negative_tokens);
    bool run_denoiser(const std::vector<float>& packed_latents,
                      const std::vector<float>& text_embeddings, int32_t real_tokens,
                      float timestep, std::vector<float>& output);
    bool decode_vae(const std::vector<float>& packed_latents, LTXVideoResult& result);

    bool denoise_loop(std::vector<float>& latents, const std::vector<float>& prompt_embeddings,
                      int32_t prompt_tokens, const std::vector<float>& negative_embeddings,
                      int32_t negative_tokens, int32_t num_steps, float guidance_scale);
    bool compute_velocity_for_step(const std::vector<float>& latents,
                                   const std::vector<float>& prompt_embeddings,
                                   int32_t prompt_tokens,
                                   const std::vector<float>& negative_embeddings,
                                   int32_t negative_tokens, float timestep, int32_t step,
                                   int32_t num_steps, float guidance_scale,
                                   std::vector<float>& cond, std::vector<float>& uncond,
                                   std::vector<float>& velocity);

    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    LTXVideoDiffusionConfig config_;
    LTXVideoOptions options_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtmc
