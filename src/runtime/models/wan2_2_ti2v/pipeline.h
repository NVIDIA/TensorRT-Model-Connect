/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/backend/prebound_backend.h"
#include "runtime/models/wan2_2_ti2v/options.h"
#include "runtime/models/wan2_2_ti2v/runtime_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using Wan22ModuleLoader = std::function<std::unique_ptr<ITrtModule>(
    const std::string&, cudaStream_t, const std::vector<ModuleExternalBinding>&)>;

struct Wan22TI2VRuntimeShape {
    int32_t latent_frames{0};
    int32_t latent_height{0};
    int32_t latent_width{0};
    int32_t video_frames{0};
    int32_t video_height{0};
    int32_t video_width{0};
    std::size_t latent_count{0};
};

Wan22TI2VRuntimeShape make_wan22_runtime_shape(const Wan22TI2VRequest& request);

std::vector<ModuleExternalBinding>
make_wan22_vae_cache_bindings(const std::vector<void*>& input_addresses,
                              const std::vector<void*>& output_addresses,
                              const Wan22TI2VRuntimeShape& shape);

class Wan22TI2VPipeline final : public IPipeline {
  public:
    Wan22TI2VPipeline(Wan22ModuleLoader module_loader, std::unique_ptr<ITokenizer> tokenizer,
                      Wan22TI2VOptions options, wan2_2_ti2v::RuntimeConfig runtime_config,
                      std::string model_id);
    ~Wan22TI2VPipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Wan22TI2VPipeline"; }

  private:
    std::vector<int32_t> tokenize(const std::string& text) const;
    std::vector<float> encode_text(const std::vector<int32_t>& ids, ITrtModule& text_encoder);
    std::vector<float> run_denoiser(const std::vector<float>& latents,
                                    const std::vector<float>& context,
                                    const std::vector<float>& time,
                                    const Wan22TI2VRuntimeShape& shape, ITrtModule& denoiser);
    void run_denoising(std::vector<float>& latents, const std::vector<float>& prompt_context,
                       const std::vector<float>& negative_context, const Wan22TI2VRequest& request,
                       const Wan22TI2VRuntimeShape& shape, double& denoiser_ms,
                       double& scheduler_ms);
    ImageResult decode_video(const std::vector<float>& latents, const Wan22TI2VRuntimeShape& shape);
    std::unique_ptr<ITrtModule>
    load_module(const std::string& section_name,
                const std::vector<ModuleExternalBinding>& external_bindings = {}) const;
    void synchronize_stream(const char* transition) const;
    void synchronize_stream_noexcept() const noexcept;

    Wan22ModuleLoader module_loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    Wan22TI2VOptions options_;
    wan2_2_ti2v::RuntimeConfig runtime_config_;
    std::string model_id_;
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
};

} // namespace trtmc
