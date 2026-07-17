/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/wan2_2_ti2v/options.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using Wan22ModuleLoader = std::function<std::unique_ptr<ITrtModule>(
    const std::string&, cudaStream_t, const std::vector<ModuleExternalBinding>&)>;

std::vector<ModuleExternalBinding>
make_wan22_vae_cache_bindings(const std::vector<void*>& input_addresses,
                              const std::vector<void*>& output_addresses);

class Wan22TI2VPipeline final : public IPipeline {
  public:
    Wan22TI2VPipeline(Wan22ModuleLoader module_loader, std::shared_ptr<ITokenizer> tokenizer,
                      Wan22TI2VOptions options, std::string model_id);
    ~Wan22TI2VPipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Wan22TI2VPipeline"; }

  private:
    std::vector<int32_t> tokenize(const std::string& text) const;
    std::vector<float> encode_text(const std::vector<int32_t>& ids, ITrtModule& text_encoder);
    std::vector<float> run_denoiser(const std::vector<float>& latents,
                                    const std::vector<float>& context, int64_t timestep,
                                    ITrtModule& denoiser);
    ImageResult decode_video(const std::vector<float>& latents);
    std::unique_ptr<ITrtModule>
    load_module(const std::string& section_name,
                const std::vector<ModuleExternalBinding>& external_bindings = {}) const;
    void synchronize_stream(const char* transition) const;
    void synchronize_stream_noexcept() const noexcept;

    Wan22ModuleLoader module_loader_;
    std::shared_ptr<ITokenizer> tokenizer_;
    Wan22TI2VOptions options_;
    std::string model_id_;
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
};

} // namespace trtmc
