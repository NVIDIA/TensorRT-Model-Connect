/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/backend/prebound_backend.h"
#include "runtime/models/cosmos3/conditioning.h"
#include "runtime/models/cosmos3/options.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using Cosmos3ModuleLoader = std::function<std::unique_ptr<ITrtModule>(
    const std::string&, cudaStream_t, const std::vector<ModuleExternalBinding>&)>;

class Cosmos3Pipeline final : public IPipeline {
  public:
    Cosmos3Pipeline(Cosmos3ModuleLoader module_loader, std::unique_ptr<ITokenizer> tokenizer,
                    Cosmos3Options options, std::string model_id,
                    DistributedRuntimeGroup context_parallel_group = {},
                    DistributedRuntimeGroup classifier_free_group = {});
    ~Cosmos3Pipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt,
                               const GenerateConfig& config = {}) override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "Cosmos3Pipeline"; }

  private:
    std::unique_ptr<ITrtModule>
    load_module(const std::string& section_name,
                const std::vector<ModuleExternalBinding>& external_bindings = {}) const;
    std::vector<float> run_denoiser(const std::vector<float>& patches,
                                    const std::vector<float>& time_features,
                                    const cosmos3::PromptInputs& prompt_inputs,
                                    ITrtModule& denoiser) const;
    TensorMap make_denoiser_inputs(const std::vector<float>& patches,
                                   const std::vector<float>& time_features,
                                   const cosmos3::PromptInputs& prompt_inputs,
                                   const ITrtModule& denoiser) const;
    void run_denoising(std::vector<float>& latents, const cosmos3::PromptInputs& conditional_prompt,
                       const cosmos3::PromptInputs& unconditional_prompt,
                       const Cosmos3Request& request, double& engine_load_ms, double& step_prep_ms,
                       double& denoiser_ms, double& cfg_exchange_ms, double& scheduler_ms);
    ImageResult decode_video(const std::vector<float>& latents);
    void synchronize_stream(const char* transition) const;
    void synchronize_stream_noexcept() const noexcept;

    DistributedRuntimeGroup context_parallel_group_;
    DistributedRuntimeGroup classifier_free_group_;
    Cosmos3ModuleLoader module_loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    Cosmos3Options options_;
    std::string model_id_;
    int32_t context_parallel_rank_{0};
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
};

} // namespace trtmc
