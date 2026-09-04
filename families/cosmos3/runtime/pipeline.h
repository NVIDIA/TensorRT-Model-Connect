/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/cosmos3/runtime/conditioning.h"
#include "families/cosmos3/runtime/runtime_config.h"
#include "families/cosmos3/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/task.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::cosmos3 {

class Cosmos3Pipeline final : public IImageGeneration {
  public:
    Cosmos3Pipeline(BundleReader reader, IBackend& backend, std::unique_ptr<ITokenizer> tokenizer,
                    RuntimeConfig runtime, void* distributed_communicator = nullptr,
                    std::shared_ptr<void> distributed_owner = {}, int32_t distributed_rank = 0,
                    int32_t distributed_world_size = 1);
    ~Cosmos3Pipeline() override;

    ImageResult generate_image(const std::string& prompt,
                               const ImageGenerationConfig& config = {}) override;

  private:
    std::unique_ptr<ITrtModule>
    load_module(const std::string& section_name,
                const std::vector<ModuleExternalBinding>& external_bindings = {}) const;
    std::vector<float> run_denoiser(const std::vector<float>& patches,
                                    const std::vector<float>& time_features,
                                    const PromptInputs& prompt_inputs, ITrtModule& denoiser) const;
    void run_denoising(std::vector<float>& latents, const PromptInputs& conditional_prompt,
                       const PromptInputs& unconditional_prompt, const GenerationRequest& request,
                       double& engine_load_ms, double& step_prep_ms, double& denoiser_ms,
                       double& scheduler_ms);
    ImageResult decode_video(const std::vector<float>& latents);
    void synchronize_stream(const char* transition) const;
    void synchronize_stream_noexcept() const noexcept;

    BundleReader reader_;
    IBackend* backend_{nullptr};
    void* distributed_communicator_{nullptr};
    std::shared_ptr<void> distributed_owner_;
    std::unique_ptr<ITokenizer> tokenizer_;
    RuntimeConfig runtime_;
    int32_t distributed_rank_{0};
    int32_t distributed_world_size_{1};
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
};

} // namespace trtmc::cosmos3
