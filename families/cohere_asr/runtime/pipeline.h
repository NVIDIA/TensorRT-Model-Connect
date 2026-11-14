/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace trtmc::cohere_asr {
struct Config {
    int max_samples{560000}, max_tokens{256}, eos_id{3};
    std::vector<int32_t> arabic_prompt, english_prompt;
    std::vector<float> window, filters;
    std::vector<std::string> pieces;
    std::unordered_set<int32_t> special;
};
class Pipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    Pipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
             Config config);
    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISpeechTranscription>(*this)};
    }
    internal::TextResult run(const internal::SpeechTranscriptionRequest&,
                             internal::ConfigView) override;

  private:
    std::unique_ptr<ITrtModule> encoder_, decoder_;
    Config config_;
};
} // namespace trtmc::cohere_asr
