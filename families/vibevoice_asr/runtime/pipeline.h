/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/vibevoice_asr/runtime/host_plan.h"
#include "families/vibevoice_asr/runtime/prompt.h"
#include "families/vibevoice_asr/runtime/runtime_config.h"
#include "families/vibevoice_asr/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::vibevoice_asr {

class VibeVoiceAsrModel final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    VibeVoiceAsrModel(RuntimeConfig config, std::unique_ptr<ITokenizer> tokenizer,
                      std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> prefill,
                      std::unique_ptr<ITrtModule> decode);
    const char* task() const noexcept override { return ISpeechTranscription::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView options) override;

  private:
    std::vector<float> prepare_audio(const internal::AudioView& audio) const;
    // Connector rows [frames, hidden] as BF16 bit patterns.
    std::vector<std::uint16_t> encode_audio(const std::vector<float>& samples);
    // Run `rows` decoder rows written at cache row `position`; returns last-row logits.
    const float* forward(ITrtModule& module, const std::int32_t* ids, const std::uint16_t* embeds,
                         const std::int32_t* mask, std::int32_t rows, std::int32_t position);
    std::vector<std::int32_t> generate(const Prompt& prompt,
                                       const std::vector<std::uint16_t>& features,
                                       std::int64_t max_new_tokens, double& prefill_ms,
                                       double& decode_ms);

    RuntimeConfig config_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    std::vector<DeviceTensor> cache_;
    std::vector<std::uint16_t> zero_row_;
    std::vector<float> logits_;
    std::mutex mutex_;
};

} // namespace trtmc::vibevoice_asr
