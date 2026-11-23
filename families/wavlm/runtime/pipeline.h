/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/wavlm/runtime/ctc.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct WavLMRuntimeConfig {
    std::int32_t sample_rate{16000};
    bool do_normalize{true};
    std::vector<std::int32_t> conv_kernel;
    std::vector<std::int32_t> conv_stride;
    std::int64_t min_samples{0};
    std::int64_t max_samples{0};
    wavlm::CtcVocabulary vocabulary;
};

// One engine and one pass. The clip's true length goes in: the engine has no
// padding mask, so a padded clip is a different clip.
class WavLMPipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    WavLMPipeline(std::unique_ptr<ITrtModule> engine, WavLMRuntimeConfig config);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    WavLMRuntimeConfig config_;
};

} // namespace trtmc
