/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/hubert/runtime/ctc.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct HubertRuntimeConfig {
    std::int32_t sample_rate{16000};
    bool do_normalize{true};
    std::vector<std::int32_t> conv_kernel;
    std::vector<std::int32_t> conv_stride;
    std::int64_t min_samples{0};
    std::int64_t max_samples{0};
    hubert::CtcVocabulary vocabulary;
};

// One engine and one pass. The clip's true length goes in, because the base
// layout's feature encoder normalises across time and a padded clip is a
// different clip.
class HubertPipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    HubertPipeline(std::unique_ptr<ITrtModule> engine, HubertRuntimeConfig config);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    HubertRuntimeConfig config_;
};

} // namespace trtmc
