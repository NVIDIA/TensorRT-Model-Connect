/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/wav2vec2/runtime/ctc.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct Wav2Vec2RuntimeConfig {
    std::int32_t sample_rate{16000};
    bool do_normalize{true};
    std::vector<std::int32_t> conv_kernel;
    std::vector<std::int32_t> conv_stride;
    std::int64_t min_samples{0};
    std::int64_t max_samples{0};
    wav2vec2::CtcVocabulary vocabulary;
};

// One engine and one pass. The clip's true length goes in, because the feature
// encoder normalises across time and a padded clip is a different clip.
class Wav2Vec2Pipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    Wav2Vec2Pipeline(std::unique_ptr<ITrtModule> engine, Wav2Vec2RuntimeConfig config);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    Wav2Vec2RuntimeConfig config_;
};

} // namespace trtmc
