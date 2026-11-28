/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/granite_speech5/runtime/ctc.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::granite_speech5 {

// One encoder engine call per clip (front end, Conformer encoder, per-frame CTC argmax), then the
// greedy CTC collapse and the tokenizer's decoding on the host.
class Pipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    Pipeline(std::unique_ptr<ITrtModule> encoder, Geometry geometry, Vocabulary vocabulary);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView options) override;

  private:
    std::unique_ptr<ITrtModule> encoder_;
    Geometry geometry_;
    Vocabulary vocabulary_;
    std::mutex mutex_;
};

} // namespace trtmc::granite_speech5
