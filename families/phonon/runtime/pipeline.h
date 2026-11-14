/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "families/phonon/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>

namespace trtmc::phonon {
struct Assets {
    std::vector<float> text, voice, bos, projection, condition;
    std::int64_t maximum;
};
class Pipeline final : public internal::IModel, public internal::ITextToSpeech {
  public:
    Pipeline(std::unique_ptr<ITrtModule> backbone, std::unique_ptr<ITrtModule> flow,
             std::unique_ptr<ITrtModule> codec, Assets assets, Tokenizer tokenizer);
    const char* task() const noexcept override { return "text_to_speech"; }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::AudioResult run(const internal::TextToSpeechRequest&, internal::ConfigView) override;

  private:
    std::unique_ptr<ITrtModule> backbone_, flow_, codec_;
    Assets assets_;
    Tokenizer tokenizer_;
};
} // namespace trtmc::phonon
