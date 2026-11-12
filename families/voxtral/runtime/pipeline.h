/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/voxtral/runtime/frontend.h"
#include "families/voxtral/runtime/text.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::voxtral {

struct RuntimeConfig {
    FrontendConfig frontend;
    PromptContract prompt;
    DType dtype{DType::kFloat32};
    std::int64_t max_audio_chunks{0};
    std::int64_t hidden_size{0};
    std::int64_t vocab_size{0};
    std::int64_t num_hidden_layers{0};
    std::int64_t kv_width{0};
    std::int64_t capacity{0};
    std::int64_t prefill_rows{0};
    std::int32_t eos_token_id{-1};
    std::int32_t num_special_tokens{0};
};

RuntimeConfig parse_runtime_config(const std::vector<char>& json);

// Greedy transcription: log-mel front end, encoder engine, prompt prefill in
// launches of at most `prefill_rows` rows, then one decoder row per token until
// the end-of-sequence token or the token budget.
class VoxtralPipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    explicit VoxtralPipeline(const FamilyContext& context);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView config) override;

  private:
    void validate_engines() const;
    void reset_caches();
    std::vector<float> run_rows(ITrtModule& module, const char* embeds, std::int64_t rows,
                                std::int64_t position);
    std::size_t row_bytes() const noexcept;

    RuntimeConfig config_;
    std::unique_ptr<ITrtModule> encoder_, prefill_, decode_;
    std::vector<char> embeddings_;
    std::vector<float> mel_filters_;
    Vocabulary vocabulary_;
    std::mutex mutex_;
};

} // namespace trtmc::voxtral
