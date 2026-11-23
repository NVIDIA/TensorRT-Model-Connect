/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen3AsrPipeline: speech transcription over the audio encoder and the Qwen3 decoder.
//
// One request: log-mel features, one encoder launch over all attention windows, one prefill launch
// over the whole prompt (audio rows replace the placeholder embeddings), then greedy decoding one
// token per launch until a stop token or the token limit. Both decoder profiles write the same
// KV caches in place. The transcript is parsed as the official processor's
// return_format="transcription_only" does.

#include "families/qwen3_asr/runtime/frontend.h"
#include "families/qwen3_asr/runtime/runtime_config.h"
#include "families/qwen3_asr/runtime/text.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::qwen3_asr {

class Qwen3AsrPipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    Qwen3AsrPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> prefill,
                     std::unique_ptr<ITrtModule> decode, RuntimeConfig config,
                     MelFilterbank filters, Vocabulary vocabulary);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView config) override;

  private:
    std::vector<float> mono_audio(const internal::AudioView& audio) const;
    // Projected audio rows [audio_tokens, audio_embed_size].
    std::vector<float> encode(const EncoderBatch& batch);
    void bind_caches(ITrtModule& module);
    // One decoder launch over `ids` starting at cache row `position`; returns the argmax token.
    std::int32_t launch(ITrtModule& module, std::vector<std::int32_t>& ids,
                        std::vector<float>& audio_rows, std::vector<std::int32_t>& audio_mask,
                        std::int32_t position);

    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    RuntimeConfig config_;
    MelFilterbank filters_;
    Vocabulary vocabulary_;
    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<std::int32_t> key_positions_;
    std::vector<float> logits_;
    std::mutex mutex_;
};

} // namespace trtmc::qwen3_asr
