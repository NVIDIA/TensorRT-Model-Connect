/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/moss_transcribe_diarize/runtime/log_mel.h"
#include "families/moss_transcribe_diarize/runtime/prompt_plan.h"
#include "families/moss_transcribe_diarize/runtime/runtime_config.h"
#include "families/moss_transcribe_diarize/runtime/text_output.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace trtmc::moss_transcribe_diarize {

struct Engines {
    std::unique_ptr<ITrtModule> audio;   // one 30 s chunk -> audio rows
    std::unique_ptr<ITrtModule> prefill; // text engine, profile 0
    std::unique_ptr<ITrtModule> decode;  // text engine, profile 1
};

// Audio -> speaker-attributed, timestamped transcript. One request at a time:
// the KV cache and the audio rows are owned by the loaded model.
class TranscribeDiarizeModel final : public internal::IModel,
                                     public internal::ISpeechTranscription {
  public:
    TranscribeDiarizeModel(Engines engines, RuntimeConfig config, MelFilterBank mel_filters,
                           TokenDecoder decoder);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView config) override;

  private:
    std::vector<float> mono_samples(const internal::AudioView& audio) const;
    std::int32_t resolve_max_new_tokens(internal::ConfigView config,
                                        std::size_t prompt_tokens) const;
    void encode_audio(const std::vector<float>& samples, const std::vector<AudioChunk>& chunks);
    std::int32_t prefill(const PromptPlan& plan);
    std::int32_t decode_step(std::int32_t token, std::int32_t position);
    std::int32_t key_rows(std::int32_t valid_rows) const;
    std::int32_t next_token(const TensorMap& outputs) const;
    bool is_end_of_turn(std::int32_t token) const;
    void validate_engines() const;
    void bind_cache(ITrtModule& module);

    Engines engines_;
    RuntimeConfig config_;
    LogMelExtractor mel_;
    TokenDecoder decoder_;
    DType work_dtype_;
    std::vector<DeviceTensor> cache_; // key_0, value_0, key_1, value_1, ...
    DeviceTensor audio_rows_;
    DeviceTensor decode_audio_row_;
    std::vector<std::int32_t> key_positions_;
    std::mutex mutex_;
};

} // namespace trtmc::moss_transcribe_diarize
