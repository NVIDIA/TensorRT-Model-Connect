/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/granite_speech/runtime/text.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace trtmc::granite_speech {

struct RuntimeConfig {
    DType work{DType::kBFloat16};
    std::int32_t sample_rate{0};
    std::int32_t n_fft{0};
    std::int32_t hop_length{0};
    std::int32_t window_size{0};
    std::int32_t num_queries{0};
    std::int64_t min_samples{0};
    std::int64_t max_samples{0};
    std::int32_t capacity{0};
    std::int32_t hidden_size{0};
    std::int32_t num_layers{0};
    std::int32_t kv_width{0};
    std::int32_t vocab_size{0};
    std::int32_t audio_token_id{0};
    std::int32_t eos_token_id{0};
    std::vector<std::int32_t> prefix_ids;
    std::vector<std::int32_t> suffix_ids;

    static RuntimeConfig from_json(std::string_view json);
    // Projector outputs for `samples` PCM values, as the reference processor counts them.
    std::int64_t audio_tokens(std::int64_t samples) const;
};

// numpy/torch "reflect" padding by `half` samples on each side (needs more than `half` samples).
std::vector<float> reflect_pad(const float* samples, std::size_t count, std::size_t half);
// Index of the first maximum, as torch.argmax returns it.
std::int32_t first_argmax(const float* values, std::size_t count);

// Audio engine, then the decoder prefill over the prompt with the audio rows in place, then greedy
// decode steps until EOS or max_new_tokens. Both decoder profiles read and write one device KV
// cache directly: step outputs are bound onto the cache row they belong to.
class Pipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    Pipeline(std::unique_ptr<ITrtModule> audio, std::unique_ptr<ITrtModule> prefill,
             std::unique_ptr<ITrtModule> decode, RuntimeConfig config, Vocabulary vocabulary);

    const char* task() const noexcept override {
        return internal::ISpeechTranscription::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest& request,
                   internal::ConfigView options) override;

  private:
    std::vector<float> prefill(const std::vector<float>& padded, std::int64_t audio_rows,
                               std::int64_t& prompt_rows);
    std::vector<float> step(std::int32_t token, std::int64_t cached_rows);
    void bind_cache(ITrtModule& module, std::int64_t cached_rows);

    std::unique_ptr<ITrtModule> audio_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> decode_;
    RuntimeConfig config_;
    Vocabulary vocabulary_;
    std::vector<DeviceTensor> keys_;
    std::vector<DeviceTensor> values_;
    DeviceTensor prompt_embeds_;
    DeviceTensor zero_row_;
    std::mutex mutex_;
};

} // namespace trtmc::granite_speech
