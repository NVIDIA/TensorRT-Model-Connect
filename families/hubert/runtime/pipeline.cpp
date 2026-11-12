/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hubert/runtime/pipeline.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("hubert engine did not return ") + name);
    if (found->second.data == nullptr || found->second.dtype != DType::kFloat32 ||
        found->second.numel() <= 0)
        throw std::runtime_error(std::string("hubert engine returned an unusable ") + name);
    return found->second;
}

} // namespace

HubertPipeline::HubertPipeline(std::unique_ptr<ITrtModule> engine, HubertRuntimeConfig config)
    : engine_(std::move(engine)), config_(std::move(config)) {
    if (!engine_)
        throw std::runtime_error("HubertPipeline: missing engine");
    if (config_.vocabulary.tokens.empty())
        throw std::runtime_error("HubertPipeline: the bundle carries no vocabulary");
}

std::vector<internal::TaskInstance> HubertPipeline::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this)};
}

TextResult HubertPipeline::run(const internal::SpeechTranscriptionRequest& request,
                               internal::ConfigView) {
    const internal::AudioView& audio = request.audio;
    if (audio.samples.data() == nullptr || audio.samples.empty())
        throw std::invalid_argument("hubert requires float32 PCM input");

    // The convolution strides and the normalisation are fixed to the rate the
    // checkpoint was trained at. This family ships no resampler; the caller
    // supplies audio at the declared rate.
    const auto rate = static_cast<std::int32_t>(
        audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.sample_rate)));
    if (rate != config_.sample_rate) {
        throw std::invalid_argument("hubert requires " + std::to_string(config_.sample_rate) +
                                    " Hz audio, not " + std::to_string(rate) + " Hz");
    }

    const auto channels = static_cast<std::int32_t>(audio.channels == 0 ? 1U : audio.channels);
    auto samples = hubert::mix_mono(audio.samples.data(),
                                    static_cast<std::int64_t>(audio.samples.size()), channels);

    const auto count = static_cast<std::int64_t>(samples.size());
    if (count < config_.min_samples || count > config_.max_samples) {
        // The engine's profile is built for a sample range; a clip outside it
        // is refused by name rather than failing inside TensorRT.
        throw std::invalid_argument("hubert accepts " + std::to_string(config_.min_samples) +
                                    " to " + std::to_string(config_.max_samples) +
                                    " samples, not " + std::to_string(count));
    }

    if (config_.do_normalize)
        hubert::normalize_waveform(samples);

    Tensor input{samples.data(), {1, 1, 1, count}, DType::kFloat32};
    auto outputs = engine_->forward({{"audio", input}});

    const Tensor& logits = require_output(outputs, "logits");
    const auto vocab_size = static_cast<std::int64_t>(config_.vocabulary.tokens.size());
    const auto expected = hubert::feature_frames(count, config_.conv_kernel, config_.conv_stride);
    const auto expected_logits = static_cast<std::size_t>(expected * vocab_size);
    if (expected <= 0 || logits.numel() != expected_logits) {
        throw std::runtime_error("hubert engine returned " + std::to_string(logits.numel()) +
                                 " logits, not the " + std::to_string(expected_logits) +
                                 " its convolution strides imply");
    }

    auto decoded = hubert::greedy_decode(static_cast<const float*>(logits.data), expected,
                                         vocab_size, config_.vocabulary);
    TextResult result;
    result.text = std::move(decoded.text);
    result.token_ids = std::move(decoded.token_ids);
    return result;
}

} // namespace trtmc
