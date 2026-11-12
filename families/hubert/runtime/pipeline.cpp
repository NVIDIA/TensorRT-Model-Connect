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

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string text = "[";
    for (std::size_t index = 0; index < shape.size(); ++index)
        text += (index ? ", " : "") + std::to_string(shape[index]);
    return text + "]";
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
    const auto values = static_cast<std::int64_t>(audio.samples.size());
    if (values % channels != 0)
        throw std::invalid_argument("hubert input is not a whole number of audio frames");
    const auto count = values / channels;
    if (count < config_.min_samples || count > config_.max_samples) {
        // The engine's profile is built for a sample range; a clip outside it
        // is refused by name, before it is mixed down, rather than failing
        // inside TensorRT.
        throw std::invalid_argument("hubert accepts " + std::to_string(config_.min_samples) +
                                    " to " + std::to_string(config_.max_samples) +
                                    " samples, not " + std::to_string(count));
    }
    auto samples = hubert::mix_mono(audio.samples.data(), values, channels);

    if (config_.do_normalize)
        hubert::normalize_waveform(samples);

    Tensor input{samples.data(), {1, 1, 1, count}, DType::kFloat32};
    auto outputs = engine_->forward({{"audio", input}});

    const Tensor& logits = require_output(outputs, "logits");
    const auto vocab_size = static_cast<std::int64_t>(config_.vocabulary.tokens.size());
    const auto expected = hubert::feature_frames(count, config_.conv_kernel, config_.conv_stride);
    // The decoder reads one row per frame, so the layout is checked, not only
    // the element count: [vocab, frames] holds as many values and decodes to noise.
    if (expected <= 0 || logits.shape != std::vector<std::int64_t>{expected, vocab_size}) {
        throw std::runtime_error("hubert engine returned logits of shape " +
                                 shape_text(logits.shape) + ", not the [" +
                                 std::to_string(expected) + ", " + std::to_string(vocab_size) +
                                 "] its convolution strides and vocabulary imply");
    }

    auto decoded = hubert::greedy_decode(static_cast<const float*>(logits.data), expected,
                                         vocab_size, config_.vocabulary);
    TextResult result;
    result.text = std::move(decoded.text);
    result.token_ids = std::move(decoded.token_ids);
    return result;
}

} // namespace trtmc
