/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ast/runtime/pipeline.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("ast engine did not return ") + name);
    if (found->second.data == nullptr || found->second.dtype != DType::kFloat32 ||
        found->second.numel() <= 0)
        throw std::runtime_error(std::string("ast engine returned an unusable ") + name);
    return found->second;
}

} // namespace

AstPipeline::AstPipeline(std::unique_ptr<ITrtModule> engine, AstRuntimeConfig config)
    : engine_(std::move(engine)), config_(std::move(config)) {
    if (!engine_)
        throw std::runtime_error("AstPipeline: missing engine");
    if (config_.labels.empty())
        throw std::runtime_error("AstPipeline: the bundle carries no class labels");
}

std::vector<internal::TaskInstance> AstPipeline::task_bindings() {
    return {internal::bind<internal::IAudioToClassScores>(*this)};
}

internal::LabelScoresResult AstPipeline::run(const internal::AudioToClassScoresRequest& request,
                                             internal::ConfigView) {
    const internal::AudioView& audio = request.audio;
    if (audio.samples.data() == nullptr || audio.samples.empty())
        throw std::invalid_argument("ast requires float32 PCM input");
    const auto channels = static_cast<std::int32_t>(audio.channels == 0 ? 1U : audio.channels);
    if (audio.samples.size() % static_cast<std::size_t>(channels) != 0)
        throw std::invalid_argument("ast input length is not a whole number of audio frames");

    // The filter bank's framing, mel band and normalisation statistics are all
    // tied to the rate the checkpoint was trained at, so a different rate is
    // refused rather than silently analysed at the wrong scale. This family
    // ships no resampler; the caller supplies audio at the declared rate.
    const auto rate = static_cast<std::int32_t>(
        audio.sample_rate.value_or(static_cast<std::uint32_t>(config_.fbank.sample_rate)));
    if (rate != config_.fbank.sample_rate) {
        throw std::invalid_argument("ast requires " + std::to_string(config_.fbank.sample_rate) +
                                    " Hz audio, not " + std::to_string(rate) + " Hz");
    }

    const auto mono = ast::mix_ast_mono(audio.samples.data(),
                                        static_cast<std::int64_t>(audio.samples.size()), channels);
    auto features = ast::compute_ast_features(mono.data(), static_cast<std::int64_t>(mono.size()),
                                              config_.fbank);

    Tensor input{
        features.data(), {config_.fbank.max_length, config_.fbank.num_mel_bins}, DType::kFloat32};
    auto outputs = engine_->forward({{"input_values", input}});

    const Tensor& logits = require_output(outputs, "logits");
    const auto count = static_cast<std::size_t>(config_.labels.size());
    if (logits.numel() != static_cast<std::int64_t>(count))
        throw std::runtime_error("ast engine returned a logit count the labels do not match");

    const auto* values = static_cast<const float*>(logits.data);
    internal::LabelScoresResult result;
    result.scores.assign(values, values + count);
    for (const auto& score : result.scores)
        if (!std::isfinite(score))
            throw std::runtime_error("ast engine returned a non-finite class score");
    result.labels = config_.labels;
    // Raw logits over a multi-label vocabulary: AudioSet clips carry several
    // events at once, so these are per-class and do not sum to one. Reporting
    // them as probabilities would invite a softmax that does not belong here.
    result.kind = internal::ScoreKind::Logit;
    return result;
}

} // namespace trtmc
