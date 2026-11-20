/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma/runtime/chat_templates.h"
#include "families/gemma/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <type_traits>

namespace trtmc {
namespace {

// Borrow ordered parts through the synchronous call; never flatten media to text.
template <typename Part>
std::vector<internal::ImageAudioTextPart> media_parts(Span<const Part> parts) {
    std::vector<internal::ImageAudioTextPart> result;
    for (const auto& part : parts)
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, internal::TextPartView> ||
                              std::is_same_v<T, internal::ImageView> ||
                              std::is_same_v<T, internal::AudioView>)
                    result.emplace_back(value);
                else
                    throw std::invalid_argument(
                        "Gemma4 media does not map tool or reasoning parts");
            },
            part);
    return result;
}

template <typename Message>
auto borrow_messages(Span<const Message> messages) {
    std::vector<std::vector<internal::ImageAudioTextPart>> parts;
    std::vector<internal::ImageAudioTextMessage> views;
    parts.reserve(messages.size());
    views.reserve(messages.size());
    for (const auto& message : messages) {
        parts.push_back(media_parts(message.parts));
        views.push_back({message.role, {parts.back().data(), parts.back().size()}});
    }
    return std::make_pair(std::move(parts), std::move(views));
}

} // namespace

void GemmaTextGenerationPipeline::add_media_inputs(TensorMap& inputs, int32_t length,
                                                   bool prefill) {
    if (!vision_ && !audio_)
        return;
    const auto width = vision_ ? vision_->config().hidden_size : audio_->config().hidden_size;
    // Media is consumed during the whole prompt prefill, never during decoding.
    // A text request through a media bundle uses the same plan with an empty mask.
    if (!prefill || media_mask_.empty()) {
        media_mask_.assign(length, 0);
        media_features_.assign(static_cast<std::size_t>(length) * width, 0);
    }
    if (media_mask_.size() != static_cast<std::size_t>(length) ||
        media_features_.size() != static_cast<std::size_t>(length) * width)
        throw std::runtime_error("Gemma4 conditioned decoder input dimensions do not match");
    inputs["media_features"] = {media_features_.data(), {length, width}, DType::kFloat32};
    inputs["media_mask"] = {media_mask_.data(), {length}, DType::kInt32};
}

TextResult GemmaTextGenerationPipeline::run(const internal::ImagesTextToTextRequest& input,
                                            internal::ConfigView supplied) {
    if (!vision_)
        throw std::invalid_argument("Gemma4 bundle has no native image capability");
    if (!input.tools.empty())
        throw std::invalid_argument("Gemma4 image requests do not map tool definitions");
    const auto borrowed = borrow_messages(input.messages);
    return run_media({{borrowed.second.data(), borrowed.second.size()}}, supplied);
}

TextResult GemmaTextGenerationPipeline::run(const internal::AudioTextToTextRequest& input,
                                            internal::ConfigView supplied) {
    if (!audio_)
        throw std::invalid_argument("Gemma4 bundle has no native audio capability");
    const auto borrowed = borrow_messages(input.messages);
    return run_media({{borrowed.second.data(), borrowed.second.size()}}, supplied);
}

TextResult GemmaTextGenerationPipeline::run(const internal::ImageAudioTextToTextRequest& input,
                                            internal::ConfigView supplied) {
    if (!vision_ || !audio_)
        throw std::invalid_argument("Gemma4 bundle has no combined image/audio capability");
    return run_media(input, supplied);
}

TextResult
GemmaTextGenerationPipeline::run_media(const internal::ImageAudioTextToTextRequest& input,
                                       internal::ConfigView supplied) {
    using Clock = std::chrono::steady_clock;
    if ((!vision_ && !audio_) || (config_.chat_template_format != "gemma4" &&
                                  config_.chat_template_format != "gemma4_closed_thought"))
        throw std::invalid_argument("Gemma4 bundle has no native media capability");
    media_mask_.clear();
    media_features_.clear();
    const auto config = request_config(supplied, true);
    const auto prompt =
        gemma4_media_prompt(input, config) +
        gemma4_generation_suffix(config_.chat_template_format, config.enable_thinking);
    const int width = vision_ ? vision_->config().hidden_size : audio_->config().hidden_size;
    const int image_token = vision_ ? vision_->config().image_token : -1;
    const int audio_token = audio_ ? audio_->config().audio_token : -1;
    const auto start = Clock::now();
    // Validate every media segment on the CPU before executing any model.
    using Media = std::variant<GemmaImagePatches, GemmaAudioFeatures>;
    std::vector<Media> media;
    for (const auto& message : input.messages)
        for (const auto& part : message.parts) {
            if (const auto* image = std::get_if<internal::ImageView>(&part)) {
                if (!vision_)
                    throw std::invalid_argument("Gemma4 bundle has no native image capability");
                media.emplace_back(gemma4_image_patches(*image, vision_->config()));
            }
            if (const auto* audio = std::get_if<internal::AudioView>(&part)) {
                if (!audio_)
                    throw std::invalid_argument("Gemma4 bundle has no native audio capability");
                media.emplace_back(gemma4_audio_features(*audio, audio_->config()));
            }
        }
    auto tokens = tokenizer_->encode(prompt);
    if (tokens.size() >= 2 && tokens[0] == config_.id_bos && tokens[1] == config_.id_bos)
        tokens.erase(tokens.begin());
    std::vector<int32_t> ids;
    std::vector<std::size_t> offsets;
    std::size_t next = 0;
    for (int32_t token : tokens) {
        if (token != image_token && token != audio_token) {
            ids.push_back(token);
            continue;
        }
        const bool image = token == image_token;
        if (next >= media.size() || image != std::holds_alternative<GemmaImagePatches>(media[next]))
            throw std::invalid_argument("Gemma4 media markers and supplied content do not match");
        const int begin = image ? vision_->config().begin_token : audio_->config().begin_token;
        const int end = image ? vision_->config().end_token : audio_->config().end_token;
        const int count =
            std::visit([](const auto& value) { return value.soft_tokens; }, media[next]);
        ids.push_back(begin);
        offsets.push_back(ids.size());
        ids.insert(ids.end(), count, token);
        ids.push_back(end);
        ++next;
    }
    if (next != media.size())
        throw std::invalid_argument("Gemma4 tokenizer did not preserve every media marker");
    if (static_cast<int64_t>(ids.size()) + config.max_new_tokens - 1 > config_.prefill_max_length)
        throw std::invalid_argument("Gemma prompt and generation exceed the engine capacity");
    media_mask_.assign(ids.size(), 0);
    media_features_.assign(ids.size() * width, 0);
    try {
        for (std::size_t i = 0; i < media.size(); ++i) {
            const auto* image = std::get_if<GemmaImagePatches>(&media[i]);
            const auto features = image ? vision_->encode(*image)
                                        : audio_->encode(std::get<GemmaAudioFeatures>(media[i]));
            std::copy(features.begin(), features.end(),
                      media_features_.begin() + offsets[i] * width);
            // Positive block IDs permit vision-only bidirectional local attention.
            // Negative audio IDs replace embeddings but retain causal attention.
            const int32_t block = static_cast<int32_t>(i + 1) * (image ? 1 : -1);
            std::fill_n(media_mask_.begin() + offsets[i], features.size() / width, block);
        }
        last_setup_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        const auto sampling = gemma_sampling_params_from_config(config, config_.id_eos_ids);
        auto timed = generate_from_ids(ids, config.max_new_tokens, sampling, config);
        std::vector<int32_t> output(timed.token_ids.begin() + ids.size(), timed.token_ids.end());
        TextResult result{tokenizer_->decode(output), std::move(output), timed.prefill_ms,
                          timed.decode_ms};
        result.setup_ms = last_setup_ms_;
        media_mask_.clear();
        media_features_.clear();
        return result;
    } catch (...) {
        media_mask_.clear();
        media_features_.clear();
        throw;
    }
}

} // namespace trtmc
