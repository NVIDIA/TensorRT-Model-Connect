/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/gemma/runtime/edge_llm/request.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"

#include <cstring>
#include <limits>
#include <type_traits>

namespace trtmc::gemma::edge_llm {
namespace api = trtmc::internal;
namespace edge = trt_edgellm::rt;

/// These defaults match the legacy text API; media requires the provider template.
inline Span<const api::ConfigField> fields(bool media) {
    static const std::vector<api::ConfigField> text{
        {"max_new_tokens", api::ConfigKind::I64, std::nullopt, "Generated token budget"},
        {"temperature", api::ConfigKind::F64, 1.0, "Sampling temperature"},
        {"top_k", api::ConfigKind::I64, std::int64_t{1}, "Top-k sampling"},
        {"top_p", api::ConfigKind::F64, 1.0, "Nucleus sampling"},
        {"seed", api::ConfigKind::I64, std::int64_t{-1}, "Sampling seed; -1 leaves it unspecified"},
        {"use_chat_template", api::ConfigKind::Bool, false,
         "Apply the checkpoint provider template"},
        {"enable_thinking", api::ConfigKind::Bool, true, "Provider template thinking option"},
        {"system_prompt", api::ConfigKind::String, std::string_view{}, "Optional system message"},
    };
    static const auto multimodal = [] {
        auto copy = text;
        copy[5].default_value = true;
        return copy;
    }();
    const auto& selected = media ? multimodal : text;
    return {selected.data(), selected.size()};
}

inline TextGenerationConfig generation_config(api::ConfigView supplied, bool media,
                                              int default_length) {
    const auto& schema = fields(media);
    api::validate_config(schema, supplied);
    auto integer = [&](const char* name, int fallback) {
        const auto value = api::config_get<std::int64_t>(supplied, schema, name).value_or(fallback);
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument(std::string("Gemma4 option outside int32 range: ") + name);
        return static_cast<std::int32_t>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens", default_length);
    if (config.max_new_tokens <= 0)
        throw std::invalid_argument("Gemma4 max_new_tokens must be positive");
    config.top_k = integer("top_k", 1);
    config.seed = integer("seed", -1);
    config.temperature = *api::config_get<double>(supplied, schema, "temperature");
    config.top_p = *api::config_get<double>(supplied, schema, "top_p");
    config.use_chat_template = *api::config_get<bool>(supplied, schema, "use_chat_template");
    config.enable_thinking = *api::config_get<bool>(supplied, schema, "enable_thinking");
    config.system_prompt = *api::config_get<std::string_view>(supplied, schema, "system_prompt");
    if (media && !config.use_chat_template)
        throw std::invalid_argument("Gemma4 media requires its provider chat template");
    return config;
}

/// Copy borrowed RGB storage; Edge owns resize, normalization and vision execution.
inline edge::imageUtils::ImageData image_buffer(const api::ImageView& image) {
    if (!image.data || image.height == 0 || image.width == 0 || image.channels != 3 ||
        image.height > INT32_MAX || image.width > INT32_MAX ||
        (image.format != api::ImageFormat::UInt8 && image.format != api::ImageFormat::Float32))
        throw std::invalid_argument("Gemma4 requires a nonempty host RGB8 or RGB float image");
    const auto pixels = static_cast<std::uint64_t>(image.height) * image.width;
    const std::size_t item_size = image.format == api::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (pixels > std::numeric_limits<std::size_t>::max() / 3 / item_size ||
        image.byte_size != pixels * 3 * item_size)
        throw std::invalid_argument("Gemma4 RGB storage does not match its dimensions");
    const auto count = static_cast<std::size_t>(pixels * 3);
    edge::Tensor tensor({1, image.height, image.width, 3}, edge::DeviceType::kCPU,
                        nvinfer1::DataType::kUINT8);
    auto* out = tensor.dataPointer<std::uint8_t>();
    if (image.format == api::ImageFormat::UInt8) {
        std::memcpy(out, image.data, count);
    } else {
        const auto* bytes = static_cast<const unsigned char*>(image.data);
        for (std::size_t i = 0; i < count; ++i) {
            float value;
            std::memcpy(&value, bytes + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("Gemma4 RGB float pixels must be finite in [0,1]");
            out[i] = static_cast<std::uint8_t>(std::lround(value * 255));
        }
    }
    return edge::imageUtils::ImageData(std::move(tensor));
}

/// Gemma4 uses 16 kHz mono PCM. Edge owns the checkpoint's mel preprocessing.
inline edge::audioUtils::AudioData audio_buffer(const api::AudioView& audio) {
    const auto rate = audio.sample_rate.value_or(16000);
    if (audio.channels != 1 || rate != 16000 || audio.samples.empty() || !audio.samples.data() ||
        audio.samples.size() > INT32_MAX)
        throw std::invalid_argument("Gemma4 requires nonempty 16 kHz mono float PCM");
    for (const auto value : audio.samples)
        if (!std::isfinite(value))
            throw std::invalid_argument("Gemma4 PCM samples must be finite");
    auto tensor = std::make_shared<edge::Tensor>(
        edge::Coords{static_cast<std::int64_t>(audio.samples.size())}, edge::DeviceType::kCPU,
        nvinfer1::DataType::kFLOAT);
    std::memcpy(tensor->dataPointer<float>(), audio.samples.data(),
                audio.samples.size() * sizeof(float));
    return edge::audioUtils::wrapPcm(std::move(tensor), rate);
}

inline const char* message_role(api::MessageRole role) {
    switch (role) {
    case api::MessageRole::System:
        return "system";
    case api::MessageRole::User:
        return "user";
    case api::MessageRole::Assistant:
        return "assistant";
    default:
        throw std::invalid_argument("Gemma4 media accepts system, user and assistant roles");
    }
}

/// Preserve ordered typed content; reject rather than silently drop unsupported parts.
template <class Part>
edge::LLMGenerationRequest media_request(Span<const api::MediaMessage<Part>> messages,
                                         const TextGenerationConfig& config, int default_length,
                                         bool vision, bool audio) {
    auto request = make_request("", config, default_length, true);
    auto& item = request.requests.front();
    item.messages.pop_back(); // Keep an explicitly supplied system prompt, not the dummy user.
    if (messages.empty())
        throw std::invalid_argument("Gemma4 media messages cannot be empty");
    for (const auto& source : messages) {
        edge::Message message;
        message.role = message_role(source.role);
        message.contentIsArray = true;
        if (source.parts.empty())
            throw std::invalid_argument("Gemma4 media message cannot have empty content");
        for (const auto& part : source.parts) {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, api::TextPartView>) {
                        message.contents.push_back({"text", std::string(value.text)});
                    } else if constexpr (std::is_same_v<T, api::ImageView>) {
                        if (!vision)
                            throw std::invalid_argument("Gemma4 bundle has no vision encoder");
                        item.imageBuffers.push_back(image_buffer(value));
                        message.contents.push_back({"image", ""});
                    } else if constexpr (std::is_same_v<T, api::AudioView>) {
                        if (!audio)
                            throw std::invalid_argument("Gemma4 bundle has no audio encoder");
                        item.audioBuffers.push_back(audio_buffer(value));
                        message.contents.push_back({"audio", ""});
                    } else {
                        throw std::invalid_argument(
                            "Gemma4 media does not map tool or reasoning parts");
                    }
                },
                part);
        }
        item.messages.push_back(std::move(message));
    }
    return request;
}

} // namespace trtmc::gemma::edge_llm
