/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/qwen3_5_moe/runtime/edge_llm/request.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"

#include <cstring>
#include <limits>
#include <type_traits>

namespace trtmc::qwen3_5_moe::edge_llm {
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
    };
    static const auto multimodal = [] {
        auto copy = text;
        for (auto& field : copy)
            if (field.name == "use_chat_template")
                field.default_value = true;
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
            throw std::invalid_argument(std::string("Qwen3.5-MoE option outside int32 range: ") +
                                        name);
        return static_cast<std::int32_t>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens", default_length);
    if (config.max_new_tokens <= 0)
        throw std::invalid_argument("Qwen3.5-MoE max_new_tokens must be positive");
    config.top_k = integer("top_k", 1);
    config.seed = integer("seed", -1);
    config.temperature = *api::config_get<double>(supplied, schema, "temperature");
    config.top_p = *api::config_get<double>(supplied, schema, "top_p");
    config.use_chat_template = *api::config_get<bool>(supplied, schema, "use_chat_template");
    config.enable_thinking = *api::config_get<bool>(supplied, schema, "enable_thinking");
    if (media && !config.use_chat_template)
        throw std::invalid_argument("Qwen3.5-MoE media requires its provider chat template");
    return config;
}

/// Copy borrowed RGB storage; Edge owns resize, normalization and vision execution.
inline edge::imageUtils::ImageData image_buffer(const api::ImageView& image) {
    if (!image.data || image.height == 0 || image.width == 0 || image.channels != 3 ||
        image.height > INT32_MAX || image.width > INT32_MAX ||
        (image.format != api::ImageFormat::UInt8 && image.format != api::ImageFormat::Float32))
        throw std::invalid_argument("Qwen3.5-MoE requires a nonempty host RGB8 or RGB float image");
    const auto pixels = static_cast<std::uint64_t>(image.height) * image.width;
    const std::size_t item_size = image.format == api::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (pixels > std::numeric_limits<std::size_t>::max() / 3 / item_size ||
        image.byte_size != pixels * 3 * item_size)
        throw std::invalid_argument("Qwen3.5-MoE RGB storage does not match its dimensions");
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
                throw std::invalid_argument("Qwen3.5-MoE RGB float pixels must be finite in [0,1]");
            out[i] = static_cast<std::uint8_t>(std::lround(value * 255));
        }
    }
    return edge::imageUtils::ImageData(std::move(tensor));
}

/// Pack ordered RGB frames without substituting independent images for a video.
inline edge::imageUtils::ImageData video_buffer(const api::VideoView& video) {
    const auto count = video.frames.size();
    if (count == 0 || count > INT32_MAX ||
        (!video.timestamps_seconds.empty() && video.timestamps_seconds.size() != count))
        throw std::invalid_argument("Qwen3.5-MoE video needs frames and matching timestamps");
    for (std::size_t i = 0; i < video.timestamps_seconds.size(); ++i)
        if (!std::isfinite(video.timestamps_seconds[i]) ||
            (i && video.timestamps_seconds[i] < video.timestamps_seconds[i - 1]))
            throw std::invalid_argument("Qwen3.5-MoE video timestamps must be finite and ordered");
    auto first = image_buffer(video.frames[0]);
    const auto stride = static_cast<std::size_t>(first.height) * first.width * 3;
    if (count > static_cast<std::size_t>(INT64_MAX) / stride)
        throw std::invalid_argument("Qwen3.5-MoE video dimensions overflow");
    edge::Tensor tensor({static_cast<int64_t>(count), first.height, first.width, 3},
                        edge::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    auto* output = tensor.dataPointer<std::uint8_t>();
    std::memcpy(output, first.data(), stride);
    for (std::size_t i = 1; i < count; ++i) {
        if (video.frames[i].height != video.frames[0].height ||
            video.frames[i].width != video.frames[0].width)
            throw std::invalid_argument("Qwen3.5-MoE video frames must have matching dimensions");
        auto frame = image_buffer(video.frames[i]);
        std::memcpy(output + i * stride, frame.data(), stride);
    }
    edge::imageUtils::ImageData result(std::move(tensor));
    result.isVideo = true;
    // Unspecified timing follows the official frame-list default of one FPS.
    result.fps = 1.0;
    result.timestamps.assign(video.timestamps_seconds.begin(), video.timestamps_seconds.end());
    return result;
}

/// Preserve image, video and text order in the checkpoint provider messages.
template <class Request>
inline edge::LLMGenerationRequest
media_request(const Request& input, const TextGenerationConfig& config, int default_length) {
    if constexpr (!std::is_same_v<Request, api::ImageVideoTextToTextRequest>) {
        if (!input.tools.empty())
            throw std::invalid_argument("Qwen3.5-MoE media does not map tools");
    }
    if (input.messages.empty())
        throw std::invalid_argument("Qwen3.5-MoE media requires messages");
    (void)default_length;
    auto request = make_request("", config, true);
    auto& item = request.requests.front();
    item.messages.clear();
    for (const auto& source : input.messages) {
        edge::Message message;
        switch (source.role) {
        case api::MessageRole::System:
            message.role = "system";
            break;
        case api::MessageRole::User:
            message.role = "user";
            break;
        case api::MessageRole::Assistant:
            message.role = "assistant";
            break;
        default:
            throw std::invalid_argument(
                "Qwen3.5-MoE media accepts system, user and assistant roles");
        }
        message.contentIsArray = true;
        if (source.parts.empty())
            throw std::invalid_argument("Qwen3.5-MoE media message content cannot be empty");
        for (const auto& part : source.parts) {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, api::TextPartView>) {
                        message.contents.push_back({"text", std::string(value.text)});
                    } else if constexpr (std::is_same_v<T, api::ImageView>) {
                        item.imageBuffers.push_back(image_buffer(value));
                        message.contents.push_back({"image", ""});
                    } else if constexpr (std::is_same_v<T, api::VideoView>) {
                        item.imageBuffers.push_back(video_buffer(value));
                        message.contents.push_back({"video", ""});
                    } else {
                        throw std::invalid_argument(
                            "Qwen3.5-MoE media does not map tool or reasoning parts");
                    }
                },
                part);
        }
        item.messages.push_back(std::move(message));
    }
    return request;
}
} // namespace trtmc::qwen3_5_moe::edge_llm
