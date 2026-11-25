/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minicpmv4_7/runtime/request.h"

#include "families/minicpmv4_7/runtime/prompt.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace trtmc::minicpmv4_7 {

Span<const internal::ConfigField> request_config_fields() {
    static const std::vector<internal::ConfigField> fields{
        {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
         "Generated token budget; by default up to 128 within the decoder capacity"},
        {"temperature", internal::ConfigKind::F64, 1.0, "Sampling temperature"},
        {"top_k", internal::ConfigKind::I64, int64_t{1},
         "Top-k sampling; 1 selects greedy decoding"},
        {"top_p", internal::ConfigKind::F64, 1.0, "Nucleus sampling"},
        {"seed", internal::ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 leaves it unspecified"},
        {"enable_thinking", internal::ConfigKind::Bool, false, "Provider template thinking mode"},
    };
    return {fields.data(), fields.size()};
}

std::vector<uint8_t> rgb_bytes(const internal::ImageView& image) {
    if (image.data == nullptr || image.height == 0 || image.width == 0 || image.channels != 3)
        throw std::invalid_argument("MiniCPM-V 4.7 accepts non-empty RGB images");
    constexpr auto kMaxSide = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
    if (image.height > kMaxSide || image.width > kMaxSide)
        throw std::invalid_argument("MiniCPM-V 4.7 image sides must not exceed 2147483647");
    // Sides below 2^31 keep height * width * 3 below 2^64.
    const uint64_t count = static_cast<uint64_t>(image.height) * image.width * 3U;
    const bool bytes = image.format == internal::ImageFormat::UInt8;
    const uint64_t value_size = bytes ? 1U : sizeof(float);
    if (count > std::numeric_limits<std::size_t>::max() / value_size ||
        image.byte_size != count * value_size)
        throw std::invalid_argument("MiniCPM-V 4.7 image byte size disagrees with its shape");
    std::vector<uint8_t> out(static_cast<std::size_t>(count));
    if (bytes) {
        std::memcpy(out.data(), image.data, out.size());
        return out;
    }
    const auto* values = static_cast<const float*>(image.data);
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(values[i]) || values[i] < 0.0F || values[i] > 1.0F)
            throw std::invalid_argument("MiniCPM-V 4.7 Float32 pixels must lie in [0, 1]");
        out[i] = static_cast<uint8_t>(std::lround(values[i] * 255.0F));
    }
    return out;
}

PreparedRequest prepare_request(const internal::ImagesTextToTextRequest& request,
                                internal::ConfigView supplied, const RequestContext& context) {
    if (!request.tools.empty())
        throw std::invalid_argument("MiniCPM-V 4.7 does not map tool definitions");
    const auto schema = request_config_fields();
    internal::validate_config(schema, supplied);
    const auto requested = internal::config_get<int64_t>(supplied, schema, "max_new_tokens");
    if (requested && (*requested <= 0 || *requested > context.capacity))
        throw std::invalid_argument("MiniCPM-V 4.7 max_new_tokens must be in [1, capacity]");
    // The sampler checks its options here, before any image is decoded or preprocessed.
    Sampler sampler(SamplingConfig{*internal::config_get<double>(supplied, schema, "temperature"),
                                   *internal::config_get<int64_t>(supplied, schema, "top_k"),
                                   *internal::config_get<double>(supplied, schema, "top_p"),
                                   *internal::config_get<int64_t>(supplied, schema, "seed")});
    const bool thinking = *internal::config_get<bool>(supplied, schema, "enable_thinking");

    std::vector<ChatMessage> messages;
    std::vector<ImageCrops> images;
    for (const auto& message : request.messages) {
        ChatMessage item;
        if (message.role == internal::MessageRole::System)
            item.role = Role::System;
        else if (message.role == internal::MessageRole::User)
            item.role = Role::User;
        else if (message.role == internal::MessageRole::Assistant)
            item.role = Role::Assistant;
        else
            throw std::invalid_argument("MiniCPM-V 4.7 accepts system, user and assistant roles");
        std::vector<std::string> parts;
        for (const auto& part : message.parts) {
            std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, internal::TextPartView>) {
                        parts.emplace_back(value.text);
                    } else if constexpr (std::is_same_v<T, internal::ImageView>) {
                        if (item.role != Role::User)
                            throw std::invalid_argument(
                                "MiniCPM-V 4.7 accepts images in user messages only");
                        const auto pixels = rgb_bytes(value);
                        images.push_back(
                            preprocess_image(pixels.data(), static_cast<int32_t>(value.height),
                                             static_cast<int32_t>(value.width), context.processor));
                        parts.emplace_back(kImagePlaceholder);
                    } else {
                        throw std::invalid_argument(
                            "MiniCPM-V 4.7 does not map reasoning or tool parts");
                    }
                },
                part);
        }
        item.content = join_parts(parts);
        messages.push_back(std::move(item));
    }

    const auto formatted =
        format_conversation(context.prompts.at(thinking ? "true" : "false"), messages);
    const auto expanded = expand_images(formatted, images, context.processor.merge_factor);
    auto ids = context.tokenizer.encode(expanded);
    const int64_t max_new =
        generation_budget(requested, context.capacity, static_cast<int64_t>(ids.size()));
    return {std::move(sampler), std::move(images), std::move(ids), max_new};
}

} // namespace trtmc::minicpmv4_7
