/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_vl_moe/runtime/pipeline.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc {
namespace {

Span<const internal::ConfigField> task_fields() {
    using internal::ConfigField;
    using internal::ConfigKind;
    static const std::vector<ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, std::nullopt, "Generated token budget"},
        {"temperature", ConfigKind::F64, 1.0, "Sampling temperature"},
        {"top_k", ConfigKind::I64, std::int64_t{1}, "Top-k sampling"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus sampling"},
        {"min_p", ConfigKind::F64, 0.0, "Minimum relative probability"},
        {"seed", ConfigKind::I64, std::int64_t{-1}, "Sampling seed; -1 is unspecified"},
    };
    return {fields.data(), fields.size()};
}

TextGenerationConfig request_config(internal::ConfigView supplied, int default_tokens) {
    const auto fields = task_fields();
    internal::validate_config(fields, supplied);
    const auto integer = [&](const char* key, int fallback) {
        const auto value =
            internal::config_get<std::int64_t>(supplied, fields, key).value_or(fallback);
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw internal::ConfigError(std::string("Qwen3-VL-MoE option outside int32: ") + key);
        return static_cast<std::int32_t>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens", default_tokens);
    config.top_k = integer("top_k", 1);
    config.seed = integer("seed", -1);
    const auto temperature = *internal::config_get<double>(supplied, fields, "temperature");
    const auto top_p = *internal::config_get<double>(supplied, fields, "top_p");
    const auto min_p = *internal::config_get<double>(supplied, fields, "min_p");
    if (config.max_new_tokens <= 0 || config.top_k < 1 || config.seed < -1 ||
        !std::isfinite(temperature) || temperature <= 0 ||
        temperature > std::numeric_limits<float>::max() || !std::isfinite(top_p) || top_p <= 0 ||
        top_p > 1 || !std::isfinite(min_p) || min_p < 0 || min_p > 1)
        throw internal::ConfigError("Qwen3-VL-MoE invalid generation options");
    config.temperature = static_cast<float>(temperature);
    config.top_p = static_cast<float>(top_p);
    config.min_p = static_cast<float>(min_p);
    return config;
}

} // namespace

std::vector<internal::TaskInstance> Qwen3VlMoePipeline::task_bindings() {
    std::vector<internal::TaskInstance> bindings{
        internal::bind<internal::ITextContinuation>(*this, task_fields())};
    if (has_vision_encoder())
        bindings.push_back(internal::bind<internal::IImagesTextToText>(*this, task_fields()));
    return bindings;
}

TextResult Qwen3VlMoePipeline::run(const internal::TextContinuationRequest& request,
                                   internal::ConfigView supplied) {
    const auto* prefix = std::get_if<std::string_view>(&request.prefix);
    if (prefix == nullptr)
        throw std::invalid_argument("Qwen3-VL-MoE continuation accepts UTF-8 text prefixes");
    return generate(std::string(*prefix), request_config(supplied, default_max_new_tokens()));
}

TextResult Qwen3VlMoePipeline::run(const internal::ImagesTextToTextRequest& request,
                                   internal::ConfigView supplied) {
    const auto config = request_config(supplied, default_max_new_tokens());
    if (request.messages.size() != 1 || !request.tools.empty() ||
        request.messages[0].role != internal::MessageRole::User)
        throw std::invalid_argument("Qwen3-VL-MoE accepts one user turn without tools");
    const internal::ImageView* image = nullptr;
    std::string prompt;
    for (const auto& part : request.messages[0].parts) {
        if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
            prompt.append(text->text);
        } else if (const auto* media = std::get_if<internal::ImageView>(&part)) {
            if (image != nullptr)
                throw std::invalid_argument("Qwen3-VL-MoE accepts exactly one image");
            image = media;
        } else {
            throw std::invalid_argument("Qwen3-VL-MoE accepts image and text parts only");
        }
    }
    if (image == nullptr || image->data == nullptr || image->height == 0 || image->width == 0 ||
        image->channels != 3 || image->height > static_cast<std::uint32_t>(INT32_MAX) ||
        image->width > static_cast<std::uint32_t>(INT32_MAX))
        throw std::invalid_argument("Qwen3-VL-MoE requires a nonempty RGB image");
    const auto width = static_cast<std::size_t>(image->width);
    const auto height = static_cast<std::size_t>(image->height);
    if (height > std::numeric_limits<std::size_t>::max() / width / 3)
        throw std::invalid_argument("Qwen3-VL-MoE image size overflows");
    const auto count = height * width * 3;
    const auto element_size = image->format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (image->format != internal::ImageFormat::UInt8 &&
        image->format != internal::ImageFormat::Float32)
        throw std::invalid_argument("Qwen3-VL-MoE unsupported image format");
    if (count > std::numeric_limits<std::size_t>::max() / element_size ||
        image->byte_size != count * element_size)
        throw std::invalid_argument("Qwen3-VL-MoE RGB image byte size differs from its dimensions");
    std::vector<float> pixels(count);
    if (image->format == internal::ImageFormat::UInt8) {
        const auto* bytes = static_cast<const std::uint8_t*>(image->data);
        for (std::size_t i = 0; i < count; ++i)
            pixels[i] = bytes[i] / 255.0F;
    } else {
        const auto* values = static_cast<const float*>(image->data);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i]) || values[i] < 0 || values[i] > 1)
                throw std::invalid_argument("Qwen3-VL-MoE float image values must be in [0,1]");
            pixels[i] = values[i];
        }
    }
    return generate(prompt, pixels.data(), static_cast<int32_t>(height),
                    static_cast<int32_t>(width), config);
}

} // namespace trtmc
