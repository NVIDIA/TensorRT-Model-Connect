/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma/runtime/gemma4_image.h"

#include "families/gemma/runtime/chat_templates.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc {
namespace {

// Separable antialiased cubic resampling in RGB8, with the provider's
// half-pixel coordinates and 22-bit coefficients. Round/clamp after each pass.
struct Filter {
    int start;
    std::vector<int32_t> weights;
    int precision{22};
};

double cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.5 * x - 2.5) * x) * x + 1;
    return x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0;
}

std::vector<Filter> filters(int source, int target, bool unified) {
    const double scale = static_cast<double>(source) / target;
    const double stretch = std::max(1.0, scale);
    std::vector<Filter> result;
    std::vector<std::vector<double>> normalized;
    double largest = 0;
    for (int out = 0; out < target; ++out) {
        const double center = (out + 0.5) * scale;
        const int start = std::max(0, static_cast<int>(center - 2 * stretch + 0.5));
        const int end = std::min(source, static_cast<int>(center + 2 * stretch + 0.5));
        std::vector<double> values;
        double sum = 0;
        for (int in = start; in < end; ++in) {
            const double value = cubic((in - center + 0.5) / stretch);
            values.push_back(value);
            sum += value;
        }
        for (auto& value : values) {
            value /= sum;
            largest = std::max(largest, value);
        }
        normalized.push_back(std::move(values));
        result.push_back({start, {}, 22});
    }
    int precision = 22;
    if (unified) {
        // Match torchvision's RGB8 antialias coefficient quantization, not
        // Pillow's fixed 22-bit coefficients. This is input preprocessing
        // semantics, independent of the GPU executing the model.
        // Reference: PyTorch UpSampleKernel.cpp, _compute_index_ranges_int16_weights.
        precision = 0;
        while (precision < 22 && std::lround(largest * (int64_t{1} << (precision + 1))) < 32768)
            ++precision;
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].precision = precision;
        for (double value : normalized[i])
            result[i].weights.push_back(
                static_cast<int32_t>(std::lround(value * (int64_t{1} << precision))));
    }
    return result;
}

std::vector<uint8_t> resize_rgb(const std::vector<uint8_t>& source, int width, int height,
                                int target_width, int target_height, bool unified) {
    const auto horizontal = filters(width, target_width, unified);
    std::vector<uint8_t> intermediate(static_cast<std::size_t>(height) * target_width * 3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < target_width; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto& filter = horizontal[x];
                int64_t value = int64_t{1} << (filter.precision - 1);
                for (std::size_t i = 0; i < filter.weights.size(); ++i)
                    value +=
                        static_cast<int64_t>(filter.weights[i]) *
                        source[(static_cast<std::size_t>(y) * width + filter.start + i) * 3 + c];
                intermediate[(static_cast<std::size_t>(y) * target_width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp<int64_t>(value >> filter.precision, 0, 255));
            }
    const auto vertical = filters(height, target_height, unified);
    std::vector<uint8_t> output(static_cast<std::size_t>(target_height) * target_width * 3);
    for (int y = 0; y < target_height; ++y)
        for (int x = 0; x < target_width; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto& filter = vertical[y];
                int64_t value = int64_t{1} << (filter.precision - 1);
                for (std::size_t i = 0; i < filter.weights.size(); ++i)
                    value += static_cast<int64_t>(filter.weights[i]) *
                             intermediate[((filter.start + i) * target_width + x) * 3 + c];
                output[(static_cast<std::size_t>(y) * target_width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp<int64_t>(value >> filter.precision, 0, 255));
            }
    return output;
}

std::string strip_thinking(const std::string& text) {
    std::string result;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        auto end = text.find("<channel|>", begin);
        if (end == std::string::npos)
            end = text.size();
        const auto thinking = text.find("<|channel>", begin);
        result += text.substr(begin, std::min(end, thinking) - begin);
        if (end == text.size())
            break;
        begin = end + std::strlen("<channel|>");
    }
    return gemma_trim_chat_text(result);
}

const char* role_name(internal::MessageRole role) {
    switch (role) {
    case internal::MessageRole::User:
        return "user";
    case internal::MessageRole::Assistant:
        return "model";
    case internal::MessageRole::System:
        return "system";
    default:
        throw std::invalid_argument("Gemma4 media accepts system, user and assistant roles");
    }
}

} // namespace

GemmaImagePatches gemma4_image_patches(const internal::ImageView& image,
                                       const GemmaImageConfig& config) {
    if (!image.data || !image.width || !image.height || image.channels != 3 ||
        image.width > INT32_MAX || image.height > INT32_MAX ||
        (image.format != internal::ImageFormat::UInt8 &&
         image.format != internal::ImageFormat::Float32))
        throw std::invalid_argument("Gemma4 requires a nonempty host RGB8 or RGB float image");
    const uint64_t count = static_cast<uint64_t>(image.width) * image.height * 3;
    const std::size_t item = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (count > std::numeric_limits<std::size_t>::max() / item || image.byte_size != count * item)
        throw std::invalid_argument("Gemma4 RGB storage does not match its dimensions");
    if (config.patch_size <= 0 || config.pooling_kernel < 2 || config.soft_tokens <= 0)
        throw std::invalid_argument("Gemma4 invalid image patch geometry");
    std::vector<uint8_t> pixels(static_cast<std::size_t>(count));
    if (image.format == internal::ImageFormat::UInt8) {
        std::memcpy(pixels.data(), image.data, pixels.size());
    } else {
        const auto* bytes = static_cast<const unsigned char*>(image.data);
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            float value;
            std::memcpy(&value, bytes + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("Gemma4 RGB float pixels must be finite in [0,1]");
            pixels[i] = static_cast<uint8_t>(std::lround(value * 255));
        }
    }
    const int teacher_patch = config.patch_size, kernel = config.pooling_kernel;
    const int patch = teacher_patch * (config.merged_patches ? kernel : 1);
    const int patches = config.soft_tokens * kernel * kernel;
    const int multiple = teacher_patch * kernel;
    const double scale = std::sqrt(static_cast<double>(patches) * teacher_patch * teacher_patch /
                                   (static_cast<double>(image.width) * image.height));
    int height = static_cast<int>(std::floor(scale * image.height / multiple)) * multiple;
    int width = static_cast<int>(std::floor(scale * image.width / multiple)) * multiple;
    if (height == 0) {
        height = multiple;
        width =
            static_cast<int>(std::min<double>(
                std::floor(static_cast<double>(image.width) / image.height), config.soft_tokens)) *
            multiple;
    } else if (width == 0) {
        width = multiple;
        height =
            static_cast<int>(std::min<double>(
                std::floor(static_cast<double>(image.height) / image.width), config.soft_tokens)) *
            multiple;
    }
    if (width <= 0 || height <= 0 ||
        static_cast<int64_t>(width) * height > int64_t{patches} * teacher_patch * teacher_patch)
        throw std::invalid_argument("Gemma4 resized image exceeds its patch budget");
    const auto resized =
        resize_rgb(pixels, image.width, image.height, width, height, config.merged_patches);
    const int columns = width / patch, rows = height / patch;
    GemmaImagePatches result;
    result.soft_tokens = rows * columns / (config.merged_patches ? 1 : kernel * kernel);
    const int capacity = config.merged_patches ? config.soft_tokens : patches;
    result.pixels.assign(static_cast<std::size_t>(capacity) * patch * patch * 3, 0);
    result.positions.assign(static_cast<std::size_t>(capacity) * 2, -1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < columns; ++x) {
            const int index = y * columns + x;
            result.positions[2 * index] = x;
            result.positions[2 * index + 1] = y;
            for (int py = 0; py < patch; ++py)
                for (int px = 0; px < patch; ++px)
                    for (int c = 0; c < 3; ++c)
                        result
                            .pixels[((static_cast<std::size_t>(index) * patch + py) * patch + px) *
                                        3 +
                                    c] =
                            config.merged_patches
                                ? resized[((y * patch + py) * width + x * patch + px) * 3 + c] *
                                      (1.0F / 255.0F)
                                : static_cast<float>(
                                      resized[((y * patch + py) * width + x * patch + px) * 3 + c] /
                                      255.0);
        }
    return result;
}

std::string gemma4_image_prompt(const internal::ImagesTextToTextRequest& input,
                                const TextGenerationConfig& config) {
    using namespace internal;
    if (!input.tools.empty())
        throw std::invalid_argument("Gemma4 image requests do not map tool definitions");
    std::vector<std::vector<ImageAudioTextPart>> parts;
    std::vector<ImageAudioTextMessage> messages;
    parts.reserve(input.messages.size());
    for (const auto& message : input.messages) {
        parts.emplace_back();
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<TextPartView>(&part))
                parts.back().emplace_back(*text);
            else if (const auto* image = std::get_if<ImageView>(&part))
                parts.back().emplace_back(*image);
            else
                throw std::invalid_argument("Gemma4 media does not map tool or reasoning parts");
        }
        messages.push_back({message.role, {parts.back().data(), parts.back().size()}});
    }
    return gemma4_media_prompt({{messages.data(), messages.size()}}, config);
}

std::string gemma4_media_prompt(const internal::ImageAudioTextToTextRequest& input,
                                const TextGenerationConfig& config) {
    using namespace internal;
    if (!config.use_chat_template || input.messages.empty())
        throw std::invalid_argument(
            "Gemma4 media requires messages and its provider chat template");
    for (const auto& message : input.messages) {
        role_name(message.role);
        if (message.parts.empty())
            throw std::invalid_argument("Gemma4 media message cannot have empty content");
        for (const auto& part : message.parts)
            if (!std::holds_alternative<TextPartView>(part) &&
                !std::holds_alternative<ImageView>(part) &&
                !std::holds_alternative<AudioView>(part))
                throw std::invalid_argument("Gemma4 media does not map tool or reasoning parts");
    }
    std::size_t first = 0;
    std::string prompt = "<bos>";
    const bool first_system =
        config.system_prompt.empty() && input.messages[0].role == MessageRole::System;
    if (config.enable_thinking || !config.system_prompt.empty() || first_system) {
        prompt += "<|turn>system\n";
        if (config.enable_thinking)
            prompt += "<|think|>\n";
        if (!config.system_prompt.empty())
            prompt += gemma_trim_chat_text(config.system_prompt);
        else if (first_system) {
            for (const auto& part : input.messages[0].parts) {
                const auto* text = std::get_if<TextPartView>(&part);
                if (!text)
                    throw std::invalid_argument("Gemma4 first system message accepts text only");
                prompt += gemma_trim_chat_text(std::string(text->text)) + " ";
            }
            first = 1;
        }
        prompt += "<turn|>\n";
    }
    for (std::size_t i = first; i < input.messages.size(); ++i) {
        const auto& message = input.messages[i];
        const bool assistant = message.role == MessageRole::Assistant;
        const bool continuation =
            assistant && i > first && input.messages[i - 1].role == MessageRole::Assistant;
        if (!continuation)
            prompt += std::string("<|turn>") + role_name(message.role) + "\n";
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<TextPartView>(&part))
                prompt += assistant ? strip_thinking(std::string(text->text))
                                    : gemma_trim_chat_text(std::string(text->text));
            else if (std::holds_alternative<ImageView>(part))
                prompt += "<|image|>";
            else
                prompt += "<|audio|>";
        }
        if (!(assistant && i + 1 < input.messages.size() &&
              input.messages[i + 1].role == MessageRole::Assistant))
            prompt += "<turn|>\n";
    }
    return prompt + "<|turn>model\n";
}

GemmaVisionEncoder::GemmaVisionEncoder(std::unique_ptr<ITrtModule> module, GemmaImageConfig config)
    : module_(std::move(module)), config_(config) {
    if (!module_ || !module_->ok() || config_.hidden_size <= 0)
        throw std::runtime_error("Gemma4 invalid native vision engine");
}

std::vector<float> GemmaVisionEncoder::encode(const GemmaImagePatches& input) {
    const int kernel = config_.merged_patches ? 1 : config_.pooling_kernel;
    const int patch = config_.patch_size * (config_.merged_patches ? config_.pooling_kernel : 1);
    const int patches = config_.soft_tokens * kernel * kernel;
    TensorMap inputs;
    inputs["pixel_values"] = {
        const_cast<float*>(input.pixels.data()), {patches, 3 * patch * patch}, DType::kFloat32};
    inputs["pixel_position_ids"] = {
        const_cast<int32_t*>(input.positions.data()), {patches, 2}, DType::kInt32};
    const auto outputs = module_->forward(inputs);
    const auto& features = outputs.at("image_features");
    const auto& mask = outputs.at("image_feature_mask");
    if (features.dtype != DType::kFloat32 || mask.dtype != DType::kInt32 ||
        features.numel() != static_cast<std::size_t>(config_.soft_tokens) * config_.hidden_size ||
        mask.numel() != static_cast<std::size_t>(config_.soft_tokens))
        throw std::runtime_error("Gemma4 vision engine output contract mismatch");
    const auto* values = static_cast<const float*>(features.data);
    const auto* valid = static_cast<const int32_t*>(mask.data);
    std::vector<float> result;
    for (int i = 0; i < config_.soft_tokens; ++i)
        if (valid[i])
            result.insert(result.end(), values + i * config_.hidden_size,
                          values + (i + 1) * config_.hidden_size);
    if (result.size() != static_cast<std::size_t>(input.soft_tokens) * config_.hidden_size ||
        !std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Gemma4 vision engine returned invalid conditioned features");
    return result;
}

} // namespace trtmc
