/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::blip {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
Tensor require_output(const TensorMap& outputs, const char* name,
                      const std::vector<std::int64_t>& shape) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || !it->second.data || it->second.dtype != DType::kFloat32 ||
        it->second.shape != shape)
        throw std::runtime_error(std::string("blip invalid engine output: ") + name);
    const auto* data = static_cast<const float*>(it->second.data);
    if (!std::all_of(data, data + it->second.numel(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("blip engine output contains nonfinite values");
    return it->second;
}
} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> decoder,
                   std::unique_ptr<ITokenizer> tokenizer, RuntimeConfig config)
    : vision_(std::move(vision)), decoder_(std::move(decoder)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)) {
    if (!vision_ || !decoder_ || !vision_->ok() || !decoder_->ok() || !tokenizer_)
        throw std::runtime_error("blip requires two loaded engines and a tokenizer");
    if (config_.max_length < 2 || config_.max_length > 512 || config_.image_tokens <= 0 ||
        config_.vision_width <= 0 || config_.vocab_size <= 0 || config_.bos_token_id < 0 ||
        config_.bos_token_id >= config_.vocab_size || config_.eos_token_id < 0 ||
        config_.eos_token_id >= config_.vocab_size)
        throw std::runtime_error("blip invalid runtime dimensions or special tokens");
    fields_ = {{"max_new_tokens", internal::ConfigKind::I64,
                internal::ConfigValue{std::int64_t{20}}, "Maximum generated caption tokens"}};
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::IImagesTextToText>(*this, {fields_.data(), fields_.size()})};
}

TextResult Pipeline::run(const internal::ImagesTextToTextRequest& request,
                         internal::ConfigView supplied) {
    const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
    internal::validate_config(fields, supplied);
    const auto max_new = *internal::config_get<std::int64_t>(supplied, fields, "max_new_tokens");
    if (max_new <= 0 || max_new > config_.max_length)
        throw internal::ConfigError("blip max_new_tokens must fit the bundle sequence capacity");
    if (request.messages.size() != 1 || request.messages[0].role != internal::MessageRole::User ||
        !request.tools.empty())
        throw std::invalid_argument("blip expects one user message without tools");
    const internal::ImageView* image = nullptr;
    std::string prompt;
    bool has_text = false;
    for (const auto& part : request.messages[0].parts) {
        if (const auto* value = std::get_if<internal::ImageView>(&part)) {
            if (image)
                throw std::invalid_argument("blip expects exactly one image");
            image = value;
        } else if (const auto* value = std::get_if<internal::TextPartView>(&part)) {
            if (has_text)
                throw std::invalid_argument("blip expects at most one caption prefix");
            prompt = value->text;
            has_text = true;
        } else {
            throw std::invalid_argument(
                "blip accepts only an image and an optional caption prefix");
        }
    }
    if (!image || !image->data || image->channels != 3 || !image->height || !image->width ||
        image->height > 32768 || image->width > 32768 ||
        (image->format != internal::ImageFormat::Float32 &&
         image->format != internal::ImageFormat::UInt8))
        throw std::invalid_argument("blip requires a nonempty contiguous RGB image");
    const std::size_t count = std::size_t{image->height} * image->width * 3;
    const auto element_size = image->format == internal::ImageFormat::Float32 ? sizeof(float) : 1;
    if (image->byte_size != count * element_size)
        throw std::invalid_argument("blip image byte size does not match its dimensions");

    auto ids = tokenizer_->encode(prompt);
    // The tokenizer supplies CLS ... SEP. Upstream generate replaces CLS with
    // the decoder BOS and removes SEP, preserving the supplied caption prefix.
    if (ids.size() < 2)
        throw std::runtime_error("blip tokenizer omitted its special-token frame");
    ids.front() = config_.bos_token_id;
    ids.pop_back();
    if (ids.size() + static_cast<std::size_t>(max_new) >
        static_cast<std::size_t>(config_.max_length))
        throw std::invalid_argument("blip caption prefix and output exceed sequence capacity");
    if (std::any_of(ids.begin(), ids.end(),
                    [&](auto id) { return id < 0 || id >= config_.vocab_size; }))
        throw std::invalid_argument("blip tokenizer produced an out-of-vocabulary token");

    const auto start = Clock::now();
    std::vector<float> source(count);
    if (image->format == internal::ImageFormat::UInt8) {
        const auto* data = static_cast<const std::uint8_t*>(image->data);
        std::transform(data, data + count, source.begin(), [](auto v) { return v / 255.0F; });
    } else {
        const auto* data = static_cast<const float*>(image->data);
        if (!std::all_of(data, data + count,
                         [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; }))
            throw std::invalid_argument("blip float RGB values must be finite and in [0,1]");
        std::copy(data, data + count, source.begin());
    }
    auto pixels =
        preprocess_blip_image(source.data(), image->height, image->width, config_.preprocess);
    TextResult result;
    result.setup_ms = elapsed(start);
    const auto prefill = Clock::now();
    auto vision_outputs = vision_->forward(
        {{"pixel_values",
          {pixels.data(),
           {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w},
           DType::kFloat32}}});
    const auto embeds = require_output(vision_outputs, "image_embeds",
                                       {config_.image_tokens, config_.vision_width});
    std::vector<float> image_embeds(embeds.numel());
    std::memcpy(image_embeds.data(), embeds.data, embeds.nbytes());
    result.prefill_ms = elapsed(prefill);
    const auto decode_start = Clock::now();
    for (std::int64_t step = 0; step < max_new; ++step) {
        const auto length = static_cast<std::int32_t>(ids.size());
        std::vector<std::int32_t> positions(length);
        std::iota(positions.begin(), positions.end(), 0);
        std::vector<float> mask(static_cast<std::size_t>(length) * length, 0);
        for (std::int32_t q = 0; q < length; ++q)
            for (std::int32_t k = q + 1; k < length; ++k)
                mask[static_cast<std::size_t>(q) * length + k] = -std::numeric_limits<float>::max();
        std::int32_t last_index = length - 1;
        auto outputs = decoder_->forward(
            {{"image_embeds",
              {image_embeds.data(), {config_.image_tokens, config_.vision_width}, DType::kFloat32}},
             {"input_ids", {ids.data(), {length}, DType::kInt32}},
             {"position_ids", {positions.data(), {length}, DType::kInt32}},
             {"causal_mask", {mask.data(), {1, length, length}, DType::kFloat32}},
             {"last_index", {&last_index, {1}, DType::kInt32}}});
        const auto logits = require_output(outputs, "logits", {1, config_.vocab_size});
        const auto* values = static_cast<const float*>(logits.data);
        const auto next = static_cast<std::int32_t>(
            std::max_element(values, values + config_.vocab_size) - values);
        ids.push_back(next);
        result.token_ids.push_back(next);
        if (next == config_.eos_token_id)
            break;
    }
    result.decode_ms = elapsed(decode_start);
    // Native generation returns the continuation; the conditioning prefix is
    // retained only inside the decoder, like other text-generation tasks.
    result.text = tokenizer_->decode(result.token_ids);
    return result;
}
} // namespace trtmc::blip
