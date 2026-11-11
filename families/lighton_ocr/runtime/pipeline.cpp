/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace trtmc::lighton_ocr {
namespace {

Span<const internal::ConfigField> fields() {
    static const std::vector<internal::ConfigField> result{
        {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
         "Maximum generated tokens including EOS; default from the checkpoint bundle."},
        {"temperature", internal::ConfigKind::F64, internal::ConfigValue{0.0},
         "Only deterministic greedy generation (0) is supported."},
    };
    return {result.data(), result.size()};
}

internal::ImageView request_image(const internal::ImagesTextToTextRequest& request) {
    if (!request.tools.empty() || request.messages.size() != 1 ||
        request.messages[0].role != internal::MessageRole::User)
        throw std::invalid_argument("LightOnOCR requires exactly one user message without tools");
    const internal::ImageView* image = nullptr;
    for (const auto& part : request.messages[0].parts) {
        if (const auto* candidate = std::get_if<internal::ImageView>(&part)) {
            if (image)
                throw std::invalid_argument("LightOnOCR supports exactly one image");
            image = candidate;
        } else if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
            if (!text->text.empty())
                throw std::invalid_argument("LightOnOCR accepts image-only OCR with empty text");
        } else {
            throw std::invalid_argument("LightOnOCR does not accept reasoning or tool input");
        }
    }
    if (!image)
        throw std::invalid_argument("LightOnOCR requires an image");
    return *image;
}

std::vector<float> output(const TensorMap& outputs, const char* name,
                          const std::vector<std::int64_t>& shape) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.shape != shape || (!found->second.data && found->second.numel()))
        throw std::runtime_error(std::string("LightOnOCR engine output violates its contract: ") +
                                 name);
    const auto* data = static_cast<const float*>(found->second.data);
    std::vector<float> result(data, data + found->second.numel());
    if (!std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("LightOnOCR engine output contains nonfinite values");
    return result;
}

bool same_prefix(const std::vector<float>& before, const std::vector<float>& after,
                 std::int32_t groups, std::int32_t length, std::int32_t added,
                 std::int32_t dimension) {
    const auto prefix = static_cast<std::size_t>(length) * dimension;
    const auto stride = static_cast<std::size_t>(length + added) * dimension;
    if (!prefix)
        return true;
    for (std::int32_t group = 0; group < groups; ++group)
        if (std::memcmp(before.data() + group * prefix, after.data() + group * stride,
                        prefix * sizeof(float)) != 0)
            return false;
    return true;
}

double milliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> decoder,
                   std::vector<float> embedding, Tokenizer tokenizer, RuntimeConfig config)
    : vision_(std::move(vision)), decoder_(std::move(decoder)), embedding_(std::move(embedding)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!vision_ || !decoder_ || !vision_->ok() || !decoder_->ok() || config_.hidden_size <= 0 ||
        config_.vocab_size <= 0 ||
        embedding_.size() != static_cast<std::size_t>(config_.hidden_size) * config_.vocab_size)
        throw std::invalid_argument("LightOnOCR invalid native engine or embedding contract");
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::IImagesTextToText>(*this, fields())};
}

std::vector<float> Pipeline::embed(std::int32_t id) const {
    if (id < 0 || id >= config_.vocab_size)
        throw std::invalid_argument("LightOnOCR input token is outside the vocabulary");
    const auto begin = embedding_.begin() + static_cast<std::size_t>(id) * config_.hidden_size;
    return {begin, begin + config_.hidden_size};
}

Pipeline::Prompt Pipeline::prompt(const internal::ImageView& image, std::int32_t output_budget) {
    Prompt result;
    result.image = preprocess(image, config_.image);
    const auto multiple = config_.image.patch_size * config_.image.spatial_merge_size;
    const auto count = (result.image.height / multiple) * (result.image.width / multiple);
    result.ids = config_.prompt_prefix_ids;
    result.ids.insert(result.ids.end(), static_cast<std::size_t>(count), config_.image_token_id);
    result.ids.insert(result.ids.end(), config_.prompt_suffix_ids.begin(),
                      config_.prompt_suffix_ids.end());
    if (result.ids.size() + static_cast<std::size_t>(output_budget) >
        static_cast<std::size_t>(config_.max_sequence_length))
        throw std::invalid_argument(
            "LightOnOCR prompt and output budget exceed the bundle capacity");
    auto vision = vision_->forward({
        {"pixel_values",
         {result.image.pixels.data(),
          {1, 3, result.image.height, result.image.width},
          DType::kFloat32}},
        {"vision_position_ids",
         {result.image.position_ids.data(),
          {static_cast<std::int64_t>(result.image.position_ids.size())},
          DType::kInt32}},
    });
    result.features = output(vision, "image_features", {count, config_.hidden_size});
    for (const auto id : config_.prompt_prefix_ids) {
        const auto row = embed(id);
        result.embeddings.insert(result.embeddings.end(), row.begin(), row.end());
    }
    result.embeddings.insert(result.embeddings.end(), result.features.begin(),
                             result.features.end());
    for (const auto id : config_.prompt_suffix_ids) {
        const auto row = embed(id);
        result.embeddings.insert(result.embeddings.end(), row.begin(), row.end());
    }
    return result;
}

std::vector<float> Pipeline::decode(const std::vector<float>& embedding, State& state,
                                    bool* prefix_unchanged) {
    if (embedding.empty() || embedding.size() % static_cast<std::size_t>(config_.hidden_size))
        throw std::invalid_argument("LightOnOCR decoder embedding shape is invalid");
    const auto length = static_cast<std::int32_t>(embedding.size() / config_.hidden_size);
    if (state.length < 0 || length > config_.max_sequence_length - state.length)
        throw std::invalid_argument(
            "LightOnOCR decoder input exceeds the bundle sequence capacity");
    std::vector<std::int32_t> positions(static_cast<std::size_t>(length));
    std::iota(positions.begin(), positions.end(), state.length);
    const auto total = state.length + length;
    std::vector<float> mask(static_cast<std::size_t>(length) * total, 0.0F);
    for (std::int32_t row = 0; row < length; ++row)
        std::fill(mask.begin() + static_cast<std::size_t>(row) * total + state.length + row + 1,
                  mask.begin() + static_cast<std::size_t>(row + 1) * total, -1.0e30F);
    float empty = 0;
    const std::vector<std::int64_t> cache_shape{
        config_.num_hidden_layers, config_.num_key_value_heads, state.length, config_.head_dim};
    const auto outputs = decoder_->forward({
        {"inputs_embeds",
         {const_cast<float*>(embedding.data()), {length, config_.hidden_size}, DType::kFloat32}},
        {"position_ids", {positions.data(), {length}, DType::kInt32}},
        {"attention_mask", {mask.data(), {length, total}, DType::kFloat32}},
        {"past_key", {state.key.empty() ? &empty : state.key.data(), cache_shape, DType::kFloat32}},
        {"past_value",
         {state.value.empty() ? &empty : state.value.data(), cache_shape, DType::kFloat32}},
    });
    auto logits = output(outputs, "logits", {1, config_.vocab_size});
    const std::vector<std::int64_t> present_shape{
        config_.num_hidden_layers, config_.num_key_value_heads, total, config_.head_dim};
    auto key = output(outputs, "present_key", present_shape);
    auto value = output(outputs, "present_value", present_shape);
    if (prefix_unchanged) {
        const auto groups = config_.num_hidden_layers * config_.num_key_value_heads;
        *prefix_unchanged =
            *prefix_unchanged &&
            same_prefix(state.key, key, groups, state.length, length, config_.head_dim) &&
            same_prefix(state.value, value, groups, state.length, length, config_.head_dim);
    }
    state = {std::move(key), std::move(value), total};
    return logits;
}

bool Pipeline::is_eos(std::int32_t id) const {
    return std::find(config_.eos_token_ids.begin(), config_.eos_token_ids.end(), id) !=
           config_.eos_token_ids.end();
}

TextResult Pipeline::run(const internal::ImagesTextToTextRequest& request,
                         internal::ConfigView supplied) {
    internal::validate_config(fields(), supplied);
    const auto maximum = internal::config_get<std::int64_t>(supplied, fields(), "max_new_tokens")
                             .value_or(config_.max_new_tokens);
    const auto temperature =
        internal::config_get<double>(supplied, fields(), "temperature").value();
    if (maximum <= 0 || maximum > config_.max_sequence_length || !std::isfinite(temperature) ||
        temperature != 0)
        throw std::invalid_argument(
            "LightOnOCR requires a positive token budget and greedy temperature 0");
    const auto image = request_image(request);
    const auto started = std::chrono::steady_clock::now();
    auto prepared = prompt(image, static_cast<std::int32_t>(maximum));
    State state;
    auto logits = decode(prepared.embeddings, state);
    const auto prefilled = std::chrono::steady_clock::now();
    std::vector<std::int32_t> tokens;
    for (std::int64_t index = 0; index < maximum; ++index) {
        const auto token = static_cast<std::int32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        tokens.push_back(token);
        if (is_eos(token) || index + 1 == maximum)
            break;
        logits = decode(embed(token), state);
    }
    const auto finished = std::chrono::steady_clock::now();
    return {tokenizer_.decode(tokens), tokens, milliseconds(started, prefilled),
            milliseconds(prefilled, finished)};
}

Observations Pipeline::observe(const internal::ImageView& image,
                               const std::vector<std::int32_t>& forced_ids) {
    if (forced_ids.size() != 1)
        throw std::invalid_argument(
            "LightOnOCR diagnostic requires exactly one forced continuation token");
    const auto forced = embed(forced_ids[0]);
    auto prepared = prompt(image, 1);
    State state;
    Observations result;
    result.image = std::move(prepared.image);
    result.image_features = std::move(prepared.features);
    result.prompt_ids = std::move(prepared.ids);
    result.logits.push_back(decode(prepared.embeddings, state, &result.cache_prefix_unchanged));
    result.logits.push_back(decode(forced, state, &result.cache_prefix_unchanged));
    return result;
}

} // namespace trtmc::lighton_ocr
