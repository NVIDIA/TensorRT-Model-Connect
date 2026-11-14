/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/owlv2/runtime/pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

const internal::ConfigField kFields[] = {
    {"score_threshold", internal::ConfigKind::F64, std::nullopt,
     "Minimum score for a patch to be reported; the bundle's value is used when unset"},
};

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

float threshold_from(internal::ConfigView supplied, std::string_view name, float fallback) {
    const auto value = internal::config_get<double>(supplied, kFields, name);
    if (!value.has_value())
        return fallback;
    if (!std::isfinite(*value) || *value < 0.0 || *value > 1.0)
        throw internal::ConfigError("owlv2 score_threshold must lie in [0, 1]");
    return static_cast<float>(*value);
}

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("owlv2 engine did not return ") + name);
    if (found->second.data == nullptr || found->second.dtype != DType::kFloat32 ||
        found->second.numel() <= 0)
        throw std::runtime_error(std::string("owlv2 engine returned an unusable ") + name);
    return found->second;
}

const float* require_matrix(const TensorMap& outputs, const char* name, std::int64_t rows,
                            std::int64_t columns) {
    const Tensor& tensor = require_output(outputs, name);
    if (tensor.shape.size() != 2 || tensor.shape[0] != rows || tensor.shape[1] != columns)
        throw std::runtime_error(std::string("owlv2 output has unexpected shape: ") + name);
    return static_cast<const float*>(tensor.data);
}

std::string trim(std::string_view value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string_view::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\n\r");
    return std::string(value.substr(first, last - first + 1));
}

} // namespace

std::vector<std::string> split_owlv2_phrases(std::string_view text) {
    std::vector<std::string> phrases;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto stop = text.find('.', start);
        auto phrase = trim(text.substr(
            start, stop == std::string_view::npos ? std::string_view::npos : stop - start));
        if (!phrase.empty())
            phrases.push_back(std::move(phrase));
        if (stop == std::string_view::npos)
            break;
        start = stop + 1;
    }
    return phrases;
}

Owlv2Pipeline::Owlv2Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> text,
                             std::shared_ptr<ITokenizer> tokenizer, Owlv2RuntimeConfig config)
    : vision_(std::move(vision)), text_(std::move(text)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)) {
    if (!vision_ || !text_)
        throw std::runtime_error("Owlv2Pipeline: missing engine");
    if (!tokenizer_)
        throw std::runtime_error("Owlv2Pipeline: missing tokenizer");
    if (config_.num_patches != config_.grid * config_.grid)
        throw std::runtime_error("Owlv2Pipeline: patch count is not the square of the grid");
}

std::vector<internal::TaskInstance> Owlv2Pipeline::task_bindings() {
    std::vector<internal::TaskInstance> bindings;
    bindings.push_back(internal::bind<internal::IImageTextToBoxes>(*this, kFields));
    return bindings;
}

std::vector<float> Owlv2Pipeline::embed_query(const std::string& phrase) {
    auto ids = tokenizer_->encode(phrase);
    if (ids.empty())
        throw std::runtime_error("owlv2 tokenizer returned no tokens");

    // The reference locates the pooled row with argmax over the ids, which
    // works only because the end marker is the largest id and the pad is 0. A
    // phrase long enough to be truncated would lose that marker off the end,
    // so the marker is put back on the last row and the index is passed to the
    // engine rather than rediscovered there.
    const auto end_id = ids.back();
    const auto limit = static_cast<std::size_t>(config_.max_text_len);
    if (ids.size() > limit) {
        ids.resize(limit);
        ids.back() = end_id;
    }
    const auto eos_index = static_cast<std::int32_t>(ids.size() - 1);
    ids.resize(limit, static_cast<std::int32_t>(config_.pad_token_id));

    Tensor input_ids{ids.data(), {config_.max_text_len}, DType::kInt32};
    std::int32_t index = eos_index;
    Tensor eos{&index, {1}, DType::kInt32};
    auto outputs = text_->forward({{"input_ids", input_ids}, {"eos_index", eos}});
    const float* values = require_matrix(outputs, "query_embeds", 1, config_.query_dim);
    return std::vector<float>(values, values + config_.query_dim);
}

internal::GroundedBoxesResult Owlv2Pipeline::run(const internal::ImageTextToBoxesRequest& request,
                                                 internal::ConfigView config) {
    const float threshold = threshold_from(config, "score_threshold", config_.score_threshold);

    const internal::ImageView& image = request.image;
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("owlv2 requires contiguous float32 RGB input");
    }
    if (request.text.empty())
        throw std::invalid_argument("owlv2 requires a text prompt");

    const auto phrases = split_owlv2_phrases(request.text);
    if (phrases.empty())
        throw std::invalid_argument("owlv2 prompt contains no phrases");

    auto pixel_values = preprocess_owlv2_image(
        static_cast<const float*>(image.data), static_cast<std::int32_t>(image.height),
        static_cast<std::int32_t>(image.width), config_.preprocess);
    Tensor pixels{pixel_values.data(),
                  {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w},
                  DType::kFloat32};
    auto outputs = vision_->forward({{"pixel_values", pixels}});

    const std::int32_t patches = config_.num_patches;
    const float* boxes = require_matrix(outputs, "pred_boxes", patches, 4);
    const float* embeds = require_matrix(outputs, "class_embeds", patches, config_.query_dim);
    const float* shift = require_matrix(outputs, "logit_shift", patches, 1);
    const float* scale = require_matrix(outputs, "logit_scale", patches, 1);

    std::vector<std::vector<float>> queries;
    queries.reserve(phrases.size());
    for (const auto& phrase : phrases)
        queries.push_back(embed_query(phrase));

    // The reference scales the boxes by the side of the square the image was
    // padded to, so they land in the original image's pixel frame. They are
    // not clipped to it: a box may run past the edge, and that is what the
    // model predicted.
    const auto side = static_cast<float>(compute_owlv2_pad_side(
        static_cast<std::int32_t>(image.height), static_cast<std::int32_t>(image.width)));

    internal::GroundedBoxesResult result;
    for (std::int32_t patch = 0; patch < patches; ++patch) {
        const float* row = embeds + static_cast<std::size_t>(patch) * config_.query_dim;
        float best = -std::numeric_limits<float>::infinity();
        std::size_t label = 0;
        for (std::size_t q = 0; q < queries.size(); ++q) {
            float dot = 0.0F;
            for (std::int32_t d = 0; d < config_.query_dim; ++d)
                dot += row[d] * queries[q][static_cast<std::size_t>(d)];
            const float logit = (dot + shift[patch]) * scale[patch];
            if (logit > best) {
                best = logit;
                label = q;
            }
        }
        const float score = sigmoid(best);
        if (score <= threshold)
            continue;

        const float* box = boxes + static_cast<std::size_t>(patch) * 4;
        internal::GroundedBox grounded;
        grounded.box.x_min = (box[0] - box[2] * 0.5F) * side;
        grounded.box.y_min = (box[1] - box[3] * 0.5F) * side;
        grounded.box.x_max = (box[0] + box[2] * 0.5F) * side;
        grounded.box.y_max = (box[1] + box[3] * 0.5F) * side;
        grounded.label = phrases[label];
        grounded.confidence = score;
        result.boxes.push_back(std::move(grounded));
    }
    std::sort(result.boxes.begin(), result.boxes.end(),
              [](const internal::GroundedBox& left, const internal::GroundedBox& right) {
                  return left.confidence.value_or(0.0F) > right.confidence.value_or(0.0F);
              });
    return result;
}

} // namespace trtmc
