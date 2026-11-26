/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mask2former/runtime/pipeline.h"

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace trtmc::mask2former {
namespace {

constexpr const char* kInput = "pixel_values";
constexpr const char* kOutput = "semantic_scores";

using Json = nlohmann::json;

[[noreturn]] void invalid_metadata(const std::string& message) {
    throw std::runtime_error("Mask2Former runtime.json " + message);
}

// The default parser keeps the last of two equal keys; refuse them instead.
Json parse_unique_keys(const std::vector<char>& data) {
    std::vector<std::set<std::string>> keys;
    try {
        return Json::parse(data.begin(), data.end(),
                           [&keys](int, Json::parse_event_t event, Json& parsed) {
                               if (event == Json::parse_event_t::object_start) {
                                   keys.emplace_back();
                               } else if (event == Json::parse_event_t::object_end) {
                                   keys.pop_back();
                               } else if (event == Json::parse_event_t::key &&
                                          !keys.back().insert(parsed.get<std::string>()).second) {
                                   invalid_metadata("repeats the key " + parsed.get<std::string>());
                               }
                               return true;
                           });
    } catch (const Json::exception& error) {
        invalid_metadata(std::string("is not valid JSON: ") + error.what());
    }
}

const Json& field(const Json& json, const char* name) {
    const auto found = json.find(name);
    if (found == json.end())
        invalid_metadata(std::string("is missing ") + name);
    return *found;
}

// A JSON integer only: get<std::int32_t>() would truncate 384.5 and wrap values beyond int32.
// The parser stores every nonnegative integer as unsigned.
std::int32_t positive_int32(const Json& json, const char* name) {
    const auto& value = field(json, name);
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number > 0 &&
            number <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            return static_cast<std::int32_t>(number);
    }
    invalid_metadata(std::string(name) + " must be a positive integer within int32");
}

double finite_number(const Json& value, const char* name) {
    if (!value.is_number())
        invalid_metadata(std::string(name) + " must be a number");
    const auto number = value.get<double>();
    if (!std::isfinite(number))
        invalid_metadata(std::string(name) + " must be finite");
    return number;
}

std::array<float, 3> float_channels(const Json& json, const char* name) {
    const auto& values = field(json, name);
    if (!values.is_array() || values.size() != 3)
        invalid_metadata(std::string(name) + " must hold three numbers");
    std::array<float, 3> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto number = finite_number(values[index], name);
        // Converting a double beyond float range to float is undefined.
        if (std::fabs(number) > static_cast<double>(std::numeric_limits<float>::max()))
            invalid_metadata(std::string(name) + " must fit float");
        result[index] = static_cast<float>(number);
    }
    return result;
}

std::vector<std::string> class_names(const Json& json) {
    const auto& values = field(json, "labels");
    if (!values.is_array() || values.empty())
        invalid_metadata("labels must be a nonempty array");
    std::vector<std::string> names;
    names.reserve(values.size());
    for (const auto& value : values) {
        if (!value.is_string() || value.get_ref<const std::string&>().empty())
            invalid_metadata("labels must be nonempty strings");
        names.push_back(value.get<std::string>());
    }
    return names;
}

// The checkpoint's processor resizes 8-bit images. Float32 pixels in [0, 1] are quantized to the
// nearest 8-bit value, which recovers the exact bytes of an 8-bit image decoded as value / 255.
std::vector<std::uint8_t> rgb8(const internal::ImageView& image) {
    if (image.data == nullptr || image.channels != 3 || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float))
        throw std::invalid_argument("Mask2Former requires a nonempty contiguous RGB image");
    const auto count = static_cast<std::size_t>(image.height) * image.width * 3;
    if (image.format == internal::ImageFormat::UInt8) {
        if (image.byte_size != count)
            throw std::invalid_argument("Mask2Former 8-bit RGB input has the wrong byte size");
        const auto* values = static_cast<const std::uint8_t*>(image.data);
        return {values, values + count};
    }
    if (image.format != internal::ImageFormat::Float32 || image.byte_size != count * sizeof(float))
        throw std::invalid_argument("Mask2Former requires 8-bit or float32 RGB input");
    const auto* values = static_cast<const float*>(image.data);
    std::vector<std::uint8_t> bytes(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float value = values[index];
        if (!(value >= 0.0F && value <= 1.0F))
            throw std::invalid_argument("Mask2Former float32 RGB values must lie in [0, 1]");
        bytes[index] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
    }
    return bytes;
}

bool has_shape(const ITrtModule& engine, const char* name, const std::vector<std::int64_t>& shape) {
    return engine.tensor_dtype(name) == DType::kFloat32 && engine.tensor_shape(name) == shape;
}

} // namespace

RuntimeMetadata parse_runtime_json(const std::vector<char>& data) {
    const auto json = parse_unique_keys(data);
    if (!json.is_object())
        invalid_metadata("must be a JSON object");
    RuntimeMetadata metadata;
    metadata.preprocess.height = positive_int32(json, "input_height");
    metadata.preprocess.width = positive_int32(json, "input_width");
    metadata.preprocess.rescale_factor =
        finite_number(field(json, "rescale_factor"), "rescale_factor");
    metadata.preprocess.mean = float_channels(json, "image_mean");
    metadata.preprocess.std = float_channels(json, "image_std");
    metadata.head.score_height = positive_int32(json, "score_height");
    metadata.head.score_width = positive_int32(json, "score_width");
    metadata.head.labels = class_names(json);
    if (static_cast<std::size_t>(positive_int32(json, "num_classes")) !=
        metadata.head.labels.size())
        invalid_metadata("labels do not match num_classes");
    return metadata;
}

Mask2FormerSemanticSegmentation::Mask2FormerSemanticSegmentation(std::unique_ptr<ITrtModule> engine,
                                                                 PreprocessConfig preprocess,
                                                                 SemanticHead head)
    : engine_(std::move(engine)), preprocess_(preprocess), head_(std::move(head)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("Mask2Former engine failed to load");
    const auto classes = static_cast<std::int64_t>(head_.labels.size());
    if (preprocess_.height <= 0 || preprocess_.width <= 0 || head_.score_height <= 0 ||
        head_.score_width <= 0 || classes <= 0 ||
        classes > std::numeric_limits<std::int32_t>::max() || !(preprocess_.rescale_factor > 0.0) ||
        !std::isfinite(preprocess_.rescale_factor))
        throw std::runtime_error("Mask2Former runtime.json does not match its runtime contract");
    if (!normalization_is_finite(preprocess_))
        throw std::runtime_error("Mask2Former rescale_factor, image_mean and image_std must "
                                 "normalize 8-bit values to finite float32 values");
    if (!engine_->has_input(kInput) || !engine_->has_output(kOutput) ||
        !has_shape(*engine_, kInput, {1, 3, preprocess_.height, preprocess_.width}) ||
        !has_shape(*engine_, kOutput, {1, classes, head_.score_height, head_.score_width}))
        throw std::runtime_error("Mask2Former engine I/O does not match runtime.json");
}

internal::SemanticSegmentationResult
Mask2FormerSemanticSegmentation::run(const internal::ImageToSemanticSegmentationRequest& request,
                                     internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("Mask2Former has no runtime configuration");
    const auto& image = request.image;
    const auto bytes = rgb8(image);
    const auto height = static_cast<std::int32_t>(image.height);
    const auto width = static_cast<std::int32_t>(image.width);
    auto pixels = preprocess(bytes.data(), height, width, preprocess_);

    Tensor input;
    input.data = pixels.data();
    input.shape = {1, 3, preprocess_.height, preprocess_.width};
    input.dtype = DType::kFloat32;
    const auto outputs = engine_->forward({{kInput, input}});
    const auto found = outputs.find(kOutput);
    const auto classes = static_cast<std::int32_t>(head_.labels.size());
    const std::vector<std::int64_t> expected{1, classes, head_.score_height, head_.score_width};
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != DType::kFloat32 || found->second.shape != expected)
        throw std::runtime_error("Mask2Former engine returned no complete float32 semantic scores");

    internal::SemanticSegmentationResult result;
    result.labels = resize_argmax(static_cast<const float*>(found->second.data), classes,
                                  head_.score_height, head_.score_width, height, width);
    result.height = image.height;
    result.width = image.width;
    result.class_ids.resize(head_.labels.size());
    std::iota(result.class_ids.begin(), result.class_ids.end(), 0);
    result.class_names = head_.labels;
    return result;
}

} // namespace trtmc::mask2former
