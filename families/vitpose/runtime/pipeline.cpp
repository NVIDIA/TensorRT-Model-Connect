/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vitpose/runtime/pipeline.h"

#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <utility>

namespace trtmc::vitpose {
namespace {

constexpr const char* kInput = "pixel_values";
constexpr const char* kOutput = "heatmaps";

const internal::ConfigField kFields[] = {
    {"person_box", internal::ConfigKind::F64List, std::nullopt,
     "Person box [x_min, y_min, x_max, y_max] in original-image pixels; the whole image when "
     "unset"},
};

using Json = nlohmann::json;

[[noreturn]] void invalid_metadata(const std::string& message) {
    throw std::runtime_error("vitpose runtime.json " + message);
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

// A JSON integer only: get<std::int32_t>() would truncate 256.5 and wrap values beyond int32. The
// parser stores every non-negative integer as unsigned.
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

std::vector<std::string> keypoint_labels(const Json& json) {
    const auto& values = field(json, "labels");
    if (!values.is_array() || values.empty())
        invalid_metadata("labels must be a non-empty array");
    std::vector<std::string> names;
    std::set<std::string> seen;
    for (const auto& value : values) {
        if (!value.is_string() || value.get_ref<const std::string&>().empty())
            invalid_metadata("labels must be non-empty strings");
        if (!seen.insert(value.get<std::string>()).second)
            invalid_metadata("labels repeat " + value.get<std::string>());
        names.push_back(value.get<std::string>());
    }
    return names;
}

bool has_shape(const ITrtModule& engine, const char* name, const std::vector<std::int64_t>& shape) {
    return engine.tensor_dtype(name) == DType::kFloat32 && engine.tensor_shape(name) == shape;
}

// The reference processor reads 8-bit images: 8-bit input is read in place, and float32 pixels in
// [0, 1] are rounded to the nearest 8-bit value into `storage`, which recovers the exact bytes of
// an 8-bit image decoded as value / 255.
const std::uint8_t* rgb8(const internal::ImageView& image, std::vector<std::uint8_t>& storage) {
    if (image.data == nullptr || image.channels != 3 || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float))
        throw std::invalid_argument("vitpose requires a non-empty contiguous RGB image");
    const auto count = static_cast<std::size_t>(image.height) * image.width * 3;
    if (image.format == internal::ImageFormat::UInt8) {
        if (image.byte_size != count)
            throw std::invalid_argument("vitpose 8-bit RGB input has the wrong byte size");
        return static_cast<const std::uint8_t*>(image.data);
    }
    if (image.format != internal::ImageFormat::Float32 || image.byte_size != count * sizeof(float))
        throw std::invalid_argument("vitpose requires 8-bit or float32 RGB input");
    const auto* values = static_cast<const float*>(image.data);
    storage.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float value = values[index];
        if (!(value >= 0.0F && value <= 1.0F))
            throw std::invalid_argument("vitpose float32 RGB values must lie in [0, 1]");
        storage[index] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
    }
    return storage.data();
}

} // namespace

RuntimeMetadata parse_runtime_json(const std::vector<char>& data) {
    const auto json = parse_unique_keys(data);
    if (!json.is_object())
        invalid_metadata("must be a JSON object");
    static const std::set<std::string> kKeys = {
        "input_height", "input_width",    "rescale_factor", "image_mean",  "image_std",
        "box_padding",  "heatmap_height", "heatmap_width",  "dark_kernel", "labels"};
    for (const auto& item : json.items()) {
        if (kKeys.count(item.key()) == 0)
            invalid_metadata("has the unexpected key " + item.key());
    }
    RuntimeMetadata metadata;
    metadata.preprocess.height = positive_int32(json, "input_height");
    metadata.preprocess.width = positive_int32(json, "input_width");
    metadata.preprocess.rescale_factor =
        finite_number(field(json, "rescale_factor"), "rescale_factor");
    metadata.preprocess.padding_factor = finite_number(field(json, "box_padding"), "box_padding");
    metadata.preprocess.mean = float_channels(json, "image_mean");
    metadata.preprocess.std = float_channels(json, "image_std");
    metadata.heatmap_height = positive_int32(json, "heatmap_height");
    metadata.heatmap_width = positive_int32(json, "heatmap_width");
    metadata.dark_kernel = positive_int32(json, "dark_kernel");
    metadata.labels = keypoint_labels(json);
    return metadata;
}

PersonBox request_box(internal::ConfigView config, std::uint32_t height, std::uint32_t width) {
    const auto supplied = internal::config_get<Span<const double>>(config, kFields, "person_box");
    if (!supplied)
        return {0.0, 0.0, static_cast<double>(width), static_cast<double>(height)};
    const auto values = *supplied;
    if (values.size() != 4)
        throw internal::ConfigError("vitpose person_box must hold x_min, y_min, x_max and y_max");
    for (const double value : values) {
        if (!std::isfinite(value))
            throw internal::ConfigError("vitpose person_box values must be finite");
    }
    const double x_min = values[0];
    const double y_min = values[1];
    const double x_max = values[2];
    const double y_max = values[3];
    if (x_min < 0.0 || y_min < 0.0 || x_max > static_cast<double>(width) ||
        y_max > static_cast<double>(height))
        throw internal::ConfigError("vitpose person_box must lie inside the image");
    if (!(x_max - x_min >= 1.0) || !(y_max - y_min >= 1.0))
        throw internal::ConfigError("vitpose person_box must be at least one pixel wide and tall");
    return {x_min, y_min, x_max - x_min, y_max - y_min};
}

VitPosePoints::VitPosePoints(std::unique_ptr<ITrtModule> engine, RuntimeMetadata metadata)
    : engine_(std::move(engine)), metadata_(std::move(metadata)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("vitpose engine failed to load");
    const auto& preprocess = metadata_.preprocess;
    const auto keypoints = static_cast<std::int64_t>(metadata_.labels.size());
    if (preprocess.height < 2 || preprocess.width < 2 || metadata_.heatmap_height < 2 ||
        metadata_.heatmap_width < 2 || keypoints <= 0 ||
        keypoints > std::numeric_limits<std::int32_t>::max() ||
        !(preprocess.padding_factor > 0.0) ||
        !(preprocess.padding_factor <= static_cast<double>(std::numeric_limits<float>::max())) ||
        metadata_.dark_kernel < 3 || metadata_.dark_kernel > 127 || metadata_.dark_kernel % 2 == 0)
        throw std::runtime_error("vitpose runtime.json does not match its runtime contract");
    if (!normalization_is_finite(preprocess))
        throw std::runtime_error("vitpose rescale_factor, image_mean and image_std must normalize "
                                 "8-bit values to finite float32 values");
    if (!engine_->has_input(kInput) || !engine_->has_output(kOutput) ||
        !has_shape(*engine_, kInput, {1, 3, preprocess.height, preprocess.width}) ||
        !has_shape(*engine_, kOutput,
                   {1, keypoints, metadata_.heatmap_height, metadata_.heatmap_width}))
        throw std::runtime_error("vitpose engine I/O does not match runtime.json");
}

std::vector<internal::TaskInstance> VitPosePoints::task_bindings() {
    return {internal::bind<internal::IImageTextToPoints>(*this, kFields)};
}

internal::GroundedPointsResult VitPosePoints::run(const internal::ImageTextToPointsRequest& request,
                                                  internal::ConfigView config) {
    internal::validate_config(kFields, config);
    if (!request.text.empty() && request.text != kQuery)
        throw std::invalid_argument("vitpose locates the body keypoints of one person; the query "
                                    "must be \"person\" or empty");
    std::vector<std::uint8_t> converted;
    const auto* bytes = rgb8(request.image, converted);
    const auto box = request_box(config, request.image.height, request.image.width);
    const auto& preprocess = metadata_.preprocess;
    const auto geometry = crop_geometry(box, preprocess);
    const auto crop =
        warp_crop(bytes, static_cast<std::int32_t>(request.image.height),
                  static_cast<std::int32_t>(request.image.width), geometry, preprocess);
    auto pixels = normalize_crop(crop, preprocess);

    Tensor input;
    input.data = pixels.data();
    input.shape = {1, 3, preprocess.height, preprocess.width};
    input.dtype = DType::kFloat32;
    const auto outputs = engine_->forward({{kInput, input}});
    const auto found = outputs.find(kOutput);
    const auto keypoints = static_cast<std::int32_t>(metadata_.labels.size());
    const std::vector<std::int64_t> expected{1, keypoints, metadata_.heatmap_height,
                                             metadata_.heatmap_width};
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != DType::kFloat32 || found->second.shape != expected)
        throw std::runtime_error("vitpose engine returned no complete float32 heatmaps");
    const auto* heatmaps = static_cast<const float*>(found->second.data);
    for (std::size_t index = 0; index < static_cast<std::size_t>(found->second.numel()); ++index) {
        if (!std::isfinite(heatmaps[index]))
            throw std::runtime_error("vitpose engine returned non-finite heatmaps");
    }

    const auto decoded = decode_heatmaps(heatmaps, keypoints, metadata_.heatmap_height,
                                         metadata_.heatmap_width, geometry, metadata_.dark_kernel);
    internal::GroundedPointsResult result;
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        // A heatmap without a positive value locates nothing; the reference marks it -1.
        if (!decoded[index].located)
            continue;
        result.points.push_back(
            {{decoded[index].x, decoded[index].y}, metadata_.labels[index], decoded[index].score});
    }
    result.parse_complete = true;
    return result;
}

} // namespace trtmc::vitpose
