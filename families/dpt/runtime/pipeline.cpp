/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dpt/runtime/pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::dpt {
namespace {

constexpr std::array<const char*, 5> kRuntimeKeys = {"input_image_h", "input_image_w",
                                                     "rescale_factor", "image_mean", "image_std"};

[[noreturn]] void contract_error(const std::string& what) {
    throw std::runtime_error("dpt runtime.json " + what);
}

// nlohmann keeps the last of two equal keys; refuse them instead.
nlohmann::json parse_unique_keys(const std::string& text) {
    std::vector<std::unordered_set<std::string>> open_objects;
    const nlohmann::json::parser_callback_t callback =
        [&open_objects](int /*depth*/, nlohmann::json::parse_event_t event,
                        nlohmann::json& parsed) {
            if (event == nlohmann::json::parse_event_t::object_start) {
                open_objects.emplace_back();
            } else if (event == nlohmann::json::parse_event_t::object_end) {
                open_objects.pop_back();
            } else if (event == nlohmann::json::parse_event_t::key) {
                const auto key = parsed.get<std::string>();
                if (!open_objects.back().insert(key).second)
                    contract_error("repeats the key " + key);
            }
            return true;
        };
    try {
        return nlohmann::json::parse(text, callback);
    } catch (const nlohmann::json::exception& error) {
        contract_error(std::string("is not valid JSON: ") + error.what());
    }
}

// A JSON integer (not 384.0) from 1 to INT32_MAX, checked before narrowing.
std::int32_t positive_int32(const nlohmann::json& json, const char* key) {
    const auto& value = json.at(key);
    constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number >= 1U && number <= kMax)
            return static_cast<std::int32_t>(number);
    } else if (value.is_number_integer()) {
        const auto number = value.get<std::int64_t>();
        if (number >= 1 && static_cast<std::uint64_t>(number) <= kMax)
            return static_cast<std::int32_t>(number);
    }
    contract_error(std::string(key) + " must be an integer from 1 to 2147483647");
}

double finite_number(const nlohmann::json& value, const std::string& what) {
    if (!value.is_number())
        contract_error(what + " must be a number");
    const auto number = value.get<double>();
    if (!std::isfinite(number))
        contract_error(what + " must be finite");
    return number;
}

std::array<float, 3> channel_values(const nlohmann::json& json, const char* key) {
    const auto& value = json.at(key);
    if (!value.is_array() || value.size() != 3U)
        contract_error(std::string(key) + " must hold three numbers");
    std::array<float, 3> result{};
    for (std::size_t channel = 0; channel < 3U; ++channel) {
        const double number = finite_number(value[channel], key);
        if (std::fabs(number) > static_cast<double>(std::numeric_limits<float>::max()))
            contract_error(std::string(key) + " must fit float32");
        result[channel] = static_cast<float>(number);
    }
    return result;
}

bool all_finite(const float* values, std::size_t count) {
    return std::all_of(values, values + count, [](float value) { return std::isfinite(value); });
}

const Tensor& require_output(const std::unordered_map<std::string, Tensor>& outputs,
                             const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("dpt engine did not produce " + std::string(name));
    return found->second;
}

} // namespace

PreprocessConfig parse_runtime_config(const std::string& text) {
    const auto json = parse_unique_keys(text);
    if (!json.is_object())
        contract_error("must hold a JSON object");
    for (const auto& item : json.items()) {
        if (std::find(kRuntimeKeys.begin(), kRuntimeKeys.end(), item.key()) == kRuntimeKeys.end())
            contract_error("has the unknown key " + item.key());
    }
    for (const char* key : kRuntimeKeys) {
        if (!json.contains(key))
            contract_error(std::string("is missing ") + key);
    }
    PreprocessConfig config;
    config.input_image_h = positive_int32(json, "input_image_h");
    config.input_image_w = positive_int32(json, "input_image_w");
    config.rescale_factor = finite_number(json.at("rescale_factor"), "rescale_factor");
    config.image_mean = channel_values(json, "image_mean");
    config.image_std = channel_values(json, "image_std");
    try {
        check_preprocess_config(config);
    } catch (const std::invalid_argument& error) {
        contract_error(std::string("does not hold a usable image pipeline: ") + error.what());
    }
    return config;
}

void check_engine_io(const ITrtModule& module, const PreprocessConfig& config) {
    const std::vector<std::int64_t> input_shape{1, 3, config.input_image_h, config.input_image_w};
    const std::vector<std::int64_t> output_shape{1, 1, config.input_image_h, config.input_image_w};
    const auto inputs = module.input_info();
    if (inputs.size() != 1U || inputs[0].name != "pixel_values" ||
        inputs[0].dtype != DType::kFloat32 || inputs[0].shape != input_shape) {
        throw std::runtime_error(
            "dpt engine must take one float32 pixel_values [1, 3, H, W] of runtime.json");
    }
    const auto outputs = module.output_info();
    if (outputs.size() != 1U || outputs[0].name != "predicted_depth" ||
        outputs[0].dtype != DType::kFloat32 || outputs[0].shape != output_shape) {
        throw std::runtime_error(
            "dpt engine must return one float32 predicted_depth [1, 1, H, W] of runtime.json");
    }
}

DptPipeline::DptPipeline(std::unique_ptr<ITrtModule> model, PreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("DptPipeline: invalid model");
    try {
        check_preprocess_config(preprocess_config_);
    } catch (const std::invalid_argument& error) {
        throw std::runtime_error(std::string("DptPipeline: ") + error.what());
    }
    check_engine_io(*model_, preprocess_config_);
}

MonocularDepthResult DptPipeline::estimate_depth(const float* pixels, std::int32_t height,
                                                 std::int32_t width) {
    auto pixel_values = preprocess_image(pixels, height, width, preprocess_config_);

    Tensor image;
    image.data = pixel_values.data();
    image.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    image.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", image}});
    const Tensor& depth = require_output(outputs, "predicted_depth");
    if (depth.data == nullptr || depth.dtype != DType::kFloat32 || depth.shape.size() != 4 ||
        depth.shape[0] != 1 || depth.shape[1] != 1 ||
        depth.shape[2] != preprocess_config_.input_image_h ||
        depth.shape[3] != preprocess_config_.input_image_w) {
        throw std::runtime_error("dpt engine output does not match the engine input size");
    }
    const auto* engine_depth = static_cast<const float*>(depth.data);
    const auto engine_count = static_cast<std::size_t>(preprocess_config_.input_image_h) *
                              static_cast<std::size_t>(preprocess_config_.input_image_w);
    if (!all_finite(engine_depth, engine_count))
        throw std::runtime_error("dpt engine produced a non-finite depth value");

    MonocularDepthResult result;
    result.height = height;
    result.width = width;
    // post_process_depth_estimation: bicubic, align_corners=False, onto the
    // caller's image.
    result.depth = resize_bicubic(engine_depth, preprocess_config_.input_image_h,
                                  preprocess_config_.input_image_w, height, width);
    if (!all_finite(result.depth.data(), result.depth.size()))
        throw std::runtime_error("dpt depth map overflows float32 when resized to the image");
    return result;
}

} // namespace trtmc::dpt
