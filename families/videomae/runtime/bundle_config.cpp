/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/videomae/runtime/bundle_config.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::videomae {
namespace {

using Json = nlohmann::json;

[[noreturn]] void refuse(const std::string& what) {
    throw std::runtime_error("videomae runtime.json " + what);
}

// nlohmann keeps the last of repeated keys; refuse them instead.
Json parse_without_repeated_keys(std::string_view text) {
    std::vector<std::set<std::string>> keys;
    return Json::parse(text.begin(), text.end(),
                       [&keys](int, Json::parse_event_t event, Json& parsed) {
                           if (event == Json::parse_event_t::object_start) {
                               keys.emplace_back();
                           } else if (event == Json::parse_event_t::object_end) {
                               keys.pop_back();
                           } else if (event == Json::parse_event_t::key &&
                                      !keys.back().insert(parsed.get<std::string>()).second) {
                               refuse("repeats the key " + parsed.get<std::string>());
                           }
                           return true;
                       });
}

std::int64_t integer(const Json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() &&
         value.get<std::uint64_t>() >
             static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        refuse(std::string("needs an integer ") + key);
    return value.get<std::int64_t>();
}

std::array<float, 3> channels(const Json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_array() || value.size() != 3)
        refuse(std::string("needs three values for ") + key);
    std::array<float, 3> result{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const auto& item = value[channel];
        if (!item.is_number())
            refuse(std::string("needs numbers for ") + key);
        const double number = item.get<double>();
        if (!std::isfinite(number) || std::fabs(number) > std::numeric_limits<float>::max())
            refuse(std::string("needs finite float32 values for ") + key);
        result[channel] = static_cast<float>(number);
    }
    return result;
}

} // namespace

RuntimeConfig parse_runtime_config(std::string_view text) {
    Json json;
    try {
        json = parse_without_repeated_keys(text);
    } catch (const Json::exception& error) {
        refuse(std::string("is not valid JSON: ") + error.what());
    }
    static const std::set<std::string> kKeys{"num_frames",   "image_size", "patch_size",
                                             "tubelet_size", "image_mean", "image_std"};
    if (!json.is_object())
        refuse("must be an object");
    std::set<std::string> present;
    for (const auto& item : json.items())
        present.insert(item.key());
    if (present != kKeys)
        refuse("must hold exactly num_frames, image_size, patch_size, tubelet_size, image_mean "
               "and image_std");
    RuntimeConfig config;
    config.geometry = {integer(json, "num_frames"), integer(json, "image_size"),
                       integer(json, "patch_size"), integer(json, "tubelet_size")};
    validate_geometry(config.geometry);
    config.normalization = {channels(json, "image_mean"), channels(json, "image_std")};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float mean = config.normalization.mean[channel];
        const float std = config.normalization.std[channel];
        if (!(std > 0.0F))
            refuse("needs positive image_std values");
        // The runtime normalizes in float32: (pixel - mean) / std for pixels in [0, 1].
        for (const float pixel : {0.0F, 1.0F})
            if (!std::isfinite((pixel - mean) / std))
                refuse("has channel statistics that overflow float32 for pixels in [0, 1]");
    }
    return config;
}

void validate_engine(const ITrtModule& engine, const ClipGeometry& geometry) {
    const auto check = [](bool condition, const char* what) {
        if (!condition)
            throw std::runtime_error(std::string("videomae engine ") + what);
    };
    const auto inputs = engine.input_info();
    const auto outputs = engine.output_info();
    check(inputs.size() == 3 && outputs.size() == 1 && engine.has_input("pixel_values") &&
              engine.has_input("visible_index") && engine.has_input("masked_index") &&
              engine.has_output("logits"),
          "must have the inputs pixel_values, visible_index, masked_index and the output logits");
    check(engine.tensor_dtype("pixel_values") == DType::kFloat32 &&
              engine.tensor_shape("pixel_values") ==
                  std::vector<std::int64_t>{1, geometry.frames, ClipGeometry::kChannels,
                                            geometry.size, geometry.size},
          "pixel_values does not match runtime.json");
    // Every split of the tubes into conditioning and reconstructed ones, 1 to tubes - 1
    // of each, must fit the engine's dynamic index inputs.
    const std::vector<std::int64_t> fewest{1}, most{geometry.tubes() - 1};
    for (const char* name : {"visible_index", "masked_index"})
        check(engine.tensor_dtype(name) == DType::kInt32 && engine.input_rank(name) == 1 &&
                  engine.input_is_dynamic(name) && engine.optimization_profile_count() >= 1 &&
                  engine.input_profile_shape(name, 0, ProfileShapeSelector::kMin) == fewest &&
                  engine.input_profile_shape(name, 0, ProfileShapeSelector::kMax) == most,
              "tube index inputs must be dynamic int32 vectors of 1 to tubes - 1 entries");
    // The module sizes outputs for the largest profile, so logits must report tubes - 1
    // reconstructed tubes here: a fixed masked length would only fail at inference.
    const auto logits = engine.tensor_shape("logits");
    check(engine.tensor_dtype("logits") == DType::kFloat32 && logits.size() == 3 &&
              logits[0] == 1 && logits[1] == geometry.tubes() - 1 &&
              logits[2] == geometry.tube_values(),
          "logits must be float32 [1, masked tubes, tube values] for runtime.json");
}

} // namespace trtmc::videomae
