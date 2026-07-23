/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/tool/action_request_json.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace trtmc::openpi::tool {
namespace {

using Json = nlohmann::json;

constexpr std::array<const char*, 3> kCameraNames = {
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
};

Json parse_strict_document(std::string_view text) {
    std::vector<std::unordered_set<std::string>> object_keys;
    Json::parser_callback_t reject_duplicate_keys = [&](int, Json::parse_event_t event,
                                                        Json& value) {
        if (event == Json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == Json::parse_event_t::key) {
            if (object_keys.empty() ||
                !object_keys.back().insert(value.get<std::string>()).second) {
                throw std::invalid_argument("duplicate JSON object key");
            }
        } else if (event == Json::parse_event_t::object_end) {
            if (!object_keys.empty())
                object_keys.pop_back();
        }
        return true;
    };

    try {
        return Json::parse(text.begin(), text.end(), reject_duplicate_keys);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("action request is not valid strict JSON: ") +
                                    error.what());
    }
}

void require_exact_keys(const Json& object, std::initializer_list<const char*> required,
                        std::initializer_list<const char*> optional, const std::string& context) {
    if (!object.is_object())
        throw std::invalid_argument(context + " must be a JSON object");

    std::unordered_set<std::string> accepted;
    for (const char* key : required) {
        accepted.emplace(key);
        if (!object.contains(key))
            throw std::invalid_argument(context + " is missing '" + key + "'");
    }
    for (const char* key : optional)
        accepted.emplace(key);

    for (const auto& [key, unused] : object.items()) {
        (void)unused;
        if (accepted.find(key) == accepted.end())
            throw std::invalid_argument(context + " has unexpected field '" + key + "'");
    }
}

const Json& require_field(const Json& object, const char* key, std::string_view context) {
    const auto iterator = object.find(key);
    if (iterator == object.end())
        throw std::invalid_argument(std::string(context) + " is missing '" + key + "'");
    return *iterator;
}

std::string require_nonempty_string(const Json& object, const char* key,
                                    const std::string& context) {
    const auto& value = require_field(object, key, context);
    if (!value.is_string() || value.get_ref<const std::string&>().empty())
        throw std::invalid_argument(context + " field '" + key + "' must be a non-empty string");
    return value.get<std::string>();
}

bool require_bool(const Json& object, const char* key, const std::string& context) {
    const auto& value = require_field(object, key, context);
    if (!value.is_boolean())
        throw std::invalid_argument(context + " field '" + key + "' must be a boolean");
    return value.get<bool>();
}

int32_t require_nonnegative_int32(const Json& object, const char* key, const std::string& context,
                                  bool strictly_positive) {
    const auto& value = require_field(object, key, context);
    if (!value.is_number_integer() && !value.is_number_unsigned())
        throw std::invalid_argument(context + " field '" + key + "' must be an integer");

    std::uint64_t parsed = 0;
    if (value.is_number_unsigned()) {
        parsed = value.get<std::uint64_t>();
    } else {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < 0)
            throw std::invalid_argument(context + " field '" + key + "' must be non-negative");
        parsed = static_cast<std::uint64_t>(signed_value);
    }
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()) ||
        (strictly_positive && parsed == 0U)) {
        throw std::invalid_argument(
            context + " field '" + key +
            (strictly_positive ? "' must be a positive int32" : "' is outside int32 range"));
    }
    return static_cast<int32_t>(parsed);
}

std::vector<float> require_finite_float_array(const Json& object, const char* key,
                                              const std::string& context) {
    const auto& value = require_field(object, key, context);
    if (!value.is_array() || value.empty())
        throw std::invalid_argument(context + " field '" + key +
                                    "' must be a non-empty numeric array");

    constexpr std::size_t kMaximumElements = 1U << 20U;
    if (value.size() > kMaximumElements)
        throw std::invalid_argument(context + " field '" + key + "' is unreasonably large");

    std::vector<float> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_number())
            throw std::invalid_argument(context + " field '" + key + "' must contain only numbers");
        const double parsed = element.get<double>();
        if (!std::isfinite(parsed) ||
            std::fabs(parsed) > static_cast<double>(std::numeric_limits<float>::max())) {
            throw std::invalid_argument(context + " field '" + key +
                                        "' contains a non-finite or out-of-range value");
        }
        result.push_back(static_cast<float>(parsed));
    }
    return result;
}

void validate_timing(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string("action result timing '") + name +
                                    "' must be finite and non-negative");
}

} // namespace

ActionRequestFile parse_action_request_json(std::string_view text) {
    const Json root = parse_strict_document(text);
    require_exact_keys(root, {"prompt", "cameras", "state"},
                       {"initial_noise", "seed", "denoise_steps"}, "action request");

    ActionRequestFile result;
    result.prompt = require_nonempty_string(root, "prompt", "action request");
    result.state = require_finite_float_array(root, "state", "action request");

    if (root.contains("initial_noise"))
        result.initial_noise = require_finite_float_array(root, "initial_noise", "action request");
    if (root.contains("seed"))
        result.seed = require_nonnegative_int32(root, "seed", "action request", false);
    if (root.contains("denoise_steps"))
        result.denoise_steps =
            require_nonnegative_int32(root, "denoise_steps", "action request", true);

    const auto& cameras = require_field(root, "cameras", "action request");
    require_exact_keys(cameras, {kCameraNames[0], kCameraNames[1], kCameraNames[2]}, {},
                       "action request cameras");
    for (std::size_t index = 0; index < kCameraNames.size(); ++index) {
        const char* name = kCameraNames[index];
        const auto& camera = require_field(cameras, name, "action request cameras");
        const std::string context = std::string("action request camera '") + name + "'";
        require_exact_keys(camera, {"path", "valid"}, {}, context);
        result.cameras[index] = ActionCameraFile{
            name,
            require_nonempty_string(camera, "path", context),
            require_bool(camera, "valid", context),
        };
    }

    return result;
}

ActionRequestFile read_action_request_json(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open action request JSON: " + path);
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (!input.good() && !input.eof())
        throw std::runtime_error("failed to read action request JSON: " + path);
    return parse_action_request_json(text);
}

std::string serialize_action_result_json(const ActionResult& result) {
    if (result.horizon <= 0 || result.action_dim <= 0)
        throw std::invalid_argument("action result horizon and action_dim must be positive");
    const auto horizon = static_cast<std::size_t>(result.horizon);
    const auto action_dim = static_cast<std::size_t>(result.action_dim);
    if (horizon > std::numeric_limits<std::size_t>::max() / action_dim ||
        result.actions.size() != horizon * action_dim) {
        throw std::invalid_argument("action result data size does not match horizon * action_dim");
    }
    if (std::any_of(result.actions.begin(), result.actions.end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("action result contains a non-finite action value");
    }
    validate_timing(result.timings.preprocess_ms, "preprocess_ms");
    validate_timing(result.timings.prefill_ms, "prefill_ms");
    validate_timing(result.timings.denoise_ms, "denoise_ms");
    validate_timing(result.timings.postprocess_ms, "postprocess_ms");

    Json output = {
        {"actions", result.actions},
        {"horizon", result.horizon},
        {"action_dim", result.action_dim},
        {"timings",
         {
             {"preprocess_ms", result.timings.preprocess_ms},
             {"prefill_ms", result.timings.prefill_ms},
             {"denoise_ms", result.timings.denoise_ms},
             {"postprocess_ms", result.timings.postprocess_ms},
         }},
    };
    return output.dump();
}

} // namespace trtmc::openpi::tool
