/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace trtmc::openpi {
namespace {

using Json = nlohmann::json;

Json parse_object(std::string_view text, const char* label) {
    Json document;
    std::vector<std::unordered_set<std::string>> object_keys;
    try {
        const Json::parser_callback_t reject_duplicate_keys =
            [&object_keys](int, Json::parse_event_t event, Json& parsed) {
                if (event == Json::parse_event_t::object_start) {
                    object_keys.emplace_back();
                } else if (event == Json::parse_event_t::key) {
                    if (object_keys.empty() ||
                        !object_keys.back().emplace(parsed.get<std::string>()).second) {
                        throw std::invalid_argument("duplicate JSON object key");
                    }
                } else if (event == Json::parse_event_t::object_end) {
                    if (object_keys.empty()) {
                        throw std::invalid_argument("invalid JSON object nesting");
                    }
                    object_keys.pop_back();
                }
                return true;
            };
        document = Json::parse(text.begin(), text.end(), reject_duplicate_keys);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("OpenPI ") + label +
                                    " is not valid JSON: " + error.what());
    }
    if (!document.is_object()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " must contain a JSON object");
    }
    return document;
}

const Json& require_field(const Json& object, const char* key, const char* label) {
    const auto iterator = object.find(key);
    if (iterator == object.end()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " is missing '" + key + "'");
    }
    return *iterator;
}

std::string require_string(const Json& object, const char* key, const char* label) {
    const auto& value = require_field(object, key, label);
    if (!value.is_string()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' must be a string");
    }
    return value.get<std::string>();
}

int32_t require_int32(const Json& object, const char* key, const char* label) {
    const auto& value = require_field(object, key, label);
    if (!value.is_number_integer()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' must be an integer");
    }
    const auto parsed = value.get<int64_t>();
    if (parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' is outside int32 range");
    }
    return static_cast<int32_t>(parsed);
}

bool require_bool(const Json& object, const char* key, const char* label) {
    const auto& value = require_field(object, key, label);
    if (!value.is_boolean()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' must be a boolean");
    }
    return value.get<bool>();
}

void require_string_value(const Json& object, const char* key, std::string_view expected,
                          const char* label) {
    const auto actual = require_string(object, key, label);
    if (actual != expected) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' mismatch: expected '" + std::string(expected) + "', got '" +
                                    actual + "'");
    }
}

void require_int_value(const Json& object, const char* key, int32_t expected, const char* label) {
    const int32_t actual = require_int32(object, key, label);
    if (actual != expected) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' mismatch: expected " + std::to_string(expected) + ", got " +
                                    std::to_string(actual));
    }
}

bool is_lower_sha256(std::string_view digest) {
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

void require_digest(const std::string& digest, const char* field, bool nonzero = false) {
    if (!is_lower_sha256(digest) ||
        (nonzero &&
         std::all_of(digest.begin(), digest.end(), [](char value) { return value == '0'; }))) {
        throw std::invalid_argument(std::string("OpenPI config.json field '") + field +
                                    "' must be a valid lowercase SHA-256 digest");
    }
}

std::vector<float> require_float_array(const Json& object, const char* key, const char* label) {
    const auto& value = require_field(object, key, label);
    if (!value.is_array() || value.empty()) {
        throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                    "' must be a non-empty array");
    }
    std::vector<float> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_number()) {
            throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                        "' must contain only numbers");
        }
        const double parsed = element.get<double>();
        if (!std::isfinite(parsed) || parsed < -std::numeric_limits<float>::max() ||
            parsed > std::numeric_limits<float>::max()) {
            throw std::invalid_argument(std::string("OpenPI ") + label + " field '" + key +
                                        "' contains a non-finite or out-of-range value");
        }
        result.push_back(static_cast<float>(parsed));
    }
    return result;
}

void require_quantile_dimensions(const QuantileStats& quantiles, const char* field,
                                 int32_t minimum_dimension) {
    const auto dimension = quantiles.q01.size();
    if (dimension != quantiles.q99.size() ||
        dimension < static_cast<std::size_t>(minimum_dimension) ||
        dimension > kModelActionDimension) {
        throw std::invalid_argument(std::string("OpenPI normalization field '") + field +
                                    "' has an invalid quantile dimension");
    }
}

void require_quantile_order(float q01, float q99, std::size_t index, std::size_t external_dimension,
                            const char* field) {
    const bool externally_used = index < external_dimension;
    const bool valid = externally_used ? q99 > q01 : q99 >= q01;
    if (valid) {
        return;
    }
    throw std::invalid_argument(std::string("OpenPI normalization field '") + field +
                                (externally_used ? "' has q99 <= q01 at externally used dimension "
                                                 : "' has q99 < q01 at padded dimension ") +
                                std::to_string(index));
}

QuantileStats parse_quantiles(const Json& norm_stats, const char* field,
                              int32_t minimum_dimension) {
    const auto& entry = require_field(norm_stats, field, "normalization statistics");
    if (!entry.is_object()) {
        throw std::invalid_argument(std::string("OpenPI normalization field '") + field +
                                    "' must be an object");
    }
    QuantileStats result;
    result.q01 = require_float_array(entry, "q01", field);
    result.q99 = require_float_array(entry, "q99", field);
    require_quantile_dimensions(result, field, minimum_dimension);
    for (std::size_t index = 0; index < result.q01.size(); ++index) {
        require_quantile_order(result.q01[index], result.q99[index], index,
                               static_cast<std::size_t>(minimum_dimension), field);
    }
    return result;
}

struct ProfileContract {
    int32_t action_horizon;
    int32_t external_action_dim;
    int32_t external_state_dim;
    bool discrete_state_input;
};

ProfileContract contract_for_profile(std::string_view profile) {
    if (profile == "pi05_droid") {
        return ProfileContract{15, 8, 8, true};
    }
    throw std::invalid_argument("Unsupported OpenPI profile '" + std::string(profile) +
                                "'; expected pi05_droid");
}

OpenPIConfig parse_config_fields(const Json& root) {
    OpenPIConfig config;
    config.profile = require_string(root, "openpi_profile", "config.json");
    config.precision = require_string(root, "precision", "config.json");
    config.parameter_dtype = require_string(root, "openpi_parameter_dtype", "config.json");
    config.tokenizer_sha256 = require_string(root, "openpi_tokenizer_sha256", "config.json");
    config.normalization_sha256 =
        require_string(root, "openpi_normalization_sha256", "config.json");
    config.prefill_engine_sha256 =
        require_string(root, "openpi_prefill_engine_sha256", "config.json");
    config.action_engine_sha256 =
        require_string(root, "openpi_action_engine_sha256", "config.json");
    config.action_horizon = require_int32(root, "openpi_action_horizon", "config.json");
    config.internal_action_dim = require_int32(root, "openpi_internal_action_dim", "config.json");
    config.external_action_dim = require_int32(root, "openpi_external_action_dim", "config.json");
    config.external_state_dim = require_int32(root, "openpi_external_state_dim", "config.json");
    config.prefix_length = require_int32(root, "openpi_prefix_length", "config.json");
    config.max_token_length = require_int32(root, "openpi_max_token_length", "config.json");
    config.num_layers = require_int32(root, "openpi_num_layers", "config.json");
    config.num_heads = require_int32(root, "openpi_num_heads", "config.json");
    config.num_kv_heads = require_int32(root, "openpi_num_kv_heads", "config.json");
    config.head_dim = require_int32(root, "openpi_head_dim", "config.json");
    config.denoise_steps = require_int32(root, "openpi_denoise_steps", "config.json");
    config.batch_size = require_int32(root, "openpi_batch_size", "config.json");
    config.discrete_state_input = require_bool(root, "openpi_discrete_state_input", "config.json");
    return config;
}

void require_config_dimensions(const Json& root, const ProfileContract& expected) {
    require_int_value(root, "openpi_action_horizon", expected.action_horizon, "config.json");
    require_int_value(root, "openpi_internal_action_dim",
                      static_cast<int32_t>(kModelActionDimension), "config.json");
    require_int_value(root, "openpi_external_action_dim", expected.external_action_dim,
                      "config.json");
    require_int_value(root, "openpi_external_state_dim", expected.external_state_dim,
                      "config.json");
    require_int_value(root, "openpi_prefix_length", kPrefixLength, "config.json");
    require_int_value(root, "openpi_max_token_length", kMaximumPromptTokens, "config.json");
    require_int_value(root, "openpi_num_layers", kTransformerLayers, "config.json");
    require_int_value(root, "openpi_num_heads", 8, "config.json");
    require_int_value(root, "openpi_num_kv_heads", kKvHeads, "config.json");
    require_int_value(root, "openpi_head_dim", kHeadDimension, "config.json");
    require_int_value(root, "openpi_denoise_steps", 10, "config.json");
    require_int_value(root, "openpi_batch_size", 1, "config.json");
}

void require_config_scalar_contract(const OpenPIConfig& config, const ProfileContract& expected) {
    if (config.discrete_state_input != expected.discrete_state_input) {
        throw std::invalid_argument("OpenPI config discrete-state setting does not match profile");
    }
    if (config.precision != "bf16" && config.precision != "bfloat16") {
        throw std::invalid_argument("OpenPI config precision must be bf16 or bfloat16");
    }
    if (config.parameter_dtype != "bfloat16") {
        throw std::invalid_argument("OpenPI parameter dtype must be bfloat16");
    }
    require_digest(config.tokenizer_sha256, "openpi_tokenizer_sha256");
    require_digest(config.normalization_sha256, "openpi_normalization_sha256");
    require_digest(config.prefill_engine_sha256, "openpi_prefill_engine_sha256");
    require_digest(config.action_engine_sha256, "openpi_action_engine_sha256");
}

void require_camera_array_shapes(const Json& names, const Json& masks, std::size_t count) {
    if (!names.is_array() || names.size() != count || !masks.is_array() || masks.size() != count) {
        throw std::invalid_argument(
            "OpenPI config must declare exactly three camera names and masks");
    }
}

void parse_camera_entry(OpenPIConfig& config, const Json& names, const Json& masks,
                        std::size_t index) {
    if (!names[index].is_string() || !masks[index].is_boolean()) {
        throw std::invalid_argument("OpenPI config camera names/masks have invalid types");
    }
    config.camera_names[index] = names[index].get<std::string>();
    config.camera_mask[index] = masks[index].get<bool>();
    if (config.camera_names[index] != kCameraNames[index]) {
        throw std::invalid_argument(
            "OpenPI config camera ordering does not match the audited contract");
    }
}

void parse_camera_contract(OpenPIConfig& config, const Json& root) {
    const auto& names = require_field(root, "openpi_camera_names", "config.json");
    const auto& masks = require_field(root, "openpi_camera_mask", "config.json");
    require_camera_array_shapes(names, masks, config.camera_names.size());
    for (std::size_t index = 0; index < config.camera_names.size(); ++index) {
        parse_camera_entry(config, names, masks, index);
    }
    if (config.camera_mask != std::array<bool, 3>{true, true, false}) {
        throw std::invalid_argument("OpenPI config camera mask must be [true, true, false]");
    }
}

} // namespace

OpenPIConfig parse_openpi_config(std::string_view config_json) {
    const Json root = parse_object(config_json, "config.json");
    require_string_value(root, "runtime_strategy", "openpi_vla", "config.json");
    require_string_value(root, "task_strategy", "robot_action_generation", "config.json");
    require_string_value(root, "user_contract", "robot_action_chunk", "config.json");
    require_string_value(root, "model_type", "openpi_pi05_flow", "config.json");
    require_string_value(root, "openpi_upstream_commit", kAuditedUpstreamCommit, "config.json");
    require_string_value(root, "openpi_runtime_contract", "native_cpp_device_resident_flow",
                         "config.json");

    OpenPIConfig config = parse_config_fields(root);
    const auto expected = contract_for_profile(config.profile);
    require_config_dimensions(root, expected);
    require_config_scalar_contract(config, expected);
    parse_camera_contract(config, root);
    return config;
}

OpenPINormalization parse_openpi_normalization(std::string_view normalization_json,
                                               const OpenPIConfig& config) {
    const Json root = parse_object(normalization_json, "normalization statistics");
    const auto& stats = require_field(root, "norm_stats", "normalization statistics");
    if (!stats.is_object()) {
        throw std::invalid_argument("OpenPI normalization statistics require a norm_stats object");
    }
    OpenPINormalization normalization;
    normalization.state = parse_quantiles(stats, "state", config.external_state_dim);
    normalization.actions = parse_quantiles(stats, "actions", config.external_action_dim);
    return normalization;
}

} // namespace trtmc::openpi
