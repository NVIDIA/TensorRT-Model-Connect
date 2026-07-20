/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/plugin_contract.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr int32_t kPluginContractSchema = 1;
constexpr const char* kPluginFamily = "wan2_2_ti2v";
constexpr const char* kConfigContractKey = "_trtmc_wan22_plugin_contract";

bool has_exact_keys(const nlohmann::json& object, std::initializer_list<const char*> expected) {
    if (!object.is_object() || object.size() != expected.size())
        return false;
    return std::all_of(expected.begin(), expected.end(),
                       [&object](const char* key) { return object.contains(key); });
}

[[noreturn]] void invalid_contract(const std::string& detail) {
    throw std::runtime_error("Invalid Wan2.2 AOT plugin contract: " + detail);
}

const nlohmann::json& require_object_member(const nlohmann::json& object, const char* key) {
    const auto iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_object())
        invalid_contract(std::string("missing object field '") + key + "'");
    return *iterator;
}

std::string require_string(const nlohmann::json& object, const char* key) {
    const auto iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_string())
        invalid_contract(std::string("missing string field '") + key + "'");
    const auto value = iterator->get<std::string>();
    if (value.empty())
        invalid_contract(std::string("empty string field '") + key + "'");
    return value;
}

int32_t require_positive_integer(const nlohmann::json& object, const char* key) {
    const auto iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_number_integer())
        invalid_contract(std::string("missing integer field '") + key + "'");
    const auto value = iterator->get<int32_t>();
    if (value <= 0)
        invalid_contract(std::string("non-positive integer field '") + key + "'");
    return value;
}

int32_t require_nonnegative_integer(const nlohmann::json& object, const char* key) {
    const auto iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_number_integer())
        invalid_contract(std::string("missing integer field '") + key + "'");
    const auto value = iterator->get<int32_t>();
    if (value < 0)
        invalid_contract(std::string("negative integer field '") + key + "'");
    return value;
}

void validate_contract_root(const nlohmann::json& object) {
    if (!has_exact_keys(object, {"schema", "family", "semantic_abi", "source_digest", "creator_set",
                                 "runtime_abi", "cuda_architectures"})) {
        invalid_contract("contract root has unsupported fields");
    }
}

int32_t require_supported_schema(const nlohmann::json& object) {
    const auto schema = require_positive_integer(object, "schema");
    if (schema != kPluginContractSchema)
        invalid_contract("unsupported schema " + std::to_string(schema));
    return schema;
}

std::string require_plugin_family(const nlohmann::json& object) {
    auto family = require_string(object, "family");
    if (family != kPluginFamily)
        invalid_contract("family must be wan2_2_ti2v");
    return family;
}

std::string require_source_digest(const nlohmann::json& object) {
    auto digest = require_string(object, "source_digest");
    if (digest.size() != 64 || !std::all_of(digest.begin(), digest.end(), [](char value) {
            return std::isdigit(static_cast<unsigned char>(value)) != 0 ||
                   (value >= 'a' && value <= 'f');
        })) {
        invalid_contract("source_digest must be a lowercase SHA-256 digest");
    }
    return digest;
}

std::string require_creator_set(const nlohmann::json& object) {
    auto creator_set = require_string(object, "creator_set");
    std::vector<std::string> creators;
    std::size_t begin = 0;
    while (begin <= creator_set.size()) {
        const auto end = creator_set.find(';', begin);
        const auto entry = creator_set.substr(begin, end - begin);
        if (entry.empty() || std::count(entry.begin(), entry.end(), ':') != 2)
            invalid_contract("creator_set must contain canonical name:version:namespace data");
        creators.push_back(entry);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    if (!std::is_sorted(creators.begin(), creators.end()) ||
        std::adjacent_find(creators.begin(), creators.end()) != creators.end()) {
        invalid_contract("creator_set must be sorted and unique");
    }
    return creator_set;
}

PluginRuntimeAbi require_runtime_abi(const nlohmann::json& object) {
    const auto& abi = require_object_member(object, "runtime_abi");
    if (!has_exact_keys(abi, {"tensorrt_major", "tensorrt_minor", "cuda_major", "cudnn_major"})) {
        invalid_contract("runtime_abi has unsupported fields");
    }
    PluginRuntimeAbi result;
    result.tensorrt_major = require_positive_integer(abi, "tensorrt_major");
    result.tensorrt_minor = require_nonnegative_integer(abi, "tensorrt_minor");
    result.cuda_major = require_positive_integer(abi, "cuda_major");
    result.cudnn_major = require_positive_integer(abi, "cudnn_major");
    return result;
}

std::vector<int32_t> parse_cuda_architectures(const nlohmann::json& object) {
    const auto architectures = object.find("cuda_architectures");
    if (architectures == object.end() || !architectures->is_array() || architectures->empty())
        invalid_contract("missing non-empty array field 'cuda_architectures'");

    std::vector<int32_t> result;
    for (const auto& value : *architectures) {
        if (!value.is_number_integer() || value.get<int32_t>() <= 0)
            invalid_contract("cuda_architectures must contain positive integers");
        result.push_back(value.get<int32_t>());
    }
    return result;
}

void validate_cuda_architecture_set(const std::vector<int32_t>& architectures) {
    if (!std::is_sorted(architectures.begin(), architectures.end()) ||
        std::adjacent_find(architectures.begin(), architectures.end()) != architectures.end()) {
        invalid_contract("cuda_architectures must be sorted and unique");
    }
    if (architectures != std::vector<int32_t>{103, 110})
        invalid_contract("cuda_architectures must be exactly [103,110]");
}

PluginContract parse_contract_object(const nlohmann::json& object) {
    validate_contract_root(object);
    PluginContract contract;
    contract.schema = require_supported_schema(object);
    contract.family = require_plugin_family(object);
    contract.semantic_abi = require_string(object, "semantic_abi");
    contract.source_digest = require_source_digest(object);
    contract.creator_set = require_creator_set(object);
    contract.runtime_abi = require_runtime_abi(object);
    contract.cuda_architectures = parse_cuda_architectures(object);
    validate_cuda_architecture_set(contract.cuda_architectures);
    return contract;
}

std::string mismatch(const char* field, const std::string& expected, const std::string& actual) {
    return std::string("Wan2.2 AOT plugin ") + field + " mismatch: bundle='" + expected +
           "', installed='" + actual + "'";
}

void validate_plugin_identity(const PluginContract& expected, const PluginContract& installed) {
    if (expected.schema != installed.schema)
        throw std::runtime_error("Wan2.2 AOT plugin contract schema mismatch");
    if (expected.family != installed.family)
        throw std::runtime_error(mismatch("family", expected.family, installed.family));
    if (expected.semantic_abi != installed.semantic_abi) {
        throw std::runtime_error(
            mismatch("semantic ABI", expected.semantic_abi, installed.semantic_abi));
    }
    if (expected.source_digest != installed.source_digest) {
        throw std::runtime_error(
            mismatch("source digest", expected.source_digest, installed.source_digest));
    }
    if (expected.creator_set != installed.creator_set) {
        throw std::runtime_error(
            mismatch("creator set", expected.creator_set, installed.creator_set));
    }
}

void validate_plugin_runtime(const PluginContract& expected, const PluginContract& installed,
                             const std::string& loaded_runtime_abi) {
    if (!(expected.runtime_abi == installed.runtime_abi)) {
        throw std::runtime_error(mismatch("declared runtime ABI",
                                          canonical_runtime_abi(expected.runtime_abi),
                                          canonical_runtime_abi(installed.runtime_abi)));
    }
    const auto expected_runtime = canonical_runtime_abi(expected.runtime_abi);
    if (loaded_runtime_abi != expected_runtime) {
        throw std::runtime_error(
            mismatch("loaded runtime ABI", expected_runtime, loaded_runtime_abi));
    }
}

void validate_plugin_architectures(const PluginContract& expected, const PluginContract& installed,
                                   int32_t current_sm) {
    if (current_sm <= 0 ||
        std::find(installed.cuda_architectures.begin(), installed.cuda_architectures.end(),
                  current_sm) == installed.cuda_architectures.end()) {
        throw std::runtime_error("Wan2.2 AOT plugin does not contain current GPU sm_" +
                                 std::to_string(current_sm));
    }
    if (expected.cuda_architectures != installed.cuda_architectures) {
        throw std::runtime_error("Wan2.2 AOT plugin CUDA architecture set mismatch");
    }
}

} // namespace

PluginContract parse_bundle_plugin_contract(const std::string& config_json) {
    try {
        const auto config = nlohmann::json::parse(config_json);
        return parse_contract_object(require_object_member(config, kConfigContractKey));
    } catch (const nlohmann::json::exception& error) {
        invalid_contract(error.what());
    }
}

PluginContract parse_companion_plugin_contract(const std::string& manifest_json) {
    try {
        return parse_contract_object(nlohmann::json::parse(manifest_json));
    } catch (const nlohmann::json::exception& error) {
        invalid_contract(error.what());
    }
}

std::string canonical_runtime_abi(const PluginRuntimeAbi& abi) {
    std::ostringstream result;
    result << "tensorrt=" << abi.tensorrt_major << '.' << abi.tensorrt_minor
           << ";cuda=" << abi.cuda_major << ";cudnn=" << abi.cudnn_major;
    return result.str();
}

void validate_plugin_contract(const PluginContract& expected, const PluginContract& installed,
                              const std::string& loaded_runtime_abi, int32_t current_sm) {
    validate_plugin_identity(expected, installed);
    validate_plugin_runtime(expected, installed, loaded_runtime_abi);
    validate_plugin_architectures(expected, installed, current_sm);
}

} // namespace trtmc::wan2_2_ti2v
