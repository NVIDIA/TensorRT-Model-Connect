/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"

#include "utils/json_helpers.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace trtmc {

namespace {

using BundleSectionLocation = std::pair<std::size_t, std::size_t>;
using BundleSectionEntry = std::pair<std::string, BundleSectionLocation>;
using BundleSectionTable = std::vector<BundleSectionEntry>;

uint64_t read_u64_le(std::ifstream& in) {
    unsigned char bytes[8];
    in.read(reinterpret_cast<char*>(bytes), 8);
    if (!in) {
        throw std::runtime_error("Failed to read uint64 from bundle file");
    }
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

std::size_t find_matching_object_end(const std::string& json, std::size_t brace_start) {
    int depth = 1;
    std::size_t pos = brace_start + 1;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') {
            ++depth;
        } else if (json[pos] == '}') {
            --depth;
        }
        ++pos;
    }
    return pos;
}

int64_t parse_int64_field(const std::string& inner, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = inner.find(needle);
    if (key_pos == std::string::npos) {
        return 0;
    }

    const auto colon = inner.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return 0;
    }

    const auto start = inner.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) {
        return 0;
    }

    try {
        return std::stoll(inner.substr(start));
    } catch (...) {
        return 0;
    }
}

bool parse_section_entry(const std::string& sections_json, std::size_t& search_pos,
                         BundleSectionEntry& entry) {
    const auto quote_start = sections_json.find('"', search_pos);
    if (quote_start == std::string::npos) {
        return false;
    }
    const auto quote_end = sections_json.find('"', quote_start + 1);
    if (quote_end == std::string::npos) {
        return false;
    }

    const auto inner_brace = sections_json.find('{', quote_end + 1);
    if (inner_brace == std::string::npos) {
        return false;
    }
    const auto inner_brace_end = sections_json.find('}', inner_brace + 1);
    if (inner_brace_end == std::string::npos) {
        return false;
    }

    const std::string section_name =
        sections_json.substr(quote_start + 1, quote_end - quote_start - 1);
    const std::string inner = sections_json.substr(inner_brace, inner_brace_end - inner_brace + 1);
    const int64_t offset_val = parse_int64_field(inner, "offset");
    const int64_t size_val = parse_int64_field(inner, "size");
    entry = {section_name,
             {static_cast<std::size_t>(offset_val), static_cast<std::size_t>(size_val)}};

    search_pos = inner_brace_end + 1;
    return true;
}

void parse_sections_table(const std::string& json, BundleSectionTable& sections_out) {
    const std::string sections_key = "\"sections\"";
    const auto sections_pos = json.find(sections_key);
    if (sections_pos == std::string::npos) {
        return;
    }

    const auto brace_start = json.find('{', sections_pos + sections_key.size());
    if (brace_start == std::string::npos) {
        return;
    }

    const std::size_t brace_end = find_matching_object_end(json, brace_start);
    const std::string sections_json = json.substr(brace_start, brace_end - brace_start);
    std::size_t search_pos = 0;
    while (search_pos < sections_json.size()) {
        BundleSectionEntry entry;
        if (!parse_section_entry(sections_json, search_pos, entry)) {
            break;
        }
        sections_out.push_back(std::move(entry));
    }
}

bool is_lower_hex(const std::string& value, std::size_t expected_size) {
    return value.size() == expected_size &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 ||
                      (character >= static_cast<unsigned char>('a') &&
                       character <= static_cast<unsigned char>('f'));
           });
}

[[noreturn]] void invalid_runtime_memory(const std::string& detail) {
    throw std::runtime_error("Invalid runtime_memory contract: " + detail);
}

nlohmann::json parse_unique_json(const std::string& text, const std::string& context) {
    bool duplicate_key = false;
    std::vector<std::set<std::string>> object_keys;
    const auto callback =
        [&duplicate_key, &object_keys](int, nlohmann::json::parse_event_t event,
                                      nlohmann::json& parsed) {
            if (event == nlohmann::json::parse_event_t::object_start) {
                object_keys.emplace_back();
            } else if (event == nlohmann::json::parse_event_t::key) {
                if (object_keys.empty() ||
                    !object_keys.back().insert(parsed.get<std::string>()).second) {
                    duplicate_key = true;
                }
            } else if (event == nlohmann::json::parse_event_t::object_end &&
                       !object_keys.empty()) {
                object_keys.pop_back();
            }
            return true;
        };

    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(text, callback);
    } catch (const nlohmann::json::exception& error) {
        invalid_runtime_memory(context + " JSON is invalid: " + std::string(error.what()));
    }
    if (duplicate_key)
        invalid_runtime_memory(context + " JSON contains a duplicate object key");
    return parsed;
}

void require_exact_keys(const nlohmann::json& value,
                        const std::set<std::string>& expected_keys,
                        const std::string& context) {
    if (!value.is_object())
        invalid_runtime_memory(context + " must be an object");
    std::set<std::string> actual_keys;
    for (auto item = value.begin(); item != value.end(); ++item)
        actual_keys.insert(item.key());
    if (actual_keys != expected_keys)
        invalid_runtime_memory(context + " has an incompatible schema");
}

std::int32_t require_int32(const nlohmann::json& value, const std::string& context) {
    if (value.is_number_unsigned()) {
        const auto unsigned_value = value.get<std::uint64_t>();
        if (unsigned_value >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
            invalid_runtime_memory(context + " is outside the int32 range");
        }
        return static_cast<std::int32_t>(unsigned_value);
    }
    if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < std::numeric_limits<std::int32_t>::min() ||
            signed_value > std::numeric_limits<std::int32_t>::max()) {
            invalid_runtime_memory(context + " is outside the int32 range");
        }
        return static_cast<std::int32_t>(signed_value);
    }
    invalid_runtime_memory(context + " must be an integer");
}

std::string qualified_runtime_stack_digest(const QualifiedRuntimeStack& stack) {
    internal::Sha256 digest;
    const auto add = [&digest](std::string_view key, const std::string& value) {
        const auto record = std::to_string(key.size()) + ":" + std::string(key) + "=" +
                            std::to_string(value.size()) + ":" + value + "\n";
        digest.update(record);
    };
    add("sm", stack.sm);
    add("tensorrt", stack.tensorrt);
    add("cuda_runtime", stack.cuda_runtime);
    add("cudnn_backend", stack.cudnn_backend);
    add("cudnn_frontend_revision", stack.cudnn_frontend_revision);
    add("nvrtc", stack.nvrtc);
    add("driver", stack.driver);
    return digest.hex_digest();
}

std::string module_residency_plan_set_digest(
    const std::vector<ModuleResidencyPlanCalibration>& plans) {
    internal::Sha256 digest;
    for (const auto& plan : plans) {
        digest.update(plan.section_name);
        const char separator = '\0';
        digest.update(&separator, 1);
        digest.update(plan.section_sha256);
        digest.update(&separator, 1);
        digest.update(plan.role);
        digest.update(&separator, 1);
        digest.update(std::to_string(plan.optimization_profile_count));
        const char newline = '\n';
        digest.update(&newline, 1);
    }
    return digest.hex_digest();
}

void parse_module_residency_calibration(const nlohmann::json& runtime_memory,
                                        RuntimeMemoryContract& contract) {
    const auto found = runtime_memory.find("module_residency_calibration");
    if (found == runtime_memory.end() || !found->is_object()) {
        invalid_runtime_memory("contract version 2 requires module_residency_calibration");
    }
    const auto& value = *found;
    static const std::set<std::string> kExpectedKeys = {
        "schema_version",
        "measurement_kind",
        "cuda_module_loading_mode",
        "qualified_runtime_stack_sha256",
        "plan_set_sha256",
        "evidence_sha256",
        "plans",
        "profile_reserves",
    };
    auto expected_keys = kExpectedKeys;
    if (value.contains("evidence_provenance"))
        expected_keys.insert("evidence_provenance");
    require_exact_keys(value, expected_keys, "module_residency_calibration");

    auto& calibration = contract.module_residency_calibration;
    calibration.present = true;
    try {
        calibration.schema_version =
            require_int32(value.at("schema_version"),
                          "module_residency_calibration.schema_version");
        calibration.measurement_kind = value.at("measurement_kind").get<std::string>();
        calibration.cuda_module_loading_mode =
            value.at("cuda_module_loading_mode").get<std::string>();
        calibration.evidence_provenance =
            value.value("evidence_provenance", std::string("external_manifest_v1"));
        calibration.qualified_runtime_stack_sha256 =
            value.at("qualified_runtime_stack_sha256").get<std::string>();
        calibration.plan_set_sha256 = value.at("plan_set_sha256").get<std::string>();
        calibration.evidence_sha256 = value.at("evidence_sha256").get<std::string>();
    } catch (const nlohmann::json::exception& error) {
        invalid_runtime_memory("module_residency_calibration scalar field is invalid: " +
                               std::string(error.what()));
    }
    if (calibration.schema_version != 1)
        invalid_runtime_memory("unsupported module_residency_calibration schema_version");
    if (calibration.measurement_kind != "nvml_process_cumulative_first_use") {
        invalid_runtime_memory("unsupported module-residency measurement_kind");
    }
    if (calibration.cuda_module_loading_mode != "lazy" &&
        calibration.cuda_module_loading_mode != "eager") {
        invalid_runtime_memory("unsupported CUDA module-loading mode");
    }
    if (calibration.evidence_provenance != "external_manifest_v1" &&
        calibration.evidence_provenance != "embedded_bundle_v1") {
        invalid_runtime_memory("unsupported module-residency evidence_provenance");
    }
    if (!is_lower_hex(calibration.qualified_runtime_stack_sha256, 64) ||
        !is_lower_hex(calibration.plan_set_sha256, 64) ||
        !is_lower_hex(calibration.evidence_sha256, 64)) {
        invalid_runtime_memory("module-residency provenance requires lowercase SHA-256 values");
    }

    const auto& plans = value.at("plans");
    if (!plans.is_array() || plans.size() != 2)
        invalid_runtime_memory("module-residency calibration requires two ordered plans");
    static const std::set<std::string> kExpectedPlanKeys = {
        "section_name",
        "section_sha256",
        "role",
        "optimization_profile_count",
    };
    for (std::size_t index = 0; index < plans.size(); ++index) {
        const auto& plan_value = plans.at(index);
        if (!plan_value.is_object())
            invalid_runtime_memory("module-residency plan entry must be an object");
        require_exact_keys(plan_value, kExpectedPlanKeys, "module-residency plan entry");
        ModuleResidencyPlanCalibration plan;
        try {
            plan.section_name = plan_value.at("section_name").get<std::string>();
            plan.section_sha256 = plan_value.at("section_sha256").get<std::string>();
            plan.role = plan_value.at("role").get<std::string>();
            plan.optimization_profile_count =
                require_int32(plan_value.at("optimization_profile_count"),
                              "module-residency plan optimization_profile_count");
        } catch (const nlohmann::json::exception& error) {
            invalid_runtime_memory("module-residency plan entry is invalid: " +
                                   std::string(error.what()));
        }
        const bool expected_decode =
            index == 0 && plan.section_name == "engine_plan" && plan.role == "decode" &&
            plan.optimization_profile_count ==
                static_cast<int32_t>(contract.active_kv_profile_limits.size());
        const bool expected_prefill =
            index == 1 && plan.section_name == "prefill_engine_plan" && plan.role == "prefill" &&
            plan.optimization_profile_count == 1;
        if ((!expected_decode && !expected_prefill) ||
            !is_lower_hex(plan.section_sha256, 64)) {
            invalid_runtime_memory(
                "module-residency plans do not match the split engine/profile topology");
        }
        calibration.plans.push_back(std::move(plan));
    }
    if (calibration.qualified_runtime_stack_sha256 !=
        qualified_runtime_stack_digest(contract.qualified_runtime_stack)) {
        invalid_runtime_memory(
            "module-residency calibration runtime-stack digest does not match the contract");
    }
    if (calibration.plan_set_sha256 !=
        module_residency_plan_set_digest(calibration.plans)) {
        invalid_runtime_memory(
            "module-residency calibration plan-set digest does not match its plan entries");
    }

    const auto& reserves = value.at("profile_reserves");
    if (!reserves.is_array() || reserves.size() != contract.active_kv_profile_limits.size()) {
        invalid_runtime_memory(
            "module-residency profile reserves must align with active KV profiles");
    }
    static const std::set<std::string> kExpectedReserveKeys = {
        "covering_profile_limit",
        "cumulative_reserve_bytes",
    };
    std::uint64_t previous_reserve = 0;
    for (std::size_t index = 0; index < reserves.size(); ++index) {
        const auto& reserve_value = reserves.at(index);
        if (!reserve_value.is_object())
            invalid_runtime_memory("module-residency reserve entry must be an object");
        require_exact_keys(reserve_value, kExpectedReserveKeys,
                           "module-residency reserve entry");
        ModuleResidencyProfileReserve reserve;
        try {
            reserve.covering_profile_limit =
                require_int32(reserve_value.at("covering_profile_limit"),
                              "module-residency reserve covering_profile_limit");
            const auto& reserve_bytes = reserve_value.at("cumulative_reserve_bytes");
            if (!reserve_bytes.is_number_unsigned() &&
                (!reserve_bytes.is_number_integer() ||
                 reserve_bytes.get<std::int64_t>() <= 0)) {
                invalid_runtime_memory(
                    "module-residency cumulative reserve must be a positive integer");
            }
            reserve.cumulative_reserve_bytes =
                reserve_bytes.get<std::uint64_t>();
        } catch (const nlohmann::json::exception& error) {
            invalid_runtime_memory("module-residency reserve entry is invalid: " +
                                   std::string(error.what()));
        }
        if (reserve.covering_profile_limit != contract.active_kv_profile_limits[index] ||
            reserve.cumulative_reserve_bytes == 0 ||
            reserve.cumulative_reserve_bytes < previous_reserve) {
            invalid_runtime_memory(
                "module-residency reserves must be positive, aligned, and nondecreasing");
        }
        previous_reserve = reserve.cumulative_reserve_bytes;
        calibration.profile_reserves.push_back(reserve);
    }
}

void parse_runtime_memory_contract(const std::string& json, BundleInfo& info) {
    if (json.find("\"runtime_memory\"") == std::string::npos)
        return;

    const auto header = parse_unique_json(json, "bundle header");
    if (!header.is_object())
        invalid_runtime_memory("bundle header must be an object");
    const auto found = header.find("runtime_memory");
    if (found == header.end())
        return;
    const auto& value = *found;
    if (!value.is_object())
        invalid_runtime_memory("runtime_memory must be an object");

    const auto version_value = value.find("contract_version");
    if (version_value == value.end())
        invalid_runtime_memory("contract_version is required");

    auto& contract = info.runtime_memory;
    contract.present = true;
    contract.contract_version = require_int32(*version_value, "contract_version");
    if (contract.contract_version != 1 && contract.contract_version != 2)
        invalid_runtime_memory("unsupported contract_version");

    static const std::set<std::string> kVersionOneKeys = {
        "contract_version",
        "qualified_model_id",
        "qualified_model_revision",
        "qualified_config_sha256",
        "qualified_target",
        "qualified_runtime_stack",
        "native_kv_plugin_abi",
        "model_context_limit",
        "prefill_chunk_limit",
        "kv_layout",
        "kv_dtype",
        "kv_bytes_per_token",
        "active_kv_profile_limits",
        "runtime_owned",
    };
    auto expected_keys = kVersionOneKeys;
    if (contract.contract_version == 2)
        expected_keys.insert("module_residency_calibration");
    require_exact_keys(value, expected_keys, "runtime_memory");

    const auto& runtime_stack_value = value.at("qualified_runtime_stack");
    static const std::set<std::string> kRuntimeStackKeys = {
        "sm",
        "tensorrt",
        "cuda_runtime",
        "cudnn_backend",
        "cudnn_frontend_revision",
        "nvrtc",
        "driver",
    };
    require_exact_keys(runtime_stack_value, kRuntimeStackKeys, "qualified_runtime_stack");

    auto& runtime_stack = contract.qualified_runtime_stack;
    try {
        contract.qualified_model_id = value.at("qualified_model_id").get<std::string>();
        contract.qualified_model_revision =
            value.at("qualified_model_revision").get<std::string>();
        contract.qualified_config_sha256 =
            value.at("qualified_config_sha256").get<std::string>();
        contract.qualified_target = value.at("qualified_target").get<std::string>();
        runtime_stack.sm = runtime_stack_value.at("sm").get<std::string>();
        runtime_stack.tensorrt = runtime_stack_value.at("tensorrt").get<std::string>();
        runtime_stack.cuda_runtime =
            runtime_stack_value.at("cuda_runtime").get<std::string>();
        runtime_stack.cudnn_backend =
            runtime_stack_value.at("cudnn_backend").get<std::string>();
        runtime_stack.cudnn_frontend_revision =
            runtime_stack_value.at("cudnn_frontend_revision").get<std::string>();
        runtime_stack.nvrtc = runtime_stack_value.at("nvrtc").get<std::string>();
        runtime_stack.driver = runtime_stack_value.at("driver").get<std::string>();
        contract.native_kv_plugin_abi =
            require_int32(value.at("native_kv_plugin_abi"), "native_kv_plugin_abi");
        contract.model_context_limit =
            require_int32(value.at("model_context_limit"), "model_context_limit");
        contract.prefill_chunk_limit =
            require_int32(value.at("prefill_chunk_limit"), "prefill_chunk_limit");
        contract.kv_layout = value.at("kv_layout").get<std::string>();
        contract.kv_dtype = value.at("kv_dtype").get<std::string>();
        const auto& bytes_per_token = value.at("kv_bytes_per_token");
        if (!bytes_per_token.is_number_unsigned() &&
            (!bytes_per_token.is_number_integer() ||
             bytes_per_token.get<std::int64_t>() <= 0)) {
            invalid_runtime_memory("kv_bytes_per_token must be a positive integer");
        }
        contract.kv_bytes_per_token = bytes_per_token.get<std::uint64_t>();
        const auto& profile_limits = value.at("active_kv_profile_limits");
        if (!profile_limits.is_array())
            invalid_runtime_memory("active_kv_profile_limits must be an array");
        contract.active_kv_profile_limits.clear();
        contract.active_kv_profile_limits.reserve(profile_limits.size());
        for (const auto& profile_limit : profile_limits) {
            contract.active_kv_profile_limits.push_back(
                require_int32(profile_limit, "active_kv_profile_limits[]"));
        }
        contract.runtime_owned = value.at("runtime_owned").get<bool>();
    } catch (const nlohmann::json::exception& error) {
        invalid_runtime_memory("runtime_memory scalar or array field is invalid: " +
                               std::string(error.what()));
    }

    if (contract.qualified_model_id.empty())
        invalid_runtime_memory("qualified_model_id is required");
    if (!is_lower_hex(contract.qualified_model_revision, 40))
        invalid_runtime_memory("qualified_model_revision must be a lowercase 40-character SHA");
    if (!is_lower_hex(contract.qualified_config_sha256, 64))
        invalid_runtime_memory("qualified_config_sha256 must be a lowercase SHA-256");
    if (contract.qualified_target.empty())
        invalid_runtime_memory("qualified_target is required");
    if (runtime_stack.sm.empty() || runtime_stack.tensorrt.empty() ||
        runtime_stack.cuda_runtime.empty() || runtime_stack.cudnn_backend.empty() ||
        !is_lower_hex(runtime_stack.cudnn_frontend_revision, 40) || runtime_stack.nvrtc.empty() ||
        runtime_stack.driver.empty()) {
        invalid_runtime_memory("qualified_runtime_stack requires non-empty "
                               "SM/TensorRT/CUDA/cuDNN/Frontend/NVRTC/driver fields");
    }
    if (contract.native_kv_plugin_abi <= 0)
        invalid_runtime_memory("native_kv_plugin_abi must be positive");
    if (contract.model_context_limit <= 0)
        invalid_runtime_memory("model_context_limit must be positive");
    if (contract.prefill_chunk_limit <= 0 ||
        contract.prefill_chunk_limit > contract.model_context_limit) {
        invalid_runtime_memory("prefill_chunk_limit must be within model_context_limit");
    }
    if (contract.kv_layout.empty())
        invalid_runtime_memory("kv_layout is required");
    if (contract.kv_dtype != "bfloat16" && contract.kv_dtype != "float16" &&
        contract.kv_dtype != "float32") {
        invalid_runtime_memory("kv_dtype is unsupported");
    }
    if (contract.kv_bytes_per_token == 0)
        invalid_runtime_memory("kv_bytes_per_token must be positive");
    if (contract.active_kv_profile_limits.empty())
        invalid_runtime_memory("active_kv_profile_limits is required");
    if (!std::is_sorted(contract.active_kv_profile_limits.begin(),
                        contract.active_kv_profile_limits.end()) ||
        std::adjacent_find(contract.active_kv_profile_limits.begin(),
                           contract.active_kv_profile_limits.end()) !=
            contract.active_kv_profile_limits.end() ||
        contract.active_kv_profile_limits.front() <= 0) {
        invalid_runtime_memory("active_kv_profile_limits must be positive and strictly increasing");
    }
    if (contract.active_kv_profile_limits.back() != contract.model_context_limit) {
        invalid_runtime_memory("active_kv_profile_limits must end at model_context_limit");
    }
    if (std::find(contract.active_kv_profile_limits.begin(),
                  contract.active_kv_profile_limits.end(),
                  contract.prefill_chunk_limit) == contract.active_kv_profile_limits.end()) {
        invalid_runtime_memory("prefill_chunk_limit must be an active KV profile limit");
    }
    if (!contract.runtime_owned)
        invalid_runtime_memory("runtime_owned must be true");
    if (contract.contract_version == 2) {
        parse_module_residency_calibration(value, contract);
    }
    if (info.model_id != contract.qualified_model_id)
        invalid_runtime_memory("qualified_model_id does not match bundle model_id");
    if (info.max_cache_length != contract.model_context_limit)
        invalid_runtime_memory("model_context_limit does not match max_cache_length");
    if (contract.kv_dtype == "bfloat16" && info.precision != "bf16")
        invalid_runtime_memory("bfloat16 KV contract requires bf16 bundle precision");
}

BundleInfo BundleInfoFromJson(const std::string& json, BundleSectionTable& sections_out) {
    BundleInfo info;
    info.model_id = extract_json_string(json, "model_id", "");
    info.model_type = extract_json_string(json, "model_type", "");
    info.family = extract_json_string(json, "family", "");
    info.precision = extract_json_string(json, "precision", "");
    info.trt_version = extract_json_string(json, "trt_version", "");
    info.trt_abi = extract_json_string(json, "trt_abi", "");
    info.gpu_name = extract_json_string(json, "gpu_name", "");
    info.created_at = extract_json_string(json, "created_at", "");
    info.vocab_size = extract_json_int(json, "vocab_size", 0);
    info.hidden_size = extract_json_int(json, "hidden_size", 0);
    info.num_layers = extract_json_int(json, "num_layers", 0);
    info.num_attention_heads = extract_json_int(json, "num_attention_heads", 1);
    info.num_key_value_heads = extract_json_int(json, "num_key_value_heads", 1);
    info.max_cache_length = extract_json_int(json, "max_cache_length", 32);
    info.runtime_strategy = extract_json_string(json, "runtime_strategy", "");
    const int32_t tokenizer_add_special =
        extract_json_int(json, "tokenizer_add_special_tokens", -1);
    if (tokenizer_add_special >= 0) {
        info.tokenizer_add_special_tokens = (tokenizer_add_special != 0);
        info.tokenizer_add_special_tokens_present = true;
    }
    parse_runtime_memory_contract(json, info);

    // Per-component diffusion batch caps (see design doc Decision C).
    // Absent => leave the default {1, 1, 1} so legacy bundles run unchanged.
    const std::string mbs_text = extract_json_object_text(json, "max_batch_size");
    if (!mbs_text.empty()) {
        info.max_batch_size.dit = extract_json_int(mbs_text, "dit", 1);
        info.max_batch_size.text_encoder = extract_json_int(mbs_text, "text_encoder", 1);
        info.max_batch_size.vae = extract_json_int(mbs_text, "vae", 1);
    }

    sections_out.clear();
    parse_sections_table(json, sections_out);
    info.sections.clear();
    info.sections.reserve(sections_out.size());
    for (const auto& [name, offset_size] : sections_out) {
        const auto& [offset, size] = offset_size;
        info.sections.push_back(BundleSectionInfo{name, static_cast<std::uint64_t>(offset),
                                                  static_cast<std::uint64_t>(size)});
    }

    return info;
}

std::uint64_t read_bundle_data_start(std::ifstream& in, const std::string& path,
                                     std::uint64_t file_size) {
    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0)
        throw std::runtime_error("Invalid bundle magic in: " + path);

    const std::uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024)
        throw std::runtime_error("Bundle header too large: " + path);
    if (header_length > std::numeric_limits<std::uint64_t>::max() - kBundleHeaderOffset)
        throw std::runtime_error("Bundle header offset overflow: " + path);

    const std::uint64_t data_start = kBundleHeaderOffset + header_length;
    if (data_start > file_size)
        throw std::runtime_error("Bundle data offset extends outside file: " + path);
    return data_start;
}

std::uint64_t checked_section_file_offset(const BundleSectionInfo& section,
                                          std::uint64_t data_start, std::uint64_t file_size,
                                          const std::string& path) {
    if (section.offset > file_size - data_start ||
        section.size > file_size - data_start - section.offset) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' extends outside file: " + path);
    }
    if (section.size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        section.size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' is too large to read: " + path);
    }
    return data_start + section.offset;
}

std::ifstream open_bundle_section(const std::string& path, const BundleSectionInfo& section) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("Failed to open bundle file: " + path);

    const auto file_end = in.tellg();
    if (file_end < 0)
        throw std::runtime_error("Failed to determine bundle size: " + path);
    const auto file_size = static_cast<std::uint64_t>(file_end);
    in.seekg(0);

    const std::uint64_t data_start = read_bundle_data_start(in, path, file_size);
    const std::uint64_t file_offset =
        checked_section_file_offset(section, data_start, file_size, path);
    in.seekg(static_cast<std::streamoff>(file_offset));
    if (!in) {
        throw std::runtime_error("Failed to seek to bundle section '" + section.name +
                                 "' in: " + path);
    }
    return in;
}

} // namespace

BundleFile ReadBundleFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open bundle file: " + path);
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0) {
        throw std::runtime_error("Invalid bundle magic in: " + path);
    }

    const uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024) {
        throw std::runtime_error("Bundle header too large: " + path);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!in) {
        throw std::runtime_error("Failed to read bundle header: " + path);
    }

    BundleSectionTable section_table;
    BundleFile bundle;
    bundle.info = BundleInfoFromJson(header_json, section_table);

    in.seekg(0, std::ios::end);
    const auto file_end = in.tellg();
    if (file_end < 0)
        throw std::runtime_error("Failed to determine bundle size: " + path);
    const std::uint64_t file_size = static_cast<std::uint64_t>(file_end);
    const std::uint64_t data_start = kBundleHeaderOffset + header_length;

    for (const auto& [name, offset_size] : section_table) {
        const auto& [offset, size] = offset_size;
        BundleSection section;
        section.name = name;
        const BundleSectionInfo section_info{name, offset, size};
        const std::uint64_t file_offset =
            checked_section_file_offset(section_info, data_start, file_size, path);
        section.data.resize(static_cast<std::size_t>(size));

        in.seekg(static_cast<std::streamoff>(file_offset));
        if (!section.data.empty())
            in.read(section.data.data(), static_cast<std::streamsize>(section.data.size()));
        if (!in) {
            throw std::runtime_error("Failed to read bundle section '" + name + "' from: " + path);
        }

        bundle.sections.push_back(std::move(section));
    }

    return bundle;
}

BundleInfo ReadBundleHeader(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open bundle file: " + path);
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0) {
        throw std::runtime_error("Invalid bundle magic in: " + path);
    }

    const uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024) {
        throw std::runtime_error("Bundle header too large: " + path);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!in) {
        throw std::runtime_error("Failed to read bundle header: " + path);
    }

    BundleSectionTable sections_ignored;
    return BundleInfoFromJson(header_json, sections_ignored);
}

std::vector<char> ReadBundleSection(const std::string& path, const BundleSectionInfo& section) {
    std::ifstream in = open_bundle_section(path, section);
    std::vector<char> data(static_cast<std::size_t>(section.size));
    if (!data.empty()) {
        in.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    if (!in) {
        throw std::runtime_error("Failed to read bundle section '" + section.name +
                                 "' from: " + path);
    }
    return data;
}

void CopyBundleSection(const std::string& path, const BundleSectionInfo& section,
                       std::ostream& output) {
    std::ifstream in = open_bundle_section(path, section);
    std::array<char, 1024 * 1024> buffer{};
    std::uint64_t remaining = section.size;
    while (remaining != 0) {
        const auto chunk_size =
            static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
        in.read(buffer.data(), chunk_size);
        if (in.gcount() != chunk_size) {
            throw std::runtime_error("Failed to read bundle section '" + section.name +
                                     "' from: " + path);
        }
        output.write(buffer.data(), chunk_size);
        if (!output) {
            throw std::runtime_error("Failed to write materialized bundle section '" +
                                     section.name + "'");
        }
        remaining -= static_cast<std::uint64_t>(chunk_size);
    }
}

bool HasBundleMagic(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in) {
        return false;
    }

    return std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) == 0;
}

// Public API implementations from bundle.h

bool IsBundle(const std::string& path) {
    return HasBundleMagic(path);
}

BundleInfo InspectBundle(const std::string& bundle_path) {
    return ReadBundleHeader(bundle_path);
}

} // namespace trtmc
