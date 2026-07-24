/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"

#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/backend/trt_version.h"
#include "trtmc/bundle.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <nvrtc.h>
#include <sstream>
#include <string_view>

namespace trtmc {

RuntimeMemoryQualificationAdmissionError::~RuntimeMemoryQualificationAdmissionError() = default;
IRuntimeMemoryQualificationV1::~IRuntimeMemoryQualificationV1() = default;

std::string runtime_memory_execution_plan_identity(const std::string& bundle_section_name,
                                                   const ITrtModule& module) {
    if (bundle_section_name.empty())
        throw std::invalid_argument("qualification execution-plan section name is empty");
    const auto* introspection = dynamic_cast<const IRuntimeMemoryEngineIntrospectionV1*>(&module);
    if (introspection == nullptr) {
        throw std::logic_error(
            "qualification execution plan has no runtime-memory engine introspection");
    }
    const auto stats = introspection->runtime_memory_engine_stats();
    if (stats.struct_size < sizeof(RuntimeMemoryEngineStatsV1) ||
        stats.api_version != kRuntimeMemoryBackendApiVersionV1 || stats.engine_identity == 0) {
        throw std::logic_error(
            "qualification execution plan has no valid deserialized-engine identity");
    }
    std::ostringstream identity;
    identity << bundle_section_name << "@engine=0x" << std::hex << std::nouppercase
             << stats.engine_identity;
    return identity.str();
}

namespace {

std::string format_compute_capability(std::int32_t major, std::int32_t minor) {
    if (major < 0 || minor < 0)
        return "unavailable";
    return "sm" + std::to_string(major) + std::to_string(minor);
}

void require_exact_stack_field(const char* name, const std::string& expected,
                               const std::string& actual) {
    if (expected.empty())
        throw std::invalid_argument(std::string("qualified runtime-memory tuple has no expected ") +
                                    name);
    if (actual != expected) {
        throw std::runtime_error("Qualified runtime-memory stack mismatch: expected " +
                                 std::string(name) + " " + expected + ", actual " +
                                 (actual.empty() ? std::string("unavailable") : actual));
    }
}

std::string required_json_string(const nlohmann::json& value, const char* name) {
    const auto found = value.find(name);
    if (found == value.end() || !found->is_string() ||
        found->get_ref<const std::string&>().empty()) {
        throw std::runtime_error("Selected TensorRT backend returned incomplete runtime-stack "
                                 "evidence: missing " +
                                 std::string(name));
    }
    return found->get<std::string>();
}

std::string linked_nvrtc_version() {
    int major = 0;
    int minor = 0;
    const nvrtcResult status = nvrtcVersion(&major, &minor);
    if (status != NVRTC_SUCCESS || major <= 0 || minor < 0) {
        throw std::runtime_error(
            "Core NVRTC load-order anchor could not query its linked runtime");
    }
    return std::to_string(major) + "." + std::to_string(minor);
}

bool developer_chunk_variant_enabled() {
    const char* value = std::getenv("TRTMC_DEVELOPER_CHUNK_VARIANT");
    return value != nullptr && std::string_view(value) == "C/2";
}

bool matches_developer_chunk_variant(const RuntimeMemoryContract& contract,
                                     const RuntimeMemoryQualifiedTuple& expected) {
    if (!developer_chunk_variant_enabled() ||
        (expected.model_id != "Qwen/Qwen3-0.6B" &&
         expected.model_id != "TinyLlama/TinyLlama-1.1B-Chat-v1.0") ||
        expected.prefill_chunk_limit <= 1 || expected.prefill_chunk_limit % 2 != 0) {
        return false;
    }
    const auto variant_chunk = expected.prefill_chunk_limit / 2;
    auto variant_buckets = expected.active_kv_profile_limits;
    variant_buckets.push_back(variant_chunk);
    std::sort(variant_buckets.begin(), variant_buckets.end());
    variant_buckets.erase(
        std::unique(variant_buckets.begin(), variant_buckets.end()),
        variant_buckets.end());
    return contract.prefill_chunk_limit == variant_chunk &&
           contract.active_kv_profile_limits == variant_buckets;
}

} // namespace

void validate_runtime_memory_qualified_tuple(const RuntimeMemoryContract& contract,
                                             const RuntimeMemoryQualifiedTuple& expected) {
    const bool invariant_mismatch =
        !contract.present || contract.qualified_model_id != expected.model_id ||
        contract.qualified_model_revision != expected.revision ||
        contract.qualified_config_sha256 != expected.config_sha256 ||
        contract.qualified_target != expected.target ||
        contract.qualified_runtime_stack.sm != expected.gpu_architecture ||
        contract.qualified_runtime_stack.tensorrt != expected.trt_runtime_version ||
        contract.qualified_runtime_stack.cuda_runtime != expected.cuda_runtime_version ||
        contract.qualified_runtime_stack.cudnn_backend != expected.cudnn_backend_version ||
        contract.qualified_runtime_stack.cudnn_frontend_revision !=
            expected.cudnn_frontend_revision ||
        contract.qualified_runtime_stack.nvrtc != expected.nvrtc_version ||
        contract.qualified_runtime_stack.driver != expected.driver_version ||
        contract.contract_version != expected.contract_version ||
        contract.native_kv_plugin_abi != expected.native_kv_plugin_abi ||
        contract.model_context_limit != expected.model_context_limit ||
        contract.kv_layout != expected.kv_layout || contract.kv_dtype != expected.kv_dtype ||
        !contract.runtime_owned;
    const bool default_profiles =
        contract.prefill_chunk_limit == expected.prefill_chunk_limit &&
        contract.active_kv_profile_limits == expected.active_kv_profile_limits;
    if (invariant_mismatch ||
        (!default_profiles &&
         !matches_developer_chunk_variant(contract, expected))) {
        throw std::runtime_error("runtime_memory contract does not match the exact qualified "
                                 "model/revision/config/target tuple for " +
                                 expected.model_id);
    }
}

RuntimeMemoryRuntimeTarget
parse_runtime_memory_runtime_stack_json(const std::string& json_text) {
    if (json_text.empty())
        throw std::runtime_error(
            "Selected TensorRT backend does not expose runtime-stack evidence V1");

    nlohmann::json value;
    try {
        value = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Selected TensorRT backend returned invalid runtime-stack JSON: " +
            std::string(error.what()));
    }
    if (!value.is_object() || value.size() != 7) {
        throw std::runtime_error(
            "Selected TensorRT backend returned an incompatible runtime-stack schema");
    }

    RuntimeMemoryRuntimeTarget actual;
    const auto sm = required_json_string(value, "sm");
    if (sm.size() < 4 || sm.rfind("sm", 0) != 0 ||
        !std::all_of(sm.begin() + 2, sm.end(),
                     [](unsigned char character) { return std::isdigit(character) != 0; })) {
        throw std::runtime_error(
            "Selected TensorRT backend returned invalid SM runtime evidence: " + sm);
    }
    const int encoded_sm = std::stoi(sm.substr(2));
    actual.compute_capability_major = encoded_sm / 10;
    actual.compute_capability_minor = encoded_sm % 10;
    actual.trt_runtime_version = required_json_string(value, "tensorrt");
    actual.cuda_runtime_version = required_json_string(value, "cuda_runtime");
    actual.cudnn_backend_version = required_json_string(value, "cudnn_backend");
    actual.cudnn_frontend_revision =
        required_json_string(value, "cudnn_frontend_revision");
    actual.nvrtc_version = required_json_string(value, "nvrtc");
    actual.driver_version = required_json_string(value, "driver");
    const auto anchored_nvrtc = linked_nvrtc_version();
    if (actual.nvrtc_version != anchored_nvrtc) {
        throw std::runtime_error(
            "Selected TensorRT backend/plugin NVRTC disagrees with the "
            "core load-order anchor: core=" +
            anchored_nvrtc + ", backend=" + actual.nvrtc_version);
    }
    return actual;
}

void validate_runtime_memory_runtime_stack(
    const QualifiedRuntimeStack& expected,
    const RuntimeMemoryRuntimeTarget& actual) {
    const auto actual_gpu =
        format_compute_capability(actual.compute_capability_major, actual.compute_capability_minor);
    require_exact_stack_field("GPU SM", expected.sm, actual_gpu);
    require_exact_stack_field("TensorRT runtime", expected.tensorrt,
                              actual.trt_runtime_version);
    require_exact_stack_field("CUDA runtime", expected.cuda_runtime,
                              actual.cuda_runtime_version);
    require_exact_stack_field("cuDNN backend", expected.cudnn_backend,
                              actual.cudnn_backend_version);
    require_exact_stack_field("cuDNN Frontend revision",
                              expected.cudnn_frontend_revision,
                              actual.cudnn_frontend_revision);
    require_exact_stack_field("NVRTC", expected.nvrtc, actual.nvrtc_version);
    require_exact_stack_field("NVIDIA driver", expected.driver,
                              actual.driver_version);
}

void validate_runtime_memory_runtime_target(const RuntimeMemoryQualifiedTuple& expected,
                                            const RuntimeMemoryRuntimeTarget& actual) {
    const auto expected_trt = parse_trt_version(expected.trt_runtime_version);
    if (!expected_trt || expected_trt->patch < 0 || expected_trt->build < 0)
        throw std::invalid_argument(
            "qualified runtime-memory tuple has no exact expected TensorRT runtime version");
    QualifiedRuntimeStack stack;
    stack.sm = expected.gpu_architecture;
    stack.tensorrt = expected.trt_runtime_version;
    stack.cuda_runtime = expected.cuda_runtime_version;
    stack.cudnn_backend = expected.cudnn_backend_version;
    stack.cudnn_frontend_revision = expected.cudnn_frontend_revision;
    stack.nvrtc = expected.nvrtc_version;
    stack.driver = expected.driver_version;
    validate_runtime_memory_runtime_stack(stack, actual);
}

void finalize_runtime_memory_invocation_traces(RuntimeMemoryQualificationResultV1& result) {
    if (result.runtime_memory_receipt_json.empty())
        throw std::runtime_error("qualification invocation trace requires a memory receipt");
    const auto receipt = nlohmann::json::parse(result.runtime_memory_receipt_json);
    const auto allocation_id = receipt.at("kv_allocation_id").get<std::uint64_t>();
    const auto reserved_tokens = receipt.at("runtime_kv_capacity_tokens").get<std::uint64_t>();
    const auto bytes_per_token = receipt.at("kv_bytes_per_token").get<std::uint64_t>();
    const auto shared_context_bytes =
        receipt.at("context_device_memory_bytes").get<std::uint64_t>();
    if (allocation_id == 0 || reserved_tokens == 0 || bytes_per_token == 0)
        throw std::runtime_error(
            "qualification invocation trace requires non-zero allocation ID, R, and B");
    if (result.invocations.empty())
        throw std::runtime_error("qualification completed without an invocation trace");

    std::uint64_t stable_base_address = 0;
    for (std::size_t index = 0; index < result.invocations.size(); ++index) {
        auto& trace = result.invocations[index];
        if (trace.invocation_index != index)
            throw std::logic_error("qualification invocation indices are not contiguous");
        if (trace.role.empty() || trace.plan_id.empty() || trace.profile_id < 0 ||
            trace.cuda_graph_status.empty()) {
            throw std::logic_error(
                "qualification invocation is missing role/profile/graph identity");
        }
        if (trace.chunk_end <= trace.chunk_begin)
            throw std::logic_error("qualification invocation has an empty chunk range");
        const auto query_tokens = trace.chunk_end - trace.chunk_begin;
        if (trace.history_tokens > trace.active_tokens ||
            trace.active_tokens - trace.history_tokens != query_tokens ||
            trace.active_tokens > reserved_tokens ||
            (trace.history_tokens == 0
                 ? trace.bound_tokens != 1
                 : trace.bound_tokens < std::max<std::uint64_t>(trace.history_tokens, 2)) ||
            trace.bound_tokens > reserved_tokens) {
            throw std::logic_error("qualification invocation violates H/A/T/R bounds");
        }
        if (query_tokens > std::numeric_limits<std::uint64_t>::max() / bytes_per_token)
            throw std::overflow_error("qualification KV append byte count overflows");
        if (trace.kv_base_address == 0 || trace.kv_base_address % 256U != 0) {
            throw std::logic_error("qualification invocation has an invalid KV base address");
        }
        if (index == 0) {
            stable_base_address = trace.kv_base_address;
        } else if (trace.kv_base_address != stable_base_address) {
            throw std::logic_error("qualification invocation replaced the shared KV base address");
        }
        if (trace.context_device_memory_bytes == 0 ||
            trace.context_device_memory_bytes > shared_context_bytes) {
            throw std::logic_error(
                "qualification invocation has invalid actual-shape context bytes");
        }
        if (trace.kv_allocation_id == 0 || trace.kv_allocation_id != allocation_id ||
            trace.reserved_tokens == 0 || trace.reserved_tokens != reserved_tokens) {
            throw std::logic_error(
                "qualification invocation did not sample the active KV allocation");
        }
        const auto expected_append_bytes = query_tokens * bytes_per_token;
        if (trace.kv_append_bytes != expected_append_bytes || trace.kv_append_events == 0) {
            throw std::logic_error(
                "qualification invocation did not measure the exact current-row commit");
        }
        if (trace.kv_device_to_host_bytes != 0 || trace.full_history_device_to_device_bytes != 0) {
            throw std::logic_error(
                "contiguous runtime qualification observed forbidden KV transfer traffic");
        }
    }
}

std::int32_t
resolve_runtime_memory_post_step_trace_rows(std::int32_t completed_position,
                                            std::int32_t max_length, std::int32_t current_rows,
                                            const std::function<std::int32_t()>& next_rows) {
    if (current_rows <= 0)
        throw std::invalid_argument("current trace rows must be positive");
    if (max_length >= 0 && completed_position >= max_length)
        return current_rows;
    if (!next_rows)
        throw std::invalid_argument("next trace rows callback is missing");
    const auto resolved = next_rows();
    if (resolved <= 0)
        throw std::logic_error("next trace rows must be positive");
    return resolved;
}

} // namespace trtmc
