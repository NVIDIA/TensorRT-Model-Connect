/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/backend/trt_version.h"
#include "trtmc/bundle.h"
#include "utils/sha256.h"

#include <algorithm>
#include <cctype>
#include <cuda.h>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <nvrtc.h>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace trtmc {

namespace {

constexpr std::string_view kExecutionAttemptEvidenceSource =
    "runtime_memory_transfer_snapshot_v1.execution_attempt_events";
constexpr std::string_view kEmbeddedCalibrationEvidenceSection =
    "runtime_memory_calibration/evidence.json";
constexpr std::string_view kEmbeddedCalibrationEvidenceSchema =
    "trtmc.native-dynamic-memory-build-calibration-evidence/v2";

std::uint64_t checked_execution_attempt_sum(std::uint64_t total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total) {
        throw std::runtime_error("Qualification execution-attempt ledger aggregate overflowed");
    }
    return total + value;
}

std::uint64_t checked_receipt_sum(std::uint64_t total, std::uint64_t value,
                                  const char* description) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total) {
        throw std::overflow_error(std::string("Qualification receipt ") + description +
                                  " overflowed");
    }
    return total + value;
}

std::uint64_t checked_receipt_product(std::uint64_t lhs, std::uint64_t rhs,
                                      const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string("Qualification receipt ") + description +
                                  " overflowed");
    }
    return lhs * rhs;
}

RuntimeMemoryTransferSnapshotV1 require_execution_attempt_snapshot(const ITrtModule& module,
                                                                   std::size_t module_index) {
    const auto* ledger = dynamic_cast<const IRuntimeMemoryTransferLedgerV1*>(&module);
    if (ledger == nullptr) {
        throw std::runtime_error("Qualification execution-attempt ledger unavailable for module " +
                                 std::to_string(module_index));
    }
    auto snapshot = ledger->runtime_memory_transfer_snapshot();
    if (snapshot.struct_size < sizeof(RuntimeMemoryTransferSnapshotV1) ||
        snapshot.api_version != kRuntimeMemoryBackendApiVersionCurrent) {
        throw std::runtime_error(
            "Qualification execution-attempt ledger has incompatible ABI for module " +
            std::to_string(module_index));
    }
    return snapshot;
}

} // namespace

RuntimeMemoryQualificationAdmissionError::RuntimeMemoryQualificationAdmissionError(
    const std::string& message,
    RuntimeMemoryQualificationExecutionAttemptEvidence execution_attempt_evidence)
    : std::runtime_error(message),
      execution_attempt_evidence_(std::move(execution_attempt_evidence)) {
    if (execution_attempt_evidence_.source != kExecutionAttemptEvidenceSource ||
        !execution_attempt_evidence_.available || execution_attempt_evidence_.module_count == 0 ||
        execution_attempt_evidence_.after < execution_attempt_evidence_.before ||
        execution_attempt_evidence_.delta !=
            execution_attempt_evidence_.after - execution_attempt_evidence_.before) {
        throw std::invalid_argument(
            "Qualification admission error requires complete execution-attempt evidence");
    }
}

RuntimeMemoryQualificationAdmissionError::~RuntimeMemoryQualificationAdmissionError() = default;
IRuntimeMemoryQualificationV1::~IRuntimeMemoryQualificationV1() = default;

RuntimeMemoryQualificationExecutionAttemptBaseline
capture_runtime_memory_qualification_execution_attempts(
    const std::vector<const ITrtModule*>& modules) {
    RuntimeMemoryQualificationExecutionAttemptBaseline baseline;
    baseline.modules.reserve(modules.size());
    baseline.before_by_module.reserve(modules.size());

    std::unordered_set<const ITrtModule*> seen;
    seen.reserve(modules.size());
    for (const auto* module : modules) {
        if (module == nullptr) {
            throw std::runtime_error("Qualification execution-attempt ledger has a null module");
        }
        if (!seen.insert(module).second)
            continue;

        const auto snapshot = require_execution_attempt_snapshot(*module, baseline.modules.size());
        baseline.before =
            checked_execution_attempt_sum(baseline.before, snapshot.execution_attempt_events);
        baseline.modules.push_back(module);
        baseline.before_by_module.push_back(snapshot.execution_attempt_events);
    }
    if (baseline.modules.empty()) {
        throw std::runtime_error(
            "Qualification execution-attempt ledger has no prefill/decode modules");
    }
    return baseline;
}

RuntimeMemoryQualificationExecutionAttemptEvidence
finish_runtime_memory_qualification_execution_attempts(
    const RuntimeMemoryQualificationExecutionAttemptBaseline& baseline) {
    if (baseline.modules.empty() || baseline.modules.size() != baseline.before_by_module.size()) {
        throw std::runtime_error("Qualification execution-attempt ledger baseline is incomplete");
    }

    std::uint64_t after = 0;
    for (std::size_t index = 0; index < baseline.modules.size(); ++index) {
        const auto snapshot = require_execution_attempt_snapshot(*baseline.modules[index], index);
        if (snapshot.execution_attempt_events < baseline.before_by_module[index]) {
            throw std::runtime_error(
                "Qualification execution-attempt ledger regressed for module " +
                std::to_string(index));
        }
        after = checked_execution_attempt_sum(after, snapshot.execution_attempt_events);
    }
    if (after < baseline.before) {
        throw std::runtime_error("Qualification execution-attempt ledger aggregate regressed");
    }

    RuntimeMemoryQualificationExecutionAttemptEvidence evidence;
    evidence.source = std::string(kExecutionAttemptEvidenceSource);
    evidence.available = true;
    evidence.module_count = baseline.modules.size();
    evidence.before = baseline.before;
    evidence.after = after;
    evidence.delta = after - baseline.before;
    return evidence;
}

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
        stats.api_version != kRuntimeMemoryBackendApiVersionCurrent || stats.engine_identity == 0) {
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
        throw std::runtime_error("Core NVRTC load-order anchor could not query its linked runtime");
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
    variant_buckets.erase(std::unique(variant_buckets.begin(), variant_buckets.end()),
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
        (!default_profiles && !matches_developer_chunk_variant(contract, expected))) {
        throw std::runtime_error("runtime_memory contract does not match the exact qualified "
                                 "model/revision/config/target tuple for " +
                                 expected.model_id);
    }
}

void validate_runtime_memory_embedded_calibration_evidence(
    const RuntimeMemoryContract& contract, const BundleFile& bundle) {
    const auto& calibration = contract.module_residency_calibration;
    const auto* evidence_bytes =
        find_section(bundle, std::string(kEmbeddedCalibrationEvidenceSection));
    const bool requires_embedded =
        calibration.evidence_provenance == "embedded_bundle_v1";
    if (evidence_bytes == nullptr) {
        if (requires_embedded) {
            throw std::runtime_error(
                "Embedded module-residency calibration is missing bundle section " +
                std::string(kEmbeddedCalibrationEvidenceSection));
        }
        return;
    }
    if (evidence_bytes->empty()) {
        throw std::runtime_error(
            "Module-residency calibration evidence section is empty");
    }

    internal::Sha256 evidence_digest;
    evidence_digest.update(evidence_bytes->data(), evidence_bytes->size());
    if (evidence_digest.hex_digest() != calibration.evidence_sha256) {
        throw std::runtime_error(
            "Module-residency calibration evidence hash mismatch for " +
            std::string(kEmbeddedCalibrationEvidenceSection));
    }
    if (!requires_embedded)
        return;

    nlohmann::json evidence;
    try {
        evidence = nlohmann::json::parse(evidence_bytes->begin(), evidence_bytes->end());
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Embedded module-residency calibration evidence is invalid JSON: " +
            std::string(error.what()));
    }
    try {
        if (!evidence.is_object() ||
            evidence.value("schema", std::string()) !=
                kEmbeddedCalibrationEvidenceSchema ||
            evidence.value("measurement_kind", std::string()) !=
                calibration.measurement_kind ||
            evidence.value("model_id", std::string()) != contract.qualified_model_id ||
            !evidence.contains("model_context_limit") ||
            !evidence.at("model_context_limit").is_number_integer() ||
            evidence.at("model_context_limit").get<std::int64_t>() !=
                contract.model_context_limit ||
            !evidence.contains("passed") || !evidence.at("passed").is_boolean() ||
            !evidence.at("passed").get<bool>()) {
            throw std::runtime_error(
                "Embedded module-residency calibration evidence does not match its sealed "
                "contract");
        }

        const nlohmann::json expected_limits = contract.active_kv_profile_limits;
        nlohmann::json expected_plans = nlohmann::json::array();
        for (const auto& plan : calibration.plans) {
            expected_plans.push_back({
                {"section_name", plan.section_name},
                {"section_sha256", plan.section_sha256},
                {"role", plan.role},
                {"optimization_profile_count", plan.optimization_profile_count},
            });
        }
        const nlohmann::json expected_provenance = {
            {"qualified_runtime_stack_sha256", calibration.qualified_runtime_stack_sha256},
            {"plan_set_sha256", calibration.plan_set_sha256},
            {"cuda_module_loading_mode", calibration.cuda_module_loading_mode},
            {"plans", expected_plans},
        };
        nlohmann::json expected_reserves = nlohmann::json::array();
        nlohmann::json expected_bootstrap_reserves = nlohmann::json::array();
        for (const auto& reserve : calibration.profile_reserves) {
            expected_reserves.push_back({
                {"covering_profile_limit", reserve.covering_profile_limit},
                {"cumulative_reserve_bytes", reserve.cumulative_reserve_bytes},
            });
            expected_bootstrap_reserves.push_back({
                {"covering_profile_limit", reserve.covering_profile_limit},
                {"cumulative_reserve_bytes", 1},
            });
        }
        if (!evidence.contains("active_kv_profile_limits") ||
            evidence.at("active_kv_profile_limits") != expected_limits ||
            !evidence.contains("contract_provenance") ||
            evidence.at("contract_provenance") != expected_provenance ||
            !evidence.contains("recommended_profile_reserves") ||
            evidence.at("recommended_profile_reserves") != expected_reserves) {
            throw std::runtime_error(
                "Embedded module-residency calibration evidence does not bind the exact "
                "plan/profile/reserve contract");
        }

        const nlohmann::json expected_stack = {
            {"sm", contract.qualified_runtime_stack.sm},
            {"tensorrt", contract.qualified_runtime_stack.tensorrt},
            {"cuda_runtime", contract.qualified_runtime_stack.cuda_runtime},
            {"cudnn_backend", contract.qualified_runtime_stack.cudnn_backend},
            {"cudnn_frontend_revision",
             contract.qualified_runtime_stack.cudnn_frontend_revision},
            {"nvrtc", contract.qualified_runtime_stack.nvrtc},
            {"driver", contract.qualified_runtime_stack.driver},
        };
        const nlohmann::json expected_contract_invariants = {
            {"contract_version", contract.contract_version},
            {"qualified_model_id", contract.qualified_model_id},
            {"qualified_model_revision", contract.qualified_model_revision},
            {"qualified_config_sha256", contract.qualified_config_sha256},
            {"qualified_target", contract.qualified_target},
            {"qualified_runtime_stack", expected_stack},
            {"native_kv_plugin_abi", contract.native_kv_plugin_abi},
            {"model_context_limit", contract.model_context_limit},
            {"prefill_chunk_limit", contract.prefill_chunk_limit},
            {"kv_layout", contract.kv_layout},
            {"kv_dtype", contract.kv_dtype},
            {"kv_bytes_per_token", contract.kv_bytes_per_token},
            {"active_kv_profile_limits", expected_limits},
            {"runtime_owned", contract.runtime_owned},
        };
        const auto bootstrap_found = evidence.find("bootstrap_contract");
        if (bootstrap_found == evidence.end() || !bootstrap_found->is_object()) {
            throw std::runtime_error(
                "Embedded module-residency calibration evidence has no bootstrap contract");
        }
        const auto& bootstrap = *bootstrap_found;
        for (const auto& [name, expected] : expected_contract_invariants.items()) {
            const auto found = bootstrap.find(name);
            if (found == bootstrap.end() || *found != expected) {
                throw std::runtime_error(
                    "Embedded module-residency calibration bootstrap contract does not bind " +
                    name);
            }
        }
        const auto bootstrap_calibration_found =
            bootstrap.find("module_residency_calibration");
        if (bootstrap_calibration_found == bootstrap.end() ||
            !bootstrap_calibration_found->is_object()) {
            throw std::runtime_error(
                "Embedded module-residency calibration evidence has no bootstrap calibration");
        }
        const auto& bootstrap_calibration = *bootstrap_calibration_found;
        const auto bootstrap_provenance =
            bootstrap_calibration.value("evidence_provenance",
                                        std::string("external_manifest_v1"));
        if (bootstrap_calibration.value("schema_version", 0) !=
                calibration.schema_version ||
            bootstrap_calibration.value("measurement_kind", std::string()) !=
                calibration.measurement_kind ||
            bootstrap_calibration.value("cuda_module_loading_mode", std::string()) !=
                calibration.cuda_module_loading_mode ||
            bootstrap_provenance != "external_manifest_v1" ||
            bootstrap_calibration.value("qualified_runtime_stack_sha256",
                                        std::string()) !=
                calibration.qualified_runtime_stack_sha256 ||
            bootstrap_calibration.value("plan_set_sha256", std::string()) !=
                calibration.plan_set_sha256 ||
            !bootstrap_calibration.contains("plans") ||
            bootstrap_calibration.at("plans") != expected_plans ||
            !bootstrap_calibration.contains("profile_reserves") ||
            bootstrap_calibration.at("profile_reserves") !=
                expected_bootstrap_reserves) {
            throw std::runtime_error(
                "Embedded module-residency calibration bootstrap is not the exact "
                "pre-measurement contract");
        }
        const auto bootstrap_evidence_sha256 =
            bootstrap_calibration.value("evidence_sha256", std::string());
        const bool valid_bootstrap_sha =
            bootstrap_evidence_sha256.size() == 64 &&
            std::all_of(bootstrap_evidence_sha256.begin(),
                        bootstrap_evidence_sha256.end(), [](unsigned char character) {
                            return std::isdigit(character) != 0 ||
                                   (character >= static_cast<unsigned char>('a') &&
                                    character <= static_cast<unsigned char>('f'));
                        });
        const nlohmann::json expected_bootstrap_only = {
            {"profile_reserve_bytes", 1},
            {"evidence_sha256", bootstrap_evidence_sha256},
            {"never_published", true},
        };
        if (!valid_bootstrap_sha || !evidence.contains("bootstrap_only") ||
            evidence.at("bootstrap_only") != expected_bootstrap_only) {
            throw std::runtime_error(
                "Embedded module-residency calibration bootstrap receipt is invalid");
        }
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            "Embedded module-residency calibration evidence has an invalid field: " +
            std::string(error.what()));
    }
}

void validate_runtime_memory_module_residency_calibration(
    const RuntimeMemoryContract& contract, const BundleFile& bundle) {
    const auto& calibration = contract.module_residency_calibration;
    if (contract.contract_version != 2 || !calibration.present ||
        calibration.schema_version != 1 || calibration.plans.size() != 2 ||
        calibration.profile_reserves.size() != contract.active_kv_profile_limits.size()) {
        throw std::runtime_error(
            "Qualified runtime-memory bundle has no complete module-residency calibration");
    }
    if (calibration.evidence_provenance != "external_manifest_v1" &&
        calibration.evidence_provenance != "embedded_bundle_v1") {
        throw std::runtime_error(
            "Qualified runtime-memory bundle has unsupported evidence provenance");
    }
    validate_runtime_memory_embedded_calibration_evidence(contract, bundle);

    for (const auto& plan : calibration.plans) {
        const auto* bytes = find_section(bundle, plan.section_name);
        if (bytes == nullptr || bytes->empty()) {
            throw std::runtime_error("Module-residency calibration references missing bundle "
                                     "section " +
                                     plan.section_name);
        }
        internal::Sha256 digest;
        digest.update(bytes->data(), bytes->size());
        if (digest.hex_digest() != plan.section_sha256) {
            throw std::runtime_error("Module-residency calibration plan hash mismatch for " +
                                     plan.section_name);
        }
    }

    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (driver == nullptr) {
        throw std::runtime_error(
            "Unable to load the CUDA driver for module-residency calibration validation");
    }
    const auto close_driver = [](void* handle) {
        if (handle != nullptr)
            dlclose(handle);
    };
    std::unique_ptr<void, decltype(close_driver)> driver_owner(driver, close_driver);
    using CuInit = CUresult(CUDAAPI*)(unsigned int);
    using CuModuleGetLoadingMode = CUresult(CUDAAPI*)(CUmoduleLoadingMode*);
    const auto cu_init = reinterpret_cast<CuInit>(dlsym(driver, "cuInit"));
    const auto get_loading_mode =
        reinterpret_cast<CuModuleGetLoadingMode>(dlsym(driver, "cuModuleGetLoadingMode"));
    if (cu_init == nullptr || get_loading_mode == nullptr || cu_init(0) != CUDA_SUCCESS) {
        throw std::runtime_error(
            "CUDA driver cannot report the effective module-loading mode");
    }
    CUmoduleLoadingMode mode{};
    if (get_loading_mode(&mode) != CUDA_SUCCESS) {
        throw std::runtime_error(
            "CUDA driver failed to report the effective module-loading mode");
    }
    const std::string actual_mode =
        mode == CU_MODULE_LAZY_LOADING
            ? "lazy"
            : mode == CU_MODULE_EAGER_LOADING ? "eager" : "unknown";
    if (actual_mode != calibration.cuda_module_loading_mode) {
        throw std::runtime_error(
            "Module-residency calibration CUDA loading mode mismatch: expected " +
            calibration.cuda_module_loading_mode + ", actual " + actual_mode);
    }
}

RuntimeMemoryRuntimeTarget parse_runtime_memory_runtime_stack_json(const std::string& json_text) {
    if (json_text.empty())
        throw std::runtime_error(
            "Selected TensorRT backend does not expose runtime-stack evidence V1");

    nlohmann::json value;
    try {
        value = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("Selected TensorRT backend returned invalid runtime-stack JSON: " +
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
    actual.cudnn_frontend_revision = required_json_string(value, "cudnn_frontend_revision");
    actual.nvrtc_version = required_json_string(value, "nvrtc");
    actual.driver_version = required_json_string(value, "driver");
    const auto anchored_nvrtc = linked_nvrtc_version();
    if (actual.nvrtc_version != anchored_nvrtc) {
        throw std::runtime_error("Selected TensorRT backend/plugin NVRTC disagrees with the "
                                 "core load-order anchor: core=" +
                                 anchored_nvrtc + ", backend=" + actual.nvrtc_version);
    }
    return actual;
}

void validate_runtime_memory_runtime_stack(const QualifiedRuntimeStack& expected,
                                           const RuntimeMemoryRuntimeTarget& actual) {
    const auto actual_gpu =
        format_compute_capability(actual.compute_capability_major, actual.compute_capability_minor);
    require_exact_stack_field("GPU SM", expected.sm, actual_gpu);
    require_exact_stack_field("TensorRT runtime", expected.tensorrt, actual.trt_runtime_version);
    require_exact_stack_field("CUDA runtime", expected.cuda_runtime, actual.cuda_runtime_version);
    require_exact_stack_field("cuDNN backend", expected.cudnn_backend,
                              actual.cudnn_backend_version);
    require_exact_stack_field("cuDNN Frontend revision", expected.cudnn_frontend_revision,
                              actual.cudnn_frontend_revision);
    require_exact_stack_field("NVRTC", expected.nvrtc, actual.nvrtc_version);
    require_exact_stack_field("NVIDIA driver", expected.driver, actual.driver_version);
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
    if (!receipt.is_object()) {
        throw std::runtime_error(
            "qualification invocation trace requires a schema-v4 memory receipt object");
    }
    const auto required_u64 = [&receipt](const char* name) {
        const auto found = receipt.find(name);
        if (found == receipt.end() ||
            (!found->is_number_unsigned() &&
             (!found->is_number_integer() || found->get<std::int64_t>() < 0))) {
            throw std::runtime_error(
                "qualification invocation trace schema-v4 receipt has no valid " +
                std::string(name));
        }
        try {
            return found->get<std::uint64_t>();
        } catch (const nlohmann::json::exception&) {
            throw std::runtime_error(
                "qualification invocation trace schema-v4 receipt has no valid " +
                std::string(name));
        }
    };
    const auto required_string = [&receipt](const char* name) -> const std::string& {
        const auto found = receipt.find(name);
        if (found == receipt.end() || !found->is_string() ||
            found->get_ref<const std::string&>().empty()) {
            throw std::runtime_error(
                "qualification invocation trace schema-v4 receipt has no valid " +
                std::string(name));
        }
        return found->get_ref<const std::string&>();
    };
    const auto optional_u64 = [&receipt](const char* name) -> std::optional<std::uint64_t> {
        const auto found = receipt.find(name);
        if (found == receipt.end()) {
            throw std::runtime_error(
                "qualification invocation trace schema-v4 receipt has no " +
                std::string(name));
        }
        if (found->is_null())
            return std::nullopt;
        if (!found->is_number_unsigned() &&
            (!found->is_number_integer() || found->get<std::int64_t>() < 0)) {
            throw std::runtime_error(
                "qualification invocation trace schema-v4 receipt has invalid " +
                std::string(name));
        }
        try {
            return found->get<std::uint64_t>();
        } catch (const nlohmann::json::exception&) {
            throw std::runtime_error(
                "qualification invocation trace schema-v4 receipt has invalid " +
                std::string(name));
        }
    };
    const auto is_lower_sha256 = [](const std::string& value) {
        return value.size() == 64 &&
               std::all_of(value.begin(), value.end(), [](unsigned char character) {
                   return std::isdigit(character) != 0 ||
                          (character >= static_cast<unsigned char>('a') &&
                           character <= static_cast<unsigned char>('f'));
               });
    };

    if (required_u64("receipt_schema_version") != 4 ||
        required_u64("contract_version") != 2) {
        throw std::runtime_error(
            "qualification invocation trace requires receipt schema 4 and runtime-memory "
            "contract version 2");
    }
    const auto allocation_id = required_u64("kv_allocation_id");
    const auto reserved_tokens = required_u64("runtime_kv_capacity_tokens");
    const auto bytes_per_token = required_u64("kv_bytes_per_token");
    const auto shared_context_bytes = required_u64("context_device_memory_bytes");
    const auto module_residency_reserve_bytes =
        required_u64("module_residency_reserve_bytes");
    const auto module_residency_profile_limit =
        required_u64("module_residency_reserve_profile_limit");
    const auto& plan_set_sha256 =
        required_string("module_residency_plan_set_sha256");
    const auto& evidence_sha256 =
        required_string("module_residency_evidence_sha256");
    const auto& module_loading_mode =
        required_string("module_residency_cuda_module_loading_mode");
    if (allocation_id == 0 || reserved_tokens == 0 || bytes_per_token == 0)
        throw std::runtime_error(
            "qualification invocation trace requires non-zero allocation ID, R, and B");
    if (module_residency_reserve_bytes == 0 ||
        module_residency_profile_limit < reserved_tokens ||
        !is_lower_sha256(plan_set_sha256) || !is_lower_sha256(evidence_sha256) ||
        (module_loading_mode != "lazy" && module_loading_mode != "eager")) {
        throw std::runtime_error(
            "qualification invocation trace requires complete plan-bound module-residency "
            "provenance");
    }

    const auto kv_budget_bytes = required_u64("kv_budget_bytes");
    const auto kv_reserved_bytes = required_u64("kv_reserved_bytes");
    const auto kv_committed_bytes = required_u64("kv_committed_bytes");
    const auto expected_kv_reserved_bytes =
        checked_receipt_product(reserved_tokens, bytes_per_token, "R*B");
    if (kv_reserved_bytes != expected_kv_reserved_bytes ||
        kv_committed_bytes != kv_reserved_bytes || kv_budget_bytes < kv_reserved_bytes) {
        throw std::runtime_error(
            "qualification invocation trace receipt does not allocate exactly R*B KV bytes");
    }

    const auto safety_reserve_bytes = required_u64("safety_reserve_bytes");
    const auto capacity_decision_free_bytes =
        required_u64("capacity_decision_free_bytes");
    const auto capacity_decision_total_bytes =
        required_u64("capacity_decision_total_bytes");
    const auto capacity_decision_device_used_bytes =
        required_u64("capacity_decision_device_used_bytes");
    const auto capacity_decision_resident_overhead_bytes =
        required_u64("capacity_decision_resident_overhead_bytes");
    const auto final_non_kv_overhead_delta_bytes =
        required_u64("final_non_kv_overhead_delta_bytes");
    if (capacity_decision_free_bytes == 0 || capacity_decision_total_bytes == 0 ||
        capacity_decision_free_bytes > capacity_decision_total_bytes ||
        capacity_decision_device_used_bytes !=
            capacity_decision_total_bytes - capacity_decision_free_bytes) {
        throw std::runtime_error(
            "qualification invocation trace receipt has an incoherent capacity-decision "
            "snapshot");
    }

    auto required_capacity_bytes = checked_receipt_sum(
        safety_reserve_bytes, module_residency_reserve_bytes,
        "capacity-decision safety plus module-residency reserve");
    required_capacity_bytes = checked_receipt_sum(
        required_capacity_bytes, final_non_kv_overhead_delta_bytes,
        "capacity-decision reserve plus final overhead delta");
    required_capacity_bytes = checked_receipt_sum(
        required_capacity_bytes, kv_reserved_bytes,
        "capacity-decision reserve plus KV allocation");
    if (required_capacity_bytes > capacity_decision_free_bytes) {
        throw std::runtime_error(
            "qualification invocation trace receipt violates the capacity-decision memory "
            "invariant");
    }

    const auto settled_free_bytes = required_u64("settled_free_bytes");
    const auto settled_total_bytes = required_u64("settled_total_bytes");
    const auto settled_device_used_bytes = required_u64("settled_device_used_bytes");
    if (settled_free_bytes == 0 || settled_total_bytes != capacity_decision_total_bytes ||
        settled_free_bytes > settled_total_bytes ||
        settled_device_used_bytes != settled_total_bytes - settled_free_bytes) {
        throw std::runtime_error(
            "qualification invocation trace receipt has an incoherent settled snapshot");
    }

    const auto final_free_bytes = required_u64("final_free_bytes");
    const auto final_total_bytes = required_u64("final_total_bytes");
    const auto final_device_used_bytes = required_u64("final_device_used_bytes");
    if (final_free_bytes != capacity_decision_free_bytes ||
        final_total_bytes != capacity_decision_total_bytes ||
        final_device_used_bytes != capacity_decision_device_used_bytes) {
        throw std::runtime_error(
            "qualification invocation trace receipt deprecated final aliases do not match "
            "the capacity-decision snapshot");
    }

    const auto ordinary_device_input_bytes =
        optional_u64("ordinary_device_input_bytes");
    const auto ordinary_device_output_bytes =
        optional_u64("ordinary_device_output_bytes");
    const auto external_device_output_bytes =
        optional_u64("external_device_output_bytes");
    const auto graph_private_device_bytes =
        optional_u64("graph_private_device_bytes");
    if (ordinary_device_input_bytes && ordinary_device_output_bytes &&
        external_device_output_bytes && graph_private_device_bytes) {
        auto final_overhead_bytes = checked_receipt_sum(
            shared_context_bytes, *ordinary_device_input_bytes, "final non-KV overhead");
        final_overhead_bytes = checked_receipt_sum(
            final_overhead_bytes, *ordinary_device_output_bytes, "final non-KV overhead");
        final_overhead_bytes = checked_receipt_sum(
            final_overhead_bytes, *external_device_output_bytes, "final non-KV overhead");
        final_overhead_bytes = checked_receipt_sum(
            final_overhead_bytes, *graph_private_device_bytes, "final non-KV overhead");
        const auto expected_delta =
            final_overhead_bytes > capacity_decision_resident_overhead_bytes
                ? final_overhead_bytes - capacity_decision_resident_overhead_bytes
                : std::uint64_t{0};
        if (final_non_kv_overhead_delta_bytes != expected_delta) {
            throw std::runtime_error(
                "qualification invocation trace receipt does not report the exact positive "
                "final non-KV overhead delta");
        }
    }

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
