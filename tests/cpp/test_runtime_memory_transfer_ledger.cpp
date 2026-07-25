/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"
#include "trtmc/bundle.h"
#include "utils/sha256.h"

#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class ScopedEnvironment {
  public:
    ScopedEnvironment(std::string name, std::optional<std::string> value) : name_(std::move(name)) {
        if (const char* existing = std::getenv(name_.c_str()))
            previous_ = std::string(existing);
        apply(value);
    }

    ~ScopedEnvironment() { apply(previous_); }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

  private:
    void apply(const std::optional<std::string>& value) const {
        if (value.has_value()) {
            (void)setenv(name_.c_str(), value->c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    std::string name_;
    std::optional<std::string> previous_;
};

trtmc::RuntimeMemoryQualificationResultV1 result_with_transfer(std::uint64_t d2h_bytes,
                                                               std::uint64_t d2d_bytes) {
    trtmc::RuntimeMemoryQualificationResultV1 result;
    result.runtime_memory_receipt_json =
        R"({"receipt_schema_version":4,"contract_version":2,"kv_allocation_id":9,"runtime_kv_capacity_tokens":128,"kv_bytes_per_token":16,"kv_budget_bytes":2048,"kv_reserved_bytes":2048,"kv_committed_bytes":2048,"safety_reserve_bytes":0,"context_device_memory_bytes":4096,"ordinary_device_input_bytes":0,"ordinary_device_output_bytes":0,"external_device_output_bytes":0,"graph_private_device_bytes":0,"module_residency_reserve_bytes":268435456,"module_residency_reserve_profile_limit":128,"module_residency_plan_set_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","module_residency_evidence_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","module_residency_cuda_module_loading_mode":"lazy","capacity_decision_free_bytes":536870912,"capacity_decision_total_bytes":1073741824,"capacity_decision_device_used_bytes":536870912,"capacity_decision_resident_overhead_bytes":4096,"final_non_kv_overhead_delta_bytes":0,"settled_free_bytes":536868864,"settled_total_bytes":1073741824,"settled_device_used_bytes":536872960,"final_free_bytes":536870912,"final_total_bytes":1073741824,"final_device_used_bytes":536870912})";
    trtmc::RuntimeMemoryInvocationTraceV1 trace;
    trace.role = "prefill";
    trace.plan_id = "prefill_engine_plan@engine=0x1234";
    trace.profile_id = 1;
    trace.chunk_end = 1;
    trace.kv_allocation_id = 9;
    trace.active_tokens = 1;
    trace.bound_tokens = 1;
    trace.reserved_tokens = 128;
    trace.kv_base_address = 4096;
    trace.context_device_memory_bytes = 4096;
    trace.cuda_graph_status = "uncaptured";
    trace.kv_append_bytes = 16;
    trace.kv_append_events = 2;
    trace.kv_device_to_host_bytes = d2h_bytes;
    trace.full_history_device_to_device_bytes = d2d_bytes;
    result.invocations.push_back(trace);
    return result;
}

trtmc::RuntimeMemoryContract embedded_evidence_contract() {
    trtmc::RuntimeMemoryContract contract;
    contract.present = true;
    contract.contract_version = 2;
    contract.qualified_model_id = "Qwen/Qwen3-0.6B";
    contract.qualified_model_revision = std::string(40, '1');
    contract.qualified_config_sha256 = std::string(64, '2');
    const std::string runtime_config = R"({"model_type":"qwen3"})";
    trtmc::internal::Sha256 runtime_config_digest;
    runtime_config_digest.update(runtime_config.data(), runtime_config.size());
    contract.runtime_config_sha256 = runtime_config_digest.hex_digest();
    contract.qualified_target = "gb300-trt-11.2";
    contract.qualified_runtime_stack.sm = "sm103";
    contract.qualified_runtime_stack.tensorrt = "11.2.0.113";
    contract.qualified_runtime_stack.cuda_runtime = "13.3";
    contract.qualified_runtime_stack.cudnn_backend = "9.20.0";
    contract.qualified_runtime_stack.cudnn_frontend_revision = std::string(40, '3');
    contract.qualified_runtime_stack.nvrtc = "13.3";
    contract.qualified_runtime_stack.driver = "580.105.08";
    contract.native_kv_plugin_abi = 2;
    contract.model_context_limit = 512;
    contract.prefill_chunk_limit = 256;
    contract.kv_layout = "contiguous_runtime_v1";
    contract.kv_dtype = "bfloat16";
    contract.kv_bytes_per_token = 16;
    contract.active_kv_profile_limits = {128, 256, 512};
    contract.runtime_owned = true;

    auto& calibration = contract.module_residency_calibration;
    calibration.present = true;
    calibration.schema_version = 1;
    calibration.measurement_kind = "nvml_process_cumulative_first_use";
    calibration.cuda_module_loading_mode = "lazy";
    calibration.evidence_provenance = "embedded_bundle_v1";
    calibration.qualified_runtime_stack_sha256 = std::string(64, 'a');
    calibration.plan_set_sha256 = std::string(64, 'b');
    calibration.plans = {
        {"engine_plan", std::string(64, 'c'), "decode", 3},
        {"prefill_engine_plan", std::string(64, 'd'), "prefill", 1},
    };
    calibration.profile_reserves = {
        {128, 1024},
        {256, 2048},
        {512, 4096},
    };
    return contract;
}

nlohmann::json embedded_evidence_document(const trtmc::RuntimeMemoryContract& contract) {
    const auto& calibration = contract.module_residency_calibration;
    nlohmann::json plans = nlohmann::json::array();
    for (const auto& plan : calibration.plans) {
        plans.push_back({
            {"section_name", plan.section_name},
            {"section_sha256", plan.section_sha256},
            {"role", plan.role},
            {"optimization_profile_count", plan.optimization_profile_count},
        });
    }
    nlohmann::json reserves = nlohmann::json::array();
    nlohmann::json bootstrap_reserves = nlohmann::json::array();
    for (const auto& reserve : calibration.profile_reserves) {
        reserves.push_back({
            {"covering_profile_limit", reserve.covering_profile_limit},
            {"cumulative_reserve_bytes", reserve.cumulative_reserve_bytes},
        });
        bootstrap_reserves.push_back({
            {"covering_profile_limit", reserve.covering_profile_limit},
            {"cumulative_reserve_bytes", 1},
        });
    }
    const auto bootstrap_evidence_sha256 = std::string(64, 'e');
    const nlohmann::json stack = {
        {"sm", contract.qualified_runtime_stack.sm},
        {"tensorrt", contract.qualified_runtime_stack.tensorrt},
        {"cuda_runtime", contract.qualified_runtime_stack.cuda_runtime},
        {"cudnn_backend", contract.qualified_runtime_stack.cudnn_backend},
        {"cudnn_frontend_revision",
         contract.qualified_runtime_stack.cudnn_frontend_revision},
        {"nvrtc", contract.qualified_runtime_stack.nvrtc},
        {"driver", contract.qualified_runtime_stack.driver},
    };
    const nlohmann::json bootstrap_calibration = {
        {"schema_version", calibration.schema_version},
        {"measurement_kind", calibration.measurement_kind},
        {"cuda_module_loading_mode", calibration.cuda_module_loading_mode},
        {"evidence_provenance", "external_manifest_v1"},
        {"qualified_runtime_stack_sha256",
         calibration.qualified_runtime_stack_sha256},
        {"plan_set_sha256", calibration.plan_set_sha256},
        {"plans", plans},
        {"profile_reserves", bootstrap_reserves},
        {"evidence_sha256", bootstrap_evidence_sha256},
    };
    const nlohmann::json bootstrap_contract = {
        {"contract_version", contract.contract_version},
        {"qualified_model_id", contract.qualified_model_id},
        {"qualified_model_revision", contract.qualified_model_revision},
        {"qualified_config_sha256", contract.qualified_config_sha256},
        {"runtime_config_sha256", contract.runtime_config_sha256},
        {"qualified_target", contract.qualified_target},
        {"qualified_runtime_stack", stack},
        {"native_kv_plugin_abi", contract.native_kv_plugin_abi},
        {"model_context_limit", contract.model_context_limit},
        {"prefill_chunk_limit", contract.prefill_chunk_limit},
        {"kv_layout", contract.kv_layout},
        {"kv_dtype", contract.kv_dtype},
        {"kv_bytes_per_token", contract.kv_bytes_per_token},
        {"active_kv_profile_limits", contract.active_kv_profile_limits},
        {"runtime_owned", contract.runtime_owned},
        {"module_residency_calibration", bootstrap_calibration},
    };
    return {
        {"schema", "trtmc.native-dynamic-memory-build-calibration-evidence/v2"},
        {"measurement_kind", calibration.measurement_kind},
        {"model_id", contract.qualified_model_id},
        {"model_context_limit", contract.model_context_limit},
        {"active_kv_profile_limits", contract.active_kv_profile_limits},
        {"contract_provenance",
         {
             {"qualified_runtime_stack_sha256",
              calibration.qualified_runtime_stack_sha256},
             {"plan_set_sha256", calibration.plan_set_sha256},
             {"cuda_module_loading_mode", calibration.cuda_module_loading_mode},
             {"plans", plans},
         }},
        {"bootstrap_contract", bootstrap_contract},
        {"bootstrap_only",
         {
             {"profile_reserve_bytes", 1},
             {"evidence_sha256", bootstrap_evidence_sha256},
             {"never_published", true},
         }},
        {"recommended_profile_reserves", reserves},
        {"passed", true},
    };
}

trtmc::BundleFile bundle_with_embedded_evidence(
    trtmc::RuntimeMemoryContract& contract, const nlohmann::json& document) {
    const auto serialized = document.dump();
    std::vector<char> bytes(serialized.begin(), serialized.end());
    trtmc::internal::Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    contract.module_residency_calibration.evidence_sha256 = digest.hex_digest();
    trtmc::BundleFile bundle;
    bundle.sections.push_back(
        {"runtime_memory_calibration/evidence.json", std::move(bytes)});
    const std::string runtime_config = R"({"model_type":"qwen3"})";
    bundle.sections.push_back(
        {"config.json",
         std::vector<char>(runtime_config.begin(), runtime_config.end())});
    return bundle;
}

bool rejects_embedded_evidence(const trtmc::RuntimeMemoryContract& contract,
                               const trtmc::BundleFile& bundle,
                               const std::string& expected_message) {
    try {
        trtmc::validate_runtime_memory_embedded_calibration_evidence(contract, bundle);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(expected_message) != std::string::npos;
    }
    return false;
}

void test_embedded_calibration_evidence_is_bound_before_deserialization() {
    auto external = embedded_evidence_contract();
    external.module_residency_calibration.evidence_provenance =
        "external_manifest_v1";
    trtmc::BundleFile empty_bundle;
    bool external_rejected = false;
    try {
        trtmc::validate_runtime_memory_embedded_calibration_evidence(external,
                                                                    empty_bundle);
    } catch (const std::runtime_error& error) {
        external_rejected =
            std::string(error.what()).find("Product runtime requires embedded") !=
            std::string::npos;
    }
    check(external_rejected,
          "product runtime rejects legacy external calibration provenance");
    bool internal_bootstrap_accepted = true;
    try {
        trtmc::InternalRuntimeMemoryCalibrationBootstrapScope bootstrap_scope;
        trtmc::validate_runtime_memory_embedded_calibration_evidence(external,
                                                                    empty_bundle);
    } catch (...) {
        internal_bootstrap_accepted = false;
    }
    check(internal_bootstrap_accepted,
          "private calibrator scope accepts its ephemeral external bootstrap");

    auto contract = embedded_evidence_contract();
    check(rejects_embedded_evidence(contract, empty_bundle, "missing bundle section"),
          "embedded provenance cannot silently downgrade when evidence is absent");

    auto document = embedded_evidence_document(contract);
    auto bundle = bundle_with_embedded_evidence(contract, document);
    bool valid_accepted = true;
    try {
        trtmc::validate_runtime_memory_embedded_calibration_evidence(contract, bundle);
    } catch (...) {
        valid_accepted = false;
    }
    check(valid_accepted, "exact embedded calibration evidence is accepted");

    auto tampered_bundle = bundle;
    tampered_bundle.sections.front().data.back() ^= 1;
    check(rejects_embedded_evidence(contract, tampered_bundle, "hash mismatch"),
          "embedded evidence byte tamper is rejected");

    auto reserve_tamper = contract;
    reserve_tamper.module_residency_calibration.profile_reserves.back()
        .cumulative_reserve_bytes += 1;
    check(rejects_embedded_evidence(reserve_tamper, bundle, "reserve contract"),
          "embedded evidence binds the final cumulative reserve table");

    auto contract_tamper = contract;
    contract_tamper.kv_dtype = "float16";
    check(rejects_embedded_evidence(contract_tamper, bundle, "kv_dtype"),
          "embedded evidence binds outer runtime-memory contract invariants");

    auto wrong_schema = document;
    wrong_schema["schema"] = 2;
    auto wrong_schema_contract = embedded_evidence_contract();
    auto wrong_schema_bundle =
        bundle_with_embedded_evidence(wrong_schema_contract, wrong_schema);
    check(rejects_embedded_evidence(wrong_schema_contract, wrong_schema_bundle,
                                    "invalid field"),
          "wrong-typed embedded evidence schema fails closed as runtime_error");
}

void test_runtime_config_bytes_are_bound_before_deserialization() {
    auto contract = embedded_evidence_contract();
    auto document = embedded_evidence_document(contract);
    auto bundle = bundle_with_embedded_evidence(contract, document);

    bool valid_accepted = true;
    try {
        trtmc::validate_runtime_memory_runtime_config(contract, bundle);
    } catch (...) {
        valid_accepted = false;
    }
    check(valid_accepted, "exact embedded config.json bytes are accepted");

    auto tampered_bundle = bundle;
    tampered_bundle.sections.back().data.back() ^= 1;
    bool tamper_rejected = false;
    try {
        trtmc::validate_runtime_memory_runtime_config(contract, tampered_bundle);
    } catch (const std::runtime_error& error) {
        tamper_rejected =
            std::string(error.what()).find("config.json hash mismatch") !=
            std::string::npos;
    }
    check(tamper_rejected, "runtime config.json byte tamper is rejected");

    auto missing_bundle = bundle;
    missing_bundle.sections.pop_back();
    bool missing_rejected = false;
    try {
        trtmc::validate_runtime_memory_runtime_config(contract, missing_bundle);
    } catch (const std::runtime_error& error) {
        missing_rejected =
            std::string(error.what()).find("missing config.json") != std::string::npos;
    }
    check(missing_rejected, "qualified runtime-memory bundle requires config.json");
}

void test_exact_qualified_tuple_rejects_tampered_bundle_identity() {
    trtmc::RuntimeMemoryQualifiedTuple expected;
    expected.model_id = "Qwen/Qwen3-0.6B";
    expected.revision = "qualified-revision";
    expected.config_sha256 = "qualified-config";
    expected.target = "gb300-trt-11.2";
    expected.gpu_architecture = "sm103";
    expected.trt_runtime_version = "11.2.0.113";
    expected.cuda_runtime_version = "13.3";
    expected.cudnn_backend_version = "9.20.0";
    expected.cudnn_frontend_revision = "7b9b711c22b6823e87150213ecd8449260db8610";
    expected.nvrtc_version = "13.3";
    expected.driver_version = "580.105.08";
    expected.model_context_limit = 40960;
    expected.prefill_chunk_limit = 2048;
    expected.active_kv_profile_limits = {128, 512, 2048, 40960};

    trtmc::RuntimeMemoryContract contract;
    contract.present = true;
    contract.qualified_model_id = expected.model_id;
    contract.qualified_model_revision = expected.revision;
    contract.qualified_config_sha256 = expected.config_sha256;
    contract.qualified_target = expected.target;
    contract.qualified_runtime_stack.sm = expected.gpu_architecture;
    contract.qualified_runtime_stack.tensorrt = expected.trt_runtime_version;
    contract.qualified_runtime_stack.cuda_runtime = expected.cuda_runtime_version;
    contract.qualified_runtime_stack.cudnn_backend = expected.cudnn_backend_version;
    contract.qualified_runtime_stack.cudnn_frontend_revision = expected.cudnn_frontend_revision;
    contract.qualified_runtime_stack.nvrtc = expected.nvrtc_version;
    contract.qualified_runtime_stack.driver = expected.driver_version;
    contract.contract_version = expected.contract_version;
    contract.native_kv_plugin_abi = expected.native_kv_plugin_abi;
    contract.model_context_limit = expected.model_context_limit;
    contract.prefill_chunk_limit = expected.prefill_chunk_limit;
    contract.kv_layout = expected.kv_layout;
    contract.kv_dtype = expected.kv_dtype;
    contract.active_kv_profile_limits = expected.active_kv_profile_limits;
    contract.runtime_owned = true;

    trtmc::validate_runtime_memory_qualified_tuple(contract, expected);
    using Mutation = void (*)(trtmc::RuntimeMemoryContract&);
    for (const Mutation mutation : {
             +[](trtmc::RuntimeMemoryContract& value) {
                 value.qualified_model_revision = "tampered";
             },
             +[](trtmc::RuntimeMemoryContract& value) {
                 value.qualified_target = "unqualified-target";
             },
             +[](trtmc::RuntimeMemoryContract& value) {
                 value.active_kv_profile_limits.back() = 32768;
             },
             +[](trtmc::RuntimeMemoryContract& value) {
                 value.qualified_runtime_stack.nvrtc = "13.0";
             },
         }) {
        auto tampered = contract;
        mutation(tampered);
        bool rejected = false;
        try {
            trtmc::validate_runtime_memory_qualified_tuple(tampered, expected);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("exact qualified") != std::string::npos;
        }
        check(rejected, "runtime rejects a self-consistent but unqualified tuple");
    }
}

void test_developer_c_div_2_tuple_requires_exact_opt_in_and_buckets() {
    trtmc::RuntimeMemoryQualifiedTuple expected;
    expected.model_id = "Qwen/Qwen3-0.6B";
    expected.revision = "qualified-revision";
    expected.config_sha256 = "qualified-config";
    expected.target = "gb300-trt-11.2";
    expected.gpu_architecture = "sm103";
    expected.trt_runtime_version = "11.2.0.113";
    expected.cuda_runtime_version = "13.3";
    expected.cudnn_backend_version = "9.20.0";
    expected.cudnn_frontend_revision = "7b9b711c22b6823e87150213ecd8449260db8610";
    expected.nvrtc_version = "13.3";
    expected.driver_version = "580.105.08";
    expected.model_context_limit = 40960;
    expected.prefill_chunk_limit = 2048;
    expected.active_kv_profile_limits = {128, 512, 2048, 40960};

    trtmc::RuntimeMemoryContract contract;
    contract.present = true;
    contract.qualified_model_id = expected.model_id;
    contract.qualified_model_revision = expected.revision;
    contract.qualified_config_sha256 = expected.config_sha256;
    contract.qualified_target = expected.target;
    contract.qualified_runtime_stack.sm = expected.gpu_architecture;
    contract.qualified_runtime_stack.tensorrt = expected.trt_runtime_version;
    contract.qualified_runtime_stack.cuda_runtime = expected.cuda_runtime_version;
    contract.qualified_runtime_stack.cudnn_backend = expected.cudnn_backend_version;
    contract.qualified_runtime_stack.cudnn_frontend_revision = expected.cudnn_frontend_revision;
    contract.qualified_runtime_stack.nvrtc = expected.nvrtc_version;
    contract.qualified_runtime_stack.driver = expected.driver_version;
    contract.contract_version = expected.contract_version;
    contract.native_kv_plugin_abi = expected.native_kv_plugin_abi;
    contract.model_context_limit = expected.model_context_limit;
    contract.prefill_chunk_limit = 1024;
    contract.kv_layout = expected.kv_layout;
    contract.kv_dtype = expected.kv_dtype;
    contract.active_kv_profile_limits = {128, 512, 1024, 2048, 40960};
    contract.runtime_owned = true;

    const auto rejected = [&](const trtmc::RuntimeMemoryContract& value) {
        try {
            trtmc::validate_runtime_memory_qualified_tuple(value, expected);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    {
        ScopedEnvironment opt_in("TRTMC_DEVELOPER_CHUNK_VARIANT", std::nullopt);
        check(rejected(contract), "C/2 tuple is rejected without developer opt-in");
    }
    {
        ScopedEnvironment opt_in("TRTMC_DEVELOPER_CHUNK_VARIANT", "true");
        check(rejected(contract), "C/2 tuple rejects a non-canonical opt-in value");
    }
    {
        ScopedEnvironment opt_in("TRTMC_DEVELOPER_CHUNK_VARIANT", "C/2");
        trtmc::validate_runtime_memory_qualified_tuple(contract, expected);

        auto existing_bucket_expected = expected;
        existing_bucket_expected.prefill_chunk_limit = 1024;
        existing_bucket_expected.active_kv_profile_limits = {128, 256, 512, 1024, 2048, 40960};
        auto existing_bucket_variant = contract;
        existing_bucket_variant.prefill_chunk_limit = 512;
        existing_bucket_variant.active_kv_profile_limits =
            existing_bucket_expected.active_kv_profile_limits;
        trtmc::validate_runtime_memory_qualified_tuple(existing_bucket_variant,
                                                       existing_bucket_expected);

        auto noncanonical = contract;
        noncanonical.active_kv_profile_limits = {128, 512, 1024, 40960};
        check(rejected(noncanonical), "C/2 tuple rejects non-canonical profile buckets");

        auto unsupported_expected = expected;
        unsupported_expected.model_id = "other/model";
        auto unsupported_contract = contract;
        unsupported_contract.qualified_model_id = unsupported_expected.model_id;
        bool unsupported_rejected = false;
        try {
            trtmc::validate_runtime_memory_qualified_tuple(unsupported_contract,
                                                           unsupported_expected);
        } catch (const std::runtime_error&) {
            unsupported_rejected = true;
        }
        check(unsupported_rejected, "C/2 tuple remains limited to the two qualified models");
    }
}

void test_exact_runtime_target_rejects_gpu_and_trt_drift() {
    trtmc::RuntimeMemoryQualifiedTuple expected;
    expected.gpu_architecture = "sm103";
    expected.trt_runtime_version = "11.2.0.113";
    expected.cuda_runtime_version = "13.3";
    expected.cudnn_backend_version = "9.20.0";
    expected.cudnn_frontend_revision = "7b9b711c22b6823e87150213ecd8449260db8610";
    expected.nvrtc_version = "13.3";
    expected.driver_version = "580.105.08";

    trtmc::RuntimeMemoryRuntimeTarget actual;
    actual.cuda_device = 2;
    actual.compute_capability_major = 10;
    actual.compute_capability_minor = 3;
    actual.trt_runtime_version = "11.2.0.113";
    actual.cuda_runtime_version = "13.3";
    actual.cudnn_backend_version = "9.20.0";
    actual.cudnn_frontend_revision = "7b9b711c22b6823e87150213ecd8449260db8610";
    actual.nvrtc_version = "13.3";
    actual.driver_version = "580.105.08";
    trtmc::validate_runtime_memory_runtime_target(expected, actual);

    {
        auto mismatched = actual;
        mismatched.compute_capability_minor = 0;
        bool rejected = false;
        try {
            trtmc::validate_runtime_memory_runtime_target(expected, mismatched);
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            rejected = message.find("expected GPU SM sm103") != std::string::npos &&
                       message.find("actual sm100") != std::string::npos;
        }
        check(rejected, "runtime target guard reports expected and actual GPU architecture");
    }

    for (const std::string& actual_version : {"11.2.1.113", "11.2.0.114", "11.2"}) {
        auto mismatched = actual;
        mismatched.trt_runtime_version = actual_version;
        bool rejected = false;
        try {
            trtmc::validate_runtime_memory_runtime_target(expected, mismatched);
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            rejected = message.find("expected TensorRT runtime 11.2.0.113") != std::string::npos &&
                       message.find("actual " + actual_version) != std::string::npos;
        }
        check(rejected, "runtime target guard rejects TensorRT patch/build drift");
    }

    {
        auto unavailable = actual;
        unavailable.trt_runtime_version.clear();
        bool rejected = false;
        try {
            trtmc::validate_runtime_memory_runtime_target(expected, unavailable);
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            rejected = message.find("expected TensorRT runtime 11.2.0.113") != std::string::npos &&
                       message.find("actual unavailable") != std::string::npos;
        }
        check(rejected, "runtime target guard fails closed when TensorRT version is unavailable");
    }

    struct StackMutation {
        const char* expected_name;
        void (*mutate)(trtmc::RuntimeMemoryRuntimeTarget&);
    };
    for (const auto& mutation : {
             StackMutation{"CUDA runtime",
                           +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.cuda_runtime_version = "13.1";
                           }},
             StackMutation{"cuDNN backend",
                           +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.cudnn_backend_version = "9.20.1";
                           }},
             StackMutation{"cuDNN Frontend revision",
                           +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.cudnn_frontend_revision = "unqualified";
                           }},
             StackMutation{
                 "NVRTC",
                 +[](trtmc::RuntimeMemoryRuntimeTarget& value) { value.nvrtc_version = "13.0"; }},
             StackMutation{"NVIDIA driver",
                           +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.driver_version = "580.105.09";
                           }},
         }) {
        auto mismatched = actual;
        mutation.mutate(mismatched);
        bool rejected = false;
        try {
            trtmc::validate_runtime_memory_runtime_target(expected, mismatched);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find(mutation.expected_name) != std::string::npos;
        }
        check(rejected, "runtime stack guard rejects ancillary library/driver drift");
    }
}

void test_backend_stack_json_is_independent_and_fail_closed() {
    const std::string valid =
        R"({"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"})";
    const auto actual = trtmc::parse_runtime_memory_runtime_stack_json(valid);
    check(actual.compute_capability_major == 10 && actual.compute_capability_minor == 3,
          "backend runtime stack parses independently detected SM");
    check(actual.cudnn_frontend_revision == "7b9b711c22b6823e87150213ecd8449260db8610",
          "backend runtime stack preserves compiled Frontend revision");

    const std::vector<std::string> invalid_values = {
        "",
        R"({"sm":"sm103"})",
        R"({"sm":"unavailable","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"})",
    };
    for (const std::string& invalid : invalid_values) {
        bool rejected = false;
        try {
            (void)trtmc::parse_runtime_memory_runtime_stack_json(invalid);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "missing or malformed backend stack evidence fails closed");
    }
}

void test_cache_copy_events_are_measured_and_rejected() {
    trtmc::RuntimeMemoryTransferSnapshotV1 before;
    trtmc::RuntimeMemoryTransferSnapshotV1 after;
    after.event_sequence = 2;
    after.counters.push_back({"cache_k_0", 64, 128, 1, 1, true});
    const auto delta = trtmc::runtime_memory_transfer_delta(before, after);
    check(delta.runtime_kv_device_to_host_bytes == 64,
          "cache D2H event contributes its actual bytes");
    check(delta.runtime_kv_device_to_device_bytes == 128,
          "cache D2D event contributes its actual bytes");
    check(delta.runtime_kv_device_to_host_events == 1, "cache D2H event count is retained");
    check(delta.runtime_kv_device_to_device_events == 1, "cache D2D event count is retained");

    for (const auto [d2h, d2d] : {std::pair<std::uint64_t, std::uint64_t>{64, 0},
                                  std::pair<std::uint64_t, std::uint64_t>{0, 128}}) {
        auto result = result_with_transfer(d2h, d2d);
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(result);
        } catch (const std::logic_error& error) {
            rejected = std::string(error.what()).find("forbidden KV transfer") != std::string::npos;
        }
        check(rejected, "one injected cache transfer event fails qualification");
    }
}

void test_non_kv_transfer_is_filtered() {
    trtmc::RuntimeMemoryTransferSnapshotV1 before;
    trtmc::RuntimeMemoryTransferSnapshotV1 after;
    after.event_sequence = 1;
    after.counters.push_back({"logits", 4096, 0, 1, 0, false});
    const auto delta = trtmc::runtime_memory_transfer_delta(before, after);
    check(delta.runtime_kv_device_to_host_bytes == 0, "logits D2H is not mislabeled as KV traffic");
}

void test_measured_current_row_commit_must_match_sq_times_b() {
    auto exact = result_with_transfer(0, 0);
    trtmc::finalize_runtime_memory_invocation_traces(exact);
    check(exact.invocations[0].kv_allocation_id == 9 && exact.invocations[0].reserved_tokens == 128,
          "per-invocation allocation sample is preserved after receipt cross-check");
    check(exact.invocations[0].kv_append_bytes == 16,
          "measured exact-Sq append bytes are preserved");
    check(exact.invocations[0].kv_append_events == 2,
          "measured per-span append events are preserved");

    for (const auto [bytes, events] : {std::pair<std::uint64_t, std::uint64_t>{15, 2},
                                       std::pair<std::uint64_t, std::uint64_t>{16, 0}}) {
        auto invalid = result_with_transfer(0, 0);
        invalid.invocations[0].kv_append_bytes = bytes;
        invalid.invocations[0].kv_append_events = events;
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid);
        } catch (const std::logic_error& error) {
            rejected =
                std::string(error.what()).find("exact current-row commit") != std::string::npos;
        }
        check(rejected, "qualification rejects synthesized or missing current-row commit traffic");
    }
}

void test_qualification_requires_v2_schema4_module_residency_receipt() {
    const auto replace_receipt_field =
        [](trtmc::RuntimeMemoryQualificationResultV1& result, const std::string& expected,
           const std::string& replacement) {
            const auto offset = result.runtime_memory_receipt_json.find(expected);
            if (offset == std::string::npos)
                throw std::logic_error("test receipt does not contain expected field");
            result.runtime_memory_receipt_json.replace(offset, expected.size(), replacement);
        };
    {
        auto legacy_schema = result_with_transfer(0, 0);
        replace_receipt_field(legacy_schema, "\"receipt_schema_version\":4",
                              "\"receipt_schema_version\":3");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(legacy_schema);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("schema 4") != std::string::npos;
        }
        check(rejected, "qualification rejects a legacy receipt schema");
    }
    {
        auto provisional_contract = result_with_transfer(0, 0);
        replace_receipt_field(provisional_contract, "\"contract_version\":2",
                              "\"contract_version\":1");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(provisional_contract);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("contract version 2") !=
                       std::string::npos;
        }
        check(rejected, "qualification rejects a provisional runtime-memory contract");
    }
    {
        auto missing_provenance = result_with_transfer(0, 0);
        replace_receipt_field(missing_provenance,
                              "\"module_residency_reserve_bytes\":268435456",
                              "\"module_residency_reserve_bytes\":0");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(missing_provenance);
        } catch (const std::runtime_error& error) {
            rejected =
                std::string(error.what()).find("module-residency provenance") !=
                std::string::npos;
        }
        check(rejected, "qualification rejects an unbounded module-residency receipt");
    }
    {
        auto stale_profile = result_with_transfer(0, 0);
        replace_receipt_field(stale_profile,
                              "\"module_residency_reserve_profile_limit\":128",
                              "\"module_residency_reserve_profile_limit\":127");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(stale_profile);
        } catch (const std::runtime_error& error) {
            rejected =
                std::string(error.what()).find("module-residency provenance") !=
                std::string::npos;
        }
        check(rejected, "qualification rejects a reserve row that does not cover runtime R");
    }
    for (const auto& [expected, replacement] :
         std::vector<std::pair<std::string, std::string>>{
             {
                 "\"module_residency_plan_set_sha256\":"
                 "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"",
                 "\"module_residency_plan_set_sha256\":\"not-a-sha256\"",
             },
             {
                 "\"module_residency_evidence_sha256\":"
                 "\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"",
                 "\"module_residency_evidence_sha256\":\""
                 "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\"",
             },
             {
                 "\"module_residency_cuda_module_loading_mode\":\"lazy\"",
                 "\"module_residency_cuda_module_loading_mode\":\"unknown\"",
             },
         }) {
        auto invalid_provenance = result_with_transfer(0, 0);
        replace_receipt_field(invalid_provenance, expected, replacement);
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid_provenance);
        } catch (const std::runtime_error& error) {
            rejected =
                std::string(error.what()).find("module-residency provenance") !=
                std::string::npos;
        }
        check(rejected,
              "qualification rejects malformed plan, evidence, or loading-mode provenance");
    }

    {
        auto invalid_kv_allocation = result_with_transfer(0, 0);
        replace_receipt_field(invalid_kv_allocation, "\"kv_reserved_bytes\":2048",
                              "\"kv_reserved_bytes\":2049");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid_kv_allocation);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("exactly R*B") != std::string::npos;
        }
        check(rejected, "qualification rejects a KV allocation that is not exactly R*B");
    }
    {
        auto invalid_capacity_snapshot = result_with_transfer(0, 0);
        replace_receipt_field(invalid_capacity_snapshot,
                              "\"capacity_decision_device_used_bytes\":536870912",
                              "\"capacity_decision_device_used_bytes\":536870911");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid_capacity_snapshot);
        } catch (const std::runtime_error& error) {
            rejected =
                std::string(error.what()).find("capacity-decision snapshot") !=
                std::string::npos;
        }
        check(rejected, "qualification rejects an incoherent capacity-decision snapshot");
    }
    {
        auto missing_settled_snapshot = result_with_transfer(0, 0);
        replace_receipt_field(missing_settled_snapshot, "\"settled_free_bytes\":536868864",
                              "\"settled_free_bytes\":null");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(missing_settled_snapshot);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("settled_free_bytes") !=
                       std::string::npos;
        }
        check(rejected, "qualification fails closed when the settled snapshot is unavailable");
    }
    {
        auto invalid_settled_snapshot = result_with_transfer(0, 0);
        replace_receipt_field(invalid_settled_snapshot,
                              "\"settled_device_used_bytes\":536872960",
                              "\"settled_device_used_bytes\":536872959");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid_settled_snapshot);
        } catch (const std::runtime_error& error) {
            rejected =
                std::string(error.what()).find("settled snapshot") != std::string::npos;
        }
        check(rejected, "qualification rejects an incoherent settled snapshot");
    }
    {
        auto stale_final_alias = result_with_transfer(0, 0);
        replace_receipt_field(stale_final_alias, "\"final_free_bytes\":536870912",
                              "\"final_free_bytes\":536870911");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(stale_final_alias);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("deprecated final aliases") !=
                       std::string::npos;
        }
        check(rejected,
              "qualification requires deprecated final aliases to equal capacity-decision");
    }
    {
        auto invalid_overhead_delta = result_with_transfer(0, 0);
        replace_receipt_field(invalid_overhead_delta, "\"final_non_kv_overhead_delta_bytes\":0",
                              "\"final_non_kv_overhead_delta_bytes\":1");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid_overhead_delta);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("final non-KV overhead delta") !=
                       std::string::npos;
        }
        check(rejected, "qualification rejects double-charged or omitted non-KV overhead");
    }
    {
        auto overflowed_kv_product = result_with_transfer(0, 0);
        replace_receipt_field(overflowed_kv_product, "\"kv_bytes_per_token\":16",
                              "\"kv_bytes_per_token\":18446744073709551615");
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(overflowed_kv_product);
        } catch (const std::overflow_error& error) {
            rejected = std::string(error.what()).find("R*B overflowed") != std::string::npos;
        }
        check(rejected, "qualification fails closed when receipt R*B overflows");
    }
}

void test_invocation_must_sample_active_allocation() {
    for (const auto [allocation_id, reserved_tokens] :
         {std::pair<std::uint64_t, std::uint64_t>{0, 128},
          std::pair<std::uint64_t, std::uint64_t>{8, 128},
          std::pair<std::uint64_t, std::uint64_t>{9, 0},
          std::pair<std::uint64_t, std::uint64_t>{9, 127}}) {
        auto invalid = result_with_transfer(0, 0);
        invalid.invocations[0].kv_allocation_id = allocation_id;
        invalid.invocations[0].reserved_tokens = reserved_tokens;
        bool rejected = false;
        try {
            trtmc::finalize_runtime_memory_invocation_traces(invalid);
        } catch (const std::logic_error& error) {
            rejected = std::string(error.what()).find("active KV allocation") != std::string::npos;
        }
        check(rejected, "qualification rejects a synthesized or stale allocation sample");
    }
}

void test_exact_m_observability_does_not_query_m_plus_one() {
    bool queried_next = false;
    const auto rows = trtmc::resolve_runtime_memory_post_step_trace_rows(2048, 2048, 2048, [&] {
        queried_next = true;
        throw std::runtime_error("M+1 must not be queried");
        return 0;
    });
    check(rows == 2048, "exact-M trace keeps the completed invocation rows");
    check(!queried_next, "exact-M trace does not resolve an M+1 profile");

    const auto below_limit =
        trtmc::resolve_runtime_memory_post_step_trace_rows(127, 2048, 128, [&] {
            queried_next = true;
            return 512;
        });
    check(below_limit == 512, "below-M trace still resolves the next profile");
    check(queried_next, "below-M trace queries the next invocation rows");
}

} // namespace

int main() {
    test_embedded_calibration_evidence_is_bound_before_deserialization();
    test_runtime_config_bytes_are_bound_before_deserialization();
    test_exact_qualified_tuple_rejects_tampered_bundle_identity();
    test_developer_c_div_2_tuple_requires_exact_opt_in_and_buckets();
    test_exact_runtime_target_rejects_gpu_and_trt_drift();
    test_backend_stack_json_is_independent_and_fail_closed();
    test_cache_copy_events_are_measured_and_rejected();
    test_non_kv_transfer_is_filtered();
    test_measured_current_row_commit_must_match_sq_times_b();
    test_qualification_requires_v2_schema4_module_residency_receipt();
    test_invocation_must_sample_active_allocation();
    test_exact_m_observability_does_not_query_m_plus_one();
    if (failures != 0)
        return 1;
    std::cout << "runtime memory transfer ledger checks passed\n";
    return 0;
}
