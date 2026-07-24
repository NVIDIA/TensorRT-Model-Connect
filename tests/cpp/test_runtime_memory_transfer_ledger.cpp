/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"
#include "trtmc/bundle.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

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
    ScopedEnvironment(std::string name, std::optional<std::string> value)
        : name_(std::move(name)) {
        if (const char* existing = std::getenv(name_.c_str()))
            previous_ = std::string(existing);
        apply(value);
    }

    ~ScopedEnvironment() {
        apply(previous_);
    }

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
        R"({"kv_allocation_id":9,"runtime_kv_capacity_tokens":128,"kv_bytes_per_token":16,"context_device_memory_bytes":4096})";
    trtmc::RuntimeMemoryInvocationTraceV1 trace;
    trace.role = "prefill";
    trace.plan_id = "engine_plan:prefill";
    trace.profile_id = 1;
    trace.chunk_end = 1;
    trace.active_tokens = 1;
    trace.bound_tokens = 1;
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
    expected.cudnn_frontend_revision =
        "7b9b711c22b6823e87150213ecd8449260db8610";
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
    contract.qualified_runtime_stack.cuda_runtime =
        expected.cuda_runtime_version;
    contract.qualified_runtime_stack.cudnn_backend =
        expected.cudnn_backend_version;
    contract.qualified_runtime_stack.cudnn_frontend_revision =
        expected.cudnn_frontend_revision;
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
    expected.cudnn_frontend_revision =
        "7b9b711c22b6823e87150213ecd8449260db8610";
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
    contract.qualified_runtime_stack.cuda_runtime =
        expected.cuda_runtime_version;
    contract.qualified_runtime_stack.cudnn_backend =
        expected.cudnn_backend_version;
    contract.qualified_runtime_stack.cudnn_frontend_revision =
        expected.cudnn_frontend_revision;
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
    expected.cudnn_frontend_revision =
        "7b9b711c22b6823e87150213ecd8449260db8610";
    expected.nvrtc_version = "13.3";
    expected.driver_version = "580.105.08";

    trtmc::RuntimeMemoryRuntimeTarget actual;
    actual.cuda_device = 2;
    actual.compute_capability_major = 10;
    actual.compute_capability_minor = 3;
    actual.trt_runtime_version = "11.2.0.113";
    actual.cuda_runtime_version = "13.3";
    actual.cudnn_backend_version = "9.20.0";
    actual.cudnn_frontend_revision =
        "7b9b711c22b6823e87150213ecd8449260db8610";
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
             StackMutation{"CUDA runtime", +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.cuda_runtime_version = "13.1";
                           }},
             StackMutation{"cuDNN backend", +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.cudnn_backend_version = "9.20.1";
                           }},
             StackMutation{"cuDNN Frontend revision",
                           +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.cudnn_frontend_revision = "unqualified";
                           }},
             StackMutation{"NVRTC", +[](trtmc::RuntimeMemoryRuntimeTarget& value) {
                               value.nvrtc_version = "13.0";
                           }},
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
            rejected = std::string(error.what()).find(mutation.expected_name) !=
                       std::string::npos;
        }
        check(rejected, "runtime stack guard rejects ancillary library/driver drift");
    }
}

void test_backend_stack_json_is_independent_and_fail_closed() {
    const std::string valid =
        R"({"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"})";
    const auto actual = trtmc::parse_runtime_memory_runtime_stack_json(valid);
    check(actual.compute_capability_major == 10 &&
              actual.compute_capability_minor == 3,
          "backend runtime stack parses independently detected SM");
    check(actual.cudnn_frontend_revision ==
              "7b9b711c22b6823e87150213ecd8449260db8610",
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
    test_exact_qualified_tuple_rejects_tampered_bundle_identity();
    test_developer_c_div_2_tuple_requires_exact_opt_in_and_buckets();
    test_exact_runtime_target_rejects_gpu_and_trt_drift();
    test_backend_stack_json_is_independent_and_fail_closed();
    test_cache_copy_events_are_measured_and_rejected();
    test_non_kv_transfer_is_filtered();
    test_measured_current_row_commit_must_match_sq_times_b();
    test_exact_m_observability_does_not_query_m_plus_one();
    if (failures != 0)
        return 1;
    std::cout << "runtime memory transfer ledger checks passed\n";
    return 0;
}
