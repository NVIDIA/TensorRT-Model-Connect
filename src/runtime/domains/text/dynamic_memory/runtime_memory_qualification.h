/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Private, test-only qualification surface for native text runtime-memory
// pipelines. This deliberately lives outside include/trtmc and outside the
// public IPipeline vtable: release qualification needs raw token IDs and full
// logits, while product callers should continue to use the normal generation
// APIs.

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr std::uint32_t kRuntimeMemoryQualificationApiVersionV1 = 1U;

struct RuntimeMemoryContract;
struct QualifiedRuntimeStack;
class ITrtModule;

struct RuntimeMemoryQualifiedTuple {
    std::string model_id;
    std::string revision;
    std::string config_sha256;
    std::string target;
    std::string gpu_architecture;
    std::string trt_runtime_version;
    std::string cuda_runtime_version;
    std::string cudnn_backend_version;
    std::string cudnn_frontend_revision;
    std::string nvrtc_version;
    std::string driver_version;
    std::int32_t contract_version{1};
    std::int32_t native_kv_plugin_abi{2};
    std::int32_t model_context_limit{0};
    std::int32_t prefill_chunk_limit{0};
    std::string kv_layout{"contiguous_runtime_v1"};
    std::string kv_dtype{"bfloat16"};
    std::vector<std::int32_t> active_kv_profile_limits;
};

void validate_runtime_memory_qualified_tuple(const RuntimeMemoryContract& contract,
                                             const RuntimeMemoryQualifiedTuple& expected);

struct RuntimeMemoryRuntimeTarget {
    std::int32_t cuda_device{-1};
    std::int32_t compute_capability_major{-1};
    std::int32_t compute_capability_minor{-1};
    std::string trt_runtime_version;
    std::string cuda_runtime_version;
    std::string cudnn_backend_version;
    std::string cudnn_frontend_revision;
    std::string nvrtc_version;
    std::string driver_version;
};

// Exact runtime target gate for the two qualified native dynamic-memory
// pipelines. Production receives independently detected JSON from the
// selected backend/common-plugin DSO; bundle metadata is never accepted as
// evidence for the live runtime.
RuntimeMemoryRuntimeTarget parse_runtime_memory_runtime_stack_json(const std::string& json_text);
void validate_runtime_memory_runtime_stack(const QualifiedRuntimeStack& expected,
                                           const RuntimeMemoryRuntimeTarget& actual);
void validate_runtime_memory_runtime_target(const RuntimeMemoryQualifiedTuple& expected,
                                            const RuntimeMemoryRuntimeTarget& actual);

struct RuntimeMemoryQualificationExecutionAttemptEvidence {
    std::string source;
    bool available{false};
    std::uint64_t module_count{0};
    std::uint64_t before{0};
    std::uint64_t after{0};
    std::uint64_t delta{0};
};

class RuntimeMemoryQualificationAdmissionError : public std::runtime_error {
  public:
    RuntimeMemoryQualificationAdmissionError(
        const std::string& message,
        RuntimeMemoryQualificationExecutionAttemptEvidence execution_attempt_evidence);
    ~RuntimeMemoryQualificationAdmissionError() override;

    const std::string& execution_attempt_source() const noexcept {
        return execution_attempt_evidence_.source;
    }
    bool execution_attempt_available() const noexcept {
        return execution_attempt_evidence_.available;
    }
    std::uint64_t execution_attempt_module_count() const noexcept {
        return execution_attempt_evidence_.module_count;
    }
    std::uint64_t execution_attempt_before() const noexcept {
        return execution_attempt_evidence_.before;
    }
    std::uint64_t execution_attempt_after() const noexcept {
        return execution_attempt_evidence_.after;
    }
    std::uint64_t execution_attempt_delta() const noexcept {
        return execution_attempt_evidence_.delta;
    }

  private:
    RuntimeMemoryQualificationExecutionAttemptEvidence execution_attempt_evidence_;
};

// A request-local baseline over every unique prefill/decode module owned by a
// qualification pipeline. It is intentionally private to the native
// qualification surface and retains per-module values so aggregate arithmetic
// cannot hide one module's counter regression behind another module's advance.
struct RuntimeMemoryQualificationExecutionAttemptBaseline {
    std::vector<const ITrtModule*> modules;
    std::vector<std::uint64_t> before_by_module;
    std::uint64_t before{0};
};

RuntimeMemoryQualificationExecutionAttemptBaseline
capture_runtime_memory_qualification_execution_attempts(
    const std::vector<const ITrtModule*>& modules);
RuntimeMemoryQualificationExecutionAttemptEvidence
finish_runtime_memory_qualification_execution_attempts(
    const RuntimeMemoryQualificationExecutionAttemptBaseline& baseline);

struct RuntimeMemoryQualificationRequestV1 {
    std::vector<std::int32_t> input_ids;
    std::int32_t max_new_tokens{0};
};

// One real TensorRT invocation made by the split prefill/decode pipeline.
// This remains a qualification-only contract: product callers do not pay for
// tracing and the installed IPipeline/C ABI remain unchanged.
struct RuntimeMemoryInvocationTraceV1 {
    std::uint64_t invocation_index{0};
    std::string role;
    std::string plan_id;
    std::int32_t profile_id{-1};
    std::uint64_t chunk_begin{0};
    std::uint64_t chunk_end{0};
    std::uint32_t launch_count{1};
    std::uint64_t kv_allocation_id{0};
    std::uint64_t kv_base_address{0};
    std::uint64_t history_tokens{0};  // H
    std::uint64_t active_tokens{0};   // A
    std::uint64_t bound_tokens{0};    // T
    std::uint64_t reserved_tokens{0}; // R
    std::uint64_t context_device_memory_bytes{0};
    std::string cuda_graph_status;
    std::uint64_t kv_device_to_host_bytes{0};
    std::uint64_t kv_append_bytes{0};
    std::uint64_t kv_append_events{0};
    std::uint64_t full_history_device_to_device_bytes{0};
};

// Bind a qualification trace to the actual deserialized TensorRT engine, not
// merely to a caller-supplied role label. The section name is stable bundle
// identity; the opaque engine identity proves which loaded plan served the
// invocation inside this process.
std::string runtime_memory_execution_plan_identity(const std::string& bundle_section_name,
                                                   const ITrtModule& module);

struct RuntimeMemoryQualificationResultV1 {
    // Greedy tokens selected from step_logits[0..max_new_tokens-1].
    std::vector<std::int32_t> selected_token_ids;

    // Row 0 is the final prompt-position logits. Row i+1 is the logits after
    // executing selected_token_ids[i]. Thus a decode request of N tokens
    // returns N+1 complete vocabulary rows and proves that the Nth token was
    // actually executed (including position M-1 at the model boundary).
    std::vector<std::vector<float>> step_logits;

    std::uint64_t prompt_tokens{0};
    std::uint64_t runtime_kv_capacity_tokens{0};
    std::uint64_t effective_request_limit{0};
    std::uint32_t prefill_chunk_limit{0};
    std::uint32_t prefill_launches{0};
    std::uint32_t decode_launches{0};
    std::uint64_t final_kv_position{0};
    std::vector<RuntimeMemoryInvocationTraceV1> invocations;

    // The versioned runtime receipt includes the allocation ID and exact
    // reserved/committed byte trace. Keeping it intact prevents a second,
    // qualification-only interpretation of memory accounting.
    std::string runtime_memory_receipt_json;
};

class IRuntimeMemoryQualificationV1 {
  public:
    virtual ~IRuntimeMemoryQualificationV1();

    virtual std::uint32_t runtime_memory_qualification_api_version() const {
        return kRuntimeMemoryQualificationApiVersionV1;
    }

    virtual RuntimeMemoryQualificationResultV1
    qualify_runtime_memory(const RuntimeMemoryQualificationRequestV1& request) = 0;
};

// Cross-check per-invocation observations against the finalized versioned
// runtime receipt. Model pipelines call this only after the real execution
// has finished and the receipt has been captured.
void finalize_runtime_memory_invocation_traces(RuntimeMemoryQualificationResultV1& result);

std::int32_t
resolve_runtime_memory_post_step_trace_rows(std::int32_t completed_position,
                                            std::int32_t max_length, std::int32_t current_rows,
                                            const std::function<std::int32_t()>& next_rows);

} // namespace trtmc
