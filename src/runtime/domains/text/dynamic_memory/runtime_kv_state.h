/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/domains/text/dynamic_memory/runtime_kv_allocation.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_plan.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;

struct RuntimeKvIoNames {
    std::string token_id{"token_id"};
    std::string position_id{"position_id"};
    // Exact number of valid read-only history rows. The qualified segmented
    // graph consumes H directly; cache commit offsets remain runtime-owned and
    // are never TensorRT inputs.
    std::string history_length{"history_length"};
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    // Exact-Sq token-major engine outputs. They are ordinary deferred outputs,
    // not aliases of the read-only cache inputs.
    std::vector<std::string> cache_k_output;
    std::vector<std::string> cache_v_output;
};

struct RuntimeKvGraphLayout {
    std::uint32_t layer_count{0};
    std::uint32_t kv_head_count{0};
    std::uint32_t head_dim{0};
    std::uint64_t capacity_tokens{0};
    std::uint32_t prefill_chunk_limit{0};
    // Persistent build contract. At runtime, the physical allocation may
    // resolve to R smaller than the terminal build profile M. An invocation
    // binds T to the first profile bucket covering history H, clamped to R.
    // Cold history is represented only by the T=1 sentinel.
    std::vector<std::uint64_t> active_kv_profile_limits;
    DType dtype{DType::kBFloat16};
    RuntimeKvIoNames names;
};

// Set the actual Sq/T shapes required for an invocation and return TensorRT's
// USER_MANAGED context-memory requirement. No KV pointer is required at this
// stage, so the runtime planner can solve R before allocating the slab.
RuntimeMemoryContextRequirementV1 plan_runtime_kv_invocation(ITrtModule& module,
                                                             const RuntimeKvGraphLayout& layout,
                                                             std::uint64_t history_tokens,
                                                             std::uint64_t query_tokens,
                                                             std::uint64_t bound_tokens);

// Internal seam for deterministic copy/red-zone tests. Production uses one
// checked cudaMemcpy2DAsync(..., cudaMemcpyDeviceToDevice, stream) call to
// scatter the exact-Sq rows across every contiguous layer K/V span.
using RuntimeKvDeviceCopy = std::function<cudaError_t(
    void* destination, std::size_t destination_pitch, const void* source, std::size_t source_pitch,
    std::size_t width_bytes, std::size_t height, void* stream)>;
using RuntimeKvStreamSynchronize = std::function<cudaError_t(void* stream)>;

struct RuntimeKvCommitSnapshot {
    std::uint64_t device_to_device_bytes{0};
    std::uint64_t device_to_device_events{0};
};

// Owns the one contiguous KV slab and shared TensorRT context block used by
// the qualified prefill/decode roles.
class RuntimeKvStateCore {
  public:
    RuntimeKvStateCore(RuntimeKvGraphLayout layout, std::unique_ptr<RuntimeKvAllocation> allocation,
                       std::unique_ptr<RuntimeKvAllocation> staging,
                       RuntimeMemoryContextBlockV1 context_block, RuntimeMemoryReceipt receipt,
                       void* stream, RuntimeKvDeviceCopy device_copy = {},
                       RuntimeKvStreamSynchronize stream_synchronize = {});

    void prepare_invocation(ITrtModule& module, std::uint64_t history_tokens,
                            std::uint64_t query_tokens, std::uint64_t bound_tokens);
    // Enqueue exactly Sq rows from per-layer staging into persistent cache
    // offsets [H, H+Sq). The logical caller may advance only after this
    // returns successfully.
    void commit_current_rows(std::uint64_t history_tokens, std::uint64_t query_tokens);
    // Request-completion barrier for the last asynchronous commit. Any
    // delayed CUDA error poisons the state and is propagated to the caller.
    void synchronize_commits();
    void reset_request_state();

    const RuntimeKvGraphLayout& layout() const { return layout_; }
    std::uint64_t bound_tokens_for(std::uint64_t history_tokens) const;
    std::uint64_t capacity_tokens() const { return layout_.capacity_tokens; }
    std::uint64_t allocation_id() const { return allocation_->allocation_id(); }
    std::uint64_t allocation_base_address() const { return allocation_->base_address(); }
    std::uint64_t allocation_bytes() const { return allocation_->total_bytes(); }
    void* cache_key_pointer(std::uint32_t layer) const { return allocation_->key_pointer(layer); }
    void* cache_value_pointer(std::uint32_t layer) const {
        return allocation_->value_pointer(layer);
    }
    std::uint64_t staging_capacity_tokens() const { return staging_->capacity_tokens(); }
    std::uint64_t staging_bytes() const { return staging_->total_bytes(); }
    std::uint64_t context_allocation_bytes() const { return context_block_.capacity_bytes; }
    void* staging_key_pointer(std::uint32_t layer) const { return staging_->key_pointer(layer); }
    void* staging_value_pointer(std::uint32_t layer) const {
        return staging_->value_pointer(layer);
    }
    std::uint64_t last_commit_bytes() const { return last_commit_bytes_; }
    std::uint64_t total_commit_bytes() const { return total_commit_bytes_; }
    RuntimeKvCommitSnapshot commit_snapshot() const {
        return RuntimeKvCommitSnapshot{total_commit_bytes_, total_commit_events_};
    }
    std::uint64_t last_context_device_memory_bytes() const {
        return last_context_device_memory_bytes_;
    }
    void sample_request_completion_device_memory() noexcept;
    const RuntimeMemoryReceipt& receipt() const { return receipt_; }
    std::string receipt_json() const { return receipt_.to_json(); }
    bool valid() const;

  private:
    RuntimeMemoryBindingV1 make_binding(const RuntimeKvAllocation& owner, const std::string& name,
                                        void* pointer, const std::vector<int64_t>& shape,
                                        std::uint64_t valid_tokens, std::uint64_t bound_tokens,
                                        std::uint64_t capacity_tokens, int32_t sequence_axis) const;
    void bind_one(IRuntimeMemoryModuleV1& module, const RuntimeKvAllocation& owner,
                  const std::string& name, void* pointer, const std::vector<int64_t>& shape,
                  std::uint64_t valid_tokens, std::uint64_t bound_tokens,
                  std::uint64_t capacity_tokens, int32_t sequence_axis);

    RuntimeKvGraphLayout layout_;
    std::unique_ptr<RuntimeKvAllocation> allocation_;
    std::unique_ptr<RuntimeKvAllocation> staging_;
    RuntimeMemoryContextBlockV1 context_block_;
    RuntimeMemoryReceipt receipt_;
    void* stream_{nullptr};
    RuntimeKvDeviceCopy device_copy_;
    RuntimeKvStreamSynchronize stream_synchronize_;
    std::uint64_t last_context_device_memory_bytes_{0};
    std::uint64_t prepared_history_tokens_{0};
    std::uint64_t prepared_query_tokens_{0};
    std::uint64_t last_commit_bytes_{0};
    std::uint64_t total_commit_bytes_{0};
    std::uint64_t total_commit_events_{0};
    bool invocation_prepared_{false};
    bool commit_pending_{false};
    bool commit_poisoned_{false};
};

} // namespace trtmc
