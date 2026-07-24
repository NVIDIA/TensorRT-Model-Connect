/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_kv_state.h"

#include "runtime/domains/text/dynamic_memory/runtime_kv_setup.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace {

struct CacheShape {
    std::vector<int64_t> dimensions;
    int32_t sequence_axis{-1};
};

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
        throw std::overflow_error(std::string(what) + " overflows uint64");
    return lhs + rhs;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error(std::string(what) + " overflows uint64");
    return lhs * rhs;
}

std::uint64_t staging_token_capacity_for_layout(const RuntimeKvGraphLayout& layout) {
    return std::min<std::uint64_t>(layout.prefill_chunk_limit, layout.capacity_tokens);
}

void validate_layout(const RuntimeKvGraphLayout& layout) {
    if (layout.layer_count == 0 || layout.kv_head_count == 0 || layout.head_dim == 0)
        throw std::invalid_argument("Runtime KV graph layout has a zero model dimension");
    if (layout.capacity_tokens == 0)
        throw std::invalid_argument("Runtime KV graph layout has zero capacity");
    // C is a build-time per-launch maximum while capacity_tokens is runtime R.
    // A small runtime budget may validly resolve R<C; the scheduler then uses
    // Sq=min(C, remaining request tokens).
    if (layout.prefill_chunk_limit == 0)
        throw std::invalid_argument("Runtime KV prefill chunk limit is zero");
    if (layout.active_kv_profile_limits.empty()) {
        throw std::invalid_argument(
            "Runtime KV graph layout has no history-bound profile contract");
    }
    for (std::size_t index = 0; index < layout.active_kv_profile_limits.size(); ++index) {
        const auto limit = layout.active_kv_profile_limits[index];
        if (limit == 0 || (index > 0 && limit <= layout.active_kv_profile_limits[index - 1])) {
            throw std::invalid_argument(
                "Runtime KV history-bound profile limits must be positive and "
                "strictly increasing");
        }
    }
    if (layout.active_kv_profile_limits.back() < layout.capacity_tokens) {
        throw std::invalid_argument(
            "Runtime KV history-bound profiles do not cover runtime capacity");
    }
    const auto layers = static_cast<std::size_t>(layout.layer_count);
    if (layout.names.cache_k.size() != layers || layout.names.cache_v.size() != layers) {
        throw std::invalid_argument("Runtime KV graph layout has the wrong cache tensor count");
    }
    if (layout.names.cache_k_output.size() != layers ||
        layout.names.cache_v_output.size() != layers) {
        throw std::invalid_argument(
            "Runtime KV graph layout requires one exact-Sq K/V output per layer");
    }
    if (layout.names.history_length.empty()) {
        throw std::invalid_argument("Runtime KV graph layout requires a history_length input");
    }
}

CacheShape cache_shape_for_module(const ITrtModule& module, const std::string& name,
                                  const RuntimeKvGraphLayout& layout, std::uint64_t bound_tokens) {
    if (bound_tokens > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("Runtime KV bound extent exceeds int64");
    const auto bound = static_cast<int64_t>(bound_tokens);
    const auto heads = static_cast<int64_t>(layout.kv_head_count);
    const auto dim = static_cast<int64_t>(layout.head_dim);
    const int32_t rank = module.input_rank(name);
    if (rank == 2)
        return CacheShape{{bound, heads * dim}, 0};
    if (rank == 4)
        return CacheShape{{1, heads, bound, dim}, 2};
    throw std::invalid_argument("Runtime KV tensor '" + name + "' must have rank 2 or rank 4");
}

CacheShape current_output_shape_for_module(const ITrtModule& module, const std::string& name,
                                           const RuntimeKvGraphLayout& layout,
                                           std::uint64_t query_tokens) {
    if (!module.has_output(name)) {
        throw std::invalid_argument("Runtime KV current-row tensor '" + name +
                                    "' is not an engine output");
    }
    if (query_tokens > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("Runtime KV query extent exceeds int64");
    }
    const auto declared = module.tensor_shape(name);
    if (declared.size() != 2) {
        throw std::invalid_argument("Runtime KV current-row output '" + name +
                                    "' must be token-major rank 2 [Sq, kv_dim]");
    }
    const auto width =
        checked_mul(layout.kv_head_count, layout.head_dim, "Runtime KV current-row width");
    if (width > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("Runtime KV current-row width exceeds int64");
    }
    return CacheShape{{static_cast<int64_t>(query_tokens), static_cast<int64_t>(width)}, 0};
}

RuntimeMemoryShapeV1 make_shape(const std::string& name, const std::vector<int64_t>& dimensions,
                                DType dtype, std::uint64_t valid_tokens, std::uint64_t bound_tokens,
                                std::uint64_t capacity_tokens, int32_t sequence_axis) {
    RuntimeMemoryShapeV1 shape;
    shape.name = name;
    shape.shape = dimensions;
    shape.dtype = dtype;
    shape.valid_tokens = valid_tokens;
    shape.bound_tokens = bound_tokens;
    shape.capacity_tokens = capacity_tokens;
    shape.sequence_axis = sequence_axis;
    return shape;
}

void set_dynamic_input_shape(IRuntimeMemoryModuleV1& runtime_module, const ITrtModule& module,
                             const std::string& name, std::uint64_t query_tokens) {
    if (name.empty() || !module.has_input(name) || !module.input_is_dynamic(name))
        return;
    if (query_tokens > static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("Runtime query extent exceeds int64");
    RuntimeInputShapeV1 input_shape;
    input_shape.name = name;
    input_shape.shape = {static_cast<int64_t>(query_tokens)};
    runtime_module.set_runtime_input_shape(input_shape);
}

IRuntimeMemoryModuleV1& require_runtime_module(ITrtModule& module) {
    auto* runtime_module = dynamic_cast<IRuntimeMemoryModuleV1*>(&module);
    if (runtime_module == nullptr)
        throw std::invalid_argument(
            "Qualified runtime-memory graph requires the standard TensorRT backend");
    return *runtime_module;
}

} // namespace

RuntimeMemoryContextRequirementV1 plan_runtime_kv_invocation(ITrtModule& module,
                                                             const RuntimeKvGraphLayout& layout,
                                                             std::uint64_t history_tokens,
                                                             std::uint64_t query_tokens,
                                                             std::uint64_t bound_tokens) {
    validate_layout(layout);
    if (query_tokens == 0)
        throw std::invalid_argument("Runtime KV invocation has zero query tokens");
    const auto active_tokens = checked_add(history_tokens, query_tokens, "Runtime KV H + Sq");
    const bool valid_history_binding =
        history_tokens == 0 ? bound_tokens == 1
                            : bound_tokens >= std::max<std::uint64_t>(history_tokens, 2);
    if (active_tokens > layout.capacity_tokens || bound_tokens == 0 ||
        bound_tokens > layout.capacity_tokens || !valid_history_binding) {
        throw std::invalid_argument("Runtime KV invocation violates the cold-sentinel H/T contract "
                                    "or H + Sq <= R");
    }

    auto& runtime_module = require_runtime_module(module);
    if (!module.has_input(layout.names.history_length) ||
        module.tensor_dtype(layout.names.history_length) != DType::kInt32) {
        throw std::invalid_argument("Runtime KV graph requires an INT32 history_length input");
    }
    const auto history_shape = module.tensor_shape(layout.names.history_length);
    if (history_shape.size() != 1 || (history_shape.front() >= 0 && history_shape.front() != 1)) {
        throw std::invalid_argument("Runtime KV history_length input must have shape [1]");
    }
    set_dynamic_input_shape(runtime_module, module, layout.names.token_id, query_tokens);
    set_dynamic_input_shape(runtime_module, module, layout.names.position_id, query_tokens);

    // Set every dynamic input before asking TensorRT to infer any output. An
    // output shape query performed while another cache input is unresolved is
    // invalid even when the output's Sq dimension is otherwise obvious.
    for (std::uint32_t layer = 0; layer < layout.layer_count; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (int32_t value = 0; value < 2; ++value) {
            const auto& input_name =
                value == 0 ? layout.names.cache_k[index] : layout.names.cache_v[index];
            const auto cache = cache_shape_for_module(module, input_name, layout, bound_tokens);
            runtime_module.set_runtime_binding_shape(
                make_shape(input_name, cache.dimensions, layout.dtype, history_tokens, bound_tokens,
                           layout.capacity_tokens, cache.sequence_axis));
        }
    }

    const auto staging_tokens = staging_token_capacity_for_layout(layout);
    if (query_tokens > staging_tokens) {
        throw std::invalid_argument("Runtime KV query extent exceeds exact-row staging capacity");
    }
    for (std::uint32_t layer = 0; layer < layout.layer_count; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (int32_t value = 0; value < 2; ++value) {
            const auto& output_name = value == 0 ? layout.names.cache_k_output[index]
                                                 : layout.names.cache_v_output[index];
            const auto current =
                current_output_shape_for_module(module, output_name, layout, query_tokens);
            runtime_module.set_runtime_binding_shape(
                make_shape(output_name, current.dimensions, layout.dtype, query_tokens,
                           query_tokens, staging_tokens, current.sequence_axis));
        }
    }
    return runtime_module.context_memory_requirement();
}

RuntimeKvStateCore::RuntimeKvStateCore(RuntimeKvGraphLayout layout,
                                       std::unique_ptr<RuntimeKvAllocation> allocation,
                                       std::unique_ptr<RuntimeKvAllocation> staging,
                                       RuntimeMemoryContextBlockV1 context_block,
                                       RuntimeMemoryReceipt receipt, void* stream,
                                       RuntimeKvDeviceCopy device_copy,
                                       RuntimeKvStreamSynchronize stream_synchronize)
    : layout_(std::move(layout)), allocation_(std::move(allocation)), staging_(std::move(staging)),
      context_block_(std::move(context_block)), receipt_(std::move(receipt)), stream_(stream),
      device_copy_(std::move(device_copy)), stream_synchronize_(std::move(stream_synchronize)) {
    validate_layout(layout_);
    if (!allocation_ || !allocation_->valid())
        throw std::invalid_argument("Runtime KV state requires a valid allocation");
    if (!staging_ || !staging_->valid())
        throw std::invalid_argument("Runtime KV state requires valid current-row staging");
    if (allocation_->layer_count() != layout_.layer_count ||
        allocation_->capacity_tokens() != layout_.capacity_tokens ||
        allocation_->row_width() !=
            static_cast<std::uint64_t>(layout_.kv_head_count) * layout_.head_dim) {
        throw std::invalid_argument("Runtime KV allocation does not match graph layout");
    }
    if (staging_->layer_count() != layout_.layer_count ||
        staging_->capacity_tokens() != staging_token_capacity_for_layout(layout_) ||
        staging_->row_width() != allocation_->row_width() ||
        staging_->row_bytes() != allocation_->row_bytes() ||
        staging_->device() != allocation_->device()) {
        throw std::invalid_argument("Runtime KV staging allocation does not match graph layout");
    }
    if (context_block_.capacity_bytes > 0 &&
        (!context_block_.pointer || !context_block_.lifetime)) {
        throw std::invalid_argument("Runtime KV state has an invalid context block");
    }
    if (context_block_.device != static_cast<int32_t>(allocation_->device())) {
        throw std::invalid_argument(
            "Runtime KV context block and allocations use different devices");
    }
    if (!device_copy_) {
        device_copy_ = [](void* destination, std::size_t destination_pitch, const void* source,
                          std::size_t source_pitch, std::size_t width_bytes, std::size_t height,
                          void* stream) {
            return cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch,
                                     width_bytes, height, cudaMemcpyDeviceToDevice,
                                     static_cast<cudaStream_t>(stream));
        };
    }
    if (!stream_synchronize_) {
        stream_synchronize_ = [](void* stream) {
            return cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        };
    }
    receipt_.runtime_kv_capacity_tokens = layout_.capacity_tokens;
    receipt_.effective_request_limit =
        std::min(receipt_.effective_request_limit, layout_.capacity_tokens);
    receipt_.kv_reserved_bytes = allocation_->total_bytes();
    receipt_.kv_committed_bytes = allocation_->total_bytes();
    receipt_.kv_allocation_id = allocation_->allocation_id();
}

std::uint64_t RuntimeKvStateCore::bound_tokens_for(std::uint64_t history_tokens) const {
    if (history_tokens > layout_.capacity_tokens) {
        throw std::invalid_argument("Runtime KV history length must satisfy 0 <= H <= R");
    }
    // TensorRT does not accept a zero sequence extent. The plugin observes
    // history_length=0 and skips this one-row sentinel without reading it.
    if (history_tokens == 0)
        return 1;
    const auto found = std::lower_bound(layout_.active_kv_profile_limits.begin(),
                                        layout_.active_kv_profile_limits.end(), history_tokens);
    if (found == layout_.active_kv_profile_limits.end()) {
        throw std::logic_error("Runtime KV profile contract does not cover history length");
    }
    return std::min(*found, layout_.capacity_tokens);
}

RuntimeMemoryBindingV1
RuntimeKvStateCore::make_binding(const RuntimeKvAllocation& owner, const std::string& name,
                                 void* pointer, const std::vector<int64_t>& shape,
                                 std::uint64_t valid_tokens, std::uint64_t bound_tokens,
                                 std::uint64_t capacity_tokens, int32_t sequence_axis) const {
    RuntimeMemoryBindingV1 binding;
    binding.name = name;
    binding.pointer = pointer;
    binding.capacity_bytes = static_cast<std::size_t>(owner.layer_span_bytes());
    binding.shape = shape;
    binding.dtype = layout_.dtype;
    binding.alignment = static_cast<std::size_t>(owner.alignment());
    binding.device = static_cast<int32_t>(owner.device());
    binding.lifetime = owner.lifetime();
    binding.valid_tokens = valid_tokens;
    binding.bound_tokens = bound_tokens;
    binding.capacity_tokens = capacity_tokens;
    binding.sequence_axis = sequence_axis;
    return binding;
}

void RuntimeKvStateCore::bind_one(IRuntimeMemoryModuleV1& module, const RuntimeKvAllocation& owner,
                                  const std::string& name, void* pointer,
                                  const std::vector<int64_t>& shape, std::uint64_t valid_tokens,
                                  std::uint64_t bound_tokens, std::uint64_t capacity_tokens,
                                  int32_t sequence_axis) {
    module.bind_runtime_memory(make_binding(owner, name, pointer, shape, valid_tokens, bound_tokens,
                                            capacity_tokens, sequence_axis));
}

void RuntimeKvStateCore::prepare_invocation(ITrtModule& module, std::uint64_t history_tokens,
                                            std::uint64_t query_tokens,
                                            std::uint64_t bound_tokens) {
    if (commit_poisoned_) {
        throw std::logic_error("Runtime KV state is poisoned after a failed current-row commit");
    }
    if (invocation_prepared_) {
        throw std::logic_error("Runtime KV invocation was prepared without a successful commit");
    }
    const auto active_tokens = checked_add(history_tokens, query_tokens, "Runtime KV H + Sq");
    if (active_tokens > layout_.capacity_tokens) {
        throw std::invalid_argument("Runtime KV invocation exceeds physical capacity");
    }
    const auto expected_bound = bound_tokens_for(history_tokens);
    if (bound_tokens != expected_bound) {
        throw std::invalid_argument("Runtime KV invocation bound does not match the history "
                                    "profile contract: expected " +
                                    std::to_string(expected_bound) + ", got " +
                                    std::to_string(bound_tokens));
    }
    const auto requirement =
        plan_runtime_kv_invocation(module, layout_, history_tokens, query_tokens, bound_tokens);
    if (requirement.device != context_block_.device) {
        throw std::invalid_argument("Runtime KV context requirement is on a different CUDA device");
    }
    if (requirement.capacity_bytes > context_block_.capacity_bytes) {
        throw std::runtime_error("Actual TensorRT invocation requires " +
                                 std::to_string(requirement.capacity_bytes) +
                                 " context bytes, exceeding the planned shared block " +
                                 std::to_string(context_block_.capacity_bytes));
    }
    last_context_device_memory_bytes_ = static_cast<std::uint64_t>(requirement.capacity_bytes);

    auto& runtime_module = require_runtime_module(module);
    for (std::uint32_t layer = 0; layer < layout_.layer_count; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (int32_t value = 0; value < 2; ++value) {
            const auto& input_name =
                value == 0 ? layout_.names.cache_k[index] : layout_.names.cache_v[index];
            void* pointer =
                value == 0 ? allocation_->key_pointer(layer) : allocation_->value_pointer(layer);
            const auto cache = cache_shape_for_module(module, input_name, layout_, bound_tokens);
            bind_one(runtime_module, *allocation_, input_name, pointer, cache.dimensions,
                     history_tokens, bound_tokens, layout_.capacity_tokens, cache.sequence_axis);
        }
    }
    const auto staging_tokens = staging_->capacity_tokens();
    for (std::uint32_t layer = 0; layer < layout_.layer_count; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        for (int32_t value = 0; value < 2; ++value) {
            const auto& output_name = value == 0 ? layout_.names.cache_k_output[index]
                                                 : layout_.names.cache_v_output[index];
            void* pointer =
                value == 0 ? staging_->key_pointer(layer) : staging_->value_pointer(layer);
            const auto current =
                current_output_shape_for_module(module, output_name, layout_, query_tokens);
            bind_one(runtime_module, *staging_, output_name, pointer, current.dimensions,
                     query_tokens, query_tokens, staging_tokens, current.sequence_axis);
        }
    }
    runtime_module.bind_context_memory(context_block_);
    if (!runtime_module.runtime_memory_ready()) {
        throw std::runtime_error(
            "Runtime KV module is not ready after cache/staging/context binding");
    }
    prepared_history_tokens_ = history_tokens;
    prepared_query_tokens_ = query_tokens;
    invocation_prepared_ = true;
    last_commit_bytes_ = 0;
}

void RuntimeKvStateCore::commit_current_rows(std::uint64_t history_tokens,
                                             std::uint64_t query_tokens) {
    if (commit_poisoned_) {
        throw std::logic_error("Runtime KV state is poisoned after a failed current-row commit");
    }
    if (!invocation_prepared_) {
        throw std::logic_error("Runtime KV current-row commit has no prepared invocation");
    }
    if (history_tokens != prepared_history_tokens_ || query_tokens != prepared_query_tokens_) {
        throw std::logic_error("Runtime KV current-row commit does not match the prepared H/Sq");
    }
    const auto active_tokens = checked_add(history_tokens, query_tokens, "Runtime KV commit range");
    if (query_tokens == 0 || active_tokens > layout_.capacity_tokens ||
        query_tokens > staging_->capacity_tokens()) {
        throw std::invalid_argument("Runtime KV current-row commit violates H + Sq <= R");
    }

    const auto destination_offset = checked_mul(history_tokens, allocation_->row_bytes(),
                                                "Runtime KV commit destination offset");
    const auto bytes_per_span =
        checked_mul(query_tokens, allocation_->row_bytes(), "Runtime KV commit span bytes");
    auto* destination_base =
        static_cast<std::byte*>(allocation_->key_pointer(0)) + destination_offset;
    const void* source_base = staging_->key_pointer(0);
    const auto span_count =
        checked_mul(layout_.layer_count, std::uint64_t{2}, "Runtime KV commit span count");
    const auto status = device_copy_(
        destination_base, static_cast<std::size_t>(allocation_->layer_span_bytes()), source_base,
        static_cast<std::size_t>(staging_->layer_span_bytes()),
        static_cast<std::size_t>(bytes_per_span), static_cast<std::size_t>(span_count), stream_);
    if (status != cudaSuccess) {
        commit_poisoned_ = true;
        throw std::runtime_error("Runtime KV current-row D2D commit failed: " +
                                 std::string(cudaGetErrorString(status)));
    }

    commit_pending_ = true;
    try {
        const auto per_layer =
            checked_mul(bytes_per_span, std::uint64_t{2}, "Runtime KV commit K/V bytes");
        last_commit_bytes_ =
            checked_mul(per_layer, layout_.layer_count, "Runtime KV commit total bytes");
        total_commit_bytes_ = checked_add(total_commit_bytes_, last_commit_bytes_,
                                          "Runtime KV cumulative commit bytes");
        total_commit_events_ = checked_add(total_commit_events_, std::uint64_t{1},
                                           "Runtime KV cumulative commit events");
        invocation_prepared_ = false;
        prepared_history_tokens_ = 0;
        prepared_query_tokens_ = 0;
    } catch (...) {
        commit_poisoned_ = true;
        throw;
    }
}

void RuntimeKvStateCore::synchronize_commits() {
    if (commit_poisoned_) {
        throw std::logic_error("Runtime KV state is poisoned after a failed current-row commit");
    }
    if (!commit_pending_)
        return;

    cudaError_t status = cudaErrorUnknown;
    try {
        status = stream_synchronize_(stream_);
    } catch (...) {
        commit_poisoned_ = true;
        throw;
    }
    if (status != cudaSuccess) {
        commit_poisoned_ = true;
        throw std::runtime_error("Runtime KV asynchronous D2D commit failed: " +
                                 std::string(cudaGetErrorString(status)));
    }
    commit_pending_ = false;
}

void RuntimeKvStateCore::reset_request_state() {
    synchronize_commits();
    if (commit_poisoned_) {
        throw std::logic_error(
            "Runtime KV state cannot be reset after a failed current-row commit");
    }
    invocation_prepared_ = false;
    prepared_history_tokens_ = 0;
    prepared_query_tokens_ = 0;
    last_commit_bytes_ = 0;
    total_commit_bytes_ = 0;
    total_commit_events_ = 0;
}

void RuntimeKvStateCore::sample_request_completion_device_memory() noexcept {
    try {
        const auto sample = query_runtime_device_memory_snapshot(
            "after successful runtime-memory request completion");
        if (sample.device != allocation_->device() || sample.free_bytes == 0 ||
            sample.total_bytes == 0 || sample.free_bytes > sample.total_bytes ||
            (receipt_.pre_load_snapshot_available &&
             sample.total_bytes != receipt_.pre_load_total_bytes)) {
            receipt_.mark_peak_device_sampling_failed(
                "request_completion_cuda_memory_snapshot_invalid");
            return;
        }
        receipt_.observe_peak_device_memory(sample.free_bytes,
                                            RuntimeMemoryPeakSampleBoundary::kRequestCompletion);
    } catch (...) {
        // Observability must never turn a successful inference into a failure.
        receipt_.mark_peak_device_sampling_failed("request_completion_cuda_mem_get_info_failed");
    }
}

bool RuntimeKvStateCore::valid() const {
    const auto exact_product = [](std::uint64_t lhs, std::uint64_t rhs,
                                  std::uint64_t expected) {
        return lhs == 0 || rhs <= std::numeric_limits<std::uint64_t>::max() / lhs
                   ? lhs * rhs == expected
                   : false;
    };
    return !commit_poisoned_ && allocation_ && allocation_->valid() && staging_ &&
           staging_->valid() &&
           exact_product(layout_.capacity_tokens, receipt_.kv_bytes_per_token,
                         allocation_->total_bytes()) &&
           exact_product(staging_token_capacity_for_layout(layout_), receipt_.kv_bytes_per_token,
                         staging_->total_bytes()) &&
           receipt_.runtime_kv_capacity_tokens == layout_.capacity_tokens &&
           receipt_.kv_allocation_id == allocation_->allocation_id();
}

} // namespace trtmc
