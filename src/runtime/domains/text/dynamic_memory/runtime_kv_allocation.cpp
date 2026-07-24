/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_kv_allocation.h"

#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

constexpr std::uint64_t kRuntimeKvAlignment = 256;

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error(std::string(what) + " overflows uint64");
    return lhs * rhs;
}

std::atomic<std::uint64_t>& next_allocation_id() {
    static std::atomic<std::uint64_t> next{1};
    return next;
}

} // namespace

RuntimeKvAllocation::RuntimeKvAllocation(std::uint32_t layer_count, std::uint64_t capacity_tokens,
                                         std::uint64_t row_width, std::uint64_t element_bytes,
                                         std::uint32_t device, void* stream,
                                         std::shared_ptr<IRuntimeDeviceAllocator> allocator)
    : layer_count_(layer_count), capacity_tokens_(capacity_tokens), row_width_(row_width) {
    if (layer_count_ == 0)
        throw std::invalid_argument("Runtime KV allocation requires at least one layer");
    if (capacity_tokens_ == 0)
        throw std::invalid_argument("Runtime KV allocation requires positive token capacity");
    if (row_width_ == 0)
        throw std::invalid_argument("Runtime KV allocation requires positive row width");
    if (element_bytes == 0)
        throw std::invalid_argument("Runtime KV allocation requires positive element size");
    if (!allocator)
        throw std::invalid_argument("Runtime KV allocation requires an allocator");

    row_bytes_ = checked_mul(row_width_, element_bytes, "Runtime KV row size");
    layer_span_bytes_ = checked_mul(capacity_tokens_, row_bytes_, "Runtime KV layer span size");
    // Every K/V span is bound as an independent TensorRT tensor address.
    // The qualified v1 layouts therefore require a naturally aligned row so
    // all span starts remain 256-byte aligned without adding padding that
    // would make the physical allocation differ from R * B.
    if (row_bytes_ % kRuntimeKvAlignment != 0) {
        throw std::invalid_argument(
            "Runtime KV row size must be a multiple of the TensorRT alignment");
    }
    const auto pair_bytes =
        checked_mul(layer_span_bytes_, std::uint64_t{2}, "Runtime KV layer pair size");
    const auto total_bytes = checked_mul(pair_bytes, layer_count_, "Runtime KV allocation size");

    allocation_ = allocator->allocate(total_bytes, kRuntimeKvAlignment, device, stream);
    if (!allocation_.valid())
        throw std::runtime_error("Runtime KV allocator returned an invalid allocation");
    if (allocation_.bytes < total_bytes) {
        throw std::runtime_error("Runtime KV allocator returned fewer bytes than requested");
    }
    if (allocation_.device != device) {
        throw std::runtime_error("Runtime KV allocator returned memory on the wrong device");
    }
    if (allocation_.alignment < kRuntimeKvAlignment ||
        (reinterpret_cast<std::uintptr_t>(allocation_.pointer) % kRuntimeKvAlignment) != 0) {
        throw std::runtime_error("Runtime KV allocator returned an under-aligned allocation");
    }

    allocation_id_ = next_allocation_id().fetch_add(1, std::memory_order_relaxed);
    if (allocation_id_ == 0)
        allocation_id_ = next_allocation_id().fetch_add(1, std::memory_order_relaxed);
}

void* RuntimeKvAllocation::span_pointer(std::uint32_t layer, bool value) const {
    if (layer >= layer_count_)
        throw std::out_of_range("Runtime KV layer index is out of range");
    const auto span =
        checked_mul(static_cast<std::uint64_t>(layer), std::uint64_t{2}, "Runtime KV span index") +
        static_cast<std::uint64_t>(value);
    const auto offset = checked_mul(span, layer_span_bytes_, "Runtime KV span offset");
    return static_cast<void*>(static_cast<std::byte*>(allocation_.pointer) + offset);
}

void* RuntimeKvAllocation::key_pointer(std::uint32_t layer) const {
    return span_pointer(layer, false);
}

void* RuntimeKvAllocation::value_pointer(std::uint32_t layer) const {
    return span_pointer(layer, true);
}

} // namespace trtmc
