/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/domains/text/dynamic_memory/runtime_memory_plan.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace trtmc {

// One runtime-owned, dense K/V allocation shared by every execution role of a
// native decoder. Storage is laid out as:
//
//   layer 0 K | layer 0 V | layer 1 K | layer 1 V | ...
//
// Every span contains exactly R rows. TensorRT binds a read-only history view
// with H <= T <= R (using the unique H=0,T=1 cold sentinel), while admission
// separately proves A=H+Sq <= R.
class RuntimeKvAllocation {
  public:
    RuntimeKvAllocation(
        std::uint32_t layer_count, std::uint64_t capacity_tokens, std::uint64_t row_width,
        std::uint64_t element_bytes, std::uint32_t device, void* stream,
        std::shared_ptr<IRuntimeDeviceAllocator> allocator = make_cuda_runtime_device_allocator());

    RuntimeKvAllocation(const RuntimeKvAllocation&) = delete;
    RuntimeKvAllocation& operator=(const RuntimeKvAllocation&) = delete;
    RuntimeKvAllocation(RuntimeKvAllocation&&) noexcept = default;
    RuntimeKvAllocation& operator=(RuntimeKvAllocation&&) noexcept = default;

    void* key_pointer(std::uint32_t layer) const;
    void* value_pointer(std::uint32_t layer) const;

    std::uint64_t layer_span_bytes() const { return layer_span_bytes_; }
    std::uint64_t row_bytes() const { return row_bytes_; }
    std::uint64_t total_bytes() const { return allocation_.bytes; }
    std::uint64_t capacity_tokens() const { return capacity_tokens_; }
    std::uint64_t row_width() const { return row_width_; }
    std::uint32_t layer_count() const { return layer_count_; }
    std::uint32_t device() const { return allocation_.device; }
    std::uint64_t alignment() const { return allocation_.alignment; }
    std::uint64_t allocation_id() const { return allocation_id_; }
    std::uint64_t base_address() const {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(allocation_.pointer));
    }
    const std::shared_ptr<void>& lifetime() const { return allocation_.owner; }
    bool valid() const { return allocation_.valid(); }

  private:
    void* span_pointer(std::uint32_t layer, bool value) const;

    RuntimeDeviceAllocation allocation_;
    std::uint32_t layer_count_{0};
    std::uint64_t capacity_tokens_{0};
    std::uint64_t row_width_{0};
    std::uint64_t row_bytes_{0};
    std::uint64_t layer_span_bytes_{0};
    std::uint64_t allocation_id_{0};
};

} // namespace trtmc
