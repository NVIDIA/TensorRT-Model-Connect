/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_kv_allocation.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace {

class HostAllocator final : public trtmc::IRuntimeDeviceAllocator {
  public:
    trtmc::RuntimeDeviceAllocation allocate(std::uint64_t bytes, std::uint64_t alignment,
                                            std::uint32_t device, void*) override {
        void* pointer = nullptr;
        if (posix_memalign(&pointer, static_cast<std::size_t>(alignment),
                           static_cast<std::size_t>(bytes)) != 0) {
            throw std::bad_alloc();
        }
        auto owner = std::shared_ptr<void>(pointer, std::free);
        return {pointer, bytes, device, alignment, std::move(owner)};
    }
};

class WrongDeviceAllocator final : public trtmc::IRuntimeDeviceAllocator {
  public:
    trtmc::RuntimeDeviceAllocation allocate(std::uint64_t bytes, std::uint64_t alignment,
                                            std::uint32_t device, void*) override {
        void* pointer = nullptr;
        if (posix_memalign(&pointer, static_cast<std::size_t>(alignment),
                           static_cast<std::size_t>(bytes)) != 0) {
            throw std::bad_alloc();
        }
        auto owner = std::shared_ptr<void>(pointer, std::free);
        return {pointer, bytes, device + 1, alignment, std::move(owner)};
    }
};

template <typename Fn>
void expect_invalid(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    auto allocator = std::make_shared<HostAllocator>();
    trtmc::RuntimeKvAllocation slab(/*layers=*/3, /*R=*/17, /*width=*/128,
                                    /*element_bytes=*/2, /*device=*/4,
                                    /*stream=*/nullptr, allocator);

    assert(slab.valid());
    assert(slab.row_bytes() == 256);
    assert(slab.layer_span_bytes() == 4352);
    assert(slab.total_bytes() == 26112);
    assert(slab.capacity_tokens() == 17);
    assert(slab.layer_count() == 3);
    assert(slab.device() == 4);
    assert(slab.allocation_id() != 0);
    assert(reinterpret_cast<std::uintptr_t>(slab.key_pointer(0)) % 256 == 0);

    auto* base = static_cast<std::byte*>(slab.key_pointer(0));
    assert(static_cast<std::byte*>(slab.value_pointer(0)) - base == 4352);
    assert(static_cast<std::byte*>(slab.key_pointer(1)) - base == 8704);
    assert(static_cast<std::byte*>(slab.value_pointer(2)) - base == 21760);
    for (std::uint32_t layer = 0; layer < slab.layer_count(); ++layer) {
        assert(reinterpret_cast<std::uintptr_t>(slab.key_pointer(layer)) % 256 == 0);
        assert(reinterpret_cast<std::uintptr_t>(slab.value_pointer(layer)) % 256 == 0);
    }

    bool range_threw = false;
    try {
        (void)slab.key_pointer(3);
    } catch (const std::out_of_range&) {
        range_threw = true;
    }
    assert(range_threw);

    expect_invalid([&] { trtmc::RuntimeKvAllocation invalid(0, 1, 1, 2, 0, nullptr, allocator); });
    expect_invalid([&] { trtmc::RuntimeKvAllocation invalid(1, 0, 1, 2, 0, nullptr, allocator); });
    expect_invalid([&] { trtmc::RuntimeKvAllocation invalid(1, 1, 0, 2, 0, nullptr, allocator); });
    expect_invalid([&] { trtmc::RuntimeKvAllocation invalid(1, 1, 1, 2, 0, nullptr, allocator); });

    bool wrong_device_threw = false;
    try {
        auto wrong_device_allocator = std::make_shared<WrongDeviceAllocator>();
        trtmc::RuntimeKvAllocation invalid(1, 1, 128, 2, 0, nullptr, wrong_device_allocator);
    } catch (const std::runtime_error&) {
        wrong_device_threw = true;
    }
    assert(wrong_device_threw);

    trtmc::RuntimeKvAllocation second(/*layers=*/1, /*R=*/1, /*width=*/128,
                                      /*element_bytes=*/2, /*device=*/0,
                                      /*stream=*/nullptr, allocator);
    assert(second.allocation_id() != slab.allocation_id());
    return 0;
}
