/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/cosmos3/vae_cache_storage.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::cosmos3 {
namespace {

[[noreturn]] void throw_cuda_error(cudaError_t status, const char* operation) {
    throw std::runtime_error(std::string("Cosmos3 recurrent VAE cache ") + operation +
                             " failed: " + cudaGetErrorString(status));
}

void require_cuda_success(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw_cuda_error(status, operation);
}

std::size_t checked_align_up(std::size_t value) {
    constexpr std::size_t kMask = kVaeCacheAlignment - 1;
    static_assert((kVaeCacheAlignment & kMask) == 0, "VAE cache alignment must be a power of two");
    if (value > std::numeric_limits<std::size_t>::max() - kMask)
        throw std::overflow_error("Cosmos3 recurrent VAE cache layout overflow");
    return (value + kMask) & ~kMask;
}

} // namespace

VaeCacheLayout make_vae_cache_layout(const std::vector<std::size_t>& capacities) {
    if (capacities.empty())
        throw std::invalid_argument("Cosmos3 recurrent VAE cache bank must not be empty");

    VaeCacheLayout layout;
    layout.offsets.reserve(capacities.size());
    std::size_t cursor = 0;
    for (const std::size_t capacity : capacities) {
        if (capacity == 0)
            throw std::invalid_argument("Cosmos3 recurrent VAE cache capacity must be positive");
        cursor = checked_align_up(cursor);
        layout.offsets.push_back(cursor);
        if (capacity > std::numeric_limits<std::size_t>::max() - cursor)
            throw std::overflow_error("Cosmos3 recurrent VAE cache layout overflow");
        cursor += capacity;
    }
    layout.total_bytes = checked_align_up(cursor);
    return layout;
}

VaeCacheMemoryKind select_vae_cache_memory_kind(bool integrated, bool can_map_host_memory,
                                                int compute_capability_major,
                                                int compute_capability_minor) {
    if (!integrated || (compute_capability_major == 11 && compute_capability_minor == 0))
        return VaeCacheMemoryKind::kDevice;
    if (!can_map_host_memory) {
        throw std::runtime_error(
            "Cosmos3 integrated GPU cannot map host memory for recurrent VAE caches");
    }
    return VaeCacheMemoryKind::kMappedHost;
}

VaeCacheBank VaeCacheBank::allocate_for_current_device(const std::vector<std::size_t>& capacities) {
    int device = 0;
    require_cuda_success(cudaGetDevice(&device), "device query");

    int integrated = 0;
    require_cuda_success(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device),
                         "integrated-device query");
    int compute_capability_major = 0;
    require_cuda_success(cudaDeviceGetAttribute(&compute_capability_major,
                                                cudaDevAttrComputeCapabilityMajor, device),
                         "compute-capability-major query");
    int compute_capability_minor = 0;
    require_cuda_success(cudaDeviceGetAttribute(&compute_capability_minor,
                                                cudaDevAttrComputeCapabilityMinor, device),
                         "compute-capability-minor query");
    int can_map_host_memory = 0;
    if (integrated != 0) {
        require_cuda_success(
            cudaDeviceGetAttribute(&can_map_host_memory, cudaDevAttrCanMapHostMemory, device),
            "mapped-host capability query");
    }

    auto layout = make_vae_cache_layout(capacities);
    const auto kind =
        select_vae_cache_memory_kind(integrated != 0, can_map_host_memory != 0,
                                     compute_capability_major, compute_capability_minor);
    return VaeCacheBank(std::move(layout), kind);
}

VaeCacheBank::VaeCacheBank(VaeCacheLayout layout, VaeCacheMemoryKind memory_kind)
    : layout_(std::move(layout)), memory_kind_(memory_kind) {
    if (memory_kind_ == VaeCacheMemoryKind::kDevice) {
        require_cuda_success(cudaMalloc(&allocation_, layout_.total_bytes), "cudaMalloc");
        device_base_ = allocation_;
    } else {
        require_cuda_success(cudaHostAlloc(&allocation_, layout_.total_bytes, cudaHostAllocMapped),
                             "cudaHostAllocMapped");
        const cudaError_t alias_status = cudaHostGetDevicePointer(&device_base_, allocation_, 0);
        if (alias_status != cudaSuccess || device_base_ == nullptr) {
            (void)cudaFreeHost(allocation_);
            allocation_ = nullptr;
            device_base_ = nullptr;
            if (alias_status != cudaSuccess)
                throw_cuda_error(alias_status, "cudaHostGetDevicePointer");
            throw std::runtime_error(
                "Cosmos3 recurrent VAE cache mapped allocation has no CUDA device alias");
        }
    }

    const auto address = reinterpret_cast<std::uintptr_t>(device_base_);
    if ((address & (kVaeCacheAlignment - 1)) != 0) {
        release();
        throw std::runtime_error(
            "Cosmos3 recurrent VAE cache allocation does not meet CUDA alignment");
    }
}

VaeCacheBank::~VaeCacheBank() {
    release();
}

void* VaeCacheBank::device_address(std::size_t index) const {
    if (index >= layout_.offsets.size() || device_base_ == nullptr)
        throw std::out_of_range("Cosmos3 recurrent VAE cache index is out of range");
    auto* bytes = static_cast<unsigned char*>(device_base_);
    return bytes + layout_.offsets[index];
}

void VaeCacheBank::zero_async(cudaStream_t stream) const {
    if (device_base_ == nullptr || layout_.total_bytes == 0)
        throw std::runtime_error("Cosmos3 recurrent VAE cache bank is not allocated");
    require_cuda_success(cudaMemsetAsync(device_base_, 0, layout_.total_bytes, stream),
                         "device memset");
}

void VaeCacheBank::copy_from_async(const VaeCacheBank& source, cudaStream_t stream) const {
    if (device_base_ == nullptr || source.device_base_ == nullptr ||
        layout_.offsets != source.layout_.offsets ||
        layout_.total_bytes != source.layout_.total_bytes) {
        throw std::invalid_argument("Cosmos3 recurrent VAE cache bank copy is incompatible");
    }
    require_cuda_success(cudaMemcpyAsync(device_base_, source.device_base_, layout_.total_bytes,
                                         cudaMemcpyDeviceToDevice, stream),
                         "device-to-device copy");
}

void VaeCacheBank::release() noexcept {
    if (allocation_ != nullptr) {
        if (memory_kind_ == VaeCacheMemoryKind::kMappedHost)
            (void)cudaFreeHost(allocation_);
        else
            (void)cudaFree(allocation_);
    }
}

} // namespace trtmc::cosmos3
