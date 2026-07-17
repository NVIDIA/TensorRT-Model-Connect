/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <vector>

namespace trtmc::wan2_2_ti2v {

// TensorRT device addresses are kept at CUDA's documented minimum allocation
// alignment even when multiple recurrent-cache tensors share one allocation.
inline constexpr std::size_t kVaeCacheAlignment = 256;

enum class VaeCacheMemoryKind {
    kDevice,
    kMappedHost,
};

struct VaeCacheLayout {
    std::vector<std::size_t> offsets;
    std::size_t total_bytes{0};
};

// Pure helpers are public inside this model-owned header so policy, overflow,
// and alignment can be tested without depending on the current CUDA device.
VaeCacheLayout make_vae_cache_layout(const std::vector<std::size_t>& capacities);
VaeCacheMemoryKind select_vae_cache_memory_kind(bool integrated, bool can_map_host_memory);

// One contiguous recurrent-cache bank. Discrete GPUs retain the historical
// cudaMalloc behavior. Integrated GPUs use a mapped pinned system allocation
// and expose only its CUDA device alias to TensorRT.
class VaeCacheBank final {
  public:
    static VaeCacheBank allocate_for_current_device(const std::vector<std::size_t>& capacities);

    VaeCacheBank() = default;
    ~VaeCacheBank();

    VaeCacheBank(VaeCacheBank&& other) noexcept;
    VaeCacheBank& operator=(VaeCacheBank&& other) noexcept;
    VaeCacheBank(const VaeCacheBank&) = delete;
    VaeCacheBank& operator=(const VaeCacheBank&) = delete;

    void* device_address(std::size_t index) const;
    std::size_t capacity(std::size_t index) const;
    std::size_t total_bytes() const { return layout_.total_bytes; }
    std::size_t size() const { return capacities_.size(); }
    VaeCacheMemoryKind memory_kind() const { return memory_kind_; }

    void zero_async(cudaStream_t stream) const;
    void copy_from_async(const VaeCacheBank& source, cudaStream_t stream) const;

  private:
    VaeCacheBank(std::vector<std::size_t> capacities, VaeCacheLayout layout,
                 VaeCacheMemoryKind memory_kind);
    void release() noexcept;

    std::vector<std::size_t> capacities_;
    VaeCacheLayout layout_;
    VaeCacheMemoryKind memory_kind_{VaeCacheMemoryKind::kDevice};
    void* allocation_{nullptr};
    void* device_base_{nullptr};
};

} // namespace trtmc::wan2_2_ti2v
