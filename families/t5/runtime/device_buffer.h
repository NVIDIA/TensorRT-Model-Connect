/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>

namespace trtmc {
namespace t5 {

// Owns one device allocation. Holding the cross-attention buffers in these
// rather than in raw pointers means a constructor that throws part way through
// still releases what it already allocated: members that finished constructing
// are destroyed during unwinding, and a raw pointer in a vector is not.
class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    cudaError_t allocate(std::size_t bytes) {
        cudaFree(ptr_);
        ptr_ = nullptr;
        return cudaMalloc(&ptr_, bytes);
    }

    void* get() const { return ptr_; }

  private:
    void* ptr_{nullptr};
};

// The pipeline's own allocation paths, kept here so the test can drive the
// same code the constructor and setup_cross_attention run.
inline void allocate_cross_kv(std::vector<DeviceBuffer>& keys, std::vector<DeviceBuffer>& values,
                              int32_t layers, std::size_t bytes) {
    keys.resize(static_cast<std::size_t>(layers));
    values.resize(static_cast<std::size_t>(layers));
    for (int32_t i = 0; i < layers; ++i) {
        const std::size_t layer = static_cast<std::size_t>(i);
        if (keys[layer].allocate(bytes) != cudaSuccess)
            throw std::runtime_error("T5Pipeline: unable to allocate cross-attention key buffer");
        if (values[layer].allocate(bytes) != cudaSuccess)
            throw std::runtime_error("T5Pipeline: unable to allocate cross-attention value buffer");
    }
}

inline void ensure_encoder_mask(DeviceBuffer& mask, std::size_t bytes) {
    if (mask.get() == nullptr && mask.allocate(bytes) != cudaSuccess)
        throw std::runtime_error("T5Pipeline: unable to allocate encoder mask buffer");
}

} // namespace t5
} // namespace trtmc
