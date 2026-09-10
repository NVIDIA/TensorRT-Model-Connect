/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

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

} // namespace t5
} // namespace trtmc
