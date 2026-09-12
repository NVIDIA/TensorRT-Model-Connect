/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace trtmc {
namespace flux {

// Owns one device allocation. Growing it frees the previous allocation before
// attempting the new one and leaves the pointer null on failure, so a failed
// grow can never leave a stale or dangling pointer for the caller to reuse.
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

// A device buffer reused and grown across calls (unlike the per-request
// buffers the other families own): `bytes` tracks the capacity actually
// allocated, not the size of the most recent request.
struct GrowableBuffer {
    DeviceBuffer buf;
    std::size_t bytes = 0;
};

// Grows `gb` to at least `need` bytes. A no-op when it is already large
// enough. On allocation failure the previous (too-small) buffer is already
// gone -- `bytes` is left unchanged so the next call retries the grow rather
// than treating the missing buffer as already sized.
inline void ensure_buf(GrowableBuffer& gb, std::size_t need) {
    if (gb.bytes >= need)
        return;
    if (gb.buf.allocate(need) != cudaSuccess)
        throw std::runtime_error("flux_gpu_matmul: unable to allocate device buffer");
    gb.bytes = need;
}

} // namespace flux
} // namespace trtmc
