/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// CUDA RAII wrappers — no TRT dependency.
// CudaStream and CudaBuffer with move semantics.

#include <cstddef>
#include <cuda_runtime_api.h>

namespace trtmc {

class CudaStream final {
  public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    bool ok() const;
    cudaStream_t get() const;

  private:
    cudaStream_t mStream{nullptr};
    cudaError_t mStatus{cudaSuccess};
};

class CudaBuffer final {
  public:
    explicit CudaBuffer(std::size_t bytes);
    ~CudaBuffer();
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
    bool ok() const;
    void* data() const;
    std::size_t size() const;

  private:
    void* mPtr{nullptr};
    std::size_t mBytes{0};
    cudaError_t mStatus{cudaSuccess};
};

/// Let stream-ordered allocations (cudaMallocAsync, which TensorRT's default allocator uses) on an
/// integrated GPU reach the device's whole memory.
///
/// A memory pool created with maxSize 0, the default pool included, holds about a third of the
/// memory on Jetson AGX Thor (40.5 of 122.8 GiB), while cudaMalloc can still use the rest: two
/// engines of 27 and 29 GiB do not deserialize in one process. On an integrated device whose
/// current pool is still the default one, this makes a pool with maxSize = the device memory the
/// current pool; the pool lives until the process exits. Thread-safe and idempotent; a device
/// that is not integrated, or already has another current pool, is left alone.
/// @param device CUDA device ordinal.
/// @return true when the device's current pool was replaced by this call.
bool use_full_size_mem_pool(int device);

} // namespace trtmc
