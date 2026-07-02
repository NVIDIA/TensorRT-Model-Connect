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

} // namespace trtmc
