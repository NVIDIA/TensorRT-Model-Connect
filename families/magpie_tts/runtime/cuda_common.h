/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Magpie-owned CUDA RAII helpers.

#include <cstddef>
#include <cuda_runtime_api.h>

namespace trtmc {

class MagpieCudaStream final {
  public:
    MagpieCudaStream();
    ~MagpieCudaStream();
    MagpieCudaStream(const MagpieCudaStream&) = delete;
    MagpieCudaStream& operator=(const MagpieCudaStream&) = delete;
    MagpieCudaStream(MagpieCudaStream&& other) noexcept;
    MagpieCudaStream& operator=(MagpieCudaStream&& other) noexcept;
    bool ok() const;
    cudaStream_t get() const;

  private:
    cudaStream_t stream_{nullptr};
    cudaError_t status_{cudaSuccess};
};

class MagpieCudaBuffer final {
  public:
    explicit MagpieCudaBuffer(std::size_t bytes);
    ~MagpieCudaBuffer();
    MagpieCudaBuffer(const MagpieCudaBuffer&) = delete;
    MagpieCudaBuffer& operator=(const MagpieCudaBuffer&) = delete;
    MagpieCudaBuffer(MagpieCudaBuffer&& other) noexcept;
    MagpieCudaBuffer& operator=(MagpieCudaBuffer&& other) noexcept;
    bool ok() const;
    void* data() const;
    std::size_t size() const;

  private:
    void* ptr_{nullptr};
    std::size_t bytes_{0};
    cudaError_t status_{cudaSuccess};
};

} // namespace trtmc
