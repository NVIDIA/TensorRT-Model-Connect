/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// DeviceTensor: GPU-resident tensor — like torch.Tensor on CUDA.
// Owns its device memory via CudaBuffer. Supports H2D, D2H, D2D transfers.

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

class DeviceTensor {
  public:
    // Allocate GPU memory for the given shape and dtype.
    DeviceTensor(std::vector<int64_t> shape, DType dtype, cudaStream_t stream);

    // Default (empty/moved-from state).
    DeviceTensor() = default;
    ~DeviceTensor();

    // Move-only.
    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&& other) noexcept;
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // --- Transfers ---
    bool copy_from_host(const void* src);
    bool copy_to_host(void* dst) const;
    bool copy_from(const DeviceTensor& other); // D2D

    // --- Static factories ---
    static DeviceTensor zeros(std::vector<int64_t> shape, DType dtype, cudaStream_t stream);

    // --- Access ---
    void* data();
    const void* data() const;
    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    bool ok() const { return ptr_ != nullptr; }

    int64_t numel() const;
    std::size_t nbytes() const;
    cudaStream_t stream() const { return stream_; }

  private:
    void* ptr_{nullptr};
    std::vector<int64_t> shape_;
    DType dtype_{DType::kFloat32};
    std::size_t nbytes_{0};
    cudaStream_t stream_{nullptr};

    void free();
};

using DeviceTensorMap = std::unordered_map<std::string, DeviceTensor*>;

} // namespace trtmc
