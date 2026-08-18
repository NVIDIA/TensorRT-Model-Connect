/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

namespace trtmc::sam2_hoi {

class CudaStream final {
  public:
    CudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~CudaStream() {
        if (stream_ != nullptr)
            (void)cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    bool ok() const noexcept { return status_ == cudaSuccess; }
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
    cudaError_t status_{cudaSuccess};
};

} // namespace trtmc::sam2_hoi
