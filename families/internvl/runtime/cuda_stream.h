/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// internvl-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class InternVlCudaStream final {
  public:
    InternVlCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~InternVlCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    InternVlCudaStream(const InternVlCudaStream&) = delete;
    InternVlCudaStream& operator=(const InternVlCudaStream&) = delete;
    InternVlCudaStream(InternVlCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    InternVlCudaStream& operator=(InternVlCudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            status_ = other.status_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    bool ok() const { return status_ == cudaSuccess; }
    cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
    cudaError_t status_{cudaSuccess};
};

} // namespace trtmc
