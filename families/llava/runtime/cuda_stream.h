/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// llava-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class LlavaCudaStream final {
  public:
    LlavaCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~LlavaCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    LlavaCudaStream(const LlavaCudaStream&) = delete;
    LlavaCudaStream& operator=(const LlavaCudaStream&) = delete;
    LlavaCudaStream(LlavaCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    LlavaCudaStream& operator=(LlavaCudaStream&& other) noexcept {
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
