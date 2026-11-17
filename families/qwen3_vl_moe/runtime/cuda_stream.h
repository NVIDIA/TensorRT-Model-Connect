/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// qwen3_vl_moe-owned CUDA stream RAII helper.

#include <cuda_runtime_api.h>

namespace trtmc {

class Qwen3VlMoeCudaStream final {
  public:
    Qwen3VlMoeCudaStream() { status_ = cudaStreamCreate(&stream_); }
    ~Qwen3VlMoeCudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    Qwen3VlMoeCudaStream(const Qwen3VlMoeCudaStream&) = delete;
    Qwen3VlMoeCudaStream& operator=(const Qwen3VlMoeCudaStream&) = delete;
    Qwen3VlMoeCudaStream(Qwen3VlMoeCudaStream&& other) noexcept
        : stream_(other.stream_), status_(other.status_) {
        other.stream_ = nullptr;
    }
    Qwen3VlMoeCudaStream& operator=(Qwen3VlMoeCudaStream&& other) noexcept {
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
