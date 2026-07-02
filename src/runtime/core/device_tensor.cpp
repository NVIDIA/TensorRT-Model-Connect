/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/device_tensor.h"

#include <algorithm>
#include <cstring>

namespace trtmc {

DeviceTensor::DeviceTensor(std::vector<int64_t> shape, DType dtype, cudaStream_t stream)
    : shape_(std::move(shape)), dtype_(dtype), stream_(stream) {
    int64_t n = 1;
    for (auto s : shape_)
        n *= s;
    nbytes_ = static_cast<std::size_t>(n) * dtype_size(dtype_);

    if (nbytes_ > 0) {
        auto err = cudaMalloc(&ptr_, nbytes_);
        if (err != cudaSuccess)
            ptr_ = nullptr;
    }
}

DeviceTensor::~DeviceTensor() {
    free();
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : ptr_(other.ptr_), shape_(std::move(other.shape_)), dtype_(other.dtype_),
      nbytes_(other.nbytes_), stream_(other.stream_) {
    other.ptr_ = nullptr;
    other.nbytes_ = 0;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        nbytes_ = other.nbytes_;
        stream_ = other.stream_;
        other.ptr_ = nullptr;
        other.nbytes_ = 0;
    }
    return *this;
}

bool DeviceTensor::copy_from_host(const void* src) {
    if (!ptr_ || !src || nbytes_ == 0)
        return false;
    auto err = cudaMemcpyAsync(ptr_, src, nbytes_, cudaMemcpyHostToDevice, stream_);
    return err == cudaSuccess;
}

bool DeviceTensor::copy_to_host(void* dst) const {
    if (!ptr_ || !dst || nbytes_ == 0)
        return false;
    auto err = cudaMemcpy(dst, ptr_, nbytes_, cudaMemcpyDeviceToHost);
    return err == cudaSuccess;
}

bool DeviceTensor::copy_from(const DeviceTensor& other) {
    if (!ptr_ || !other.ptr_ || nbytes_ == 0)
        return false;
    auto copy_bytes = std::min(nbytes_, other.nbytes_);
    auto err = cudaMemcpyAsync(ptr_, other.ptr_, copy_bytes, cudaMemcpyDeviceToDevice, stream_);
    return err == cudaSuccess;
}

DeviceTensor DeviceTensor::zeros(std::vector<int64_t> shape, DType dtype, cudaStream_t stream) {
    DeviceTensor t(std::move(shape), dtype, stream);
    if (t.ptr_ && t.nbytes_ > 0) {
        cudaMemsetAsync(t.ptr_, 0, t.nbytes_, stream);
    }
    return t;
}

void* DeviceTensor::data() {
    return ptr_;
}
const void* DeviceTensor::data() const {
    return ptr_;
}

int64_t DeviceTensor::numel() const {
    if (shape_.empty())
        return 0;
    int64_t n = 1;
    for (auto s : shape_)
        n *= s;
    return n;
}

std::size_t DeviceTensor::nbytes() const {
    return nbytes_;
}

void DeviceTensor::free() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
    nbytes_ = 0;
}

} // namespace trtmc
