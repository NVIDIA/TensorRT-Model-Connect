/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/primitives/cuda_common.h"

#include <mutex>

namespace trtmc {

CudaStream::CudaStream() {
    mStatus = cudaStreamCreate(&mStream);
}

CudaStream::~CudaStream() {
    if (mStream != nullptr) {
        cudaStreamDestroy(mStream);
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : mStream(other.mStream), mStatus(other.mStatus) {
    other.mStream = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (mStream != nullptr) {
            cudaStreamDestroy(mStream);
        }
        mStream = other.mStream;
        mStatus = other.mStatus;
        other.mStream = nullptr;
    }
    return *this;
}

bool CudaStream::ok() const {
    return mStatus == cudaSuccess;
}

cudaStream_t CudaStream::get() const {
    return mStream;
}

CudaBuffer::CudaBuffer(std::size_t bytes) : mBytes(bytes) {
    if (mBytes == 0) {
        return;
    }
    mStatus = cudaMalloc(&mPtr, mBytes);
}

CudaBuffer::~CudaBuffer() {
    if (mPtr != nullptr) {
        cudaFree(mPtr);
    }
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : mPtr(other.mPtr), mBytes(other.mBytes), mStatus(other.mStatus) {
    other.mPtr = nullptr;
    other.mBytes = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        if (mPtr != nullptr) {
            cudaFree(mPtr);
        }
        mPtr = other.mPtr;
        mBytes = other.mBytes;
        mStatus = other.mStatus;
        other.mPtr = nullptr;
        other.mBytes = 0;
    }
    return *this;
}

bool CudaBuffer::ok() const {
    return mStatus == cudaSuccess;
}

void* CudaBuffer::data() const {
    return mPtr;
}

std::size_t CudaBuffer::size() const {
    return mBytes;
}

bool use_full_size_mem_pool(int device) {
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    int integrated = 0;
    cudaMemPool_t current = nullptr;
    cudaMemPool_t fallback = nullptr;
    cudaDeviceProp properties{};
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess ||
        integrated == 0 || cudaDeviceGetMemPool(&current, device) != cudaSuccess ||
        cudaDeviceGetDefaultMemPool(&fallback, device) != cudaSuccess || current != fallback ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        // Leave no sticky error from a query the caller did not make.
        cudaGetLastError();
        return false;
    }
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    props.maxSize = properties.totalGlobalMem;
    cudaMemPool_t pool = nullptr;
    if (cudaMemPoolCreate(&pool, &props) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaDeviceSetMemPool(device, pool) != cudaSuccess) {
        cudaMemPoolDestroy(pool);
        cudaGetLastError();
        return false;
    }
    return true;
}

} // namespace trtmc
