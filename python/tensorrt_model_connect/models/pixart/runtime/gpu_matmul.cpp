/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_matmul.h"

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace trtmc {

namespace {

struct DeviceBuffer {
    float* data{nullptr};
    std::size_t bytes{0};
};

bool ensure_capacity(DeviceBuffer& buffer, std::size_t required) {
    if (buffer.bytes >= required)
        return true;
    if (buffer.data != nullptr)
        cudaFree(buffer.data);
    buffer.data = nullptr;
    buffer.bytes = 0;
    if (cudaMalloc(reinterpret_cast<void**>(&buffer.data), required) != cudaSuccess)
        return false;
    buffer.bytes = required;
    return true;
}

void release(DeviceBuffer& buffer) {
    if (buffer.data != nullptr)
        cudaFree(buffer.data);
    buffer = {};
}

bool request_is_valid(const float* lhs, const float* rhs, const float* output, int32_t rows,
                      int32_t inner, int32_t columns) {
    return lhs != nullptr && rhs != nullptr && output != nullptr && rows > 0 && inner > 0 &&
           columns > 0;
}

void add_bias(float* output, const float* bias, int32_t rows, int32_t columns) {
    if (bias == nullptr)
        return;
    for (int32_t row = 0; row < rows; ++row) {
        float* output_row =
            output + static_cast<std::size_t>(row) * static_cast<std::size_t>(columns);
        for (int32_t column = 0; column < columns; ++column)
            output_row[column] += bias[column];
    }
}

} // namespace

struct PixArtGpuMatmul::Impl {
    cublasHandle_t handle{nullptr};
    cudaStream_t stream{nullptr};
    DeviceBuffer lhs;
    DeviceBuffer rhs;
    DeviceBuffer output;
    bool ready{false};

    bool prepare(const float* lhs_data, const float* rhs_data, std::size_t lhs_bytes,
                 std::size_t rhs_bytes, std::size_t output_bytes) {
        if (!ensure_capacity(lhs, lhs_bytes) || !ensure_capacity(rhs, rhs_bytes) ||
            !ensure_capacity(output, output_bytes)) {
            return false;
        }
        if (cudaMemcpyAsync(lhs.data, lhs_data, lhs_bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
            return false;
        }
        if (cudaMemcpyAsync(rhs.data, rhs_data, rhs_bytes, cudaMemcpyHostToDevice, stream) ==
            cudaSuccess) {
            return true;
        }
        cudaStreamSynchronize(stream);
        return false;
    }

    bool multiply(int32_t rows, int32_t inner, int32_t columns) {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, columns, rows, inner, &alpha, rhs.data,
                           columns, lhs.data, inner, &beta, output.data,
                           columns) == CUBLAS_STATUS_SUCCESS;
    }

    bool download(float* host_output, std::size_t output_bytes) {
        const auto copy_status =
            cudaMemcpyAsync(host_output, output.data, output_bytes, cudaMemcpyDeviceToHost, stream);
        const auto sync_status = cudaStreamSynchronize(stream);
        return copy_status == cudaSuccess && sync_status == cudaSuccess;
    }
};

PixArtGpuMatmul::PixArtGpuMatmul() : impl_(std::make_unique<Impl>()) {
    if (cudaStreamCreate(&impl_->stream) != cudaSuccess)
        return;
    if (cublasCreate(&impl_->handle) != CUBLAS_STATUS_SUCCESS)
        return;
    if (cublasSetStream(impl_->handle, impl_->stream) != CUBLAS_STATUS_SUCCESS)
        return;
    if (cublasSetMathMode(impl_->handle, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS)
        return;
    impl_->ready = true;
}

PixArtGpuMatmul::~PixArtGpuMatmul() {
    release(impl_->lhs);
    release(impl_->rhs);
    release(impl_->output);
    if (impl_->handle != nullptr)
        cublasDestroy(impl_->handle);
    if (impl_->stream != nullptr)
        cudaStreamDestroy(impl_->stream);
}

bool PixArtGpuMatmul::run(const float* lhs, const float* rhs, const float* bias, float* output,
                          int32_t rows, int32_t inner, int32_t columns) {
    if (!impl_->ready || !request_is_valid(lhs, rhs, output, rows, inner, columns))
        return false;

    const auto lhs_bytes =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(inner) * sizeof(float);
    const auto rhs_bytes =
        static_cast<std::size_t>(inner) * static_cast<std::size_t>(columns) * sizeof(float);
    const auto output_count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    const auto output_bytes = output_count * sizeof(float);
    if (!impl_->prepare(lhs, rhs, lhs_bytes, rhs_bytes, output_bytes))
        return false;
    if (!impl_->multiply(rows, inner, columns)) {
        cudaStreamSynchronize(impl_->stream);
        return false;
    }
    if (!impl_->download(output, output_bytes))
        return false;

    add_bias(output, bias, rows, columns);
    return true;
}

} // namespace trtmc
