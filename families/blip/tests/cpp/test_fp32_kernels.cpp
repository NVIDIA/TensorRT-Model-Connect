/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/fp32_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <utility>
#include <vector>

namespace {
int failures = 0;

void require_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

template <typename Function>
void rejects(Function call, const char* message) {
    bool rejected = false;
    try {
        call();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}

struct Buffer {
    std::vector<std::int64_t> shape;
    std::vector<float> host_values;
    DLTensor tensor{};

    Buffer(std::vector<std::int64_t> dimensions, const std::vector<float>& values,
           cudaStream_t stream)
        : shape(std::move(dimensions)), host_values(values) {
        const auto count =
            std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>{});
        if (!values.empty() && static_cast<std::int64_t>(values.size()) != count)
            throw std::invalid_argument("test tensor size mismatch");
        require_cuda(cudaGetDevice(&tensor.device.device_id));
        tensor.device.device_type = kDLCUDA;
        tensor.ndim = static_cast<int>(shape.size());
        tensor.shape = shape.data();
        tensor.dtype = {kDLFloat, 32, 1};
        require_cuda(cudaMalloc(&tensor.data, static_cast<std::size_t>(count) * sizeof(float)));
        if (!values.empty())
            require_cuda(cudaMemcpyAsync(tensor.data, host_values.data(),
                                         values.size() * sizeof(float), cudaMemcpyHostToDevice,
                                         stream));
    }
    ~Buffer() { cudaFree(tensor.data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    tvm::ffi::TensorView view() const { return tvm::ffi::TensorView(&tensor); }
    std::vector<float> read(cudaStream_t stream) const {
        const auto count =
            std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>{});
        std::vector<float> result(static_cast<std::size_t>(count));
        require_cuda(cudaMemcpyAsync(result.data(), tensor.data, result.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
        require_cuda(cudaStreamSynchronize(stream));
        return result;
    }
};

void test_math_and_contracts(cudaStream_t stream) {
    using tvm::ffi::Function;
    using tvm::ffi::TensorView;
    auto norm = Function::GetGlobalRequired("blip.vision.norm_fp32");
    auto linear = Function::GetGlobalRequired("blip.vision.linear_fp32");
    auto softmax = Function::GetGlobalRequired("blip.vision.softmax_fp32");
    auto gelu = Function::GetGlobalRequired("blip.vision.gelu_fp32");
    const std::vector<float> values{1, 2, 3, 4, -1, 1, -1, 1};
    const std::vector<float> scales{1, 2, 1, 0.5F};
    const std::vector<float> biases{0.1F, 0.2F, -0.1F, 0};
    Buffer x({2, 4}, values, stream), gamma({4}, scales, stream), beta({4}, biases, stream);
    Buffer out({2, 4}, {}, stream);
    norm(x.view(), gamma.view(), beta.view(), out.view(), 1e-5);
    auto actual = out.read(stream);
    for (int row = 0; row < 2; ++row) {
        double mean = 0, variance = 0;
        for (int col = 0; col < 4; ++col)
            mean += values[row * 4 + col] / 4.0;
        for (int col = 0; col < 4; ++col)
            variance += std::pow(values[row * 4 + col] - mean, 2) / 4;
        for (int col = 0; col < 4; ++col) {
            const auto expected =
                (values[row * 4 + col] - mean) / std::sqrt(variance + 1e-5) * scales[col] +
                biases[col];
            check(std::abs(actual[row * 4 + col] - expected) < 1e-6, "normalization mismatch");
        }
    }
    const std::vector<float> weights{1, 0, -1, 2, 0.5F, 1, 0, -1, 0, 2, 1, 0};
    const std::vector<float> offsets{0.25F, -0.5F, 1};
    Buffer w({3, 4}, weights, stream), b({3}, offsets, stream), projected({2, 3}, {}, stream);
    linear(x.view(), w.view(), b.view(), projected.view());
    actual = projected.read(stream);
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 3; ++col) {
            double expected = offsets[col];
            for (int k = 0; k < 4; ++k)
                expected += values[row * 4 + k] * weights[col * 4 + k];
            check(std::abs(actual[row * 3 + col] - expected) < 1e-6, "linear mismatch");
        }
    softmax(x.view(), out.view());
    actual = out.read(stream);
    for (int row = 0; row < 2; ++row) {
        double sum = 0;
        for (int col = 0; col < 4; ++col)
            sum += std::exp(values[row * 4 + col]);
        for (int col = 0; col < 4; ++col)
            check(std::abs(actual[row * 4 + col] - std::exp(values[row * 4 + col]) / sum) < 1e-6,
                  "softmax mismatch");
    }
    gelu(x.view(), out.view());
    actual = out.read(stream);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto expected = values[i] * 0.5 * (1 + std::erf(values[i] / std::sqrt(2.0)));
        check(std::abs(actual[i] - expected) < 1e-6, "GELU mismatch");
    }
    Buffer wrong_shape({1, 4}, {}, stream), wrong_scale({3}, {1, 1, 1}, stream);
    rejects([&] { norm(x.view(), gamma.view(), beta.view(), wrong_shape.view(), 1e-5); },
            "broadcasted normalization output was accepted");
    rejects([&] { norm(x.view(), wrong_scale.view(), beta.view(), out.view(), 1e-5); },
            "wrong normalization width was accepted");
    rejects([&] { norm(x.view(), gamma.view(), beta.view(), out.view(), -1.0); },
            "negative normalization epsilon was accepted");
    rejects(
        [&] {
            norm(x.view(), gamma.view(), beta.view(), out.view(),
                 std::numeric_limits<double>::quiet_NaN());
        },
        "nonfinite normalization epsilon was accepted");
    rejects([&] { linear(x.view(), w.view(), wrong_scale.view(), wrong_shape.view()); },
            "wrong linear output was accepted");
    auto invalid = x.tensor;
    invalid.dtype.bits = 16;
    rejects([&] { gelu(TensorView(&invalid), out.view()); }, "non-FP32 input was accepted");
    invalid = x.tensor;
    std::int64_t strides[]{5, 1};
    invalid.strides = strides;
    rejects([&] { softmax(TensorView(&invalid), out.view()); }, "strided input was accepted");
    invalid = out.tensor;
    invalid.device.device_id += 1;
    rejects([&] { gelu(x.view(), TensorView(&invalid)); }, "different-device output was accepted");
    invalid = out.tensor;
    invalid.data = nullptr;
    rejects([&] { gelu(x.view(), TensorView(&invalid)); }, "null output was accepted");
    norm(x.view(), gamma.view(), beta.view(), out.view(), 1e-5);
    actual = out.read(stream);
    check(
        std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }),
        "valid request failed after rejected requests");
}
} // namespace

int main() {
    cudaStream_t stream = nullptr;
    TVMFFIStreamHandle previous = nullptr;
    int device = 0;
    try {
        require_cuda(cudaGetDevice(&device));
        require_cuda(cudaStreamCreate(&stream));
        if (TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous) != 0)
            throw std::runtime_error("failed to set the kernel stream");
        trtmc::blip::initialize_fp32_kernels();
        test_math_and_contracts(stream);
        if (TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr) != 0)
            throw std::runtime_error("failed to restore the kernel stream");
        require_cuda(cudaStreamDestroy(stream));
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr);
        if (stream)
            cudaStreamDestroy(stream);
        return 1;
    }
}
