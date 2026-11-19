/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/fp32_kernels.h"
#include "families/blip/runtime/fp32_plugin.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace trtmc::blip;
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}
void require_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

class Buffer {
  public:
    Buffer(std::size_t count, const std::vector<float>& values, cudaStream_t stream)
        : values_(values), size_(count) {
        if (!values.empty() && values.size() != count)
            throw std::invalid_argument("test buffer shape mismatch");
        require_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(float)));
        if (!values.empty())
            require_cuda(cudaMemcpyAsync(data_, values_.data(), count * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
    }
    ~Buffer() { cudaFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    float* data() const { return data_; }
    std::vector<float> read(cudaStream_t stream) const {
        std::vector<float> result(size_);
        require_cuda(cudaMemcpyAsync(result.data(), data_, size_ * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
        require_cuda(cudaStreamSynchronize(stream));
        return result;
    }

  private:
    std::vector<float> values_;
    std::size_t size_;
    float* data_{};
};

void normalization(cudaStream_t stream) {
    {
        Buffer input(2, {1e20F, -1e20F}, stream), scale(1, {1.0F}, stream),
            bias(1, {0.375F}, stream), output(2, {}, stream);
        require_cuda(
            norm_fp32(input.data(), scale.data(), bias.data(), output.data(), 2, 1, 1e-5F, stream));
        const auto actual = output.read(stream);
        check(actual == std::vector<float>({0.375F, 0.375F}),
              "empty normalization lanes corrupted single-element rows");
    }
    constexpr int rows = 3;
    for (const int width : {4, 7, 768}) {
        std::vector<float> values(rows * width), gamma(width), bias(width);
        for (int i = 0; i < rows * width; ++i)
            values[i] = std::sin(i * 0.13F) * 0.7F + (i / width) * 0.2F;
        for (int i = 0; i < width; ++i) {
            gamma[i] = 0.5F + (i % 7) * 0.1F;
            bias[i] = (i % 5 - 2) * 0.03F;
        }
        Buffer input(values.size(), values, stream), scale(width, gamma, stream),
            offset(width, bias, stream), output(values.size(), {}, stream);
        require_cuda(norm_fp32(input.data(), scale.data(), offset.data(), output.data(), rows,
                               width, 1e-5F, stream));
        const auto actual = output.read(stream);
        for (int row = 0; row < rows; ++row) {
            double mean = 0.0, variance = 0.0;
            for (int column = 0; column < width; ++column)
                mean += values[row * width + column] / static_cast<double>(width);
            for (int column = 0; column < width; ++column)
                variance += std::pow(values[row * width + column] - mean, 2) / width;
            for (int column = 0; column < width; ++column) {
                const double expected = (values[row * width + column] - mean) /
                                            std::sqrt(variance + 1e-5F) * gamma[column] +
                                        bias[column];
                check(std::abs(actual[row * width + column] - expected) < 3e-6,
                      "normalization mismatch");
            }
        }
        check(norm_fp32(input.data(), scale.data(), offset.data(), output.data(), rows, width,
                        -1.0F, stream) == cudaErrorInvalidValue,
              "negative normalization epsilon accepted");
        check(norm_fp32(input.data(), scale.data(), offset.data(), output.data(), rows, width,
                        std::numeric_limits<float>::quiet_NaN(), stream) == cudaErrorInvalidValue,
              "nonfinite normalization epsilon accepted");
        check(norm_fp32(nullptr, scale.data(), offset.data(), output.data(), rows, width, 1e-5F,
                        stream) == cudaErrorInvalidValue,
              "null normalization input accepted");
    }
}

void softmax_and_gelu(cudaStream_t stream) {
    for (const int width : {4, 7, 577, 1025, 2049}) {
        std::vector<float> values(width * 3);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = std::sin(static_cast<float>(i) * 0.073F) * 3.1F;
        Buffer input(values.size(), values, stream), output(values.size(), {}, stream);
        require_cuda(softmax_fp32(input.data(), output.data(), 3, width, stream));
        const auto actual = output.read(stream);
        for (int row = 0; row < 3; ++row) {
            const auto first = values.begin() + row * width;
            const double maximum = *std::max_element(first, first + width);
            double sum = 0.0;
            for (int column = 0; column < width; ++column)
                sum += std::exp(values[row * width + column] - maximum);
            for (int column = 0; column < width; ++column)
                check(std::abs(actual[row * width + column] -
                               std::exp(values[row * width + column] - maximum) / sum) < 1e-6,
                      "softmax mismatch");
        }
    }
    std::vector<float> values(4001);
    for (int i = 0; i < static_cast<int>(values.size()); ++i)
        values[i] = (i - 2000) * 0.01F;
    Buffer input(values.size(), values, stream), output(values.size(), {}, stream);
    require_cuda(gelu_fp32(input.data(), output.data(), static_cast<int>(values.size()), stream));
    const auto actual = output.read(stream);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double expected = values[i] * 0.5 * (1.0 + std::erf(values[i] / std::sqrt(2.0)));
        check(std::abs(actual[i] - expected) < 1e-6, "exact GELU mismatch");
    }
    check(gelu_fp32(input.data(), nullptr, 1, stream) == cudaErrorInvalidValue,
          "null GELU output accepted");
    check(softmax_fp32(input.data(), output.data(), 1, 0, stream) == cudaErrorInvalidValue,
          "empty softmax dimension accepted");
}

void projection(cudaStream_t stream) {
    auto context = create_linear_context();
    Buffer workspace(kLinearWorkspaceBytes / sizeof(float), {}, stream);
    for (const int width : {4, 7}) {
        constexpr int rows = 3, outputs = 5;
        std::vector<float> input_values(rows * width), weights(outputs * width), biases(outputs);
        for (std::size_t i = 0; i < input_values.size(); ++i)
            input_values[i] = std::sin(static_cast<float>(i) * 0.17F) * 0.31F;
        for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i] = std::cos(static_cast<float>(i) * 0.23F) * 0.63F;
        for (int i = 0; i < outputs; ++i)
            biases[i] = (i - 2) * 0.101F;
        Buffer input(input_values.size(), input_values, stream),
            weight(weights.size(), weights, stream), bias(outputs, biases, stream),
            output(rows * outputs, {}, stream);
        check(linear_fp32(*context, input.data(), weight.data(), bias.data(), output.data(), rows,
                          width, outputs, workspace.data(), kLinearWorkspaceBytes - 1, stream) != 0,
              "undersized GEMM workspace accepted");
        check(linear_fp32(*context, input.data(), weight.data(), bias.data(), output.data(), rows,
                          width, outputs, workspace.data(), kLinearWorkspaceBytes, stream) == 0,
              "projection execution failed after rejected workspace");
        const auto actual = output.read(stream);
        std::vector<float> changed_biases = biases;
        for (auto& value : changed_biases)
            value += 0.233F;
        Buffer changed_bias(outputs, changed_biases, stream);
        check(linear_fp32(*context, input.data(), weight.data(), changed_bias.data(), output.data(),
                          rows, width, outputs, workspace.data(), kLinearWorkspaceBytes,
                          stream) == 0,
              "projection reuse with changed bias failed");
        const auto changed = output.read(stream);
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < outputs; ++column) {
                double expected = biases[column];
                for (int inner = 0; inner < width; ++inner)
                    expected += static_cast<double>(input_values[row * width + inner]) *
                                weights[column * width + inner];
                check(std::abs(actual[row * outputs + column] - expected) < 3e-6,
                      "projection mismatch");
                const double changed_expected = expected - biases[column] + changed_biases[column];
                check(std::abs(changed[row * outputs + column] - changed_expected) < 3e-6,
                      "cached projection used the previous bias pointer");
            }
    }
}

void plugin_contracts() {
    initialize_fp32_kernels();
    auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(
        getPluginRegistry()->getCreator("BlipFp32", "1", ""));
    if (!creator)
        throw std::runtime_error("native plugin creator missing");
    Fp32Plugin original(static_cast<int>(Fp32Operation::Norm), 1e-5F);
    std::unique_ptr<Fp32Plugin> clone(original.clone());
    std::unique_ptr<nvinfer1::IPluginV3> restored(creator->createPlugin(
        "roundtrip", clone->getFieldsToSerialize(), nvinfer1::TensorRTPhase::kRUNTIME));
    check(restored != nullptr, "plugin serialization roundtrip failed");
    std::int32_t bad_operation = 99;
    float epsilon = 1e-5F;
    std::array<nvinfer1::PluginField, 2> fields{
        {{"operation", &bad_operation, nvinfer1::PluginFieldType::kINT32, 1},
         {"epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1}}};
    nvinfer1::PluginFieldCollection invalid{2, fields.data()};
    check(creator->createPlugin("bad", &invalid, nvinfer1::TensorRTPhase::kBUILD) == nullptr,
          "unknown serialized operation accepted");
    fields[0].data = nullptr;
    check(creator->createPlugin("bad", &invalid, nvinfer1::TensorRTPhase::kBUILD) == nullptr,
          "null serialized data accepted");
    std::array<nvinfer1::DynamicPluginTensorDesc, 4> descriptors{};
    for (auto& descriptor : descriptors) {
        descriptor.desc.type = nvinfer1::DataType::kFLOAT;
        descriptor.desc.format = nvinfer1::TensorFormat::kLINEAR;
        descriptor.desc.dims = nvinfer1::Dims2{2, 4};
    }
    descriptors[1].desc.dims = descriptors[2].desc.dims = nvinfer1::Dims2{1, 4};
    check(original.configurePlugin(descriptors.data(), 3, &descriptors[3], 1) == 0,
          "valid plugin shape rejected");
    descriptors[3].desc.dims = nvinfer1::Dims2{1, 4};
    check(original.configurePlugin(descriptors.data(), 3, &descriptors[3], 1) != 0,
          "broadcasted plugin output accepted");
    descriptors[3].desc.dims = nvinfer1::Dims2{2, 4};
    descriptors[0].desc.type = nvinfer1::DataType::kHALF;
    check(!original.supportsFormatCombination(0, descriptors.data(), 3, 1),
          "non-FP32 plugin input accepted");
    descriptors[0].desc.type = nvinfer1::DataType::kFLOAT;
    descriptors[0].desc.dims = nvinfer1::Dims2{INT_MAX, 4};
    descriptors[3].desc.dims = descriptors[0].desc.dims;
    check(original.configurePlugin(descriptors.data(), 3, &descriptors[3], 1) != 0,
          "overflowing plugin dimensions accepted");
    descriptors[0].desc.dims = descriptors[3].desc.dims = nvinfer1::Dims2{65536, 65537};
    descriptors[1].desc.dims = descriptors[2].desc.dims = nvinfer1::Dims2{1, 65537};
    check(original.configurePlugin(descriptors.data(), 3, &descriptors[3], 1) != 0,
          "overflowing volume that wraps to a positive count accepted");
}

void dependencies() {
    std::ifstream maps("/proc/self/maps");
    if (!maps)
        throw std::runtime_error("cannot inspect loaded native libraries");
    std::string line;
    while (std::getline(maps, line))
        for (const auto* library :
             {"libtorch", "libc10", "libtvm_ffi", "libtrtmc_byok", "libpython"})
            check(line.find(library) == std::string::npos, "unexpected native runtime dependency");
}
} // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && devices == 0)) {
            std::cout << "BLIP CUDA kernels require a visible CUDA device\n";
            return 77;
        }
        require_cuda(status);
        plugin_contracts();
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        normalization(stream);
        softmax_and_gelu(stream);
        projection(stream);
        dependencies();
        require_cuda(cudaStreamDestroy(stream));
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (stream)
            cudaStreamDestroy(stream);
        return 1;
    }
}
