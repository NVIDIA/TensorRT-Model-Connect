/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Forward reduction and affine arithmetic are adapted from PyTorch v2.7.1,
 * commit e2d141dbde55c2a4370fac5165b0561b6af4798b:
 * aten/src/ATen/native/cuda/layer_norm_kernel.cu and
 * aten/src/ATen/native/cuda/block_reduce.cuh.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the BSD 3-Clause License.
 */

#include "layer_norm_plugin.h"

#include <cstdint>
#include <cuda_bf16.h>

namespace trtmc::sam2_hoi {
namespace {

constexpr int32_t kWidth = 256;
constexpr int32_t kWarp = 32;
constexpr int32_t kWarps = 4;

struct WelfordData {
    float mean;
    float sigma2;
    float count;
};

__device__ __forceinline__ WelfordData online_sum(float value, const WelfordData& current) {
    const float delta = value - current.mean;
    const float new_count = current.count + 1.0F;
    const float new_mean = current.mean + delta * (1.0F / new_count);
    return {new_mean, current.sigma2 + delta * (value - new_mean), new_count};
}

__device__ __forceinline__ WelfordData combine(const WelfordData data_b, const WelfordData data_a) {
    const float delta = data_b.mean - data_a.mean;
    const float count = data_a.count + data_b.count;
    if (count <= 0.0F)
        return {0.0F, 0.0F, 0.0F};
    const float coefficient = 1.0F / count;
    const float n_a = data_a.count * coefficient;
    const float n_b = data_b.count * coefficient;
    const float mean = n_a * data_a.mean + n_b * data_b.mean;
    const float sigma2 = data_a.sigma2 + data_b.sigma2 + delta * delta * data_a.count * n_b;
    return {mean, sigma2, count};
}

__device__ __forceinline__ WelfordData shuffle_down(const WelfordData& value, int32_t offset) {
    return {
        __shfl_down_sync(0xffffffffU, value.mean, offset, kWarp),
        __shfl_down_sync(0xffffffffU, value.sigma2, offset, kWarp),
        __shfl_down_sync(0xffffffffU, value.count, offset, kWarp),
    };
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__global__ void layer_norm_256_kernel(int64_t rows, const T* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias, float* __restrict__ output) {
    const int64_t row = blockIdx.x;
    if (row >= rows)
        return;
    const int32_t lane = threadIdx.x;
    const int32_t warp = threadIdx.y;
    const int32_t linear_thread = lane + warp * kWarp;
    const T* row_input = input + row * kWidth;

    WelfordData stats{0.0F, 0.0F, 0.0F};
    if (linear_thread < kWidth / 4) {
#pragma unroll
        for (int32_t offset = 0; offset < 4; ++offset) {
            const float value = to_float(row_input[linear_thread * 4 + offset]);
            stats = online_sum(value, stats);
        }
    }
#pragma unroll
    for (int32_t offset = kWarp / 2; offset > 0; offset >>= 1)
        stats = combine(stats, shuffle_down(stats, offset));

    extern __shared__ float shared[];
    float* mean_sigma = shared;
    float* counts = shared + kWarps;
#pragma unroll
    for (int32_t offset = kWarps / 2; offset > 0; offset >>= 1) {
        if (lane == 0 && warp >= offset && warp < 2 * offset) {
            const int32_t target = warp - offset;
            mean_sigma[2 * target] = stats.mean;
            mean_sigma[2 * target + 1] = stats.sigma2;
            counts[target] = stats.count;
        }
        __syncthreads();
        if (lane == 0 && warp < offset) {
            const WelfordData other{mean_sigma[2 * warp], mean_sigma[2 * warp + 1], counts[warp]};
            stats = combine(stats, other);
        }
        __syncthreads();
    }
    if (lane == 0 && warp == 0) {
        mean_sigma[0] = stats.mean;
        mean_sigma[1] = stats.sigma2 / static_cast<float>(kWidth);
    }
    __syncthreads();

    const float mean = mean_sigma[0];
    const float reciprocal_stddev = rsqrtf(mean_sigma[1] + 1.0e-5F);
    if (linear_thread < kWidth / 4) {
#pragma unroll
        for (int32_t offset = 0; offset < 4; ++offset) {
            const int32_t column = linear_thread * 4 + offset;
            const float value = to_float(row_input[column]);
            output[row * kWidth + column] =
                weight[column] * (reciprocal_stddev * (value - mean)) + bias[column];
        }
    }
}

bool valid_dimensions(nvinfer1::PluginTensorDesc const* inputs) {
    if (inputs[0].dims.nbDims < 1 || inputs[0].dims.d[inputs[0].dims.nbDims - 1] != kWidth ||
        inputs[1].dims.nbDims != 1 || inputs[1].dims.d[0] != kWidth || inputs[2].dims.nbDims != 1 ||
        inputs[2].dims.d[0] != kWidth)
        return false;
    int64_t elements = 1;
    for (int32_t index = 0; index < inputs[0].dims.nbDims; ++index) {
        if (inputs[0].dims.d[index] <= 0)
            return false;
        elements *= inputs[0].dims.d[index];
    }
    return elements % kWidth == 0;
}

} // namespace

LayerNorm256Plugin::LayerNorm256Plugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* LayerNorm256Plugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* LayerNorm256Plugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t LayerNorm256Plugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t LayerNorm256Plugin::initialize() noexcept {
    return 0;
}
void LayerNorm256Plugin::terminate() noexcept {}
void LayerNorm256Plugin::destroy() noexcept {
    delete this;
}
std::size_t LayerNorm256Plugin::getSerializationSize() const noexcept {
    return 0;
}
void LayerNorm256Plugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void LayerNorm256Plugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}
char const* LayerNorm256Plugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType LayerNorm256Plugin::getOutputDataType(int32_t index,
                                                         nvinfer1::DataType const* input_types,
                                                         int32_t num_inputs) const noexcept {
    (void)input_types;
    (void)index;
    (void)num_inputs;
    return nvinfer1::DataType::kFLOAT;
}
LayerNorm256Plugin* LayerNorm256Plugin::clone() const noexcept {
    auto* cloned = new LayerNorm256Plugin();
    cloned->setPluginNamespace(namespace_.c_str());
    return cloned;
}
nvinfer1::DimsExprs
LayerNorm256Plugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                        int32_t num_inputs,
                                        nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    nvinfer1::DimsExprs output{};
    if (output_index == 0 && num_inputs == 3)
        output = inputs[0];
    return output;
}
bool LayerNorm256Plugin::supportsFormatCombination(int32_t position,
                                                   nvinfer1::PluginTensorDesc const* inputs_outputs,
                                                   int32_t num_inputs,
                                                   int32_t num_outputs) noexcept {
    if (num_inputs != 3 || num_outputs != 1 || position < 0 || position >= 4)
        return false;
    const auto& descriptor = inputs_outputs[position];
    if (descriptor.format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    return position == 0 ? (descriptor.type == nvinfer1::DataType::kBF16 ||
                            descriptor.type == nvinfer1::DataType::kFLOAT)
                         : descriptor.type == nvinfer1::DataType::kFLOAT;
}
void LayerNorm256Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                         int32_t num_inputs,
                                         nvinfer1::DynamicPluginTensorDesc const* outputs,
                                         int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t LayerNorm256Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                                 int32_t num_inputs,
                                                 nvinfer1::PluginTensorDesc const* outputs,
                                                 int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}
int32_t LayerNorm256Plugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                    nvinfer1::PluginTensorDesc const* output_descriptors,
                                    void const* const* inputs, void* const* outputs,
                                    void* workspace, cudaStream_t stream) noexcept {
    (void)output_descriptors;
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr || !valid_dimensions(input_descriptors))
        return 1;
    int64_t elements = 1;
    for (int32_t index = 0; index < input_descriptors[0].dims.nbDims; ++index)
        elements *= input_descriptors[0].dims.d[index];
    const int64_t rows = elements / kWidth;
    const dim3 threads(kWarp, kWarps, 1);
    constexpr int32_t shared_bytes = kWarps * 3 / 2 * sizeof(float);
    if (input_descriptors[0].type == nvinfer1::DataType::kBF16) {
        layer_norm_256_kernel<<<rows, threads, shared_bytes, stream>>>(
            rows, static_cast<const __nv_bfloat16*>(inputs[0]),
            static_cast<const float*>(inputs[1]), static_cast<const float*>(inputs[2]),
            static_cast<float*>(outputs[0]));
    } else if (input_descriptors[0].type == nvinfer1::DataType::kFLOAT) {
        layer_norm_256_kernel<<<rows, threads, shared_bytes, stream>>>(
            rows, static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), static_cast<float*>(outputs[0]));
    } else {
        return 1;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::sam2_hoi
