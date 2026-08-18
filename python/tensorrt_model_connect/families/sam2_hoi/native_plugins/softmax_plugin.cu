/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Forward reduction and arithmetic are adapted from PyTorch v2.7.1
 * aten/src/ATen/native/cuda/PersistentSoftmax.cuh at commit
 * e2d141dbde55c2a4370fac5165b0561b6af4798b, BSD-3-Clause.
 * Copyright (c) PyTorch contributors.
 */

#include "softmax_plugin.h"

#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace trtmc::sam2_hoi {
namespace {

template <typename T>
__device__ __forceinline__ float load_float(const T* input, int64_t index) {
    return static_cast<float>(input[index]);
}
template <>
__device__ __forceinline__ float load_float(const __nv_bfloat16* input, int64_t index) {
    return __bfloat162float(input[index]);
}
template <typename T>
__device__ __forceinline__ void store_float(T* output, int64_t index, float value) {
    output[index] = static_cast<T>(value);
}
template <>
__device__ __forceinline__ void store_float(__nv_bfloat16* output, int64_t index, float value) {
    output[index] = __float2bfloat16_rn(value);
}

__device__ __forceinline__ float warp_max(float value, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2) {
        const float other = __shfl_xor_sync(0xffffffffU, value, offset, width);
        value = value > other ? value : other;
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value, int width) {
    for (int offset = width / 2; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffffU, value, offset, width);
    return value;
}

template <typename T, int LOG2_ELEMENTS>
__global__ void softmax_warp_forward(T* output, const T* input, int32_t rows, int32_t elements) {
    constexpr int next_power_of_two = 1 << LOG2_ELEMENTS;
    constexpr int warp_width = next_power_of_two < 32 ? next_power_of_two : 32;
    constexpr int iterations = next_power_of_two / warp_width;
    constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;

    const int32_t first_row =
        (static_cast<int32_t>(blockDim.y) * blockIdx.x + threadIdx.y) * warp_batch;
    int32_t local_rows = rows - first_row;
    if (local_rows > warp_batch)
        local_rows = warp_batch;
    const int32_t lane = threadIdx.x;
    const int64_t row_offset = static_cast<int64_t>(first_row) * elements + lane;

    float values[warp_batch][iterations];
#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
        const int32_t row_elements = batch < local_rows ? elements : 0;
#pragma unroll
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int32_t element = lane + iteration * warp_width;
            values[batch][iteration] =
                element < row_elements
                    ? load_float(input, row_offset + static_cast<int64_t>(batch) * elements +
                                            iteration * warp_width)
                    : -CUDART_INF_F;
        }
    }

    float maxima[warp_batch];
#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
        maxima[batch] = values[batch][0];
#pragma unroll
        for (int iteration = 0; iteration < iterations; ++iteration)
            maxima[batch] =
                maxima[batch] > values[batch][iteration] ? maxima[batch] : values[batch][iteration];
        maxima[batch] = warp_max(maxima[batch], warp_width);
    }

    float sums[warp_batch] = {0.0F};
#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
#pragma unroll
        for (int iteration = 0; iteration < iterations; ++iteration) {
            values[batch][iteration] = expf(values[batch][iteration] - maxima[batch]);
            sums[batch] += values[batch][iteration];
        }
        sums[batch] = warp_sum(sums[batch], warp_width);
    }

#pragma unroll
    for (int batch = 0; batch < warp_batch; ++batch) {
        if (batch >= local_rows)
            break;
#pragma unroll
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const int32_t element = lane + iteration * warp_width;
            if (element < elements) {
                store_float(output,
                            row_offset + static_cast<int64_t>(batch) * elements +
                                iteration * warp_width,
                            values[batch][iteration] / sums[batch]);
            } else {
                break;
            }
        }
    }
}

template <typename T>
int32_t launch_softmax(T* output, const T* input, int32_t rows, int32_t elements,
                       cudaStream_t stream) {
    int32_t log2_elements = 0;
    while ((1 << log2_elements) < elements)
        ++log2_elements;
    const int32_t next_power_of_two = 1 << log2_elements;
    const int32_t warp_width = next_power_of_two < 32 ? next_power_of_two : 32;
    const int32_t warp_batch = next_power_of_two <= 128 ? 2 : 1;
    const int32_t warps_per_block = 128 / warp_width;
    const int32_t rows_per_block = warps_per_block * warp_batch;
    const int32_t blocks = (rows + rows_per_block - 1) / rows_per_block;
    const dim3 threads(warp_width, warps_per_block, 1);
#define TRTMC_LAUNCH_SOFTMAX(LOG2)                                                                 \
    case LOG2:                                                                                     \
        softmax_warp_forward<T, LOG2>                                                              \
            <<<blocks, threads, 0, stream>>>(output, input, rows, elements);                       \
        break
    switch (log2_elements) {
        TRTMC_LAUNCH_SOFTMAX(0);
        TRTMC_LAUNCH_SOFTMAX(1);
        TRTMC_LAUNCH_SOFTMAX(2);
        TRTMC_LAUNCH_SOFTMAX(3);
        TRTMC_LAUNCH_SOFTMAX(4);
        TRTMC_LAUNCH_SOFTMAX(5);
        TRTMC_LAUNCH_SOFTMAX(6);
        TRTMC_LAUNCH_SOFTMAX(7);
        TRTMC_LAUNCH_SOFTMAX(8);
        TRTMC_LAUNCH_SOFTMAX(9);
        TRTMC_LAUNCH_SOFTMAX(10);
        TRTMC_LAUNCH_SOFTMAX(11);
    default:
        return 1;
    }
#undef TRTMC_LAUNCH_SOFTMAX
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

int64_t volume(const nvinfer1::Dims& dimensions) {
    int64_t elements = 1;
    for (int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0)
            return 0;
        elements *= dimensions.d[index];
    }
    return elements;
}

} // namespace

SoftmaxPlugin::SoftmaxPlugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* SoftmaxPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SoftmaxPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SoftmaxPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SoftmaxPlugin::initialize() noexcept {
    return 0;
}
void SoftmaxPlugin::terminate() noexcept {}
void SoftmaxPlugin::destroy() noexcept {
    delete this;
}
std::size_t SoftmaxPlugin::getSerializationSize() const noexcept {
    return 0;
}
void SoftmaxPlugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void SoftmaxPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}
char const* SoftmaxPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SoftmaxPlugin::getOutputDataType(int32_t index,
                                                    nvinfer1::DataType const* input_types,
                                                    int32_t num_inputs) const noexcept {
    return index == 0 && num_inputs == 1 ? input_types[0] : nvinfer1::DataType::kFLOAT;
}
SoftmaxPlugin* SoftmaxPlugin::clone() const noexcept {
    auto* cloned = new SoftmaxPlugin();
    cloned->setPluginNamespace(namespace_.c_str());
    return cloned;
}
nvinfer1::DimsExprs
SoftmaxPlugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                   int32_t num_inputs,
                                   nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    nvinfer1::DimsExprs output{};
    if (output_index == 0 && num_inputs == 1)
        output = inputs[0];
    return output;
}
bool SoftmaxPlugin::supportsFormatCombination(int32_t position,
                                              nvinfer1::PluginTensorDesc const* inputs_outputs,
                                              int32_t num_inputs, int32_t num_outputs) noexcept {
    if (num_inputs != 1 || num_outputs != 1 || position < 0 || position >= 2)
        return false;
    const auto type = inputs_outputs[0].type;
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kBF16) &&
           inputs_outputs[position].type == type &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}
void SoftmaxPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                    int32_t num_inputs,
                                    nvinfer1::DynamicPluginTensorDesc const* outputs,
                                    int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t SoftmaxPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                            int32_t num_inputs,
                                            nvinfer1::PluginTensorDesc const* outputs,
                                            int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}
int32_t SoftmaxPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                               nvinfer1::PluginTensorDesc const* output_descriptors,
                               void const* const* inputs, void* const* outputs, void* workspace,
                               cudaStream_t stream) noexcept {
    (void)output_descriptors;
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr || input_descriptors[0].dims.nbDims < 1)
        return 1;
    const int32_t elements = input_descriptors[0].dims.d[input_descriptors[0].dims.nbDims - 1];
    const int64_t total = volume(input_descriptors[0].dims);
    if (elements <= 0 || elements > 2048 || total <= 0 || total % elements != 0 ||
        total / elements > INT32_MAX)
        return 1;
    const int32_t rows = static_cast<int32_t>(total / elements);
    if (input_descriptors[0].type == nvinfer1::DataType::kFLOAT)
        return launch_softmax(static_cast<float*>(outputs[0]), static_cast<const float*>(inputs[0]),
                              rows, elements, stream);
    if (input_descriptors[0].type == nvinfer1::DataType::kBF16)
        return launch_softmax(static_cast<__nv_bfloat16*>(outputs[0]),
                              static_cast<const __nv_bfloat16*>(inputs[0]), rows, elements, stream);
    return 1;
}

} // namespace trtmc::sam2_hoi
