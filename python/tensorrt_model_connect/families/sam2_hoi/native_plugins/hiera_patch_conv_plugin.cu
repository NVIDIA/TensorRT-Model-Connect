/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This fixed-site kernel preserves the exact PyTorch 2.7.1/cuDNN 9.7.1
 * Hiera patch-convolution boundary. Each output accumulates the 147 BF16
 * input/weight products in c, ky, kx order with FP32 fmaf. The unbiased
 * accumulator is rounded to BF16 before a BF16-rounded bias is added and the
 * result is rounded to BF16 again.
 */

#include "hiera_patch_conv_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <initializer_list>

namespace trtmc::sam2_hoi {
namespace {

constexpr int32_t kINPUT_CHANNELS = 3;
constexpr int32_t kINPUT_HEIGHT = 1024;
constexpr int32_t kINPUT_WIDTH = 1024;
constexpr int32_t kOUTPUT_CHANNELS = 96;
constexpr int32_t kOUTPUT_HEIGHT = 256;
constexpr int32_t kOUTPUT_WIDTH = 256;
constexpr int32_t kKERNEL_SIZE = 7;
constexpr int32_t kSTRIDE = 4;
constexpr int32_t kPADDING = 3;
constexpr int32_t kREDUCTION = kINPUT_CHANNELS * kKERNEL_SIZE * kKERNEL_SIZE;
constexpr int32_t kINTERIOR_OUTPUTS_PER_THREAD = 24;
constexpr int32_t kINTERIOR_CHANNEL_GROUPS = kOUTPUT_CHANNELS / kINTERIOR_OUTPUTS_PER_THREAD;
static_assert(kOUTPUT_CHANNELS % kINTERIOR_OUTPUTS_PER_THREAD == 0);

__device__ __forceinline__ void accumulate_interior_patch(const __nv_bfloat16* input,
                                                          const __nv_bfloat16* weight,
                                                          int32_t output_channel_base,
                                                          int32_t output_y, int32_t output_x,
                                                          float* accumulators) {
    const __nv_bfloat16* group_weight = weight + output_channel_base * kREDUCTION;
    const int32_t input_y_base = output_y * kSTRIDE - kPADDING;
    const int32_t input_x_base = output_x * kSTRIDE - kPADDING;

#pragma unroll 1
    for (int32_t input_channel = 0; input_channel < kINPUT_CHANNELS; ++input_channel) {
#pragma unroll 1
        for (int32_t kernel_y = 0; kernel_y < kKERNEL_SIZE; ++kernel_y) {
            const int32_t input_index =
                (input_channel * kINPUT_HEIGHT + input_y_base + kernel_y) * kINPUT_WIDTH +
                input_x_base;
            const __nv_bfloat16* input_row = input + input_index;
            const int32_t reduction_base = (input_channel * kKERNEL_SIZE + kernel_y) * kKERNEL_SIZE;
#pragma unroll
            for (int32_t kernel_x = 0; kernel_x < kKERNEL_SIZE; ++kernel_x) {
                const float input_value = __bfloat162float(input_row[kernel_x]);
#pragma unroll
                for (int32_t output_slot = 0; output_slot < kINTERIOR_OUTPUTS_PER_THREAD;
                     ++output_slot) {
                    const int32_t weight_index =
                        output_slot * kREDUCTION + reduction_base + kernel_x;
                    accumulators[output_slot] =
                        fmaf(input_value, __bfloat162float(group_weight[weight_index]),
                             accumulators[output_slot]);
                }
            }
        }
    }
}

__device__ __forceinline__ void accumulate_boundary_patch(const __nv_bfloat16* input,
                                                          const __nv_bfloat16* weight,
                                                          int32_t output_channel, int32_t output_y,
                                                          int32_t output_x, float* accumulator) {
    const __nv_bfloat16* channel_weight = weight + output_channel * kREDUCTION;

#pragma unroll 1
    for (int32_t input_channel = 0; input_channel < kINPUT_CHANNELS; ++input_channel) {
#pragma unroll 1
        for (int32_t kernel_y = 0; kernel_y < kKERNEL_SIZE; ++kernel_y) {
            const int32_t input_y = output_y * kSTRIDE + kernel_y - kPADDING;
            const int32_t reduction_base = (input_channel * kKERNEL_SIZE + kernel_y) * kKERNEL_SIZE;
#pragma unroll
            for (int32_t kernel_x = 0; kernel_x < kKERNEL_SIZE; ++kernel_x) {
                const int32_t input_x = output_x * kSTRIDE + kernel_x - kPADDING;
                float input_value = 0.0F;
                if (input_y >= 0 && input_y < kINPUT_HEIGHT && input_x >= 0 &&
                    input_x < kINPUT_WIDTH) {
                    const int32_t input_index =
                        (input_channel * kINPUT_HEIGHT + input_y) * kINPUT_WIDTH + input_x;
                    input_value = __bfloat162float(input[input_index]);
                }
                const int32_t weight_index = reduction_base + kernel_x;
                // Invalid padding still contributes fmaf(+0.0F, weight, accumulator).
                // Skipping it would change NaN, infinity, and signed-zero behavior.
                *accumulator =
                    fmaf(input_value, __bfloat162float(channel_weight[weight_index]), *accumulator);
            }
        }
    }
}

__device__ __forceinline__ void store_patch_output(const __nv_bfloat16* bias, __nv_bfloat16* output,
                                                   int32_t output_channel, int32_t output_y,
                                                   int32_t output_x, float accumulator) {
    const int32_t output_index =
        (output_channel * kOUTPUT_HEIGHT + output_y) * kOUTPUT_WIDTH + output_x;
    const __nv_bfloat16 unbiased = __float2bfloat16_rn(accumulator);
    output[output_index] =
        __float2bfloat16_rn(__bfloat162float(unbiased) + __bfloat162float(bias[output_channel]));
}

__global__ void hiera_patch_conv_interior_kernel(const __nv_bfloat16* input,
                                                 const __nv_bfloat16* weight,
                                                 const __nv_bfloat16* bias, __nv_bfloat16* output) {
    const int32_t output_x = static_cast<int32_t>(threadIdx.x) + 1;
    if (output_x >= kOUTPUT_WIDTH)
        return;

    const int32_t output_y = static_cast<int32_t>(blockIdx.y) + 1;
    const int32_t output_channel_base =
        static_cast<int32_t>(blockIdx.x) * kINTERIOR_OUTPUTS_PER_THREAD;
    float accumulators[kINTERIOR_OUTPUTS_PER_THREAD]{};
    accumulate_interior_patch(input, weight, output_channel_base, output_y, output_x, accumulators);
#pragma unroll
    for (int32_t output_slot = 0; output_slot < kINTERIOR_OUTPUTS_PER_THREAD; ++output_slot) {
        store_patch_output(bias, output, output_channel_base + output_slot, output_y, output_x,
                           accumulators[output_slot]);
    }
}

__global__ void hiera_patch_conv_boundary_kernel(const __nv_bfloat16* input,
                                                 const __nv_bfloat16* weight,
                                                 const __nv_bfloat16* bias, __nv_bfloat16* output) {
    int32_t output_y = 0;
    int32_t output_x = static_cast<int32_t>(threadIdx.x);
    if (blockIdx.y != 0) {
        output_y = static_cast<int32_t>(threadIdx.x) + 1;
        output_x = 0;
        if (output_y >= kOUTPUT_HEIGHT)
            return;
    }

    const int32_t output_channel = static_cast<int32_t>(blockIdx.x);
    float accumulator = 0.0F;
    accumulate_boundary_patch(input, weight, output_channel, output_y, output_x, &accumulator);
    store_patch_output(bias, output, output_channel, output_y, output_x, accumulator);
}

bool dimensions_equal(const nvinfer1::Dims& dimensions,
                      const std::initializer_list<int32_t>& expected) {
    if (dimensions.nbDims != static_cast<int32_t>(expected.size()))
        return false;
    int32_t index = 0;
    for (const int32_t value : expected) {
        if (dimensions.d[index++] != value)
            return false;
    }
    return true;
}

} // namespace

HieraPatchConvPlugin::HieraPatchConvPlugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* HieraPatchConvPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* HieraPatchConvPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t HieraPatchConvPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t HieraPatchConvPlugin::initialize() noexcept {
    return 0;
}
void HieraPatchConvPlugin::terminate() noexcept {}
void HieraPatchConvPlugin::destroy() noexcept {
    delete this;
}
std::size_t HieraPatchConvPlugin::getSerializationSize() const noexcept {
    return 0;
}
void HieraPatchConvPlugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void HieraPatchConvPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}
char const* HieraPatchConvPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType HieraPatchConvPlugin::getOutputDataType(int32_t index,
                                                           nvinfer1::DataType const* input_types,
                                                           int32_t num_inputs) const noexcept {
    if (index == 0 && num_inputs == 3 && input_types[0] == nvinfer1::DataType::kBF16 &&
        input_types[1] == nvinfer1::DataType::kBF16 && input_types[2] == nvinfer1::DataType::kBF16)
        return nvinfer1::DataType::kBF16;
    return nvinfer1::DataType::kFLOAT;
}
HieraPatchConvPlugin* HieraPatchConvPlugin::clone() const noexcept {
    auto* cloned = new HieraPatchConvPlugin();
    cloned->setPluginNamespace(namespace_.c_str());
    return cloned;
}
nvinfer1::DimsExprs
HieraPatchConvPlugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                          int32_t num_inputs,
                                          nvinfer1::IExprBuilder& expression_builder) noexcept {
    nvinfer1::DimsExprs output{};
    if (output_index != 0 || num_inputs != 3 || inputs[0].nbDims != 4)
        return output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = expression_builder.constant(kOUTPUT_CHANNELS);
    output.d[2] = expression_builder.constant(kOUTPUT_HEIGHT);
    output.d[3] = expression_builder.constant(kOUTPUT_WIDTH);
    return output;
}
bool HieraPatchConvPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* inputs_outputs, int32_t num_inputs,
    int32_t num_outputs) noexcept {
    if (num_inputs != 3 || num_outputs != 1 || position < 0 || position >= 4)
        return false;
    return inputs_outputs[position].type == nvinfer1::DataType::kBF16 &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}
void HieraPatchConvPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                           int32_t num_inputs,
                                           nvinfer1::DynamicPluginTensorDesc const* outputs,
                                           int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t HieraPatchConvPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                                   int32_t num_inputs,
                                                   nvinfer1::PluginTensorDesc const* outputs,
                                                   int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}
int32_t HieraPatchConvPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                      nvinfer1::PluginTensorDesc const* output_descriptors,
                                      void const* const* inputs, void* const* outputs,
                                      void* workspace, cudaStream_t stream) noexcept {
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr || input_descriptors == nullptr ||
        output_descriptors == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || outputs[0] == nullptr)
        return 1;
    if (input_descriptors[0].type != nvinfer1::DataType::kBF16 ||
        input_descriptors[1].type != nvinfer1::DataType::kBF16 ||
        input_descriptors[2].type != nvinfer1::DataType::kBF16 ||
        output_descriptors[0].type != nvinfer1::DataType::kBF16 ||
        !dimensions_equal(input_descriptors[0].dims, {1, 3, 1024, 1024}) ||
        !dimensions_equal(input_descriptors[1].dims, {96, 3, 7, 7}) ||
        !dimensions_equal(input_descriptors[2].dims, {96}) ||
        !dimensions_equal(output_descriptors[0].dims, {1, 96, 256, 256}))
        return 1;

    constexpr int32_t threads = kOUTPUT_WIDTH;
    const dim3 interior_blocks(kINTERIOR_CHANNEL_GROUPS, kOUTPUT_HEIGHT - 1);
    hiera_patch_conv_interior_kernel<<<interior_blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(inputs[0]), static_cast<const __nv_bfloat16*>(inputs[1]),
        static_cast<const __nv_bfloat16*>(inputs[2]), static_cast<__nv_bfloat16*>(outputs[0]));
    const dim3 boundary_blocks(kOUTPUT_CHANNELS, 2);
    hiera_patch_conv_boundary_kernel<<<boundary_blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(inputs[0]), static_cast<const __nv_bfloat16*>(inputs[1]),
        static_cast<const __nv_bfloat16*>(inputs[2]), static_cast<__nv_bfloat16*>(outputs[0]));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::sam2_hoi
