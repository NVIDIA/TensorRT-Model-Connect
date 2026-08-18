/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hiera_layer_norm_plugin.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trtmc::sam2_hoi {
namespace {

constexpr int32_t kVecSize = 4;
constexpr int32_t kWarp = 32;
constexpr int32_t kWarps = 4;
constexpr float kEpsilon = 1.0e-6F;

template <typename T, int32_t VecSize>
struct alignas(sizeof(T) * VecSize) AlignedVector {
    T val[VecSize];
};

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
    float mean;
    float sigma2;
    if (count > 0.0F) {
        const float coefficient = 1.0F / count;
        const float n_a = data_a.count * coefficient;
        const float n_b = data_b.count * coefficient;
        mean = n_a * data_a.mean + n_b * data_b.mean;
        sigma2 = data_a.sigma2 + data_b.sigma2 + delta * delta * data_a.count * n_b;
    } else {
        mean = 0.0F;
        sigma2 = 0.0F;
    }
    return {mean, sigma2, count};
}

__device__ __forceinline__ WelfordData shuffle_down(const WelfordData& value, int32_t offset) {
    return {
        __shfl_down_sync(0xffffffffU, value.mean, offset, kWarp),
        __shfl_down_sync(0xffffffffU, value.sigma2, offset, kWarp),
        __shfl_down_sync(0xffffffffU, value.count, offset, kWarp),
    };
}

__device__ __forceinline__ WelfordData compute_stats(const float* __restrict__ input, int32_t width,
                                                     float* shared) {
    using Vec = AlignedVector<float, kVecSize>;
    const auto* input_vec = reinterpret_cast<const Vec*>(input);
    const int32_t linear_thread = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t threads = blockDim.x * blockDim.y;
    const int32_t vectors = width / kVecSize;
    WelfordData stats{0.0F, 0.0F, 0.0F};
    for (int32_t index = linear_thread; index < vectors; index += threads) {
        const Vec data = input_vec[index];
#pragma unroll
        for (int32_t lane = 0; lane < kVecSize; ++lane)
            stats = online_sum(data.val[lane], stats);
    }
#pragma unroll
    for (int32_t offset = kWarp / 2; offset > 0; offset >>= 1)
        stats = combine(stats, shuffle_down(stats, offset));

    float* mean_sigma = shared;
    float* counts = shared + blockDim.y;
#pragma unroll
    for (int32_t offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int32_t target = threadIdx.y - offset;
            mean_sigma[2 * target] = stats.mean;
            mean_sigma[2 * target + 1] = stats.sigma2;
            counts[target] = stats.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            const WelfordData other{mean_sigma[2 * threadIdx.y], mean_sigma[2 * threadIdx.y + 1],
                                    counts[threadIdx.y]};
            stats = combine(stats, other);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mean_sigma[0] = stats.mean;
        mean_sigma[1] = stats.sigma2 / static_cast<float>(width);
    }
    __syncthreads();
    return {mean_sigma[0], mean_sigma[1], 0.0F};
}

__global__ void hiera_layer_norm_kernel(int64_t rows, int32_t width,
                                        const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output) {
    const int64_t row = blockIdx.x;
    if (row >= rows)
        return;
    extern __shared__ float shared[];
    const float* row_input = input + row * width;
    const WelfordData stats = compute_stats(row_input, width, shared);
    const float reciprocal_stddev = rsqrtf(stats.sigma2 + kEpsilon);

    using Vec = AlignedVector<float, kVecSize>;
    const auto* input_vec = reinterpret_cast<const Vec*>(row_input);
    const auto* weight_vec = reinterpret_cast<const Vec*>(weight);
    const auto* bias_vec = reinterpret_cast<const Vec*>(bias);
    auto* output_vec = reinterpret_cast<Vec*>(output + row * width);
    const int32_t linear_thread = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t threads = blockDim.x * blockDim.y;
    const int32_t vectors = width / kVecSize;
    for (int32_t index = linear_thread; index < vectors; index += threads) {
        const Vec data = input_vec[index];
        Vec result;
#pragma unroll
        for (int32_t lane = 0; lane < kVecSize; ++lane) {
            result.val[lane] =
                weight_vec[index].val[lane] * (reciprocal_stddev * (data.val[lane] - stats.mean)) +
                bias_vec[index].val[lane];
        }
        output_vec[index] = result;
    }
}

bool allowed_width(int32_t width) {
    return width == 96 || width == 192 || width == 384 || width == 768;
}

bool valid_descriptors(nvinfer1::PluginTensorDesc const* inputs,
                       nvinfer1::PluginTensorDesc const* outputs) {
    if (inputs == nullptr || outputs == nullptr) {
        return false;
    }
    for (int32_t index = 0; index < 3; ++index) {
        if (inputs[index].type != nvinfer1::DataType::kFLOAT ||
            inputs[index].format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
    }
    if (outputs[0].type != nvinfer1::DataType::kFLOAT ||
        outputs[0].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    const auto& input = inputs[0].dims;
    const auto& weight = inputs[1].dims;
    const auto& bias = inputs[2].dims;
    const auto& output = outputs[0].dims;
    if (input.nbDims < 1)
        return false;
    if (output.nbDims != input.nbDims) {
        return false;
    }
    for (int32_t index = 0; index < input.nbDims; ++index) {
        if (input.d[index] != output.d[index]) {
            return false;
        }
    }
    const int32_t width = input.d[input.nbDims - 1];
    if (!allowed_width(width) || weight.nbDims != 1 || weight.d[0] != width || bias.nbDims != 1 ||
        bias.d[0] != width)
        return false;
    int64_t elements = 1;
    for (int32_t index = 0; index < input.nbDims; ++index) {
        if (input.d[index] <= 0)
            return false;
        elements *= input.d[index];
    }
    return elements % width == 0;
}

} // namespace

HieraLayerNormPlugin::HieraLayerNormPlugin(const void* data, std::size_t length) {
    (void)data;
    (void)length;
}
char const* HieraLayerNormPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* HieraLayerNormPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t HieraLayerNormPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t HieraLayerNormPlugin::initialize() noexcept {
    return 0;
}
void HieraLayerNormPlugin::terminate() noexcept {}
void HieraLayerNormPlugin::destroy() noexcept {
    delete this;
}
std::size_t HieraLayerNormPlugin::getSerializationSize() const noexcept {
    return 0;
}
void HieraLayerNormPlugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}
void HieraLayerNormPlugin::setPluginNamespace(char const* value) noexcept {
    namespace_ = value != nullptr ? value : "";
}
char const* HieraLayerNormPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType HieraLayerNormPlugin::getOutputDataType(int32_t index,
                                                           nvinfer1::DataType const* input_types,
                                                           int32_t num_inputs) const noexcept {
    (void)index;
    (void)input_types;
    (void)num_inputs;
    return nvinfer1::DataType::kFLOAT;
}
HieraLayerNormPlugin* HieraLayerNormPlugin::clone() const noexcept {
    auto* plugin = new HieraLayerNormPlugin();
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}
nvinfer1::DimsExprs
HieraLayerNormPlugin::getOutputDimensions(int32_t output_index, nvinfer1::DimsExprs const* inputs,
                                          int32_t num_inputs,
                                          nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    nvinfer1::DimsExprs output{};
    if (output_index == 0 && num_inputs == 3 && inputs != nullptr)
        output = inputs[0];
    return output;
}
bool HieraLayerNormPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* inputs_outputs, int32_t num_inputs,
    int32_t num_outputs) noexcept {
    if (inputs_outputs == nullptr || num_inputs != 3 || num_outputs != 1 || position < 0 ||
        position >= 4)
        return false;
    const auto& descriptor = inputs_outputs[position];
    return descriptor.format == nvinfer1::TensorFormat::kLINEAR &&
           descriptor.type == nvinfer1::DataType::kFLOAT;
}
void HieraLayerNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                           int32_t num_inputs,
                                           nvinfer1::DynamicPluginTensorDesc const* outputs,
                                           int32_t num_outputs) noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
}
std::size_t HieraLayerNormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                                                   int32_t num_inputs,
                                                   nvinfer1::PluginTensorDesc const* outputs,
                                                   int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return 0;
}
int32_t HieraLayerNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                      nvinfer1::PluginTensorDesc const* output_descriptors,
                                      void const* const* inputs, void* const* outputs,
                                      void* workspace, cudaStream_t stream) noexcept {
    (void)workspace;
    if (inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        inputs[2] == nullptr || outputs[0] == nullptr ||
        !valid_descriptors(input_descriptors, output_descriptors))
        return 1;
    const int32_t rank = input_descriptors[0].dims.nbDims;
    const int32_t width = input_descriptors[0].dims.d[rank - 1];
    int64_t elements = 1;
    for (int32_t index = 0; index < rank; ++index)
        elements *= input_descriptors[0].dims.d[index];
    const int64_t rows = elements / width;
    const dim3 threads(kWarp, kWarps, 1);
    constexpr int32_t shared_bytes = kWarps * 3 / 2 * sizeof(float);
    hiera_layer_norm_kernel<<<rows, threads, shared_bytes, stream>>>(
        rows, width, static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
        static_cast<const float*>(inputs[2]), static_cast<float*>(outputs[0]));
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::sam2_hoi
