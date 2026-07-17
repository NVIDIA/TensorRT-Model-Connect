/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace trtmc::wan22 {
namespace {

constexpr int32_t kRows = 27'280;
constexpr int32_t kColumns = 3'072;
constexpr float kEpsilon = 1.0e-6F;
constexpr int32_t kVectorSize = 4;

template <typename T, int32_t Size>
struct alignas(sizeof(T) * Size) AlignedVector {
    T values[Size];
};

struct WelfordData {
    float mean;
    float sigma2;
    float count;
};

__device__ __forceinline__ WelfordData welford_online(float value, WelfordData current) {
    const float delta = value - current.mean;
    const float new_count = current.count + 1.0F;
    const float new_mean = current.mean + delta * (1.0F / new_count);
    return {new_mean, current.sigma2 + delta * (value - new_mean), new_count};
}

__device__ __forceinline__ WelfordData welford_combine(WelfordData data_b,
                                                        WelfordData data_a) {
    const float delta = data_b.mean - data_a.mean;
    const float count = data_a.count + data_b.count;
    if (count <= 0.0F)
        return {0.0F, 0.0F, count};
    const float coefficient = 1.0F / count;
    const float n_a = data_a.count * coefficient;
    const float n_b = data_b.count * coefficient;
    const float mean = n_a * data_a.mean + n_b * data_b.mean;
    const float sigma2 = data_a.sigma2 + data_b.sigma2 +
                         delta * delta * data_a.count * n_b;
    return {mean, sigma2, count};
}

__device__ __forceinline__ WelfordData compute_stats(const float* input, float* shared) {
    using Vector = AlignedVector<float, kVectorSize>;
    const auto* input_vectors = reinterpret_cast<const Vector*>(input);
    const int32_t threads = static_cast<int32_t>(blockDim.x * blockDim.y);
    const int32_t thread = static_cast<int32_t>(threadIdx.x + threadIdx.y * blockDim.x);
    constexpr int32_t vector_count = kColumns / kVectorSize;
    WelfordData state{0.0F, 0.0F, 0.0F};
    for (int32_t index = thread; index < vector_count; index += threads) {
        const Vector data = input_vectors[index];
#pragma unroll
        for (int32_t lane = 0; lane < kVectorSize; ++lane)
            state = welford_online(data.values[lane], state);
    }

    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        const WelfordData other{
            __shfl_down_sync(0xFFFFFFFFU, state.mean, offset),
            __shfl_down_sync(0xFFFFFFFFU, state.sigma2, offset),
            __shfl_down_sync(0xFFFFFFFFU, state.count, offset),
        };
        state = welford_combine(state, other);
    }

    float* mean_sigma = shared;
    float* counts = shared + blockDim.y;
    for (int32_t offset = static_cast<int32_t>(blockDim.y / 2); offset > 0; offset >>= 1) {
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
            const int32_t destination = static_cast<int32_t>(threadIdx.y) - offset;
            mean_sigma[2 * destination] = state.mean;
            mean_sigma[2 * destination + 1] = state.sigma2;
            counts[destination] = state.count;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y < offset) {
            const WelfordData other{
                mean_sigma[2 * threadIdx.y],
                mean_sigma[2 * threadIdx.y + 1],
                counts[threadIdx.y],
            };
            state = welford_combine(state, other);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mean_sigma[0] = state.mean;
        mean_sigma[1] = state.sigma2 / static_cast<float>(kColumns);
    }
    __syncthreads();
    return {mean_sigma[0], mean_sigma[1], 0.0F};
}

__global__ void layer_norm_fp32_kernel(const float* input, float* output) {
    extern __shared__ float shared[];
    const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kColumns;
    const float* row = input + row_offset;
    const WelfordData stats = compute_stats(row, shared);
    const float inverse_stddev = ::rsqrt(stats.sigma2 + kEpsilon);

    using Vector = AlignedVector<float, kVectorSize>;
    const auto* input_vectors = reinterpret_cast<const Vector*>(row);
    auto* output_vectors = reinterpret_cast<Vector*>(output + row_offset);
    const int32_t threads = static_cast<int32_t>(blockDim.x * blockDim.y);
    const int32_t thread = static_cast<int32_t>(threadIdx.x + threadIdx.y * blockDim.x);
    constexpr int32_t vector_count = kColumns / kVectorSize;
    for (int32_t index = thread; index < vector_count; index += threads) {
        const Vector data = input_vectors[index];
        Vector result{};
#pragma unroll
        for (int32_t lane = 0; lane < kVectorSize; ++lane)
            result.values[lane] = inverse_stddev * (data.values[lane] - stats.mean);
        output_vectors[index] = result;
    }
}

int32_t launch_layer_norm(const float* input, float* output, int32_t rows, int32_t columns,
                          float epsilon, cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows != kRows || columns != kColumns ||
        epsilon != kEpsilon)
        return 1;
    constexpr dim3 threads(32, 4, 1);
    constexpr int32_t shared_bytes = 4 * 3 / 2 * static_cast<int32_t>(sizeof(float));
    layer_norm_fp32_kernel<<<rows, threads, shared_bytes, stream>>>(input, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace

class DitLayerNormFp32Plugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitLayerNormFp32";
    static constexpr const char* kVERSION = "1";

    DitLayerNormFp32Plugin() = default;
    DitLayerNormFp32Plugin(const void*, size_t) {}
    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void setPluginNamespace(char const* value) noexcept override { namespace_ = value ? value : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kFLOAT;
    }
    DitLayerNormFp32Plugin* clone() const noexcept override {
        auto* result = new DitLayerNormFp32Plugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 1 && output_count == 1 && position >= 0 && position < 2 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& dims = input_desc[0].dims;
        if (dims.nbDims != 2)
            return 1;
        return launch_layer_norm(static_cast<const float*>(inputs[0]),
                                 static_cast<float*>(outputs[0]), dims.d[0], dims.d[1],
                                 kEpsilon, stream);
    }

  private:
    std::string namespace_;
};

class DitLayerNormFp32Creator final : public nvinfer1::IPluginCreator {
  public:
    DitLayerNormFp32Creator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return DitLayerNormFp32Plugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return DitLayerNormFp32Plugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitLayerNormFp32Plugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitLayerNormFp32Plugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override { namespace_ = value ? value : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22

extern "C" int trtmc_wan22_dit_layer_norm_fp32_launch(const float* input, float* output,
                                                       int32_t rows, int32_t columns,
                                                       float epsilon, void* stream) {
    return trtmc::wan22::launch_layer_norm(input, output, rows, columns, epsilon,
                                           static_cast<cudaStream_t>(stream));
}

static nvinfer1::PluginRegistrar<trtmc::wan22::DitLayerNormFp32Creator>
    plugin_registrar_wan22_dit_layer_norm_fp32{};
