/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <string>

namespace trtmc::wan22 {
namespace {

constexpr int32_t kTokenRows = 27'280;
constexpr int32_t kTextRows = 512;
constexpr int32_t kColumns = 3'072;
constexpr int32_t kWarpSize = 32;
constexpr int32_t kVectorSize = 4;
constexpr int32_t kVectors = kColumns / kVectorSize;
constexpr float kEpsilon = 1.0e-6F;

__device__ __forceinline__ float fp32(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 bf16(float value) {
    return __float2bfloat16_rn(value);
}

// Match PyTorch 2.12 Reduce.cuh's contiguous FP32 mean for [27280,3072].
// Each warp reduces one row.  Four vector lanes are accumulated independently,
// combined in element order, and then reduced with shuffle-down additions.
__global__ void rms_norm_fp32_kernel(const __nv_bfloat16* input, const float* gamma, float* output,
                                     float* optional_means, int32_t rows) {
    const int32_t row = static_cast<int32_t>(blockIdx.x * blockDim.y + threadIdx.y);
    if (row >= rows)
        return;
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    const __nv_bfloat16* row_input = input + static_cast<int64_t>(row) * kColumns;
    float* row_output = output + static_cast<int64_t>(row) * kColumns;
    float sums[kVectorSize] = {0.0F, 0.0F, 0.0F, 0.0F};

    for (int32_t vector = lane; vector < kVectors; vector += kWarpSize) {
        const int32_t base = vector * kVectorSize;
#pragma unroll
        for (int32_t element = 0; element < kVectorSize; ++element) {
            const float value = fp32(row_input[base + element]);
            const float square = __fmul_rn(value, value);
            sums[element] = __fadd_rn(sums[element], square);
        }
    }

    float sum = sums[0];
#pragma unroll
    for (int32_t element = 1; element < kVectorSize; ++element)
        sum = __fadd_rn(sum, sums[element]);
#pragma unroll
    for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xFFFFFFFFU, sum, offset, kWarpSize);
        sum = __fadd_rn(sum, other);
    }
    sum = __shfl_sync(0xFFFFFFFFU, sum, 0, kWarpSize);

    const float mean = __fmul_rn(sum, 1.0F / kColumns);
    if (lane == 0 && optional_means != nullptr)
        optional_means[row] = mean;
    const float inverse = rsqrtf(__fadd_rn(mean, kEpsilon));
    for (int32_t element = lane; element < kColumns; element += kWarpSize) {
        const __nv_bfloat16 normalized = bf16(__fmul_rn(fp32(row_input[element]), inverse));
        row_output[element] = __fmul_rn(fp32(normalized), gamma[element]);
    }
}

int32_t launch_rms_norm(const __nv_bfloat16* input, const float* gamma, float* output,
                        float* optional_means, int32_t rows, int32_t columns, float epsilon,
                        cudaStream_t stream) {
    if (input == nullptr || gamma == nullptr || output == nullptr ||
        (rows != kTokenRows && rows != kTextRows) || columns != kColumns || epsilon != kEpsilon)
        return 1;
    constexpr int32_t warps_per_block = 16;
    const int32_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    constexpr dim3 threads(kWarpSize, warps_per_block, 1);
    rms_norm_fp32_kernel<<<blocks, threads, 0, stream>>>(input, gamma, output, optional_means,
                                                         rows);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace

class DitRmsNormFp32Plugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitRmsNormFp32";
    static constexpr const char* kVERSION = "1";

    DitRmsNormFp32Plugin() = default;
    DitRmsNormFp32Plugin(const void*, size_t) {}
    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kFLOAT;
    }
    DitRmsNormFp32Plugin* clone() const noexcept override {
        auto* result = new DitRmsNormFp32Plugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        if (input_count != 2 || output_count != 1 || position < 0 || position >= 3 ||
            in_out[position].format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        return position == 0 ? in_out[position].type == nvinfer1::DataType::kBF16
                             : in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& hidden = input_desc[0].dims;
        const auto& gamma = input_desc[1].dims;
        if (hidden.nbDims != 2 || (hidden.d[0] != kTokenRows && hidden.d[0] != kTextRows) ||
            hidden.d[1] != kColumns || gamma.nbDims != 2 || gamma.d[0] != 1 ||
            gamma.d[1] != kColumns)
            return 1;
        return launch_rms_norm(
            static_cast<const __nv_bfloat16*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<float*>(outputs[0]), nullptr, hidden.d[0], kColumns, kEpsilon, stream);
    }

  private:
    std::string namespace_;
};

class DitRmsNormFp32Creator final : public nvinfer1::IPluginCreator {
  public:
    DitRmsNormFp32Creator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return DitRmsNormFp32Plugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return DitRmsNormFp32Plugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitRmsNormFp32Plugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitRmsNormFp32Plugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22

extern "C" int trtmc_wan22_dit_rms_norm_fp32_launch(const __nv_bfloat16* input, const float* gamma,
                                                    float* output, float* optional_means,
                                                    int32_t rows, int32_t columns, float epsilon,
                                                    void* stream) {
    return trtmc::wan22::launch_rms_norm(input, gamma, output, optional_means, rows, columns,
                                         epsilon, static_cast<cudaStream_t>(stream));
}

static nvinfer1::PluginRegistrar<trtmc::wan22::DitRmsNormFp32Creator>
    plugin_registrar_wan22_dit_rms_norm_fp32{};
