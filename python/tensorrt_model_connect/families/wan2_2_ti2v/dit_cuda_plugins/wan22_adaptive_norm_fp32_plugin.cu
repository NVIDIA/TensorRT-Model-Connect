/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace trtmc::wan22 {
namespace {

constexpr int32_t kRows = 27'280;
constexpr int32_t kColumns = 3'072;
constexpr int64_t kElements = static_cast<int64_t>(kRows) * kColumns;
constexpr size_t kWorkspaceBytes = static_cast<size_t>(kColumns) * sizeof(float);

__global__ void scale_plus_one_kernel(const float* scale, float* scale_plus_one) {
    for (int32_t column = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         column < kColumns; column += static_cast<int32_t>(blockDim.x * gridDim.x))
        scale_plus_one[column] = scale[column] + 1.0F;
}

__global__ void adaptive_multiply_kernel(const float* normalized, const float* scale_plus_one,
                                         float* output) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < kElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = normalized[index] * scale_plus_one[index % kColumns];
}

__global__ void adaptive_add_kernel(const float* shift, float* output) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < kElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = output[index] + shift[index % kColumns];
}

int32_t launch_adaptive_norm(const float* normalized, const float* shift, const float* scale,
                             float* output, void* workspace, int32_t rows, int32_t columns,
                             cudaStream_t stream) {
    if (normalized == nullptr || shift == nullptr || scale == nullptr || output == nullptr ||
        workspace == nullptr || rows != kRows || columns != kColumns)
        return 1;
    auto* scale_plus_one = static_cast<float*>(workspace);
    constexpr int32_t threads = 256;
    constexpr int32_t scale_blocks = (kColumns + threads - 1) / threads;
    const int64_t required_blocks = (kElements + threads - 1) / threads;
    const int32_t blocks = static_cast<int32_t>(std::min<int64_t>(required_blocks, 65'535));

    // Preserve the three separate FP32 materialization boundaries in upstream
    // eager execution.  In particular, do not contract the multiply and final
    // add into an FMA; that contraction is the measured TensorRT divergence.
    scale_plus_one_kernel<<<scale_blocks, threads, 0, stream>>>(scale, scale_plus_one);
    adaptive_multiply_kernel<<<blocks, threads, 0, stream>>>(normalized, scale_plus_one, output);
    adaptive_add_kernel<<<blocks, threads, 0, stream>>>(shift, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace

class DitAdaptiveNormFp32Plugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitAdaptiveNormFp32";
    static constexpr const char* kVERSION = "1";

    DitAdaptiveNormFp32Plugin() = default;
    DitAdaptiveNormFp32Plugin(const void*, size_t) {}
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
    DitAdaptiveNormFp32Plugin* clone() const noexcept override {
        auto* result = new DitAdaptiveNormFp32Plugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 3 && output_count == 1 && position >= 0 && position < 4 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return kWorkspaceBytes;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& normalized_dims = input_desc[0].dims;
        const auto& shift_dims = input_desc[1].dims;
        const auto& scale_dims = input_desc[2].dims;
        if (normalized_dims.nbDims != 2 || normalized_dims.d[0] != kRows ||
            normalized_dims.d[1] != kColumns || shift_dims.nbDims != 2 || shift_dims.d[0] != 1 ||
            shift_dims.d[1] != kColumns || scale_dims.nbDims != 2 || scale_dims.d[0] != 1 ||
            scale_dims.d[1] != kColumns)
            return 1;
        return launch_adaptive_norm(
            static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), static_cast<float*>(outputs[0]), workspace, kRows,
            kColumns, stream);
    }

  private:
    std::string namespace_;
};

class DitAdaptiveNormFp32Creator final : public nvinfer1::IPluginCreator {
  public:
    DitAdaptiveNormFp32Creator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return DitAdaptiveNormFp32Plugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return DitAdaptiveNormFp32Plugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitAdaptiveNormFp32Plugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitAdaptiveNormFp32Plugin(data, length);
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

extern "C" int trtmc_wan22_dit_adaptive_norm_fp32_launch(const float* normalized,
                                                         const float* shift, const float* scale,
                                                         float* output, void* workspace,
                                                         int32_t rows, int32_t columns,
                                                         void* stream) {
    return trtmc::wan22::launch_adaptive_norm(normalized, shift, scale, output, workspace, rows,
                                              columns, static_cast<cudaStream_t>(stream));
}

static nvinfer1::PluginRegistrar<trtmc::wan22::DitAdaptiveNormFp32Creator>
    plugin_registrar_wan22_dit_adaptive_norm_fp32{};
