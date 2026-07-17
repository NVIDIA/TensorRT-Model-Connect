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

__global__ void gated_update_kernel(const float* update, const float* gate, float* output) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < kElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = update[index] * gate[index % kColumns];
}

__global__ void residual_add_kernel(const float* residual, float* output) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < kElements; index += static_cast<int64_t>(blockDim.x) * gridDim.x)
        output[index] = residual[index] + output[index];
}

int32_t launch_gated_residual(const float* residual, const float* update, const float* gate,
                              float* output, int32_t rows, int32_t columns,
                              cudaStream_t stream) {
    if (residual == nullptr || update == nullptr || gate == nullptr || output == nullptr ||
        rows != kRows || columns != kColumns)
        return 1;
    constexpr int32_t threads = 256;
    const int64_t required_blocks = (kElements + threads - 1) / threads;
    const int32_t blocks = static_cast<int32_t>(std::min<int64_t>(required_blocks, 65'535));
    // Upstream eager materializes the multiplication before the residual add.
    // Separate kernels prevent TensorRT from contracting the expression to FMA.
    gated_update_kernel<<<blocks, threads, 0, stream>>>(update, gate, output);
    residual_add_kernel<<<blocks, threads, 0, stream>>>(residual, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace

class DitGatedResidualFp32Plugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitGatedResidualFp32";
    static constexpr const char* kVERSION = "1";

    DitGatedResidualFp32Plugin() = default;
    DitGatedResidualFp32Plugin(const void*, size_t) {}
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
    DitGatedResidualFp32Plugin* clone() const noexcept override {
        auto* result = new DitGatedResidualFp32Plugin();
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
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr)
            return 1;
        const auto& residual = input_desc[0].dims;
        const auto& update = input_desc[1].dims;
        const auto& gate = input_desc[2].dims;
        if (residual.nbDims != 2 || residual.d[0] != kRows || residual.d[1] != kColumns ||
            update.nbDims != 2 || update.d[0] != kRows || update.d[1] != kColumns ||
            gate.nbDims != 2 || gate.d[0] != 1 || gate.d[1] != kColumns)
            return 1;
        return launch_gated_residual(
            static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), static_cast<float*>(outputs[0]), kRows,
            kColumns, stream);
    }

  private:
    std::string namespace_;
};

class DitGatedResidualFp32Creator final : public nvinfer1::IPluginCreator {
  public:
    DitGatedResidualFp32Creator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override {
        return DitGatedResidualFp32Plugin::kNAME;
    }
    char const* getPluginVersion() const noexcept override {
        return DitGatedResidualFp32Plugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitGatedResidualFp32Plugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitGatedResidualFp32Plugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override { namespace_ = value ? value : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace trtmc::wan22

extern "C" int trtmc_wan22_dit_gated_residual_fp32_launch(
    const float* residual, const float* update, const float* gate, float* output, int32_t rows,
    int32_t columns, void* stream) {
    return trtmc::wan22::launch_gated_residual(residual, update, gate, output, rows, columns,
                                               static_cast<cudaStream_t>(stream));
}

static nvinfer1::PluginRegistrar<trtmc::wan22::DitGatedResidualFp32Creator>
    plugin_registrar_wan22_dit_gated_residual_fp32{};
