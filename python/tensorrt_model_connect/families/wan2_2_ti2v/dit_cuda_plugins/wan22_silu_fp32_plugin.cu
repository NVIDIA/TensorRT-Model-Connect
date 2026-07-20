/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace trtmc::wan22 {
namespace {

int64_t silu_volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0)
        return 0;
    int64_t count = 1;
    for (int32_t index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0)
            return 0;
        count *= dims.d[index];
    }
    return count;
}

__global__ void dit_silu_fp32_kernel(const float* input, float* output, int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        // Exact PyTorch FP32 CUDA SiLU operation order: FP32 negation,
        // CUDA device exp, FP32 addition, then precise FP32 division.
        const float x_acc = static_cast<float>(input[index]);
        output[index] = x_acc / (float(1) + ::exp(-x_acc));
    }
}

int32_t silu_launch_blocks(int64_t count, int32_t threads) {
    const int64_t required = (count + threads - 1) / threads;
    return static_cast<int32_t>(required < 65535 ? required : 65535);
}

} // namespace

class DitSiluFp32Plugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitSiluFp32";
    static constexpr const char* kVERSION = "1";

    DitSiluFp32Plugin() = default;
    DitSiluFp32Plugin(const void*, size_t) {}
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
    DitSiluFp32Plugin* clone() const noexcept override {
        auto* result = new DitSiluFp32Plugin();
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
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        const int64_t count = silu_volume(input_desc[0].dims);
        if (count <= 0 || inputs == nullptr || outputs == nullptr)
            return 1;
        constexpr int32_t threads = 256;
        dit_silu_fp32_kernel<<<silu_launch_blocks(count, threads), threads, 0, stream>>>(
            static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), count);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class DitSiluFp32Creator final : public nvinfer1::IPluginCreator {
  public:
    DitSiluFp32Creator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return DitSiluFp32Plugin::kNAME; }
    char const* getPluginVersion() const noexcept override { return DitSiluFp32Plugin::kVERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitSiluFp32Plugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitSiluFp32Plugin(data, length);
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

static nvinfer1::PluginRegistrar<trtmc::wan22::DitSiluFp32Creator>
    plugin_registrar_wan22_dit_silu_fp32{};
