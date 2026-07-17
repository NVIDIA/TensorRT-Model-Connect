/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace trtmc::wan22 {
namespace {

int64_t volume(const nvinfer1::Dims& dims) {
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

} // namespace

class VaeFp32BarrierPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22VaeFp32Barrier";
    static constexpr const char* kVERSION = "1";

    VaeFp32BarrierPlugin() = default;
    VaeFp32BarrierPlugin(const void*, size_t) {}

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
    VaeFp32BarrierPlugin* clone() const noexcept override {
        auto* result = new VaeFp32BarrierPlugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs,
                                            int32_t, nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count,
                                   int32_t output_count) noexcept override {
        return input_count == 1 && output_count == 1 && position >= 0 && position < 2 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*,
                            int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        const int64_t count = volume(input_desc[0].dims);
        if (count <= 0 || inputs == nullptr || outputs == nullptr)
            return 1;
        const cudaError_t status =
            cudaMemcpyAsync(outputs[0], inputs[0], static_cast<size_t>(count) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        return status == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class VaeFp32BarrierCreator final : public nvinfer1::IPluginCreator {
  public:
    VaeFp32BarrierCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return VaeFp32BarrierPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return VaeFp32BarrierPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(
        char const*, nvinfer1::PluginFieldCollection const*) noexcept override {
        return new VaeFp32BarrierPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new VaeFp32BarrierPlugin(data, length);
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

static nvinfer1::PluginRegistrar<trtmc::wan22::VaeFp32BarrierCreator>
    plugin_registrar_wan22_vae_fp32_barrier{};

extern "C" void trtmc_wan22_vae_cuda_plugin_force_link() {}
