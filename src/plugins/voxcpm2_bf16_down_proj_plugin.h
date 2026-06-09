#pragma once

// TensorRT plugin for the VoxCPM2 TSLM MLP down projection.

#if TRTMC_HAS_TRT

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>

namespace trtmc {

class VoxCPM2Bf16DownProjPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    VoxCPM2Bf16DownProjPlugin() = default;
    VoxCPM2Bf16DownProjPlugin(const void* data, size_t length);

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(int32_t index,
                                         nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;

    VoxCPM2Bf16DownProjPlugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
                                            nvinfer1::DimsExprs const* inputs,
                                            int32_t nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                         nvinfer1::DynamicPluginTensorDesc const* out,
                         int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
                            nvinfer1::PluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    static constexpr const char* kPLUGIN_NAME = "VoxCPM2Bf16DownProj";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
    int32_t num_inputs_{2};
};

} // namespace trtmc

#endif // TRTMC_HAS_TRT
