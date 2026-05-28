#pragma once

#if TRTMC_HAS_TRT

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>

namespace trtmc {

class SanaWmRopePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmRopePlugin(int32_t frames, int32_t spatial, int32_t heads, int32_t head_dim,
                     bool inverse = false, bool use_double = false, bool output_bf16 = false);
    SanaWmRopePlugin(const void* data, size_t length);
    ~SanaWmRopePlugin() override = default;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void setPluginNamespace(char const* ns) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                         int32_t nbInputs) const noexcept override;
    SanaWmRopePlugin* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmRope";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t heads_{0};
    int32_t head_dim_{0};
    bool inverse_{false};
    bool use_double_{false};
    bool output_bf16_{false};
    std::string namespace_;
};

} // namespace trtmc

#endif // TRTMC_HAS_TRT
