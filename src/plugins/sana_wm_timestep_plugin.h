#pragma once

#if TRTMC_HAS_TRT

#include <NvInferRuntime.h>
#include <cublasLt.h>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class SanaWmTimestepEmbedPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmTimestepEmbedPlugin(int32_t frequency_dim, int32_t hidden_size,
                              std::vector<float> freqs,
                              std::vector<uint16_t> w0, std::vector<uint16_t> b0,
                              std::vector<uint16_t> w1, std::vector<uint16_t> b1,
                              std::vector<uint16_t> w2, std::vector<uint16_t> b2);
    SanaWmTimestepEmbedPlugin(const void* data, size_t length);
    ~SanaWmTimestepEmbedPlugin() override = default;

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
    SanaWmTimestepEmbedPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmTimestepEmbed";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frequency_dim_{0};
    int32_t hidden_size_{0};
    std::vector<float> freqs_;
    std::vector<uint16_t> w0_;
    std::vector<uint16_t> b0_;
    std::vector<uint16_t> w1_;
    std::vector<uint16_t> b1_;
    std::vector<uint16_t> w2_;
    std::vector<uint16_t> b2_;
    cublasLtHandle_t lt_handle_{nullptr};
    void* device_w0_{nullptr};
    void* device_freqs_{nullptr};
    void* device_b0_{nullptr};
    void* device_w1_{nullptr};
    void* device_b1_{nullptr};
    void* device_w2_{nullptr};
    void* device_b2_{nullptr};
    std::string namespace_;
};

} // namespace trtmc

#endif // TRTMC_HAS_TRT
