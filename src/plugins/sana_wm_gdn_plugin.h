#pragma once

#if TRTMC_HAS_TRT

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>

namespace trtmc {

class SanaWmGdnPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    enum class Mode : int32_t {
        kMain = 0,
        kCamera = 1,
        kMainCombined = 2,
        kMainRawCombined = 3,
        kCameraCombined = 4,
    };

    SanaWmGdnPlugin() = default;
    SanaWmGdnPlugin(Mode mode, bool reverse_output, float eps = 1.0e-6F);
    SanaWmGdnPlugin(Mode mode, bool reverse_output, float eps, int32_t frames, int32_t head_dim,
                    float norm_eps);
    SanaWmGdnPlugin(const void* data, size_t length);

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

    SanaWmGdnPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGdnScan";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool is_main() const noexcept { return mode_ == Mode::kMain; }
    bool is_main_combined() const noexcept { return mode_ == Mode::kMainCombined; }
    bool is_main_raw_combined() const noexcept { return mode_ == Mode::kMainRawCombined; }
    bool is_camera_combined() const noexcept { return mode_ == Mode::kCameraCombined; }

    Mode mode_{Mode::kMain};
    bool reverse_output_{false};
    float eps_{1.0e-6F};
    int32_t frames_{0};
    int32_t head_dim_{0};
    float norm_eps_{1.0e-5F};
    std::string namespace_;
};

class SanaWmUcpePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmUcpePlugin() = default;
    SanaWmUcpePlugin(int32_t frames, int32_t spatial, int32_t heads, int32_t head_dim, bool inverse,
                     bool tree_reduce, bool downscale);
    SanaWmUcpePlugin(const void* data, size_t length);

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

    SanaWmUcpePlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmUcpe";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t heads_{0};
    int32_t head_dim_{0};
    bool inverse_{false};
    bool tree_reduce_{true};
    bool downscale_{false};
    std::string namespace_;
};

} // namespace trtmc

#endif // TRTMC_HAS_TRT
