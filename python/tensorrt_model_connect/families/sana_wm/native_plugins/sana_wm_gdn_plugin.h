/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if TRTMC_HAS_TRT

#include <NvInferRuntime.h>
#include <cstdint>
#include <string>
#if TRTMC_HAS_LIBTORCH_CONV
#include <vector>
#endif

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
                     bool tree_reduce, bool downscale, bool double_rope, bool rope_only);
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
    bool double_rope_{false};
    bool rope_only_{false};
    std::string namespace_;
};

class SanaWmCamPrepPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmCamPrepPlugin() = default;
    SanaWmCamPrepPlugin(int32_t frames, int32_t spatial, int32_t heads, int32_t head_dim,
                        float norm_eps);
    SanaWmCamPrepPlugin(const void* data, size_t length);

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

    SanaWmCamPrepPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmCamPrep";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t heads_{0};
    int32_t head_dim_{0};
    float norm_eps_{1.0e-6F};
    std::string namespace_;
};

#if TRTMC_HAS_LIBTORCH_CONV
int32_t launch_sana_wm_short_conv(void* output, const void* input, const void* weight,
                                  const void* bias, int32_t batch, int32_t frames, int32_t spatial,
                                  int32_t channels, int32_t kernel_size,
                                  cudaStream_t stream) noexcept;
int32_t launch_sana_wm_bias_silu(void* values, const void* bias, int32_t rows, int32_t spatial,
                                 int32_t channels, cudaStream_t stream) noexcept;
int32_t launch_sana_wm_gated_silu(void* output, const void* input, const void* bias, int32_t rows,
                                  int32_t spatial, int32_t hidden, cudaStream_t stream) noexcept;
int32_t launch_sana_wm_t2i_modulate(void* output, const void* input, const void* shift,
                                    const void* scale, int32_t batch, int32_t frames,
                                    int32_t tokens, int32_t hidden, cudaStream_t stream) noexcept;

class SanaWmTorchConv2dPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmTorchConv2dPlugin() = default;
    SanaWmTorchConv2dPlugin(int32_t out_channels, int32_t in_channels, int32_t kernel_h,
                            int32_t kernel_w, int32_t pad_h, int32_t pad_w, int32_t groups,
                            const float* weight, int32_t weight_count, const float* bias,
                            int32_t bias_count);
    SanaWmTorchConv2dPlugin(const void* data, size_t length);

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

    SanaWmTorchConv2dPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmTorchConv2d";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t out_channels_{0};
    int32_t in_channels_{0};
    int32_t kernel_h_{0};
    int32_t kernel_w_{0};
    int32_t pad_h_{0};
    int32_t pad_w_{0};
    int32_t groups_{1};
    std::vector<uint16_t> weight_;
    std::vector<uint16_t> bias_;
    void* weight_device_{nullptr};
    void* bias_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmTorchConv3dPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmTorchConv3dPlugin() = default;
    SanaWmTorchConv3dPlugin(int32_t out_channels, int32_t in_channels, int32_t kernel_t,
                            int32_t kernel_h, int32_t kernel_w, int32_t stride_t, int32_t stride_h,
                            int32_t stride_w, int32_t pad_t, int32_t pad_h, int32_t pad_w,
                            int32_t dilation_t, int32_t dilation_h, int32_t dilation_w,
                            int32_t groups, int32_t output_t, int32_t output_h, int32_t output_w,
                            const float* weight, int32_t weight_count, const float* bias,
                            int32_t bias_count);
    SanaWmTorchConv3dPlugin(const void* data, size_t length);

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

    SanaWmTorchConv3dPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmTorchConv3d";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t out_channels_{0};
    int32_t in_channels_{0};
    int32_t kernel_t_{0};
    int32_t kernel_h_{0};
    int32_t kernel_w_{0};
    int32_t stride_t_{1};
    int32_t stride_h_{1};
    int32_t stride_w_{1};
    int32_t pad_t_{0};
    int32_t pad_h_{0};
    int32_t pad_w_{0};
    int32_t dilation_t_{1};
    int32_t dilation_h_{1};
    int32_t dilation_w_{1};
    int32_t groups_{1};
    int32_t output_t_{0};
    int32_t output_h_{0};
    int32_t output_w_{0};
    std::vector<uint16_t> weight_;
    std::vector<uint16_t> bias_;
    void* weight_device_{nullptr};
    void* bias_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmVaeRmsSiluPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmVaeRmsSiluPlugin() = default;
    explicit SanaWmVaeRmsSiluPlugin(float eps);
    SanaWmVaeRmsSiluPlugin(const void* data, size_t length);

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

    SanaWmVaeRmsSiluPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmVaeRmsSilu";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    float eps_{1.0e-8F};
    std::string namespace_;
};

class SanaWmVaeDenormalizePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmVaeDenormalizePlugin() = default;
    SanaWmVaeDenormalizePlugin(int32_t channels, float scaling_factor, const float* mean,
                               int32_t mean_count, const float* std, int32_t std_count);
    SanaWmVaeDenormalizePlugin(const void* data, size_t length);
    ~SanaWmVaeDenormalizePlugin() override;

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
    SanaWmVaeDenormalizePlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmVaeDenormalize";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t channels_{0};
    float scaling_factor_{1.0F};
    std::vector<uint16_t> mean_;
    std::vector<uint16_t> std_;
    void* mean_device_{nullptr};
    void* std_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmVaeLayerNormPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmVaeLayerNormPlugin() = default;
    SanaWmVaeLayerNormPlugin(int32_t channels, float eps, const float* weight, int32_t weight_count,
                             const float* bias, int32_t bias_count);
    SanaWmVaeLayerNormPlugin(const void* data, size_t length);
    ~SanaWmVaeLayerNormPlugin() override;

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
    SanaWmVaeLayerNormPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmVaeLayerNorm";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t channels_{0};
    float eps_{1.0e-6F};
    std::vector<uint16_t> weight_;
    std::vector<uint16_t> bias_;
    void* weight_device_{nullptr};
    void* bias_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmGlumbconvTempPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmGlumbconvTempPlugin() = default;
    SanaWmGlumbconvTempPlugin(int32_t batch, int32_t frames, int32_t height, int32_t width,
                              int32_t channels, int32_t hidden, int32_t t_kernel,
                              const float* inverted_weight, int32_t inverted_weight_count,
                              const float* inverted_bias, int32_t inverted_bias_count,
                              const float* depth_weight, int32_t depth_weight_count,
                              const float* depth_bias, int32_t depth_bias_count,
                              const float* point_weight, int32_t point_weight_count,
                              const float* t_weight, int32_t t_weight_count);
    SanaWmGlumbconvTempPlugin(const void* data, size_t length);

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

    SanaWmGlumbconvTempPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGlumbconvTemp";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t batch_{0};
    int32_t frames_{0};
    int32_t height_{0};
    int32_t width_{0};
    int32_t channels_{0};
    int32_t hidden_{0};
    int32_t t_kernel_{3};
    std::vector<uint16_t> inverted_weight_;
    std::vector<uint16_t> inverted_bias_;
    std::vector<uint16_t> depth_weight_;
    std::vector<uint16_t> depth_bias_;
    std::vector<uint16_t> point_weight_;
    std::vector<uint16_t> t_weight_;
    void* inverted_weight_device_{nullptr};
    void* inverted_bias_device_{nullptr};
    void* depth_weight_device_{nullptr};
    void* depth_bias_device_{nullptr};
    void* point_weight_device_{nullptr};
    void* t_weight_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmTimestepEmbedPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmTimestepEmbedPlugin() = default;
    SanaWmTimestepEmbedPlugin(int32_t frequency_dim, int32_t hidden_size, const float* freqs,
                              int32_t freqs_count, const float* w0, int32_t w0_count,
                              const float* b0, int32_t b0_count, const float* w1, int32_t w1_count,
                              const float* b1, int32_t b1_count, const float* w2, int32_t w2_count,
                              const float* b2, int32_t b2_count);
    SanaWmTimestepEmbedPlugin(const void* data, size_t length);

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
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t frequency_dim_{0};
    int32_t hidden_size_{0};
    std::vector<float> freqs_;
    std::vector<uint16_t> w0_;
    std::vector<uint16_t> b0_;
    std::vector<uint16_t> w1_;
    std::vector<uint16_t> b1_;
    std::vector<uint16_t> w2_;
    std::vector<uint16_t> b2_;
    void* freqs_device_{nullptr};
    void* w0_device_{nullptr};
    void* b0_device_{nullptr};
    void* w1_device_{nullptr};
    void* b1_device_{nullptr};
    void* w2_device_{nullptr};
    void* b2_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmT2IModulatePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmT2IModulatePlugin() = default;
    SanaWmT2IModulatePlugin(const void* data, size_t length);

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

    SanaWmT2IModulatePlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmT2IModulate";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
};

class SanaWmCaptionEmbedPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmCaptionEmbedPlugin() = default;
    explicit SanaWmCaptionEmbedPlugin(int32_t hidden_size);
    SanaWmCaptionEmbedPlugin(const void* data, size_t length);

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

    SanaWmCaptionEmbedPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmCaptionEmbed";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t hidden_size_{0};
    std::string namespace_;
};

class SanaWmCrossAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmCrossAttentionPlugin() = default;
    explicit SanaWmCrossAttentionPlugin(int32_t num_heads);
    SanaWmCrossAttentionPlugin(const void* data, size_t length);

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

    SanaWmCrossAttentionPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmCrossAttention";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t num_heads_{0};
    std::string namespace_;
};

class SanaWmSoftmaxAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmSoftmaxAttentionPlugin() = default;
    SanaWmSoftmaxAttentionPlugin(int32_t frames, int32_t spatial, int32_t heads, int32_t head_dim,
                                 float norm_eps, bool camera);
    SanaWmSoftmaxAttentionPlugin(const void* data, size_t length);

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

    SanaWmSoftmaxAttentionPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmSoftmaxAttention";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t heads_{0};
    int32_t head_dim_{0};
    float norm_eps_{1.0e-5F};
    bool camera_{false};
    std::string namespace_;
};

class SanaWmTorchCamPrepPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmTorchCamPrepPlugin() = default;
    SanaWmTorchCamPrepPlugin(int32_t frames, int32_t spatial, int32_t heads, int32_t head_dim,
                             float norm_eps);
    SanaWmTorchCamPrepPlugin(const void* data, size_t length);

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

    SanaWmTorchCamPrepPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmTorchCamPrep";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t heads_{0};
    int32_t head_dim_{0};
    float norm_eps_{1.0e-5F};
    std::string namespace_;
};

class SanaWmCameraBetaDiscountPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmCameraBetaDiscountPlugin() = default;
    SanaWmCameraBetaDiscountPlugin(int32_t frames, int32_t spatial, int32_t heads);
    SanaWmCameraBetaDiscountPlugin(const void* data, size_t length);

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

    SanaWmCameraBetaDiscountPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmCameraBetaDiscount";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t heads_{0};
    std::string namespace_;
};

class SanaWmFrameGatePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmFrameGatePlugin() = default;
    SanaWmFrameGatePlugin(const void* data, size_t length);

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

    SanaWmFrameGatePlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmFrameGate";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
};

class SanaWmFrameMeanPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmFrameMeanPlugin() = default;
    SanaWmFrameMeanPlugin(int32_t frames, int32_t spatial);
    SanaWmFrameMeanPlugin(const void* data, size_t length);

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

    SanaWmFrameMeanPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmFrameMean";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frames_{0};
    int32_t spatial_{0};
    std::string namespace_;
};

class SanaWmLayerNormPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLayerNormPlugin() = default;
    explicit SanaWmLayerNormPlugin(float eps);
    SanaWmLayerNormPlugin(const void* data, size_t length);

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

    SanaWmLayerNormPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLayerNorm";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    float eps_{1.0e-6F};
    std::string namespace_;
};

class SanaWmShortConvPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmShortConvPlugin() = default;
    SanaWmShortConvPlugin(int32_t frames, int32_t spatial, int32_t channels, int32_t kernel_size,
                          const float* weight, int32_t weight_count, const float* bias,
                          int32_t bias_count);
    SanaWmShortConvPlugin(const void* data, size_t length);

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

    SanaWmShortConvPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmShortConv";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t frames_{0};
    int32_t spatial_{0};
    int32_t channels_{0};
    int32_t kernel_size_{0};
    std::vector<uint16_t> weight_;
    std::vector<uint16_t> bias_;
    void* weight_device_{nullptr};
    void* bias_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmGateProjPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmGateProjPlugin() = default;
    SanaWmGateProjPlugin(int32_t input_dim, int32_t output_dim, int32_t activation,
                         int32_t use_matmul_bias, const float* weight, int32_t weight_count,
                         const float* bias, int32_t bias_count);
    SanaWmGateProjPlugin(const void* data, size_t length);

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

    SanaWmGateProjPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGateProj";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t input_dim_{0};
    int32_t output_dim_{0};
    int32_t activation_{0};
    int32_t use_matmul_bias_{0};
    std::vector<uint16_t> weight_;
    std::vector<uint16_t> bias_;
    void* weight_device_{nullptr};
    void* bias_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmDecayPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmDecayPlugin() = default;
    SanaWmDecayPlugin(int32_t heads, const float* a_log_values, int32_t a_count);
    SanaWmDecayPlugin(const void* data, size_t length);

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

    SanaWmDecayPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmDecay";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t heads_{0};
    std::vector<float> a_log_values_;
    void* a_log_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmGemmaRmsNormPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmGemmaRmsNormPlugin() = default;
    explicit SanaWmGemmaRmsNormPlugin(float eps);
    SanaWmGemmaRmsNormPlugin(const void* data, size_t length);

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
    SanaWmGemmaRmsNormPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGemmaRmsNorm";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    float eps_{1.0e-6F};
    std::string namespace_;
};

class SanaWmGemmaGatedGeluPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmGemmaGatedGeluPlugin() = default;
    SanaWmGemmaGatedGeluPlugin(const void* data, size_t length);

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
    SanaWmGemmaGatedGeluPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGemmaGatedGelu";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    std::string namespace_;
};

class SanaWmGemmaRopePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmGemmaRopePlugin() = default;
    SanaWmGemmaRopePlugin(int32_t heads, int32_t head_dim, int32_t rotary_dim, bool interleaved);
    SanaWmGemmaRopePlugin(const void* data, size_t length);

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
    SanaWmGemmaRopePlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGemmaRope";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t heads_{0};
    int32_t head_dim_{0};
    int32_t rotary_dim_{0};
    bool interleaved_{false};
    std::string namespace_;
};

class SanaWmGemmaAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmGemmaAttentionPlugin() = default;
    SanaWmGemmaAttentionPlugin(int32_t heads, int32_t kv_heads, int32_t head_dim, float scale);
    SanaWmGemmaAttentionPlugin(const void* data, size_t length);

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
    SanaWmGemmaAttentionPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmGemmaAttention";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t heads_{0};
    int32_t kv_heads_{0};
    int32_t head_dim_{0};
    float scale_{1.0F};
    std::string namespace_;
};

class SanaWmLtxTextNormalizePlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxTextNormalizePlugin() = default;
    SanaWmLtxTextNormalizePlugin(int32_t caption_channels, int32_t layer_count, float scale_factor,
                                 float eps);
    SanaWmLtxTextNormalizePlugin(const void* data, size_t length);

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
    SanaWmLtxTextNormalizePlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxTextNormalize";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t caption_channels_{0};
    int32_t layer_count_{0};
    float scale_factor_{8.0F};
    float eps_{1.0e-6F};
    std::string namespace_;
};

class SanaWmLtxRegisterPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxRegisterPlugin() = default;
    SanaWmLtxRegisterPlugin(int32_t register_count, int32_t hidden_dim, const float* registers,
                            int32_t value_count);
    SanaWmLtxRegisterPlugin(const void* data, size_t length);
    ~SanaWmLtxRegisterPlugin() override;

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
    SanaWmLtxRegisterPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxRegister";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;

    int32_t register_count_{0};
    int32_t hidden_dim_{0};
    std::vector<uint16_t> registers_;
    void* registers_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmLtxConnectorBlockPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxConnectorBlockPlugin() = default;
    SanaWmLtxConnectorBlockPlugin(int32_t hidden_dim, int32_t num_heads, int32_t head_dim,
                                  int32_t ff_dim, const float* packed_weights,
                                  int32_t weight_count);
    SanaWmLtxConnectorBlockPlugin(const void* data, size_t length);
    ~SanaWmLtxConnectorBlockPlugin() override;

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
    SanaWmLtxConnectorBlockPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxConnectorBlock";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;
    std::size_t expectedWeightCount() const noexcept;

    int32_t hidden_dim_{0};
    int32_t num_heads_{0};
    int32_t head_dim_{0};
    int32_t ff_dim_{0};
    std::vector<uint16_t> packed_weights_;
    void* weights_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmLtxRmsNormPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxRmsNormPlugin() = default;
    explicit SanaWmLtxRmsNormPlugin(float eps);
    SanaWmLtxRmsNormPlugin(const void* data, size_t length);

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
    SanaWmLtxRmsNormPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxRmsNorm";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    float eps_{1.0e-6F};
    std::string namespace_;
};

class SanaWmLtxTimestepFrequencyPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxTimestepFrequencyPlugin() = default;
    SanaWmLtxTimestepFrequencyPlugin(int32_t frequency_dim, float max_period);
    SanaWmLtxTimestepFrequencyPlugin(const void* data, size_t length);

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
    SanaWmLtxTimestepFrequencyPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxTimestepFrequency";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    int32_t frequency_dim_{256};
    float max_period_{10000.0F};
    std::string namespace_;
};

class SanaWmLtxVideoBlockPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxVideoBlockPlugin() = default;
    SanaWmLtxVideoBlockPlugin(int32_t hidden_dim, int32_t num_heads, int32_t head_dim,
                              int32_t ff_dim, int32_t context_tokens, bool debug,
                              const float* packed_weights, int32_t weight_count);
    SanaWmLtxVideoBlockPlugin(const void* data, size_t length);
    ~SanaWmLtxVideoBlockPlugin() override;

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
    SanaWmLtxVideoBlockPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxVideoBlock";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;
    std::size_t expectedWeightCount() const noexcept;

    int32_t hidden_dim_{0};
    int32_t num_heads_{0};
    int32_t head_dim_{0};
    int32_t ff_dim_{0};
    int32_t context_tokens_{0};
    bool debug_{false};
    std::vector<uint16_t> packed_weights_;
    void* weights_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};

class SanaWmLtxVideoOutputPlugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    SanaWmLtxVideoOutputPlugin() = default;
    SanaWmLtxVideoOutputPlugin(int32_t hidden_dim, int32_t output_dim, const float* packed_weights,
                               int32_t weight_count);
    SanaWmLtxVideoOutputPlugin(const void* data, size_t length);
    ~SanaWmLtxVideoOutputPlugin() override;

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
    SanaWmLtxVideoOutputPlugin* clone() const noexcept override;
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

    static constexpr const char* kPLUGIN_NAME = "SanaWmLtxVideoOutput";
    static constexpr const char* kPLUGIN_VERSION = "1";

  private:
    bool ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept;
    void releaseDeviceCache() noexcept;
    std::size_t expectedWeightCount() const noexcept;

    int32_t hidden_dim_{0};
    int32_t output_dim_{0};
    std::vector<uint16_t> packed_weights_;
    void* weights_device_{nullptr};
    int32_t cached_device_{-1};
    std::string namespace_;
};
#endif

} // namespace trtmc

#endif // TRTMC_HAS_TRT
