/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iterator>
#include <limits>
#include <new>
#include <string>

namespace trtmc::wan22 {
namespace {

constexpr uint32_t kSERIALIZATION_MAGIC = 0x57435632U; // "WCV2"
constexpr uint32_t kSERIALIZATION_VERSION = 1U;
constexpr size_t kMAX_WORKSPACE_BYTES = size_t{512} << 20;
constexpr int32_t kTHREADS = 256;

struct Conv3dConfig {
    int32_t batch{};
    int32_t input_channels{};
    int32_t output_channels{};
    int32_t input_depth{};
    int32_t input_height{};
    int32_t input_width{};
};

struct SerializedConfig {
    uint32_t magic;
    uint32_t version;
    Conv3dConfig config;
};

static_assert(sizeof(SerializedConfig) == 32);

bool isSupportedConfig(const Conv3dConfig& config) {
    const bool channels_ok = (config.input_channels == 256 || config.input_channels == 512) &&
                             config.output_channels == 256;
    const bool temporal_ok = config.input_depth == 3 || config.input_depth == 6;
    const bool spatial_ok = (config.input_height == 18 && config.input_width == 18) ||
                            (config.input_height == 354 && config.input_width == 642);
    return config.batch == 1 && channels_ok && temporal_ok && spatial_ok;
}

bool matches(const nvinfer1::Dims& dims, const std::array<int32_t, 5>& expected) {
    if (dims.nbDims != static_cast<int32_t>(expected.size()))
        return false;
    for (int32_t index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] != expected[static_cast<size_t>(index)])
            return false;
    }
    return true;
}

bool matches(const nvinfer1::Dims& dims, const std::array<int32_t, 1>& expected) {
    return dims.nbDims == 1 && dims.d[0] == expected[0];
}

__global__ void addChannelBias(float* output, const float* bias, int64_t element_count,
                               int64_t spatial_volume) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count)
        return;
    const int32_t channel = static_cast<int32_t>((index / spatial_volume) % 256);
    output[index] += bias[channel];
}

const char* algorithmName(cudnnConvolutionFwdAlgo_t algorithm) {
    switch (algorithm) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
        return "IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        return "IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        return "GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        return "DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
        return "FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        return "FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        return "WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
        return "WINOGRAD_NONFUSED";
    default:
        return "UNKNOWN";
    }
}

void reportAlgorithmOnce(const Conv3dConfig& config, cudnnConvolutionFwdAlgo_t algorithm,
                         size_t workspace_bytes) {
    // TensorRT clones plugins several times during build.  Report each static
    // contract once per process so target-local selection remains auditable
    // without emitting hundreds of identical lines.
    static std::array<std::atomic_bool, 8> reported{};
    const size_t spatial_index = config.input_height == 354 ? 4 : 0;
    const size_t channel_index = config.input_channels == 512 ? 2 : 0;
    const size_t temporal_index = config.input_depth == 6 ? 1 : 0;
    const size_t index = spatial_index + channel_index + temporal_index;
    if (reported[index].exchange(true, std::memory_order_relaxed))
        return;
    std::fprintf(stderr,
                 "[Wan22VaeConv3d] target-local cuDNN algorithm=%s(%d) workspace=%zu "
                 "shape=[%d,%d,%d,%d,%d]->[%d,%d,%d,%d,%d]\n",
                 algorithmName(algorithm), static_cast<int>(algorithm), workspace_bytes,
                 config.batch, config.input_channels, config.input_depth, config.input_height,
                 config.input_width, config.batch, config.output_channels, config.input_depth - 2,
                 config.input_height - 2, config.input_width - 2);
}

} // namespace

class VaeConv3dPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22VaeConv3d";
    static constexpr const char* kVERSION = "1";

    explicit VaeConv3dPlugin(Conv3dConfig config) : config_(config) { prepare(); }

    VaeConv3dPlugin(const void* data, size_t length) {
        if (data != nullptr && length == sizeof(SerializedConfig)) {
            SerializedConfig serialized{};
            std::memcpy(&serialized, data, sizeof(serialized));
            if (serialized.magic == kSERIALIZATION_MAGIC &&
                serialized.version == kSERIALIZATION_VERSION) {
                config_ = serialized.config;
            }
        }
        prepare();
    }

    ~VaeConv3dPlugin() override { release(); }

    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return prepare() ? 0 : 1; }
    void terminate() noexcept override { release(); }
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override { return sizeof(SerializedConfig); }
    void serialize(void* buffer) const noexcept override {
        if (buffer == nullptr)
            return;
        const SerializedConfig serialized{kSERIALIZATION_MAGIC, kSERIALIZATION_VERSION, config_};
        std::memcpy(buffer, &serialized, sizeof(serialized));
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kFLOAT;
    }
    VaeConv3dPlugin* clone() const noexcept override {
        auto* result = new (std::nothrow) VaeConv3dPlugin(config_);
        if (result != nullptr)
            result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const*, int32_t,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        nvinfer1::DimsExprs output{};
        output.nbDims = 5;
        output.d[0] = builder.constant(config_.batch);
        output.d[1] = builder.constant(config_.output_channels);
        output.d[2] = builder.constant(config_.input_depth - 2);
        output.d[3] = builder.constant(config_.input_height - 2);
        output.d[4] = builder.constant(config_.input_width - 2);
        return output;
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 3 && output_count == 1 && position >= 0 && position < 4 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
                         nvinfer1::DynamicPluginTensorDesc const* outputs,
                         int32_t output_count) noexcept override {
        configured_ = validateDescriptors(inputs, input_count, outputs, output_count);
    }
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return prepared_ ? workspace_bytes_ : 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        if (!prepared_ && !prepare())
            return 1;
        if (!configured_ || input_desc == nullptr || output_desc == nullptr || inputs == nullptr ||
            outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
            inputs[2] == nullptr || outputs[0] == nullptr ||
            (workspace_bytes_ != 0 && workspace == nullptr))
            return 1;
        if (!validateRuntimeDescriptors(input_desc, output_desc))
            return 1;
        if (cudnnSetStream(handle_, stream) != CUDNN_STATUS_SUCCESS)
            return 1;

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const cudnnStatus_t status = cudnnConvolutionForward(
            handle_, &alpha, input_descriptor_, inputs[0], filter_descriptor_, inputs[1],
            convolution_descriptor_, algorithm_, workspace, workspace_bytes_, &beta,
            output_descriptor_, outputs[0]);
        if (status != CUDNN_STATUS_SUCCESS)
            return 1;

        const int64_t spatial_volume = static_cast<int64_t>(config_.input_depth - 2) *
                                       (config_.input_height - 2) * (config_.input_width - 2);
        const int64_t element_count =
            static_cast<int64_t>(config_.batch) * config_.output_channels * spatial_volume;
        const int64_t block_count = (element_count + kTHREADS - 1) / kTHREADS;
        if (block_count <= 0 || block_count > std::numeric_limits<uint32_t>::max())
            return 1;
        addChannelBias<<<static_cast<uint32_t>(block_count), kTHREADS, 0, stream>>>(
            static_cast<float*>(outputs[0]), static_cast<const float*>(inputs[2]), element_count,
            spatial_volume);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    bool validateRuntimeDescriptors(nvinfer1::PluginTensorDesc const* inputs,
                                    nvinfer1::PluginTensorDesc const* outputs) const noexcept {
        const std::array<int32_t, 5> activation{config_.batch, config_.input_channels,
                                                config_.input_depth, config_.input_height,
                                                config_.input_width};
        const std::array<int32_t, 5> weight{config_.output_channels, config_.input_channels, 3, 3,
                                            3};
        const std::array<int32_t, 1> bias{config_.output_channels};
        const std::array<int32_t, 5> output{config_.batch, config_.output_channels,
                                            config_.input_depth - 2, config_.input_height - 2,
                                            config_.input_width - 2};
        return isSupportedConfig(config_) && matches(inputs[0].dims, activation) &&
               matches(inputs[1].dims, weight) && matches(inputs[2].dims, bias) &&
               matches(outputs[0].dims, output);
    }

    bool validateDescriptors(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t input_count,
                             nvinfer1::DynamicPluginTensorDesc const* outputs,
                             int32_t output_count) const noexcept {
        if (inputs == nullptr || outputs == nullptr || input_count != 3 || output_count != 1)
            return false;
        const std::array<nvinfer1::PluginTensorDesc, 3> input_desc{inputs[0].desc, inputs[1].desc,
                                                                   inputs[2].desc};
        return validateRuntimeDescriptors(input_desc.data(), &outputs[0].desc);
    }

    bool prepare() noexcept {
        if (prepared_)
            return true;
        release();
        if (!isSupportedConfig(config_))
            return false;
        if (cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS ||
            cudnnCreateTensorDescriptor(&input_descriptor_) != CUDNN_STATUS_SUCCESS ||
            cudnnCreateTensorDescriptor(&output_descriptor_) != CUDNN_STATUS_SUCCESS ||
            cudnnCreateFilterDescriptor(&filter_descriptor_) != CUDNN_STATUS_SUCCESS ||
            cudnnCreateConvolutionDescriptor(&convolution_descriptor_) != CUDNN_STATUS_SUCCESS) {
            release();
            return false;
        }

        const int input_dims[5] = {config_.batch, config_.input_channels, config_.input_depth,
                                   config_.input_height, config_.input_width};
        const int input_strides[5] = {
            config_.input_channels * config_.input_depth * config_.input_height *
                config_.input_width,
            config_.input_depth * config_.input_height * config_.input_width,
            config_.input_height * config_.input_width, config_.input_width, 1};
        const int output_dims[5] = {config_.batch, config_.output_channels, config_.input_depth - 2,
                                    config_.input_height - 2, config_.input_width - 2};
        const int output_strides[5] = {config_.output_channels * output_dims[2] * output_dims[3] *
                                           output_dims[4],
                                       output_dims[2] * output_dims[3] * output_dims[4],
                                       output_dims[3] * output_dims[4], output_dims[4], 1};
        const int filter_dims[5] = {config_.output_channels, config_.input_channels, 3, 3, 3};
        const int pad[3] = {0, 0, 0};
        const int stride[3] = {1, 1, 1};
        const int dilation[3] = {1, 1, 1};
        if (cudnnSetTensorNdDescriptor(input_descriptor_, CUDNN_DATA_FLOAT, 5, input_dims,
                                       input_strides) != CUDNN_STATUS_SUCCESS ||
            cudnnSetTensorNdDescriptor(output_descriptor_, CUDNN_DATA_FLOAT, 5, output_dims,
                                       output_strides) != CUDNN_STATUS_SUCCESS ||
            cudnnSetFilterNdDescriptor(filter_descriptor_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5,
                                       filter_dims) != CUDNN_STATUS_SUCCESS ||
            cudnnSetConvolutionNdDescriptor(convolution_descriptor_, 3, pad, stride, dilation,
                                            CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS ||
            cudnnSetConvolutionMathType(convolution_descriptor_,
                                        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) !=
                CUDNN_STATUS_SUCCESS) {
            release();
            return false;
        }

        std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT> performance{};
        int returned = 0;
        if (cudnnGetConvolutionForwardAlgorithm_v7(handle_, input_descriptor_, filter_descriptor_,
                                                   convolution_descriptor_, output_descriptor_,
                                                   static_cast<int>(performance.size()), &returned,
                                                   performance.data()) != CUDNN_STATUS_SUCCESS) {
            release();
            return false;
        }
        bool selected = false;
        for (int index = 0; index < returned; ++index) {
            const auto& candidate = performance[static_cast<size_t>(index)];
            if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.memory > kMAX_WORKSPACE_BYTES)
                continue;
            size_t queried_workspace = 0;
            if (cudnnGetConvolutionForwardWorkspaceSize(
                    handle_, input_descriptor_, filter_descriptor_, convolution_descriptor_,
                    output_descriptor_, candidate.algo,
                    &queried_workspace) != CUDNN_STATUS_SUCCESS ||
                queried_workspace > kMAX_WORKSPACE_BYTES)
                continue;
            algorithm_ = candidate.algo;
            workspace_bytes_ = std::max(candidate.memory, queried_workspace);
            selected = true;
            break;
        }
        if (!selected) {
            std::fprintf(stderr,
                         "[Wan22VaeConv3d] no target-local cuDNN forward algorithm within "
                         "%zu-byte workspace bound for shape=[%d,%d,%d,%d,%d]\n",
                         kMAX_WORKSPACE_BYTES, config_.batch, config_.input_channels,
                         config_.input_depth, config_.input_height, config_.input_width);
            release();
            return false;
        }
        prepared_ = true;
        reportAlgorithmOnce(config_, algorithm_, workspace_bytes_);
        return true;
    }

    void release() noexcept {
        prepared_ = false;
        workspace_bytes_ = 0;
        if (convolution_descriptor_ != nullptr) {
            cudnnDestroyConvolutionDescriptor(convolution_descriptor_);
            convolution_descriptor_ = nullptr;
        }
        if (filter_descriptor_ != nullptr) {
            cudnnDestroyFilterDescriptor(filter_descriptor_);
            filter_descriptor_ = nullptr;
        }
        if (output_descriptor_ != nullptr) {
            cudnnDestroyTensorDescriptor(output_descriptor_);
            output_descriptor_ = nullptr;
        }
        if (input_descriptor_ != nullptr) {
            cudnnDestroyTensorDescriptor(input_descriptor_);
            input_descriptor_ = nullptr;
        }
        if (handle_ != nullptr) {
            cudnnDestroy(handle_);
            handle_ = nullptr;
        }
    }

    Conv3dConfig config_{};
    cudnnHandle_t handle_{};
    cudnnTensorDescriptor_t input_descriptor_{};
    cudnnTensorDescriptor_t output_descriptor_{};
    cudnnFilterDescriptor_t filter_descriptor_{};
    cudnnConvolutionDescriptor_t convolution_descriptor_{};
    cudnnConvolutionFwdAlgo_t algorithm_{CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};
    size_t workspace_bytes_{};
    bool prepared_{};
    bool configured_{true};
    std::string namespace_;
};

class VaeConv3dCreator final : public nvinfer1::IPluginCreator {
  public:
    VaeConv3dCreator() {
        fields_[0] = {"batch", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_[1] = {"input_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_[2] = {"output_channels", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_[3] = {"input_depth", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_[4] = {"input_height", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_[5] = {"input_width", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        collection_.nbFields = static_cast<int32_t>(fields_.size());
        collection_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override { return VaeConv3dPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override { return VaeConv3dPlugin::kVERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &collection_;
    }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* collection) noexcept override {
        Conv3dConfig config{};
        bool seen[6]{};
        if (collection == nullptr)
            return nullptr;
        for (int32_t index = 0; index < collection->nbFields; ++index) {
            const nvinfer1::PluginField& field = collection->fields[index];
            if (field.name == nullptr || field.data == nullptr || field.length != 1 ||
                field.type != nvinfer1::PluginFieldType::kINT32)
                return nullptr;
            const int32_t value = *static_cast<const int32_t*>(field.data);
            if (std::strcmp(field.name, "batch") == 0) {
                config.batch = value;
                seen[0] = true;
            } else if (std::strcmp(field.name, "input_channels") == 0) {
                config.input_channels = value;
                seen[1] = true;
            } else if (std::strcmp(field.name, "output_channels") == 0) {
                config.output_channels = value;
                seen[2] = true;
            } else if (std::strcmp(field.name, "input_depth") == 0) {
                config.input_depth = value;
                seen[3] = true;
            } else if (std::strcmp(field.name, "input_height") == 0) {
                config.input_height = value;
                seen[4] = true;
            } else if (std::strcmp(field.name, "input_width") == 0) {
                config.input_width = value;
                seen[5] = true;
            } else {
                return nullptr;
            }
        }
        if (!std::all_of(std::begin(seen), std::end(seen), [](bool value) { return value; }) ||
            !isSupportedConfig(config))
            return nullptr;
        return new (std::nothrow) VaeConv3dPlugin(config);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        auto* plugin = new (std::nothrow) VaeConv3dPlugin(data, length);
        return plugin;
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    std::array<nvinfer1::PluginField, 6> fields_{};
    nvinfer1::PluginFieldCollection collection_{};
    std::string namespace_;
};

} // namespace trtmc::wan22

static nvinfer1::PluginRegistrar<trtmc::wan22::VaeConv3dCreator>
    plugin_registrar_wan22_vae_conv3d{};
