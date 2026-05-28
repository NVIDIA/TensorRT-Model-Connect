#if TRTMC_HAS_TRT && TRTMC_HAS_CUDNN

#include "plugins/sana_wm_patch_embed_plugin.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace trtmc {
namespace {

template <typename T>
void write_value(char*& ptr, const T& value) {
    std::memcpy(ptr, &value, sizeof(T));
    ptr += sizeof(T);
}

template <typename T>
T read_value(const char*& ptr, const char* end, T fallback = T{}) {
    if (ptr + sizeof(T) > end)
        return fallback;
    T value{};
    std::memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

void write_vector(char*& ptr, const std::vector<uint16_t>& values) {
    const auto size = static_cast<uint64_t>(values.size());
    write_value(ptr, size);
    if (!values.empty()) {
        const std::size_t bytes = values.size() * sizeof(uint16_t);
        std::memcpy(ptr, values.data(), bytes);
        ptr += bytes;
    }
}

std::vector<uint16_t> read_vector(const char*& ptr, const char* end) {
    const uint64_t size = read_value<uint64_t>(ptr, end, 0);
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(uint16_t);
    if (ptr + bytes > end)
        return {};
    std::vector<uint16_t> values(static_cast<std::size_t>(size));
    if (bytes != 0) {
        std::memcpy(values.data(), ptr, bytes);
        ptr += bytes;
    }
    return values;
}

cudnnConvolutionFwdAlgo_t parse_algo(int32_t value) {
    switch (value) {
    case 1:
        return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    case 2:
        return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    case 3:
        return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    case 4:
        return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    case 5:
        return CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    case 6:
        return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    case 7:
        return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    default:
        return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }
}

bool set_conv_descriptors(const nvinfer1::Dims& input_dims, int32_t out_channels,
                          int32_t in_channels, int32_t kernel_t, int32_t kernel_h,
                          int32_t kernel_w, cudnnTensorDescriptor_t input_desc,
                          cudnnFilterDescriptor_t filter_desc, cudnnConvolutionDescriptor_t conv_desc,
                          cudnnTensorDescriptor_t output_desc,
                          cudnnTensorDescriptor_t bias_desc) {
    if (input_dims.nbDims != 5 || input_dims.d[1] != in_channels)
        return false;
    const int32_t batch = static_cast<int32_t>(input_dims.d[0]);
    const int32_t input_frames = static_cast<int32_t>(input_dims.d[2]);
    const int32_t input_height = static_cast<int32_t>(input_dims.d[3]);
    const int32_t input_width = static_cast<int32_t>(input_dims.d[4]);
    const int32_t frames = input_frames - kernel_t + 1;
    const int32_t height = input_height - kernel_h + 1;
    const int32_t width = input_width - kernel_w + 1;
    if (batch <= 0 || frames <= 0 || height <= 0 || width <= 0)
        return false;

    const int input_dims_arr[5] = {batch, in_channels, input_frames, input_height, input_width};
    const int input_strides[5] = {in_channels * input_frames * input_height * input_width,
                                  input_frames * input_height * input_width,
                                  input_height * input_width, input_width, 1};
    const int filter_dims[5] = {out_channels, in_channels, kernel_t, kernel_h, kernel_w};
    const int pad[3] = {0, 0, 0};
    const int stride[3] = {1, 1, 1};
    const int dilation[3] = {1, 1, 1};
    const int output_dims_arr[5] = {batch, out_channels, frames, height, width};
    const int output_strides[5] = {out_channels * frames * height * width,
                                   frames * height * width, height * width, width, 1};
    const int bias_dims[5] = {1, out_channels, 1, 1, 1};
    const int bias_strides[5] = {out_channels, 1, 1, 1, 1};

    return cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_BFLOAT16, 5, input_dims_arr,
                                      input_strides) == CUDNN_STATUS_SUCCESS &&
           cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW, 5,
                                      filter_dims) == CUDNN_STATUS_SUCCESS &&
           cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad, stride, dilation,
                                           CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) ==
               CUDNN_STATUS_SUCCESS &&
           cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_BFLOAT16, 5, output_dims_arr,
                                      output_strides) == CUDNN_STATUS_SUCCESS &&
           cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_BFLOAT16, 5, bias_dims,
                                      bias_strides) == CUDNN_STATUS_SUCCESS;
}

} // namespace

SanaWmPatchEmbed3dPlugin::SanaWmPatchEmbed3dPlugin(int32_t out_channels, int32_t in_channels,
                                                   int32_t kernel_t, int32_t kernel_h,
                                                   int32_t kernel_w, int32_t algo,
                                                   std::vector<uint16_t> weight,
                                                   std::vector<uint16_t> bias)
    : out_channels_(out_channels), in_channels_(in_channels), kernel_t_(kernel_t),
      kernel_h_(kernel_h), kernel_w_(kernel_w), algo_(algo), weight_(std::move(weight)),
      bias_(std::move(bias)) {}

SanaWmPatchEmbed3dPlugin::SanaWmPatchEmbed3dPlugin(const void* data, size_t length) {
    const char* ptr = static_cast<const char*>(data);
    const char* end = ptr + length;
    const uint32_t magic = read_value<uint32_t>(ptr, end, 0);
    const uint32_t version = read_value<uint32_t>(ptr, end, 0);
    if (magic != 0x53415750U || version != 1U)
        return;
    out_channels_ = read_value<int32_t>(ptr, end, 0);
    in_channels_ = read_value<int32_t>(ptr, end, 0);
    kernel_t_ = read_value<int32_t>(ptr, end, 1);
    kernel_h_ = read_value<int32_t>(ptr, end, 1);
    kernel_w_ = read_value<int32_t>(ptr, end, 1);
    algo_ = read_value<int32_t>(ptr, end, 0);
    weight_ = read_vector(ptr, end);
    bias_ = read_vector(ptr, end);
}

char const* SanaWmPatchEmbed3dPlugin::getPluginType() const noexcept { return kPLUGIN_NAME; }

char const* SanaWmPatchEmbed3dPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }

int32_t SanaWmPatchEmbed3dPlugin::getNbOutputs() const noexcept { return 1; }

int32_t SanaWmPatchEmbed3dPlugin::initialize() noexcept {
    if (handle_ != nullptr && device_weight_ != nullptr &&
        (bias_.empty() || device_bias_ != nullptr)) {
        return 0;
    }
    terminate();
    if (cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS)
        return 1;
    if (!weight_.empty()) {
        const std::size_t bytes = weight_.size() * sizeof(uint16_t);
        if (cudaMalloc(&device_weight_, bytes) != cudaSuccess ||
            cudaMemcpy(device_weight_, weight_.data(), bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            terminate();
            return 1;
        }
    }
    if (!bias_.empty()) {
        const std::size_t bytes = bias_.size() * sizeof(uint16_t);
        if (cudaMalloc(&device_bias_, bytes) != cudaSuccess ||
            cudaMemcpy(device_bias_, bias_.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            terminate();
            return 1;
        }
    }
    return 0;
}

void SanaWmPatchEmbed3dPlugin::terminate() noexcept {
    if (device_bias_ != nullptr) {
        cudaFree(device_bias_);
        device_bias_ = nullptr;
    }
    if (device_weight_ != nullptr) {
        cudaFree(device_weight_);
        device_weight_ = nullptr;
    }
    if (handle_ != nullptr) {
        cudnnDestroy(handle_);
        handle_ = nullptr;
    }
}

void SanaWmPatchEmbed3dPlugin::destroy() noexcept { delete this; }

size_t SanaWmPatchEmbed3dPlugin::getSerializationSize() const noexcept {
    return sizeof(uint32_t) * 2 + sizeof(int32_t) * 6 + sizeof(uint64_t) * 2 +
           (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmPatchEmbed3dPlugin::serialize(void* buffer) const noexcept {
    auto* ptr = static_cast<char*>(buffer);
    write_value<uint32_t>(ptr, 0x53415750U);
    write_value<uint32_t>(ptr, 1U);
    write_value(ptr, out_channels_);
    write_value(ptr, in_channels_);
    write_value(ptr, kernel_t_);
    write_value(ptr, kernel_h_);
    write_value(ptr, kernel_w_);
    write_value(ptr, algo_);
    write_vector(ptr, weight_);
    write_vector(ptr, bias_);
}

void SanaWmPatchEmbed3dPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmPatchEmbed3dPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmPatchEmbed3dPlugin::getOutputDataType(
    int32_t /*index*/, nvinfer1::DataType const* inputTypes, int32_t /*nbInputs*/) const noexcept {
    return inputTypes[0];
}

SanaWmPatchEmbed3dPlugin* SanaWmPatchEmbed3dPlugin::clone() const noexcept {
    auto* plugin = new SanaWmPatchEmbed3dPlugin(out_channels_, in_channels_, kernel_t_, kernel_h_,
                                                kernel_w_, algo_, weight_, bias_);
    plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}

nvinfer1::DimsExprs SanaWmPatchEmbed3dPlugin::getOutputDimensions(
    int32_t /*outputIndex*/, nvinfer1::DimsExprs const* inputs, int32_t /*nbInputs*/,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    out.nbDims = 5;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(out_channels_);
    out.d[2] = inputs[0].d[2];
    out.d[3] = inputs[0].d[3];
    out.d[4] = inputs[0].d[4];
    return out;
}

bool SanaWmPatchEmbed3dPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t /*nbInputs*/,
    int32_t /*nbOutputs*/) noexcept {
    const auto& desc = inOut[pos];
    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (pos == 0)
        return desc.type == nvinfer1::DataType::kBF16;
    return desc.type == inOut[0].type;
}

void SanaWmPatchEmbed3dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* /*in*/,
                                               int32_t /*nbInputs*/,
                                               nvinfer1::DynamicPluginTensorDesc const* /*out*/,
                                               int32_t /*nbOutputs*/) noexcept {}

size_t SanaWmPatchEmbed3dPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    nvinfer1::PluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept {
    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnTensorDescriptor_t bias_desc = nullptr;
    size_t workspace = 0;
    if (cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS &&
        cudnnCreateTensorDescriptor(&input_desc) == CUDNN_STATUS_SUCCESS &&
        cudnnCreateFilterDescriptor(&filter_desc) == CUDNN_STATUS_SUCCESS &&
        cudnnCreateConvolutionDescriptor(&conv_desc) == CUDNN_STATUS_SUCCESS &&
        cudnnCreateTensorDescriptor(&output_desc) == CUDNN_STATUS_SUCCESS &&
        cudnnCreateTensorDescriptor(&bias_desc) == CUDNN_STATUS_SUCCESS &&
        set_conv_descriptors(inputs[0].dims, out_channels_, in_channels_, kernel_t_, kernel_h_,
                             kernel_w_, input_desc, filter_desc, conv_desc, output_desc,
                             bias_desc)) {
        cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc,
                                                output_desc, parse_algo(algo_), &workspace);
    }
    if (bias_desc != nullptr)
        cudnnDestroyTensorDescriptor(bias_desc);
    if (output_desc != nullptr)
        cudnnDestroyTensorDescriptor(output_desc);
    if (conv_desc != nullptr)
        cudnnDestroyConvolutionDescriptor(conv_desc);
    if (filter_desc != nullptr)
        cudnnDestroyFilterDescriptor(filter_desc);
    if (input_desc != nullptr)
        cudnnDestroyTensorDescriptor(input_desc);
    if (handle != nullptr)
        cudnnDestroy(handle);
    return workspace;
}

int32_t SanaWmPatchEmbed3dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                          nvinfer1::PluginTensorDesc const* /*outputDesc*/,
                                          void const* const* inputs, void* const* outputs,
                                          void* workspace, cudaStream_t stream) noexcept {
    if ((handle_ == nullptr || device_weight_ == nullptr ||
         (!bias_.empty() && device_bias_ == nullptr)) &&
        initialize() != 0) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d lazy initialize failed\n");
        return 1;
    }
    if (handle_ == nullptr || inputs == nullptr || outputs == nullptr || device_weight_ == nullptr) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d enqueue missing state handle=%p inputs=%p "
                             "outputs=%p weight=%p\n",
                     static_cast<void*>(handle_), static_cast<void const*>(inputs),
                     static_cast<void const*>(outputs), device_weight_);
        return 1;
    }
    if (static_cast<int64_t>(weight_.size()) !=
        static_cast<int64_t>(out_channels_) * in_channels_ * kernel_t_ * kernel_h_ * kernel_w_) {
        return 1;
    }
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnTensorDescriptor_t bias_desc = nullptr;
    size_t workspace_size = 0;
    int32_t status = 1;
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cudnnStatus_t cudnn_status = cudnnSetStream(handle_, stream);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d cudnnSetStream failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        return 1;
    }
    cudnn_status = cudnnCreateTensorDescriptor(&input_desc);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d create input desc failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    cudnn_status = cudnnCreateFilterDescriptor(&filter_desc);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d create filter desc failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    cudnn_status = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d create conv desc failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    cudnn_status = cudnnCreateTensorDescriptor(&output_desc);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d create output desc failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    cudnn_status = cudnnCreateTensorDescriptor(&bias_desc);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d create bias desc failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    if (!set_conv_descriptors(inputDesc[0].dims, out_channels_, in_channels_, kernel_t_,
                              kernel_h_, kernel_w_, input_desc, filter_desc, conv_desc,
                              output_desc, bias_desc)) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d descriptor setup failed dims=(%ld,%ld,%ld,%ld,%ld) "
                             "channels=(%d,%d) kernel=(%d,%d,%d)\n",
                     static_cast<long>(inputDesc[0].dims.d[0]),
                     static_cast<long>(inputDesc[0].dims.d[1]),
                     static_cast<long>(inputDesc[0].dims.d[2]),
                     static_cast<long>(inputDesc[0].dims.d[3]),
                     static_cast<long>(inputDesc[0].dims.d[4]), out_channels_, in_channels_,
                     kernel_t_, kernel_h_, kernel_w_);
        goto cleanup;
    }
    cudnn_status = cudnnGetConvolutionForwardWorkspaceSize(handle_, input_desc, filter_desc,
                                                           conv_desc, output_desc,
                                                           parse_algo(algo_), &workspace_size);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d workspace failed algo=%d: %s\n", algo_,
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    cudnn_status = cudnnConvolutionForward(handle_, &alpha, input_desc, inputs[0], filter_desc,
                                           device_weight_, conv_desc, parse_algo(algo_),
                                           workspace, workspace_size, &beta, output_desc,
                                           outputs[0]);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d forward failed algo=%d workspace=%zu ptr=%p: %s\n",
                     algo_, workspace_size, workspace, cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    if (!bias_.empty()) {
        cudnn_status =
            cudnnAddTensor(handle_, &alpha, bias_desc, device_bias_, &alpha, output_desc,
                           outputs[0]);
    }
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "SanaWmPatchEmbed3d bias failed: %s\n",
                     cudnnGetErrorString(cudnn_status));
        goto cleanup;
    }
    status = 0;

cleanup:
    if (bias_desc != nullptr)
        cudnnDestroyTensorDescriptor(bias_desc);
    if (output_desc != nullptr)
        cudnnDestroyTensorDescriptor(output_desc);
    if (conv_desc != nullptr)
        cudnnDestroyConvolutionDescriptor(conv_desc);
    if (filter_desc != nullptr)
        cudnnDestroyFilterDescriptor(filter_desc);
    if (input_desc != nullptr)
        cudnnDestroyTensorDescriptor(input_desc);
    return status;
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT && TRTMC_HAS_CUDNN
