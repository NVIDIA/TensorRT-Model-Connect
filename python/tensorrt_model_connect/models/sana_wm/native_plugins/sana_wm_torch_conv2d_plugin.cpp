/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if TRTMC_HAS_TRT && TRTMC_HAS_LIBTORCH_CONV

#include "sana_wm_gdn_plugin.h"

#include <ATen/ATen.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addcmul.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/amin.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv2d.h>
#include <ATen/ops/conv3d.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/div.h>
#include <ATen/ops/einsum.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_real.h>
#include <algorithm>
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <optional>

namespace trtmc {
namespace {

uint16_t float_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

void append_bf16_values(std::vector<uint16_t>& out, const float* values, int32_t count) {
    out.clear();
    if (values == nullptr || count <= 0)
        return;
    out.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i)
        out.push_back(float_to_bf16_bits(values[i]));
}

void append_float_values(std::vector<float>& out, const float* values, int32_t count) {
    out.clear();
    if (values == nullptr || count <= 0)
        return;
    out.assign(values, values + count);
}

bool copy_to_device_cache(void*& device_ptr, const void* host_ptr, size_t bytes,
                          cudaStream_t stream) noexcept {
    if (device_ptr != nullptr)
        return true;
    if (host_ptr == nullptr || bytes == 0)
        return false;
    if (cudaMalloc(&device_ptr, bytes) != cudaSuccess)
        return false;
    if (cudaMemcpyAsync(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return false;
    return true;
}

void free_device_cache(void*& device_ptr) noexcept {
    if (device_ptr != nullptr) {
        cudaFree(device_ptr);
        device_ptr = nullptr;
    }
}

struct Conv2dShape {
    int32_t batch{0};
    int32_t channels{0};
    int32_t height{0};
    int32_t width{0};
};

struct Conv3dShape {
    int32_t batch{0};
    int32_t channels{0};
    int32_t frames{0};
    int32_t height{0};
    int32_t width{0};
};

Conv2dShape parse_conv2d_shape(const nvinfer1::Dims& dims) {
    Conv2dShape shape;
    if (dims.nbDims == 4) {
        shape.batch = dims.d[0];
        shape.channels = dims.d[1];
        shape.height = dims.d[2];
        shape.width = dims.d[3];
    }
    return shape;
}

Conv3dShape parse_conv3d_shape(const nvinfer1::Dims& dims) {
    Conv3dShape shape;
    if (dims.nbDims == 5) {
        shape.batch = dims.d[0];
        shape.channels = dims.d[1];
        shape.frames = dims.d[2];
        shape.height = dims.d[3];
        shape.width = dims.d[4];
    }
    return shape;
}

bool report_torch_conv_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_torch_conv2d] %s failed: %s\n", op, detail);
    return true;
}

bool report_torch_conv3d_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_torch_conv3d] %s failed: %s\n", op, detail);
    return true;
}

bool report_vae_rms_silu_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_vae_rms_silu] %s failed: %s\n", op, detail);
    return true;
}

bool report_timestep_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_timestep] %s failed: %s\n", op, detail);
    return true;
}

bool report_frame_gate_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_frame_gate] %s failed: %s\n", op, detail);
    return true;
}

bool report_caption_embed_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_caption_embed] %s failed: %s\n", op, detail);
    return true;
}

bool report_cross_attention_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_cross_attention] %s failed: %s\n", op, detail);
    return true;
}

bool report_softmax_attention_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_softmax_attention] %s failed: %s\n", op, detail);
    return true;
}

bool report_torch_cam_prep_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_torch_cam_prep] %s failed: %s\n", op, detail);
    return true;
}

bool report_camera_beta_discount_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_camera_beta_discount] %s failed: %s\n", op, detail);
    return true;
}

bool report_frame_mean_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_frame_mean] %s failed: %s\n", op, detail);
    return true;
}

bool report_layer_norm_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_layer_norm] %s failed: %s\n", op, detail);
    return true;
}

bool report_gate_proj_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_gate_proj] %s failed: %s\n", op, detail);
    return true;
}

bool report_decay_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_decay] %s failed: %s\n", op, detail);
    return true;
}

bool report_glumbconvtemp_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_glumbconvtemp] %s failed: %s\n", op, detail);
    return true;
}

bool report_gemma_rope_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_gemma_rope] %s failed: %s\n", op, detail);
    return true;
}

bool report_gemma_rms_norm_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_gemma_rms_norm] %s failed: %s\n", op, detail);
    return true;
}

bool report_gemma_gated_gelu_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_gemma_gated_gelu] %s failed: %s\n", op, detail);
    return true;
}

bool report_gemma_attention_error(const char* op, const char* detail) {
    std::fprintf(stderr, "[trtmc.sana_wm_gemma_attention] %s failed: %s\n", op, detail);
    return true;
}

} // namespace

SanaWmTorchConv2dPlugin::SanaWmTorchConv2dPlugin(int32_t out_channels, int32_t in_channels,
                                                 int32_t kernel_h, int32_t kernel_w, int32_t pad_h,
                                                 int32_t pad_w, int32_t groups, const float* weight,
                                                 int32_t weight_count, const float* bias,
                                                 int32_t bias_count)
    : out_channels_(out_channels), in_channels_(in_channels), kernel_h_(kernel_h),
      kernel_w_(kernel_w), pad_h_(pad_h), pad_w_(pad_w), groups_(std::max(1, groups)) {
    const int32_t expected_weight =
        out_channels_ * (in_channels_ / groups_) * kernel_h_ * kernel_w_;
    if (weight_count == expected_weight)
        append_bf16_values(weight_, weight, weight_count);
    if (bias_count == out_channels_)
        append_bf16_values(bias_, bias, bias_count);
}

SanaWmTorchConv2dPlugin::SanaWmTorchConv2dPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 8;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    out_channels_ = header[0];
    in_channels_ = header[1];
    kernel_h_ = header[2];
    kernel_w_ = header[3];
    pad_h_ = header[4];
    pad_w_ = header[5];
    groups_ = std::max(1, header[6]);
    const bool has_bias = header[7] != 0;
    const int32_t weight_count = out_channels_ * (in_channels_ / groups_) * kernel_h_ * kernel_w_;
    const int32_t bias_count = has_bias ? out_channels_ : 0;
    const size_t expected = kHeaderCount * sizeof(int32_t) +
                            static_cast<size_t>(weight_count + bias_count) * sizeof(uint16_t);
    if (weight_count <= 0 || length < expected)
        return;
    const auto* payload = reinterpret_cast<const uint16_t*>(static_cast<const char*>(data) +
                                                            kHeaderCount * sizeof(int32_t));
    weight_.assign(payload, payload + weight_count);
    if (has_bias)
        bias_.assign(payload + weight_count, payload + weight_count + bias_count);
}

char const* SanaWmTorchConv2dPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmTorchConv2dPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmTorchConv2dPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmTorchConv2dPlugin::initialize() noexcept {
    return 0;
}

void SanaWmTorchConv2dPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmTorchConv2dPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmTorchConv2dPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 8;
    return kHeaderCount * sizeof(int32_t) + (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmTorchConv2dPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 8;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = out_channels_;
    header[1] = in_channels_;
    header[2] = kernel_h_;
    header[3] = kernel_w_;
    header[4] = pad_h_;
    header[5] = pad_w_;
    header[6] = groups_;
    header[7] = bias_.empty() ? 0 : 1;
    auto* payload =
        reinterpret_cast<uint16_t*>(static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t));
    if (!weight_.empty()) {
        std::memcpy(payload, weight_.data(), weight_.size() * sizeof(uint16_t));
        payload += weight_.size();
    }
    if (!bias_.empty())
        std::memcpy(payload, bias_.data(), bias_.size() * sizeof(uint16_t));
}

void SanaWmTorchConv2dPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmTorchConv2dPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmTorchConv2dPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                              int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmTorchConv2dPlugin* SanaWmTorchConv2dPlugin::clone() const noexcept {
    auto* p = new SanaWmTorchConv2dPlugin();
    p->out_channels_ = out_channels_;
    p->in_channels_ = in_channels_;
    p->kernel_h_ = kernel_h_;
    p->kernel_w_ = kernel_w_;
    p->pad_h_ = pad_h_;
    p->pad_w_ = pad_w_;
    p->groups_ = groups_;
    p->weight_ = weight_;
    p->bias_ = bias_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmTorchConv2dPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
                                             int32_t,
                                             nvinfer1::IExprBuilder& exprBuilder) noexcept {
    (void)outputIndex;
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(out_channels_);
    out.d[2] = inputs[0].d[2];
    out.d[3] = inputs[0].d[3];
    return out;
}

bool SanaWmTorchConv2dPlugin::supportsFormatCombination(int32_t pos,
                                                        nvinfer1::PluginTensorDesc const* inOut,
                                                        int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmTorchConv2dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                              nvinfer1::DynamicPluginTensorDesc const*,
                                              int32_t) noexcept {}

size_t SanaWmTorchConv2dPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                 nvinfer1::PluginTensorDesc const*,
                                                 int32_t) const noexcept {
    return 0;
}

void SanaWmTorchConv2dPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    if (weight_device_ != nullptr) {
        cudaFree(weight_device_);
        weight_device_ = nullptr;
    }
    if (bias_device_ != nullptr) {
        cudaFree(bias_device_);
        bias_device_ = nullptr;
    }
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmTorchConv2dPlugin::ensureDeviceCache(cudaStream_t stream,
                                                int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    cached_device_ = device_index;
    if (weight_device_ == nullptr) {
        const size_t bytes = weight_.size() * sizeof(uint16_t);
        if (bytes == 0 || cudaMalloc(&weight_device_, bytes) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync(weight_device_, weight_.data(), bytes, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess)
            return false;
    }
    if (!bias_.empty() && bias_device_ == nullptr) {
        const size_t bytes = bias_.size() * sizeof(uint16_t);
        if (cudaMalloc(&bias_device_, bytes) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync(bias_device_, bias_.data(), bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess)
            return false;
    }
    return true;
}

int32_t SanaWmTorchConv2dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                         nvinfer1::PluginTensorDesc const*,
                                         void const* const* inputs, void* const* outputs, void*,
                                         cudaStream_t stream) noexcept {
    try {
        const auto shape = parse_conv2d_shape(inputDesc[0].dims);
        const int32_t expected_weight =
            out_channels_ * (in_channels_ / groups_) * kernel_h_ * kernel_w_;
        if (shape.batch <= 0 || shape.channels != in_channels_ ||
            static_cast<int32_t>(weight_.size()) != expected_weight) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        if (!ensureDeviceCache(stream, device_index))
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto input = at::from_blob(const_cast<void*>(inputs[0]),
                                   {shape.batch, in_channels_, shape.height, shape.width}, options);
        auto input_cl = input.contiguous(at::MemoryFormat::ChannelsLast);
        auto weight = at::from_blob(
            weight_device_, {out_channels_, in_channels_ / groups_, kernel_h_, kernel_w_}, options);

        std::optional<at::Tensor> bias;
        if (bias_device_ != nullptr)
            bias = at::from_blob(bias_device_, {out_channels_}, options);

        auto output = at::conv2d(input_cl, weight, bias, {1, 1}, {pad_h_, pad_w_}, {1, 1}, groups_);
        auto output_view = at::from_blob(
            outputs[0], {shape.batch, out_channels_, shape.height, shape.width}, options);
        output_view.copy_(output);
        return 0;
    } catch (const c10::Error& e) {
        report_torch_conv_error("aten conv2d", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_torch_conv_error("aten conv2d", e.what());
        return 1;
    }
}

SanaWmTorchConv3dPlugin::SanaWmTorchConv3dPlugin(
    int32_t out_channels, int32_t in_channels, int32_t kernel_t, int32_t kernel_h, int32_t kernel_w,
    int32_t stride_t, int32_t stride_h, int32_t stride_w, int32_t pad_t, int32_t pad_h,
    int32_t pad_w, int32_t dilation_t, int32_t dilation_h, int32_t dilation_w, int32_t groups,
    int32_t output_t, int32_t output_h, int32_t output_w, const float* weight, int32_t weight_count,
    const float* bias, int32_t bias_count)
    : out_channels_(out_channels), in_channels_(in_channels), kernel_t_(kernel_t),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_t_(std::max(1, stride_t)),
      stride_h_(std::max(1, stride_h)), stride_w_(std::max(1, stride_w)),
      pad_t_(std::max(0, pad_t)), pad_h_(std::max(0, pad_h)), pad_w_(std::max(0, pad_w)),
      dilation_t_(std::max(1, dilation_t)), dilation_h_(std::max(1, dilation_h)),
      dilation_w_(std::max(1, dilation_w)), groups_(std::max(1, groups)), output_t_(output_t),
      output_h_(output_h), output_w_(output_w) {
    const int32_t expected_weight =
        out_channels_ * (in_channels_ / groups_) * kernel_t_ * kernel_h_ * kernel_w_;
    if (weight_count == expected_weight)
        append_bf16_values(weight_, weight, weight_count);
    if (bias_count == out_channels_)
        append_bf16_values(bias_, bias, bias_count);
}

SanaWmTorchConv3dPlugin::SanaWmTorchConv3dPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 19;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    out_channels_ = header[0];
    in_channels_ = header[1];
    kernel_t_ = header[2];
    kernel_h_ = header[3];
    kernel_w_ = header[4];
    stride_t_ = std::max(1, header[5]);
    stride_h_ = std::max(1, header[6]);
    stride_w_ = std::max(1, header[7]);
    pad_t_ = std::max(0, header[8]);
    pad_h_ = std::max(0, header[9]);
    pad_w_ = std::max(0, header[10]);
    dilation_t_ = std::max(1, header[11]);
    dilation_h_ = std::max(1, header[12]);
    dilation_w_ = std::max(1, header[13]);
    groups_ = std::max(1, header[14]);
    output_t_ = header[15];
    output_h_ = header[16];
    output_w_ = header[17];
    const bool has_bias = header[18] != 0;
    const int32_t weight_count =
        out_channels_ * (in_channels_ / groups_) * kernel_t_ * kernel_h_ * kernel_w_;
    const int32_t bias_count = has_bias ? out_channels_ : 0;
    const size_t expected = kHeaderCount * sizeof(int32_t) +
                            static_cast<size_t>(weight_count + bias_count) * sizeof(uint16_t);
    if (weight_count <= 0 || output_t_ <= 0 || output_h_ <= 0 || output_w_ <= 0 ||
        length < expected) {
        return;
    }
    const auto* payload = reinterpret_cast<const uint16_t*>(static_cast<const char*>(data) +
                                                            kHeaderCount * sizeof(int32_t));
    weight_.assign(payload, payload + weight_count);
    if (has_bias)
        bias_.assign(payload + weight_count, payload + weight_count + bias_count);
}

char const* SanaWmTorchConv3dPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmTorchConv3dPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmTorchConv3dPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmTorchConv3dPlugin::initialize() noexcept {
    return 0;
}

void SanaWmTorchConv3dPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmTorchConv3dPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmTorchConv3dPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 19;
    return kHeaderCount * sizeof(int32_t) + (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmTorchConv3dPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 19;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = out_channels_;
    header[1] = in_channels_;
    header[2] = kernel_t_;
    header[3] = kernel_h_;
    header[4] = kernel_w_;
    header[5] = stride_t_;
    header[6] = stride_h_;
    header[7] = stride_w_;
    header[8] = pad_t_;
    header[9] = pad_h_;
    header[10] = pad_w_;
    header[11] = dilation_t_;
    header[12] = dilation_h_;
    header[13] = dilation_w_;
    header[14] = groups_;
    header[15] = output_t_;
    header[16] = output_h_;
    header[17] = output_w_;
    header[18] = bias_.empty() ? 0 : 1;
    auto* payload =
        reinterpret_cast<uint16_t*>(static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t));
    if (!weight_.empty()) {
        std::memcpy(payload, weight_.data(), weight_.size() * sizeof(uint16_t));
        payload += weight_.size();
    }
    if (!bias_.empty())
        std::memcpy(payload, bias_.data(), bias_.size() * sizeof(uint16_t));
}

void SanaWmTorchConv3dPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmTorchConv3dPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmTorchConv3dPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                              int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmTorchConv3dPlugin* SanaWmTorchConv3dPlugin::clone() const noexcept {
    auto* p = new SanaWmTorchConv3dPlugin();
    p->out_channels_ = out_channels_;
    p->in_channels_ = in_channels_;
    p->kernel_t_ = kernel_t_;
    p->kernel_h_ = kernel_h_;
    p->kernel_w_ = kernel_w_;
    p->stride_t_ = stride_t_;
    p->stride_h_ = stride_h_;
    p->stride_w_ = stride_w_;
    p->pad_t_ = pad_t_;
    p->pad_h_ = pad_h_;
    p->pad_w_ = pad_w_;
    p->dilation_t_ = dilation_t_;
    p->dilation_h_ = dilation_h_;
    p->dilation_w_ = dilation_w_;
    p->groups_ = groups_;
    p->output_t_ = output_t_;
    p->output_h_ = output_h_;
    p->output_w_ = output_w_;
    p->weight_ = weight_;
    p->bias_ = bias_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmTorchConv3dPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
                                             int32_t,
                                             nvinfer1::IExprBuilder& exprBuilder) noexcept {
    (void)outputIndex;
    nvinfer1::DimsExprs out;
    out.nbDims = 5;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(out_channels_);
    out.d[2] = exprBuilder.constant(output_t_);
    out.d[3] = exprBuilder.constant(output_h_);
    out.d[4] = exprBuilder.constant(output_w_);
    return out;
}

bool SanaWmTorchConv3dPlugin::supportsFormatCombination(int32_t pos,
                                                        nvinfer1::PluginTensorDesc const* inOut,
                                                        int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmTorchConv3dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                              nvinfer1::DynamicPluginTensorDesc const*,
                                              int32_t) noexcept {}

size_t SanaWmTorchConv3dPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                 nvinfer1::PluginTensorDesc const*,
                                                 int32_t) const noexcept {
    return 0;
}

void SanaWmTorchConv3dPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    if (weight_device_ != nullptr) {
        cudaFree(weight_device_);
        weight_device_ = nullptr;
    }
    if (bias_device_ != nullptr) {
        cudaFree(bias_device_);
        bias_device_ = nullptr;
    }
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmTorchConv3dPlugin::ensureDeviceCache(cudaStream_t stream,
                                                int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    cached_device_ = device_index;
    if (weight_device_ == nullptr) {
        const size_t bytes = weight_.size() * sizeof(uint16_t);
        if (bytes == 0 || cudaMalloc(&weight_device_, bytes) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync(weight_device_, weight_.data(), bytes, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            return false;
        }
    }
    if (!bias_.empty() && bias_device_ == nullptr) {
        const size_t bytes = bias_.size() * sizeof(uint16_t);
        if (cudaMalloc(&bias_device_, bytes) != cudaSuccess)
            return false;
        if (cudaMemcpyAsync(bias_device_, bias_.data(), bytes, cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
            return false;
        }
    }
    return true;
}

int32_t SanaWmTorchConv3dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                         nvinfer1::PluginTensorDesc const*,
                                         void const* const* inputs, void* const* outputs, void*,
                                         cudaStream_t stream) noexcept {
    try {
        const auto shape = parse_conv3d_shape(inputDesc[0].dims);
        const int32_t expected_weight =
            out_channels_ * (in_channels_ / groups_) * kernel_t_ * kernel_h_ * kernel_w_;
        if (shape.batch <= 0 || shape.channels != in_channels_ || shape.frames <= 0 ||
            shape.height <= 0 || shape.width <= 0 || output_t_ <= 0 || output_h_ <= 0 ||
            output_w_ <= 0 || static_cast<int32_t>(weight_.size()) != expected_weight) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        if (!ensureDeviceCache(stream, device_index))
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto input = at::from_blob(
            const_cast<void*>(inputs[0]),
            {shape.batch, in_channels_, shape.frames, shape.height, shape.width}, options);
        auto weight = at::from_blob(
            weight_device_,
            {out_channels_, in_channels_ / groups_, kernel_t_, kernel_h_, kernel_w_}, options);

        std::optional<at::Tensor> bias;
        if (bias_device_ != nullptr)
            bias = at::from_blob(bias_device_, {out_channels_}, options);

        auto output =
            at::conv3d(input, weight, bias, {stride_t_, stride_h_, stride_w_},
                       {pad_t_, pad_h_, pad_w_}, {dilation_t_, dilation_h_, dilation_w_}, groups_);
        auto output_view = at::from_blob(
            outputs[0], {shape.batch, out_channels_, output_t_, output_h_, output_w_}, options);
        output_view.copy_(output);
        return 0;
    } catch (const c10::Error& e) {
        report_torch_conv3d_error("aten conv3d", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_torch_conv3d_error("aten conv3d", e.what());
        return 1;
    }
}

SanaWmVaeRmsSiluPlugin::SanaWmVaeRmsSiluPlugin(float eps) : eps_(eps) {}

SanaWmVaeRmsSiluPlugin::SanaWmVaeRmsSiluPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(eps_))
        std::memcpy(&eps_, data, sizeof(eps_));
}

char const* SanaWmVaeRmsSiluPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmVaeRmsSiluPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmVaeRmsSiluPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmVaeRmsSiluPlugin::initialize() noexcept {
    return 0;
}

void SanaWmVaeRmsSiluPlugin::terminate() noexcept {}

void SanaWmVaeRmsSiluPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmVaeRmsSiluPlugin::getSerializationSize() const noexcept {
    return sizeof(eps_);
}

void SanaWmVaeRmsSiluPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &eps_, sizeof(eps_));
}

void SanaWmVaeRmsSiluPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmVaeRmsSiluPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmVaeRmsSiluPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                             int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmVaeRmsSiluPlugin* SanaWmVaeRmsSiluPlugin::clone() const noexcept {
    auto* p = new SanaWmVaeRmsSiluPlugin(eps_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs SanaWmVaeRmsSiluPlugin::getOutputDimensions(int32_t outputIndex,
                                                                nvinfer1::DimsExprs const* inputs,
                                                                int32_t,
                                                                nvinfer1::IExprBuilder&) noexcept {
    (void)outputIndex;
    return inputs[0];
}

bool SanaWmVaeRmsSiluPlugin::supportsFormatCombination(int32_t pos,
                                                       nvinfer1::PluginTensorDesc const* inOut,
                                                       int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmVaeRmsSiluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                             nvinfer1::DynamicPluginTensorDesc const*,
                                             int32_t) noexcept {}

size_t SanaWmVaeRmsSiluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                nvinfer1::PluginTensorDesc const*,
                                                int32_t) const noexcept {
    return 0;
}

int32_t SanaWmVaeRmsSiluPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                        nvinfer1::PluginTensorDesc const*,
                                        void const* const* inputs, void* const* outputs, void*,
                                        cudaStream_t stream) noexcept {
    try {
        const auto shape = parse_conv3d_shape(inputDesc[0].dims);
        if (shape.batch <= 0 || shape.channels <= 0 || shape.frames <= 0 || shape.height <= 0 ||
            shape.width <= 0) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const std::array<int64_t, 5> dimensions = {shape.batch, shape.channels, shape.frames,
                                                   shape.height, shape.width};
        auto input = at::from_blob(const_cast<void*>(inputs[0]), dimensions, options);
        auto squared = at::pow(input, 2);
        auto mean_sq = at::mean(squared, {1}, true);
        auto rms = at::sqrt(at::add(mean_sq, eps_));
        auto output = at::silu(at::div(input, rms));
        auto output_view = at::from_blob(outputs[0], dimensions, options);
        output_view.copy_(output);
        return 0;
    } catch (const c10::Error& e) {
        report_vae_rms_silu_error("aten rms norm + silu", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_vae_rms_silu_error("aten rms norm + silu", e.what());
        return 1;
    }
}

SanaWmVaeDenormalizePlugin::SanaWmVaeDenormalizePlugin(int32_t channels, float scaling_factor,
                                                       const float* mean, int32_t mean_count,
                                                       const float* std, int32_t std_count)
    : channels_(channels), scaling_factor_(scaling_factor) {
    if (mean_count == channels_)
        append_bf16_values(mean_, mean, mean_count);
    if (std_count == channels_)
        append_bf16_values(std_, std, std_count);
}
SanaWmVaeDenormalizePlugin::SanaWmVaeDenormalizePlugin(const void* data, size_t length) {
    constexpr size_t kHeader = sizeof(int32_t) + sizeof(float);
    if (data == nullptr || length < kHeader)
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::memcpy(&channels_, cursor, sizeof(channels_));
    cursor += sizeof(channels_);
    std::memcpy(&scaling_factor_, cursor, sizeof(scaling_factor_));
    cursor += sizeof(scaling_factor_);
    const size_t expected = kHeader + 2 * static_cast<size_t>(channels_) * sizeof(uint16_t);
    if (channels_ <= 0 || length != expected)
        return;
    const auto* values = reinterpret_cast<const uint16_t*>(cursor);
    mean_.assign(values, values + channels_);
    values += channels_;
    std_.assign(values, values + channels_);
}
SanaWmVaeDenormalizePlugin::~SanaWmVaeDenormalizePlugin() {
    releaseDeviceCache();
}
char const* SanaWmVaeDenormalizePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmVaeDenormalizePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmVaeDenormalizePlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmVaeDenormalizePlugin::initialize() noexcept {
    return 0;
}
void SanaWmVaeDenormalizePlugin::terminate() noexcept {
    releaseDeviceCache();
}
void SanaWmVaeDenormalizePlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmVaeDenormalizePlugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) + sizeof(float) + (mean_.size() + std_.size()) * sizeof(uint16_t);
}
void SanaWmVaeDenormalizePlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &channels_, sizeof(channels_));
    cursor += sizeof(channels_);
    std::memcpy(cursor, &scaling_factor_, sizeof(scaling_factor_));
    cursor += sizeof(scaling_factor_);
    if (!mean_.empty()) {
        std::memcpy(cursor, mean_.data(), mean_.size() * sizeof(uint16_t));
        cursor += mean_.size() * sizeof(uint16_t);
    }
    if (!std_.empty())
        std::memcpy(cursor, std_.data(), std_.size() * sizeof(uint16_t));
}
void SanaWmVaeDenormalizePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmVaeDenormalizePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmVaeDenormalizePlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                 int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmVaeDenormalizePlugin* SanaWmVaeDenormalizePlugin::clone() const noexcept {
    auto* plugin = new SanaWmVaeDenormalizePlugin();
    plugin->channels_ = channels_;
    plugin->scaling_factor_ = scaling_factor_;
    plugin->mean_ = mean_;
    plugin->std_ = std_;
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs
SanaWmVaeDenormalizePlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmVaeDenormalizePlugin::supportsFormatCombination(int32_t pos,
                                                           nvinfer1::PluginTensorDesc const* inOut,
                                                           int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}
void SanaWmVaeDenormalizePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) noexcept {}
size_t SanaWmVaeDenormalizePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                    nvinfer1::PluginTensorDesc const*,
                                                    int32_t) const noexcept {
    return 0;
}
void SanaWmVaeDenormalizePlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const auto have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(mean_device_);
    free_device_cache(std_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}
bool SanaWmVaeDenormalizePlugin::ensureDeviceCache(cudaStream_t stream,
                                                   int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(mean_device_, mean_.data(), mean_.size() * sizeof(uint16_t),
                                stream) &&
           copy_to_device_cache(std_device_, std_.data(), std_.size() * sizeof(uint16_t), stream);
}
int32_t SanaWmVaeDenormalizePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                            nvinfer1::PluginTensorDesc const*,
                                            void const* const* inputs, void* const* outputs, void*,
                                            cudaStream_t stream) noexcept {
    try {
        const auto shape = parse_conv3d_shape(inputDesc[0].dims);
        if (shape.batch <= 0 || shape.channels != channels_ || shape.frames <= 0 ||
            shape.height <= 0 || shape.width <= 0 || scaling_factor_ == 0.0F ||
            mean_.size() != static_cast<size_t>(channels_) ||
            std_.size() != static_cast<size_t>(channels_)) {
            return 1;
        }
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess ||
            !ensureDeviceCache(stream, device_index)) {
            return 1;
        }
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const std::array<int64_t, 5> dimensions = {shape.batch, shape.channels, shape.frames,
                                                   shape.height, shape.width};
        auto input = at::from_blob(const_cast<void*>(inputs[0]), dimensions, options);
        auto mean = at::from_blob(mean_device_, {1, channels_, 1, 1, 1}, options);
        auto std = at::from_blob(std_device_, {1, channels_, 1, 1, 1}, options);
        auto result = input * std / scaling_factor_ + mean;
        auto output = at::from_blob(outputs[0], dimensions, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& error) {
        report_vae_rms_silu_error("aten VAE denormalize", error.what());
        return 1;
    } catch (const std::exception& error) {
        report_vae_rms_silu_error("aten VAE denormalize", error.what());
        return 1;
    }
}

SanaWmVaeLayerNormPlugin::SanaWmVaeLayerNormPlugin(int32_t channels, float eps, const float* weight,
                                                   int32_t weight_count, const float* bias,
                                                   int32_t bias_count)
    : channels_(channels), eps_(eps) {
    if (weight_count == channels_)
        append_bf16_values(weight_, weight, weight_count);
    if (bias_count == channels_)
        append_bf16_values(bias_, bias, bias_count);
}
SanaWmVaeLayerNormPlugin::SanaWmVaeLayerNormPlugin(const void* data, size_t length) {
    constexpr size_t kHeader = sizeof(int32_t) + sizeof(float);
    if (data == nullptr || length < kHeader)
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::memcpy(&channels_, cursor, sizeof(channels_));
    cursor += sizeof(channels_);
    std::memcpy(&eps_, cursor, sizeof(eps_));
    cursor += sizeof(eps_);
    const size_t expected = kHeader + 2 * static_cast<size_t>(channels_) * sizeof(uint16_t);
    if (channels_ <= 0 || length != expected)
        return;
    const auto* values = reinterpret_cast<const uint16_t*>(cursor);
    weight_.assign(values, values + channels_);
    values += channels_;
    bias_.assign(values, values + channels_);
}
SanaWmVaeLayerNormPlugin::~SanaWmVaeLayerNormPlugin() {
    releaseDeviceCache();
}
char const* SanaWmVaeLayerNormPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmVaeLayerNormPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmVaeLayerNormPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmVaeLayerNormPlugin::initialize() noexcept {
    return 0;
}
void SanaWmVaeLayerNormPlugin::terminate() noexcept {
    releaseDeviceCache();
}
void SanaWmVaeLayerNormPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmVaeLayerNormPlugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) + sizeof(float) + (weight_.size() + bias_.size()) * sizeof(uint16_t);
}
void SanaWmVaeLayerNormPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &channels_, sizeof(channels_));
    cursor += sizeof(channels_);
    std::memcpy(cursor, &eps_, sizeof(eps_));
    cursor += sizeof(eps_);
    if (!weight_.empty()) {
        std::memcpy(cursor, weight_.data(), weight_.size() * sizeof(uint16_t));
        cursor += weight_.size() * sizeof(uint16_t);
    }
    if (!bias_.empty())
        std::memcpy(cursor, bias_.data(), bias_.size() * sizeof(uint16_t));
}
void SanaWmVaeLayerNormPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmVaeLayerNormPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmVaeLayerNormPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                               int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmVaeLayerNormPlugin* SanaWmVaeLayerNormPlugin::clone() const noexcept {
    auto* plugin = new SanaWmVaeLayerNormPlugin();
    plugin->channels_ = channels_;
    plugin->eps_ = eps_;
    plugin->weight_ = weight_;
    plugin->bias_ = bias_;
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs
SanaWmVaeLayerNormPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                              nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmVaeLayerNormPlugin::supportsFormatCombination(int32_t pos,
                                                         nvinfer1::PluginTensorDesc const* inOut,
                                                         int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}
void SanaWmVaeLayerNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                               nvinfer1::DynamicPluginTensorDesc const*,
                                               int32_t) noexcept {}
size_t SanaWmVaeLayerNormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                  nvinfer1::PluginTensorDesc const*,
                                                  int32_t) const noexcept {
    return 0;
}
void SanaWmVaeLayerNormPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const auto have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(weight_device_);
    free_device_cache(bias_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}
bool SanaWmVaeLayerNormPlugin::ensureDeviceCache(cudaStream_t stream,
                                                 int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(weight_device_, weight_.data(), weight_.size() * sizeof(uint16_t),
                                stream) &&
           copy_to_device_cache(bias_device_, bias_.data(), bias_.size() * sizeof(uint16_t),
                                stream);
}
int32_t SanaWmVaeLayerNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                          nvinfer1::PluginTensorDesc const*,
                                          void const* const* inputs, void* const* outputs, void*,
                                          cudaStream_t stream) noexcept {
    try {
        const auto shape = parse_conv3d_shape(inputDesc[0].dims);
        const auto channels = static_cast<std::size_t>(channels_);
        if (shape.batch <= 0 || shape.channels != channels_ || shape.frames <= 0 ||
            shape.height <= 0 || shape.width <= 0 || weight_.size() != channels ||
            bias_.size() != channels) {
            return 1;
        }
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess ||
            !ensureDeviceCache(stream, device_index)) {
            return 1;
        }
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const std::array<int64_t, 5> dimensions = {shape.batch, shape.channels, shape.frames,
                                                   shape.height, shape.width};
        auto input = at::from_blob(const_cast<void*>(inputs[0]), dimensions, options);
        auto weight = at::from_blob(weight_device_, {channels_}, options);
        auto bias = at::from_blob(bias_device_, {channels_}, options);
        auto channels_last = input.movedim(1, -1);
        auto result = at::layer_norm(channels_last, {channels_}, weight, bias,
                                     static_cast<double>(eps_), true)
                          .movedim(-1, 1);
        auto output = at::from_blob(outputs[0], dimensions, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& error) {
        report_layer_norm_error("aten VAE channel layer_norm", error.what());
        return 1;
    } catch (const std::exception& error) {
        report_layer_norm_error("aten VAE channel layer_norm", error.what());
        return 1;
    }
}

SanaWmGlumbconvTempPlugin::SanaWmGlumbconvTempPlugin(
    int32_t batch, int32_t frames, int32_t height, int32_t width, int32_t channels, int32_t hidden,
    int32_t t_kernel, const float* inverted_weight, int32_t inverted_weight_count,
    const float* inverted_bias, int32_t inverted_bias_count, const float* depth_weight,
    int32_t depth_weight_count, const float* depth_bias, int32_t depth_bias_count,
    const float* point_weight, int32_t point_weight_count, const float* t_weight,
    int32_t t_weight_count)
    : batch_(batch), frames_(frames), height_(height), width_(width), channels_(channels),
      hidden_(hidden), t_kernel_(t_kernel) {
    const int32_t expanded = hidden_ * 2;
    if (inverted_weight_count == expanded * channels_)
        append_bf16_values(inverted_weight_, inverted_weight, inverted_weight_count);
    if (inverted_bias_count == expanded)
        append_bf16_values(inverted_bias_, inverted_bias, inverted_bias_count);
    if (depth_weight_count == expanded * 3 * 3)
        append_bf16_values(depth_weight_, depth_weight, depth_weight_count);
    if (depth_bias_count == expanded)
        append_bf16_values(depth_bias_, depth_bias, depth_bias_count);
    if (point_weight_count == channels_ * hidden_)
        append_bf16_values(point_weight_, point_weight, point_weight_count);
    if (t_weight_count == channels_ * channels_ * t_kernel_)
        append_bf16_values(t_weight_, t_weight, t_weight_count);
}

SanaWmGlumbconvTempPlugin::SanaWmGlumbconvTempPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 9;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    batch_ = header[0];
    frames_ = header[1];
    height_ = header[2];
    width_ = header[3];
    channels_ = header[4];
    hidden_ = header[5];
    t_kernel_ = header[6];
    const bool has_inverted_bias = header[7] != 0;
    const bool has_depth_bias = header[8] != 0;
    const int32_t expanded = hidden_ * 2;
    const int32_t inverted_weight_count = expanded * channels_;
    const int32_t inverted_bias_count = has_inverted_bias ? expanded : 0;
    const int32_t depth_weight_count = expanded * 3 * 3;
    const int32_t depth_bias_count = has_depth_bias ? expanded : 0;
    const int32_t point_weight_count = channels_ * hidden_;
    const int32_t t_weight_count = channels_ * channels_ * t_kernel_;
    const size_t payload_count =
        static_cast<size_t>(inverted_weight_count + inverted_bias_count + depth_weight_count +
                            depth_bias_count + point_weight_count + t_weight_count);
    const size_t expected = kHeaderCount * sizeof(int32_t) + payload_count * sizeof(uint16_t);
    if (batch_ <= 0 || frames_ <= 0 || height_ <= 0 || width_ <= 0 || channels_ <= 0 ||
        hidden_ <= 0 || t_kernel_ <= 0 || length < expected)
        return;
    const auto* payload = reinterpret_cast<const uint16_t*>(static_cast<const char*>(data) +
                                                            kHeaderCount * sizeof(int32_t));
    inverted_weight_.assign(payload, payload + inverted_weight_count);
    payload += inverted_weight_count;
    if (has_inverted_bias) {
        inverted_bias_.assign(payload, payload + inverted_bias_count);
        payload += inverted_bias_count;
    }
    depth_weight_.assign(payload, payload + depth_weight_count);
    payload += depth_weight_count;
    if (has_depth_bias) {
        depth_bias_.assign(payload, payload + depth_bias_count);
        payload += depth_bias_count;
    }
    point_weight_.assign(payload, payload + point_weight_count);
    payload += point_weight_count;
    t_weight_.assign(payload, payload + t_weight_count);
}

char const* SanaWmGlumbconvTempPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGlumbconvTempPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGlumbconvTempPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmGlumbconvTempPlugin::initialize() noexcept {
    return 0;
}

void SanaWmGlumbconvTempPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmGlumbconvTempPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmGlumbconvTempPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 9;
    return kHeaderCount * sizeof(int32_t) +
           (inverted_weight_.size() + inverted_bias_.size() + depth_weight_.size() +
            depth_bias_.size() + point_weight_.size() + t_weight_.size()) *
               sizeof(uint16_t);
}

void SanaWmGlumbconvTempPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 9;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = batch_;
    header[1] = frames_;
    header[2] = height_;
    header[3] = width_;
    header[4] = channels_;
    header[5] = hidden_;
    header[6] = t_kernel_;
    header[7] = inverted_bias_.empty() ? 0 : 1;
    header[8] = depth_bias_.empty() ? 0 : 1;
    auto* payload =
        reinterpret_cast<uint16_t*>(static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t));
    auto write = [&payload](const std::vector<uint16_t>& values) {
        if (!values.empty()) {
            std::memcpy(payload, values.data(), values.size() * sizeof(uint16_t));
            payload += values.size();
        }
    };
    write(inverted_weight_);
    write(inverted_bias_);
    write(depth_weight_);
    write(depth_bias_);
    write(point_weight_);
    write(t_weight_);
}

void SanaWmGlumbconvTempPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGlumbconvTempPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGlumbconvTempPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmGlumbconvTempPlugin* SanaWmGlumbconvTempPlugin::clone() const noexcept {
    auto* p = new SanaWmGlumbconvTempPlugin();
    p->batch_ = batch_;
    p->frames_ = frames_;
    p->height_ = height_;
    p->width_ = width_;
    p->channels_ = channels_;
    p->hidden_ = hidden_;
    p->t_kernel_ = t_kernel_;
    p->inverted_weight_ = inverted_weight_;
    p->inverted_bias_ = inverted_bias_;
    p->depth_weight_ = depth_weight_;
    p->depth_bias_ = depth_bias_;
    p->point_weight_ = point_weight_;
    p->t_weight_ = t_weight_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmGlumbconvTempPlugin::getOutputDimensions(int32_t outputIndex,
                                               nvinfer1::DimsExprs const* inputs, int32_t,
                                               nvinfer1::IExprBuilder&) noexcept {
    (void)outputIndex;
    return inputs[0];
}

bool SanaWmGlumbconvTempPlugin::supportsFormatCombination(int32_t pos,
                                                          nvinfer1::PluginTensorDesc const* inOut,
                                                          int32_t, int32_t) noexcept {
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmGlumbconvTempPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                nvinfer1::DynamicPluginTensorDesc const*,
                                                int32_t) noexcept {}

size_t SanaWmGlumbconvTempPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                   nvinfer1::PluginTensorDesc const*,
                                                   int32_t) const noexcept {
    return 0;
}

void SanaWmGlumbconvTempPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(inverted_weight_device_);
    free_device_cache(inverted_bias_device_);
    free_device_cache(depth_weight_device_);
    free_device_cache(depth_bias_device_);
    free_device_cache(point_weight_device_);
    free_device_cache(t_weight_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmGlumbconvTempPlugin::ensureDeviceCache(cudaStream_t stream,
                                                  int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    cached_device_ = device_index;
    if (!copy_to_device_cache(inverted_weight_device_, inverted_weight_.data(),
                              inverted_weight_.size() * sizeof(uint16_t), stream))
        return false;
    if (!inverted_bias_.empty() &&
        !copy_to_device_cache(inverted_bias_device_, inverted_bias_.data(),
                              inverted_bias_.size() * sizeof(uint16_t), stream))
        return false;
    if (!copy_to_device_cache(depth_weight_device_, depth_weight_.data(),
                              depth_weight_.size() * sizeof(uint16_t), stream))
        return false;
    if (!depth_bias_.empty() &&
        !copy_to_device_cache(depth_bias_device_, depth_bias_.data(),
                              depth_bias_.size() * sizeof(uint16_t), stream))
        return false;
    if (!copy_to_device_cache(point_weight_device_, point_weight_.data(),
                              point_weight_.size() * sizeof(uint16_t), stream))
        return false;
    if (!copy_to_device_cache(t_weight_device_, t_weight_.data(),
                              t_weight_.size() * sizeof(uint16_t), stream))
        return false;
    return true;
}

int32_t SanaWmGlumbconvTempPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                           nvinfer1::PluginTensorDesc const*,
                                           void const* const* inputs, void* const* outputs, void*,
                                           cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        const int32_t token_count = frames_ * height_ * width_;
        if (dims.nbDims != 3 || dims.d[0] != batch_ || dims.d[1] != token_count ||
            dims.d[2] != channels_)
            return 1;
        const int32_t expanded = hidden_ * 2;
        if (static_cast<int32_t>(inverted_weight_.size()) != expanded * channels_ ||
            static_cast<int32_t>(depth_weight_.size()) != expanded * 3 * 3 ||
            static_cast<int32_t>(point_weight_.size()) != channels_ * hidden_ ||
            static_cast<int32_t>(t_weight_.size()) != channels_ * channels_ * t_kernel_)
            return 1;

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        if (!ensureDeviceCache(stream, device_index))
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);

        auto input =
            at::from_blob(const_cast<void*>(inputs[0]), {batch_, token_count, channels_}, options);
        auto inverted_weight =
            at::from_blob(inverted_weight_device_, {expanded, channels_, 1, 1}, options);
        auto depth_weight = at::from_blob(depth_weight_device_, {expanded, 1, 3, 3}, options);
        auto point_weight =
            at::from_blob(point_weight_device_, {channels_, hidden_, 1, 1}, options);
        auto t_weight =
            at::from_blob(t_weight_device_, {channels_, channels_, t_kernel_, 1}, options);

        const std::array<int64_t, 2> stride{1, 1};
        const std::array<int64_t, 2> dilation{1, 1};
        const std::array<int64_t, 2> pad_none{0, 0};
        const std::array<int64_t, 2> pad_depth{1, 1};
        const std::array<int64_t, 2> pad_temporal{t_kernel_ / 2, 0};
        const int32_t rows = batch_ * frames_;
        const int32_t spatial = height_ * width_;

        auto x =
            input.reshape({batch_ * frames_, height_, width_, channels_}).permute({0, 3, 1, 2});
        auto inverted = at::conv2d(x, inverted_weight, std::nullopt, stride, pad_none, dilation, 1);
        if (launch_sana_wm_bias_silu(inverted.data_ptr(), inverted_bias_device_, rows, spatial,
                                     expanded, stream) != 0) {
            return 1;
        }
        auto depth =
            at::conv2d(inverted, depth_weight, std::nullopt, stride, pad_depth, dilation, expanded);
        auto gated =
            at::empty({rows, hidden_, height_, width_}, options, at::MemoryFormat::ChannelsLast);
        if (launch_sana_wm_gated_silu(gated.data_ptr(), depth.data_ptr(), depth_bias_device_, rows,
                                      spatial, hidden_, stream) != 0) {
            return 1;
        }
        auto point = at::conv2d(gated, point_weight, std::nullopt, stride, pad_none, dilation, 1);
        auto point_bcts = point.view({batch_, frames_, channels_, spatial}).permute({0, 2, 1, 3});
        auto temporal =
            at::conv2d(point_bcts, t_weight, std::nullopt, stride, pad_temporal, dilation, 1);
        auto out =
            (point_bcts + temporal).permute({0, 2, 3, 1}).reshape({batch_, token_count, channels_});
        auto output_view = at::from_blob(outputs[0], {batch_, token_count, channels_}, options);
        output_view.copy_(out);
        return 0;
    } catch (const c10::Error& e) {
        report_glumbconvtemp_error("aten glumbconvtemp", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_glumbconvtemp_error("aten glumbconvtemp", e.what());
        return 1;
    }
}

SanaWmTimestepEmbedPlugin::SanaWmTimestepEmbedPlugin(
    int32_t frequency_dim, int32_t hidden_size, const float* freqs, int32_t freqs_count,
    const float* w0, int32_t w0_count, const float* b0, int32_t b0_count, const float* w1,
    int32_t w1_count, const float* b1, int32_t b1_count, const float* w2, int32_t w2_count,
    const float* b2, int32_t b2_count)
    : frequency_dim_(frequency_dim), hidden_size_(hidden_size) {
    const int32_t half = frequency_dim_ / 2;
    if (frequency_dim_ > 0 && hidden_size_ > 0 && frequency_dim_ == half * 2 &&
        freqs_count == half) {
        append_float_values(freqs_, freqs, freqs_count);
    }
    if (w0_count == frequency_dim_ * hidden_size_)
        append_bf16_values(w0_, w0, w0_count);
    if (b0_count == hidden_size_)
        append_bf16_values(b0_, b0, b0_count);
    if (w1_count == hidden_size_ * hidden_size_)
        append_bf16_values(w1_, w1, w1_count);
    if (b1_count == hidden_size_)
        append_bf16_values(b1_, b1, b1_count);
    if (w2_count == hidden_size_ * 6 * hidden_size_)
        append_bf16_values(w2_, w2, w2_count);
    if (b2_count == 6 * hidden_size_)
        append_bf16_values(b2_, b2, b2_count);
}

SanaWmTimestepEmbedPlugin::SanaWmTimestepEmbedPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 2;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    frequency_dim_ = header[0];
    hidden_size_ = header[1];
    const int32_t half = frequency_dim_ / 2;
    if (frequency_dim_ <= 0 || hidden_size_ <= 0 || frequency_dim_ != half * 2)
        return;
    const int32_t w0_count = frequency_dim_ * hidden_size_;
    const int32_t b0_count = hidden_size_;
    const int32_t w1_count = hidden_size_ * hidden_size_;
    const int32_t b1_count = hidden_size_;
    const int32_t w2_count = hidden_size_ * 6 * hidden_size_;
    const int32_t b2_count = 6 * hidden_size_;
    const size_t expected =
        kHeaderCount * sizeof(int32_t) + static_cast<size_t>(half) * sizeof(float) +
        static_cast<size_t>(w0_count + b0_count + w1_count + b1_count + w2_count + b2_count) *
            sizeof(uint16_t);
    if (length < expected)
        return;

    const char* cursor = static_cast<const char*>(data) + kHeaderCount * sizeof(int32_t);
    const auto* freqs = reinterpret_cast<const float*>(cursor);
    freqs_.assign(freqs, freqs + half);
    cursor += static_cast<size_t>(half) * sizeof(float);

    auto read_bf16 = [&cursor](std::vector<uint16_t>& out, int32_t count) {
        const auto* values = reinterpret_cast<const uint16_t*>(cursor);
        out.assign(values, values + count);
        cursor += static_cast<size_t>(count) * sizeof(uint16_t);
    };
    read_bf16(w0_, w0_count);
    read_bf16(b0_, b0_count);
    read_bf16(w1_, w1_count);
    read_bf16(b1_, b1_count);
    read_bf16(w2_, w2_count);
    read_bf16(b2_, b2_count);
}

char const* SanaWmTimestepEmbedPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmTimestepEmbedPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmTimestepEmbedPlugin::getNbOutputs() const noexcept {
    return 2;
}

int32_t SanaWmTimestepEmbedPlugin::initialize() noexcept {
    return 0;
}

void SanaWmTimestepEmbedPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmTimestepEmbedPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmTimestepEmbedPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 2;
    return kHeaderCount * sizeof(int32_t) + freqs_.size() * sizeof(float) +
           (w0_.size() + b0_.size() + w1_.size() + b1_.size() + w2_.size() + b2_.size()) *
               sizeof(uint16_t);
}

void SanaWmTimestepEmbedPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 2;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = frequency_dim_;
    header[1] = hidden_size_;
    char* cursor = static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t);
    if (!freqs_.empty()) {
        std::memcpy(cursor, freqs_.data(), freqs_.size() * sizeof(float));
        cursor += freqs_.size() * sizeof(float);
    }
    auto write_bf16 = [&cursor](const std::vector<uint16_t>& values) {
        if (!values.empty()) {
            std::memcpy(cursor, values.data(), values.size() * sizeof(uint16_t));
            cursor += values.size() * sizeof(uint16_t);
        }
    };
    write_bf16(w0_);
    write_bf16(b0_);
    write_bf16(w1_);
    write_bf16(b1_);
    write_bf16(w2_);
    write_bf16(b2_);
}

void SanaWmTimestepEmbedPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmTimestepEmbedPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmTimestepEmbedPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmTimestepEmbedPlugin* SanaWmTimestepEmbedPlugin::clone() const noexcept {
    auto* p = new SanaWmTimestepEmbedPlugin();
    p->frequency_dim_ = frequency_dim_;
    p->hidden_size_ = hidden_size_;
    p->freqs_ = freqs_;
    p->w0_ = w0_;
    p->b0_ = b0_;
    p->w1_ = w1_;
    p->b1_ = b1_;
    p->w2_ = w2_;
    p->b2_ = b2_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmTimestepEmbedPlugin::getOutputDimensions(int32_t outputIndex,
                                               nvinfer1::DimsExprs const* inputs, int32_t,
                                               nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out;
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = inputs[0].d[2];
    out.d[3] = exprBuilder.constant(outputIndex == 0 ? hidden_size_ : 6 * hidden_size_);
    return out;
}

bool SanaWmTimestepEmbedPlugin::supportsFormatCombination(int32_t pos,
                                                          nvinfer1::PluginTensorDesc const* inOut,
                                                          int32_t, int32_t) noexcept {
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (pos == 0)
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    return inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmTimestepEmbedPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                nvinfer1::DynamicPluginTensorDesc const*,
                                                int32_t) noexcept {}

size_t SanaWmTimestepEmbedPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                   nvinfer1::PluginTensorDesc const*,
                                                   int32_t) const noexcept {
    return 0;
}

void SanaWmTimestepEmbedPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(freqs_device_);
    free_device_cache(w0_device_);
    free_device_cache(b0_device_);
    free_device_cache(w1_device_);
    free_device_cache(b1_device_);
    free_device_cache(w2_device_);
    free_device_cache(b2_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmTimestepEmbedPlugin::ensureDeviceCache(cudaStream_t stream,
                                                  int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(freqs_device_, freqs_.data(), freqs_.size() * sizeof(float),
                                stream) &&
           copy_to_device_cache(w0_device_, w0_.data(), w0_.size() * sizeof(uint16_t), stream) &&
           copy_to_device_cache(b0_device_, b0_.data(), b0_.size() * sizeof(uint16_t), stream) &&
           copy_to_device_cache(w1_device_, w1_.data(), w1_.size() * sizeof(uint16_t), stream) &&
           copy_to_device_cache(b1_device_, b1_.data(), b1_.size() * sizeof(uint16_t), stream) &&
           copy_to_device_cache(w2_device_, w2_.data(), w2_.size() * sizeof(uint16_t), stream) &&
           copy_to_device_cache(b2_device_, b2_.data(), b2_.size() * sizeof(uint16_t), stream);
}

int32_t SanaWmTimestepEmbedPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                           nvinfer1::PluginTensorDesc const*,
                                           void const* const* inputs, void* const* outputs, void*,
                                           cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        const int32_t half = frequency_dim_ / 2;
        const int32_t w0_count = frequency_dim_ * hidden_size_;
        const int32_t w1_count = hidden_size_ * hidden_size_;
        const int32_t w2_count = hidden_size_ * 6 * hidden_size_;
        if (dims.nbDims != 3 || dims.d[1] != 1 || frequency_dim_ != half * 2 ||
            static_cast<int32_t>(freqs_.size()) != half ||
            static_cast<int32_t>(w0_.size()) != w0_count ||
            static_cast<int32_t>(b0_.size()) != hidden_size_ ||
            static_cast<int32_t>(w1_.size()) != w1_count ||
            static_cast<int32_t>(b1_.size()) != hidden_size_ ||
            static_cast<int32_t>(w2_.size()) != w2_count ||
            static_cast<int32_t>(b2_.size()) != 6 * hidden_size_) {
            return 1;
        }
        const int32_t batch = dims.d[0];
        const int32_t frames = dims.d[2];
        if (batch <= 0 || frames <= 0)
            return 1;

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        if (!ensureDeviceCache(stream, device_index))
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto f32_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);

        auto timestep = at::from_blob(const_cast<void*>(inputs[0]), {batch, 1, frames}, f32_options)
                            .reshape({batch * frames})
                            .to(at::kLong)
                            .to(at::kFloat);
        auto freqs = at::from_blob(freqs_device_, {half}, f32_options);
        auto args = timestep.reshape({batch * frames, 1}) * freqs.reshape({1, half});
        auto t_freq = at::cat({at::cos(args), at::sin(args)}, -1).to(at::kBFloat16);

        auto w0_trt = at::from_blob(w0_device_, {frequency_dim_, hidden_size_}, bf16_options);
        auto b0 = at::from_blob(b0_device_, {hidden_size_}, bf16_options);
        auto h0 = at::linear(t_freq, w0_trt.transpose(0, 1).contiguous(), b0);
        auto h1 = at::silu(h0);

        auto w1_trt = at::from_blob(w1_device_, {hidden_size_, hidden_size_}, bf16_options);
        auto b1 = at::from_blob(b1_device_, {hidden_size_}, bf16_options);
        auto t = at::linear(h1, w1_trt.transpose(0, 1).contiguous(), b1);

        auto w2_trt = at::from_blob(w2_device_, {hidden_size_, 6 * hidden_size_}, bf16_options);
        auto b2 = at::from_blob(b2_device_, {6 * hidden_size_}, bf16_options);
        auto t0 = at::linear(at::silu(t), w2_trt.transpose(0, 1).contiguous(), b2);

        auto output_t = at::from_blob(outputs[0], {batch, 1, frames, hidden_size_}, bf16_options);
        output_t.copy_(t.reshape({batch, 1, frames, hidden_size_}));
        auto output_t0 =
            at::from_blob(outputs[1], {batch, 1, frames, 6 * hidden_size_}, bf16_options);
        output_t0.copy_(t0.reshape({batch, 1, frames, 6 * hidden_size_}));
        return 0;
    } catch (const c10::Error& e) {
        report_timestep_error("aten timestep embed", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_timestep_error("aten timestep embed", e.what());
        return 1;
    }
}

SanaWmT2IModulatePlugin::SanaWmT2IModulatePlugin(const void*, size_t) {}

char const* SanaWmT2IModulatePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmT2IModulatePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmT2IModulatePlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmT2IModulatePlugin::initialize() noexcept {
    return 0;
}

void SanaWmT2IModulatePlugin::terminate() noexcept {}

void SanaWmT2IModulatePlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmT2IModulatePlugin::getSerializationSize() const noexcept {
    return 0;
}

void SanaWmT2IModulatePlugin::serialize(void*) const noexcept {}

void SanaWmT2IModulatePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmT2IModulatePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmT2IModulatePlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                              int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmT2IModulatePlugin* SanaWmT2IModulatePlugin::clone() const noexcept {
    auto* p = new SanaWmT2IModulatePlugin();
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs SanaWmT2IModulatePlugin::getOutputDimensions(int32_t,
                                                                 nvinfer1::DimsExprs const* inputs,
                                                                 int32_t,
                                                                 nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmT2IModulatePlugin::supportsFormatCombination(int32_t pos,
                                                        nvinfer1::PluginTensorDesc const* inOut,
                                                        int32_t, int32_t) noexcept {
    (void)pos;
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmT2IModulatePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                              nvinfer1::DynamicPluginTensorDesc const*,
                                              int32_t) noexcept {}

size_t SanaWmT2IModulatePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                 nvinfer1::PluginTensorDesc const*,
                                                 int32_t) const noexcept {
    return 0;
}

int32_t SanaWmT2IModulatePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                         nvinfer1::PluginTensorDesc const*,
                                         void const* const* inputs, void* const* outputs, void*,
                                         cudaStream_t stream) noexcept {
    const auto x_dims = inputDesc[0].dims;
    const auto shift_dims = inputDesc[1].dims;
    const auto scale_dims = inputDesc[2].dims;
    if (x_dims.nbDims != 4 || shift_dims.nbDims != 4 || scale_dims.nbDims != 4)
        return 1;
    const int32_t batch = x_dims.d[0];
    const int32_t frames = x_dims.d[1];
    const int32_t tokens = x_dims.d[2];
    const int32_t hidden = x_dims.d[3];
    if (batch <= 0 || frames <= 0 || tokens <= 0 || hidden <= 0)
        return 1;
    if (shift_dims.d[0] != batch || shift_dims.d[1] != frames || shift_dims.d[2] != 1 ||
        shift_dims.d[3] != hidden || scale_dims.d[0] != batch || scale_dims.d[1] != frames ||
        scale_dims.d[2] != 1 || scale_dims.d[3] != hidden) {
        return 1;
    }
    return launch_sana_wm_t2i_modulate(outputs[0], inputs[0], inputs[1], inputs[2], batch, frames,
                                       tokens, hidden, stream);
}

SanaWmCaptionEmbedPlugin::SanaWmCaptionEmbedPlugin(int32_t hidden_size)
    : hidden_size_(hidden_size) {}

SanaWmCaptionEmbedPlugin::SanaWmCaptionEmbedPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(hidden_size_))
        std::memcpy(&hidden_size_, data, sizeof(hidden_size_));
}

char const* SanaWmCaptionEmbedPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmCaptionEmbedPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmCaptionEmbedPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmCaptionEmbedPlugin::initialize() noexcept {
    return 0;
}

void SanaWmCaptionEmbedPlugin::terminate() noexcept {}

void SanaWmCaptionEmbedPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmCaptionEmbedPlugin::getSerializationSize() const noexcept {
    return sizeof(hidden_size_);
}

void SanaWmCaptionEmbedPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &hidden_size_, sizeof(hidden_size_));
}

void SanaWmCaptionEmbedPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmCaptionEmbedPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmCaptionEmbedPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                               int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmCaptionEmbedPlugin* SanaWmCaptionEmbedPlugin::clone() const noexcept {
    auto* p = new SanaWmCaptionEmbedPlugin(hidden_size_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmCaptionEmbedPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                              nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out = inputs[0];
    if (out.nbDims > 0)
        out.d[out.nbDims - 1] = exprBuilder.constant(hidden_size_);
    return out;
}

bool SanaWmCaptionEmbedPlugin::supportsFormatCombination(int32_t pos,
                                                         nvinfer1::PluginTensorDesc const* inOut,
                                                         int32_t nbInputs, int32_t) noexcept {
    return nbInputs == 6 && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmCaptionEmbedPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                               nvinfer1::DynamicPluginTensorDesc const*,
                                               int32_t) noexcept {}

size_t SanaWmCaptionEmbedPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                  nvinfer1::PluginTensorDesc const*,
                                                  int32_t) const noexcept {
    return 0;
}

int32_t SanaWmCaptionEmbedPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                          nvinfer1::PluginTensorDesc const*,
                                          void const* const* inputs, void* const* outputs, void*,
                                          cudaStream_t stream) noexcept {
    try {
        const auto y_dims = inputDesc[0].dims;
        if (y_dims.nbDims != 4 || hidden_size_ <= 0)
            return 1;
        const int32_t batch = y_dims.d[0];
        const int32_t groups = y_dims.d[1];
        const int32_t length = y_dims.d[2];
        const int32_t input_dim = y_dims.d[3];
        if (batch <= 0 || groups <= 0 || length <= 0 || input_dim <= 0)
            return 1;
        const auto w0_dims = inputDesc[1].dims;
        const auto b0_dims = inputDesc[2].dims;
        const auto w1_dims = inputDesc[3].dims;
        const auto b1_dims = inputDesc[4].dims;
        const auto norm_dims = inputDesc[5].dims;
        if (w0_dims.nbDims != 2 || w0_dims.d[0] != input_dim || w0_dims.d[1] != hidden_size_ ||
            b0_dims.nbDims != 1 || b0_dims.d[0] != hidden_size_ || w1_dims.nbDims != 2 ||
            w1_dims.d[0] != hidden_size_ || w1_dims.d[1] != hidden_size_ || b1_dims.nbDims != 1 ||
            b1_dims.d[0] != hidden_size_ || norm_dims.nbDims != 1 ||
            norm_dims.d[0] != hidden_size_) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto y = at::from_blob(const_cast<void*>(inputs[0]), {batch, groups, length, input_dim},
                               options);
        auto w0 = at::from_blob(const_cast<void*>(inputs[1]), {input_dim, hidden_size_}, options);
        auto b0 = at::from_blob(const_cast<void*>(inputs[2]), {hidden_size_}, options);
        auto w1 =
            at::from_blob(const_cast<void*>(inputs[3]), {hidden_size_, hidden_size_}, options);
        auto b1 = at::from_blob(const_cast<void*>(inputs[4]), {hidden_size_}, options);
        auto norm_weight = at::from_blob(const_cast<void*>(inputs[5]), {hidden_size_}, options);

        auto result = at::linear(y, w0.transpose(0, 1).contiguous(), b0);
        result = at::gelu(result, "tanh");
        result = at::linear(result, w1.transpose(0, 1).contiguous(), b1);
        auto result_float = result.to(at::kFloat);
        auto normalized = result_float * at::rsqrt(result_float.pow(2).mean(-1, true) + 1.0e-5);
        result = (norm_weight.to(at::kFloat) * normalized).to(at::kBFloat16);

        auto output = at::from_blob(outputs[0], {batch, groups, length, hidden_size_}, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_caption_embed_error("aten caption embed", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_caption_embed_error("aten caption embed", e.what());
        return 1;
    }
}

SanaWmCrossAttentionPlugin::SanaWmCrossAttentionPlugin(int32_t num_heads) : num_heads_(num_heads) {}

SanaWmCrossAttentionPlugin::SanaWmCrossAttentionPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(num_heads_))
        std::memcpy(&num_heads_, data, sizeof(num_heads_));
}

char const* SanaWmCrossAttentionPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmCrossAttentionPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmCrossAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmCrossAttentionPlugin::initialize() noexcept {
    return 0;
}

void SanaWmCrossAttentionPlugin::terminate() noexcept {}

void SanaWmCrossAttentionPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmCrossAttentionPlugin::getSerializationSize() const noexcept {
    return sizeof(num_heads_);
}

void SanaWmCrossAttentionPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &num_heads_, sizeof(num_heads_));
}

void SanaWmCrossAttentionPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmCrossAttentionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmCrossAttentionPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                 int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmCrossAttentionPlugin* SanaWmCrossAttentionPlugin::clone() const noexcept {
    auto* p = new SanaWmCrossAttentionPlugin(num_heads_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmCrossAttentionPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmCrossAttentionPlugin::supportsFormatCombination(int32_t pos,
                                                           nvinfer1::PluginTensorDesc const* inOut,
                                                           int32_t nbInputs, int32_t) noexcept {
    if (nbInputs != 11 || inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    return inOut[pos].type == (pos == 2 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kBF16);
}

void SanaWmCrossAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) noexcept {}

size_t SanaWmCrossAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                    nvinfer1::PluginTensorDesc const*,
                                                    int32_t) const noexcept {
    return 0;
}

int32_t SanaWmCrossAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                            nvinfer1::PluginTensorDesc const*,
                                            void const* const* inputs, void* const* outputs, void*,
                                            cudaStream_t stream) noexcept {
    try {
        const auto x_dims = inputDesc[0].dims;
        const auto cond_dims = inputDesc[1].dims;
        const auto mask_dims = inputDesc[2].dims;
        if (x_dims.nbDims != 3 || cond_dims.nbDims != 4 || mask_dims.nbDims != 2 ||
            num_heads_ <= 0) {
            return 1;
        }
        const int32_t batch = x_dims.d[0];
        const int32_t token_count = x_dims.d[1];
        const int32_t hidden = x_dims.d[2];
        const int32_t text_length = cond_dims.d[2];
        if (batch <= 0 || token_count <= 0 || hidden <= 0 || text_length <= 0 ||
            hidden % num_heads_ != 0 || cond_dims.d[0] != batch || cond_dims.d[1] != 1 ||
            cond_dims.d[3] != hidden || mask_dims.d[0] != batch || mask_dims.d[1] != text_length) {
            return 1;
        }
        const int32_t head_dim = hidden / num_heads_;
        const auto valid_matrix = [](const nvinfer1::Dims& dims, int32_t rows, int32_t cols) {
            return dims.nbDims == 2 && dims.d[0] == rows && dims.d[1] == cols;
        };
        const auto valid_vector = [](const nvinfer1::Dims& dims, int32_t size) {
            return dims.nbDims == 1 && dims.d[0] == size;
        };
        if (!valid_matrix(inputDesc[3].dims, hidden, hidden) ||
            !valid_vector(inputDesc[4].dims, hidden) ||
            !valid_matrix(inputDesc[5].dims, hidden, 2 * hidden) ||
            !valid_vector(inputDesc[6].dims, 2 * hidden) ||
            !valid_vector(inputDesc[7].dims, hidden) || !valid_vector(inputDesc[8].dims, hidden) ||
            !valid_matrix(inputDesc[9].dims, hidden, hidden) ||
            !valid_vector(inputDesc[10].dims, hidden)) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto int_options =
            at::TensorOptions().dtype(at::kInt).device(at::kCUDA, device_index);

        auto x =
            at::from_blob(const_cast<void*>(inputs[0]), {batch, token_count, hidden}, bf16_options);
        auto cond = at::from_blob(const_cast<void*>(inputs[1]), {batch, 1, text_length, hidden},
                                  bf16_options);
        auto mask = at::from_blob(const_cast<void*>(inputs[2]), {batch, text_length}, int_options);
        auto q_weight = at::from_blob(const_cast<void*>(inputs[3]), {hidden, hidden}, bf16_options);
        auto q_bias = at::from_blob(const_cast<void*>(inputs[4]), {hidden}, bf16_options);
        auto kv_weight =
            at::from_blob(const_cast<void*>(inputs[5]), {hidden, 2 * hidden}, bf16_options);
        auto kv_bias = at::from_blob(const_cast<void*>(inputs[6]), {2 * hidden}, bf16_options);
        auto q_norm_weight = at::from_blob(const_cast<void*>(inputs[7]), {hidden}, bf16_options);
        auto k_norm_weight = at::from_blob(const_cast<void*>(inputs[8]), {hidden}, bf16_options);
        auto proj_weight =
            at::from_blob(const_cast<void*>(inputs[9]), {hidden, hidden}, bf16_options);
        auto proj_bias = at::from_blob(const_cast<void*>(inputs[10]), {hidden}, bf16_options);

        const auto rms_norm = [](const at::Tensor& value, const at::Tensor& weight) {
            auto value_float = value.to(at::kFloat);
            auto normalized = value_float * at::rsqrt(value_float.pow(2).mean(-1, true) + 1.0e-6);
            return (weight.to(at::kFloat) * normalized).to(at::kBFloat16);
        };

        auto q = at::matmul(x, q_weight) + q_bias;
        auto kv = at::linear(cond, kv_weight.transpose(0, 1).contiguous(), kv_bias)
                      .view({batch, -1, 2, hidden});
        auto k = kv.select(2, 0);
        auto v = kv.select(2, 1);
        q = rms_norm(q, q_norm_weight)
                .view({batch, token_count, num_heads_, head_dim})
                .transpose(1, 2);
        k = rms_norm(k, k_norm_weight)
                .view({batch, text_length, num_heads_, head_dim})
                .transpose(1, 2);
        v = v.view({batch, text_length, num_heads_, head_dim}).transpose(1, 2);

        auto mask_bf16 = mask.to(at::kBFloat16);
        auto attn_mask = (1 - mask_bf16) * -10000.0;
        attn_mask = attn_mask.view({batch, 1, 1, text_length}).repeat({1, num_heads_, 1, 1});
        auto result = at::scaled_dot_product_attention(q, k, v, attn_mask, 0.0, false)
                          .transpose(1, 2)
                          .reshape({batch, token_count, hidden});
        result = at::linear(result, proj_weight.transpose(0, 1).contiguous(), proj_bias);

        auto output = at::from_blob(outputs[0], {batch, token_count, hidden}, bf16_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_cross_attention_error("aten cross attention", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_cross_attention_error("aten cross attention", e.what());
        return 1;
    }
}

SanaWmSoftmaxAttentionPlugin::SanaWmSoftmaxAttentionPlugin(int32_t frames, int32_t spatial,
                                                           int32_t heads, int32_t head_dim,
                                                           float norm_eps, bool camera)
    : frames_(frames), spatial_(spatial), heads_(heads), head_dim_(head_dim), norm_eps_(norm_eps),
      camera_(camera) {}

SanaWmSoftmaxAttentionPlugin::SanaWmSoftmaxAttentionPlugin(const void* data, size_t length) {
    struct Header {
        int32_t frames;
        int32_t spatial;
        int32_t heads;
        int32_t head_dim;
        float norm_eps;
        int32_t camera;
    };
    if (data == nullptr || length < sizeof(Header))
        return;
    Header header{};
    std::memcpy(&header, data, sizeof(header));
    frames_ = header.frames;
    spatial_ = header.spatial;
    heads_ = header.heads;
    head_dim_ = header.head_dim;
    norm_eps_ = header.norm_eps;
    camera_ = header.camera != 0;
}

char const* SanaWmSoftmaxAttentionPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmSoftmaxAttentionPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmSoftmaxAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmSoftmaxAttentionPlugin::initialize() noexcept {
    return 0;
}

void SanaWmSoftmaxAttentionPlugin::terminate() noexcept {}

void SanaWmSoftmaxAttentionPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmSoftmaxAttentionPlugin::getSerializationSize() const noexcept {
    return 5 * sizeof(int32_t) + sizeof(float);
}

void SanaWmSoftmaxAttentionPlugin::serialize(void* buffer) const noexcept {
    char* cursor = static_cast<char*>(buffer);
    for (const int32_t value : {frames_, spatial_, heads_, head_dim_}) {
        std::memcpy(cursor, &value, sizeof(value));
        cursor += sizeof(value);
    }
    std::memcpy(cursor, &norm_eps_, sizeof(norm_eps_));
    cursor += sizeof(norm_eps_);
    const int32_t camera = camera_ ? 1 : 0;
    std::memcpy(cursor, &camera, sizeof(camera));
}

void SanaWmSoftmaxAttentionPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmSoftmaxAttentionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmSoftmaxAttentionPlugin::getOutputDataType(int32_t,
                                                                   nvinfer1::DataType const*,
                                                                   int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmSoftmaxAttentionPlugin* SanaWmSoftmaxAttentionPlugin::clone() const noexcept {
    auto* plugin =
        new SanaWmSoftmaxAttentionPlugin(frames_, spatial_, heads_, head_dim_, norm_eps_, camera_);
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs
SanaWmSoftmaxAttentionPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs,
                                                  int32_t, nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmSoftmaxAttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    const int32_t expected_inputs = camera_ ? 10 : 9;
    if (nbInputs != expected_inputs || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    if (pos < 5 || pos == nbInputs)
        return inOut[pos].type == nvinfer1::DataType::kBF16;
    return inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

void SanaWmSoftmaxAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*,
                                                   int32_t,
                                                   nvinfer1::DynamicPluginTensorDesc const*,
                                                   int32_t) noexcept {}

size_t SanaWmSoftmaxAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                      nvinfer1::PluginTensorDesc const*,
                                                      int32_t) const noexcept {
    return 0;
}

int32_t SanaWmSoftmaxAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                              nvinfer1::PluginTensorDesc const*,
                                              void const* const* inputs, void* const* outputs,
                                              void*, cudaStream_t stream) noexcept {
    try {
        const auto raw_dims = inputDesc[0].dims;
        if (raw_dims.nbDims != 3 || frames_ <= 0 || spatial_ <= 0 || heads_ <= 0 ||
            head_dim_ <= 0 || head_dim_ % 4 != 0) {
            return 1;
        }
        const int32_t batch = raw_dims.d[0];
        const int32_t tokens = raw_dims.d[1];
        const int32_t channels = raw_dims.d[2];
        if (batch <= 0 || tokens != frames_ * spatial_ || channels != heads_ * head_dim_)
            return 1;
        for (int32_t i = 1; i < 3; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 3 || dims.d[0] != batch || dims.d[1] != tokens ||
                dims.d[2] != channels) {
                return 1;
            }
        }
        for (int32_t i = 3; i < 5; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 1 || dims.d[0] != channels)
                return 1;
        }

        const int32_t matrix_index = camera_ ? 5 : -1;
        const int32_t rope_index = camera_ ? 6 : 5;
        if (camera_) {
            const auto dims = inputDesc[matrix_index].dims;
            if (dims.nbDims != 4 || dims.d[0] != batch || dims.d[1] != tokens || dims.d[2] != 4 ||
                dims.d[3] != 4) {
                return 1;
            }
        }
        const int32_t rope_pairs = camera_ ? head_dim_ / 4 : head_dim_ / 2;
        for (int32_t i = rope_index; i < rope_index + 4; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 2 || dims.d[0] != tokens || dims.d[1] != rope_pairs)
                return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);

        auto q_raw =
            at::from_blob(const_cast<void*>(inputs[0]), {batch, tokens, channels}, bf16_options);
        auto k_raw =
            at::from_blob(const_cast<void*>(inputs[1]), {batch, tokens, channels}, bf16_options);
        auto v_raw =
            at::from_blob(const_cast<void*>(inputs[2]), {batch, tokens, channels}, bf16_options);
        auto q_weight = at::from_blob(const_cast<void*>(inputs[3]), {channels}, bf16_options);
        auto k_weight = at::from_blob(const_cast<void*>(inputs[4]), {channels}, bf16_options);
        auto cos_high = at::from_blob(const_cast<void*>(inputs[rope_index]), {tokens, rope_pairs},
                                      float_options);
        auto sin_high = at::from_blob(const_cast<void*>(inputs[rope_index + 1]),
                                      {tokens, rope_pairs}, float_options);
        auto cos_low = at::from_blob(const_cast<void*>(inputs[rope_index + 2]),
                                     {tokens, rope_pairs}, float_options);
        auto sin_low = at::from_blob(const_cast<void*>(inputs[rope_index + 3]),
                                     {tokens, rope_pairs}, float_options);
        auto cos_double = cos_high.to(at::kDouble) + cos_low.to(at::kDouble);
        auto sin_double = sin_high.to(at::kDouble) + sin_low.to(at::kDouble);

        const auto rms_norm = [this](const at::Tensor& value, const at::Tensor& weight) {
            auto value_float = value.to(at::kFloat);
            auto normalized = value_float * at::rsqrt(value_float.pow(2).mean(-1, true) +
                                                      static_cast<double>(norm_eps_));
            return (normalized * weight.to(at::kFloat)).to(value.scalar_type());
        };
        const auto apply_rope_bnhd = [batch, tokens, this, &cos_double,
                                      &sin_double](const at::Tensor& value) {
            auto pairs = value.to(at::kDouble)
                             .contiguous()
                             .reshape({batch, tokens, heads_, head_dim_ / 2, 2});
            auto complex_value = at::view_as_complex(pairs);
            auto frequency =
                at::complex(cos_double, sin_double).view({1, tokens, 1, head_dim_ / 2});
            return at::view_as_real(complex_value * frequency)
                .flatten(-2, -1)
                .to(value.scalar_type());
        };
        const auto apply_rope_bhnd = [tokens, rope_pairs, &cos_double,
                                      &sin_double](const at::Tensor& value, bool inverse) {
            auto sizes = value.sizes();
            auto pairs = value.to(at::kDouble)
                             .contiguous()
                             .reshape({sizes[0], sizes[1], sizes[2], rope_pairs, 2});
            auto complex_value = at::view_as_complex(pairs);
            auto sin = inverse ? -sin_double : sin_double;
            auto frequency = at::complex(cos_double, sin).view({1, 1, tokens, rope_pairs});
            return at::view_as_real(complex_value * frequency)
                .flatten(-2, -1)
                .to(value.scalar_type());
        };

        at::Tensor result;
        if (!camera_) {
            auto q = apply_rope_bnhd(
                         rms_norm(q_raw, q_weight).reshape({batch, tokens, heads_, head_dim_}))
                         .transpose(1, 2);
            auto k = apply_rope_bnhd(
                         rms_norm(k_raw, k_weight).reshape({batch, tokens, heads_, head_dim_}))
                         .transpose(1, 2);
            auto v = v_raw.reshape({batch, tokens, heads_, head_dim_}).transpose(1, 2);
            result = at::scaled_dot_product_attention(q, k, v, std::nullopt, 0.0, false)
                         .transpose(1, 2)
                         .reshape({batch, tokens, channels});
        } else {
            auto p = at::from_blob(const_cast<void*>(inputs[matrix_index]), {batch, tokens, 4, 4},
                                   float_options)
                         .to(at::kBFloat16);
            auto p_t = p.transpose(-1, -2);
            auto rotation = p.slice(-2, 0, 3).slice(-1, 0, 3);
            auto rotation_inv = rotation.transpose(-1, -2);
            auto translation = p.slice(-2, 0, 3).select(-1, 3);
            auto translation_inv = -at::einsum("bnij,bnj->bni", {rotation_inv, translation});
            auto upper = at::cat({rotation_inv, translation_inv.unsqueeze(-1)}, -1);
            auto lower = at::cat({at::zeros({batch, tokens, 1, 3}, p.options()),
                                  at::ones({batch, tokens, 1, 1}, p.options())},
                                 -1);
            auto p_inv = at::cat({upper, lower}, -2);

            const auto apply_ray = [batch, tokens](const at::Tensor& value,
                                                   const at::Tensor& matrix) {
                const int64_t num_heads = value.size(1);
                const int64_t channels = value.size(-1);
                return at::einsum(
                           "bnij,bhnkj->bhnki",
                           {matrix, value.reshape({batch, num_heads, tokens, channels / 4, 4})})
                    .reshape({batch, num_heads, tokens, channels});
            };
            const auto apply_camera = [this, &apply_ray, &apply_rope_bhnd](const at::Tensor& value,
                                                                           const at::Tensor& matrix,
                                                                           bool inverse_rope) {
                const int32_t half = head_dim_ / 2;
                auto geometric = apply_ray(value.slice(-1, 0, half), matrix);
                auto rope = apply_rope_bhnd(value.slice(-1, half, head_dim_), inverse_rope);
                return at::cat({geometric, rope}, -1);
            };

            auto q_bhnd = rms_norm(q_raw, q_weight)
                              .reshape({batch, tokens, heads_, head_dim_})
                              .permute({0, 2, 1, 3});
            auto k_bhnd = rms_norm(k_raw, k_weight)
                              .reshape({batch, tokens, heads_, head_dim_})
                              .permute({0, 2, 1, 3});
            auto v_bhnd = v_raw.reshape({batch, tokens, heads_, head_dim_}).permute({0, 2, 1, 3});
            auto q = apply_camera(q_bhnd, p_t, false);
            auto kv_bhnd = at::cat({k_bhnd, v_bhnd}, 1);
            auto kv_trans = apply_camera(kv_bhnd, p_inv, false);
            auto k = kv_trans.slice(1, 0, heads_);
            auto v = kv_trans.slice(1, heads_, 2 * heads_);
            const int32_t padded_dim = head_dim_ == 32 || head_dim_ == 64 || head_dim_ == 128 ||
                                               head_dim_ == 256 || head_dim_ >= 256
                                           ? head_dim_
                                           : (head_dim_ <= 128 ? 128 : 256);
            if (padded_dim != head_dim_) {
                const int32_t padding = padded_dim - head_dim_;
                q = at::constant_pad_nd(q, {0, padding}, 0.0);
                k = at::constant_pad_nd(k, {0, padding}, 0.0);
                v = at::constant_pad_nd(v, {0, padding}, 0.0);
            }
            auto out = at::scaled_dot_product_attention(q, k, v, std::nullopt, 0.0, false);
            if (padded_dim != head_dim_)
                out = out.slice(-1, 0, head_dim_);
            auto out_trans = apply_camera(out, p, true);
            auto output =
                at::from_blob(outputs[0], {batch, tokens, heads_, head_dim_}, bf16_options);
            output.copy_(out_trans.permute({0, 2, 1, 3}));
            return 0;
        }

        auto output = at::from_blob(outputs[0], {batch, tokens, channels}, bf16_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_softmax_attention_error("aten softmax attention", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_softmax_attention_error("aten softmax attention", e.what());
        return 1;
    }
}

SanaWmTorchCamPrepPlugin::SanaWmTorchCamPrepPlugin(int32_t frames, int32_t spatial, int32_t heads,
                                                   int32_t head_dim, float norm_eps)
    : frames_(frames), spatial_(spatial), heads_(heads), head_dim_(head_dim), norm_eps_(norm_eps) {}

SanaWmTorchCamPrepPlugin::SanaWmTorchCamPrepPlugin(const void* data, size_t length) {
    struct Header {
        int32_t frames;
        int32_t spatial;
        int32_t heads;
        int32_t head_dim;
        float norm_eps;
    };
    if (data == nullptr || length < sizeof(Header))
        return;
    Header header{};
    std::memcpy(&header, data, sizeof(header));
    frames_ = header.frames;
    spatial_ = header.spatial;
    heads_ = header.heads;
    head_dim_ = header.head_dim;
    norm_eps_ = header.norm_eps;
}

char const* SanaWmTorchCamPrepPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmTorchCamPrepPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmTorchCamPrepPlugin::getNbOutputs() const noexcept {
    return 4;
}

int32_t SanaWmTorchCamPrepPlugin::initialize() noexcept {
    return 0;
}

void SanaWmTorchCamPrepPlugin::terminate() noexcept {}

void SanaWmTorchCamPrepPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmTorchCamPrepPlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int32_t) + sizeof(float);
}

void SanaWmTorchCamPrepPlugin::serialize(void* buffer) const noexcept {
    char* cursor = static_cast<char*>(buffer);
    for (const int32_t value : {frames_, spatial_, heads_, head_dim_}) {
        std::memcpy(cursor, &value, sizeof(value));
        cursor += sizeof(value);
    }
    std::memcpy(cursor, &norm_eps_, sizeof(norm_eps_));
}

void SanaWmTorchCamPrepPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmTorchCamPrepPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmTorchCamPrepPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                               int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmTorchCamPrepPlugin* SanaWmTorchCamPrepPlugin::clone() const noexcept {
    auto* p = new SanaWmTorchCamPrepPlugin(frames_, spatial_, heads_, head_dim_, norm_eps_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmTorchCamPrepPlugin::getOutputDimensions(int32_t outputIndex,
                                              nvinfer1::DimsExprs const* inputs, int32_t,
                                              nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out{};
    if (outputIndex == 3) {
        out.nbDims = 3;
        out.d[0] = inputs[0].d[0];
        out.d[1] = exprBuilder.constant(heads_);
        out.d[2] = inputs[0].d[1];
        return out;
    }
    out.nbDims = 4;
    out.d[0] = inputs[0].d[0];
    out.d[1] = exprBuilder.constant(heads_);
    out.d[2] = exprBuilder.constant(head_dim_);
    out.d[3] = inputs[0].d[1];
    return out;
}

bool SanaWmTorchCamPrepPlugin::supportsFormatCombination(int32_t pos,
                                                         nvinfer1::PluginTensorDesc const* inOut,
                                                         int32_t nbInputs, int32_t) noexcept {
    if (nbInputs != 9 || inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    return inOut[pos].type == (pos < 3 ? nvinfer1::DataType::kBF16 : nvinfer1::DataType::kFLOAT);
}

void SanaWmTorchCamPrepPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                               nvinfer1::DynamicPluginTensorDesc const*,
                                               int32_t) noexcept {}

size_t SanaWmTorchCamPrepPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                  nvinfer1::PluginTensorDesc const*,
                                                  int32_t) const noexcept {
    return 0;
}

int32_t SanaWmTorchCamPrepPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                          nvinfer1::PluginTensorDesc const*,
                                          void const* const* inputs, void* const* outputs, void*,
                                          cudaStream_t stream) noexcept {
    try {
        const auto raw_dims = inputDesc[0].dims;
        if (raw_dims.nbDims != 3 || frames_ <= 0 || spatial_ <= 0 || heads_ <= 0 ||
            head_dim_ <= 0 || head_dim_ % 8 != 0) {
            return 1;
        }
        const int32_t batch = raw_dims.d[0];
        const int32_t tokens = raw_dims.d[1];
        const int32_t channels = raw_dims.d[2];
        const int32_t half_dim = head_dim_ / 2;
        const int32_t groups = half_dim / 4;
        if (batch <= 0 || tokens != frames_ * spatial_ || channels != heads_ * head_dim_ ||
            groups <= 0) {
            return 1;
        }
        for (int32_t i = 1; i < 3; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 3 || dims.d[0] != batch || dims.d[1] != tokens ||
                dims.d[2] != channels) {
                return 1;
            }
        }
        for (int32_t i = 3; i < 5; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 4 || dims.d[0] != batch || dims.d[1] != tokens || dims.d[2] != 4 ||
                dims.d[3] != 4) {
                return 1;
            }
        }
        for (int32_t i = 5; i < 7; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 5 || dims.d[0] != 1 || dims.d[1] != 1 || dims.d[2] != tokens ||
                dims.d[3] != half_dim / 2 || dims.d[4] != 1) {
                return 1;
            }
        }
        for (int32_t i = 7; i < 9; ++i) {
            const auto dims = inputDesc[i].dims;
            if (dims.nbDims != 1 || dims.d[0] != channels)
                return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);

        auto q_raw = at::from_blob(const_cast<void*>(inputs[0]), {batch, tokens, heads_, head_dim_},
                                   bf16_options);
        auto k_raw = at::from_blob(const_cast<void*>(inputs[1]), {batch, tokens, heads_, head_dim_},
                                   bf16_options);
        auto v_raw = at::from_blob(const_cast<void*>(inputs[2]), {batch, tokens, heads_, head_dim_},
                                   bf16_options);
        auto proj_q =
            at::from_blob(const_cast<void*>(inputs[3]), {batch, tokens, 4, 4}, float_options);
        auto proj_kv =
            at::from_blob(const_cast<void*>(inputs[4]), {batch, tokens, 4, 4}, float_options);
        auto rope_cos_base =
            at::from_blob(const_cast<void*>(inputs[5]), {tokens, half_dim / 2}, float_options);
        auto rope_sin_base =
            at::from_blob(const_cast<void*>(inputs[6]), {tokens, half_dim / 2}, float_options);
        auto q_norm_weight = at::from_blob(const_cast<void*>(inputs[7]), {channels}, float_options)
                                 .view({1, 1, heads_, head_dim_});
        auto k_norm_weight = at::from_blob(const_cast<void*>(inputs[8]), {channels}, float_options)
                                 .view({1, 1, heads_, head_dim_});

        auto q32 = q_raw.to(at::kFloat);
        auto k32 = k_raw.to(at::kFloat);
        auto v32 = v_raw.to(at::kFloat);
        const std::vector<int64_t> norm_dims{-1, -2};
        auto q_inv = at::rsqrt(at::sum(q32 * q32, norm_dims) / static_cast<double>(channels) +
                               static_cast<double>(norm_eps_));
        auto k_inv = at::rsqrt(at::sum(k32 * k32, norm_dims) / static_cast<double>(channels) +
                               static_cast<double>(norm_eps_));
        auto q_normed = q32 * q_inv.view({batch, tokens, 1, 1}) * q_norm_weight;
        auto k_normed = k32 * k_inv.view({batch, tokens, 1, 1}) * k_norm_weight;
        q_normed = at::relu(q_normed);
        const double k_scale = std::pow(static_cast<double>(head_dim_), -0.5) *
                               std::pow(static_cast<double>(spatial_), -0.5);
        k_normed = at::relu(k_normed) * k_scale;

        auto q_first = q_normed.slice(-1, 0, half_dim).reshape({batch, tokens, heads_, groups, 4});
        auto k_first = k_normed.slice(-1, 0, half_dim).reshape({batch, tokens, heads_, groups, 4});
        auto v_first = v32.slice(-1, 0, half_dim).reshape({batch, tokens, heads_, groups, 4});
        const auto project_first_half = [batch, tokens, this, half_dim](const at::Tensor& value,
                                                                        const at::Tensor& matrix) {
            std::vector<at::Tensor> rows;
            rows.reserve(4);
            for (int64_t row = 0; row < 4; ++row) {
                auto matrix_row = matrix.select(-2, row);
                const auto matrix_value = [&matrix_row](int64_t col) {
                    return matrix_row.select(-1, col).unsqueeze(-1).unsqueeze(-1);
                };
                auto pair_02 = at::addcmul(value.select(-1, 2) * matrix_value(2),
                                           value.select(-1, 0), matrix_value(0));
                auto pair_13 = at::addcmul(value.select(-1, 3) * matrix_value(3),
                                           value.select(-1, 1), matrix_value(1));
                rows.push_back(pair_02 + pair_13);
            }
            return at::stack(rows, -1).reshape({batch, tokens, heads_, half_dim});
        };
        auto q_first_proj = project_first_half(q_first, proj_q);
        auto k_first_proj = project_first_half(k_first, proj_kv);
        auto v_first_proj = at::einsum("bnij,bnhgj->bnhgi", {proj_kv, v_first})
                                .reshape({batch, tokens, heads_, half_dim});

        const auto pair_swap = [batch, tokens, this, half_dim](const at::Tensor& value) {
            return value.reshape({batch, tokens, heads_, half_dim / 2, 2})
                .flip({-1})
                .reshape({batch, tokens, heads_, half_dim});
        };
        auto cos = at::cat({rope_cos_base.unsqueeze(-1), rope_cos_base.unsqueeze(-1)}, -1)
                       .reshape({1, tokens, 1, half_dim});
        auto sin = at::cat({-rope_sin_base.unsqueeze(-1), rope_sin_base.unsqueeze(-1)}, -1)
                       .reshape({1, tokens, 1, half_dim});
        auto q_second = q_normed.slice(-1, half_dim, head_dim_);
        auto k_second = k_normed.slice(-1, half_dim, head_dim_);
        auto v_second = v32.slice(-1, half_dim, head_dim_);
        const auto sum_squares_64 = [](const at::Tensor& values, bool adjacent) {
            auto padded = at::constant_pad_nd(values, {0, 64 - values.size(-1)}, 0.0);
            at::Tensor first;
            at::Tensor second;
            if (adjacent) {
                auto pairs =
                    padded.reshape({values.size(0), values.size(1), values.size(2), 32, 2});
                first = pairs.select(-1, 0);
                second = pairs.select(-1, 1);
            } else {
                first = padded.slice(-1, 0, 32);
                second = padded.slice(-1, 32, 64);
            }
            auto reduced = at::addcmul(second * second, first, first);
            for (int64_t stride = 16; stride >= 1; stride >>= 1) {
                reduced = reduced.slice(-1, 0, stride) + reduced.slice(-1, stride, 2 * stride);
            }
            return reduced.squeeze(-1);
        };
        auto pre_k_sq = sum_squares_64(k_first.reshape({batch, tokens, heads_, half_dim}), true) +
                        sum_squares_64(k_second, true);
        auto q_rope = at::addcmul(pair_swap(q_second) * sin, q_second, cos);
        auto k_rope = at::addcmul(pair_swap(k_second) * sin, k_second, cos);
        auto v_rope = v_second * cos + pair_swap(v_second) * sin;

        auto q_bnhd = at::cat({q_first_proj, q_rope}, -1);
        auto k_bnhd = at::cat({k_first_proj, k_rope}, -1);
        auto v_bnhd = at::cat({v_first_proj, v_rope}, -1);
        auto post_k_sq = sum_squares_64(k_first_proj, false) + sum_squares_64(k_rope, true);
        auto q_out = q_bnhd.to(at::kBFloat16).permute({0, 2, 3, 1}).contiguous();
        auto k_out = k_bnhd.to(at::kBFloat16).permute({0, 2, 3, 1}).contiguous();
        auto v_out = v_bnhd.to(at::kBFloat16).permute({0, 2, 3, 1}).contiguous();
        auto inflation = post_k_sq.clamp_min(1.0e-12) / pre_k_sq.clamp_min(1.0e-12);
        inflation = inflation.permute({0, 2, 1}).contiguous();

        const std::vector<int64_t> qkv_shape{batch, heads_, head_dim_, tokens};
        at::from_blob(outputs[0], qkv_shape, float_options).copy_(q_out.to(at::kFloat));
        at::from_blob(outputs[1], qkv_shape, float_options).copy_(k_out.to(at::kFloat));
        at::from_blob(outputs[2], qkv_shape, float_options).copy_(v_out.to(at::kFloat));
        at::from_blob(outputs[3], {batch, heads_, tokens}, float_options).copy_(inflation);
        return 0;
    } catch (const c10::Error& e) {
        report_torch_cam_prep_error("aten camera prep", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_torch_cam_prep_error("aten camera prep", e.what());
        return 1;
    }
}

SanaWmCameraBetaDiscountPlugin::SanaWmCameraBetaDiscountPlugin(int32_t frames, int32_t spatial,
                                                               int32_t heads)
    : frames_(frames), spatial_(spatial), heads_(heads) {}

SanaWmCameraBetaDiscountPlugin::SanaWmCameraBetaDiscountPlugin(const void* data, size_t length) {
    if (data == nullptr || length < 3 * sizeof(int32_t))
        return;
    const char* cursor = static_cast<const char*>(data);
    for (int32_t* value : {&frames_, &spatial_, &heads_}) {
        std::memcpy(value, cursor, sizeof(*value));
        cursor += sizeof(*value);
    }
}

char const* SanaWmCameraBetaDiscountPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmCameraBetaDiscountPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmCameraBetaDiscountPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmCameraBetaDiscountPlugin::initialize() noexcept {
    return 0;
}

void SanaWmCameraBetaDiscountPlugin::terminate() noexcept {}

void SanaWmCameraBetaDiscountPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmCameraBetaDiscountPlugin::getSerializationSize() const noexcept {
    return 3 * sizeof(int32_t);
}

void SanaWmCameraBetaDiscountPlugin::serialize(void* buffer) const noexcept {
    char* cursor = static_cast<char*>(buffer);
    for (const int32_t value : {frames_, spatial_, heads_}) {
        std::memcpy(cursor, &value, sizeof(value));
        cursor += sizeof(value);
    }
}

void SanaWmCameraBetaDiscountPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmCameraBetaDiscountPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmCameraBetaDiscountPlugin::getOutputDataType(int32_t,
                                                                     nvinfer1::DataType const*,
                                                                     int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmCameraBetaDiscountPlugin* SanaWmCameraBetaDiscountPlugin::clone() const noexcept {
    auto* plugin = new SanaWmCameraBetaDiscountPlugin(frames_, spatial_, heads_);
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs
SanaWmCameraBetaDiscountPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs,
                                                    int32_t, nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmCameraBetaDiscountPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3 ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type == (pos == 0 ? nvinfer1::DataType::kBF16 : nvinfer1::DataType::kFLOAT);
}

void SanaWmCameraBetaDiscountPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*,
                                                     int32_t,
                                                     nvinfer1::DynamicPluginTensorDesc const*,
                                                     int32_t) noexcept {}

size_t SanaWmCameraBetaDiscountPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                        nvinfer1::PluginTensorDesc const*,
                                                        int32_t) const noexcept {
    return 0;
}

int32_t SanaWmCameraBetaDiscountPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                                nvinfer1::PluginTensorDesc const*,
                                                void const* const* inputs, void* const* outputs,
                                                void*, cudaStream_t stream) noexcept {
    try {
        const auto beta_dims = inputDesc[0].dims;
        const auto inflation_dims = inputDesc[1].dims;
        if (beta_dims.nbDims != 4 || inflation_dims.nbDims != 3 || frames_ <= 0 || spatial_ <= 0 ||
            heads_ <= 0) {
            return 1;
        }
        const int32_t batch = beta_dims.d[0];
        if (batch <= 0 || beta_dims.d[1] != heads_ || beta_dims.d[2] != frames_ ||
            beta_dims.d[3] != spatial_ || inflation_dims.d[0] != batch ||
            inflation_dims.d[1] != heads_ || inflation_dims.d[2] != frames_ * spatial_) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);

        auto beta = at::from_blob(const_cast<void*>(inputs[0]), {batch, heads_, frames_, spatial_},
                                  bf16_options);
        auto inflation = at::from_blob(const_cast<void*>(inputs[1]),
                                       {batch, heads_, frames_, spatial_}, float_options);
        auto frame_inflation = inflation.mean(-1).clamp_min(1.0);
        auto discounted = beta.to(at::kFloat) / frame_inflation.unsqueeze(-1);
        at::from_blob(outputs[0], {batch, heads_, frames_, spatial_}, float_options)
            .copy_(discounted);
        return 0;
    } catch (const c10::Error& e) {
        report_camera_beta_discount_error("aten beta discount", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_camera_beta_discount_error("aten beta discount", e.what());
        return 1;
    }
}

SanaWmFrameGatePlugin::SanaWmFrameGatePlugin(const void*, size_t) {}

char const* SanaWmFrameGatePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmFrameGatePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmFrameGatePlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmFrameGatePlugin::initialize() noexcept {
    return 0;
}

void SanaWmFrameGatePlugin::terminate() noexcept {}

void SanaWmFrameGatePlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmFrameGatePlugin::getSerializationSize() const noexcept {
    return 0;
}

void SanaWmFrameGatePlugin::serialize(void*) const noexcept {}

void SanaWmFrameGatePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmFrameGatePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmFrameGatePlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                            int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmFrameGatePlugin* SanaWmFrameGatePlugin::clone() const noexcept {
    auto* p = new SanaWmFrameGatePlugin();
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs SanaWmFrameGatePlugin::getOutputDimensions(int32_t,
                                                               nvinfer1::DimsExprs const* inputs,
                                                               int32_t,
                                                               nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmFrameGatePlugin::supportsFormatCombination(int32_t pos,
                                                      nvinfer1::PluginTensorDesc const* inOut,
                                                      int32_t nbInputs, int32_t) noexcept {
    return nbInputs == 2 && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmFrameGatePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmFrameGatePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

int32_t SanaWmFrameGatePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    try {
        const auto token_dims = inputDesc[0].dims;
        const auto gate_dims = inputDesc[1].dims;
        if (token_dims.nbDims != 3 || gate_dims.nbDims != 4)
            return 1;
        const int32_t batch = token_dims.d[0];
        const int32_t token_count = token_dims.d[1];
        const int32_t hidden = token_dims.d[2];
        const int32_t frames = gate_dims.d[1];
        if (batch <= 0 || token_count <= 0 || hidden <= 0 || frames <= 0 ||
            token_count % frames != 0 || gate_dims.d[0] != batch || gate_dims.d[2] != 1 ||
            gate_dims.d[3] != hidden) {
            return 1;
        }
        const int32_t spatial = token_count / frames;

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto tokens =
            at::from_blob(const_cast<void*>(inputs[0]), {batch, frames, spatial, hidden}, options);
        auto gate =
            at::from_blob(const_cast<void*>(inputs[1]), {batch, frames, 1, hidden}, options);
        auto result = (tokens * gate).reshape({batch, token_count, hidden});
        auto output = at::from_blob(outputs[0], {batch, token_count, hidden}, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_frame_gate_error("aten frame gate", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_frame_gate_error("aten frame gate", e.what());
        return 1;
    }
}

SanaWmFrameMeanPlugin::SanaWmFrameMeanPlugin(int32_t frames, int32_t spatial)
    : frames_(frames), spatial_(spatial) {}

SanaWmFrameMeanPlugin::SanaWmFrameMeanPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 2;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    frames_ = header[0];
    spatial_ = header[1];
}

char const* SanaWmFrameMeanPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmFrameMeanPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmFrameMeanPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmFrameMeanPlugin::initialize() noexcept {
    return 0;
}

void SanaWmFrameMeanPlugin::terminate() noexcept {}

void SanaWmFrameMeanPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmFrameMeanPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 2;
    return kHeaderCount * sizeof(int32_t);
}

void SanaWmFrameMeanPlugin::serialize(void* buffer) const noexcept {
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = frames_;
    header[1] = spatial_;
}

void SanaWmFrameMeanPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmFrameMeanPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmFrameMeanPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                            int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmFrameMeanPlugin* SanaWmFrameMeanPlugin::clone() const noexcept {
    auto* p = new SanaWmFrameMeanPlugin(frames_, spatial_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmFrameMeanPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                           nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto out = inputs[0];
    if (out.nbDims == 3)
        out.d[1] = exprBuilder.constant(frames_);
    return out;
}

bool SanaWmFrameMeanPlugin::supportsFormatCombination(int32_t pos,
                                                      nvinfer1::PluginTensorDesc const* inOut,
                                                      int32_t, int32_t) noexcept {
    (void)pos;
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmFrameMeanPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmFrameMeanPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

int32_t SanaWmFrameMeanPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        if (dims.nbDims != 3 || frames_ <= 0 || spatial_ <= 0)
            return 1;
        const int32_t batch = dims.d[0];
        const int32_t tokens = dims.d[1];
        const int32_t hidden = dims.d[2];
        if (batch <= 0 || tokens != frames_ * spatial_ || hidden <= 0)
            return 1;

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto x = at::from_blob(const_cast<void*>(inputs[0]), {batch, frames_, spatial_, hidden},
                               options);
        auto result = x.mean(2);
        auto output = at::from_blob(outputs[0], {batch, frames_, hidden}, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_frame_mean_error("aten mean", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_frame_mean_error("aten mean", e.what());
        return 1;
    }
}

SanaWmLayerNormPlugin::SanaWmLayerNormPlugin(float eps) : eps_(eps) {}

SanaWmLayerNormPlugin::SanaWmLayerNormPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(float))
        std::memcpy(&eps_, data, sizeof(float));
}

char const* SanaWmLayerNormPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmLayerNormPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmLayerNormPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmLayerNormPlugin::initialize() noexcept {
    return 0;
}

void SanaWmLayerNormPlugin::terminate() noexcept {}

void SanaWmLayerNormPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmLayerNormPlugin::getSerializationSize() const noexcept {
    return sizeof(float);
}

void SanaWmLayerNormPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &eps_, sizeof(float));
}

void SanaWmLayerNormPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmLayerNormPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmLayerNormPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                            int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmLayerNormPlugin* SanaWmLayerNormPlugin::clone() const noexcept {
    auto* p = new SanaWmLayerNormPlugin(eps_);
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs SanaWmLayerNormPlugin::getOutputDimensions(int32_t,
                                                               nvinfer1::DimsExprs const* inputs,
                                                               int32_t,
                                                               nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmLayerNormPlugin::supportsFormatCombination(int32_t pos,
                                                      nvinfer1::PluginTensorDesc const* inOut,
                                                      int32_t, int32_t) noexcept {
    (void)pos;
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmLayerNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmLayerNormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

int32_t SanaWmLayerNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        if (dims.nbDims <= 0)
            return 1;
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(dims.nbDims));
        int64_t count = 1;
        for (int32_t i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0)
                return 1;
            shape.push_back(static_cast<int64_t>(dims.d[i]));
            count *= static_cast<int64_t>(dims.d[i]);
        }
        const int64_t hidden = shape.back();

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto x = at::from_blob(const_cast<void*>(inputs[0]), shape, options);
        auto result = at::layer_norm(x, {hidden}, {}, {}, static_cast<double>(eps_), true);
        auto output = at::from_blob(outputs[0], shape, options);
        output.copy_(result);
        (void)count;
        return 0;
    } catch (const c10::Error& e) {
        report_layer_norm_error("aten layer_norm", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_layer_norm_error("aten layer_norm", e.what());
        return 1;
    }
}

SanaWmShortConvPlugin::SanaWmShortConvPlugin(int32_t frames, int32_t spatial, int32_t channels,
                                             int32_t kernel_size, const float* weight,
                                             int32_t weight_count, const float* bias,
                                             int32_t bias_count)
    : frames_(frames), spatial_(spatial), channels_(channels), kernel_size_(kernel_size) {
    if (weight_count == channels_ * kernel_size_)
        append_bf16_values(weight_, weight, weight_count);
    if (bias_count == channels_)
        append_bf16_values(bias_, bias, bias_count);
}

SanaWmShortConvPlugin::SanaWmShortConvPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 5;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    frames_ = header[0];
    spatial_ = header[1];
    channels_ = header[2];
    kernel_size_ = header[3];
    const bool has_bias = header[4] != 0;
    const int32_t weight_count = channels_ * kernel_size_;
    const int32_t bias_count = has_bias ? channels_ : 0;
    const size_t expected = kHeaderCount * sizeof(int32_t) +
                            static_cast<size_t>(weight_count + bias_count) * sizeof(uint16_t);
    if (frames_ <= 0 || spatial_ <= 0 || channels_ <= 0 || kernel_size_ <= 0 || length < expected)
        return;
    const auto* payload = reinterpret_cast<const uint16_t*>(static_cast<const char*>(data) +
                                                            kHeaderCount * sizeof(int32_t));
    weight_.assign(payload, payload + weight_count);
    if (has_bias)
        bias_.assign(payload + weight_count, payload + weight_count + bias_count);
}

char const* SanaWmShortConvPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmShortConvPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmShortConvPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmShortConvPlugin::initialize() noexcept {
    return 0;
}

void SanaWmShortConvPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmShortConvPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmShortConvPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 5;
    return kHeaderCount * sizeof(int32_t) + (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmShortConvPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 5;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = frames_;
    header[1] = spatial_;
    header[2] = channels_;
    header[3] = kernel_size_;
    header[4] = bias_.empty() ? 0 : 1;
    auto* payload =
        reinterpret_cast<uint16_t*>(static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t));
    if (!weight_.empty()) {
        std::memcpy(payload, weight_.data(), weight_.size() * sizeof(uint16_t));
        payload += weight_.size();
    }
    if (!bias_.empty())
        std::memcpy(payload, bias_.data(), bias_.size() * sizeof(uint16_t));
}

void SanaWmShortConvPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmShortConvPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmShortConvPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                            int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmShortConvPlugin* SanaWmShortConvPlugin::clone() const noexcept {
    auto* p = new SanaWmShortConvPlugin();
    p->frames_ = frames_;
    p->spatial_ = spatial_;
    p->channels_ = channels_;
    p->kernel_size_ = kernel_size_;
    p->weight_ = weight_;
    p->bias_ = bias_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs SanaWmShortConvPlugin::getOutputDimensions(int32_t,
                                                               nvinfer1::DimsExprs const* inputs,
                                                               int32_t,
                                                               nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmShortConvPlugin::supportsFormatCombination(int32_t pos,
                                                      nvinfer1::PluginTensorDesc const* inOut,
                                                      int32_t, int32_t) noexcept {
    (void)pos;
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmShortConvPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmShortConvPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

void SanaWmShortConvPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(weight_device_);
    free_device_cache(bias_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmShortConvPlugin::ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    if (!copy_to_device_cache(weight_device_, weight_.data(), weight_.size() * sizeof(uint16_t),
                              stream)) {
        return false;
    }
    if (!bias_.empty() && !copy_to_device_cache(bias_device_, bias_.data(),
                                                bias_.size() * sizeof(uint16_t), stream)) {
        return false;
    }
    return true;
}

int32_t SanaWmShortConvPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    const auto dims = inputDesc[0].dims;
    const int32_t expected_weight = channels_ * kernel_size_;
    if (dims.nbDims != 3 || dims.d[0] <= 0 || dims.d[1] != frames_ * spatial_ ||
        dims.d[2] != channels_ || frames_ <= 0 || spatial_ <= 0 || channels_ <= 0 ||
        kernel_size_ <= 0 || static_cast<int32_t>(weight_.size()) != expected_weight) {
        return 1;
    }
    int32_t device_index = 0;
    if (cudaGetDevice(&device_index) != cudaSuccess)
        return 1;
    if (!ensureDeviceCache(stream, device_index))
        return 1;
    return launch_sana_wm_short_conv(outputs[0], inputs[0], weight_device_, bias_device_, dims.d[0],
                                     frames_, spatial_, channels_, kernel_size_, stream);
}

SanaWmGateProjPlugin::SanaWmGateProjPlugin(int32_t input_dim, int32_t output_dim,
                                           int32_t activation, int32_t use_matmul_bias,
                                           const float* weight, int32_t weight_count,
                                           const float* bias, int32_t bias_count)
    : input_dim_(input_dim), output_dim_(output_dim), activation_(activation),
      use_matmul_bias_(use_matmul_bias) {
    if (weight_count == input_dim_ * output_dim_)
        append_bf16_values(weight_, weight, weight_count);
    if (bias_count == output_dim_)
        append_bf16_values(bias_, bias, bias_count);
}

SanaWmGateProjPlugin::SanaWmGateProjPlugin(const void* data, size_t length) {
    constexpr int32_t kLegacyHeaderCount = 4;
    constexpr int32_t kHeaderCount = 5;
    if (data == nullptr || length < kLegacyHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    input_dim_ = header[0];
    output_dim_ = header[1];
    activation_ = header[2];
    const bool has_bias = header[3] != 0;
    const int32_t weight_count = input_dim_ * output_dim_;
    const int32_t bias_count = has_bias ? output_dim_ : 0;
    const size_t payload_size = static_cast<size_t>(weight_count + bias_count) * sizeof(uint16_t);
    const size_t legacy_expected = kLegacyHeaderCount * sizeof(int32_t) + payload_size;
    const size_t expected = kHeaderCount * sizeof(int32_t) + payload_size;
    if (input_dim_ <= 0 || output_dim_ <= 0 || length < legacy_expected)
        return;
    const bool has_extended_header = length >= expected;
    use_matmul_bias_ = has_extended_header ? header[4] : 0;
    const int32_t header_count = has_extended_header ? kHeaderCount : kLegacyHeaderCount;
    const auto* payload = reinterpret_cast<const uint16_t*>(static_cast<const char*>(data) +
                                                            header_count * sizeof(int32_t));
    weight_.assign(payload, payload + weight_count);
    if (has_bias)
        bias_.assign(payload + weight_count, payload + weight_count + bias_count);
}

char const* SanaWmGateProjPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGateProjPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGateProjPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmGateProjPlugin::initialize() noexcept {
    return 0;
}

void SanaWmGateProjPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmGateProjPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmGateProjPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 5;
    return kHeaderCount * sizeof(int32_t) + (weight_.size() + bias_.size()) * sizeof(uint16_t);
}

void SanaWmGateProjPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 5;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = input_dim_;
    header[1] = output_dim_;
    header[2] = activation_;
    header[3] = bias_.empty() ? 0 : 1;
    header[4] = use_matmul_bias_;
    auto* payload =
        reinterpret_cast<uint16_t*>(static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t));
    if (!weight_.empty()) {
        std::memcpy(payload, weight_.data(), weight_.size() * sizeof(uint16_t));
        payload += weight_.size();
    }
    if (!bias_.empty())
        std::memcpy(payload, bias_.data(), bias_.size() * sizeof(uint16_t));
}

void SanaWmGateProjPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGateProjPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGateProjPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                           int32_t) const noexcept {
    return activation_ == 2 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16;
}

SanaWmGateProjPlugin* SanaWmGateProjPlugin::clone() const noexcept {
    auto* p = new SanaWmGateProjPlugin();
    p->input_dim_ = input_dim_;
    p->output_dim_ = output_dim_;
    p->activation_ = activation_;
    p->use_matmul_bias_ = use_matmul_bias_;
    p->weight_ = weight_;
    p->bias_ = bias_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs
SanaWmGateProjPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs out = inputs[0];
    if (out.nbDims > 0)
        out.d[out.nbDims - 1] = exprBuilder.constant(output_dim_);
    return out;
}

bool SanaWmGateProjPlugin::supportsFormatCombination(int32_t pos,
                                                     nvinfer1::PluginTensorDesc const* inOut,
                                                     int32_t nbInputs, int32_t) noexcept {
    if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        return false;
    if (activation_ == 2)
        return nbInputs == 2 && inOut[pos].type == (pos < nbInputs ? nvinfer1::DataType::kBF16
                                                                   : nvinfer1::DataType::kFLOAT);
    return nbInputs == 1 && inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmGateProjPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                           nvinfer1::DynamicPluginTensorDesc const*,
                                           int32_t) noexcept {}

size_t SanaWmGateProjPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                              nvinfer1::PluginTensorDesc const*,
                                              int32_t) const noexcept {
    return 0;
}

void SanaWmGateProjPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(weight_device_);
    free_device_cache(bias_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmGateProjPlugin::ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    if (!copy_to_device_cache(weight_device_, weight_.data(), weight_.size() * sizeof(uint16_t),
                              stream)) {
        return false;
    }
    if (!bias_.empty() && !copy_to_device_cache(bias_device_, bias_.data(),
                                                bias_.size() * sizeof(uint16_t), stream)) {
        return false;
    }
    return true;
}

int32_t SanaWmGateProjPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                      nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                      void* const* outputs, void*, cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        if (dims.nbDims <= 0 || dims.d[dims.nbDims - 1] != input_dim_ || input_dim_ <= 0 ||
            output_dim_ <= 0 || static_cast<int32_t>(weight_.size()) != input_dim_ * output_dim_) {
            return 1;
        }
        std::vector<int64_t> input_shape;
        input_shape.reserve(static_cast<size_t>(dims.nbDims));
        int64_t prefix_count = 1;
        for (int32_t i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0)
                return 1;
            input_shape.push_back(static_cast<int64_t>(dims.d[i]));
            if (i + 1 < dims.nbDims)
                prefix_count *= static_cast<int64_t>(dims.d[i]);
        }
        std::vector<int64_t> output_shape = input_shape;
        output_shape.back() = output_dim_;
        if (activation_ == 2) {
            const auto gate_dims = inputDesc[1].dims;
            if (gate_dims.nbDims != dims.nbDims)
                return 1;
            for (int32_t i = 0; i < dims.nbDims; ++i) {
                if (gate_dims.d[i] != dims.d[i])
                    return 1;
            }
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        if (!ensureDeviceCache(stream, device_index))
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto x = at::from_blob(const_cast<void*>(inputs[0]), input_shape, options);
        auto weight_trt = at::from_blob(weight_device_, {input_dim_, output_dim_}, options);
        std::optional<at::Tensor> bias;
        if (bias_device_ != nullptr)
            bias = at::from_blob(bias_device_, {output_dim_}, options);
        at::Tensor result;
        if (activation_ == 2) {
            auto gate_x = at::from_blob(const_cast<void*>(inputs[1]), input_shape, options);
            auto gate = at::linear(gate_x.reshape({prefix_count, input_dim_}),
                                   weight_trt.transpose(0, 1).contiguous(), bias)
                            .reshape(output_shape)
                            .to(at::kFloat);
            result = x.to(at::kFloat) * at::silu(gate);
        } else if (activation_ == 6) {
            const bool restore_2d = x.dim() == 2;
            auto projection_input = restore_2d ? x.unsqueeze(0) : x;
            projection_input = projection_input.transpose(-2, -1).contiguous().transpose(-2, -1);
            result = at::linear(projection_input, weight_trt.transpose(0, 1).contiguous(), bias);
            if (restore_2d)
                result = result.squeeze(0);
        } else if (use_matmul_bias_ != 0) {
            auto projection_input = activation_ == 5 ? at::silu(x) : x;
            result = at::matmul(projection_input, weight_trt);
            if (bias.has_value())
                result = result + *bias;
        } else {
            auto projection_input = activation_ == 5 ? at::silu(x) : x;
            result = at::linear(projection_input.reshape({prefix_count, input_dim_}),
                                weight_trt.transpose(0, 1).contiguous(), bias)
                         .reshape(output_shape);
        }
        if (activation_ == 1)
            result = at::sigmoid(result);
        else if (activation_ == 3)
            result = at::gelu(result, "tanh");
        else if (activation_ == 4)
            result = at::silu(result);
        result = result.reshape(output_shape);
        const auto output_options = activation_ == 2 ? options.dtype(at::kFloat) : options;
        auto output = at::from_blob(outputs[0], output_shape, output_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_gate_proj_error("aten linear", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_gate_proj_error("aten linear", e.what());
        return 1;
    }
}

SanaWmDecayPlugin::SanaWmDecayPlugin(int32_t heads, const float* a_log_values, int32_t a_count)
    : heads_(heads) {
    if (a_count == heads_)
        append_float_values(a_log_values_, a_log_values, a_count);
}

SanaWmDecayPlugin::SanaWmDecayPlugin(const void* data, size_t length) {
    constexpr int32_t kHeaderCount = 1;
    if (data == nullptr || length < kHeaderCount * sizeof(int32_t))
        return;
    const auto* header = static_cast<const int32_t*>(data);
    heads_ = header[0];
    const size_t expected =
        kHeaderCount * sizeof(int32_t) + static_cast<size_t>(std::max(heads_, 0)) * sizeof(float);
    if (heads_ <= 0 || length < expected)
        return;
    const auto* payload = reinterpret_cast<const float*>(static_cast<const char*>(data) +
                                                         kHeaderCount * sizeof(int32_t));
    a_log_values_.assign(payload, payload + heads_);
}

char const* SanaWmDecayPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmDecayPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmDecayPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmDecayPlugin::initialize() noexcept {
    return 0;
}

void SanaWmDecayPlugin::terminate() noexcept {
    releaseDeviceCache();
}

void SanaWmDecayPlugin::destroy() noexcept {
    releaseDeviceCache();
    delete this;
}

size_t SanaWmDecayPlugin::getSerializationSize() const noexcept {
    constexpr int32_t kHeaderCount = 1;
    return kHeaderCount * sizeof(int32_t) + a_log_values_.size() * sizeof(float);
}

void SanaWmDecayPlugin::serialize(void* buffer) const noexcept {
    constexpr int32_t kHeaderCount = 1;
    auto* header = static_cast<int32_t*>(buffer);
    header[0] = heads_;
    auto* payload = static_cast<char*>(buffer) + kHeaderCount * sizeof(int32_t);
    if (!a_log_values_.empty())
        std::memcpy(payload, a_log_values_.data(), a_log_values_.size() * sizeof(float));
}

void SanaWmDecayPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmDecayPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmDecayPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                        int32_t) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

SanaWmDecayPlugin* SanaWmDecayPlugin::clone() const noexcept {
    auto* p = new SanaWmDecayPlugin();
    p->heads_ = heads_;
    p->a_log_values_ = a_log_values_;
    p->namespace_ = namespace_;
    return p;
}

nvinfer1::DimsExprs SanaWmDecayPlugin::getOutputDimensions(int32_t,
                                                           nvinfer1::DimsExprs const* inputs,
                                                           int32_t,
                                                           nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmDecayPlugin::supportsFormatCombination(int32_t pos,
                                                  nvinfer1::PluginTensorDesc const* inOut, int32_t,
                                                  int32_t) noexcept {
    (void)pos;
    return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

void SanaWmDecayPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                        nvinfer1::DynamicPluginTensorDesc const*,
                                        int32_t) noexcept {}

size_t SanaWmDecayPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                           nvinfer1::PluginTensorDesc const*,
                                           int32_t) const noexcept {
    return 0;
}

void SanaWmDecayPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const cudaError_t have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(a_log_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}

bool SanaWmDecayPlugin::ensureDeviceCache(cudaStream_t stream, int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(a_log_device_, a_log_values_.data(),
                                a_log_values_.size() * sizeof(float), stream);
}

int32_t SanaWmDecayPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                   nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                   void* const* outputs, void*, cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        if (dims.nbDims != 3 || dims.d[2] != heads_ || heads_ <= 0 ||
            static_cast<int32_t>(a_log_values_.size()) != heads_) {
            return 1;
        }
        const int32_t batch = dims.d[0];
        const int32_t frames = dims.d[1];
        if (batch <= 0 || frames <= 0)
            return 1;
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        if (!ensureDeviceCache(stream, device_index))
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        auto gate_dt =
            at::from_blob(const_cast<void*>(inputs[0]), {batch, frames, heads_}, options);
        auto a_log = at::from_blob(a_log_device_, {1, 1, heads_}, options);
        auto a = at::exp(a_log);
        auto result = at::exp(-(a * at::softplus(gate_dt, 1.0, 20.0)));
        auto output = at::from_blob(outputs[0], {batch, frames, heads_}, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_decay_error("aten decay", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_decay_error("aten decay", e.what());
        return 1;
    }
}

SanaWmGemmaRmsNormPlugin::SanaWmGemmaRmsNormPlugin(float eps) : eps_(eps) {}

SanaWmGemmaRmsNormPlugin::SanaWmGemmaRmsNormPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(eps_))
        std::memcpy(&eps_, data, sizeof(eps_));
}

char const* SanaWmGemmaRmsNormPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGemmaRmsNormPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGemmaRmsNormPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmGemmaRmsNormPlugin::initialize() noexcept {
    return 0;
}

void SanaWmGemmaRmsNormPlugin::terminate() noexcept {}

void SanaWmGemmaRmsNormPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmGemmaRmsNormPlugin::getSerializationSize() const noexcept {
    return sizeof(eps_);
}

void SanaWmGemmaRmsNormPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &eps_, sizeof(eps_));
}

void SanaWmGemmaRmsNormPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGemmaRmsNormPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGemmaRmsNormPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                               int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmGemmaRmsNormPlugin* SanaWmGemmaRmsNormPlugin::clone() const noexcept {
    auto* plugin = new SanaWmGemmaRmsNormPlugin(eps_);
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs
SanaWmGemmaRmsNormPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                              nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmGemmaRmsNormPlugin::supportsFormatCombination(int32_t pos,
                                                         nvinfer1::PluginTensorDesc const* inOut,
                                                         int32_t nbInputs,
                                                         int32_t nbOutputs) noexcept {
    if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3 ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type == (pos == 1 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}

void SanaWmGemmaRmsNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                               nvinfer1::DynamicPluginTensorDesc const*,
                                               int32_t) noexcept {}

size_t SanaWmGemmaRmsNormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                  nvinfer1::PluginTensorDesc const*,
                                                  int32_t) const noexcept {
    return 0;
}

int32_t SanaWmGemmaRmsNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                          nvinfer1::PluginTensorDesc const*,
                                          void const* const* inputs, void* const* outputs, void*,
                                          cudaStream_t stream) noexcept {
    try {
        const auto input_dims = inputDesc[0].dims;
        const auto weight_dims = inputDesc[1].dims;
        if (input_dims.nbDims <= 0 || weight_dims.nbDims != 1)
            return 1;
        const int32_t hidden = input_dims.d[input_dims.nbDims - 1];
        if (hidden <= 0 || weight_dims.d[0] != hidden)
            return 1;
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(input_dims.nbDims));
        for (int32_t i = 0; i < input_dims.nbDims; ++i) {
            if (input_dims.d[i] <= 0)
                return 1;
            shape.push_back(input_dims.d[i]);
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        auto input = at::from_blob(const_cast<void*>(inputs[0]), shape, bf16_options);
        auto weight = at::from_blob(const_cast<void*>(inputs[1]), {hidden}, float_options);
        auto input_float = input.to(at::kFloat);
        auto normalized =
            input_float * at::rsqrt(input_float.pow(2).mean(-1, true) + static_cast<double>(eps_));
        auto result = (normalized * weight).to(at::kBFloat16);
        auto output = at::from_blob(outputs[0], shape, bf16_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_gemma_rms_norm_error("aten rms norm", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_gemma_rms_norm_error("aten rms norm", e.what());
        return 1;
    }
}

SanaWmGemmaGatedGeluPlugin::SanaWmGemmaGatedGeluPlugin(const void*, size_t) {}

char const* SanaWmGemmaGatedGeluPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGemmaGatedGeluPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGemmaGatedGeluPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmGemmaGatedGeluPlugin::initialize() noexcept {
    return 0;
}

void SanaWmGemmaGatedGeluPlugin::terminate() noexcept {}

void SanaWmGemmaGatedGeluPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmGemmaGatedGeluPlugin::getSerializationSize() const noexcept {
    return 0;
}

void SanaWmGemmaGatedGeluPlugin::serialize(void*) const noexcept {}

void SanaWmGemmaGatedGeluPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGemmaGatedGeluPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGemmaGatedGeluPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                 int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmGemmaGatedGeluPlugin* SanaWmGemmaGatedGeluPlugin::clone() const noexcept {
    auto* plugin = new SanaWmGemmaGatedGeluPlugin();
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs
SanaWmGemmaGatedGeluPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmGemmaGatedGeluPlugin::supportsFormatCombination(int32_t pos,
                                                           nvinfer1::PluginTensorDesc const* inOut,
                                                           int32_t nbInputs,
                                                           int32_t nbOutputs) noexcept {
    return nbInputs == 2 && nbOutputs == 1 && pos >= 0 && pos < 3 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmGemmaGatedGeluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) noexcept {}

size_t SanaWmGemmaGatedGeluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                    nvinfer1::PluginTensorDesc const*,
                                                    int32_t) const noexcept {
    return 0;
}

int32_t SanaWmGemmaGatedGeluPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                            nvinfer1::PluginTensorDesc const*,
                                            void const* const* inputs, void* const* outputs, void*,
                                            cudaStream_t stream) noexcept {
    try {
        const auto gate_dims = inputDesc[0].dims;
        const auto up_dims = inputDesc[1].dims;
        if (gate_dims.nbDims <= 0 || gate_dims.nbDims != up_dims.nbDims)
            return 1;
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(gate_dims.nbDims));
        for (int32_t i = 0; i < gate_dims.nbDims; ++i) {
            if (gate_dims.d[i] <= 0 || gate_dims.d[i] != up_dims.d[i])
                return 1;
            shape.push_back(gate_dims.d[i]);
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto gate = at::from_blob(const_cast<void*>(inputs[0]), shape, options);
        auto up = at::from_blob(const_cast<void*>(inputs[1]), shape, options);
        auto result = at::gelu(gate, "tanh") * up;
        auto output = at::from_blob(outputs[0], shape, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_gemma_gated_gelu_error("aten gated gelu", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_gemma_gated_gelu_error("aten gated gelu", e.what());
        return 1;
    }
}

SanaWmGemmaRopePlugin::SanaWmGemmaRopePlugin(int32_t heads, int32_t head_dim, int32_t rotary_dim,
                                             bool interleaved)
    : heads_(heads), head_dim_(head_dim), rotary_dim_(rotary_dim), interleaved_(interleaved) {}

SanaWmGemmaRopePlugin::SanaWmGemmaRopePlugin(const void* data, size_t length) {
    if (data == nullptr || length < getSerializationSize())
        return;
    const char* cursor = static_cast<const char*>(data);
    std::memcpy(&heads_, cursor, sizeof(heads_));
    cursor += sizeof(heads_);
    std::memcpy(&head_dim_, cursor, sizeof(head_dim_));
    cursor += sizeof(head_dim_);
    std::memcpy(&rotary_dim_, cursor, sizeof(rotary_dim_));
    cursor += sizeof(rotary_dim_);
    int32_t interleaved = 0;
    std::memcpy(&interleaved, cursor, sizeof(interleaved));
    interleaved_ = interleaved != 0;
}

char const* SanaWmGemmaRopePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGemmaRopePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGemmaRopePlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmGemmaRopePlugin::initialize() noexcept {
    return 0;
}

void SanaWmGemmaRopePlugin::terminate() noexcept {}

void SanaWmGemmaRopePlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmGemmaRopePlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int32_t);
}

void SanaWmGemmaRopePlugin::serialize(void* buffer) const noexcept {
    char* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &heads_, sizeof(heads_));
    cursor += sizeof(heads_);
    std::memcpy(cursor, &head_dim_, sizeof(head_dim_));
    cursor += sizeof(head_dim_);
    std::memcpy(cursor, &rotary_dim_, sizeof(rotary_dim_));
    cursor += sizeof(rotary_dim_);
    const int32_t interleaved = interleaved_ ? 1 : 0;
    std::memcpy(cursor, &interleaved, sizeof(interleaved));
}

void SanaWmGemmaRopePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGemmaRopePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGemmaRopePlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                            int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmGemmaRopePlugin* SanaWmGemmaRopePlugin::clone() const noexcept {
    auto* plugin = new SanaWmGemmaRopePlugin(heads_, head_dim_, rotary_dim_, interleaved_);
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs SanaWmGemmaRopePlugin::getOutputDimensions(int32_t,
                                                               nvinfer1::DimsExprs const* inputs,
                                                               int32_t,
                                                               nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmGemmaRopePlugin::supportsFormatCombination(int32_t pos,
                                                      nvinfer1::PluginTensorDesc const* inOut,
                                                      int32_t nbInputs,
                                                      int32_t nbOutputs) noexcept {
    return nbInputs == 3 && nbOutputs == 1 && pos >= 0 && pos < 4 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmGemmaRopePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                            nvinfer1::DynamicPluginTensorDesc const*,
                                            int32_t) noexcept {}

size_t SanaWmGemmaRopePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                               nvinfer1::PluginTensorDesc const*,
                                               int32_t) const noexcept {
    return 0;
}

int32_t SanaWmGemmaRopePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                       nvinfer1::PluginTensorDesc const*, void const* const* inputs,
                                       void* const* outputs, void*, cudaStream_t stream) noexcept {
    try {
        const auto q_dims = inputDesc[0].dims;
        const auto cos_dims = inputDesc[1].dims;
        const auto sin_dims = inputDesc[2].dims;
        if (q_dims.nbDims != 2 || cos_dims.nbDims != 2 || sin_dims.nbDims != 2 || heads_ <= 0 ||
            head_dim_ <= 0 || rotary_dim_ <= 0 || rotary_dim_ > head_dim_ || rotary_dim_ % 2 != 0) {
            return 1;
        }
        const int32_t rows = q_dims.d[0];
        if (rows <= 0 || q_dims.d[1] != heads_ * head_dim_ || cos_dims.d[0] != rows ||
            sin_dims.d[0] != rows || cos_dims.d[1] != rotary_dim_ || sin_dims.d[1] != rotary_dim_) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto q = at::from_blob(const_cast<void*>(inputs[0]), {rows, heads_, head_dim_}, options);
        auto cos =
            at::from_blob(const_cast<void*>(inputs[1]), {rows, rotary_dim_}, options).unsqueeze(1);
        auto sin =
            at::from_blob(const_cast<void*>(inputs[2]), {rows, rotary_dim_}, options).unsqueeze(1);
        auto q_rot = q.slice(-1, 0, rotary_dim_);

        at::Tensor rotate_half;
        if (interleaved_) {
            auto even = q_rot.slice(-1, 0, rotary_dim_, 2);
            auto odd = q_rot.slice(-1, 1, rotary_dim_, 2);
            rotate_half = at::stack({-odd, even}, -1).flatten(-2, -1);
        } else {
            const int32_t half = rotary_dim_ / 2;
            rotate_half =
                at::cat({-q_rot.slice(-1, half, rotary_dim_), q_rot.slice(-1, 0, half)}, -1);
        }
        auto result = q_rot * cos + rotate_half * sin;
        if (rotary_dim_ != head_dim_)
            result = at::cat({result, q.slice(-1, rotary_dim_, head_dim_)}, -1);
        auto output = at::from_blob(outputs[0], {rows, heads_, head_dim_}, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_gemma_rope_error("aten rope", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_gemma_rope_error("aten rope", e.what());
        return 1;
    }
}

SanaWmGemmaAttentionPlugin::SanaWmGemmaAttentionPlugin(int32_t heads, int32_t kv_heads,
                                                       int32_t head_dim, float scale)
    : heads_(heads), kv_heads_(kv_heads), head_dim_(head_dim), scale_(scale) {}

SanaWmGemmaAttentionPlugin::SanaWmGemmaAttentionPlugin(const void* data, size_t length) {
    if (data == nullptr || length < getSerializationSize())
        return;
    const char* cursor = static_cast<const char*>(data);
    std::memcpy(&heads_, cursor, sizeof(heads_));
    cursor += sizeof(heads_);
    std::memcpy(&kv_heads_, cursor, sizeof(kv_heads_));
    cursor += sizeof(kv_heads_);
    std::memcpy(&head_dim_, cursor, sizeof(head_dim_));
    cursor += sizeof(head_dim_);
    std::memcpy(&scale_, cursor, sizeof(scale_));
}

char const* SanaWmGemmaAttentionPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* SanaWmGemmaAttentionPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t SanaWmGemmaAttentionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SanaWmGemmaAttentionPlugin::initialize() noexcept {
    return 0;
}

void SanaWmGemmaAttentionPlugin::terminate() noexcept {}

void SanaWmGemmaAttentionPlugin::destroy() noexcept {
    delete this;
}

size_t SanaWmGemmaAttentionPlugin::getSerializationSize() const noexcept {
    return 3 * sizeof(int32_t) + sizeof(float);
}

void SanaWmGemmaAttentionPlugin::serialize(void* buffer) const noexcept {
    char* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &heads_, sizeof(heads_));
    cursor += sizeof(heads_);
    std::memcpy(cursor, &kv_heads_, sizeof(kv_heads_));
    cursor += sizeof(kv_heads_);
    std::memcpy(cursor, &head_dim_, sizeof(head_dim_));
    cursor += sizeof(head_dim_);
    std::memcpy(cursor, &scale_, sizeof(scale_));
}

void SanaWmGemmaAttentionPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}

char const* SanaWmGemmaAttentionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType SanaWmGemmaAttentionPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                 int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}

SanaWmGemmaAttentionPlugin* SanaWmGemmaAttentionPlugin::clone() const noexcept {
    auto* plugin = new SanaWmGemmaAttentionPlugin(heads_, kv_heads_, head_dim_, scale_);
    plugin->namespace_ = namespace_;
    return plugin;
}

nvinfer1::DimsExprs
SanaWmGemmaAttentionPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}

bool SanaWmGemmaAttentionPlugin::supportsFormatCombination(int32_t pos,
                                                           nvinfer1::PluginTensorDesc const* inOut,
                                                           int32_t nbInputs,
                                                           int32_t nbOutputs) noexcept {
    return nbInputs == 4 && nbOutputs == 1 && pos >= 0 && pos < 5 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}

void SanaWmGemmaAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) noexcept {}

size_t SanaWmGemmaAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                    nvinfer1::PluginTensorDesc const*,
                                                    int32_t) const noexcept {
    return 0;
}

int32_t SanaWmGemmaAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                            nvinfer1::PluginTensorDesc const*,
                                            void const* const* inputs, void* const* outputs, void*,
                                            cudaStream_t stream) noexcept {
    try {
        const auto q_dims = inputDesc[0].dims;
        const auto k_dims = inputDesc[1].dims;
        const auto v_dims = inputDesc[2].dims;
        const auto mask_dims = inputDesc[3].dims;
        if (q_dims.nbDims != 2 || k_dims.nbDims != 2 || v_dims.nbDims != 2 ||
            (mask_dims.nbDims != 2 && mask_dims.nbDims != 4) || heads_ <= 0 || kv_heads_ <= 0 ||
            head_dim_ <= 0 || heads_ % kv_heads_ != 0) {
            return 1;
        }
        const int32_t query_rows = q_dims.d[0];
        const int32_t key_rows = k_dims.d[0];
        if (query_rows <= 0 || key_rows <= 0 || q_dims.d[1] != heads_ * head_dim_ ||
            k_dims.d[1] != kv_heads_ * head_dim_ || v_dims.d[0] != key_rows ||
            v_dims.d[1] != kv_heads_ * head_dim_) {
            return 1;
        }
        if ((mask_dims.nbDims == 2 &&
             (mask_dims.d[0] != query_rows || mask_dims.d[1] != key_rows)) ||
            (mask_dims.nbDims == 4 &&
             (mask_dims.d[0] != 1 || mask_dims.d[1] != 1 || mask_dims.d[2] != query_rows ||
              mask_dims.d[3] != key_rows))) {
            return 1;
        }

        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);

        auto q =
            at::from_blob(const_cast<void*>(inputs[0]), {query_rows, heads_, head_dim_}, options)
                .permute({1, 0, 2})
                .unsqueeze(0);
        auto k =
            at::from_blob(const_cast<void*>(inputs[1]), {key_rows, kv_heads_, head_dim_}, options)
                .permute({1, 0, 2})
                .unsqueeze(0);
        auto v =
            at::from_blob(const_cast<void*>(inputs[2]), {key_rows, kv_heads_, head_dim_}, options)
                .permute({1, 0, 2})
                .unsqueeze(0);
        auto mask =
            at::from_blob(const_cast<void*>(inputs[3]), {1, 1, query_rows, key_rows}, options);
        int32_t sdpa_key_rows = key_rows;
        if (query_rows > 1 && key_rows > query_rows) {
            const int32_t cache_rows = key_rows - query_rows;
            k = k.slice(2, cache_rows, key_rows);
            v = v.slice(2, cache_rows, key_rows);
            mask = mask.slice(3, cache_rows, key_rows);
            sdpa_key_rows = query_rows;
        }
        if (kv_heads_ != heads_) {
            const int32_t groups = heads_ / kv_heads_;
            k = k.unsqueeze(2)
                    .expand({1, kv_heads_, groups, sdpa_key_rows, head_dim_})
                    .reshape({1, heads_, sdpa_key_rows, head_dim_});
            v = v.unsqueeze(2)
                    .expand({1, kv_heads_, groups, sdpa_key_rows, head_dim_})
                    .reshape({1, heads_, sdpa_key_rows, head_dim_});
        }

        // Transformers passes Gemma's visibility mask to SDPA as bool. In
        // particular, an all-false padded query must produce an all-zero
        // context; a finite additive sentinel would instead softmax over the
        // masked row and change every later padded hidden state.
        auto visibility_mask = mask.eq(0);
        auto context = at::scaled_dot_product_attention(
            q, k, v, std::optional<at::Tensor>(visibility_mask), 0.0, false,
            std::optional<double>(static_cast<double>(scale_)), false);
        auto result = context.transpose(1, 2).reshape({query_rows, heads_ * head_dim_});
        auto output = at::from_blob(outputs[0], {query_rows, heads_ * head_dim_}, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& e) {
        report_gemma_attention_error("aten scaled_dot_product_attention", e.what());
        return 1;
    } catch (const std::exception& e) {
        report_gemma_attention_error("aten scaled_dot_product_attention", e.what());
        return 1;
    }
}

SanaWmLtxTextNormalizePlugin::SanaWmLtxTextNormalizePlugin(int32_t caption_channels,
                                                           int32_t layer_count, float scale_factor,
                                                           float eps)
    : caption_channels_(caption_channels), layer_count_(layer_count), scale_factor_(scale_factor),
      eps_(eps) {}

SanaWmLtxTextNormalizePlugin::SanaWmLtxTextNormalizePlugin(const void* data, size_t length) {
    constexpr size_t kSize = 2 * sizeof(int32_t) + 2 * sizeof(float);
    if (data == nullptr || length < kSize)
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::memcpy(&caption_channels_, cursor, sizeof(caption_channels_));
    cursor += sizeof(caption_channels_);
    std::memcpy(&layer_count_, cursor, sizeof(layer_count_));
    cursor += sizeof(layer_count_);
    std::memcpy(&scale_factor_, cursor, sizeof(scale_factor_));
    cursor += sizeof(scale_factor_);
    std::memcpy(&eps_, cursor, sizeof(eps_));
}

char const* SanaWmLtxTextNormalizePlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxTextNormalizePlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxTextNormalizePlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmLtxTextNormalizePlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxTextNormalizePlugin::terminate() noexcept {}
void SanaWmLtxTextNormalizePlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxTextNormalizePlugin::getSerializationSize() const noexcept {
    return 2 * sizeof(int32_t) + 2 * sizeof(float);
}
void SanaWmLtxTextNormalizePlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &caption_channels_, sizeof(caption_channels_));
    cursor += sizeof(caption_channels_);
    std::memcpy(cursor, &layer_count_, sizeof(layer_count_));
    cursor += sizeof(layer_count_);
    std::memcpy(cursor, &scale_factor_, sizeof(scale_factor_));
    cursor += sizeof(scale_factor_);
    std::memcpy(cursor, &eps_, sizeof(eps_));
}
void SanaWmLtxTextNormalizePlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxTextNormalizePlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxTextNormalizePlugin::getOutputDataType(int32_t,
                                                                   nvinfer1::DataType const*,
                                                                   int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxTextNormalizePlugin* SanaWmLtxTextNormalizePlugin::clone() const noexcept {
    auto* plugin =
        new SanaWmLtxTextNormalizePlugin(caption_channels_, layer_count_, scale_factor_, eps_);
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs
SanaWmLtxTextNormalizePlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs,
                                                  int32_t, nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmLtxTextNormalizePlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3 ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type == (pos < 2 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}
void SanaWmLtxTextNormalizePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*,
                                                   int32_t,
                                                   nvinfer1::DynamicPluginTensorDesc const*,
                                                   int32_t) noexcept {}
size_t SanaWmLtxTextNormalizePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                      nvinfer1::PluginTensorDesc const*,
                                                      int32_t) const noexcept {
    return 0;
}
int32_t SanaWmLtxTextNormalizePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                              nvinfer1::PluginTensorDesc const*,
                                              void const* const* inputs, void* const* outputs,
                                              void*, cudaStream_t stream) noexcept {
    try {
        const auto hidden_dims = inputDesc[0].dims;
        const auto mask_dims = inputDesc[1].dims;
        if (hidden_dims.nbDims != 3 || mask_dims.nbDims != 2 || caption_channels_ <= 0 ||
            layer_count_ <= 0 || hidden_dims.d[0] != mask_dims.d[0] ||
            hidden_dims.d[1] != mask_dims.d[1] ||
            hidden_dims.d[2] != caption_channels_ * layer_count_) {
            return 1;
        }
        const int64_t batch = hidden_dims.d[0];
        const int64_t seq_len = hidden_dims.d[1];
        const int64_t packed_dim = hidden_dims.d[2];
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto float_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        const auto bf16_options = float_options.dtype(at::kBFloat16);
        const auto long_options = float_options.dtype(at::kLong);
        auto hidden_float = at::from_blob(const_cast<void*>(inputs[0]),
                                          {batch, seq_len, packed_dim}, float_options);
        auto hidden =
            hidden_float.to(at::kBFloat16).view({batch, seq_len, caption_channels_, layer_count_});
        auto input_mask =
            at::from_blob(const_cast<void*>(inputs[1]), {batch, seq_len}, float_options);
        auto sequence_lengths = input_mask.to(at::kLong).sum(-1);
        auto token_indices = at::arange(seq_len, long_options).unsqueeze(0);
        auto valid = token_indices >= (seq_len - sequence_lengths.unsqueeze(1));
        auto valid4 = valid.unsqueeze(-1).unsqueeze(-1);

        auto masked = hidden.masked_fill(valid4.logical_not(), 0.0);
        const std::array<int64_t, 2> reduce_dims{1, 2};
        auto masked_sum = at::sum(masked, at::OptionalIntArrayRef(reduce_dims), true);
        auto denominator = (sequence_lengths * caption_channels_).view({batch, 1, 1, 1}) + eps_;
        auto masked_mean = masked_sum / denominator;
        auto x_min = at::amin(
            hidden.masked_fill(valid4.logical_not(), std::numeric_limits<float>::infinity()),
            reduce_dims, true);
        auto x_max = at::amax(
            hidden.masked_fill(valid4.logical_not(), -std::numeric_limits<float>::infinity()),
            reduce_dims, true);
        auto normalized = (hidden - masked_mean) / (x_max - x_min + eps_);
        normalized = normalized * scale_factor_;
        normalized = normalized.flatten(2);
        auto flat_mask = valid4.squeeze(-1).expand({batch, seq_len, packed_dim});
        normalized = normalized.masked_fill(flat_mask.logical_not(), 0.0).to(at::kBFloat16);

        auto output = at::from_blob(outputs[0], {batch, seq_len, packed_dim}, bf16_options);
        output.copy_(normalized);
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_text_normalize] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_text_normalize] %s\n", error.what());
        return 1;
    }
}

SanaWmLtxRegisterPlugin::SanaWmLtxRegisterPlugin(int32_t register_count, int32_t hidden_dim,
                                                 const float* registers, int32_t value_count)
    : register_count_(register_count), hidden_dim_(hidden_dim) {
    append_bf16_values(registers_, registers, value_count);
}
SanaWmLtxRegisterPlugin::SanaWmLtxRegisterPlugin(const void* data, size_t length) {
    constexpr size_t kHeader = 2 * sizeof(int32_t);
    if (data == nullptr || length < kHeader)
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::memcpy(&register_count_, cursor, sizeof(register_count_));
    cursor += sizeof(register_count_);
    std::memcpy(&hidden_dim_, cursor, sizeof(hidden_dim_));
    cursor += sizeof(hidden_dim_);
    const auto count = static_cast<size_t>(std::max(register_count_, 0)) *
                       static_cast<size_t>(std::max(hidden_dim_, 0));
    if (length >= kHeader + count * sizeof(uint16_t)) {
        registers_.resize(count);
        std::memcpy(registers_.data(), cursor, count * sizeof(uint16_t));
    }
}
SanaWmLtxRegisterPlugin::~SanaWmLtxRegisterPlugin() {
    releaseDeviceCache();
}
char const* SanaWmLtxRegisterPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxRegisterPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxRegisterPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmLtxRegisterPlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxRegisterPlugin::terminate() noexcept {
    releaseDeviceCache();
}
void SanaWmLtxRegisterPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxRegisterPlugin::getSerializationSize() const noexcept {
    return 2 * sizeof(int32_t) + registers_.size() * sizeof(uint16_t);
}
void SanaWmLtxRegisterPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &register_count_, sizeof(register_count_));
    cursor += sizeof(register_count_);
    std::memcpy(cursor, &hidden_dim_, sizeof(hidden_dim_));
    cursor += sizeof(hidden_dim_);
    if (!registers_.empty())
        std::memcpy(cursor, registers_.data(), registers_.size() * sizeof(uint16_t));
}
void SanaWmLtxRegisterPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxRegisterPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxRegisterPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                              int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxRegisterPlugin* SanaWmLtxRegisterPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLtxRegisterPlugin();
    plugin->register_count_ = register_count_;
    plugin->hidden_dim_ = hidden_dim_;
    plugin->registers_ = registers_;
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs SanaWmLtxRegisterPlugin::getOutputDimensions(int32_t,
                                                                 nvinfer1::DimsExprs const* inputs,
                                                                 int32_t,
                                                                 nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmLtxRegisterPlugin::supportsFormatCombination(int32_t pos,
                                                        nvinfer1::PluginTensorDesc const* inOut,
                                                        int32_t nbInputs,
                                                        int32_t nbOutputs) noexcept {
    if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3 ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type == (pos == 1 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}
void SanaWmLtxRegisterPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                              nvinfer1::DynamicPluginTensorDesc const*,
                                              int32_t) noexcept {}
size_t SanaWmLtxRegisterPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                 nvinfer1::PluginTensorDesc const*,
                                                 int32_t) const noexcept {
    return 0;
}
void SanaWmLtxRegisterPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const auto have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(registers_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}
bool SanaWmLtxRegisterPlugin::ensureDeviceCache(cudaStream_t stream,
                                                int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(registers_device_, registers_.data(),
                                registers_.size() * sizeof(uint16_t), stream);
}
int32_t SanaWmLtxRegisterPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                         nvinfer1::PluginTensorDesc const*,
                                         void const* const* inputs, void* const* outputs, void*,
                                         cudaStream_t stream) noexcept {
    try {
        const auto hidden_dims = inputDesc[0].dims;
        const auto mask_dims = inputDesc[1].dims;
        if (hidden_dims.nbDims != 3 || mask_dims.nbDims != 2 || hidden_dims.d[0] != 1 ||
            mask_dims.d[0] != 1 || hidden_dims.d[1] != mask_dims.d[1] ||
            hidden_dims.d[2] != hidden_dim_ || register_count_ <= 0 ||
            hidden_dims.d[1] % register_count_ != 0 ||
            registers_.size() != static_cast<size_t>(register_count_) * hidden_dim_) {
            return 1;
        }
        const int64_t seq_len = hidden_dims.d[1];
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess ||
            !ensureDeviceCache(stream, device_index)) {
            return 1;
        }
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options = bf16_options.dtype(at::kFloat);
        auto hidden =
            at::from_blob(const_cast<void*>(inputs[0]), {1, seq_len, hidden_dim_}, bf16_options);
        auto mask = at::from_blob(const_cast<void*>(inputs[1]), {1, seq_len}, float_options);
        auto binary = mask.ge(0.5).to(at::kInt);
        const int64_t valid_len = binary.sum().item<int64_t>();
        if (valid_len < 0 || valid_len > seq_len)
            return 1;
        auto valid = hidden.narrow(1, seq_len - valid_len, valid_len);
        auto padded = at::constant_pad_nd(valid, {0, 0, 0, seq_len - valid_len}, 0.0);
        auto registers =
            at::from_blob(registers_device_, {register_count_, hidden_dim_}, bf16_options)
                .repeat({seq_len / register_count_, 1})
                .unsqueeze(0);
        auto flipped = at::flip(binary, {1}).unsqueeze(-1);
        auto result = flipped * padded + (1 - flipped) * registers;
        auto output = at::from_blob(outputs[0], {1, seq_len, hidden_dim_}, bf16_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_register] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_register] %s\n", error.what());
        return 1;
    }
}

SanaWmLtxConnectorBlockPlugin::SanaWmLtxConnectorBlockPlugin(int32_t hidden_dim, int32_t num_heads,
                                                             int32_t head_dim, int32_t ff_dim,
                                                             const float* packed_weights,
                                                             int32_t weight_count)
    : hidden_dim_(hidden_dim), num_heads_(num_heads), head_dim_(head_dim), ff_dim_(ff_dim) {
    append_bf16_values(packed_weights_, packed_weights, weight_count);
}
SanaWmLtxConnectorBlockPlugin::SanaWmLtxConnectorBlockPlugin(const void* data, size_t length) {
    constexpr size_t kHeader = 4 * sizeof(int32_t);
    if (data == nullptr || length < kHeader)
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::memcpy(&hidden_dim_, cursor, sizeof(hidden_dim_));
    cursor += sizeof(hidden_dim_);
    std::memcpy(&num_heads_, cursor, sizeof(num_heads_));
    cursor += sizeof(num_heads_);
    std::memcpy(&head_dim_, cursor, sizeof(head_dim_));
    cursor += sizeof(head_dim_);
    std::memcpy(&ff_dim_, cursor, sizeof(ff_dim_));
    cursor += sizeof(ff_dim_);
    const auto count = expectedWeightCount();
    if (length >= kHeader + count * sizeof(uint16_t)) {
        packed_weights_.resize(count);
        std::memcpy(packed_weights_.data(), cursor, count * sizeof(uint16_t));
    }
}
SanaWmLtxConnectorBlockPlugin::~SanaWmLtxConnectorBlockPlugin() {
    releaseDeviceCache();
}
std::size_t SanaWmLtxConnectorBlockPlugin::expectedWeightCount() const noexcept {
    if (hidden_dim_ <= 0 || ff_dim_ <= 0)
        return 0;
    const auto hidden = static_cast<size_t>(hidden_dim_);
    const auto ff = static_cast<size_t>(ff_dim_);
    return 2 * hidden + 4 * (hidden * hidden + hidden) + (ff * hidden + ff) +
           (hidden * ff + hidden);
}
char const* SanaWmLtxConnectorBlockPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxConnectorBlockPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxConnectorBlockPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmLtxConnectorBlockPlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxConnectorBlockPlugin::terminate() noexcept {
    releaseDeviceCache();
}
void SanaWmLtxConnectorBlockPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxConnectorBlockPlugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int32_t) + packed_weights_.size() * sizeof(uint16_t);
}
void SanaWmLtxConnectorBlockPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &hidden_dim_, sizeof(hidden_dim_));
    cursor += sizeof(hidden_dim_);
    std::memcpy(cursor, &num_heads_, sizeof(num_heads_));
    cursor += sizeof(num_heads_);
    std::memcpy(cursor, &head_dim_, sizeof(head_dim_));
    cursor += sizeof(head_dim_);
    std::memcpy(cursor, &ff_dim_, sizeof(ff_dim_));
    cursor += sizeof(ff_dim_);
    if (!packed_weights_.empty())
        std::memcpy(cursor, packed_weights_.data(), packed_weights_.size() * sizeof(uint16_t));
}
void SanaWmLtxConnectorBlockPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxConnectorBlockPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxConnectorBlockPlugin::getOutputDataType(int32_t,
                                                                    nvinfer1::DataType const*,
                                                                    int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxConnectorBlockPlugin* SanaWmLtxConnectorBlockPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLtxConnectorBlockPlugin();
    plugin->hidden_dim_ = hidden_dim_;
    plugin->num_heads_ = num_heads_;
    plugin->head_dim_ = head_dim_;
    plugin->ff_dim_ = ff_dim_;
    plugin->packed_weights_ = packed_weights_;
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs
SanaWmLtxConnectorBlockPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs,
                                                   int32_t, nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmLtxConnectorBlockPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (nbInputs != 3 || nbOutputs != 1 || pos < 0 || pos >= 4 ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type ==
           ((pos == 1 || pos == 2) ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}
void SanaWmLtxConnectorBlockPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*,
                                                    int32_t,
                                                    nvinfer1::DynamicPluginTensorDesc const*,
                                                    int32_t) noexcept {}
size_t SanaWmLtxConnectorBlockPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                       nvinfer1::PluginTensorDesc const*,
                                                       int32_t) const noexcept {
    return 0;
}
void SanaWmLtxConnectorBlockPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const auto have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(weights_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}
bool SanaWmLtxConnectorBlockPlugin::ensureDeviceCache(cudaStream_t stream,
                                                      int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(weights_device_, packed_weights_.data(),
                                packed_weights_.size() * sizeof(uint16_t), stream);
}
int32_t SanaWmLtxConnectorBlockPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                               nvinfer1::PluginTensorDesc const*,
                                               void const* const* inputs, void* const* outputs,
                                               void*, cudaStream_t stream) noexcept {
    try {
        const auto hidden_dims = inputDesc[0].dims;
        const auto cos_dims = inputDesc[1].dims;
        const auto sin_dims = inputDesc[2].dims;
        const bool interleaved_rope = cos_dims.nbDims == 3 && sin_dims.nbDims == 3 &&
                                      cos_dims.d[0] == 1 && sin_dims.d[0] == 1 &&
                                      cos_dims.d[1] == hidden_dims.d[1] &&
                                      sin_dims.d[1] == hidden_dims.d[1] &&
                                      cos_dims.d[2] == hidden_dim_ && sin_dims.d[2] == hidden_dim_;
        const bool split_rope =
            cos_dims.nbDims == 4 && sin_dims.nbDims == 4 && cos_dims.d[0] == 1 &&
            sin_dims.d[0] == 1 && cos_dims.d[1] == num_heads_ && sin_dims.d[1] == num_heads_ &&
            cos_dims.d[2] == hidden_dims.d[1] && sin_dims.d[2] == hidden_dims.d[1] &&
            cos_dims.d[3] == head_dim_ / 2 && sin_dims.d[3] == head_dim_ / 2;
        if (hidden_dims.nbDims != 3 || hidden_dims.d[0] != 1 || hidden_dims.d[2] != hidden_dim_ ||
            (!interleaved_rope && !split_rope) || hidden_dim_ != num_heads_ * head_dim_ ||
            head_dim_ % 2 != 0 || packed_weights_.size() != expectedWeightCount()) {
            std::fprintf(
                stderr,
                "[trtmc.sana_wm_ltx_connector_block] invalid descriptors: hidden=(%ld,%ld,%ld) "
                "cos_dims=%d sin_dims=%d config=(%d,%d,%d,%d) weights=%zu/%zu\n",
                hidden_dims.nbDims > 0 ? hidden_dims.d[0] : -1,
                hidden_dims.nbDims > 1 ? hidden_dims.d[1] : -1,
                hidden_dims.nbDims > 2 ? hidden_dims.d[2] : -1, cos_dims.nbDims, sin_dims.nbDims,
                hidden_dim_, num_heads_, head_dim_, ff_dim_, packed_weights_.size(),
                expectedWeightCount());
            return 1;
        }
        const int64_t seq_len = hidden_dims.d[1];
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess ||
            !ensureDeviceCache(stream, device_index)) {
            std::fprintf(stderr,
                         "[trtmc.sana_wm_ltx_connector_block] failed to initialize weights: %s\n",
                         cudaGetErrorString(cudaPeekAtLastError()));
            return 1;
        }
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options = bf16_options.dtype(at::kFloat);
        auto hidden =
            at::from_blob(const_cast<void*>(inputs[0]), {1, seq_len, hidden_dim_}, bf16_options);
        auto cos = interleaved_rope
                       ? at::from_blob(const_cast<void*>(inputs[1]), {1, seq_len, hidden_dim_},
                                       float_options)
                       : at::from_blob(const_cast<void*>(inputs[1]),
                                       {1, num_heads_, seq_len, head_dim_ / 2}, float_options);
        auto sin = interleaved_rope
                       ? at::from_blob(const_cast<void*>(inputs[2]), {1, seq_len, hidden_dim_},
                                       float_options)
                       : at::from_blob(const_cast<void*>(inputs[2]),
                                       {1, num_heads_, seq_len, head_dim_ / 2}, float_options);

        size_t cursor = 0;
        auto take = [&](std::vector<int64_t> shape) {
            size_t count = 1;
            for (const auto value : shape)
                count *= static_cast<size_t>(value);
            auto* data = static_cast<uint16_t*>(weights_device_) + cursor;
            cursor += count;
            return at::from_blob(data, shape, bf16_options);
        };
        auto q_norm_weight = take({hidden_dim_});
        auto k_norm_weight = take({hidden_dim_});
        std::array<at::Tensor, 4> projection_weights;
        std::array<at::Tensor, 4> projection_biases;
        for (int32_t i = 0; i < 4; ++i) {
            projection_weights[static_cast<size_t>(i)] = take({hidden_dim_, hidden_dim_});
            projection_biases[static_cast<size_t>(i)] = take({hidden_dim_});
        }
        auto ff_in_weight = take({ff_dim_, hidden_dim_});
        auto ff_in_bias = take({ff_dim_});
        auto ff_out_weight = take({hidden_dim_, ff_dim_});
        auto ff_out_bias = take({hidden_dim_});

        const std::array<int64_t, 1> norm_shape{hidden_dim_};
        const auto apply_rope = [&](const at::Tensor& value) {
            if (interleaved_rope) {
                auto pairs = value.view({1, seq_len, hidden_dim_ / 2, 2});
                auto real = pairs.select(-1, 0);
                auto imag = pairs.select(-1, 1);
                auto rotated = at::stack({-imag, real}, -1).flatten(2);
                return (value.to(at::kFloat) * cos + rotated.to(at::kFloat) * sin)
                    .to(at::kBFloat16);
            }
            auto split_value = value.view({1, seq_len, num_heads_, head_dim_})
                                   .swapaxes(1, 2)
                                   .view({1, num_heads_, seq_len, 2, head_dim_ / 2})
                                   .to(at::kFloat);
            auto first = split_value.narrow(-2, 0, 1);
            auto second = split_value.narrow(-2, 1, 1);
            auto cos_u = cos.unsqueeze(-2);
            auto sin_u = sin.unsqueeze(-2);
            auto rotated = split_value * cos_u;
            auto first_out = rotated.narrow(-2, 0, 1);
            auto second_out = rotated.narrow(-2, 1, 1);
            first_out.addcmul_(-sin_u, second);
            second_out.addcmul_(sin_u, first);
            return rotated.view({1, num_heads_, seq_len, head_dim_})
                .swapaxes(1, 2)
                .reshape({1, seq_len, hidden_dim_})
                .to(at::kBFloat16);
        };

        auto norm1 = at::rms_norm(hidden, norm_shape, std::nullopt, 1.0e-6);
        auto query = at::linear(norm1, projection_weights[0], projection_biases[0]);
        auto key = at::linear(norm1, projection_weights[1], projection_biases[1]);
        auto value = at::linear(norm1, projection_weights[2], projection_biases[2]);
        query = apply_rope(at::rms_norm(query, norm_shape, q_norm_weight, 1.0e-6));
        key = apply_rope(at::rms_norm(key, norm_shape, k_norm_weight, 1.0e-6));
        query = query.view({1, seq_len, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        key = key.view({1, seq_len, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        value = value.view({1, seq_len, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto zero_mask = at::zeros({1, num_heads_, 1, seq_len}, bf16_options);
        auto attended = at::scaled_dot_product_attention(query, key, value,
                                                         std::optional<at::Tensor>(zero_mask), 0.0,
                                                         false, std::nullopt, false);
        attended = attended.permute({0, 2, 1, 3}).flatten(2);
        auto attention_output = at::linear(attended, projection_weights[3], projection_biases[3]);
        auto residual = hidden + attention_output;
        auto norm2 = at::rms_norm(residual, norm_shape, std::nullopt, 1.0e-6);
        auto ff = at::gelu(at::linear(norm2, ff_in_weight, ff_in_bias), "tanh");
        ff = at::linear(ff, ff_out_weight, ff_out_bias);
        auto result = residual + ff;

        auto output = at::from_blob(outputs[0], {1, seq_len, hidden_dim_}, bf16_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_connector_block] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_connector_block] %s\n", error.what());
        return 1;
    }
}

SanaWmLtxRmsNormPlugin::SanaWmLtxRmsNormPlugin(float eps) : eps_(eps) {}
SanaWmLtxRmsNormPlugin::SanaWmLtxRmsNormPlugin(const void* data, size_t length) {
    if (data != nullptr && length >= sizeof(eps_))
        std::memcpy(&eps_, data, sizeof(eps_));
}
char const* SanaWmLtxRmsNormPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxRmsNormPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxRmsNormPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmLtxRmsNormPlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxRmsNormPlugin::terminate() noexcept {}
void SanaWmLtxRmsNormPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxRmsNormPlugin::getSerializationSize() const noexcept {
    return sizeof(eps_);
}
void SanaWmLtxRmsNormPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &eps_, sizeof(eps_));
}
void SanaWmLtxRmsNormPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxRmsNormPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxRmsNormPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                             int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxRmsNormPlugin* SanaWmLtxRmsNormPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLtxRmsNormPlugin(eps_);
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs SanaWmLtxRmsNormPlugin::getOutputDimensions(int32_t,
                                                                nvinfer1::DimsExprs const* inputs,
                                                                int32_t,
                                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmLtxRmsNormPlugin::supportsFormatCombination(int32_t pos,
                                                       nvinfer1::PluginTensorDesc const* inOut,
                                                       int32_t nbInputs,
                                                       int32_t nbOutputs) noexcept {
    return nbInputs == 1 && nbOutputs == 1 && pos >= 0 && pos < 2 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == nvinfer1::DataType::kBF16;
}
void SanaWmLtxRmsNormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                             nvinfer1::DynamicPluginTensorDesc const*,
                                             int32_t) noexcept {}
size_t SanaWmLtxRmsNormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                nvinfer1::PluginTensorDesc const*,
                                                int32_t) const noexcept {
    return 0;
}
int32_t SanaWmLtxRmsNormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                        nvinfer1::PluginTensorDesc const*,
                                        void const* const* inputs, void* const* outputs, void*,
                                        cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        if (dims.nbDims <= 0 || dims.d[dims.nbDims - 1] <= 0)
            return 1;
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(dims.nbDims));
        for (int32_t i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0)
                return 1;
            shape.push_back(dims.d[i]);
        }
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto input = at::from_blob(const_cast<void*>(inputs[0]), shape, options);
        const std::array<int64_t, 1> normalized_shape{shape.back()};
        auto result = at::rms_norm(input, normalized_shape, std::nullopt, eps_);
        auto output = at::from_blob(outputs[0], shape, options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_rms_norm] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_rms_norm] %s\n", error.what());
        return 1;
    }
}

SanaWmLtxTimestepFrequencyPlugin::SanaWmLtxTimestepFrequencyPlugin(int32_t frequency_dim,
                                                                   float max_period)
    : frequency_dim_(frequency_dim), max_period_(max_period) {}
SanaWmLtxTimestepFrequencyPlugin::SanaWmLtxTimestepFrequencyPlugin(const void* data,
                                                                   size_t length) {
    if (data == nullptr || length < sizeof(frequency_dim_) + sizeof(max_period_))
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::memcpy(&frequency_dim_, cursor, sizeof(frequency_dim_));
    cursor += sizeof(frequency_dim_);
    std::memcpy(&max_period_, cursor, sizeof(max_period_));
}
char const* SanaWmLtxTimestepFrequencyPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxTimestepFrequencyPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxTimestepFrequencyPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmLtxTimestepFrequencyPlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxTimestepFrequencyPlugin::terminate() noexcept {}
void SanaWmLtxTimestepFrequencyPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxTimestepFrequencyPlugin::getSerializationSize() const noexcept {
    return sizeof(frequency_dim_) + sizeof(max_period_);
}
void SanaWmLtxTimestepFrequencyPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    std::memcpy(cursor, &frequency_dim_, sizeof(frequency_dim_));
    cursor += sizeof(frequency_dim_);
    std::memcpy(cursor, &max_period_, sizeof(max_period_));
}
void SanaWmLtxTimestepFrequencyPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxTimestepFrequencyPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxTimestepFrequencyPlugin::getOutputDataType(int32_t,
                                                                       nvinfer1::DataType const*,
                                                                       int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxTimestepFrequencyPlugin* SanaWmLtxTimestepFrequencyPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLtxTimestepFrequencyPlugin(frequency_dim_, max_period_);
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs SanaWmLtxTimestepFrequencyPlugin::getOutputDimensions(
    int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    auto output = inputs[0];
    if (output.nbDims > 0)
        output.d[output.nbDims - 1] = exprBuilder.constant(frequency_dim_);
    return output;
}
bool SanaWmLtxTimestepFrequencyPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    return nbInputs == 1 && nbOutputs == 1 && pos >= 0 && pos < 2 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
           inOut[pos].type == (pos == 0 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}
void SanaWmLtxTimestepFrequencyPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*,
                                                       int32_t,
                                                       nvinfer1::DynamicPluginTensorDesc const*,
                                                       int32_t) noexcept {}
size_t SanaWmLtxTimestepFrequencyPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*,
                                                          int32_t,
                                                          nvinfer1::PluginTensorDesc const*,
                                                          int32_t) const noexcept {
    return 0;
}
int32_t SanaWmLtxTimestepFrequencyPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                                  nvinfer1::PluginTensorDesc const*,
                                                  void const* const* inputs, void* const* outputs,
                                                  void*, cudaStream_t stream) noexcept {
    try {
        const auto dims = inputDesc[0].dims;
        if (dims.nbDims != 2 || dims.d[0] <= 0 || dims.d[1] != 1 || frequency_dim_ <= 0 ||
            frequency_dim_ % 2 != 0 || max_period_ <= 0.0F) {
            return 1;
        }
        const int64_t rows = dims.d[0];
        const int64_t half = frequency_dim_ / 2;
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess)
            return 1;
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto float_options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        const auto bf16_options = float_options.dtype(at::kBFloat16);
        auto timestep =
            at::from_blob(const_cast<void*>(inputs[0]), {rows, 1}, float_options).reshape({rows});
        auto exponent =
            -std::log(static_cast<double>(max_period_)) * at::arange(half, float_options);
        exponent = exponent / static_cast<double>(half);
        auto args = timestep.unsqueeze(1) * at::exp(exponent).unsqueeze(0);
        auto result = at::cat({at::cos(args), at::sin(args)}, -1).to(at::kBFloat16);
        auto output = at::from_blob(outputs[0], {rows, frequency_dim_}, bf16_options);
        output.copy_(result);
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_timestep_frequency] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_timestep_frequency] %s\n", error.what());
        return 1;
    }
}

SanaWmLtxVideoBlockPlugin::SanaWmLtxVideoBlockPlugin(int32_t hidden_dim, int32_t num_heads,
                                                     int32_t head_dim, int32_t ff_dim,
                                                     int32_t context_tokens, bool debug,
                                                     const float* packed_weights,
                                                     int32_t weight_count)
    : hidden_dim_(hidden_dim), num_heads_(num_heads), head_dim_(head_dim), ff_dim_(ff_dim),
      context_tokens_(context_tokens), debug_(debug) {
    append_bf16_values(packed_weights_, packed_weights, weight_count);
}
SanaWmLtxVideoBlockPlugin::SanaWmLtxVideoBlockPlugin(const void* data, size_t length) {
    constexpr size_t kHeader = 6 * sizeof(int32_t);
    if (data == nullptr || length < kHeader)
        return;
    const auto* cursor = static_cast<const char*>(data);
    int32_t debug = 0;
    for (int32_t* value :
         {&hidden_dim_, &num_heads_, &head_dim_, &ff_dim_, &context_tokens_, &debug}) {
        std::memcpy(value, cursor, sizeof(*value));
        cursor += sizeof(*value);
    }
    debug_ = debug != 0;
    const auto count = expectedWeightCount();
    if (length >= kHeader + count * sizeof(uint16_t)) {
        packed_weights_.resize(count);
        std::memcpy(packed_weights_.data(), cursor, count * sizeof(uint16_t));
    }
}
SanaWmLtxVideoBlockPlugin::~SanaWmLtxVideoBlockPlugin() {
    releaseDeviceCache();
}
std::size_t SanaWmLtxVideoBlockPlugin::expectedWeightCount() const noexcept {
    if (hidden_dim_ <= 0 || ff_dim_ <= 0)
        return 0;
    const auto hidden = static_cast<size_t>(hidden_dim_);
    const auto ff = static_cast<size_t>(ff_dim_);
    const auto attention = 2 * hidden + 4 * (hidden * hidden + hidden);
    return 6 * hidden + 2 * attention + (ff * hidden + ff) + (hidden * ff + hidden);
}
char const* SanaWmLtxVideoBlockPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxVideoBlockPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxVideoBlockPlugin::getNbOutputs() const noexcept {
    return debug_ ? 11 : 1;
}
int32_t SanaWmLtxVideoBlockPlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxVideoBlockPlugin::terminate() noexcept {
    releaseDeviceCache();
}
void SanaWmLtxVideoBlockPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxVideoBlockPlugin::getSerializationSize() const noexcept {
    return 6 * sizeof(int32_t) + packed_weights_.size() * sizeof(uint16_t);
}
void SanaWmLtxVideoBlockPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    const int32_t debug = debug_ ? 1 : 0;
    const std::array<int32_t, 6> header{hidden_dim_, num_heads_,      head_dim_,
                                        ff_dim_,     context_tokens_, debug};
    std::memcpy(cursor, header.data(), sizeof(header));
    cursor += sizeof(header);
    if (!packed_weights_.empty())
        std::memcpy(cursor, packed_weights_.data(), packed_weights_.size() * sizeof(uint16_t));
}
void SanaWmLtxVideoBlockPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxVideoBlockPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxVideoBlockPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxVideoBlockPlugin* SanaWmLtxVideoBlockPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLtxVideoBlockPlugin();
    plugin->hidden_dim_ = hidden_dim_;
    plugin->num_heads_ = num_heads_;
    plugin->head_dim_ = head_dim_;
    plugin->ff_dim_ = ff_dim_;
    plugin->context_tokens_ = context_tokens_;
    plugin->debug_ = debug_;
    plugin->packed_weights_ = packed_weights_;
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs
SanaWmLtxVideoBlockPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                               nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool SanaWmLtxVideoBlockPlugin::supportsFormatCombination(int32_t pos,
                                                          nvinfer1::PluginTensorDesc const* inOut,
                                                          int32_t nbInputs,
                                                          int32_t nbOutputs) noexcept {
    if (nbInputs != 5 || nbOutputs != getNbOutputs() || pos < 0 || pos >= nbInputs + nbOutputs ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type ==
           ((pos == 3 || pos == 4) ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}
void SanaWmLtxVideoBlockPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                nvinfer1::DynamicPluginTensorDesc const*,
                                                int32_t) noexcept {}
size_t SanaWmLtxVideoBlockPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                   nvinfer1::PluginTensorDesc const*,
                                                   int32_t) const noexcept {
    return 0;
}
void SanaWmLtxVideoBlockPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const auto have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(weights_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}
bool SanaWmLtxVideoBlockPlugin::ensureDeviceCache(cudaStream_t stream,
                                                  int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(weights_device_, packed_weights_.data(),
                                packed_weights_.size() * sizeof(uint16_t), stream);
}
int32_t SanaWmLtxVideoBlockPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                           nvinfer1::PluginTensorDesc const*,
                                           void const* const* inputs, void* const* outputs, void*,
                                           cudaStream_t stream) noexcept {
    try {
        const auto hidden_dims = inputDesc[0].dims;
        const auto context_dims = inputDesc[1].dims;
        const auto temb_dims = inputDesc[2].dims;
        const auto cos_dims = inputDesc[3].dims;
        const auto sin_dims = inputDesc[4].dims;
        if (hidden_dims.nbDims != 2 || context_dims.nbDims != 2 || temb_dims.nbDims != 2 ||
            cos_dims.nbDims != 4 || sin_dims.nbDims != 4 || hidden_dims.d[1] != hidden_dim_ ||
            context_dims.d[1] != hidden_dim_ || temb_dims.d[0] != hidden_dims.d[0] ||
            temb_dims.d[1] != 6 * hidden_dim_ || cos_dims.d[0] != 1 || sin_dims.d[0] != 1 ||
            cos_dims.d[1] != num_heads_ || sin_dims.d[1] != num_heads_ ||
            cos_dims.d[2] != hidden_dims.d[0] || sin_dims.d[2] != hidden_dims.d[0] ||
            cos_dims.d[3] != head_dim_ / 2 || sin_dims.d[3] != head_dim_ / 2 ||
            hidden_dim_ != num_heads_ * head_dim_ || context_tokens_ <= 0 ||
            context_tokens_ >= hidden_dims.d[0] ||
            packed_weights_.size() != expectedWeightCount()) {
            std::fprintf(stderr, "[trtmc.sana_wm_ltx_video_block] invalid descriptors\n");
            return 1;
        }
        const int64_t tokens = hidden_dims.d[0];
        const int64_t text_tokens = context_dims.d[0];
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess ||
            !ensureDeviceCache(stream, device_index)) {
            return 1;
        }
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options = bf16_options.dtype(at::kFloat);
        auto hidden =
            at::from_blob(const_cast<void*>(inputs[0]), {tokens, hidden_dim_}, bf16_options)
                .unsqueeze(0);
        auto context =
            at::from_blob(const_cast<void*>(inputs[1]), {text_tokens, hidden_dim_}, bf16_options)
                .unsqueeze(0);
        auto temb =
            at::from_blob(const_cast<void*>(inputs[2]), {tokens, 6 * hidden_dim_}, bf16_options)
                .view({1, tokens, 6, hidden_dim_});
        auto cos = at::from_blob(const_cast<void*>(inputs[3]),
                                 {1, num_heads_, tokens, head_dim_ / 2}, float_options);
        auto sin = at::from_blob(const_cast<void*>(inputs[4]),
                                 {1, num_heads_, tokens, head_dim_ / 2}, float_options);

        size_t cursor = 0;
        auto take = [&](std::vector<int64_t> shape) {
            size_t count = 1;
            for (const auto value : shape)
                count *= static_cast<size_t>(value);
            auto* data = static_cast<uint16_t*>(weights_device_) + cursor;
            cursor += count;
            return at::from_blob(data, shape, bf16_options);
        };
        auto scale_shift_table = take({6, hidden_dim_});
        struct AttentionWeights {
            at::Tensor norm_q;
            at::Tensor norm_k;
            std::array<at::Tensor, 4> weights;
            std::array<at::Tensor, 4> biases;
        };
        auto take_attention = [&]() {
            AttentionWeights result{take({hidden_dim_}), take({hidden_dim_}), {}, {}};
            for (int32_t i = 0; i < 4; ++i) {
                result.weights[static_cast<size_t>(i)] = take({hidden_dim_, hidden_dim_});
                result.biases[static_cast<size_t>(i)] = take({hidden_dim_});
            }
            return result;
        };
        auto self_weights = take_attention();
        auto cross_weights = take_attention();
        auto ff_in_weight = take({ff_dim_, hidden_dim_});
        auto ff_in_bias = take({ff_dim_});
        auto ff_out_weight = take({hidden_dim_, ff_dim_});
        auto ff_out_bias = take({hidden_dim_});
        const std::array<int64_t, 1> norm_shape{hidden_dim_};

        const auto block_rms_norm = [](const at::Tensor& value) {
            auto variance = value.to(at::kFloat).pow(2).mean(-1, true);
            return (value * at::rsqrt(variance + 1.0e-6)).to(at::kBFloat16);
        };

        const auto apply_split_rope = [&](const at::Tensor& value) {
            auto split_value = value.view({1, tokens, num_heads_, head_dim_})
                                   .swapaxes(1, 2)
                                   .view({1, num_heads_, tokens, 2, head_dim_ / 2})
                                   .to(at::kFloat);
            auto first = split_value.narrow(-2, 0, 1);
            auto second = split_value.narrow(-2, 1, 1);
            auto cos_u = cos.unsqueeze(-2);
            auto sin_u = sin.unsqueeze(-2);
            auto rotated = split_value * cos_u;
            rotated.narrow(-2, 0, 1).addcmul_(-sin_u, second);
            rotated.narrow(-2, 1, 1).addcmul_(sin_u, first);
            return rotated.view({1, num_heads_, tokens, head_dim_})
                .swapaxes(1, 2)
                .reshape({1, tokens, hidden_dim_})
                .to(at::kBFloat16);
        };
        const auto project_qkv = [&](const at::Tensor& query_input, const at::Tensor& kv_input,
                                     const AttentionWeights& weights, bool rope) {
            auto query = at::linear(query_input, weights.weights[0], weights.biases[0]);
            auto key = at::linear(kv_input, weights.weights[1], weights.biases[1]);
            auto value = at::linear(kv_input, weights.weights[2], weights.biases[2]);
            query = at::rms_norm(query, norm_shape, weights.norm_q, 1.0e-6);
            key = at::rms_norm(key, norm_shape, weights.norm_k, 1.0e-6);
            if (rope) {
                query = apply_split_rope(query);
                key = apply_split_rope(key);
            }
            return std::array<at::Tensor, 3>{query, key, value};
        };
        const auto attention_output = [&](const at::Tensor& query_input, const at::Tensor& kv_input,
                                          const AttentionWeights& weights, bool streaming,
                                          bool rope) {
            auto qkv = project_qkv(query_input, kv_input, weights, rope);
            auto query = qkv[0].view({1, tokens, num_heads_, head_dim_}).permute({0, 2, 1, 3});
            const int64_t kv_tokens = kv_input.size(1);
            auto key = qkv[1].view({1, kv_tokens, num_heads_, head_dim_}).permute({0, 2, 1, 3});
            auto value = qkv[2].view({1, kv_tokens, num_heads_, head_dim_}).permute({0, 2, 1, 3});
            at::Tensor attended;
            if (streaming) {
                auto sink = at::scaled_dot_product_attention(
                    query.narrow(2, 0, context_tokens_), key.narrow(2, 0, context_tokens_),
                    value.narrow(2, 0, context_tokens_), std::nullopt, 0.0, false, std::nullopt,
                    false);
                auto current = at::scaled_dot_product_attention(
                    query.narrow(2, context_tokens_, tokens - context_tokens_), key, value,
                    std::nullopt, 0.0, false, std::nullopt, false);
                attended = at::cat({sink, current}, 2);
            } else {
                attended = at::scaled_dot_product_attention(query, key, value, std::nullopt, 0.0,
                                                            false, std::nullopt, false);
            }
            attended = attended.permute({0, 2, 1, 3}).flatten(2);
            return at::linear(attended, weights.weights[3], weights.biases[3]);
        };

        auto ada = scale_shift_table.view({1, 1, 6, hidden_dim_}) + temb;
        auto ada_values = ada.unbind(2);
        auto norm1 = block_rms_norm(hidden);
        auto mod1 = norm1 * (1 + ada_values[1]) + ada_values[0];
        auto self_attn = attention_output(mod1, mod1, self_weights, true, true);
        auto post_self = hidden + self_attn * ada_values[2];
        auto norm2 = block_rms_norm(post_self);
        auto cross_attn = attention_output(norm2, context, cross_weights, false, false);
        auto post_cross = post_self + cross_attn;
        auto norm3 = block_rms_norm(post_cross);
        auto mod3 = norm3 * (1 + ada_values[4]) + ada_values[3];
        auto ff = at::gelu(at::linear(mod3, ff_in_weight, ff_in_bias), "tanh");
        ff = at::linear(ff, ff_out_weight, ff_out_bias);
        auto result = post_cross + ff * ada_values[5];

        const auto write = [&](int32_t index, const at::Tensor& value) {
            auto output = at::from_blob(outputs[index], {tokens, hidden_dim_}, bf16_options);
            output.copy_(value.reshape({tokens, hidden_dim_}));
        };
        write(0, result);
        if (debug_) {
            const std::array<at::Tensor, 10> debug_values{
                norm1, mod1, self_attn, post_self, norm2, cross_attn, post_cross, norm3, mod3, ff,
            };
            for (int32_t i = 0; i < static_cast<int32_t>(debug_values.size()); ++i)
                write(i + 1, debug_values[static_cast<size_t>(i)]);
        }
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_video_block] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_video_block] %s\n", error.what());
        return 1;
    }
}

SanaWmLtxVideoOutputPlugin::SanaWmLtxVideoOutputPlugin(int32_t hidden_dim, int32_t output_dim,
                                                       const float* packed_weights,
                                                       int32_t weight_count)
    : hidden_dim_(hidden_dim), output_dim_(output_dim) {
    append_bf16_values(packed_weights_, packed_weights, weight_count);
}
SanaWmLtxVideoOutputPlugin::SanaWmLtxVideoOutputPlugin(const void* data, size_t length) {
    constexpr size_t kHeader = 2 * sizeof(int32_t);
    if (data == nullptr || length < kHeader)
        return;
    const auto* cursor = static_cast<const char*>(data);
    std::array<int32_t, 2> header{};
    std::memcpy(header.data(), cursor, sizeof(header));
    hidden_dim_ = header[0];
    output_dim_ = header[1];
    cursor += sizeof(header);
    const size_t count = (length - kHeader) / sizeof(uint16_t);
    if (count == expectedWeightCount()) {
        packed_weights_.resize(count);
        std::memcpy(packed_weights_.data(), cursor, count * sizeof(uint16_t));
    }
}
SanaWmLtxVideoOutputPlugin::~SanaWmLtxVideoOutputPlugin() {
    releaseDeviceCache();
}
std::size_t SanaWmLtxVideoOutputPlugin::expectedWeightCount() const noexcept {
    if (hidden_dim_ <= 0 || output_dim_ <= 0)
        return 0;
    const auto hidden = static_cast<size_t>(hidden_dim_);
    const auto output = static_cast<size_t>(output_dim_);
    return 2 * hidden + output * hidden + output;
}
char const* SanaWmLtxVideoOutputPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}
char const* SanaWmLtxVideoOutputPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}
int32_t SanaWmLtxVideoOutputPlugin::getNbOutputs() const noexcept {
    return 1;
}
int32_t SanaWmLtxVideoOutputPlugin::initialize() noexcept {
    return 0;
}
void SanaWmLtxVideoOutputPlugin::terminate() noexcept {
    releaseDeviceCache();
}
void SanaWmLtxVideoOutputPlugin::destroy() noexcept {
    delete this;
}
size_t SanaWmLtxVideoOutputPlugin::getSerializationSize() const noexcept {
    return 2 * sizeof(int32_t) + packed_weights_.size() * sizeof(uint16_t);
}
void SanaWmLtxVideoOutputPlugin::serialize(void* buffer) const noexcept {
    auto* cursor = static_cast<char*>(buffer);
    const std::array<int32_t, 2> header{hidden_dim_, output_dim_};
    std::memcpy(cursor, header.data(), sizeof(header));
    cursor += sizeof(header);
    if (!packed_weights_.empty())
        std::memcpy(cursor, packed_weights_.data(), packed_weights_.size() * sizeof(uint16_t));
}
void SanaWmLtxVideoOutputPlugin::setPluginNamespace(char const* ns) noexcept {
    namespace_ = ns ? ns : "";
}
char const* SanaWmLtxVideoOutputPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}
nvinfer1::DataType SanaWmLtxVideoOutputPlugin::getOutputDataType(int32_t, nvinfer1::DataType const*,
                                                                 int32_t) const noexcept {
    return nvinfer1::DataType::kBF16;
}
SanaWmLtxVideoOutputPlugin* SanaWmLtxVideoOutputPlugin::clone() const noexcept {
    auto* plugin = new SanaWmLtxVideoOutputPlugin();
    plugin->hidden_dim_ = hidden_dim_;
    plugin->output_dim_ = output_dim_;
    plugin->packed_weights_ = packed_weights_;
    plugin->namespace_ = namespace_;
    return plugin;
}
nvinfer1::DimsExprs
SanaWmLtxVideoOutputPlugin::getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                                nvinfer1::IExprBuilder&) noexcept {
    return inputs[2];
}
bool SanaWmLtxVideoOutputPlugin::supportsFormatCombination(int32_t pos,
                                                           nvinfer1::PluginTensorDesc const* inOut,
                                                           int32_t nbInputs,
                                                           int32_t nbOutputs) noexcept {
    if (nbInputs != 4 || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs ||
        inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    }
    return inOut[pos].type == (pos == 3 ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kBF16);
}
void SanaWmLtxVideoOutputPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                                 nvinfer1::DynamicPluginTensorDesc const*,
                                                 int32_t) noexcept {}
size_t SanaWmLtxVideoOutputPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                                                    nvinfer1::PluginTensorDesc const*,
                                                    int32_t) const noexcept {
    return 0;
}
void SanaWmLtxVideoOutputPlugin::releaseDeviceCache() noexcept {
    int old_device = 0;
    const auto have_device = cudaGetDevice(&old_device);
    if (cached_device_ >= 0)
        cudaSetDevice(cached_device_);
    free_device_cache(weights_device_);
    cached_device_ = -1;
    if (have_device == cudaSuccess)
        cudaSetDevice(old_device);
}
bool SanaWmLtxVideoOutputPlugin::ensureDeviceCache(cudaStream_t stream,
                                                   int32_t device_index) noexcept {
    if (cached_device_ != device_index)
        releaseDeviceCache();
    if (cudaSetDevice(device_index) != cudaSuccess)
        return false;
    cached_device_ = device_index;
    return copy_to_device_cache(weights_device_, packed_weights_.data(),
                                packed_weights_.size() * sizeof(uint16_t), stream);
}
int32_t SanaWmLtxVideoOutputPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                                            nvinfer1::PluginTensorDesc const*,
                                            void const* const* inputs, void* const* outputs, void*,
                                            cudaStream_t stream) noexcept {
    try {
        const auto hidden_dims = inputDesc[0].dims;
        const auto embedded_dims = inputDesc[1].dims;
        const auto latent_dims = inputDesc[2].dims;
        const auto timestep_dims = inputDesc[3].dims;
        if (hidden_dims.nbDims != 2 || embedded_dims.nbDims != 2 || latent_dims.nbDims != 2 ||
            timestep_dims.nbDims != 2 || hidden_dims.d[1] != hidden_dim_ ||
            embedded_dims.d[0] != hidden_dims.d[0] || embedded_dims.d[1] != hidden_dim_ ||
            latent_dims.d[0] != hidden_dims.d[0] || latent_dims.d[1] != output_dim_ ||
            timestep_dims.d[0] != hidden_dims.d[0] || timestep_dims.d[1] != 1 ||
            packed_weights_.size() != expectedWeightCount()) {
            std::fprintf(stderr, "[trtmc.sana_wm_ltx_video_output] invalid descriptors\n");
            return 1;
        }
        const int64_t tokens = hidden_dims.d[0];
        int device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess ||
            !ensureDeviceCache(stream, device_index)) {
            return 1;
        }
        const auto torch_stream = c10::cuda::getStreamFromExternal(stream, device_index);
        c10::cuda::CUDAStreamGuard guard(torch_stream);
        const auto bf16_options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        const auto float_options = bf16_options.dtype(at::kFloat);
        auto hidden =
            at::from_blob(const_cast<void*>(inputs[0]), {tokens, hidden_dim_}, bf16_options)
                .unsqueeze(0);
        auto embedded =
            at::from_blob(const_cast<void*>(inputs[1]), {tokens, hidden_dim_}, bf16_options)
                .unsqueeze(0);
        auto latent =
            at::from_blob(const_cast<void*>(inputs[2]), {tokens, output_dim_}, bf16_options)
                .unsqueeze(0);
        auto raw_timestep =
            at::from_blob(const_cast<void*>(inputs[3]), {tokens, 1}, float_options).unsqueeze(0);

        auto* weight_data = static_cast<uint16_t*>(weights_device_);
        auto scale_shift_table = at::from_blob(weight_data, {2, hidden_dim_}, bf16_options);
        weight_data += 2 * hidden_dim_;
        auto projection_weight =
            at::from_blob(weight_data, {output_dim_, hidden_dim_}, bf16_options);
        weight_data += static_cast<size_t>(output_dim_) * static_cast<size_t>(hidden_dim_);
        auto projection_bias = at::from_blob(weight_data, {output_dim_}, bf16_options);

        auto scale_shift = scale_shift_table.view({1, 1, 2, hidden_dim_}) + embedded.unsqueeze(2);
        auto values = scale_shift.unbind(2);
        auto normalized = at::layer_norm(hidden, {hidden_dim_}, {}, {}, 1.0e-6, true);
        auto modulated = normalized * (1 + values[1]) + values[0];
        auto velocity = at::linear(modulated, projection_weight, projection_bias);
        auto denoised =
            (latent.to(at::kFloat) - velocity.to(at::kFloat) * raw_timestep).to(at::kBFloat16);
        auto output = at::from_blob(outputs[0], {tokens, output_dim_}, bf16_options);
        output.copy_(denoised.reshape({tokens, output_dim_}));
        return 0;
    } catch (const c10::Error& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_video_output] %s\n", error.what());
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.sana_wm_ltx_video_output] %s\n", error.what());
        return 1;
    }
}

} // namespace trtmc

#endif // TRTMC_HAS_TRT && TRTMC_HAS_LIBTORCH_CONV
