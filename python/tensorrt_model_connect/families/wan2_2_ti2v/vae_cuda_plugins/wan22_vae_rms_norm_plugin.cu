/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace trtmc::wan22 {
namespace {

constexpr int32_t kTHREADS = 256;
constexpr float kNORMALIZE_EPSILON = 1.0e-12F;

__global__ void vaeRmsNormKernel(const float* input, const float* gamma, float* output,
                                 int32_t channels, int64_t spatial_volume,
                                 int64_t position_count, float scale) {
    const int64_t position = static_cast<int64_t>(blockIdx.x);
    if (position >= position_count)
        return;

    const int32_t thread = static_cast<int32_t>(threadIdx.x);
    const int64_t batch = position / spatial_volume;
    const int64_t spatial = position - batch * spatial_volume;
    const int64_t base = batch * static_cast<int64_t>(channels) * spatial_volume + spatial;

    float sum = 0.0F;
    for (int32_t channel = thread; channel < channels; channel += blockDim.x) {
        const float value = input[base + static_cast<int64_t>(channel) * spatial_volume];
        sum = fmaf(value, value, sum);
    }

    __shared__ float reduction[kTHREADS];
    reduction[thread] = sum;
    __syncthreads();
    for (int32_t offset = kTHREADS / 2; offset > 0; offset /= 2) {
        if (thread < offset)
            reduction[thread] += reduction[thread + offset];
        __syncthreads();
    }

    __shared__ float denominator;
    if (thread == 0)
        denominator = fmaxf(sqrtf(reduction[0]), kNORMALIZE_EPSILON);
    __syncthreads();

    // Preserve the source operation order: normalize, multiply by sqrt(C),
    // then apply the learned per-channel gamma.
    for (int32_t channel = thread; channel < channels; channel += blockDim.x) {
        const int64_t index = base + static_cast<int64_t>(channel) * spatial_volume;
        const float normalized = input[index] / denominator;
        output[index] = (normalized * scale) * gamma[channel];
    }
}

} // namespace

class VaeRmsNormPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22VaeRmsNorm";
    static constexpr const char* kVERSION = "1";

    VaeRmsNormPlugin() = default;
    VaeRmsNormPlugin(const void*, size_t) {}

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
    VaeRmsNormPlugin* clone() const noexcept override {
        auto* result = new VaeRmsNormPlugin();
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
        return input_count == 2 && output_count == 1 && position >= 0 && position < 3 &&
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
        if (input_desc == nullptr || inputs == nullptr || outputs == nullptr || inputs[0] == nullptr ||
            inputs[1] == nullptr || outputs[0] == nullptr)
            return 1;
        const nvinfer1::Dims& input_dims = input_desc[0].dims;
        const nvinfer1::Dims& gamma_dims = input_desc[1].dims;
        if (input_dims.nbDims != 5 || gamma_dims.nbDims != 4 || input_dims.d[0] <= 0 ||
            input_dims.d[1] <= 0 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0 ||
            input_dims.d[4] <= 0 || gamma_dims.d[0] != input_dims.d[1])
            return 1;

        const int32_t channels = input_dims.d[1];
        const int64_t spatial_volume = static_cast<int64_t>(input_dims.d[2]) * input_dims.d[3] *
                                       input_dims.d[4];
        const int64_t position_count = static_cast<int64_t>(input_dims.d[0]) * spatial_volume;
        if (position_count <= 0 || position_count > static_cast<int64_t>(UINT32_MAX))
            return 1;
        const float scale = sqrtf(static_cast<float>(channels));
        vaeRmsNormKernel<<<static_cast<uint32_t>(position_count), kTHREADS, 0, stream>>>(
            static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<float*>(outputs[0]), channels, spatial_volume, position_count, scale);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class VaeRmsNormCreator final : public nvinfer1::IPluginCreator {
  public:
    VaeRmsNormCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return VaeRmsNormPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override { return VaeRmsNormPlugin::kVERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(
        char const*, nvinfer1::PluginFieldCollection const*) noexcept override {
        return new VaeRmsNormPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new VaeRmsNormPlugin(data, length);
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

static nvinfer1::PluginRegistrar<trtmc::wan22::VaeRmsNormCreator>
    plugin_registrar_wan22_vae_rms_norm{};
