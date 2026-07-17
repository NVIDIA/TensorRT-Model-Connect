/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
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

__global__ void dit_gelu_tanh_kernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                                     int64_t count) {
    // This is the expression used by PyTorch's CUDA GELU tanh kernel.  Its
    // BFloat16 dispatch evaluates the expression in FP32 opmath and rounds
    // only the returned value.  Wan's T5 GELU is a different sequence of
    // independently materialized BF16 pointwise operations.
    constexpr float beta = static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);
    constexpr float kappa = 0.044715F;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const float x = __bfloat162float(input[index]);
        const float x_cube = x * x * x;
        const float inner = beta * (x + kappa * x_cube);
        const float result = 0.5F * x * (1.0F + tanhf(inner));
        output[index] = __float2bfloat16_rn(result);
    }
}

__global__ void dit_rotary_kernel(const float* input, const float* cos_high, const float* sin_high,
                                  const float* cos_low, const float* sin_low, __nv_bfloat16* output,
                                  int64_t rows, int32_t heads, int32_t head_dim) {
    const int64_t half_dim = head_dim / 2;
    const int64_t pair_count = rows * static_cast<int64_t>(heads) * half_dim;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pair_count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = index / (static_cast<int64_t>(heads) * half_dim);
        const int64_t within_row = index % (static_cast<int64_t>(heads) * half_dim);
        const int64_t head = within_row / half_dim;
        const int64_t pair = within_row % half_dim;
        const int64_t scalar =
            row * static_cast<int64_t>(heads) * head_dim + head * head_dim + pair * 2;
        const int64_t frequency = row * half_dim + pair;

        // Upstream constructs complex128 values, multiplies in FP64, then
        // calls .float() before Q/K are cast to V's BFloat16 dtype.  Splitting
        // the constants into high/low FP32 parts lets the plan retain the
        // original FP64 frequency values without an FP64 TensorRT tensor.
        const double cosine =
            static_cast<double>(cos_high[frequency]) + static_cast<double>(cos_low[frequency]);
        const double sine =
            static_cast<double>(sin_high[frequency]) + static_cast<double>(sin_low[frequency]);
        const double real = static_cast<double>(input[scalar]);
        const double imag = static_cast<double>(input[scalar + 1]);
        const float rotated_real = static_cast<float>(real * cosine - imag * sine);
        const float rotated_imag = static_cast<float>(real * sine + imag * cosine);
        output[scalar] = __float2bfloat16_rn(rotated_real);
        output[scalar + 1] = __float2bfloat16_rn(rotated_imag);
    }
}

int32_t launch_blocks(int64_t count, int32_t threads) {
    const int64_t required = (count + threads - 1) / threads;
    return static_cast<int32_t>(required < 65535 ? required : 65535);
}

} // namespace

class DitGeluPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitGelu";
    static constexpr const char* kVERSION = "1";

    DitGeluPlugin() = default;
    DitGeluPlugin(const void*, size_t) {}
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
        return nvinfer1::DataType::kBF16;
    }
    DitGeluPlugin* clone() const noexcept override {
        auto* result = new DitGeluPlugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 1 && output_count == 1 && position >= 0 && position < 2 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kBF16;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        const int64_t count = volume(input_desc[0].dims);
        if (count <= 0 || inputs == nullptr || outputs == nullptr)
            return 1;
        constexpr int32_t threads = 256;
        dit_gelu_tanh_kernel<<<launch_blocks(count, threads), threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0]), static_cast<__nv_bfloat16*>(outputs[0]),
            count);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class DitRotaryPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitRotary";
    static constexpr const char* kVERSION = "1";

    DitRotaryPlugin(int32_t heads, int32_t head_dim) : heads_(heads), head_dim_(head_dim) {}
    DitRotaryPlugin(const void* data, size_t length) {
        if (data != nullptr && length == sizeof(heads_) + sizeof(head_dim_)) {
            std::memcpy(&heads_, data, sizeof(heads_));
            std::memcpy(&head_dim_, static_cast<const char*>(data) + sizeof(heads_),
                        sizeof(head_dim_));
        }
    }
    char const* getPluginType() const noexcept override { return kNAME; }
    char const* getPluginVersion() const noexcept override { return kVERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override {
        return heads_ > 0 && head_dim_ > 0 && head_dim_ % 2 == 0 ? 0 : 1;
    }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    size_t getSerializationSize() const noexcept override {
        return sizeof(heads_) + sizeof(head_dim_);
    }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &heads_, sizeof(heads_));
        std::memcpy(static_cast<char*>(buffer) + sizeof(heads_), &head_dim_, sizeof(head_dim_));
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const*,
                                         int32_t) const noexcept override {
        return nvinfer1::DataType::kBF16;
    }
    DitRotaryPlugin* clone() const noexcept override {
        auto* result = new DitRotaryPlugin(heads_, head_dim_);
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        if (input_count != 5 || output_count != 1 || position < 0 || position >= 6 ||
            in_out[position].format != nvinfer1::TensorFormat::kLINEAR)
            return false;
        return position < 5 ? in_out[position].type == nvinfer1::DataType::kFLOAT
                            : in_out[position].type == nvinfer1::DataType::kBF16;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        if (inputs == nullptr || outputs == nullptr || input_desc[0].dims.nbDims != 2)
            return 1;
        const int64_t rows = input_desc[0].dims.d[0];
        const int64_t hidden = input_desc[0].dims.d[1];
        if (rows <= 0 || hidden != static_cast<int64_t>(heads_) * head_dim_ || head_dim_ <= 0 ||
            head_dim_ % 2 != 0)
            return 1;
        const int64_t pairs = rows * heads_ * (head_dim_ / 2);
        constexpr int32_t threads = 256;
        dit_rotary_kernel<<<launch_blocks(pairs, threads), threads, 0, stream>>>(
            static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), static_cast<const float*>(inputs[3]),
            static_cast<const float*>(inputs[4]), static_cast<__nv_bfloat16*>(outputs[0]), rows,
            heads_, head_dim_);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    int32_t heads_{0};
    int32_t head_dim_{0};
    std::string namespace_;
};

class DitFp32BarrierPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22DitFp32Barrier";
    static constexpr const char* kVERSION = "1";
    DitFp32BarrierPlugin() = default;
    DitFp32BarrierPlugin(const void*, size_t) {}
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
    DitFp32BarrierPlugin* clone() const noexcept override {
        auto* result = new DitFp32BarrierPlugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 1 && output_count == 1 && position >= 0 && position < 2 &&
               in_out[position].format == nvinfer1::TensorFormat::kLINEAR &&
               in_out[position].type == nvinfer1::DataType::kFLOAT;
    }
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
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

class DitGeluCreator final : public nvinfer1::IPluginCreator {
  public:
    DitGeluCreator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return DitGeluPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override { return DitGeluPlugin::kVERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitGeluPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitGeluPlugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

class DitRotaryCreator final : public nvinfer1::IPluginCreator {
  public:
    DitRotaryCreator() {
        field_entries_[0] = {"heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        field_entries_[1] = {"head_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        fields_ = {2, field_entries_};
    }
    char const* getPluginName() const noexcept override { return DitRotaryPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override { return DitRotaryPlugin::kVERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2*
    createPlugin(char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override {
        int32_t heads = 0;
        int32_t head_dim = 0;
        if (fields != nullptr) {
            for (int32_t index = 0; index < fields->nbFields; ++index) {
                const auto& field = fields->fields[index];
                if (field.data == nullptr || field.type != nvinfer1::PluginFieldType::kINT32 ||
                    field.length != 1)
                    continue;
                if (std::strcmp(field.name, "heads") == 0)
                    std::memcpy(&heads, field.data, sizeof(heads));
                else if (std::strcmp(field.name, "head_dim") == 0)
                    std::memcpy(&head_dim, field.data, sizeof(head_dim));
            }
        }
        return heads > 0 && head_dim > 0 ? new DitRotaryPlugin(heads, head_dim) : nullptr;
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitRotaryPlugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginField field_entries_[2]{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

class DitFp32BarrierCreator final : public nvinfer1::IPluginCreator {
  public:
    DitFp32BarrierCreator() { fields_ = {0, nullptr}; }
    char const* getPluginName() const noexcept override { return DitFp32BarrierPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return DitFp32BarrierPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new DitFp32BarrierPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new DitFp32BarrierPlugin(data, length);
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

static nvinfer1::PluginRegistrar<trtmc::wan22::DitGeluCreator> plugin_registrar_wan22_dit_gelu{};
static nvinfer1::PluginRegistrar<trtmc::wan22::DitRotaryCreator>
    plugin_registrar_wan22_dit_rotary{};
static nvinfer1::PluginRegistrar<trtmc::wan22::DitFp32BarrierCreator>
    plugin_registrar_wan22_dit_fp32_barrier{};

extern "C" void trtmc_wan22_dit_cuda_plugin_force_link() {}
