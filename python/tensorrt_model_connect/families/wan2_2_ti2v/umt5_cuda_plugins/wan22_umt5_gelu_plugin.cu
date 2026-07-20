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

__device__ __forceinline__ __nv_bfloat16 bf16(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float fp32(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 rounded_mul(__nv_bfloat16 left, __nv_bfloat16 right) {
    return bf16(fp32(left) * fp32(right));
}

__device__ __forceinline__ __nv_bfloat16 rounded_add(__nv_bfloat16 left, __nv_bfloat16 right) {
    return bf16(fp32(left) + fp32(right));
}

__device__ __forceinline__ __nv_bfloat16 rounded_mul_scalar(__nv_bfloat16 value, float scalar) {
    // PyTorch's wrapped scalar remains FP32 during the BF16 pointwise kernel;
    // only the tensor result is rounded to BF16.
    return bf16(fp32(value) * scalar);
}

__global__ void source_gelu_kernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                                   int64_t count) {
    const __nv_bfloat16 one = bf16(1.0F);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const __nv_bfloat16 x = input[index];

        // Match the distinct PyTorch BF16 pointwise kernels in Wan's GELU:
        // pow -> multiply -> add -> multiply -> tanh -> add -> two multiplies.
        // Each assignment rounds to BF16 before the following operation.
        // torch.pow(BF16, 3.0) takes its integer-exponent specialization and
        // rounds x*x before the final multiply.
        const __nv_bfloat16 square = rounded_mul(x, x);
        const __nv_bfloat16 cube = rounded_mul(square, x);
        const __nv_bfloat16 cubic = rounded_mul_scalar(cube, 0.044715F);
        const __nv_bfloat16 polynomial = rounded_add(x, cubic);
        const __nv_bfloat16 tanh_input = rounded_mul_scalar(polynomial, 0.7978845608028654F);
        const __nv_bfloat16 tanh_output = bf16(tanhf(fp32(tanh_input)));
        const __nv_bfloat16 shifted = rounded_add(one, tanh_output);
        const __nv_bfloat16 scaled_x = rounded_mul_scalar(x, 0.5F);
        output[index] = rounded_mul(scaled_x, shifted);
    }
}

constexpr int32_t kUmt5SoftmaxElements = 512;
constexpr int32_t kUmt5SoftmaxRows = 64 * 512;
constexpr int32_t kUmt5SoftmaxWarpSize = 32;
constexpr int32_t kUmt5SoftmaxIterations = kUmt5SoftmaxElements / kUmt5SoftmaxWarpSize;

__device__ __forceinline__ float warp_xor_max(float value) {
#pragma unroll
    for (int32_t offset = kUmt5SoftmaxWarpSize / 2; offset > 0; offset /= 2) {
        const float other = __shfl_xor_sync(0xffffffffU, value, offset, kUmt5SoftmaxWarpSize);
        value = value < other ? other : value;
    }
    return value;
}

__device__ __forceinline__ float warp_xor_sum(float value) {
#pragma unroll
    for (int32_t offset = kUmt5SoftmaxWarpSize / 2; offset > 0; offset /= 2) {
        const float other = __shfl_xor_sync(0xffffffffU, value, offset, kUmt5SoftmaxWarpSize);
        value = value + other;
    }
    return value;
}

// Match PyTorch 2.12's persistent CUDA softmax for a contiguous FP32
// reduction of 512 values, including its lane-local accumulation and XOR
// reduction order.  The BF16 input/output encapsulate Wan's
// F.softmax(attn.float(), dim=-1).type_as(attn) contract.
__global__ void source_softmax_512_kernel(const __nv_bfloat16* input, __nv_bfloat16* output) {
    const int32_t row = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.y) +
                        static_cast<int32_t>(threadIdx.y);
    if (row >= kUmt5SoftmaxRows)
        return;

    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    const __nv_bfloat16* row_input = input + row * kUmt5SoftmaxElements;
    __nv_bfloat16* row_output = output + row * kUmt5SoftmaxElements;

    float elements[kUmt5SoftmaxIterations];
#pragma unroll
    for (int32_t iteration = 0; iteration < kUmt5SoftmaxIterations; ++iteration) {
        const int32_t element = lane + iteration * kUmt5SoftmaxWarpSize;
        elements[iteration] = fp32(row_input[element]);
    }

    float maximum = elements[0];
#pragma unroll
    for (int32_t iteration = 0; iteration < kUmt5SoftmaxIterations; ++iteration) {
        maximum = maximum > elements[iteration] ? maximum : elements[iteration];
    }
    maximum = warp_xor_max(maximum);

    float sum = 0.0F;
#pragma unroll
    for (int32_t iteration = 0; iteration < kUmt5SoftmaxIterations; ++iteration) {
        elements[iteration] = std::exp(elements[iteration] - maximum);
        sum += elements[iteration];
    }
    sum = warp_xor_sum(sum);

#pragma unroll
    for (int32_t iteration = 0; iteration < kUmt5SoftmaxIterations; ++iteration) {
        const int32_t element = lane + iteration * kUmt5SoftmaxWarpSize;
        row_output[element] = bf16(elements[iteration] / sum);
    }
}

constexpr int32_t kUmt5RmsNormRows = 512;
constexpr int32_t kUmt5RmsNormElements = 4096;
constexpr int32_t kUmt5RmsNormWarpSize = 32;
constexpr int32_t kUmt5RmsNormVectorSize = 4;
constexpr int32_t kUmt5RmsNormVectors = kUmt5RmsNormElements / kUmt5RmsNormVectorSize;

// Match PyTorch 2.12's contiguous FP32 mean reduction for [512,4096].
// Reduce.cuh vectorizes the input by four, gives each lane four independent
// accumulators over vector indices lane + 32*k, combines those accumulators in
// index order, and finishes with a shuffle-down warp reduction.  Wan then runs
// FP32 rsqrt, casts x*inverse to BF16, and performs the BF16 affine multiply.
__global__ void source_rmsnorm_512x4096_kernel(const __nv_bfloat16* input,
                                               const __nv_bfloat16* gamma, __nv_bfloat16* output) {
    const int32_t row = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.y) +
                        static_cast<int32_t>(threadIdx.y);
    if (row >= kUmt5RmsNormRows)
        return;

    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    const __nv_bfloat16* row_input = input + row * kUmt5RmsNormElements;
    __nv_bfloat16* row_output = output + row * kUmt5RmsNormElements;
    float sums[kUmt5RmsNormVectorSize] = {0.0F, 0.0F, 0.0F, 0.0F};

    for (int32_t vector = lane; vector < kUmt5RmsNormVectors; vector += kUmt5RmsNormWarpSize) {
        const int32_t base = vector * kUmt5RmsNormVectorSize;
#pragma unroll
        for (int32_t element = 0; element < kUmt5RmsNormVectorSize; ++element) {
            const float value = fp32(row_input[base + element]);
            const float square = __fmul_rn(value, value);
            sums[element] = __fadd_rn(sums[element], square);
        }
    }

    float sum = sums[0];
#pragma unroll
    for (int32_t element = 1; element < kUmt5RmsNormVectorSize; ++element) {
        sum = __fadd_rn(sum, sums[element]);
    }
#pragma unroll
    for (int32_t offset = kUmt5RmsNormWarpSize / 2; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffffU, sum, offset, kUmt5RmsNormWarpSize);
        sum = __fadd_rn(sum, other);
    }
    sum = __shfl_sync(0xffffffffU, sum, 0, kUmt5RmsNormWarpSize);

    const float mean = __fmul_rn(sum, 1.0F / kUmt5RmsNormElements);
    const float inverse = rsqrtf(__fadd_rn(mean, 1.0e-6F));
    for (int32_t element = lane; element < kUmt5RmsNormElements; element += kUmt5RmsNormWarpSize) {
        const __nv_bfloat16 normalized = bf16(__fmul_rn(fp32(row_input[element]), inverse));
        row_output[element] = bf16(__fmul_rn(fp32(gamma[element]), fp32(normalized)));
    }
}

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

} // namespace

class Umt5SourceGeluPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22Umt5SourceGelu";
    static constexpr const char* kVERSION = "1";

    Umt5SourceGeluPlugin() = default;
    Umt5SourceGeluPlugin(const void*, size_t) {}

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
    Umt5SourceGeluPlugin* clone() const noexcept override {
        auto* result = new Umt5SourceGeluPlugin();
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
        const int64_t required_blocks = (count + threads - 1) / threads;
        const int32_t blocks =
            static_cast<int32_t>(required_blocks < 65535 ? required_blocks : 65535);
        source_gelu_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0]), static_cast<__nv_bfloat16*>(outputs[0]),
            count);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class Umt5Bf16BarrierPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22Umt5Bf16Barrier";
    static constexpr const char* kVERSION = "1";

    Umt5Bf16BarrierPlugin() = default;
    Umt5Bf16BarrierPlugin(const void*, size_t) {}
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
    Umt5Bf16BarrierPlugin* clone() const noexcept override {
        auto* result = new Umt5Bf16BarrierPlugin();
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
        const cudaError_t status =
            cudaMemcpyAsync(outputs[0], inputs[0], static_cast<size_t>(count) * sizeof(uint16_t),
                            cudaMemcpyDeviceToDevice, stream);
        return status == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class Umt5SourceSoftmaxPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22Umt5SourceSoftmax";
    static constexpr const char* kVERSION = "1";

    Umt5SourceSoftmaxPlugin() = default;
    Umt5SourceSoftmaxPlugin(const void*, size_t) {}
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
    Umt5SourceSoftmaxPlugin* clone() const noexcept override {
        auto* result = new Umt5SourceSoftmaxPlugin();
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
        if (inputs == nullptr || outputs == nullptr)
            return 1;
        const nvinfer1::Dims& dims = input_desc[0].dims;
        if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 64 || dims.d[2] != 512 ||
            dims.d[3] != kUmt5SoftmaxElements) {
            return 1;
        }
        constexpr int32_t warps_per_block = 4;
        constexpr int32_t blocks = (kUmt5SoftmaxRows + warps_per_block - 1) / warps_per_block;
        const dim3 threads(kUmt5SoftmaxWarpSize, warps_per_block, 1);
        source_softmax_512_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0]), static_cast<__nv_bfloat16*>(outputs[0]));
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class Umt5SourceRmsNormPlugin final : public nvinfer1::IPluginV2DynamicExt {
  public:
    static constexpr const char* kNAME = "Wan22Umt5SourceRmsNorm";
    static constexpr const char* kVERSION = "1";

    Umt5SourceRmsNormPlugin() = default;
    Umt5SourceRmsNormPlugin(const void*, size_t) {}
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
    Umt5SourceRmsNormPlugin* clone() const noexcept override {
        auto* result = new Umt5SourceRmsNormPlugin();
        result->namespace_ = namespace_;
        return result;
    }
    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t position, nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t input_count, int32_t output_count) noexcept override {
        return input_count == 2 && output_count == 1 && position >= 0 && position < 3 &&
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
        if (inputs == nullptr || outputs == nullptr)
            return 1;
        const nvinfer1::Dims& hidden = input_desc[0].dims;
        const nvinfer1::Dims& gamma = input_desc[1].dims;
        if (hidden.nbDims != 2 || hidden.d[0] != kUmt5RmsNormRows ||
            hidden.d[1] != kUmt5RmsNormElements || gamma.nbDims != 1 ||
            gamma.d[0] != kUmt5RmsNormElements) {
            return 1;
        }
        constexpr int32_t warps_per_block = 16;
        constexpr int32_t blocks = (kUmt5RmsNormRows + warps_per_block - 1) / warps_per_block;
        const dim3 threads(kUmt5RmsNormWarpSize, warps_per_block, 1);
        source_rmsnorm_512x4096_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0]),
            static_cast<const __nv_bfloat16*>(inputs[1]), static_cast<__nv_bfloat16*>(outputs[0]));
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

  private:
    std::string namespace_;
};

class Umt5SourceGeluCreator final : public nvinfer1::IPluginCreator {
  public:
    Umt5SourceGeluCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return Umt5SourceGeluPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return Umt5SourceGeluPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new Umt5SourceGeluPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new Umt5SourceGeluPlugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

class Umt5Bf16BarrierCreator final : public nvinfer1::IPluginCreator {
  public:
    Umt5Bf16BarrierCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return Umt5Bf16BarrierPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return Umt5Bf16BarrierPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new Umt5Bf16BarrierPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new Umt5Bf16BarrierPlugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

class Umt5SourceSoftmaxCreator final : public nvinfer1::IPluginCreator {
  public:
    Umt5SourceSoftmaxCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return Umt5SourceSoftmaxPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return Umt5SourceSoftmaxPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new Umt5SourceSoftmaxPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new Umt5SourceSoftmaxPlugin(data, length);
    }
    void setPluginNamespace(char const* value) noexcept override {
        namespace_ = value ? value : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

class Umt5SourceRmsNormCreator final : public nvinfer1::IPluginCreator {
  public:
    Umt5SourceRmsNormCreator() {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return Umt5SourceRmsNormPlugin::kNAME; }
    char const* getPluginVersion() const noexcept override {
        return Umt5SourceRmsNormPlugin::kVERSION;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const*) noexcept override {
        return new Umt5SourceRmsNormPlugin();
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t length) noexcept override {
        return new Umt5SourceRmsNormPlugin(data, length);
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

static nvinfer1::PluginRegistrar<trtmc::wan22::Umt5SourceGeluCreator>
    plugin_registrar_wan22_umt5_source_gelu{};
static nvinfer1::PluginRegistrar<trtmc::wan22::Umt5Bf16BarrierCreator>
    plugin_registrar_wan22_umt5_bf16_barrier{};
static nvinfer1::PluginRegistrar<trtmc::wan22::Umt5SourceSoftmaxCreator>
    plugin_registrar_wan22_umt5_source_softmax{};
static nvinfer1::PluginRegistrar<trtmc::wan22::Umt5SourceRmsNormCreator>
    plugin_registrar_wan22_umt5_source_rmsnorm{};

extern "C" void trtmc_wan22_umt5_cuda_plugin_force_link() {}
