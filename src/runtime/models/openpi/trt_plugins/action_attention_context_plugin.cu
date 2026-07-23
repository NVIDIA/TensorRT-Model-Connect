/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <new>
#include <string>

namespace trtmc::openpi {

int32_t launch_openpi_action_attention_context(const std::uint16_t* query, const std::uint16_t* key,
                                               const std::uint16_t* value,
                                               const std::uint8_t* attention_mask,
                                               std::uint16_t* output, void* workspace,
                                               void* cublas_handle, cudaStream_t stream) noexcept;

namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kQueryHeads = 8;
constexpr int32_t kKeyHeads = 1;
constexpr int32_t kQueryRows = 15;
constexpr int32_t kKeyRows = 983;
constexpr int32_t kPaddedKeyRows = 984;
constexpr int32_t kHeadDimension = 256;
constexpr int32_t kOutputWidth = kQueryHeads * kHeadDimension;
constexpr int32_t kFlattenedQueries = kQueryHeads * kQueryRows;

constexpr std::size_t kPackedQueryElements =
    static_cast<std::size_t>(kFlattenedQueries) * kHeadDimension;
constexpr std::size_t kPackedKeyElements =
    static_cast<std::size_t>(kPaddedKeyRows) * kHeadDimension;
constexpr std::size_t kLogitElements = static_cast<std::size_t>(kPaddedKeyRows) * kFlattenedQueries;
constexpr std::size_t kProbabilityElements =
    static_cast<std::size_t>(kFlattenedQueries) * kPaddedKeyRows;
constexpr std::size_t kContextElements =
    static_cast<std::size_t>(kHeadDimension) * kFlattenedQueries;

constexpr std::size_t kPackedQueryBytes = kPackedQueryElements * sizeof(std::uint16_t);
constexpr std::size_t kPackedKeyBytes = kPackedKeyElements * sizeof(std::uint16_t);
constexpr std::size_t kLogitBytes = kLogitElements * sizeof(float);

// Exact scratch allocations attached to custom-call.65 and custom-call.66 in
// the pinned JAX 0.5.3 / XLA action executable.
constexpr std::size_t kQkCublasWorkspaceBytes = 565248;
constexpr std::size_t kPvCublasWorkspaceBytes = 739968;

// Phase liveness keeps the workspace compact without aliasing a live tensor:
// softmax probabilities overwrite dead padded K after QK, and raw PV context
// overwrites dead FP32 logits after softmax.
constexpr std::size_t kPackedQueryOffset = 0;
constexpr std::size_t kPackedKeyOffset = kPackedQueryOffset + kPackedQueryBytes;
constexpr std::size_t kPackedValueOffset = kPackedKeyOffset + kPackedKeyBytes;
constexpr std::size_t kLogitOffset = kPackedValueOffset + kPackedKeyBytes;
constexpr std::size_t kCublasWorkspaceOffset = kLogitOffset + kLogitBytes;
constexpr std::size_t kPluginWorkspaceBytes = kCublasWorkspaceOffset + kPvCublasWorkspaceBytes;
static_assert(kQkCublasWorkspaceBytes <= kPvCublasWorkspaceBytes);
static_assert(kProbabilityElements * sizeof(std::uint16_t) <= kPackedKeyBytes);
static_assert(kContextElements * sizeof(std::uint16_t) <= kLogitBytes);
static_assert(kPluginWorkspaceBytes == 2281344);

constexpr int32_t kPointwiseThreads = 256;
constexpr int32_t kSoftmaxThreads = 128;
constexpr int32_t kSoftmaxBlocks = kQueryHeads * kQueryRows / 2;
constexpr int32_t kSoftmaxTiles = 8;

__device__ __forceinline__ float bf16_to_float(std::uint16_t value) {
    return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t float_to_bf16_rn(float value) {
    std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    }
    bits += 0x00007FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ std::uint16_t bf16_multiply_rn(std::uint16_t lhs, std::uint16_t rhs) {
    return float_to_bf16_rn(__fmul_rn(bf16_to_float(lhs), bf16_to_float(rhs)));
}

__device__ __forceinline__ float full_divide(float numerator, float denominator) {
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
    return result;
}

// Exact libdevice exp lowering emitted inside action fusion.86.
__device__ __forceinline__ float openpi_action_expf(float value) {
    const float scaled =
        __fmaf_rn(value, __uint_as_float(0x3BBB989DU), __uint_as_float(0x3F000000U));
    float saturated;
    asm("cvt.sat.f32.f32 %0, %1;" : "=f"(saturated) : "f"(scaled));
    const float exponent =
        __fmaf_rd(saturated, __uint_as_float(0x437C0000U), __uint_as_float(0x4B400001U));
    const float rounded = __fadd_rn(exponent, __uint_as_float(0xCB40007FU));
    float reduced = __fmaf_rn(value, __uint_as_float(0x3FB8AA3BU), -rounded);
    reduced = __fmaf_rn(value, __uint_as_float(0x32A57060U), reduced);
    const float power_of_two = __uint_as_float(__float_as_uint(exponent) << 23U);
    float exponential;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(reduced));
    return __fmul_rn(exponential, power_of_two);
}

// Pack XLA's physical operands in one pass. Q is converted from TensorRT's
// [head,token,dimension] layout to [token,head,dimension] and multiplied at
// the BF16 boundary by 1/sqrt(256) == 0.0625. K/V receive XLA's zero row 983.
__global__ void openpi_action_attention_pack_kernel(const std::uint16_t* __restrict__ query,
                                                    const std::uint16_t* __restrict__ key,
                                                    const std::uint16_t* __restrict__ value,
                                                    std::uint16_t* __restrict__ packed_query,
                                                    std::uint16_t* __restrict__ padded_key,
                                                    std::uint16_t* __restrict__ padded_value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < kPackedQueryElements) {
        const int32_t flattened = static_cast<int32_t>(index / kHeadDimension);
        const int32_t dimension = static_cast<int32_t>(index % kHeadDimension);
        const int32_t token = flattened / kQueryHeads;
        const int32_t head = flattened % kQueryHeads;
        const std::size_t source =
            (static_cast<std::size_t>(head) * kQueryRows + token) * kHeadDimension + dimension;
        packed_query[index] = bf16_multiply_rn(query[source], 0x3D80U);
    }
    if (index < kPackedKeyElements) {
        if (index < static_cast<std::size_t>(kKeyRows) * kHeadDimension) {
            padded_key[index] = key[index];
            padded_value[index] = value[index];
        } else {
            padded_key[index] = 0;
            padded_value[index] = 0;
        }
    }
}

__device__ __forceinline__ float warp_xor_max(float value) {
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset, 32));
    }
    return value;
}

__device__ __forceinline__ float warp_xor_sum(float value) {
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset, 32));
    }
    return value;
}

// fusion.86 launches 60 four-warp CTAs. A CTA owns one query and two adjacent
// heads; each thread consumes eight keys separated by 128. Both the serial
// per-thread association and the warp/cross-warp XOR trees mirror the pinned
// PTX. Probabilities are written as padded [head*query,984] BF16 rows.
__global__ void
openpi_action_attention_softmax_kernel(const float* __restrict__ logits,
                                       const std::uint8_t* __restrict__ attention_mask,
                                       std::uint16_t* __restrict__ probabilities) {
    __shared__ float partials[8];

    const int32_t thread = threadIdx.x;
    const int32_t lane = thread & 31;
    const int32_t warp = thread >> 5;
    const int32_t query = blockIdx.x % kQueryRows;
    const int32_t first_head = (blockIdx.x / kQueryRows) * 2;
    const int32_t logit_row0 = query * kQueryHeads + first_head;
    const int32_t logit_row1 = logit_row0 + 1;
    const int32_t probability_row0 = first_head * kQueryRows + query;
    const int32_t probability_row1 = probability_row0 + kQueryRows;
    constexpr float kNegative = -2.38197633e38F;
    constexpr float kNegativeInfinity = -__builtin_inff();

    float values0[kSoftmaxTiles];
    float values1[kSoftmaxTiles];
    bool valid[kSoftmaxTiles];
#pragma unroll
    for (int32_t tile = 0; tile < kSoftmaxTiles; ++tile) {
        const int32_t key = thread + tile * kSoftmaxThreads;
        valid[tile] = key < kKeyRows;
        if (valid[tile]) {
            const bool keep = attention_mask[query * kKeyRows + key];
            const std::size_t logit_base = static_cast<std::size_t>(key) * kFlattenedQueries;
            values0[tile] = keep ? logits[logit_base + logit_row0] : kNegative;
            values1[tile] = keep ? logits[logit_base + logit_row1] : kNegative;
        } else {
            values0[tile] = kNegativeInfinity;
            values1[tile] = kNegativeInfinity;
        }
    }

    float maximum0 = values0[0];
    float maximum1 = values1[0];
#pragma unroll
    for (int32_t tile = 1; tile < kSoftmaxTiles; ++tile) {
        if (valid[tile]) {
            maximum0 = fmaxf(maximum0, values0[tile]);
            maximum1 = fmaxf(maximum1, values1[tile]);
        }
    }
    maximum0 = warp_xor_max(maximum0);
    maximum1 = warp_xor_max(maximum1);
    if (lane == 0) {
        partials[warp] = maximum0;
        partials[warp + 4] = maximum1;
    }
    __syncthreads();

    float group_maximum = thread < 8 ? partials[thread] : kNegativeInfinity;
    group_maximum = fmaxf(group_maximum, __shfl_xor_sync(0xFFFFFFFFU, group_maximum, 2, 32));
    group_maximum = fmaxf(group_maximum, __shfl_xor_sync(0xFFFFFFFFU, group_maximum, 1, 32));
    if (thread == 0 || thread == 4) {
        partials[thread] = group_maximum;
    }
    __syncthreads();
    maximum0 = partials[0];
    maximum1 = partials[4];
    // Do not let warp leaders reuse partials for the sum reduction until
    // every thread has consumed both broadcast maxima.
    __syncthreads();

    float exponentials0[kSoftmaxTiles];
    float exponentials1[kSoftmaxTiles];
#pragma unroll
    for (int32_t tile = 0; tile < kSoftmaxTiles; ++tile) {
        if (valid[tile]) {
            exponentials0[tile] = openpi_action_expf(__fsub_rn(values0[tile], maximum0));
            exponentials1[tile] = openpi_action_expf(__fsub_rn(values1[tile], maximum1));
        } else {
            exponentials0[tile] = 0.0F;
            exponentials1[tile] = 0.0F;
        }
    }

    float sum0 = exponentials0[0];
    float sum1 = exponentials1[0];
#pragma unroll
    for (int32_t tile = 1; tile < kSoftmaxTiles; ++tile) {
        sum0 = __fadd_rn(sum0, exponentials0[tile]);
        sum1 = __fadd_rn(sum1, exponentials1[tile]);
    }
    sum0 = warp_xor_sum(sum0);
    sum1 = warp_xor_sum(sum1);
    if (lane == 0) {
        partials[warp] = sum0;
        partials[warp + 4] = sum1;
    }
    __syncthreads();

    float group_sum = thread < 8 ? partials[thread] : 0.0F;
    group_sum = __fadd_rn(group_sum, __shfl_xor_sync(0xFFFFFFFFU, group_sum, 2, 32));
    group_sum = __fadd_rn(group_sum, __shfl_xor_sync(0xFFFFFFFFU, group_sum, 1, 32));
    if (thread == 0 || thread == 4) {
        partials[thread] = group_sum;
    }
    __syncthreads();
    sum0 = partials[0];
    sum1 = partials[4];

#pragma unroll
    for (int32_t tile = 0; tile < kSoftmaxTiles; ++tile) {
        const int32_t key = thread + tile * kSoftmaxThreads;
        if (key < kPaddedKeyRows) {
            const std::uint16_t probability0 =
                key < kKeyRows ? float_to_bf16_rn(full_divide(exponentials0[tile], sum0)) : 0;
            const std::uint16_t probability1 =
                key < kKeyRows ? float_to_bf16_rn(full_divide(exponentials1[tile], sum1)) : 0;
            probabilities[static_cast<std::size_t>(probability_row0) * kPaddedKeyRows + key] =
                probability0;
            probabilities[static_cast<std::size_t>(probability_row1) * kPaddedKeyRows + key] =
                probability1;
        }
    }
}

__global__ void
openpi_action_attention_context_to_rows_kernel(const std::uint16_t* __restrict__ raw_context,
                                               std::uint16_t* __restrict__ output) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kPackedQueryElements) {
        return;
    }
    const int32_t flattened = static_cast<int32_t>(index / kHeadDimension);
    const int32_t dimension = static_cast<int32_t>(index % kHeadDimension);
    const int32_t token = flattened / kQueryHeads;
    const int32_t head = flattened % kQueryHeads;
    const int32_t source_column = head * kQueryRows + token;
    output[index] =
        raw_context[static_cast<std::size_t>(dimension) * kFlattenedQueries + source_column];
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool is_bool_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBOOL &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_query_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 4 && dims.d[0] == kBatch && dims.d[1] == kQueryHeads &&
           dims.d[2] == kQueryRows && dims.d[3] == kHeadDimension;
}

bool has_key_value_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 4 && dims.d[0] == kBatch && dims.d[1] == kKeyHeads &&
           dims.d[2] == kKeyRows && dims.d[3] == kHeadDimension;
}

bool has_mask_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 4 && dims.d[0] == kBatch && dims.d[1] == 1 && dims.d[2] == kQueryRows &&
           dims.d[3] == kKeyRows;
}

bool has_output_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == kQueryRows &&
           dims.d[2] == kOutputWidth;
}

bool has_supported_dimensions(nvinfer1::Dims const* inputs, nvinfer1::Dims const& output) {
    return has_query_shape(inputs[0]) && has_key_value_shape(inputs[1]) &&
           has_key_value_shape(inputs[2]) && has_mask_shape(inputs[3]) && has_output_shape(output);
}

bool has_supported_shape(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                         nvinfer1::PluginTensorDesc const* outputs, int32_t nb_outputs) {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 4 || nb_outputs != 1 ||
        !is_bf16_linear(inputs[0]) || !is_bf16_linear(inputs[1]) || !is_bf16_linear(inputs[2]) ||
        !is_bool_linear(inputs[3]) || !is_bf16_linear(outputs[0])) {
        return false;
    }
    nvinfer1::Dims input_dims[4] = {inputs[0].dims, inputs[1].dims, inputs[2].dims, inputs[3].dims};
    return has_supported_dimensions(input_dims, outputs[0].dims);
}

bool has_supported_dynamic_shape(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
                                 nvinfer1::DynamicPluginTensorDesc const* outputs,
                                 int32_t nb_outputs) {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 4 || nb_outputs != 1) {
        return false;
    }
    nvinfer1::PluginTensorDesc input_desc[4] = {inputs[0].desc, inputs[1].desc, inputs[2].desc,
                                                inputs[3].desc};
    const nvinfer1::PluginTensorDesc output_desc[1] = {outputs[0].desc};
    nvinfer1::Dims minimum[4] = {inputs[0].min, inputs[1].min, inputs[2].min, inputs[3].min};
    nvinfer1::Dims maximum[4] = {inputs[0].max, inputs[1].max, inputs[2].max, inputs[3].max};
    return has_supported_shape(input_desc, 4, output_desc, 1) &&
           has_supported_dimensions(minimum, outputs[0].min) &&
           has_supported_dimensions(maximum, outputs[0].max);
}

int32_t configure_cublas(cublasHandle_t handle, cudaStream_t stream, void* workspace,
                         std::size_t workspace_bytes) noexcept {
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    status = cublasSetWorkspace(handle, workspace, workspace_bytes);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

class OpenPIActionAttentionContextPlugin final : public nvinfer1::IPluginV3,
                                                 public nvinfer1::IPluginV3OneCore,
                                                 public nvinfer1::IPluginV3OneBuild,
                                                 public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIActionAttentionContext";
    static constexpr const char* kVersion = "1";

    OpenPIActionAttentionContextPlugin() { serialization_fields_.nbFields = 0; }

    ~OpenPIActionAttentionContextPlugin() override {
        if (cublas_handle_ != nullptr) {
            static_cast<void>(cublasDestroy(cublas_handle_));
            cublas_handle_ = nullptr;
        }
    }

    nvinfer1::IPluginCapability*
    getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        auto* plugin = new (std::nothrow) OpenPIActionAttentionContextPlugin();
        if (plugin != nullptr) {
            plugin->namespace_ = namespace_;
        }
        return plugin;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t nb_outputs) noexcept override {
        return has_supported_dynamic_shape(inputs, nb_inputs, outputs, nb_outputs) ? 0 : 1;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 4 || input_types[0] != nvinfer1::DataType::kBF16 ||
            input_types[1] != nvinfer1::DataType::kBF16 ||
            input_types[2] != nvinfer1::DataType::kBF16 ||
            input_types[3] != nvinfer1::DataType::kBOOL) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder& expression_builder) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 4 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[2];
        outputs[0].d[2] = expression_builder.constant(kOutputWidth);
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 4 || nb_outputs != 1 || position < 0 ||
            position >= 5) {
            return false;
        }
        return position == 3 ? is_bool_linear(descriptors[position].desc)
                             : is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return kPluginWorkspaceBytes;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* outputs,
                          int32_t nb_outputs) noexcept override {
        return has_supported_shape(inputs, nb_inputs, outputs, nb_outputs) ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        if (!has_supported_shape(input_desc, 4, output_desc, 1) || inputs == nullptr ||
            outputs == nullptr || workspace == nullptr || cublas_handle_ == nullptr ||
            outputs[0] == nullptr) {
            return 1;
        }
        for (int32_t index = 0; index < 4; ++index) {
            if (inputs[index] == nullptr) {
                return 1;
            }
        }
        return launch_openpi_action_attention_context(
            static_cast<const std::uint16_t*>(inputs[0]),
            static_cast<const std::uint16_t*>(inputs[1]),
            static_cast<const std::uint16_t*>(inputs[2]),
            static_cast<const std::uint8_t*>(inputs[3]), static_cast<std::uint16_t*>(outputs[0]),
            workspace, static_cast<void*>(cublas_handle_), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        auto* plugin = new (std::nothrow) OpenPIActionAttentionContextPlugin();
        if (plugin == nullptr) {
            return nullptr;
        }
        plugin->namespace_ = namespace_;
        cublasStatus_t status = cublasCreate(&plugin->cublas_handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            delete plugin;
            return nullptr;
        }
        status = cublasSetMathMode(plugin->cublas_handle_, CUBLAS_DEFAULT_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &serialization_fields_;
    }

  private:
    std::string namespace_;
    nvinfer1::PluginFieldCollection serialization_fields_{};
    cublasHandle_t cublas_handle_{nullptr};
};

class OpenPIActionAttentionContextCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIActionAttentionContextCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields != nullptr && fields->nbFields != 0) {
            return nullptr;
        }
        return new (std::nothrow) OpenPIActionAttentionContextPlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIActionAttentionContextPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIActionAttentionContextPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

static nvinfer1::PluginRegistrar<OpenPIActionAttentionContextCreator>
    plugin_registrar_openpi_action_attention_context{};

} // namespace

int32_t launch_openpi_action_attention_context(const std::uint16_t* query, const std::uint16_t* key,
                                               const std::uint16_t* value,
                                               const std::uint8_t* attention_mask,
                                               std::uint16_t* output, void* workspace,
                                               void* cublas_handle, cudaStream_t stream) noexcept {
    if (query == nullptr || key == nullptr || value == nullptr || attention_mask == nullptr ||
        output == nullptr || workspace == nullptr || cublas_handle == nullptr) {
        return 1;
    }

    auto* workspace_bytes = static_cast<std::byte*>(workspace);
    auto* packed_query = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPackedQueryOffset);
    auto* padded_key = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPackedKeyOffset);
    auto* padded_value = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPackedValueOffset);
    auto* logits = reinterpret_cast<float*>(workspace_bytes + kLogitOffset);
    void* cublas_workspace = workspace_bytes + kCublasWorkspaceOffset;

    constexpr std::size_t kPackElements = kPackedKeyElements;
    constexpr int32_t kPackBlocks =
        static_cast<int32_t>((kPackElements + kPointwiseThreads - 1) / kPointwiseThreads);
    openpi_action_attention_pack_kernel<<<kPackBlocks, kPointwiseThreads, 0, stream>>>(
        query, key, value, packed_query, padded_key, padded_value);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return 1;
    }

    auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
    if (configure_cublas(handle, stream, cublas_workspace, kQkCublasWorkspaceBytes) != 0) {
        return 1;
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, kFlattenedQueries, kPaddedKeyRows, kHeadDimension, &alpha,
        packed_query, CUDA_R_16BF, kHeadDimension, padded_key, CUDA_R_16BF, kHeadDimension, &beta,
        logits, CUDA_R_32F, kFlattenedQueries, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    auto* probabilities = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPackedKeyOffset);
    openpi_action_attention_softmax_kernel<<<kSoftmaxBlocks, kSoftmaxThreads, 0, stream>>>(
        logits, attention_mask, probabilities);
    if (cudaPeekAtLastError() != cudaSuccess ||
        configure_cublas(handle, stream, cublas_workspace, kPvCublasWorkspaceBytes) != 0) {
        return 1;
    }

    auto* raw_context = reinterpret_cast<std::uint16_t*>(workspace_bytes + kLogitOffset);
    status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, kFlattenedQueries, kHeadDimension,
                          kPaddedKeyRows, &alpha, probabilities, CUDA_R_16BF, kPaddedKeyRows,
                          padded_value, CUDA_R_16BF, kHeadDimension, &beta, raw_context,
                          CUDA_R_16BF, kFlattenedQueries, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    constexpr int32_t kContextBlocks =
        static_cast<int32_t>((kContextElements + kPointwiseThreads - 1) / kPointwiseThreads);
    openpi_action_attention_context_to_rows_kernel<<<kContextBlocks, kPointwiseThreads, 0,
                                                     stream>>>(raw_context, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::openpi
