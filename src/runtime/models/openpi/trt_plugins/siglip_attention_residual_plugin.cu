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

int32_t launch_openpi_siglip_attention_residual(
    const std::uint16_t* hidden, const std::uint16_t* norm_gamma, const std::uint16_t* norm_beta,
    const std::uint16_t* qkv_weight, const std::uint16_t* qkv_bias,
    const std::uint16_t* output_weight, const std::uint16_t* output_bias, std::uint16_t* output,
    void* workspace, void* cublas_handle, cudaStream_t stream) noexcept;

namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kRows = 256;
constexpr int32_t kWidth = 1152;
constexpr int32_t kHeads = 16;
constexpr int32_t kHeadDimension = 72;
constexpr int32_t kQkvProjections = 3;
constexpr int32_t kQkvWidth = kQkvProjections * kWidth;

constexpr std::size_t kActivationElements = static_cast<std::size_t>(kRows) * kWidth;
constexpr std::size_t kActivationBytes = kActivationElements * sizeof(std::uint16_t);
constexpr std::size_t kQkvElements = static_cast<std::size_t>(kRows) * kQkvWidth;
constexpr std::size_t kQkvBytes = kQkvElements * sizeof(std::uint16_t);
constexpr std::size_t kAttentionElements = static_cast<std::size_t>(kHeads) * kRows * kRows;
constexpr std::size_t kAttentionBytes = kAttentionElements * sizeof(std::uint16_t);

// Exact scratch sizes attached to the corresponding __cublas$gemm thunks in
// the pinned JAX 0.5.3 / XLA executable.
constexpr std::size_t kQkvCublasWorkspaceBytes = 8552448;
constexpr std::size_t kQkCublasWorkspaceBytes = 1179648;
constexpr std::size_t kPvCublasWorkspaceBytes = 2686976;
constexpr std::size_t kOutputCublasWorkspaceBytes = 3244032;

// Keep the two QKV representations disjoint while the exact XLA bias,
// transpose, and query-scale boundary is materialized. Once packed Q/K/V are
// ready, the norm and raw-QKV regions are reused for logits and context.
constexpr std::size_t kNormOffset = 0;
constexpr std::size_t kRawQkvOffset = kNormOffset + kActivationBytes;
constexpr std::size_t kPackedQkvOffset = kRawQkvOffset + kQkvBytes;
constexpr std::size_t kCublasWorkspaceOffset = kPackedQkvOffset + kQkvBytes;
constexpr std::size_t kPluginWorkspaceBytes = kCublasWorkspaceOffset + kQkvCublasWorkspaceBytes;
static_assert(kAttentionBytes <= kRawQkvOffset + kQkvBytes,
              "reused logits region must fit before packed QKV");
static_assert(kQkCublasWorkspaceBytes <= kQkvCublasWorkspaceBytes);
static_assert(kPvCublasWorkspaceBytes + kActivationBytes <= kQkvCublasWorkspaceBytes);
static_assert(kOutputCublasWorkspaceBytes <= kQkvCublasWorkspaceBytes);

constexpr int32_t kLayerNormThreads = 96;
constexpr int32_t kLayerNormWarps = 3;
constexpr int32_t kLayerNormSegments = 3;
constexpr int32_t kLayerNormItemsPerSegment = 4;
constexpr int32_t kLayerNormSegmentWidth = 384;
constexpr int32_t kSoftmaxThreads = 128;
constexpr int32_t kSoftmaxRowsPerBlock = 8;
constexpr int32_t kSoftmaxBlocks = kHeads * kRows / kSoftmaxRowsPerBlock;

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

__device__ __forceinline__ std::uint16_t bf16_add_rn(std::uint16_t lhs, std::uint16_t rhs) {
    return float_to_bf16_rn(__fadd_rn(bf16_to_float(lhs), bf16_to_float(rhs)));
}

__device__ __forceinline__ std::uint16_t bf16_multiply_rn(std::uint16_t lhs, std::uint16_t rhs) {
    return float_to_bf16_rn(__fmul_rn(bf16_to_float(lhs), bf16_to_float(rhs)));
}

__device__ __forceinline__ float approximate_rsqrt(float value) {
    float result;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float approximate_exp2(float value) {
    float result;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float full_divide(float numerator, float denominator) {
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
    return result;
}

__device__ __forceinline__ float warp_reduce_down_sum(float value) {
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_xor_sum(float value) {
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_xor_max(float value) {
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

// Exact copy of the pinned SigLIP LayerNorm lowering. Every block owns one
// row; 96 lanes consume four adjacent elements from each of three 384-wide
// segments. The three warp totals combine as (warp0 + warp2) + warp1.
__global__ void openpi_siglip_attention_layer_norm_kernel(const std::uint16_t* __restrict__ input,
                                                          const std::uint16_t* __restrict__ gamma,
                                                          const std::uint16_t* __restrict__ beta,
                                                          std::uint16_t* __restrict__ output) {
    __shared__ float sum_partials[kLayerNormWarps];
    __shared__ float square_partials[kLayerNormWarps];
    __shared__ float mean;
    __shared__ float reciprocal;

    const int32_t row = blockIdx.x;
    const int32_t thread = threadIdx.x;
    const int32_t lane = thread & 31;
    const int32_t warp = thread >> 5;
    const int64_t row_base = static_cast<int64_t>(row) * kWidth;
    const int32_t lane_base = thread * kLayerNormItemsPerSegment;
    float values[kLayerNormSegments][kLayerNormItemsPerSegment];
    float local_sum = 0.0F;
    float local_square = 0.0F;
    bool first = true;

#pragma unroll
    for (int32_t segment = 0; segment < kLayerNormSegments; ++segment) {
#pragma unroll
        for (int32_t item = 0; item < kLayerNormItemsPerSegment; ++item) {
            const int32_t dimension = segment * kLayerNormSegmentWidth + lane_base + item;
            const float value = bf16_to_float(input[row_base + dimension]);
            values[segment][item] = value;
            const float square = __fmul_rn(value, value);
            if (first) {
                local_sum = value;
                local_square = square;
                first = false;
            } else {
                local_sum = __fadd_rn(local_sum, value);
                local_square = __fadd_rn(local_square, square);
            }
        }
    }

    const float warp_sum = warp_reduce_down_sum(local_sum);
    const float warp_square = warp_reduce_down_sum(local_square);
    if (lane == 0) {
        sum_partials[warp] = warp_sum;
        square_partials[warp] = warp_square;
    }
    __syncthreads();

    if (thread == 0) {
        const float sum = __fadd_rn(__fadd_rn(sum_partials[0], sum_partials[2]), sum_partials[1]);
        const float square_sum =
            __fadd_rn(__fadd_rn(square_partials[0], square_partials[2]), square_partials[1]);
        const float reciprocal_width = __uint_as_float(0x3A638E39U);
        const float row_mean = __fmul_rn(sum, reciprocal_width);
        const float row_mean_square = __fmul_rn(square_sum, reciprocal_width);
        const float mean_squared = __fmul_rn(row_mean, row_mean);
        const float variance = __fsub_rn(row_mean_square, mean_squared);
        const float denominator = __fadd_rn(fmaxf(variance, 0.0F), __uint_as_float(0x358637BDU));
        mean = row_mean;
        reciprocal = approximate_rsqrt(denominator);
    }
    __syncthreads();

#pragma unroll
    for (int32_t segment = 0; segment < kLayerNormSegments; ++segment) {
#pragma unroll
        for (int32_t item = 0; item < kLayerNormItemsPerSegment; ++item) {
            const int32_t dimension = segment * kLayerNormSegmentWidth + lane_base + item;
            const float centered = __fsub_rn(values[segment][item], mean);
            const float scaled_reciprocal = __fmul_rn(reciprocal, bf16_to_float(gamma[dimension]));
            const float normalized = __fmul_rn(centered, scaled_reciprocal);
            const float shifted = __fadd_rn(normalized, bf16_to_float(beta[dimension]));
            output[row_base + dimension] = float_to_bf16_rn(shifted);
        }
    }
}

// QKV is emitted row-major by cuBLAS. This kernel applies the separately
// rounded BF16 bias, applies XLA's BF16 reciprocal-sqrt scale to Q, and packs
// every projection as contiguous [head, query, dimension] storage for the
// exact batched QK/PV cuBLAS calls.
__global__ void openpi_siglip_pack_qkv_kernel(const std::uint16_t* __restrict__ raw_qkv,
                                              const std::uint16_t* __restrict__ bias,
                                              std::uint16_t* __restrict__ packed_qkv) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kQkvElements) {
        return;
    }
    const std::size_t projection = index / kActivationElements;
    const std::size_t projection_index = index % kActivationElements;
    const std::size_t row = projection_index / kWidth;
    const std::size_t feature = projection_index % kWidth;
    const std::size_t head = feature / kHeadDimension;
    const std::size_t dimension = feature % kHeadDimension;
    const std::size_t source = row * kQkvWidth + projection * kWidth + feature;
    const std::size_t destination =
        projection * kActivationElements + (head * kRows + row) * kHeadDimension + dimension;
    std::uint16_t value = bf16_add_rn(raw_qkv[source], bias[projection * kWidth + feature]);
    if (projection == 0) {
        // BF16 1/sqrt(72), exactly as constant_90_2 in the pinned HLO.
        value = bf16_multiply_rn(value, 0x3DF1U);
    }
    packed_qkv[destination] = value;
}

// fusion.257: one 128-thread CTA owns eight rows of the [16,256,256]
// logits tensor. Every warp owns two rows and every lane owns eight adjacent
// columns. Max and sum reductions use XOR butterflies, matching the Triton
// lowering exactly, including BF16 max/subtract/exp boundaries.
// This kernel intentionally supports logits == probabilities; do not mark
// either pointer restrict, because the production path normalizes in place.
__global__ void openpi_siglip_softmax_kernel(const std::uint16_t* logits,
                                             std::uint16_t* probabilities) {
    const int32_t warp = threadIdx.x >> 5;
    const int32_t lane = threadIdx.x & 31;
    const int32_t first_row = blockIdx.x * kSoftmaxRowsPerBlock + warp;

#pragma unroll
    for (int32_t row_group = 0; row_group < 2; ++row_group) {
        const int32_t row = first_row + row_group * 4;
        const int64_t row_base = static_cast<int64_t>(row) * kRows;
        const int32_t column_base = lane * 8;
        std::uint16_t input_values[8];
        float local_max;

#pragma unroll
        for (int32_t item = 0; item < 8; ++item) {
            input_values[item] = logits[row_base + column_base + item];
            const float value = bf16_to_float(input_values[item]);
            local_max = item == 0 ? value : fmaxf(local_max, value);
        }
        const std::uint16_t maximum = float_to_bf16_rn(warp_reduce_xor_max(local_max));

        std::uint16_t exponential_bf16[8];
        float local_sum = 0.0F;
#pragma unroll
        for (int32_t item = 0; item < 8; ++item) {
            const std::uint16_t difference = float_to_bf16_rn(
                __fsub_rn(bf16_to_float(input_values[item]), bf16_to_float(maximum)));
            const float exponent =
                __fmul_rn(bf16_to_float(difference), __uint_as_float(0x3FB8AA3BU));
            exponential_bf16[item] = float_to_bf16_rn(approximate_exp2(exponent));
            const float value = bf16_to_float(exponential_bf16[item]);
            local_sum = item == 0 ? value : __fadd_rn(local_sum, value);
        }
        const float denominator = warp_reduce_xor_sum(local_sum);

#pragma unroll
        for (int32_t item = 0; item < 8; ++item) {
            const float probability =
                full_divide(bf16_to_float(exponential_bf16[item]), denominator);
            probabilities[row_base + column_base + item] = float_to_bf16_rn(probability);
        }
    }
}

__global__ void openpi_siglip_context_to_rows_kernel(const std::uint16_t* __restrict__ head_major,
                                                     std::uint16_t* __restrict__ row_major) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kActivationElements) {
        return;
    }
    const std::size_t row = index / kWidth;
    const std::size_t feature = index % kWidth;
    const std::size_t head = feature / kHeadDimension;
    const std::size_t dimension = feature % kHeadDimension;
    row_major[index] = head_major[(head * kRows + row) * kHeadDimension + dimension];
}

__global__ void openpi_siglip_output_residual_kernel(const std::uint16_t* __restrict__ hidden,
                                                     const std::uint16_t* __restrict__ attended,
                                                     const std::uint16_t* __restrict__ bias,
                                                     std::uint16_t* __restrict__ output) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kActivationElements) {
        return;
    }
    const std::uint16_t shifted = bf16_add_rn(attended[index], bias[index % kWidth]);
    output[index] = bf16_add_rn(hidden[index], shifted);
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_activation_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == kRows && dims.d[2] == kWidth;
}

bool has_vector_shape(nvinfer1::Dims const& dims, int32_t width) {
    return dims.nbDims == 1 && dims.d[0] == width;
}

bool has_matrix_shape(nvinfer1::Dims const& dims, int32_t rows, int32_t columns) {
    return dims.nbDims == 2 && dims.d[0] == rows && dims.d[1] == columns;
}

bool has_supported_dimensions(nvinfer1::Dims const* input, nvinfer1::Dims const& output) {
    return has_activation_shape(input[0]) && has_vector_shape(input[1], kWidth) &&
           has_vector_shape(input[2], kWidth) && has_matrix_shape(input[3], kWidth, kQkvWidth) &&
           has_vector_shape(input[4], kQkvWidth) && has_matrix_shape(input[5], kWidth, kWidth) &&
           has_vector_shape(input[6], kWidth) && has_activation_shape(output);
}

bool has_supported_shape(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                         nvinfer1::PluginTensorDesc const* output, int32_t nb_outputs) {
    if (input == nullptr || output == nullptr || nb_inputs != 7 || nb_outputs != 1) {
        return false;
    }
    nvinfer1::Dims input_dims[7];
    for (int32_t index = 0; index < 7; ++index) {
        if (!is_bf16_linear(input[index])) {
            return false;
        }
        input_dims[index] = input[index].dims;
    }
    return is_bf16_linear(output[0]) && has_supported_dimensions(input_dims, output[0].dims);
}

bool has_supported_dynamic_shape(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                                 nvinfer1::DynamicPluginTensorDesc const* output,
                                 int32_t nb_outputs) {
    if (input == nullptr || output == nullptr || nb_inputs != 7 || nb_outputs != 1) {
        return false;
    }
    nvinfer1::PluginTensorDesc input_desc[7];
    nvinfer1::Dims minimum[7];
    nvinfer1::Dims maximum[7];
    for (int32_t index = 0; index < 7; ++index) {
        input_desc[index] = input[index].desc;
        minimum[index] = input[index].min;
        maximum[index] = input[index].max;
    }
    const nvinfer1::PluginTensorDesc output_desc[1] = {output[0].desc};
    return has_supported_shape(input_desc, 7, output_desc, 1) &&
           has_supported_dimensions(minimum, output[0].min) &&
           has_supported_dimensions(maximum, output[0].max);
}

class OpenPISiglipAttentionResidualPlugin final : public nvinfer1::IPluginV3,
                                                  public nvinfer1::IPluginV3OneCore,
                                                  public nvinfer1::IPluginV3OneBuild,
                                                  public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPISiglipAttentionResidual";
    static constexpr const char* kVersion = "1";

    OpenPISiglipAttentionResidualPlugin() { serialization_fields_.nbFields = 0; }

    ~OpenPISiglipAttentionResidualPlugin() override {
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
        auto* plugin = new (std::nothrow) OpenPISiglipAttentionResidualPlugin();
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

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* output,
                            int32_t nb_outputs) noexcept override {
        return has_supported_dynamic_shape(input, nb_inputs, output, nb_outputs) ? 0 : 1;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 7) {
            return 1;
        }
        for (int32_t index = 0; index < 7; ++index) {
            if (input_types[index] != nvinfer1::DataType::kBF16) {
                return 1;
            }
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 7 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 7 || nb_outputs != 1 || position < 0 ||
            position >= 8) {
            return false;
        }
        return is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return kPluginWorkspaceBytes;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        return has_supported_shape(input, nb_inputs, output, nb_outputs) ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        if (!has_supported_shape(input_desc, 7, output_desc, 1) || inputs == nullptr ||
            outputs == nullptr || workspace == nullptr || cublas_handle_ == nullptr ||
            outputs[0] == nullptr) {
            return 1;
        }
        for (int32_t index = 0; index < 7; ++index) {
            if (inputs[index] == nullptr) {
                return 1;
            }
        }
        return launch_openpi_siglip_attention_residual(
            static_cast<const std::uint16_t*>(inputs[0]),
            static_cast<const std::uint16_t*>(inputs[1]),
            static_cast<const std::uint16_t*>(inputs[2]),
            static_cast<const std::uint16_t*>(inputs[3]),
            static_cast<const std::uint16_t*>(inputs[4]),
            static_cast<const std::uint16_t*>(inputs[5]),
            static_cast<const std::uint16_t*>(inputs[6]), static_cast<std::uint16_t*>(outputs[0]),
            workspace, static_cast<void*>(cublas_handle_), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        auto* plugin = new (std::nothrow) OpenPISiglipAttentionResidualPlugin();
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

class OpenPISiglipAttentionResidualCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPISiglipAttentionResidualCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields != nullptr && fields->nbFields != 0) {
            return nullptr;
        }
        return new (std::nothrow) OpenPISiglipAttentionResidualPlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPISiglipAttentionResidualPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPISiglipAttentionResidualPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

static nvinfer1::PluginRegistrar<OpenPISiglipAttentionResidualCreator>
    plugin_registrar_openpi_siglip_attention_residual{};

int32_t configure_cublas(cublasHandle_t handle, cudaStream_t stream, void* workspace,
                         std::size_t workspace_bytes) noexcept {
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    status = cublasSetWorkspace(handle, workspace, workspace_bytes);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

} // namespace

int32_t launch_openpi_siglip_attention_residual(
    const std::uint16_t* hidden, const std::uint16_t* norm_gamma, const std::uint16_t* norm_beta,
    const std::uint16_t* qkv_weight, const std::uint16_t* qkv_bias,
    const std::uint16_t* output_weight, const std::uint16_t* output_bias, std::uint16_t* output,
    void* workspace, void* cublas_handle, cudaStream_t stream) noexcept {
    if (hidden == nullptr || norm_gamma == nullptr || norm_beta == nullptr ||
        qkv_weight == nullptr || qkv_bias == nullptr || output_weight == nullptr ||
        output_bias == nullptr || output == nullptr || workspace == nullptr ||
        cublas_handle == nullptr) {
        return 1;
    }

    auto* workspace_bytes = static_cast<std::byte*>(workspace);
    auto* normalized = reinterpret_cast<std::uint16_t*>(workspace_bytes + kNormOffset);
    auto* raw_qkv = reinterpret_cast<std::uint16_t*>(workspace_bytes + kRawQkvOffset);
    auto* packed_qkv = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPackedQkvOffset);
    void* cublas_workspace = workspace_bytes + kCublasWorkspaceOffset;
    auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);

    openpi_siglip_attention_layer_norm_kernel<<<kRows, kLayerNormThreads, 0, stream>>>(
        hidden, norm_gamma, norm_beta, normalized);
    if (cudaPeekAtLastError() != cudaSuccess ||
        configure_cublas(handle, stream, cublas_workspace, kQkvCublasWorkspaceBytes) != 0) {
        return 1;
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublasStatus_t status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, kQkvWidth, kRows, kWidth, &alpha, qkv_weight,
                     CUDA_R_16BF, kQkvWidth, normalized, CUDA_R_16BF, kWidth, &beta, raw_qkv,
                     CUDA_R_16BF, kQkvWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    constexpr int32_t kPointwiseThreads = 256;
    constexpr int32_t kQkvBlocks =
        static_cast<int32_t>((kQkvElements + kPointwiseThreads - 1) / kPointwiseThreads);
    openpi_siglip_pack_qkv_kernel<<<kQkvBlocks, kPointwiseThreads, 0, stream>>>(raw_qkv, qkv_bias,
                                                                                packed_qkv);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return 1;
    }

    auto* query = packed_qkv;
    auto* key = query + kActivationElements;
    auto* value = key + kActivationElements;
    // The norm + raw-QKV prefix is large enough for [16,256,256] logits and
    // is dead before this contraction starts.
    auto* logits = reinterpret_cast<std::uint16_t*>(workspace_bytes + kNormOffset);
    if (configure_cublas(handle, stream, cublas_workspace, kQkCublasWorkspaceBytes) != 0) {
        return 1;
    }
    constexpr long long kHeadOperandStride = static_cast<long long>(kRows) * kHeadDimension;
    constexpr long long kAttentionStride = static_cast<long long>(kRows) * kRows;
    status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, kRows, kRows, kHeadDimension, &alpha, key, CUDA_R_16BF,
        kHeadDimension, kHeadOperandStride, query, CUDA_R_16BF, kHeadDimension, kHeadOperandStride,
        &beta, logits, CUDA_R_16BF, kRows, kAttentionStride, kHeads, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    openpi_siglip_softmax_kernel<<<kSoftmaxBlocks, kSoftmaxThreads, 0, stream>>>(logits, logits);
    if (cudaPeekAtLastError() != cudaSuccess ||
        configure_cublas(handle, stream, cublas_workspace, kPvCublasWorkspaceBytes) != 0) {
        return 1;
    }

    // PV writes contiguous [head,query,dimension] immediately after its exact
    // cuBLAS scratch allocation. This keeps the output disjoint from both the
    // in-place probability tensor and packed V.
    auto* context_head_major = reinterpret_cast<std::uint16_t*>(
        static_cast<std::byte*>(cublas_workspace) + kPvCublasWorkspaceBytes);
    status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDimension, kRows, kRows, &alpha, value, CUDA_R_16BF,
        kHeadDimension, kHeadOperandStride, logits, CUDA_R_16BF, kRows, kAttentionStride, &beta,
        context_head_major, CUDA_R_16BF, kHeadDimension, kHeadOperandStride, kHeads,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    auto* context_rows = reinterpret_cast<std::uint16_t*>(workspace_bytes + kNormOffset);
    constexpr int32_t kActivationBlocks =
        static_cast<int32_t>((kActivationElements + kPointwiseThreads - 1) / kPointwiseThreads);
    openpi_siglip_context_to_rows_kernel<<<kActivationBlocks, kPointwiseThreads, 0, stream>>>(
        context_head_major, context_rows);
    if (cudaPeekAtLastError() != cudaSuccess ||
        configure_cublas(handle, stream, cublas_workspace, kOutputCublasWorkspaceBytes) != 0) {
        return 1;
    }

    // Reuse the dead raw-QKV region for the row-major output.
    auto* attended = reinterpret_cast<std::uint16_t*>(workspace_bytes + kRawQkvOffset);
    status =
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, kWidth, kRows, kWidth, &alpha, output_weight,
                     CUDA_R_16BF, kWidth, context_rows, CUDA_R_16BF, kWidth, &beta, attended,
                     CUDA_R_16BF, kWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    openpi_siglip_output_residual_kernel<<<kActivationBlocks, kPointwiseThreads, 0, stream>>>(
        hidden, attended, output_bias, output);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::openpi
