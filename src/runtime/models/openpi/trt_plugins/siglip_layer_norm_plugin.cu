/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <new>
#include <string>

namespace trtmc::openpi {
namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kRows = 256;
constexpr int32_t kWidth = 1152;
constexpr int32_t kThreads = 96;
constexpr int32_t kWarps = 3;
constexpr int32_t kSegments = 3;
constexpr int32_t kItemsPerSegment = 4;
constexpr int32_t kSegmentWidth = 384;
constexpr float kEpsilon = 1.0e-6F;

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

__device__ __forceinline__ float xla_warp_reduce_down(float value) {
#pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(__shfl_down_sync(0xFFFFFFFFU, value, offset), value);
    }
    return value;
}

__device__ __forceinline__ float xla_rsqrt(float value) {
    float result;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

// This fixed-shape kernel mirrors the pinned XLA input-reduction and convert
// kernels. Each of 96 lanes owns four adjacent values in each of three
// 384-wide segments. The three warp totals are combined as (warp0 + warp2) +
// warp1, then every association-sensitive FP32 operation uses an explicit RN
// intrinsic before the final BF16 round-to-nearest-even conversion.
__global__ void openpi_siglip_layer_norm_kernel(const std::uint16_t* __restrict__ input,
                                                const std::uint16_t* __restrict__ gamma,
                                                const std::uint16_t* __restrict__ beta,
                                                std::uint16_t* __restrict__ output) {
    __shared__ float sum_partials[kWarps];
    __shared__ float square_partials[kWarps];
    __shared__ float mean;
    __shared__ float reciprocal;

    const int32_t row = blockIdx.x;
    const int32_t thread = threadIdx.x;
    const int32_t lane = thread & 31;
    const int32_t warp = thread >> 5;
    const int64_t row_base = static_cast<int64_t>(row) * kWidth;
    const int32_t lane_base = thread * kItemsPerSegment;
    float values[kSegments][kItemsPerSegment];
    float local_sum = 0.0F;
    float local_square = 0.0F;
    bool first = true;

#pragma unroll
    for (int32_t segment = 0; segment < kSegments; ++segment) {
#pragma unroll
        for (int32_t item = 0; item < kItemsPerSegment; ++item) {
            const int32_t dimension = segment * kSegmentWidth + lane_base + item;
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

    const float warp_sum = xla_warp_reduce_down(local_sum);
    const float warp_square = xla_warp_reduce_down(local_square);
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
        const float clamped_variance = fmaxf(variance, 0.0F);
        const float denominator = __fadd_rn(clamped_variance, __uint_as_float(0x358637BDU));
        mean = row_mean;
        reciprocal = xla_rsqrt(denominator);
    }
    __syncthreads();

#pragma unroll
    for (int32_t segment = 0; segment < kSegments; ++segment) {
#pragma unroll
        for (int32_t item = 0; item < kItemsPerSegment; ++item) {
            const int32_t dimension = segment * kSegmentWidth + lane_base + item;
            const float centered = __fsub_rn(values[segment][item], mean);
            const float scaled_reciprocal = __fmul_rn(reciprocal, bf16_to_float(gamma[dimension]));
            const float normalized = __fmul_rn(centered, scaled_reciprocal);
            const float shifted = __fadd_rn(normalized, bf16_to_float(beta[dimension]));
            output[row_base + dimension] = float_to_bf16_rn(shifted);
        }
    }
}

int32_t launch_openpi_siglip_layer_norm(const std::uint16_t* input, const std::uint16_t* gamma,
                                        const std::uint16_t* beta, std::uint16_t* output,
                                        cudaStream_t stream) noexcept {
    openpi_siglip_layer_norm_kernel<<<kRows, kThreads, 0, stream>>>(input, gamma, beta, output);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_activation_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == kRows && dims.d[2] == kWidth;
}

bool has_parameter_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 1 && dims.d[0] == kWidth;
}

bool has_supported_shape(nvinfer1::PluginTensorDesc const& activation,
                         nvinfer1::PluginTensorDesc const& gamma,
                         nvinfer1::PluginTensorDesc const& beta,
                         nvinfer1::PluginTensorDesc const& output) {
    return is_bf16_linear(activation) && is_bf16_linear(gamma) && is_bf16_linear(beta) &&
           is_bf16_linear(output) && has_activation_shape(activation.dims) &&
           has_parameter_shape(gamma.dims) && has_parameter_shape(beta.dims) &&
           has_activation_shape(output.dims);
}

bool has_supported_shape(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                         nvinfer1::PluginTensorDesc const* output, int32_t nb_outputs) {
    return input != nullptr && output != nullptr && nb_inputs == 3 && nb_outputs == 1 &&
           has_supported_shape(input[0], input[1], input[2], output[0]);
}

bool has_supported_dynamic_shape(nvinfer1::DynamicPluginTensorDesc const* input, int32_t nb_inputs,
                                 nvinfer1::DynamicPluginTensorDesc const* output,
                                 int32_t nb_outputs) {
    if (input == nullptr || output == nullptr || nb_inputs != 3 || nb_outputs != 1) {
        return false;
    }
    return has_supported_shape(input[0].desc, input[1].desc, input[2].desc, output[0].desc) &&
           has_activation_shape(input[0].min) && has_parameter_shape(input[1].min) &&
           has_parameter_shape(input[2].min) && has_activation_shape(output[0].min) &&
           has_activation_shape(input[0].max) && has_parameter_shape(input[1].max) &&
           has_parameter_shape(input[2].max) && has_activation_shape(output[0].max);
}

class OpenPISiglipLayerNormPlugin final : public nvinfer1::IPluginV3,
                                          public nvinfer1::IPluginV3OneCore,
                                          public nvinfer1::IPluginV3OneBuild,
                                          public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPISiglipLayerNorm";
    static constexpr const char* kVersion = "1";

    explicit OpenPISiglipLayerNormPlugin(float epsilon = kEpsilon) : epsilon_(epsilon) {
        update_serialization_fields();
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
        auto* plugin = new (std::nothrow) OpenPISiglipLayerNormPlugin(epsilon_);
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
        if (epsilon_ != kEpsilon ||
            !has_supported_dynamic_shape(input, nb_inputs, output, nb_outputs)) {
            return 1;
        }
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t nb_outputs,
                               nvinfer1::DataType const* input_types,
                               int32_t nb_inputs) const noexcept override {
        if (output_types == nullptr || input_types == nullptr || nb_outputs != 1 ||
            nb_inputs != 3 || input_types[0] != nvinfer1::DataType::kBF16 ||
            input_types[1] != nvinfer1::DataType::kBF16 ||
            input_types[2] != nvinfer1::DataType::kBF16) {
            return 1;
        }
        output_types[0] = nvinfer1::DataType::kBF16;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nb_inputs,
                            nvinfer1::DimsExprs const*, int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs, int32_t nb_outputs,
                            nvinfer1::IExprBuilder&) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 3 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 3 || nb_outputs != 1 || position < 0 ||
            position >= 4) {
            return false;
        }
        return is_bf16_linear(descriptors[position].desc);
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                                 nvinfer1::DynamicPluginTensorDesc const*,
                                 int32_t) const noexcept override {
        return 0;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* output,
                          int32_t nb_outputs) noexcept override {
        if (epsilon_ != kEpsilon || !has_supported_shape(input, nb_inputs, output, nb_outputs)) {
            return 1;
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc, void const* const* inputs,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (epsilon_ != kEpsilon || !has_supported_shape(input_desc, 3, output_desc, 1) ||
            inputs == nullptr || outputs == nullptr || inputs[0] == nullptr ||
            inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr) {
            return 1;
        }
        return launch_openpi_siglip_layer_norm(static_cast<const std::uint16_t*>(inputs[0]),
                                               static_cast<const std::uint16_t*>(inputs[1]),
                                               static_cast<const std::uint16_t*>(inputs[2]),
                                               static_cast<std::uint16_t*>(outputs[0]), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        update_serialization_fields();
        return &serialization_fields_;
    }

  private:
    void update_serialization_fields() noexcept {
        serialization_field_ = {"epsilon", &epsilon_, nvinfer1::PluginFieldType::kFLOAT32, 1};
        serialization_fields_.nbFields = 1;
        serialization_fields_.fields = &serialization_field_;
    }

    float epsilon_{kEpsilon};
    std::string namespace_;
    nvinfer1::PluginField serialization_field_{};
    nvinfer1::PluginFieldCollection serialization_fields_{};
};

class OpenPISiglipLayerNormCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPISiglipLayerNormCreator() {
        field_ = {"epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1};
        fields_.nbFields = 1;
        fields_.fields = &field_;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields == nullptr || fields->fields == nullptr || fields->nbFields != 1) {
            return nullptr;
        }
        const auto& field = fields->fields[0];
        if (field.name == nullptr || std::string(field.name) != "epsilon" ||
            field.type != nvinfer1::PluginFieldType::kFLOAT32 || field.data == nullptr ||
            field.length != 1) {
            return nullptr;
        }
        const float epsilon = *static_cast<const float*>(field.data);
        if (epsilon != kEpsilon) {
            return nullptr;
        }
        return new (std::nothrow) OpenPISiglipLayerNormPlugin(epsilon);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPISiglipLayerNormPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPISiglipLayerNormPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginField field_{};
    nvinfer1::PluginFieldCollection fields_{};
};

static nvinfer1::PluginRegistrar<OpenPISiglipLayerNormCreator>
    plugin_registrar_openpi_siglip_layer_norm{};

} // namespace
} // namespace trtmc::openpi
