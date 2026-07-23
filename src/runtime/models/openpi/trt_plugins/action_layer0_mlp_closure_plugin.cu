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

int32_t launch_openpi_action_layer0_mlp_closure(
    const std::uint16_t* post_attention, const std::uint16_t* normed_ffw,
    const std::uint16_t* ffw_gate, const std::uint16_t* gate_weight, const std::uint16_t* up_weight,
    const std::uint16_t* down_weight, std::uint16_t* post_mlp, void* workspace, void* cublas_handle,
    cudaStream_t stream) noexcept;

namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kRows = 15;
constexpr int32_t kPaddedRows = 16;
constexpr int32_t kWidth = 1024;
constexpr int32_t kMlpWidth = 4096;
constexpr int32_t kCombinedWidth = 2 * kMlpWidth;

constexpr std::size_t kPaddedActivationElements = static_cast<std::size_t>(kPaddedRows) * kWidth;
constexpr std::size_t kPaddedActivationBytes = kPaddedActivationElements * sizeof(std::uint16_t);
constexpr std::size_t kCombinedWeightElements = static_cast<std::size_t>(kWidth) * kCombinedWidth;
constexpr std::size_t kCombinedWeightBytes = kCombinedWeightElements * sizeof(std::uint16_t);
constexpr std::size_t kCombinedOutputElements =
    static_cast<std::size_t>(kPaddedRows) * kCombinedWidth;
constexpr std::size_t kCombinedOutputBytes = kCombinedOutputElements * sizeof(std::uint16_t);
constexpr std::size_t kFusedOutputElements = static_cast<std::size_t>(kPaddedRows) * kMlpWidth;
constexpr std::size_t kFusedOutputBytes = kFusedOutputElements * sizeof(std::uint16_t);

// Exact scratch sizes attached to custom-call.68 and custom-call.69 in the
// pinned JAX 0.5.3 / XLA action executable. They are deliberately not rounded
// to a generic workspace bucket.
constexpr std::size_t kCombinedCublasWorkspaceBytes = 16809984;
constexpr std::size_t kDownCublasWorkspaceBytes = 8519680;

// The packed gate/up matrix is dead after the first GEMM, so its prefix can be
// reused by the fused GELU product and down-projection output. The large cuBLAS
// scratch region remains disjoint for the lifetime of enqueue.
constexpr std::size_t kCombinedWeightOffset = 0;
constexpr std::size_t kPaddedNormedOffset = kCombinedWeightOffset + kCombinedWeightBytes;
constexpr std::size_t kCombinedOutputOffset = kPaddedNormedOffset + kPaddedActivationBytes;
constexpr std::size_t kCublasWorkspaceOffset = kCombinedOutputOffset + kCombinedOutputBytes;
constexpr std::size_t kPluginWorkspaceBytes =
    kCublasWorkspaceOffset + kCombinedCublasWorkspaceBytes;
constexpr std::size_t kFusedOutputOffset = kCombinedWeightOffset;
constexpr std::size_t kDownOutputOffset = kFusedOutputOffset + kFusedOutputBytes;
static_assert(kDownCublasWorkspaceBytes <= kCombinedCublasWorkspaceBytes);
static_assert(kDownOutputOffset + kPaddedActivationBytes <= kCombinedWeightBytes);

constexpr int32_t kPointwiseThreads = 128;
constexpr int32_t kGeluBlocks = kPaddedRows * kMlpWidth / kPointwiseThreads;
constexpr int32_t kResidualBlocks = kRows * kWidth / kPointwiseThreads;

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

__device__ __forceinline__ float full_divide(float numerator, float denominator) {
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
    return result;
}

// Exact libdevice tanh lowering emitted inside XLA's loop_pad_fusion. The
// clamp, tiny-input branch, coefficient bit patterns, FMA association, and
// full-precision divide are copied from the qualified PTX rather than using a
// toolkit-version-dependent tanhf implementation.
__device__ __forceinline__ float xla_tanh(float value) {
    const float upper = __uint_as_float(0x40FFF644U);
    const float lower = __uint_as_float(0xC0FFF644U);
    const float upper_limited = value >= upper ? upper : value;
    const float clamped = upper_limited <= lower ? lower : upper_limited;
    const bool tiny = fabsf(value) < __uint_as_float(0x39D1B717U);
    const float square = __fmul_rn(clamped, clamped);

    float numerator_polynomial =
        __fmaf_rn(square, __uint_as_float(0xA59F25C0U), __uint_as_float(0x2A61337EU));
    numerator_polynomial = __fmaf_rn(square, numerator_polynomial, __uint_as_float(0xAEBD37FFU));
    numerator_polynomial = __fmaf_rn(square, numerator_polynomial, __uint_as_float(0x335C0041U));
    numerator_polynomial = __fmaf_rn(square, numerator_polynomial, __uint_as_float(0x3779434AU));
    numerator_polynomial = __fmaf_rn(square, numerator_polynomial, __uint_as_float(0x3A270DEDU));
    numerator_polynomial = __fmaf_rn(square, numerator_polynomial, __uint_as_float(0x3BA059DCU));
    const float numerator = __fmul_rn(clamped, numerator_polynomial);

    float denominator =
        __fmaf_rn(square, __uint_as_float(0x35A0D3D8U), __uint_as_float(0x38F895D6U));
    denominator = __fmaf_rn(square, denominator, __uint_as_float(0x3B14AA05U));
    denominator = __fmaf_rn(square, denominator, __uint_as_float(0x3BA059DDU));
    const float ratio = full_divide(numerator, denominator);
    return tiny ? clamped : ratio;
}

// XLA launches one 128-thread CTA per 128-feature tile over the padded
// [16,4096] result. Row 15 is the exact zero pad consumed by the down GEMM.
__global__ void openpi_action_layer0_gelu_gated_kernel(const std::uint16_t* __restrict__ combined,
                                                       std::uint16_t* __restrict__ fused) {
    const int32_t row = static_cast<int32_t>(blockIdx.x) >> 5;
    const int32_t feature =
        ((static_cast<int32_t>(blockIdx.x) << 7) & 3968) | static_cast<int32_t>(threadIdx.x);
    const int32_t output_index = row * kMlpWidth + feature;
    if (row == kRows) {
        fused[output_index] = 0;
        return;
    }

    const int32_t combined_index = row * kCombinedWidth + feature;
    const std::uint16_t gate = combined[combined_index];
    const std::uint16_t squared = bf16_multiply_rn(gate, gate);
    const std::uint16_t cubed = bf16_multiply_rn(gate, squared);
    const std::uint16_t cubic = bf16_multiply_rn(cubed, 0x3D37U);
    const std::uint16_t inner = bf16_add_rn(gate, cubic);
    const std::uint16_t scaled = bf16_multiply_rn(inner, 0x3F4CU);
    const std::uint16_t tanh = float_to_bf16_rn(xla_tanh(bf16_to_float(scaled)));
    const std::uint16_t shifted = bf16_add_rn(tanh, 0x3F80U);
    const std::uint16_t halved = bf16_multiply_rn(shifted, 0x3F00U);
    const std::uint16_t activated = bf16_multiply_rn(gate, halved);
    const std::uint16_t up = combined[combined_index + kMlpWidth];
    fused[output_index] = bf16_multiply_rn(activated, up);
}

__global__ void openpi_action_layer0_mlp_residual_kernel(
    const std::uint16_t* __restrict__ post_attention, const std::uint16_t* __restrict__ down,
    const std::uint16_t* __restrict__ ffw_gate, std::uint16_t* __restrict__ post_mlp) {
    const int32_t index = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t feature = index & (kWidth - 1);
    const std::uint16_t gated = bf16_multiply_rn(down[index], ffw_gate[feature]);
    post_mlp[index] = bf16_add_rn(post_attention[index], gated);
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_activation_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == kRows && dims.d[2] == kWidth;
}

bool has_gate_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == 1 && dims.d[2] == kWidth;
}

bool has_matrix_shape(nvinfer1::Dims const& dims, int32_t rows, int32_t columns) {
    return dims.nbDims == 2 && dims.d[0] == rows && dims.d[1] == columns;
}

bool has_supported_dimensions(nvinfer1::Dims const* input, nvinfer1::Dims const& output) {
    return has_activation_shape(input[0]) && has_activation_shape(input[1]) &&
           has_gate_shape(input[2]) && has_matrix_shape(input[3], kWidth, kMlpWidth) &&
           has_matrix_shape(input[4], kWidth, kMlpWidth) &&
           has_matrix_shape(input[5], kMlpWidth, kWidth) && has_activation_shape(output);
}

bool has_supported_shape(nvinfer1::PluginTensorDesc const* input, int32_t nb_inputs,
                         nvinfer1::PluginTensorDesc const* output, int32_t nb_outputs) {
    if (input == nullptr || output == nullptr || nb_inputs != 6 || nb_outputs != 1) {
        return false;
    }
    nvinfer1::Dims input_dims[6];
    for (int32_t index = 0; index < 6; ++index) {
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
    if (input == nullptr || output == nullptr || nb_inputs != 6 || nb_outputs != 1) {
        return false;
    }
    nvinfer1::PluginTensorDesc input_desc[6];
    nvinfer1::Dims minimum[6];
    nvinfer1::Dims maximum[6];
    for (int32_t index = 0; index < 6; ++index) {
        input_desc[index] = input[index].desc;
        minimum[index] = input[index].min;
        maximum[index] = input[index].max;
    }
    const nvinfer1::PluginTensorDesc output_desc[1] = {output[0].desc};
    return has_supported_shape(input_desc, 6, output_desc, 1) &&
           has_supported_dimensions(minimum, output[0].min) &&
           has_supported_dimensions(maximum, output[0].max);
}

class OpenPIActionLayer0MlpClosurePlugin final : public nvinfer1::IPluginV3,
                                                 public nvinfer1::IPluginV3OneCore,
                                                 public nvinfer1::IPluginV3OneBuild,
                                                 public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIActionLayer0MlpClosure";
    static constexpr const char* kVersion = "1";

    OpenPIActionLayer0MlpClosurePlugin() { serialization_fields_.nbFields = 0; }

    ~OpenPIActionLayer0MlpClosurePlugin() override {
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
        auto* plugin = new (std::nothrow) OpenPIActionLayer0MlpClosurePlugin();
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
            nb_inputs != 6) {
            return 1;
        }
        for (int32_t index = 0; index < 6; ++index) {
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
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 6 || nb_shape_inputs != 0 ||
            nb_outputs != 1) {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        if (descriptors == nullptr || nb_inputs != 6 || nb_outputs != 1 || position < 0 ||
            position >= 7) {
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
        if (!has_supported_shape(input_desc, 6, output_desc, 1) || inputs == nullptr ||
            outputs == nullptr || workspace == nullptr || cublas_handle_ == nullptr ||
            outputs[0] == nullptr) {
            return 1;
        }
        for (int32_t index = 0; index < 6; ++index) {
            if (inputs[index] == nullptr) {
                return 1;
            }
        }
        return launch_openpi_action_layer0_mlp_closure(
            static_cast<const std::uint16_t*>(inputs[0]),
            static_cast<const std::uint16_t*>(inputs[1]),
            static_cast<const std::uint16_t*>(inputs[2]),
            static_cast<const std::uint16_t*>(inputs[3]),
            static_cast<const std::uint16_t*>(inputs[4]),
            static_cast<const std::uint16_t*>(inputs[5]), static_cast<std::uint16_t*>(outputs[0]),
            workspace, static_cast<void*>(cublas_handle_), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        auto* plugin = new (std::nothrow) OpenPIActionLayer0MlpClosurePlugin();
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

class OpenPIActionLayer0MlpClosureCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIActionLayer0MlpClosureCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields != nullptr && fields->nbFields != 0) {
            return nullptr;
        }
        return new (std::nothrow) OpenPIActionLayer0MlpClosurePlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIActionLayer0MlpClosurePlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIActionLayer0MlpClosurePlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

static nvinfer1::PluginRegistrar<OpenPIActionLayer0MlpClosureCreator>
    plugin_registrar_openpi_action_layer0_mlp_closure{};

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

int32_t launch_openpi_action_layer0_mlp_closure(
    const std::uint16_t* post_attention, const std::uint16_t* normed_ffw,
    const std::uint16_t* ffw_gate, const std::uint16_t* gate_weight, const std::uint16_t* up_weight,
    const std::uint16_t* down_weight, std::uint16_t* post_mlp, void* workspace, void* cublas_handle,
    cudaStream_t stream) noexcept {
    if (post_attention == nullptr || normed_ffw == nullptr || ffw_gate == nullptr ||
        gate_weight == nullptr || up_weight == nullptr || down_weight == nullptr ||
        post_mlp == nullptr || workspace == nullptr || cublas_handle == nullptr) {
        return 1;
    }

    auto* workspace_bytes = static_cast<std::byte*>(workspace);
    auto* combined_weight =
        reinterpret_cast<std::uint16_t*>(workspace_bytes + kCombinedWeightOffset);
    auto* padded_normed = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPaddedNormedOffset);
    auto* combined_output =
        reinterpret_cast<std::uint16_t*>(workspace_bytes + kCombinedOutputOffset);
    void* cublas_workspace = workspace_bytes + kCublasWorkspaceOffset;

    constexpr std::size_t kWeightRowBytes =
        static_cast<std::size_t>(kMlpWidth) * sizeof(std::uint16_t);
    constexpr std::size_t kCombinedWeightRowBytes =
        static_cast<std::size_t>(kCombinedWidth) * sizeof(std::uint16_t);
    if (cudaMemcpy2DAsync(combined_weight, kCombinedWeightRowBytes, gate_weight, kWeightRowBytes,
                          kWeightRowBytes, kWidth, cudaMemcpyDeviceToDevice,
                          stream) != cudaSuccess ||
        cudaMemcpy2DAsync(combined_weight + kMlpWidth, kCombinedWeightRowBytes, up_weight,
                          kWeightRowBytes, kWeightRowBytes, kWidth, cudaMemcpyDeviceToDevice,
                          stream) != cudaSuccess ||
        cudaMemcpyAsync(padded_normed, normed_ffw,
                        static_cast<std::size_t>(kRows) * kWidth * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemsetAsync(padded_normed + static_cast<std::size_t>(kRows) * kWidth, 0,
                        static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t),
                        stream) != cudaSuccess) {
        return 1;
    }

    auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
    if (configure_cublas(handle, stream, cublas_workspace, kCombinedCublasWorkspaceBytes) != 0) {
        return 1;
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, kCombinedWidth, kPaddedRows, kWidth, &alpha,
        combined_weight, CUDA_R_16BF, kCombinedWidth, padded_normed, CUDA_R_16BF, kWidth, &beta,
        combined_output, CUDA_R_16BF, kCombinedWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    auto* fused = reinterpret_cast<std::uint16_t*>(workspace_bytes + kFusedOutputOffset);
    openpi_action_layer0_gelu_gated_kernel<<<kGeluBlocks, kPointwiseThreads, 0, stream>>>(
        combined_output, fused);
    if (cudaPeekAtLastError() != cudaSuccess ||
        configure_cublas(handle, stream, cublas_workspace, kDownCublasWorkspaceBytes) != 0) {
        return 1;
    }

    auto* down = reinterpret_cast<std::uint16_t*>(workspace_bytes + kDownOutputOffset);
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, kWidth, kPaddedRows, kMlpWidth, &alpha,
                          down_weight, CUDA_R_16BF, kWidth, fused, CUDA_R_16BF, kMlpWidth, &beta,
                          down, CUDA_R_16BF, kWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    openpi_action_layer0_mlp_residual_kernel<<<kResidualBlocks, kPointwiseThreads, 0, stream>>>(
        post_attention, down, ffw_gate, post_mlp);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::openpi
