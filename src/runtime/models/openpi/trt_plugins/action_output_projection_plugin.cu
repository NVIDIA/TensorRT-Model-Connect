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

int32_t launch_openpi_action_output_projection(const std::uint16_t* hidden,
                                               const std::uint16_t* weight,
                                               const std::uint16_t* bias, std::uint16_t* velocity,
                                               void* workspace, void* cublas_handle,
                                               cudaStream_t stream) noexcept;

namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kRows = 15;
constexpr int32_t kPaddedRows = 16;
constexpr int32_t kWidth = 1024;
constexpr int32_t kOutputWidth = 32;
constexpr std::size_t kPaddedInputElements = static_cast<std::size_t>(kPaddedRows) * kWidth;
constexpr std::size_t kPaddedInputBytes = kPaddedInputElements * sizeof(std::uint16_t);
constexpr std::size_t kPaddedOutputElements = static_cast<std::size_t>(kPaddedRows) * kOutputWidth;
constexpr std::size_t kPaddedOutputBytes = kPaddedOutputElements * sizeof(std::uint16_t);
// Exact scratch allocation attached to custom-call.73 in the pinned
// JAX 0.5.3 / XLA action executable.
constexpr std::size_t kCublasWorkspaceBytes = 98304;
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kWorkspaceAlignmentSlack = kWorkspaceAlignment - 1;
// The pinned XLA buffer assignment places all three cuBLAS pointers at +0x80
// modulo 256. CUBLAS_GEMM_DEFAULT is alignment-sensitive, so preserve that
// physical address geometry rather than merely keeping the regions disjoint.
// TensorRT supplies an opaque workspace pointer, so reserve enough slack to
// align its base explicitly instead of depending on an undocumented alignment.
constexpr std::size_t kPaddedInputOffset = 0x80;
constexpr std::size_t kPaddedOutputOffset = kPaddedInputOffset + kPaddedInputBytes;
constexpr std::size_t kCublasWorkspaceOffset = kPaddedOutputOffset + kPaddedOutputBytes;
constexpr std::size_t kWorkspacePayloadBytes = kCublasWorkspaceOffset + kCublasWorkspaceBytes;
constexpr std::size_t kPluginWorkspaceBytes = kWorkspaceAlignmentSlack + kWorkspacePayloadBytes;
static_assert(kPaddedInputOffset == 0x0080);
static_assert(kPaddedOutputOffset == 0x8080);
static_assert(kCublasWorkspaceOffset == 0x8480);
static_assert(kWorkspacePayloadBytes == 0x20480);
static_assert(kPluginWorkspaceBytes == 0x2057F);
static_assert((kWorkspaceAlignment & (kWorkspaceAlignment - 1)) == 0);
static_assert(kPaddedInputOffset % 256 == 128);
static_assert(kPaddedOutputOffset % 256 == 128);
static_assert(kCublasWorkspaceOffset % 256 == 128);
constexpr int32_t kPointwiseThreads = 128;
constexpr int32_t kOutputElements = kRows * kOutputWidth;
constexpr int32_t kPointwiseBlocks = (kOutputElements + kPointwiseThreads - 1) / kPointwiseThreads;

__device__ __forceinline__ std::uint16_t bf16_add_rn(std::uint16_t lhs, std::uint16_t rhs) {
    // Match the pinned sm_103a XLA PTX directly on the qualified platform.
    std::uint16_t result;
    asm("add.rn.bf16 %0, %1, %2;" : "=h"(result) : "h"(lhs), "h"(rhs));
    return result;
}

__global__ void openpi_action_output_bias_kernel(const std::uint16_t* __restrict__ projected,
                                                 const std::uint16_t* __restrict__ bias,
                                                 std::uint16_t* __restrict__ velocity) {
    const int32_t index = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < kOutputElements) {
        velocity[index] = bf16_add_rn(projected[index], bias[index % kOutputWidth]);
    }
}

bool is_bf16_linear(nvinfer1::PluginTensorDesc const& descriptor) {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR;
}

bool has_hidden_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == kRows && dims.d[2] == kWidth;
}

bool has_weight_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 2 && dims.d[0] == kWidth && dims.d[1] == kOutputWidth;
}

bool has_bias_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 1 && dims.d[0] == kOutputWidth;
}

bool has_output_shape(nvinfer1::Dims const& dims) {
    return dims.nbDims == 3 && dims.d[0] == kBatch && dims.d[1] == kRows &&
           dims.d[2] == kOutputWidth;
}

bool has_supported_dimensions(nvinfer1::Dims const* inputs, nvinfer1::Dims const& output) {
    return has_hidden_shape(inputs[0]) && has_weight_shape(inputs[1]) &&
           has_bias_shape(inputs[2]) && has_output_shape(output);
}

bool has_supported_shape(nvinfer1::PluginTensorDesc const* inputs, int32_t nb_inputs,
                         nvinfer1::PluginTensorDesc const* outputs, int32_t nb_outputs) {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 3 || nb_outputs != 1) {
        return false;
    }
    nvinfer1::Dims input_dims[3];
    for (int32_t index = 0; index < 3; ++index) {
        if (!is_bf16_linear(inputs[index])) {
            return false;
        }
        input_dims[index] = inputs[index].dims;
    }
    return is_bf16_linear(outputs[0]) && has_supported_dimensions(input_dims, outputs[0].dims);
}

bool has_supported_dynamic_shape(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nb_inputs,
                                 nvinfer1::DynamicPluginTensorDesc const* outputs,
                                 int32_t nb_outputs) {
    if (inputs == nullptr || outputs == nullptr || nb_inputs != 3 || nb_outputs != 1) {
        return false;
    }
    const nvinfer1::PluginTensorDesc input_desc[3] = {inputs[0].desc, inputs[1].desc,
                                                      inputs[2].desc};
    const nvinfer1::PluginTensorDesc output_desc[1] = {outputs[0].desc};
    const nvinfer1::Dims minimum[3] = {inputs[0].min, inputs[1].min, inputs[2].min};
    const nvinfer1::Dims maximum[3] = {inputs[0].max, inputs[1].max, inputs[2].max};
    return has_supported_shape(input_desc, 3, output_desc, 1) &&
           has_supported_dimensions(minimum, outputs[0].min) &&
           has_supported_dimensions(maximum, outputs[0].max);
}

int32_t configure_cublas(cublasHandle_t handle, cudaStream_t stream, void* workspace) noexcept {
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }
    status = cublasSetWorkspace(handle, workspace, kCublasWorkspaceBytes);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

class OpenPIActionOutputProjectionPlugin final : public nvinfer1::IPluginV3,
                                                 public nvinfer1::IPluginV3OneCore,
                                                 public nvinfer1::IPluginV3OneBuild,
                                                 public nvinfer1::IPluginV3OneRuntime {
  public:
    static constexpr const char* kName = "OpenPIActionOutputProjection";
    static constexpr const char* kVersion = "1";

    OpenPIActionOutputProjectionPlugin() { serialization_fields_.nbFields = 0; }

    ~OpenPIActionOutputProjectionPlugin() override {
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
        auto* plugin = new (std::nothrow) OpenPIActionOutputProjectionPlugin();
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
            nb_inputs != 3) {
            return 1;
        }
        for (int32_t index = 0; index < 3; ++index) {
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
                            nvinfer1::IExprBuilder& expression_builder) noexcept override {
        if (inputs == nullptr || outputs == nullptr || nb_inputs != 3 || nb_shape_inputs != 0 ||
            nb_outputs != 1 || inputs[0].nbDims != 3 || inputs[1].nbDims != 2 ||
            inputs[2].nbDims != 1 || inputs[0].d[0] == nullptr || inputs[0].d[1] == nullptr) {
            return 1;
        }
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = expression_builder.constant(kOutputWidth);
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
                                   nvinfer1::DynamicPluginTensorDesc const* descriptors,
                                   int32_t nb_inputs, int32_t nb_outputs) noexcept override {
        return descriptors != nullptr && nb_inputs == 3 && nb_outputs == 1 && position >= 0 &&
               position < 4 && is_bf16_linear(descriptors[position].desc);
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
        if (!has_supported_shape(input_desc, 3, output_desc, 1) || inputs == nullptr ||
            outputs == nullptr || workspace == nullptr || cublas_handle_ == nullptr ||
            outputs[0] == nullptr) {
            return 1;
        }
        for (int32_t index = 0; index < 3; ++index) {
            if (inputs[index] == nullptr) {
                return 1;
            }
        }
        return launch_openpi_action_output_projection(
            static_cast<const std::uint16_t*>(inputs[0]),
            static_cast<const std::uint16_t*>(inputs[1]),
            static_cast<const std::uint16_t*>(inputs[2]), static_cast<std::uint16_t*>(outputs[0]),
            workspace, static_cast<void*>(cublas_handle_), stream);
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        auto* plugin = new (std::nothrow) OpenPIActionOutputProjectionPlugin();
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
        status = cublasSetPointerMode(plugin->cublas_handle_, CUBLAS_POINTER_MODE_HOST);
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

class OpenPIActionOutputProjectionCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    OpenPIActionOutputProjectionCreator() { fields_.nbFields = 0; }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields != nullptr && fields->nbFields != 0) {
            return nullptr;
        }
        return new (std::nothrow) OpenPIActionOutputProjectionPlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return OpenPIActionOutputProjectionPlugin::kName;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return OpenPIActionOutputProjectionPlugin::kVersion;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

  private:
    nvinfer1::PluginFieldCollection fields_{};
};

static nvinfer1::PluginRegistrar<OpenPIActionOutputProjectionCreator>
    plugin_registrar_openpi_action_output_projection{};

} // namespace

int32_t launch_openpi_action_output_projection(const std::uint16_t* hidden,
                                               const std::uint16_t* weight,
                                               const std::uint16_t* bias, std::uint16_t* velocity,
                                               void* workspace, void* cublas_handle,
                                               cudaStream_t stream) noexcept {
    if (hidden == nullptr || weight == nullptr || bias == nullptr || velocity == nullptr ||
        workspace == nullptr || cublas_handle == nullptr) {
        return 1;
    }

    const auto workspace_address = reinterpret_cast<std::uintptr_t>(workspace);
    const auto aligned_workspace_address = (workspace_address + kWorkspaceAlignmentSlack) &
                                           ~static_cast<std::uintptr_t>(kWorkspaceAlignmentSlack);
    auto* workspace_bytes = reinterpret_cast<std::byte*>(aligned_workspace_address);
    auto* padded_hidden = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPaddedInputOffset);
    auto* padded_output = reinterpret_cast<std::uint16_t*>(workspace_bytes + kPaddedOutputOffset);
    void* cublas_workspace = workspace_bytes + kCublasWorkspaceOffset;

    constexpr std::size_t kInputBytes =
        static_cast<std::size_t>(kRows) * kWidth * sizeof(std::uint16_t);
    constexpr std::size_t kPaddingBytes = static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t);
    if (cudaMemcpyAsync(padded_hidden, hidden, kInputBytes, cudaMemcpyDeviceToDevice, stream) !=
            cudaSuccess ||
        cudaMemsetAsync(padded_hidden + static_cast<std::size_t>(kRows) * kWidth, 0, kPaddingBytes,
                        stream) != cudaSuccess) {
        return 1;
    }

    auto handle = reinterpret_cast<cublasHandle_t>(cublas_handle);
    if (configure_cublas(handle, stream, cublas_workspace) != 0) {
        return 1;
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, kOutputWidth, kPaddedRows, kWidth, &alpha, weight,
        CUDA_R_16BF, kOutputWidth, padded_hidden, CUDA_R_16BF, kWidth, &beta, padded_output,
        CUDA_R_16BF, kOutputWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 1;
    }

    openpi_action_output_bias_kernel<<<kPointwiseBlocks, kPointwiseThreads, 0, stream>>>(
        padded_output, bias, velocity);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

} // namespace trtmc::openpi
