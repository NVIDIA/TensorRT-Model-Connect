/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hiera_block1415_projection_plugin.h"

#include <array>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <initializer_list>

namespace trtmc::sam2_hoi {
namespace {

constexpr uint32_t kExpectedTile = static_cast<uint32_t>(CUBLASLT_MATMUL_TILE_64x128);
constexpr uint32_t kRequiredAlignment = 16;
constexpr int32_t kMaximumHeuristicResults = 32;

bool dimsEqual(const nvinfer1::Dims& dims, std::initializer_list<int32_t> expected) noexcept {
    if (dims.nbDims != static_cast<int32_t>(expected.size())) {
        return false;
    }
    int32_t index = 0;
    for (int32_t value : expected) {
        if (dims.d[index++] != value) {
            return false;
        }
    }
    return true;
}

bool tensorContract(const nvinfer1::PluginTensorDesc& descriptor,
                    std::initializer_list<int32_t> shape) noexcept {
    return descriptor.type == nvinfer1::DataType::kBF16 &&
           descriptor.format == nvinfer1::TensorFormat::kLINEAR &&
           dimsEqual(descriptor.dims, shape);
}

bool dynamicTensorContract(const nvinfer1::DynamicPluginTensorDesc& descriptor,
                           std::initializer_list<int32_t> shape) noexcept {
    return tensorContract(descriptor.desc, shape) && dimsEqual(descriptor.min, shape) &&
           dimsEqual(descriptor.max, shape);
}

bool aligned(const void* pointer) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % kRequiredAlignment == 0;
}

template <typename Value>
bool algorithmAttribute(const cublasLtMatmulAlgo_t& algorithm,
                        cublasLtMatmulAlgoConfigAttributes_t attribute, Value& value) noexcept {
    std::size_t bytes_written = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, &value, sizeof(value),
                                                &bytes_written) == CUBLAS_STATUS_SUCCESS &&
           bytes_written == sizeof(value);
}

bool exactAlgorithmConfiguration(const cublasLtMatmulAlgo_t& algorithm) noexcept {
    // The qualification trace exposes and fixes this ID/tile/swizzle tuple.
    // Other cuBLASLt configuration fields were not independently recorded, so
    // they are not claimed here; the pinned DSO identity, algoCheck, and model
    // accuracy gates remain part of the numerical qualification boundary.
    int32_t algorithm_id = -1;
    uint32_t tile = static_cast<uint32_t>(CUBLASLT_MATMUL_TILE_UNDEFINED);
    uint32_t swizzling = 0;
    return algorithmAttribute(algorithm, CUBLASLT_ALGO_CONFIG_ID, algorithm_id) &&
           algorithmAttribute(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, tile) &&
           algorithmAttribute(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzling) &&
           algorithm_id == HieraBlock1415ProjectionPlugin::kALGORITHM_ID && tile == kExpectedTile &&
           swizzling == HieraBlock1415ProjectionPlugin::kCTA_SWIZZLING;
}

} // namespace

HieraBlock1415ProjectionPlugin::LockGuard::LockGuard(std::atomic_flag& lock) noexcept
    : lock_(lock) {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
}

HieraBlock1415ProjectionPlugin::LockGuard::~LockGuard() {
    lock_.clear(std::memory_order_release);
}

HieraBlock1415ProjectionPlugin::HieraBlock1415ProjectionPlugin(const void* data, std::size_t length)
    : configured_(length == 0), serialization_valid_(length == 0) {
    (void)data;
}

HieraBlock1415ProjectionPlugin::~HieraBlock1415ProjectionPlugin() {
    LockGuard guard(lock_);
    releaseLocked();
}

char const* HieraBlock1415ProjectionPlugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* HieraBlock1415ProjectionPlugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int32_t HieraBlock1415ProjectionPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t HieraBlock1415ProjectionPlugin::initialize() noexcept {
    LockGuard guard(lock_);
    releaseLocked();
    return initializeLocked() ? 0 : 1;
}

void HieraBlock1415ProjectionPlugin::terminate() noexcept {
    LockGuard guard(lock_);
    releaseLocked();
}

void HieraBlock1415ProjectionPlugin::destroy() noexcept {
    delete this;
}

std::size_t HieraBlock1415ProjectionPlugin::getSerializationSize() const noexcept {
    return 0;
}

void HieraBlock1415ProjectionPlugin::serialize(void* buffer) const noexcept {
    (void)buffer;
}

void HieraBlock1415ProjectionPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
    namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

char const* HieraBlock1415ProjectionPlugin::getPluginNamespace() const noexcept {
    return namespace_.c_str();
}

nvinfer1::DataType HieraBlock1415ProjectionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* input_types, int32_t num_inputs) const noexcept {
    if (index == 0 && num_inputs == 3 && input_types != nullptr &&
        input_types[0] == nvinfer1::DataType::kBF16 &&
        input_types[1] == nvinfer1::DataType::kBF16 &&
        input_types[2] == nvinfer1::DataType::kBF16) {
        return nvinfer1::DataType::kBF16;
    }
    return nvinfer1::DataType::kFLOAT;
}

HieraBlock1415ProjectionPlugin* HieraBlock1415ProjectionPlugin::clone() const noexcept {
    auto* result = new HieraBlock1415ProjectionPlugin();
    result->namespace_ = namespace_;
    result->configured_ = configured_;
    result->serialization_valid_ = serialization_valid_;
    return result;
}

nvinfer1::DimsExprs HieraBlock1415ProjectionPlugin::getOutputDimensions(
    int32_t output_index, nvinfer1::DimsExprs const* inputs, int32_t num_inputs,
    nvinfer1::IExprBuilder& expression_builder) noexcept {
    (void)expression_builder;
    nvinfer1::DimsExprs output{};
    if (output_index == 0 && num_inputs == 3 && inputs != nullptr) {
        output = inputs[0];
    }
    return output;
}

bool HieraBlock1415ProjectionPlugin::supportsFormatCombination(
    int32_t position, nvinfer1::PluginTensorDesc const* inputs_outputs, int32_t num_inputs,
    int32_t num_outputs) noexcept {
    return inputs_outputs != nullptr && num_inputs == 3 && num_outputs == 1 && position >= 0 &&
           position < 4 && inputs_outputs[position].type == nvinfer1::DataType::kBF16 &&
           inputs_outputs[position].format == nvinfer1::TensorFormat::kLINEAR;
}

void HieraBlock1415ProjectionPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t num_inputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t num_outputs) noexcept {
    LockGuard guard(lock_);
    configured_ =
        serialization_valid_ && inputs != nullptr && outputs != nullptr && num_inputs == 3 &&
        num_outputs == 1 && dynamicTensorContract(inputs[0], {1, kM, kK}) &&
        dynamicTensorContract(inputs[1], {kN, kK}) && dynamicTensorContract(inputs[2], {kN}) &&
        dynamicTensorContract(outputs[0], {1, kM, kN});
}

std::size_t HieraBlock1415ProjectionPlugin::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs, int32_t num_inputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t num_outputs) const noexcept {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    return kWORKSPACE_BYTES;
}

bool HieraBlock1415ProjectionPlugin::initializeLocked() noexcept {
    if (!serialization_valid_) {
        return false;
    }
    if (cublasLtCreate(&handle_) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
            CUBLAS_STATUS_SUCCESS) {
        releaseLocked();
        return false;
    }
    cublasOperation_t transpose_weight = CUBLAS_OP_T;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    if (cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_weight,
                                       sizeof(transpose_weight)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                       sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&weight_layout_, CUDA_R_16BF, kK, kN, kK) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&input_layout_, CUDA_R_16BF, kK, kM, kK) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&output_layout_, CUDA_R_16BF, kN, kM, kN) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulPreferenceCreate(&preference_) != CUBLAS_STATUS_SUCCESS) {
        releaseLocked();
        return false;
    }

    std::size_t workspace = kWORKSPACE_BYTES;
    if (cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace,
                                             sizeof(workspace)) != CUBLAS_STATUS_SUCCESS) {
        releaseLocked();
        return false;
    }
    uint32_t alignment = kRequiredAlignment;
    for (cublasLtMatmulPreferenceAttributes_t attribute : {
             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
             CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
         }) {
        if (cublasLtMatmulPreferenceSetAttribute(preference_, attribute, &alignment,
                                                 sizeof(alignment)) != CUBLAS_STATUS_SUCCESS) {
            releaseLocked();
            return false;
        }
    }
    // The algorithm is part of the numerical contract, not a runtime tuning
    // choice. Validate it during plugin initialization so an unsupported
    // cuBLASLt/device combination fails before a usable execution context is
    // exposed. enqueue repeats initialization only for TRT 11.x runtimes
    // that deserialize a legacy V2 plugin without calling initialize().
    if (!selectExactAlgorithmLocked()) {
        releaseLocked();
        return false;
    }
    return true;
}

bool HieraBlock1415ProjectionPlugin::selectExactAlgorithmLocked() noexcept {
    std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristicResults> results{};
    int32_t returned = 0;
    if (cublasLtMatmulAlgoGetHeuristic(handle_, operation_, weight_layout_, input_layout_,
                                       output_layout_, output_layout_, preference_,
                                       kMaximumHeuristicResults, results.data(),
                                       &returned) != CUBLAS_STATUS_SUCCESS ||
        returned <= 0) {
        return false;
    }
    for (int32_t index = 0; index < returned; ++index) {
        auto const& candidate = results[static_cast<std::size_t>(index)];
        if (candidate.state != CUBLAS_STATUS_SUCCESS ||
            candidate.workspaceSize > kWORKSPACE_BYTES ||
            !exactAlgorithmConfiguration(candidate.algo)) {
            continue;
        }
        cublasLtMatmulHeuristicResult_t checked{};
        if (cublasLtMatmulAlgoCheck(handle_, operation_, weight_layout_, input_layout_,
                                    output_layout_, output_layout_, &candidate.algo,
                                    &checked) != CUBLAS_STATUS_SUCCESS ||
            checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > kWORKSPACE_BYTES) {
            continue;
        }
        algorithm_ = candidate.algo;
        algorithm_ready_ = true;
        return true;
    }
    return false;
}

int32_t
HieraBlock1415ProjectionPlugin::enqueue(nvinfer1::PluginTensorDesc const* input_descriptors,
                                        nvinfer1::PluginTensorDesc const* output_descriptors,
                                        void const* const* inputs, void* const* outputs,
                                        void* workspace, cudaStream_t stream) noexcept {
    LockGuard guard(lock_);
    if (handle_ == nullptr && !initializeLocked()) {
        return 1;
    }
    if (!configured_ || input_descriptors == nullptr || output_descriptors == nullptr ||
        inputs == nullptr || outputs == nullptr ||
        !tensorContract(input_descriptors[0], {1, kM, kK}) ||
        !tensorContract(input_descriptors[1], {kN, kK}) ||
        !tensorContract(input_descriptors[2], {kN}) ||
        !tensorContract(output_descriptors[0], {1, kM, kN}) || !aligned(inputs[0]) ||
        !aligned(inputs[1]) || !aligned(inputs[2]) || !aligned(outputs[0]) || !aligned(workspace)) {
        std::fprintf(stderr, "SAM2 HOI Hiera block14/15 projection contract mismatch\n");
        return 1;
    }

    void const* bias = inputs[2];
    if (cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                       sizeof(bias)) != CUBLAS_STATUS_SUCCESS ||
        (!algorithm_ready_ && !selectExactAlgorithmLocked())) {
        std::fprintf(stderr, "SAM2 HOI Hiera block14/15 projection exact cuBLASLt "
                             "algorithm unavailable (required id=30 tile=64x128 swizzle=1)\n");
        return 1;
    }

    float alpha = 1.0F;
    float beta = 0.0F;
    cublasStatus_t status =
        cublasLtMatmul(handle_, operation_, &alpha, inputs[1], weight_layout_, inputs[0],
                       input_layout_, &beta, outputs[0], output_layout_, outputs[0], output_layout_,
                       &algorithm_, workspace, kWORKSPACE_BYTES, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "SAM2 HOI Hiera block14/15 projection cuBLASLt failure: %d\n",
                     static_cast<int>(status));
        return 1;
    }
    return 0;
}

void HieraBlock1415ProjectionPlugin::releaseLocked() noexcept {
    if (preference_ != nullptr) {
        cublasLtMatmulPreferenceDestroy(preference_);
    }
    if (output_layout_ != nullptr) {
        cublasLtMatrixLayoutDestroy(output_layout_);
    }
    if (input_layout_ != nullptr) {
        cublasLtMatrixLayoutDestroy(input_layout_);
    }
    if (weight_layout_ != nullptr) {
        cublasLtMatrixLayoutDestroy(weight_layout_);
    }
    if (operation_ != nullptr) {
        cublasLtMatmulDescDestroy(operation_);
    }
    if (handle_ != nullptr) {
        cublasLtDestroy(handle_);
    }
    handle_ = nullptr;
    operation_ = nullptr;
    weight_layout_ = nullptr;
    input_layout_ = nullptr;
    output_layout_ = nullptr;
    preference_ = nullptr;
    algorithm_ = {};
    algorithm_ready_ = false;
}

} // namespace trtmc::sam2_hoi
