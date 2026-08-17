/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "checkpoint_reader.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "sam2_engine_builder.h"
#include "sam2_image_network.h"
#include "sam2_tracker_network.h"
#include "sam2_trt_layers.h"

#include <NvInfer.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::sam2::native {

namespace {

class Sam2Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "TensorRT SAM2: " << (message == nullptr ? "" : message) << '\n';
    }
};

template <typename T>
struct TrtDelete {
    void operator()(T* object) const noexcept {
        if (object != nullptr)
            delete object;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

void requireCuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw Sam2EngineBuildError(std::string("CUDA ") + std::string(operation) +
                                   " failed: " + cudaGetErrorString(status));
    }
}

std::string cudaVersionString(std::int32_t version) {
    if (version <= 0)
        throw Sam2EngineBuildError("CUDA returned an invalid version");
    const std::int32_t major = version / 1000;
    const std::int32_t minor = (version % 1000) / 10;
    const std::int32_t patch = version % 10;
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << major << '.' << minor << '.' << patch;
    return output.str();
}

std::string tensorrtVersionString() {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH;
#if defined(NV_TENSORRT_BUILD)
    output << '.' << NV_TENSORRT_BUILD;
#endif
    return output.str();
}

std::string tensorrtAbiString() {
    return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR);
}

Sam2RuntimeBuildFacts inspectRuntime(std::int32_t requested_device) {
    requireCuda(cudaSetDevice(requested_device), "device selection");
    std::int32_t active_device = -1;
    requireCuda(cudaGetDevice(&active_device), "active-device query");
    if (active_device != requested_device)
        throw Sam2EngineBuildError("CUDA active device does not match the requested device");

    std::int32_t runtime_version = 0;
    std::int32_t driver_version = 0;
    requireCuda(cudaRuntimeGetVersion(&runtime_version), "runtime-version query");
    requireCuda(cudaDriverGetVersion(&driver_version), "driver-version query");
    cudaDeviceProp properties{};
    requireCuda(cudaGetDeviceProperties(&properties, active_device), "device-properties query");

    Sam2RuntimeBuildFacts facts;
    facts.tensorrt_version = tensorrtVersionString();
    facts.tensorrt_abi = tensorrtAbiString();
    facts.cuda_runtime_version = cudaVersionString(runtime_version);
    facts.cuda_driver_version = cudaVersionString(driver_version);
    facts.gpu_name = properties.name;
    facts.gpu_device = active_device;
    facts.gpu_compute_major = properties.major;
    facts.gpu_compute_minor = properties.minor;
    facts.gpu_global_memory_bytes = static_cast<std::uint64_t>(properties.totalGlobalMem);
    facts.strongly_typed = true;
    facts.tf32_enabled = false;
    return facts;
}

TrtPtr<nvinfer1::INetworkDefinition> createNetwork(nvinfer1::IBuilder& builder) {
    TrtPtr<nvinfer1::INetworkDefinition> network(
        builder.createNetworkV2(sam2NetworkCreationFlags()));
    if (!network)
        throw Sam2EngineBuildError("TensorRT failed to create a SAM2 network");
    return network;
}

std::vector<std::uint8_t> serializeNetwork(nvinfer1::IBuilder& builder,
                                           nvinfer1::INetworkDefinition& network,
                                           std::uint64_t workspace_bytes,
                                           std::string_view section) {
    if (network.getNbInputs() <= 0 || network.getNbOutputs() <= 0 || network.getNbLayers() <= 0)
        throw Sam2EngineBuildError("SAM2 graph is incomplete before serialization: " +
                                   std::string(section));
    TrtPtr<nvinfer1::IBuilderConfig> config(builder.createBuilderConfig());
    if (!config)
        throw Sam2EngineBuildError("TensorRT failed to create a SAM2 builder configuration");
    config->setBuilderOptimizationLevel(trtmc::sam2::kBuilderOptimizationLevel);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<std::size_t>(workspace_bytes));
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    if (config->getFlag(nvinfer1::BuilderFlag::kTF32))
        throw Sam2EngineBuildError("TensorRT did not disable TF32 for SAM2");
    if (config->getBuilderOptimizationLevel() != trtmc::sam2::kBuilderOptimizationLevel) {
        throw Sam2EngineBuildError("TensorRT did not retain SAM2 builder optimization level 3");
    }
    if (config->getProfilingVerbosity() != nvinfer1::ProfilingVerbosity::kDETAILED) {
        throw Sam2EngineBuildError(
            "TensorRT did not retain detailed SAM2 plan profiling verbosity");
    }

    TrtPtr<nvinfer1::IHostMemory> plan(builder.buildSerializedNetwork(network, *config));
    if (!plan || plan->data() == nullptr || plan->size() == 0U)
        throw Sam2EngineBuildError("TensorRT produced an empty SAM2 plan: " + std::string(section));
    if (plan->size() > std::vector<std::uint8_t>().max_size())
        throw Sam2EngineBuildError("TensorRT SAM2 plan exceeds the host vector size limit");
    std::vector<std::uint8_t> bytes(plan->size());
    std::memcpy(bytes.data(), plan->data(), plan->size());
    return bytes;
}

bool imageOutputsComplete(const Sam2ImageNetworkOutputs& outputs) {
    if (outputs.pixel_values == nullptr)
        return false;
    for (const auto* tensor : outputs.tracker_fpn) {
        if (tensor == nullptr)
            return false;
    }
    for (const auto* tensor : outputs.bbox_classification) {
        if (tensor == nullptr)
            return false;
    }
    for (const auto* tensor : outputs.bbox_regression) {
        if (tensor == nullptr)
            return false;
    }
    return true;
}

bool trackerOutputsComplete(const Sam2TrackerNetworkOutputs& outputs) {
    return outputs.mask_logits_256 != nullptr && outputs.object_pointer != nullptr &&
           outputs.memory_features != nullptr;
}

Sam2SerializedPlan buildImagePlan(nvinfer1::IBuilder& builder, const CheckpointReader& checkpoint,
                                  std::uint64_t workspace_bytes) {
    auto network = createNetwork(builder);
    Sam2ImageNetworkBuilder graph_builder(*network, checkpoint);
    const Sam2ImageNetworkOutputs outputs = graph_builder.build();
    const bool complete =
        imageOutputsComplete(outputs) && network->getNbInputs() == 1 &&
        network->getNbOutputs() == 9 &&
        outputs.checkpoint_tensor_count == kDeliveredCheckpointTensorCount &&
        outputs.referenced_tensor_count == kImageNetworkReferencedTensorCount &&
        outputs.added_layer_count == kImageNetworkLayerCount &&
        outputs.added_layer_count == network->getNbLayers() &&
        outputs.convolution_layer_count == kImageNetworkConvolutionLayerCount &&
        outputs.activation_layer_count == kImageNetworkActivationLayerCount &&
        outputs.pooling_layer_count == kImageNetworkPoolingLayerCount &&
        outputs.element_wise_layer_count == kImageNetworkElementWiseLayerCount &&
        outputs.shuffle_layer_count == kImageNetworkShuffleLayerCount &&
        outputs.constant_layer_count == kImageNetworkConstantLayerCount &&
        outputs.slice_layer_count == kImageNetworkSliceLayerCount &&
        outputs.resize_layer_count == kImageNetworkResizeLayerCount &&
        outputs.normalization_layer_count == kImageNetworkNormalizationLayerCount &&
        outputs.cast_layer_count == kImageNetworkCastLayerCount &&
        outputs.matrix_multiply_layer_count == kImageNetworkMatrixMultiplyLayerCount &&
        outputs.softmax_layer_count == kImageNetworkSoftmaxLayerCount &&
        outputs.plugin_v3_layer_count == kImageNetworkPluginV3LayerCount &&
        outputs.attention_input_layer_count == kImageNetworkAttentionInputLayerCount &&
        outputs.attention_output_layer_count == kImageNetworkAttentionOutputLayerCount;
    if (!complete)
        throw Sam2EngineBuildError("SAM2 image graph did not satisfy its exact contract");

    Sam2SerializedPlan result;
    result.graph = {std::string(trtmc::sam2::kImagePlanSection),
                    Sam2GraphKind::kImage,
                    0,
                    network->getNbInputs(),
                    network->getNbOutputs(),
                    network->getNbLayers(),
                    outputs.referenced_tensor_count,
                    true};
    result.graph.convolution_layer_count = outputs.convolution_layer_count;
    result.graph.activation_layer_count = outputs.activation_layer_count;
    result.graph.pooling_layer_count = outputs.pooling_layer_count;
    result.graph.element_wise_layer_count = outputs.element_wise_layer_count;
    result.graph.shuffle_layer_count = outputs.shuffle_layer_count;
    result.graph.constant_layer_count = outputs.constant_layer_count;
    result.graph.slice_layer_count = outputs.slice_layer_count;
    result.graph.resize_layer_count = outputs.resize_layer_count;
    result.graph.normalization_layer_count = outputs.normalization_layer_count;
    result.graph.cast_layer_count = outputs.cast_layer_count;
    result.graph.matrix_multiply_layer_count = outputs.matrix_multiply_layer_count;
    result.graph.softmax_layer_count = outputs.softmax_layer_count;
    result.graph.plugin_v3_layer_count = outputs.plugin_v3_layer_count;
    result.graph.attention_input_layer_count = outputs.attention_input_layer_count;
    result.graph.attention_output_layer_count = outputs.attention_output_layer_count;
    result.bytes = serializeNetwork(builder, *network, workspace_bytes, result.graph.section);
    return result;
}

Sam2SerializedPlan buildTrackerPlan(nvinfer1::IBuilder& builder, const CheckpointReader& checkpoint,
                                    std::uint64_t workspace_bytes, std::int32_t history_frames) {
    auto network = createNetwork(builder);
    Sam2TrackerNetworkBuilder graph_builder(*network, checkpoint);
    const bool prompt = history_frames == 0;
    const Sam2TrackerNetworkOutputs outputs =
        prompt ? graph_builder.buildPrompt() : graph_builder.buildRecurrent(history_frames);
    const auto spec = prompt ? promptTrackerPlanSpec() : recurrentTrackerPlanSpec(history_frames);
    const bool complete =
        trackerOutputsComplete(outputs) &&
        network->getNbInputs() == static_cast<std::int32_t>(spec.inputs.size()) &&
        network->getNbOutputs() == static_cast<std::int32_t>(spec.outputs.size()) &&
        outputs.added_layer_count == network->getNbLayers() && outputs.referenced_tensor_count > 0U;
    if (!complete)
        throw Sam2EngineBuildError("SAM2 tracker graph did not satisfy its exact contract: " +
                                   std::string(spec.plan_section));

    Sam2SerializedPlan result;
    result.graph = {std::string(spec.plan_section),
                    prompt ? Sam2GraphKind::kPrompt : Sam2GraphKind::kRecurrent,
                    history_frames,
                    network->getNbInputs(),
                    network->getNbOutputs(),
                    network->getNbLayers(),
                    outputs.referenced_tensor_count,
                    true};
    result.bytes = serializeNetwork(builder, *network, workspace_bytes, result.graph.section);
    return result;
}

Sam2CompilationResult compilePlans(const Sam2EngineBuildOptions& options,
                                   const CheckpointReader& checkpoint) {
    if (!trackerGraphEmissionComplete()) {
        throw Sam2EngineBuildError(
            "SAM2 tracker graph emission is not complete; refusing to serialize partial plans");
    }
    Sam2CompilationResult result;
    result.runtime = inspectRuntime(options.gpu_device);
    result.plan_profiling_verbosity = std::string(trtmc::sam2::kPlanProfilingVerbosity);
    validateSam2RuntimeBuildFacts(result.runtime);
    Sam2Logger logger;
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
        throw Sam2EngineBuildError("TensorRT failed to create the SAM2 builder");

    result.plans.reserve(trtmc::sam2::kRequiredPlanSections.size());
    result.plans.push_back(buildImagePlan(*builder, checkpoint, options.workspace_bytes));
    result.plans.push_back(buildTrackerPlan(*builder, checkpoint, options.workspace_bytes, 0));
    for (std::int32_t history_frames = 1; history_frames <= 4; ++history_frames) {
        result.plans.push_back(
            buildTrackerPlan(*builder, checkpoint, options.workspace_bytes, history_frames));
    }
    validateSam2Compilation(result);
    return result;
}

} // namespace

Sam2NativeBundleBuildResult buildSam2NativeBundle(const Sam2EngineBuildOptions& options) {
    validateSam2EngineBuildOptions(options);
    verifySam2SourceConfig(options.source_config_path);
    CheckpointReader checkpoint = CheckpointReader::open(options.checkpoint_path);
    Sam2CompilationResult compilation = compilePlans(options, checkpoint);
    return detail::writeCompiledSam2NativeBundle(options, compilation);
}

} // namespace trtmc::sam2::native
