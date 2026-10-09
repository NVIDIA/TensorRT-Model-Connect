/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "runtime/tensorrt/trt_module_impl.h"

#include <NvInfer.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity(bool dynamic) {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT,
                                    nvinfer1::Dims{1, {dynamic ? -1 : 4}});
    auto* identity = network->addIdentity(*input);
    identity->getOutput(0)->setName("output");
    network->markOutput(*identity->getOutput(0));

    if (dynamic) {
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {2}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {4}});
        config->addOptimizationProfile(profile);
    }

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

void test_first_forward_allocates_unbound_dynamic_input(nvinfer1::ICudaEngine& engine,
                                                        cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
    check(module.device_ptr("input") == nullptr,
          "dynamic input is not allocated during module construction");
    float values[] = {1.0F, 2.0F};
    const auto outputs =
        module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
    check(module.device_ptr("input") != nullptr,
          "first forward allocates and binds an unbound dynamic input");
    const auto found = outputs.find("output");
    check(found != outputs.end() && found->second.numel() == 2,
          "dynamic identity returns its runtime shape");
    if (found != outputs.end() && found->second.numel() == 2) {
        const auto* output = static_cast<const float*>(found->second.data);
        check(std::fabs(output[0] - 1.0F) < 1.0e-6F && std::fabs(output[1] - 2.0F) < 1.0e-6F,
              "dynamic identity returns its input values");
    }
}

void test_external_dynamic_input_stays_external(nvinfer1::ICudaEngine& engine,
                                                cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        check(module.device_ptr("input") == nullptr,
              "second dynamic input also starts without an allocation");
        module.bind_external("input", external, {2});
        float values[] = {3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
        check(module.device_ptr("input") == external,
              "forward preserves the pre-bound dynamic input buffer");
    }
    cudaFree(external);
}

void test_backend_managed_graph_does_not_enable_nested_capture(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr, {},
                                true);
    module.enable_cuda_graph();
    check(!module.cuda_graph_active(), "backend-managed graph leaves stream capture disabled");
}

void test_short_static_upload_does_not_keep_the_previous_input(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    for (bool graph : {false, true}) {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        if (graph)
            module.enable_cuda_graph();
        float full[] = {1.0F, 2.0F, 3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{full, {4}, trtmc::DType::kFloat32}}});
        float short_input[] = {7.0F, 8.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{short_input, {2}, trtmc::DType::kFloat32}}});
        const auto& tensor = outputs.at("output");
        const auto* values = static_cast<const float*>(tensor.data);
        check(tensor.numel() == 4 && values[0] == 7.0F && values[1] == 8.0F,
              "static input keeps the current uploaded prefix");
        check(values[2] == 0.0F && values[3] == 0.0F,
              "short static upload clears bytes from the previous longer request");
    }
}

void test_short_external_static_upload_preserves_its_tail(nvinfer1::ICudaEngine& engine,
                                                          cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        module.bind_external("input", external, {4});
        float full[] = {1.0F, 2.0F, 3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{full, {4}, trtmc::DType::kFloat32}}});
        float prefix[] = {7.0F, 8.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{prefix, {2}, trtmc::DType::kFloat32}}});
        const auto* values = static_cast<const float*>(outputs.at("output").data);
        check(values[0] == 7.0F && values[1] == 8.0F && values[2] == 3.0F && values[3] == 4.0F,
              "short upload preserves externally owned static state beyond its prefix");
        check(module.device_ptr("input") == external, "static input keeps external ownership");
    }
    cudaFree(external);
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    auto engine = build_identity(true);
    auto static_engine = build_identity(false);
    if (!engine || !static_engine)
        return 1;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;
    test_first_forward_allocates_unbound_dynamic_input(*engine, stream);
    test_external_dynamic_input_stays_external(*engine, stream);
    test_backend_managed_graph_does_not_enable_nested_capture(*engine, stream);
    test_short_static_upload_does_not_keep_the_previous_input(*static_engine, stream);
    test_short_external_static_upload_preserves_its_tail(*static_engine, stream);
    cudaStreamDestroy(stream);
    return failures;
}
