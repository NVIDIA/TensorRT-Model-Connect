/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/opt/kv_cache.h"
#include "runtime/models/opt/pipeline.h"

#include <NvInfer.h>
#include <cstring>
#include <cuda_runtime_api.h>

#if NV_TENSORRT_MAJOR >= 11
namespace {

int test_trt_external_alias() {
    trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return 1;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config)
        return 1;
    auto* cache =
        network->addInput("cache", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 4, 1});
    auto* update =
        network->addInput("update", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 2, 1});
    auto* index = network->addInput("index", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    if (!cache || !update || !index)
        return 1;
    auto* layer =
        network->addKVCacheUpdate(*cache, *update, *index, nvinfer1::KVCacheMode::kLINEAR);
    if (!layer)
        return 1;
    layer->getOutput(0)->setName("present");
    network->markOutput(*layer->getOutput(0));
    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!plan || !runtime)
        return 1;
    auto engine = trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    const char* alias = engine ? engine->getAliasedInputTensor("present") : nullptr;
    if (!alias || std::strcmp(alias, "cache") != 0)
        return 1;
    cudaStream_t stream = nullptr;
    void* storage = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMalloc(&storage, 4 * sizeof(float)) != cudaSuccess)
        return 1;
    float values[4]{1, 2, 3, 4};
    float replacement[2]{7, 8};
    int32_t write_index[1]{1};
    cudaMemcpy(storage, values, sizeof(values), cudaMemcpyHostToDevice);
    int failures = 0;
    {
        trtmc::TrtModuleImpl module(engine.get(), engine->createExecutionContext(), stream);
        module.bind_external("cache", storage);
        if (!module.ok() || module.device_ptr("cache") != storage ||
            module.device_ptr("present") != storage)
            ++failures;
        module.forward(
            {{"update", trtmc::Tensor{replacement, {1, 1, 2, 1}, trtmc::DType::kFloat32}},
             {"index", trtmc::Tensor{write_index, {1}, trtmc::DType::kInt32}}});
        cudaMemcpy(values, storage, sizeof(values), cudaMemcpyDeviceToHost);
        if (values[0] != 1 || values[1] != 7 || values[2] != 8 || values[3] != 4)
            ++failures;
    }
    cudaFree(storage);
    cudaStreamDestroy(stream);
    return failures;
}

} // namespace
#endif

int main() {
    int failures = trtmc::test::run_native_kv_contract_tests<
        trtmc::OptTextGenerationPipeline, trtmc::OptKvCache, trtmc::OptTextGenConfig>("Opt");
#if NV_TENSORRT_MAJOR >= 11
    failures += test_trt_external_alias();
#endif
    return failures;
}
