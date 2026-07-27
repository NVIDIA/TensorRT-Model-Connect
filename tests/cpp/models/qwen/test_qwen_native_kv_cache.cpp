/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../native_kv_cache_contract_test.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/models/qwen/kv_cache.h"

#include <NvInfer.h>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

#if NV_TENSORRT_MAJOR >= 11
namespace {

trtmc::TrtLogger& test_logger() {
    static trtmc::TrtLogger logger;
    return logger;
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_kv_cache_alias_engine() {
    auto builder =
        trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(test_logger()));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* cache =
        network->addInput("cache", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 4, 1});
    auto* update =
        network->addInput("update", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 2, 1});
    auto* write_indices =
        network->addInput("write_indices", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    if (!cache || !update || !write_indices)
        return nullptr;

    auto* cache_update =
        network->addKVCacheUpdate(*cache, *update, *write_indices, nvinfer1::KVCacheMode::kLINEAR);
    if (!cache_update)
        return nullptr;
    auto* present = cache_update->getOutput(0);
    present->setName("present");
    network->markOutput(*present);

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime =
        trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(test_logger()));
    if (!runtime)
        return nullptr;
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

trtmc::Tensor make_host_tensor(void* data, std::vector<int64_t> shape, trtmc::DType dtype) {
    trtmc::Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

int run_native_trt_alias_contract_test() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL [Qwen/TensorRT alias]: " << message << '\n';
            ++failures;
        }
    };

    auto engine = build_kv_cache_alias_engine();
    check(engine != nullptr, "KV alias engine built");
    if (!engine)
        return failures;
    const char* aliased_input = engine->getAliasedInputTensor("present");
    check(aliased_input != nullptr && std::strcmp(aliased_input, "cache") == 0,
          "TensorRT reports present aliases cache");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    void* external_from_input = nullptr;
    void* external_from_output = nullptr;
    void* external_prebound = nullptr;
    cudaMalloc(&external_from_input, 4 * sizeof(float));
    cudaMalloc(&external_from_output, 4 * sizeof(float));
    cudaMalloc(&external_prebound, 4 * sizeof(float));

    {
        auto* ctx = engine->createExecutionContext();
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
        check(module.ok(), "alias module is valid");
        check(module.device_ptr("cache") == nullptr, "alias input allocation is lazy");
        check(module.device_ptr("present") == nullptr, "alias output allocation is lazy");

        float cache[4] = {1.0F, 2.0F, 3.0F, 4.0F};
        float update[2] = {5.0F, 6.0F};
        int32_t write_index[1] = {1};
        auto outputs = module.forward(
            {{"cache", make_host_tensor(cache, {1, 1, 4, 1}, trtmc::DType::kFloat32)},
             {"update", make_host_tensor(update, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        check(module.device_ptr("cache") != nullptr, "alias input allocates on first use");
        check(module.device_ptr("cache") == module.device_ptr("present"),
              "alias input and output share internal storage");
        check(outputs.count("present") == 1, "internally-owned alias returns a host output");
        if (outputs.count("present") != 0) {
            const auto* values = static_cast<const float*>(outputs.at("present").data);
            check(values[0] == 1.0F && values[1] == 5.0F && values[2] == 6.0F && values[3] == 4.0F,
                  "internal alias updates only the selected rows");
        }

        float external_cache[4] = {10.0F, 20.0F, 30.0F, 40.0F};
        cudaMemcpy(external_from_input, external_cache, sizeof(external_cache),
                   cudaMemcpyHostToDevice);
        module.bind_external("cache", external_from_input);
        check(module.device_ptr("cache") == external_from_input &&
                  module.device_ptr("present") == external_from_input,
              "external input binding cascades to alias output");

        float update_from_input[2] = {7.0F, 8.0F};
        write_index[0] = 1;
        outputs = module.forward(
            {{"update", make_host_tensor(update_from_input, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        check(outputs.count("present") == 0, "externally-bound alias remains device-only");
        cudaMemcpy(external_cache, external_from_input, sizeof(external_cache),
                   cudaMemcpyDeviceToHost);
        check(external_cache[0] == 10.0F && external_cache[1] == 7.0F &&
                  external_cache[2] == 8.0F && external_cache[3] == 40.0F,
              "external input alias updates only the selected rows");

        float second_cache[4] = {1.0F, 2.0F, 3.0F, 4.0F};
        cudaMemcpy(external_from_output, second_cache, sizeof(second_cache),
                   cudaMemcpyHostToDevice);
        module.bind_external("present", external_from_output);
        check(module.device_ptr("present") == external_from_output &&
                  module.device_ptr("cache") == external_from_output,
              "external output binding cascades to alias input");

        float update_from_output[2] = {11.0F, 12.0F};
        write_index[0] = 2;
        module.forward_async(
            {{"update", make_host_tensor(update_from_output, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        module.sync();
        cudaMemcpy(second_cache, external_from_output, sizeof(second_cache),
                   cudaMemcpyDeviceToHost);
        check(second_cache[0] == 1.0F && second_cache[1] == 2.0F && second_cache[2] == 11.0F &&
                  second_cache[3] == 12.0F,
              "external output alias updates only the selected rows");
    }

    {
        float cache[4] = {21.0F, 22.0F, 23.0F, 24.0F};
        cudaMemcpy(external_prebound, cache, sizeof(cache), cudaMemcpyHostToDevice);
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"cache", external_prebound, sizeof(cache)}};
        auto* ctx = engine->createExecutionContext();
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(module.ok(), "prebound alias module is valid");
        check(module.device_ptr("cache") == external_prebound &&
                  module.device_ptr("present") == external_prebound,
              "constructor prebinding covers the full alias group");

        float update[2] = {31.0F, 32.0F};
        int32_t write_index[1] = {0};
        const auto outputs = module.forward(
            {{"update", make_host_tensor(update, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        check(outputs.count("present") == 0, "prebound alias remains device-only");
        cudaMemcpy(cache, external_prebound, sizeof(cache), cudaMemcpyDeviceToHost);
        check(cache[0] == 31.0F && cache[1] == 32.0F && cache[2] == 23.0F && cache[3] == 24.0F,
              "prebound alias updates only the selected rows");
    }

    cudaFree(external_from_input);
    cudaFree(external_from_output);
    cudaFree(external_prebound);
    cudaStreamDestroy(stream);
    return failures;
}

} // namespace
#endif

int main() {
    int failures =
        trtmc::test::run_native_kv_cache_contract_test<trtmc::QwenKvCache>(40960, "Qwen");
#if NV_TENSORRT_MAJOR >= 11
    failures += run_native_trt_alias_contract_test();
#endif
    return failures;
}
