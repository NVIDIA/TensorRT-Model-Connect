/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Negative qualification test for the original alias-plugin proposal.
// TensorRT 11.2 accepts the BuildV2 alias declarations, but does not propagate
// PluginV3 aliases into serialized engine I/O metadata. This exact test keeps
// that limitation executable so the dynamic-memory path cannot accidentally
// regress to a same-address input/output binding.

#include "plugins/runtime_kv/native_kv_append_plugin.h"

#include <NvInfer.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, char const* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

bool cuda_ok(cudaError_t status, char const* name) {
    if (status != cudaSuccess) {
        std::cerr << "FAIL: " << name << ": " << cudaGetErrorString(status) << '\n';
        ++failures;
        return false;
    }
    return true;
}

class TestLogger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << message << '\n';
    }
};

struct EngineDeleter {
    template <typename T>
    void operator()(T* object) const noexcept {
        delete object;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, EngineDeleter>;

TrtPtr<nvinfer1::ICudaEngine> build_engine(TestLogger& logger) {
    TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder)
        return {};

    const uint32_t flags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    TrtPtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flags)};
    TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!network || !config)
        return {};

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 24);
    config->setPreviewFeature(nvinfer1::PreviewFeature::kALIASED_PLUGIN_IO_10_03, true);

    auto* cache_k =
        network->addInput("cache_k", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
    auto* cache_v =
        network->addInput("cache_v", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
    auto* new_k = network->addInput("new_k", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
    auto* new_v = network->addInput("new_v", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
    auto* write_index =
        network->addInput("write_index", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* active_length =
        network->addInput("active_length", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    if (!cache_k || !cache_v || !new_k || !new_v || !write_index || !active_length) {
        return {};
    }

    auto* profile = builder->createOptimizationProfile();
    if (!profile)
        return {};
    for (char const* name : {"cache_k", "cache_v"}) {
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 4});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{4, 4});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{8, 4});
    }
    for (char const* name : {"new_k", "new_v"}) {
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 4});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{2, 4});
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{4, 4});
    }
    config->addOptimizationProfile(profile);

    auto* creator_interface =
        getPluginRegistry()->getCreator(trtmc::runtime_kv::kNativeKvAppendPluginName,
                                        trtmc::runtime_kv::kNativeKvAppendPluginVersion, "");
    check(creator_interface != nullptr, "NativeKvAppend creator registered");
    if (!creator_interface)
        return {};
    auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(creator_interface);

    int32_t wrong_abi = trtmc::runtime_kv::kNativeKvAppendPluginAbi + 1;
    nvinfer1::PluginField wrong_field{"abi_version", &wrong_abi, nvinfer1::PluginFieldType::kINT32,
                                      1};
    nvinfer1::PluginFieldCollection wrong_fields{1, &wrong_field};
    check(creator->createPlugin("wrong_abi", &wrong_fields, nvinfer1::TensorRTPhase::kBUILD) ==
              nullptr,
          "creator rejects incompatible ABI");

    int32_t abi = trtmc::runtime_kv::kNativeKvAppendPluginAbi;
    nvinfer1::PluginField abi_field{"abi_version", &abi, nvinfer1::PluginFieldType::kINT32, 1};
    nvinfer1::PluginFieldCollection fields{1, &abi_field};
    TrtPtr<nvinfer1::IPluginV3> plugin{
        creator->createPlugin("native_kv_append", &fields, nvinfer1::TensorRTPhase::kBUILD)};
    check(plugin != nullptr, "creator returns ABI-v1 plugin");
    if (!plugin)
        return {};

    auto* build_capability = static_cast<nvinfer1::IPluginV3OneBuildV2*>(
        plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
    check(build_capability != nullptr, "BuildV2 capability present");
    if (!build_capability)
        return {};
    check(build_capability->getAliasedInput(0) == 0, "K output aliases K input");
    check(build_capability->getAliasedInput(1) == 1, "V output aliases V input");
    check(build_capability->getWorkspaceSize(nullptr, 0, nullptr, 0) == 0,
          "plugin workspace is zero");

    nvinfer1::ITensor* plugin_inputs[] = {cache_k, cache_v,     new_k,
                                          new_v,   write_index, active_length};
    auto* layer = network->addPluginV3(
        plugin_inputs, static_cast<int32_t>(std::size(plugin_inputs)), nullptr, 0, *plugin);
    check(layer != nullptr, "addPluginV3 succeeds");
    if (!layer)
        return {};
    layer->setName("NativeKvAppendV1");

    auto* present_k = layer->getOutput(0);
    auto* present_v = layer->getOutput(1);
    present_k->setName("present_k");
    present_v->setName("present_v");
    network->markOutput(*present_k);
    network->markOutput(*present_v);

    TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    check(plan != nullptr, "NativeKvAppend engine builds");
    if (!plan)
        return {};

    TrtPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime)
        return {};
    return TrtPtr<nvinfer1::ICudaEngine>{
        runtime->deserializeCudaEngine(plan->data(), plan->size())};
}

void run_update(nvinfer1::ICudaEngine& engine, int32_t write_index_value,
                int32_t active_length_value, bool expect_write) {
    constexpr int32_t bound_rows = 6;
    constexpr int32_t new_rows = 2;
    constexpr int32_t row_width = 4;

    TrtPtr<nvinfer1::IExecutionContext> context{engine.createExecutionContext()};
    check(context != nullptr, "execution context created");
    if (!context)
        return;
    check(context->setInputShape("cache_k", nvinfer1::Dims2{bound_rows, row_width}),
          "cache_k shape");
    check(context->setInputShape("cache_v", nvinfer1::Dims2{bound_rows, row_width}),
          "cache_v shape");
    check(context->setInputShape("new_k", nvinfer1::Dims2{new_rows, row_width}), "new_k shape");
    check(context->setInputShape("new_v", nvinfer1::Dims2{new_rows, row_width}), "new_v shape");

    std::vector<float> cache_k(bound_rows * row_width);
    std::vector<float> cache_v(bound_rows * row_width);
    std::vector<float> new_k(new_rows * row_width);
    std::vector<float> new_v(new_rows * row_width);
    for (size_t index = 0; index < cache_k.size(); ++index) {
        cache_k[index] = static_cast<float>(index);
        cache_v[index] = 100.0F + static_cast<float>(index);
    }
    for (size_t index = 0; index < new_k.size(); ++index) {
        new_k[index] = 1000.0F + static_cast<float>(index);
        new_v[index] = 2000.0F + static_cast<float>(index);
    }
    auto expected_k = cache_k;
    auto expected_v = cache_v;
    if (expect_write) {
        std::copy(new_k.begin(), new_k.end(), expected_k.begin() + write_index_value * row_width);
        std::copy(new_v.begin(), new_v.end(), expected_v.begin() + write_index_value * row_width);
    }

    void* device_cache_k = nullptr;
    void* device_cache_v = nullptr;
    void* device_new_k = nullptr;
    void* device_new_v = nullptr;
    void* device_write_index = nullptr;
    void* device_active_length = nullptr;
    cudaStream_t stream = nullptr;
    const size_t cache_bytes = cache_k.size() * sizeof(float);
    const size_t new_bytes = new_k.size() * sizeof(float);

    if (!cuda_ok(cudaStreamCreate(&stream), "create stream") ||
        !cuda_ok(cudaMalloc(&device_cache_k, cache_bytes), "alloc cache_k") ||
        !cuda_ok(cudaMalloc(&device_cache_v, cache_bytes), "alloc cache_v") ||
        !cuda_ok(cudaMalloc(&device_new_k, new_bytes), "alloc new_k") ||
        !cuda_ok(cudaMalloc(&device_new_v, new_bytes), "alloc new_v") ||
        !cuda_ok(cudaMalloc(&device_write_index, sizeof(int32_t)), "alloc H") ||
        !cuda_ok(cudaMalloc(&device_active_length, sizeof(int32_t)), "alloc A")) {
        return;
    }

    cudaMemcpyAsync(device_cache_k, cache_k.data(), cache_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_cache_v, cache_v.data(), cache_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_new_k, new_k.data(), new_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_new_v, new_v.data(), new_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_write_index, &write_index_value, sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(device_active_length, &active_length_value, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    check(context->setTensorAddress("cache_k", device_cache_k), "bind cache_k");
    check(context->setTensorAddress("cache_v", device_cache_v), "bind cache_v");
    check(context->setTensorAddress("new_k", device_new_k), "bind new_k");
    check(context->setTensorAddress("new_v", device_new_v), "bind new_v");
    check(context->setTensorAddress("write_index", device_write_index), "bind write_index");
    check(context->setTensorAddress("active_length", device_active_length), "bind active_length");
    check(context->setTensorAddress("present_k", device_cache_k), "bind aliased present_k");
    check(context->setTensorAddress("present_v", device_cache_v), "bind aliased present_v");
    check(context->enqueueV3(stream), "enqueue NativeKvAppend");

    cudaMemcpyAsync(cache_k.data(), device_cache_k, cache_bytes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cache_v.data(), device_cache_v, cache_bytes, cudaMemcpyDeviceToHost, stream);
    cuda_ok(cudaStreamSynchronize(stream), "synchronize");
    check(cache_k == expected_k,
          expect_write ? "K writes only requested rows" : "invalid K bounds write nothing");
    check(cache_v == expected_v,
          expect_write ? "V writes only requested rows" : "invalid V bounds write nothing");

    cudaFree(device_active_length);
    cudaFree(device_write_index);
    cudaFree(device_new_v);
    cudaFree(device_new_k);
    cudaFree(device_cache_v);
    cudaFree(device_cache_k);
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    trtmc_native_kv_append_fixture_force_link();

    TestLogger logger;
    auto engine = build_engine(logger);
    check(engine != nullptr, "engine deserializes");
    if (engine) {
        auto const* k_alias = engine->getAliasedInputTensor("present_k");
        auto const* v_alias = engine->getAliasedInputTensor("present_v");
        check(k_alias == nullptr, "TRT 11.2 omits PluginV3 K alias metadata");
        check(v_alias == nullptr, "TRT 11.2 omits PluginV3 V alias metadata");
    }

    if (failures != 0) {
        std::cerr << failures << " NativeKvAppend test(s) failed\n";
        return 1;
    }
    std::cerr << "NativeKvAppend plugin tests passed\n";
    return 0;
}
