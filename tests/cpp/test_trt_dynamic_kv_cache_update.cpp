/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Negative TensorRT 11.2 qualification for runtime-sized native KV memory.
//
// This deliberately uses the built-in IKVCacheUpdateLayer rather than a
// PluginV3 layer. TensorRT currently exports engine I/O alias metadata only
// for the built-in KV-cache update node. The test proves all properties needed
// TensorRT rejects a dynamic cache sequence dimension before its output can
// feed AttentionV2. Keeping the attempted graph executable prevents the
// qualified path from silently regressing to a static-T IKVCacheUpdate layer.

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kHeads = 2;
constexpr int32_t kHeadDim = 64;
constexpr int32_t kMinCacheRows = 2;
constexpr int32_t kOptCacheRows = 4;
constexpr int32_t kMaxCacheRows = 8;
constexpr uint16_t kZeroBf16 = 0x0000U;
constexpr uint16_t kOneBf16 = 0x3F80U;
constexpr uint16_t kTwoBf16 = 0x4000U;
constexpr uint16_t kGuard = 0x55AAU;

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
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

struct TrtDeleter {
    template <typename T>
    void operator()(T* object) const noexcept {
        delete object;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter>;

bool set_profile_shape(nvinfer1::IOptimizationProfile& profile, char const* name,
                       nvinfer1::Dims const& minimum, nvinfer1::Dims const& optimum,
                       nvinfer1::Dims const& maximum) {
    return profile.setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minimum) &&
           profile.setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optimum) &&
           profile.setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maximum);
}

TrtPtr<nvinfer1::ICudaEngine> build_engine(TestLogger& logger) {
    TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) {
        return {};
    }

    uint32_t const flags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    TrtPtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flags)};
    TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!network || !config) {
        return {};
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    nvinfer1::Dims4 const dynamic_cache{kBatch, kHeads, -1, kHeadDim};
    nvinfer1::Dims4 const dynamic_query{kBatch, kHeads, -1, kHeadDim};
    auto* cache_k = network->addInput("cache_k", nvinfer1::DataType::kBF16, dynamic_cache);
    auto* cache_v = network->addInput("cache_v", nvinfer1::DataType::kBF16, dynamic_cache);
    auto* new_k = network->addInput("new_k", nvinfer1::DataType::kBF16, dynamic_query);
    auto* new_v = network->addInput("new_v", nvinfer1::DataType::kBF16, dynamic_query);
    auto* query = network->addInput("query", nvinfer1::DataType::kBF16, dynamic_query);
    auto* write_index =
        network->addInput("write_index", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {kBatch}});
    auto* active_length =
        network->addInput("active_length", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {kBatch}});
    if (!cache_k || !cache_v || !new_k || !new_v || !query || !write_index || !active_length) {
        return {};
    }

    auto* update_k =
        network->addKVCacheUpdate(*cache_k, *new_k, *write_index, nvinfer1::KVCacheMode::kLINEAR);
    auto* update_v =
        network->addKVCacheUpdate(*cache_v, *new_v, *write_index, nvinfer1::KVCacheMode::kLINEAR);
    check(update_k != nullptr, "add dynamic K cache update");
    check(update_v != nullptr, "add dynamic V cache update");
    if (!update_k || !update_v) {
        return {};
    }
    update_k->setName("dynamic_k_cache_update");
    update_v->setName("dynamic_v_cache_update");

    auto* present_k = update_k->getOutput(0);
    auto* present_v = update_v->getOutput(0);
    present_k->setName("present_k");
    present_v->setName("present_v");

    auto* attention = network->addAttentionV2(*query, *present_k, *present_v,
                                              nvinfer1::AttentionNormalizationOp::kSOFTMAX,
                                              nvinfer1::CausalMaskKind::kLOWER_RIGHT);
    check(attention == nullptr, "dynamic-T IKVCacheUpdate output is rejected before AttentionV2");
    if (!attention) {
        return {};
    }
    check(attention->getInput(1) == present_k, "updated K connects directly to AttentionV2");
    check(attention->getInput(2) == present_v, "updated V connects directly to AttentionV2");
    check(attention->setKeyValueLengths(active_length),
          "AttentionV2 accepts runtime key/value lengths");
    check(attention->setDecomposable(false), "AttentionV2 is explicitly non-decomposable");
    check(attention->getCausalKind() == nvinfer1::CausalMaskKind::kLOWER_RIGHT,
          "AttentionV2 uses lower-right causal mask");
    attention->setName("dynamic_kv_attention");

    network->markOutput(*present_k);
    network->markOutput(*present_v);
    auto* attention_output = attention->getOutput(0);
    attention_output->setName("attention_output");
    network->markOutput(*attention_output);

    auto* profile = builder->createOptimizationProfile();
    if (!profile) {
        return {};
    }
    nvinfer1::Dims4 const cache_min{kBatch, kHeads, kMinCacheRows, kHeadDim};
    nvinfer1::Dims4 const cache_opt{kBatch, kHeads, kOptCacheRows, kHeadDim};
    nvinfer1::Dims4 const cache_max{kBatch, kHeads, kMaxCacheRows, kHeadDim};
    for (char const* name : {"cache_k", "cache_v"}) {
        check(set_profile_shape(*profile, name, cache_min, cache_opt, cache_max),
              "set dynamic cache profile");
    }
    nvinfer1::Dims4 const update_min{kBatch, kHeads, 1, kHeadDim};
    nvinfer1::Dims4 const update_opt{kBatch, kHeads, 1, kHeadDim};
    nvinfer1::Dims4 const update_max{kBatch, kHeads, 2, kHeadDim};
    for (char const* name : {"new_k", "new_v", "query"}) {
        check(set_profile_shape(*profile, name, update_min, update_opt, update_max),
              "set dynamic query/update profile");
    }
    check(profile->isValid(), "dynamic T profile is valid");
    check(config->addOptimizationProfile(profile) >= 0, "dynamic T profile added");

    TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    check(plan != nullptr, "IKVCacheUpdate + AttentionV2 engine builds with dynamic T");
    if (!plan) {
        return {};
    }

    TrtPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        return {};
    }
    return TrtPtr<nvinfer1::ICudaEngine>{
        runtime->deserializeCudaEngine(plan->data(), plan->size())};
}

bool bind(nvinfer1::IExecutionContext& context, char const* name, void* address) {
    std::string message{"bind "};
    message += name;
    bool const result = context.setTensorAddress(name, address);
    check(result, message.c_str());
    return result;
}

void run_shape(nvinfer1::ICudaEngine& engine, int32_t cache_rows, int32_t query_rows,
               int32_t write_row) {
    TrtPtr<nvinfer1::IExecutionContext> context{engine.createExecutionContext()};
    check(context != nullptr, "create dynamic-T execution context");
    if (!context) {
        return;
    }

    nvinfer1::Dims4 const cache_shape{kBatch, kHeads, cache_rows, kHeadDim};
    nvinfer1::Dims4 const query_shape{kBatch, kHeads, query_rows, kHeadDim};
    for (char const* name : {"cache_k", "cache_v"}) {
        check(context->setInputShape(name, cache_shape), "set dynamic cache T");
    }
    for (char const* name : {"new_k", "new_v", "query"}) {
        check(context->setInputShape(name, query_shape), "set dynamic Sq");
    }
    check(context->allInputDimensionsSpecified(), "all dynamic input dimensions specified");
    check(context->getTensorShape("present_k").d[2] == cache_rows, "present K preserves runtime T");
    check(context->getTensorShape("present_v").d[2] == cache_rows, "present V preserves runtime T");

    size_t const row_elements = static_cast<size_t>(kHeads) * kHeadDim;
    size_t const cache_elements = static_cast<size_t>(cache_rows) * row_elements;
    size_t constexpr guard_elements = 256;
    size_t const cache_allocation_elements = cache_elements + guard_elements;
    size_t const update_elements = static_cast<size_t>(query_rows) * row_elements;

    std::vector<uint16_t> host_cache_k(cache_allocation_elements, kZeroBf16);
    std::vector<uint16_t> host_cache_v(cache_allocation_elements, kZeroBf16);
    std::fill(host_cache_k.begin() + cache_elements, host_cache_k.end(), kGuard);
    std::fill(host_cache_v.begin() + cache_elements, host_cache_v.end(), kGuard);
    std::vector<uint16_t> host_new_k(update_elements, kOneBf16);
    std::vector<uint16_t> host_new_v(update_elements, kTwoBf16);
    std::vector<uint16_t> host_query(update_elements, kZeroBf16);
    int32_t const active_length = write_row + query_rows;

    void* device_cache_k = nullptr;
    void* device_cache_v = nullptr;
    void* device_new_k = nullptr;
    void* device_new_v = nullptr;
    void* device_query = nullptr;
    void* device_write_index = nullptr;
    void* device_active_length = nullptr;
    void* device_attention_output = nullptr;
    cudaStream_t stream = nullptr;

    size_t const cache_bytes = cache_allocation_elements * sizeof(uint16_t);
    size_t const update_bytes = update_elements * sizeof(uint16_t);
    if (!cuda_ok(cudaStreamCreate(&stream), "create spike stream") ||
        !cuda_ok(cudaMalloc(&device_cache_k, cache_bytes), "allocate K cache with red zone") ||
        !cuda_ok(cudaMalloc(&device_cache_v, cache_bytes), "allocate V cache with red zone") ||
        !cuda_ok(cudaMalloc(&device_new_k, update_bytes), "allocate new K") ||
        !cuda_ok(cudaMalloc(&device_new_v, update_bytes), "allocate new V") ||
        !cuda_ok(cudaMalloc(&device_query, update_bytes), "allocate query") ||
        !cuda_ok(cudaMalloc(&device_write_index, sizeof(int32_t)), "allocate write index") ||
        !cuda_ok(cudaMalloc(&device_active_length, sizeof(int32_t)), "allocate active length") ||
        !cuda_ok(cudaMalloc(&device_attention_output, update_bytes), "allocate attention output")) {
        return;
    }

    cudaMemcpyAsync(device_cache_k, host_cache_k.data(), cache_bytes, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(device_cache_v, host_cache_v.data(), cache_bytes, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(device_new_k, host_new_k.data(), update_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_new_v, host_new_v.data(), update_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_query, host_query.data(), update_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_write_index, &write_row, sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(device_active_length, &active_length, sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);

    bind(*context, "cache_k", device_cache_k);
    bind(*context, "cache_v", device_cache_v);
    bind(*context, "new_k", device_new_k);
    bind(*context, "new_v", device_new_v);
    bind(*context, "query", device_query);
    bind(*context, "write_index", device_write_index);
    bind(*context, "active_length", device_active_length);
    bind(*context, "present_k", device_cache_k);
    bind(*context, "present_v", device_cache_v);
    bind(*context, "attention_output", device_attention_output);

    check(context->enqueueV3(stream),
          cache_rows == kMinCacheRows ? "enqueue minimum runtime T" : "enqueue maximum runtime T");
    cudaMemcpyAsync(host_cache_k.data(), device_cache_k, cache_bytes, cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(host_cache_v.data(), device_cache_v, cache_bytes, cudaMemcpyDeviceToHost,
                    stream);
    cuda_ok(cudaStreamSynchronize(stream), "synchronize spike");

    size_t const update_start = static_cast<size_t>(write_row) * row_elements;
    size_t const update_end = update_start + update_elements;
    bool k_rows_correct = true;
    bool v_rows_correct = true;
    for (size_t index = 0; index < cache_elements; ++index) {
        uint16_t const expected_k =
            index >= update_start && index < update_end ? kOneBf16 : kZeroBf16;
        uint16_t const expected_v =
            index >= update_start && index < update_end ? kTwoBf16 : kZeroBf16;
        k_rows_correct &= host_cache_k[index] == expected_k;
        v_rows_correct &= host_cache_v[index] == expected_v;
    }
    check(k_rows_correct, "K update touches only Sq rows");
    check(v_rows_correct, "V update touches only Sq rows");
    check(std::all_of(host_cache_k.begin() + cache_elements, host_cache_k.end(),
                      [](uint16_t value) { return value == kGuard; }),
          "K red zone remains intact");
    check(std::all_of(host_cache_v.begin() + cache_elements, host_cache_v.end(),
                      [](uint16_t value) { return value == kGuard; }),
          "V red zone remains intact");

    cudaFree(device_attention_output);
    cudaFree(device_active_length);
    cudaFree(device_write_index);
    cudaFree(device_query);
    cudaFree(device_new_v);
    cudaFree(device_new_k);
    cudaFree(device_cache_v);
    cudaFree(device_cache_k);
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    TestLogger logger;
    auto engine = build_engine(logger);
    check(engine == nullptr, "TRT 11.2 rejects dynamic-T IKVCacheUpdate graph");

    if (failures != 0) {
        std::cerr << failures << " dynamic IKVCacheUpdate spike test(s) failed\n";
        return 1;
    }
    std::cerr << "Dynamic IKVCacheUpdate rejection test passed\n";
    return 0;
}
