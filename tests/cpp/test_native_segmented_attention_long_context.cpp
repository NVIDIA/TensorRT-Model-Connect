/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Long-history build/execute qualification for the selected segmented
// attention graph, including the fused-direct/cuDNN Sq=1 boundary. The
// dimensions match the Qwen and TinyLlama prototype targets.

#include "plugins/runtime_kv/cudnn_attention.h"
#include "plugins/runtime_kv/native_contiguous_attention_plugin.h"
#include "plugins/runtime_kv/runtime_kv_plugin_api.h"

#include <NvInfer.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int32_t kHq = 16;
constexpr int32_t kHkv = 8;
constexpr int32_t kD = 128;
constexpr int32_t kC = 16;
constexpr int32_t kOptT = 32768;
constexpr int32_t kMaxT = 40960;
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

uint16_t to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t const rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16U);
}

float from_bf16(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
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

struct RuntimeEngine {
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
};

struct BuildShape {
    int32_t query_heads;
    int32_t kv_heads;
    int32_t head_dim;
    int32_t chunk_limit;
    int32_t opt_history_rows;
    int32_t max_history_rows;
};

bool set_profile_shape(nvinfer1::IOptimizationProfile& profile, char const* name,
                       nvinfer1::Dims const& minimum, nvinfer1::Dims const& optimum,
                       nvinfer1::Dims const& maximum) {
    return profile.setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minimum) &&
           profile.setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optimum) &&
           profile.setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maximum);
}

RuntimeEngine build_engine(TestLogger& logger, BuildShape const& shape) {
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

    auto* history_k = network->addInput("history_k", nvinfer1::DataType::kBF16,
                                        nvinfer1::Dims2{-1, shape.kv_heads * shape.head_dim});
    auto* history_v = network->addInput("history_v", nvinfer1::DataType::kBF16,
                                        nvinfer1::Dims2{-1, shape.kv_heads * shape.head_dim});
    auto* query = network->addInput("query", nvinfer1::DataType::kBF16,
                                    nvinfer1::Dims4{1, shape.query_heads, -1, shape.head_dim});
    auto* current_k = network->addInput("current_k", nvinfer1::DataType::kBF16,
                                        nvinfer1::Dims4{1, shape.kv_heads, -1, shape.head_dim});
    auto* current_v = network->addInput("current_v", nvinfer1::DataType::kBF16,
                                        nvinfer1::Dims4{1, shape.kv_heads, -1, shape.head_dim});
    auto* history_length =
        network->addInput("history_length", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    if (!history_k || !history_v || !query || !current_k || !current_v || !history_length) {
        return {};
    }

    auto* creator_interface = getPluginRegistry()->getCreator(
        trtmc::runtime_kv::kNativeContiguousAttentionPluginName,
        trtmc::runtime_kv::kNativeContiguousAttentionPluginVersion, "");
    check(creator_interface != nullptr, "long-context creator registered");
    if (!creator_interface) {
        return {};
    }
    auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(creator_interface);
    int32_t abi = trtmc::runtime_kv::kNativeContiguousAttentionPluginAbi;
    int32_t hq = shape.query_heads;
    int32_t hkv = shape.kv_heads;
    int32_t head_dim = shape.head_dim;
    int32_t chunk_limit = shape.chunk_limit;
    nvinfer1::PluginField fields[] = {
        {"abi_version", &abi, nvinfer1::PluginFieldType::kINT32, 1},
        {"num_query_heads", &hq, nvinfer1::PluginFieldType::kINT32, 1},
        {"num_kv_heads", &hkv, nvinfer1::PluginFieldType::kINT32, 1},
        {"head_dim", &head_dim, nvinfer1::PluginFieldType::kINT32, 1},
        {"chunk_limit", &chunk_limit, nvinfer1::PluginFieldType::kINT32, 1},
    };
    nvinfer1::PluginFieldCollection collection{static_cast<int32_t>(std::size(fields)), fields};
    TrtPtr<nvinfer1::IPluginV3> plugin{creator->createPlugin(
        "long_segmented_attention", &collection, nvinfer1::TensorRTPhase::kBUILD)};
    if (!plugin) {
        return {};
    }
    nvinfer1::ITensor* inputs[] = {history_k, history_v, query,
                                   current_k, current_v, history_length};
    auto* layer =
        network->addPluginV3(inputs, static_cast<int32_t>(std::size(inputs)), nullptr, 0, *plugin);
    check(layer != nullptr, "add long-context plugin");
    if (!layer) {
        return {};
    }
    auto* context = layer->getOutput(0);
    context->setName("context");
    network->markOutput(*context);

    auto* profile = builder->createOptimizationProfile();
    if (!profile) {
        return {};
    }
    for (char const* name : {"history_k", "history_v"}) {
        check(set_profile_shape(
                  *profile, name, nvinfer1::Dims2{1, shape.kv_heads * shape.head_dim},
                  nvinfer1::Dims2{shape.opt_history_rows, shape.kv_heads * shape.head_dim},
                  nvinfer1::Dims2{shape.max_history_rows, shape.kv_heads * shape.head_dim}),
              "set long dynamic history profile");
    }
    check(set_profile_shape(
              *profile, "query", nvinfer1::Dims4{1, shape.query_heads, 1, shape.head_dim},
              nvinfer1::Dims4{1, shape.query_heads, 1, shape.head_dim},
              nvinfer1::Dims4{1, shape.query_heads, shape.chunk_limit, shape.head_dim}),
          "set long query profile");
    for (char const* name : {"current_k", "current_v"}) {
        check(set_profile_shape(
                  *profile, name, nvinfer1::Dims4{1, shape.kv_heads, 1, shape.head_dim},
                  nvinfer1::Dims4{1, shape.kv_heads, 1, shape.head_dim},
                  nvinfer1::Dims4{1, shape.kv_heads, shape.chunk_limit, shape.head_dim}),
              "set long current profile");
    }
    check(profile->isValid(), "long profile valid");
    check(config->addOptimizationProfile(profile) >= 0, "long profile added");

    TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    check(plan != nullptr, "long segmented engine builds");
    if (!plan) {
        return {};
    }
    RuntimeEngine result;
    result.runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!result.runtime) {
        return {};
    }
    result.engine.reset(result.runtime->deserializeCudaEngine(plan->data(), plan->size()));
    return result;
}

bool bind(nvinfer1::IExecutionContext& context, char const* name, void* address) {
    bool const result = context.setTensorAddress(name, address);
    if (!result) {
        std::cerr << "FAIL: bind " << name << '\n';
        ++failures;
    }
    return result;
}

void run_long_cases(nvinfer1::ICudaEngine& engine) {
    struct Case {
        int32_t tensor_rows;
        int32_t history_length;
        int32_t query_rows;
    };
    Case const cases[] = {
        {1, 0, 1}, {1, 0, kC}, {kOptT, kOptT, 1}, {kOptT, kOptT - kC, kC}, {kMaxT, kMaxT - 1, 1},
    };

    TrtPtr<nvinfer1::IExecutionContext> context{
        engine.createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED)};
    check(context != nullptr, "create USER_MANAGED long context");
    if (!context) {
        return;
    }

    size_t const cache_elements = static_cast<size_t>(kMaxT) * kHkv * kD;
    size_t const query_elements = static_cast<size_t>(kHq) * kC * kD;
    size_t const current_elements = static_cast<size_t>(kHkv) * kC * kD;
    std::vector<uint16_t> host_query(query_elements, to_bf16(0.03125F));
    std::vector<uint16_t> host_current_k(current_elements, to_bf16(0.03125F));
    std::vector<uint16_t> host_current_v(current_elements, to_bf16(0.125F));
    std::vector<uint16_t> host_context(query_elements);

    void* history_k = nullptr;
    void* history_v = nullptr;
    void* query = nullptr;
    void* current_k = nullptr;
    void* current_v = nullptr;
    void* history_length = nullptr;
    void* output = nullptr;
    void* context_memory = nullptr;
    size_t context_capacity = 0;
    cudaStream_t stream = nullptr;
    size_t const cache_bytes = cache_elements * sizeof(uint16_t);
    size_t const query_bytes = query_elements * sizeof(uint16_t);
    size_t const current_bytes = current_elements * sizeof(uint16_t);
    if (!cuda_ok(cudaStreamCreate(&stream), "create long stream") ||
        !cuda_ok(cudaMalloc(&history_k, cache_bytes), "allocate long K history") ||
        !cuda_ok(cudaMalloc(&history_v, cache_bytes), "allocate long V history") ||
        !cuda_ok(cudaMalloc(&query, query_bytes), "allocate long Q") ||
        !cuda_ok(cudaMalloc(&current_k, current_bytes), "allocate long current K") ||
        !cuda_ok(cudaMalloc(&current_v, current_bytes), "allocate long current V") ||
        !cuda_ok(cudaMalloc(&history_length, sizeof(int32_t)), "allocate long H") ||
        !cuda_ok(cudaMalloc(&output, query_bytes), "allocate long context output")) {
        return;
    }
    cudaMemsetAsync(history_k, 0, cache_bytes, stream);
    cudaMemsetAsync(history_v, 0, cache_bytes, stream);
    cudaMemcpyAsync(query, host_query.data(), query_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(current_k, host_current_k.data(), current_bytes, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(current_v, host_current_v.data(), current_bytes, cudaMemcpyHostToDevice,
                    stream);
    bind(*context, "history_k", history_k);
    bind(*context, "history_v", history_v);
    bind(*context, "query", query);
    bind(*context, "current_k", current_k);
    bind(*context, "current_v", current_v);
    bind(*context, "history_length", history_length);
    bind(*context, "context", output);

    for (auto const& item : cases) {
        check(
            context->setInputShape("history_k", nvinfer1::Dims2{item.tensor_rows, kHkv * kD}) &&
                context->setInputShape("history_v", nvinfer1::Dims2{item.tensor_rows, kHkv * kD}) &&
                context->setInputShape("query", nvinfer1::Dims4{1, kHq, item.query_rows, kD}) &&
                context->setInputShape("current_k",
                                       nvinfer1::Dims4{1, kHkv, item.query_rows, kD}) &&
                context->setInputShape("current_v", nvinfer1::Dims4{1, kHkv, item.query_rows, kD}),
            "set long runtime shapes");
        check(context->inferShapes(0, nullptr) == 0, "infer long runtime shapes");
        size_t const required_context = context->updateDeviceMemorySizeForShapes();
        if (required_context > context_capacity) {
            if (context_memory != nullptr) {
                cudaFree(context_memory);
                context_memory = nullptr;
            }
            if (required_context > 0) {
                check(cuda_ok(cudaMalloc(&context_memory, required_context),
                              "allocate long context memory"),
                      "long context allocation succeeds");
            }
            context_capacity = required_context;
        }
        context->setDeviceMemoryV2(context_memory, static_cast<int64_t>(context_capacity));
        cudaMemcpyAsync(history_length, &item.history_length, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        check(context->enqueueV3(stream), "execute first long segmented invocation");
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float first_ms = 0.0F;
        cudaEventElapsedTime(&first_ms, start, stop);
        for (int warmup = 0; warmup < 2; ++warmup) {
            check(context->enqueueV3(stream), "long segmented warmup");
        }
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < 5; ++iteration) {
            check(context->enqueueV3(stream), "long segmented benchmark");
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0F;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(stop);
        cudaEventDestroy(start);

        size_t const active_output_elements = static_cast<size_t>(kHq) * item.query_rows * kD;
        cudaMemcpyAsync(host_context.data(), output, active_output_elements * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, stream);
        cuda_ok(cudaStreamSynchronize(stream), "synchronize long context");
        bool finite = true;
        for (size_t index = 0; index < active_output_elements; ++index) {
            finite &= std::isfinite(from_bf16(host_context[index]));
        }
        check(finite, "long context output is finite");

        int32_t const padded_query = item.query_rows == 1 ? 1 : kC;
        size_t const plugin_workspace =
            trtmc::runtime_kv::cudnn_attention_workspace_size({kHq, kHkv, kD, kC}, padded_query);
        std::cerr << "long_segmented_case T=" << item.tensor_rows << " H=" << item.history_length
                  << " Sq=" << item.query_rows << " plugin_workspace_bytes=" << plugin_workspace
                  << " context_bytes=" << required_context << " first_us=" << first_ms * 1000.0F
                  << " avg_us=" << elapsed_ms * 200.0F << '\n';
    }

    if (context_memory != nullptr) {
        cudaFree(context_memory);
    }
    cudaFree(output);
    cudaFree(history_length);
    cudaFree(current_v);
    cudaFree(current_k);
    cudaFree(query);
    cudaFree(history_v);
    cudaFree(history_k);
    cudaStreamDestroy(stream);
}

void run_production_chunk_cases(nvinfer1::ICudaEngine& engine, BuildShape const& shape,
                                char const* label) {
    struct Case {
        int32_t tensor_rows;
        int32_t history_length;
        int32_t query_rows;
    };
    std::vector<Case> cases{
        {1, 0, 2},
        {2, 2, 2},
        {shape.max_history_rows, 2, 2},
    };
    // Cover both sides of every fused-direct tile/profile boundary with the
    // production Qwen and Tiny GQA geometries. T==2 is the required padded
    // extent for H==1; all other cases bind T==H.
    for (int32_t history_length : {1, 127, 128, 511, 512, 513}) {
        int32_t const tensor_rows = std::max(history_length, 2);
        if (tensor_rows <= shape.max_history_rows) {
            cases.push_back({tensor_rows, history_length, 1});
        }
    }
    if (shape.query_heads == 16 && shape.max_history_rows >= 1025) {
        // The former max/sum-exp optional-output graph selected an NVRTC plan
        // that failed only for Sq=1 at T<=1024. Keep both sides of that exact
        // boundary in the production Qwen geometry.
        cases.push_back({1024, 1023, 1});
        cases.push_back({1025, 1024, 1});
    }
    constexpr int32_t kProbeSq = 2;

    TrtPtr<nvinfer1::IExecutionContext> context{
        engine.createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED)};
    check(context != nullptr, "create production-C USER_MANAGED context");
    if (!context) {
        return;
    }

    size_t const history_elements =
        static_cast<size_t>(shape.max_history_rows) * shape.kv_heads * shape.head_dim;
    size_t const query_elements =
        static_cast<size_t>(shape.query_heads) * kProbeSq * shape.head_dim;
    size_t const current_elements = static_cast<size_t>(shape.kv_heads) * kProbeSq * shape.head_dim;
    constexpr size_t kGuardElements = 256;
    std::vector<uint16_t> host_history_k(history_elements + kGuardElements, kGuard);
    std::vector<uint16_t> host_history_v(history_elements + kGuardElements, kGuard);
    std::vector<uint16_t> returned_history_k(host_history_k.size());
    std::vector<uint16_t> returned_history_v(host_history_v.size());
    std::vector<uint16_t> host_query(query_elements);
    std::vector<uint16_t> host_current_k(current_elements);
    std::vector<uint16_t> host_current_v(current_elements);
    std::vector<uint16_t> host_context(query_elements + kGuardElements, kGuard);
    for (int32_t row = 0; row < shape.max_history_rows; ++row) {
        for (int32_t head = 0; head < shape.kv_heads; ++head) {
            for (int32_t dim = 0; dim < shape.head_dim; ++dim) {
                size_t const index =
                    (static_cast<size_t>(row) * shape.kv_heads + head) * shape.head_dim + dim;
                host_history_k[index] =
                    to_bf16(0.002F * static_cast<float>((row + 3 * head + dim) % 19 - 9));
                host_history_v[index] =
                    to_bf16(0.01F * static_cast<float>((2 * row + head + dim) % 23 - 11));
            }
        }
    }
    auto const original_history_k = host_history_k;
    auto const original_history_v = host_history_v;

    void* history_k = nullptr;
    void* history_v = nullptr;
    void* query = nullptr;
    void* current_k = nullptr;
    void* current_v = nullptr;
    void* history_length = nullptr;
    void* output = nullptr;
    void* context_memory = nullptr;
    size_t context_capacity = 0;
    size_t expected_decode_context_bytes = 0;
    size_t expected_prefill_context_bytes = 0;
    cudaStream_t stream = nullptr;
    size_t const history_bytes = host_history_k.size() * sizeof(uint16_t);
    size_t const query_bytes = query_elements * sizeof(uint16_t);
    size_t const current_bytes = current_elements * sizeof(uint16_t);
    size_t const output_bytes = host_context.size() * sizeof(uint16_t);
    if (!cuda_ok(cudaStreamCreate(&stream), "create production-C stream") ||
        !cuda_ok(cudaMalloc(&history_k, history_bytes), "allocate production-C K history") ||
        !cuda_ok(cudaMalloc(&history_v, history_bytes), "allocate production-C V history") ||
        !cuda_ok(cudaMalloc(&query, query_bytes), "allocate production-C query") ||
        !cuda_ok(cudaMalloc(&current_k, current_bytes), "allocate production-C current K") ||
        !cuda_ok(cudaMalloc(&current_v, current_bytes), "allocate production-C current V") ||
        !cuda_ok(cudaMalloc(&history_length, sizeof(int32_t)), "allocate production-C H") ||
        !cuda_ok(cudaMalloc(&output, output_bytes), "allocate production-C output")) {
        return;
    }
    cudaMemcpyAsync(history_k, host_history_k.data(), history_bytes, cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(history_v, host_history_v.data(), history_bytes, cudaMemcpyHostToDevice,
                    stream);
    bind(*context, "history_k", history_k);
    bind(*context, "history_v", history_v);
    bind(*context, "query", query);
    bind(*context, "current_k", current_k);
    bind(*context, "current_v", current_v);
    bind(*context, "history_length", history_length);
    bind(*context, "context", output);

    for (auto const& item : cases) {
        check(
            context->setInputShape(
                "history_k", nvinfer1::Dims2{item.tensor_rows, shape.kv_heads * shape.head_dim}) &&
                context->setInputShape(
                    "history_v",
                    nvinfer1::Dims2{item.tensor_rows, shape.kv_heads * shape.head_dim}) &&
                context->setInputShape("query", nvinfer1::Dims4{1, shape.query_heads,
                                                                item.query_rows, shape.head_dim}) &&
                context->setInputShape(
                    "current_k",
                    nvinfer1::Dims4{1, shape.kv_heads, item.query_rows, shape.head_dim}) &&
                context->setInputShape(
                    "current_v",
                    nvinfer1::Dims4{1, shape.kv_heads, item.query_rows, shape.head_dim}),
            "set production-C runtime shapes");
        check(context->inferShapes(0, nullptr) == 0, "infer production-C runtime shapes");
        size_t const required_context = context->updateDeviceMemorySizeForShapes();
        size_t& expected_context_bytes =
            item.query_rows == 1 ? expected_decode_context_bytes : expected_prefill_context_bytes;
        if (expected_context_bytes == 0) {
            expected_context_bytes = required_context;
        } else {
            check(required_context == expected_context_bytes,
                  "production-C context bytes are stable for one Sq across T");
        }
        if (required_context > context_capacity) {
            if (context_memory != nullptr) {
                cudaFree(context_memory);
                context_memory = nullptr;
            }
            if (required_context > 0) {
                cuda_ok(cudaMalloc(&context_memory, required_context),
                        "allocate production-C context memory");
            }
            context_capacity = required_context;
        }
        context->setDeviceMemoryV2(context_memory, static_cast<int64_t>(context_capacity));

        std::fill(host_query.begin(), host_query.end(), 0U);
        std::fill(host_current_k.begin(), host_current_k.end(), 0U);
        std::fill(host_current_v.begin(), host_current_v.end(), 0U);
        for (int32_t head = 0; head < shape.query_heads; ++head) {
            for (int32_t row = 0; row < item.query_rows; ++row) {
                for (int32_t dim = 0; dim < shape.head_dim; ++dim) {
                    size_t const index =
                        (static_cast<size_t>(head) * item.query_rows + row) * shape.head_dim + dim;
                    host_query[index] =
                        to_bf16(0.006F * static_cast<float>((head + 2 * row + dim) % 17 - 8));
                }
            }
        }
        for (int32_t head = 0; head < shape.kv_heads; ++head) {
            for (int32_t row = 0; row < item.query_rows; ++row) {
                for (int32_t dim = 0; dim < shape.head_dim; ++dim) {
                    size_t const index =
                        (static_cast<size_t>(head) * item.query_rows + row) * shape.head_dim + dim;
                    host_current_k[index] =
                        to_bf16(0.004F * static_cast<float>((3 * head + row + dim) % 13 - 6));
                    host_current_v[index] =
                        to_bf16(0.012F * static_cast<float>((head + 3 * row + dim) % 21 - 10));
                }
            }
        }
        size_t const active_query_elements =
            static_cast<size_t>(shape.query_heads) * item.query_rows * shape.head_dim;
        size_t const active_current_elements =
            static_cast<size_t>(shape.kv_heads) * item.query_rows * shape.head_dim;
        cudaMemcpyAsync(query, host_query.data(), active_query_elements * sizeof(uint16_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(current_k, host_current_k.data(),
                        active_current_elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(current_v, host_current_v.data(),
                        active_current_elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
        std::fill(host_context.begin(), host_context.end(), kGuard);
        cudaMemcpyAsync(output, host_context.data(), output_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(history_length, &item.history_length, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        check(context->enqueueV3(stream), "execute production-C probe");
        cudaMemcpyAsync(host_context.data(), output, output_bytes, cudaMemcpyDeviceToHost, stream);
        cuda_ok(cudaStreamSynchronize(stream), "synchronize production-C probe");
        check(std::all_of(host_context.begin(),
                          host_context.begin() + static_cast<std::ptrdiff_t>(active_query_elements),
                          [](uint16_t value) { return std::isfinite(from_bf16(value)); }),
              "production-C output is finite");
        check(std::all_of(host_context.begin() + static_cast<std::ptrdiff_t>(active_query_elements),
                          host_context.end(), [](uint16_t value) { return value == kGuard; }),
              "production-C exact-Sq output red zone is intact");

        if (item.query_rows == 1) {
            std::vector<float> logits(static_cast<size_t>(item.history_length) + 1U);
            float max_error = 0.0F;
            int32_t const query_group_size = shape.query_heads / shape.kv_heads;
            float const scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));
            for (int32_t query_head = 0; query_head < shape.query_heads; ++query_head) {
                int32_t const kv_head = query_head / query_group_size;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int32_t row = 0; row <= item.history_length; ++row) {
                    float dot = 0.0F;
                    for (int32_t dim = 0; dim < shape.head_dim; ++dim) {
                        size_t const query_index =
                            static_cast<size_t>(query_head) * shape.head_dim + dim;
                        size_t const key_index =
                            row == item.history_length
                                ? static_cast<size_t>(kv_head) * shape.head_dim + dim
                                : (static_cast<size_t>(row) * shape.kv_heads + kv_head) *
                                          shape.head_dim +
                                      dim;
                        auto const& key =
                            row == item.history_length ? host_current_k : host_history_k;
                        dot += from_bf16(host_query[query_index]) * from_bf16(key[key_index]);
                    }
                    logits[static_cast<size_t>(row)] = dot * scale;
                    maximum = std::max(maximum, dot * scale);
                }
                float denominator = 0.0F;
                for (float& logit : logits) {
                    logit = std::exp(logit - maximum);
                    denominator += logit;
                }
                for (int32_t dim = 0; dim < shape.head_dim; ++dim) {
                    float expected = 0.0F;
                    for (int32_t row = 0; row <= item.history_length; ++row) {
                        size_t const value_index =
                            row == item.history_length
                                ? static_cast<size_t>(kv_head) * shape.head_dim + dim
                                : (static_cast<size_t>(row) * shape.kv_heads + kv_head) *
                                          shape.head_dim +
                                      dim;
                        auto const& value =
                            row == item.history_length ? host_current_v : host_history_v;
                        expected +=
                            logits[static_cast<size_t>(row)] * from_bf16(value[value_index]);
                    }
                    expected /= denominator;
                    size_t const output_index =
                        static_cast<size_t>(query_head) * shape.head_dim + dim;
                    max_error = std::max(
                        max_error, std::abs(from_bf16(host_context[output_index]) - expected));
                }
            }
            check(max_error <= 0.015F,
                  "production direct/fallback decode matches stable full reference");
        }

        int32_t const padded_query_rows = item.query_rows == 1 ? 1 : shape.chunk_limit;
        size_t const plugin_workspace = trtmc::runtime_kv::cudnn_attention_workspace_size(
            {
                shape.query_heads,
                shape.kv_heads,
                shape.head_dim,
                shape.chunk_limit,
            },
            padded_query_rows);
        std::cerr << "production_chunk_case model=" << label << " C=" << shape.chunk_limit
                  << " T=" << item.tensor_rows << " H=" << item.history_length
                  << " Sq=" << item.query_rows << " plugin_workspace_bytes=" << plugin_workspace
                  << " context_bytes=" << required_context << '\n';
    }

    cudaMemcpyAsync(returned_history_k.data(), history_k, history_bytes, cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(returned_history_v.data(), history_v, history_bytes, cudaMemcpyDeviceToHost,
                    stream);
    cuda_ok(cudaStreamSynchronize(stream), "synchronize production history red zones");
    check(returned_history_k == original_history_k,
          "production fused/fallback decode leaves K history and red zone read-only");
    check(returned_history_v == original_history_v,
          "production fused/fallback decode leaves V history and red zone read-only");

    if (context_memory != nullptr) {
        cudaFree(context_memory);
    }
    cudaFree(output);
    cudaFree(history_length);
    cudaFree(current_v);
    cudaFree(current_k);
    cudaFree(query);
    cudaFree(history_v);
    cudaFree(history_k);
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    trtmc_runtime_kv_plugin_force_link();
    TestLogger logger;
    BuildShape const long_shape{kHq, kHkv, kD, kC, kOptT, kMaxT};
    auto runtime_engine = build_engine(logger, long_shape);
    check(runtime_engine.engine != nullptr, "deserialize long segmented engine");
    if (runtime_engine.engine) {
        std::cerr << "long_engine_static_context_bytes="
                  << runtime_engine.engine->getDeviceMemorySizeV2() << '\n';
        run_long_cases(*runtime_engine.engine);
    }
    BuildShape const qwen_production{16, 8, 128, 1024, 512, 1025};
    auto qwen_engine = build_engine(logger, qwen_production);
    check(qwen_engine.engine != nullptr, "deserialize Qwen production-C engine");
    if (qwen_engine.engine) {
        run_production_chunk_cases(*qwen_engine.engine, qwen_production, "Qwen3-0.6B");
    }
    BuildShape const tiny_production{32, 4, 64, 512, 512, 2048};
    auto tiny_engine = build_engine(logger, tiny_production);
    check(tiny_engine.engine != nullptr, "deserialize TinyLlama production-C engine");
    if (tiny_engine.engine) {
        run_production_chunk_cases(*tiny_engine.engine, tiny_production, "TinyLlama-1.1B");
    }
    if (failures != 0) {
        std::cerr << failures << " long segmented test(s) failed\n";
        return 1;
    }
    std::cerr << "Long segmented attention spike passed\n";
    return 0;
}
