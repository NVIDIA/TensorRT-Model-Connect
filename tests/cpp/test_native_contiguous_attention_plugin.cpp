/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Bounded qualification for segmented native attention. Persistent history is
// a read-only dynamic token-major input. Current K/V are exact-Sq engine
// outputs and regular plugin inputs; the runtime appends only those rows after
// enqueue. The plugin combines noncausal history SDPA and lower-right causal
// current SDPA from cuDNN log-sum-exp outputs.

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

static_assert(trtmc::runtime_kv::kCudnnAttentionControlScalarCount == 4,
              "workspace must reserve q/history/current/valid scalars");

constexpr int32_t kHq = 4;
constexpr int32_t kHkv = 2;
constexpr int32_t kD = 64;
constexpr int32_t kC = 4;
constexpr int32_t kMaxT = 8;
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

bool set_profile_shape(nvinfer1::IOptimizationProfile& profile, char const* name,
                       nvinfer1::Dims const& minimum, nvinfer1::Dims const& optimum,
                       nvinfer1::Dims const& maximum) {
    return profile.setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minimum) &&
           profile.setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optimum) &&
           profile.setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maximum);
}

nvinfer1::ITensor* token_rows_to_head_major(nvinfer1::INetworkDefinition& network,
                                            nvinfer1::ITensor& rows, char const* name) {
    auto* split_heads = network.addShuffle(rows);
    if (!split_heads) {
        return nullptr;
    }
    split_heads->setReshapeDimensions(nvinfer1::Dims3{-1, kHkv, kD});
    nvinfer1::Permutation transpose{};
    transpose.order[0] = 1;
    transpose.order[1] = 0;
    transpose.order[2] = 2;
    split_heads->setSecondTranspose(transpose);

    auto* add_batch = network.addShuffle(*split_heads->getOutput(0));
    if (!add_batch) {
        return nullptr;
    }
    add_batch->setReshapeDimensions(nvinfer1::Dims4{1, kHkv, -1, kD});
    add_batch->setName(name);
    return add_batch->getOutput(0);
}

RuntimeEngine build_engine(TestLogger& logger) {
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
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 25);

    auto* history_k =
        network->addInput("history_k", nvinfer1::DataType::kBF16, nvinfer1::Dims2{-1, kHkv * kD});
    auto* history_v =
        network->addInput("history_v", nvinfer1::DataType::kBF16, nvinfer1::Dims2{-1, kHkv * kD});
    auto* query =
        network->addInput("query", nvinfer1::DataType::kBF16, nvinfer1::Dims4{1, kHq, -1, kD});
    auto* current_k_rows = network->addInput("current_k_rows_input", nvinfer1::DataType::kBF16,
                                             nvinfer1::Dims2{-1, kHkv * kD});
    auto* current_v_rows = network->addInput("current_v_rows_input", nvinfer1::DataType::kBF16,
                                             nvinfer1::Dims2{-1, kHkv * kD});
    auto* history_length =
        network->addInput("history_length", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    if (!history_k || !history_v || !query || !current_k_rows || !current_v_rows ||
        !history_length) {
        return {};
    }

    // These identities model the pre-transpose K/V tensors produced inside a
    // real decoder. They are both consumed by attention and marked as exact-Sq
    // engine outputs for the runtime's O(Sq) append.
    auto* stage_k = network->addIdentity(*current_k_rows);
    auto* stage_v = network->addIdentity(*current_v_rows);
    if (!stage_k || !stage_v) {
        return {};
    }
    auto* staged_k = stage_k->getOutput(0);
    auto* staged_v = stage_v->getOutput(0);
    staged_k->setName("current_k_rows");
    staged_v->setName("current_v_rows");
    auto* current_k = token_rows_to_head_major(*network, *staged_k, "current_k_head_major");
    auto* current_v = token_rows_to_head_major(*network, *staged_v, "current_v_head_major");
    if (!current_k || !current_v) {
        return {};
    }

    auto* creator_interface = getPluginRegistry()->getCreator(
        trtmc::runtime_kv::kNativeContiguousAttentionPluginName,
        trtmc::runtime_kv::kNativeContiguousAttentionPluginVersion, "");
    check(creator_interface != nullptr, "NativeContiguousAttention creator registered");
    if (!creator_interface) {
        return {};
    }
    auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(creator_interface);
    int32_t abi = trtmc::runtime_kv::kNativeContiguousAttentionPluginAbi;
    int32_t hq = kHq;
    int32_t hkv = kHkv;
    int32_t head_dim = kD;
    int32_t chunk_limit = kC;
    nvinfer1::PluginField fields[] = {
        {"abi_version", &abi, nvinfer1::PluginFieldType::kINT32, 1},
        {"num_query_heads", &hq, nvinfer1::PluginFieldType::kINT32, 1},
        {"num_kv_heads", &hkv, nvinfer1::PluginFieldType::kINT32, 1},
        {"head_dim", &head_dim, nvinfer1::PluginFieldType::kINT32, 1},
        {"chunk_limit", &chunk_limit, nvinfer1::PluginFieldType::kINT32, 1},
    };
    nvinfer1::PluginFieldCollection field_collection{static_cast<int32_t>(std::size(fields)),
                                                     fields};
    TrtPtr<nvinfer1::IPluginV3> plugin{creator->createPlugin(
        "native_segmented_attention", &field_collection, nvinfer1::TensorRTPhase::kBUILD)};
    check(plugin != nullptr, "create segmented attention plugin");
    if (!plugin) {
        return {};
    }
    auto* build_capability = static_cast<nvinfer1::IPluginV3OneBuildV2*>(
        plugin->getCapabilityInterface(nvinfer1::PluginCapabilityType::kBUILD));
    check(build_capability != nullptr, "segmented BuildV2 capability present");
    if (!build_capability) {
        return {};
    }
    check(build_capability->getAliasedInput(0) == -1, "segmented context declares no alias");
    check(build_capability->getWorkspaceSize(nullptr, 0, nullptr, 0) > 0,
          "segmented workspace has a finite bound");

    nvinfer1::ITensor* plugin_inputs[] = {history_k, history_v, query,
                                          current_k, current_v, history_length};
    auto* attention = network->addPluginV3(
        plugin_inputs, static_cast<int32_t>(std::size(plugin_inputs)), nullptr, 0, *plugin);
    check(attention != nullptr, "add segmented PluginV3");
    if (!attention) {
        return {};
    }
    attention->setName("NativeSegmentedAttentionV2");
    auto* context = attention->getOutput(0);
    context->setName("context");
    network->markOutput(*context);
    network->markOutput(*staged_k);
    network->markOutput(*staged_v);

    auto* profile = builder->createOptimizationProfile();
    if (!profile) {
        return {};
    }
    for (char const* name : {"history_k", "history_v"}) {
        check(set_profile_shape(*profile, name, nvinfer1::Dims2{1, kHkv * kD},
                                nvinfer1::Dims2{4, kHkv * kD}, nvinfer1::Dims2{kMaxT, kHkv * kD}),
              "set dynamic history profile");
    }
    check(set_profile_shape(*profile, "query", nvinfer1::Dims4{1, kHq, 1, kD},
                            nvinfer1::Dims4{1, kHq, kC, kD}, nvinfer1::Dims4{1, kHq, kC, kD}),
          "set dynamic query profile");
    for (char const* name : {"current_k_rows_input", "current_v_rows_input"}) {
        check(set_profile_shape(*profile, name, nvinfer1::Dims2{1, kHkv * kD},
                                nvinfer1::Dims2{kC, kHkv * kD}, nvinfer1::Dims2{kC, kHkv * kD}),
              "set dynamic current-row profile");
    }
    check(profile->isValid(), "segmented profile valid");
    check(config->addOptimizationProfile(profile) >= 0, "segmented profile added");

    TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    check(plan != nullptr, "segmented dynamic-history engine builds");
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

size_t token_offset(int32_t head, int32_t row, int32_t dim) {
    return (static_cast<size_t>(row) * kHkv + head) * kD + dim;
}

size_t query_offset(int32_t head, int32_t row, int32_t dim, int32_t query_rows) {
    return (static_cast<size_t>(head) * query_rows + row) * kD + dim;
}

std::vector<float> reference_attention(std::vector<uint16_t> const& query,
                                       std::vector<uint16_t> const& history_k,
                                       std::vector<uint16_t> const& history_v,
                                       std::vector<uint16_t> const& current_k,
                                       std::vector<uint16_t> const& current_v,
                                       int32_t history_length, int32_t query_rows) {
    std::vector<float> result(static_cast<size_t>(kHq) * query_rows * kD);
    float const scale = 1.0F / std::sqrt(static_cast<float>(kD));
    int32_t const group_size = kHq / kHkv;
    for (int32_t qh = 0; qh < kHq; ++qh) {
        int32_t const kvh = qh / group_size;
        for (int32_t qi = 0; qi < query_rows; ++qi) {
            int32_t const visible = history_length + qi + 1;
            std::vector<float> scores(static_cast<size_t>(visible));
            float maximum = -std::numeric_limits<float>::infinity();
            for (int32_t key = 0; key < visible; ++key) {
                bool const from_history = key < history_length;
                int32_t const row = from_history ? key : key - history_length;
                auto const& keys = from_history ? history_k : current_k;
                float dot = 0.0F;
                for (int32_t dim = 0; dim < kD; ++dim) {
                    dot += from_bf16(query[query_offset(qh, qi, dim, query_rows)]) *
                           from_bf16(keys[token_offset(kvh, row, dim)]);
                }
                scores[static_cast<size_t>(key)] = dot * scale;
                maximum = std::max(maximum, dot * scale);
            }
            float denominator = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int32_t dim = 0; dim < kD; ++dim) {
                float value = 0.0F;
                for (int32_t key = 0; key < visible; ++key) {
                    bool const from_history = key < history_length;
                    int32_t const row = from_history ? key : key - history_length;
                    auto const& values = from_history ? history_v : current_v;
                    value += scores[static_cast<size_t>(key)] / denominator *
                             from_bf16(values[token_offset(kvh, row, dim)]);
                }
                result[query_offset(qh, qi, dim, query_rows)] = value;
            }
        }
    }
    return result;
}

bool bind(nvinfer1::IExecutionContext& context, char const* name, void* address) {
    bool const result = context.setTensorAddress(name, address);
    std::string message{"bind "};
    message += name;
    check(result, message.c_str());
    return result;
}

void run_sequence(nvinfer1::ICudaEngine& engine) {
    struct Step {
        int32_t query_rows;
        int32_t history_length;
        char const* name;
        bool append;
    };
    Step const steps[] = {
        {1, 0, "cold H=0 Sq=1", false},
        {2, 0, "cold H=0 Sq=2", true},
        {2, 2, "continuation", true},
        {1, 4, "decode", true},
    };

    TrtPtr<nvinfer1::IExecutionContext> context{engine.createExecutionContext()};
    check(context != nullptr, "create segmented execution context");
    if (!context) {
        return;
    }

    size_t const row_elements = static_cast<size_t>(kHkv) * kD;
    size_t const cache_elements = static_cast<size_t>(kMaxT) * row_elements;
    size_t const max_query_elements = static_cast<size_t>(kHq) * kC * kD;
    size_t const max_current_elements = static_cast<size_t>(kC) * row_elements;
    size_t constexpr guard_elements = 256;

    std::vector<uint16_t> expected_k(cache_elements + guard_elements, 0U);
    std::vector<uint16_t> expected_v(cache_elements + guard_elements, 0U);
    std::fill(expected_k.begin() + cache_elements, expected_k.end(), kGuard);
    std::fill(expected_v.begin() + cache_elements, expected_v.end(), kGuard);
    std::vector<uint16_t> actual_cache_k = expected_k;
    std::vector<uint16_t> actual_cache_v = expected_v;
    std::vector<uint16_t> query(max_query_elements);
    std::vector<uint16_t> current_k(max_current_elements);
    std::vector<uint16_t> current_v(max_current_elements);
    std::vector<uint16_t> staged_k(max_current_elements + guard_elements, kGuard);
    std::vector<uint16_t> staged_v(max_current_elements + guard_elements, kGuard);
    std::vector<uint16_t> actual_context(max_query_elements);

    void* device_cache_k = nullptr;
    void* device_cache_v = nullptr;
    void* device_query = nullptr;
    void* device_current_k_input = nullptr;
    void* device_current_v_input = nullptr;
    void* device_history_length = nullptr;
    void* device_staged_k = nullptr;
    void* device_staged_v = nullptr;
    void* device_context = nullptr;
    cudaStream_t stream = nullptr;
    size_t const cache_bytes = expected_k.size() * sizeof(uint16_t);
    size_t const query_bytes = query.size() * sizeof(uint16_t);
    size_t const current_bytes = current_k.size() * sizeof(uint16_t);
    size_t const staged_bytes = staged_k.size() * sizeof(uint16_t);
    if (!cuda_ok(cudaStreamCreate(&stream), "create segmented stream") ||
        !cuda_ok(cudaMalloc(&device_cache_k, cache_bytes), "allocate persistent K") ||
        !cuda_ok(cudaMalloc(&device_cache_v, cache_bytes), "allocate persistent V") ||
        !cuda_ok(cudaMalloc(&device_query, query_bytes), "allocate query") ||
        !cuda_ok(cudaMalloc(&device_current_k_input, current_bytes), "allocate current K input") ||
        !cuda_ok(cudaMalloc(&device_current_v_input, current_bytes), "allocate current V input") ||
        !cuda_ok(cudaMalloc(&device_history_length, sizeof(int32_t)), "allocate H") ||
        !cuda_ok(cudaMalloc(&device_staged_k, staged_bytes), "allocate exact-Sq K staging") ||
        !cuda_ok(cudaMalloc(&device_staged_v, staged_bytes), "allocate exact-Sq V staging") ||
        !cuda_ok(cudaMalloc(&device_context, query_bytes), "allocate context")) {
        return;
    }
    cudaMemcpyAsync(device_cache_k, expected_k.data(), cache_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_cache_v, expected_v.data(), cache_bytes, cudaMemcpyHostToDevice, stream);

    bind(*context, "history_k", device_cache_k);
    bind(*context, "history_v", device_cache_v);
    bind(*context, "query", device_query);
    bind(*context, "current_k_rows_input", device_current_k_input);
    bind(*context, "current_v_rows_input", device_current_v_input);
    bind(*context, "history_length", device_history_length);
    bind(*context, "current_k_rows", device_staged_k);
    bind(*context, "current_v_rows", device_staged_v);
    bind(*context, "context", device_context);

    for (size_t step_index = 0; step_index < std::size(steps); ++step_index) {
        Step const step = steps[step_index];
        int32_t const history_rows =
            step.history_length == 0 ? 1 : std::max(step.history_length, 2);
        check(context->setInputShape("history_k", nvinfer1::Dims2{history_rows, kHkv * kD}),
              "set dynamic K history");
        check(context->setInputShape("history_v", nvinfer1::Dims2{history_rows, kHkv * kD}),
              "set dynamic V history");
        check(context->setInputShape("query", nvinfer1::Dims4{1, kHq, step.query_rows, kD}),
              "set dynamic query");
        for (char const* name : {"current_k_rows_input", "current_v_rows_input"}) {
            check(context->setInputShape(name, nvinfer1::Dims2{step.query_rows, kHkv * kD}),
                  "set exact current rows");
        }
        check(context->allInputDimensionsSpecified(), "all segmented dimensions specified");
        check(context->getTensorShape("current_k_rows").d[0] == step.query_rows &&
                  context->getTensorShape("current_v_rows").d[0] == step.query_rows,
              "staging outputs follow exact Sq");
        check(context->getTensorShape("context").d[2] == step.query_rows,
              "context follows exact Sq");

        std::fill(query.begin(), query.end(), 0U);
        std::fill(current_k.begin(), current_k.end(), 0U);
        std::fill(current_v.begin(), current_v.end(), 0U);
        std::fill(staged_k.begin(), staged_k.end(), kGuard);
        std::fill(staged_v.begin(), staged_v.end(), kGuard);
        for (int32_t head = 0; head < kHq; ++head) {
            for (int32_t row = 0; row < step.query_rows; ++row) {
                for (int32_t dim = 0; dim < kD; ++dim) {
                    query[query_offset(head, row, dim, step.query_rows)] =
                        to_bf16(0.01F * (step_index + 1) * (head + 1) * (row + 1) *
                                (1.0F + 0.01F * (dim % 7)));
                }
            }
        }
        for (int32_t head = 0; head < kHkv; ++head) {
            for (int32_t row = 0; row < step.query_rows; ++row) {
                for (int32_t dim = 0; dim < kD; ++dim) {
                    size_t const index = token_offset(head, row, dim);
                    current_k[index] = to_bf16(0.02F * (step_index + 1) * (head + 1) * (row + 2) *
                                               (1.0F + 0.01F * (dim % 3)));
                    current_v[index] =
                        to_bf16(0.05F * (step_index + 1) * (head + 1) * (row + 2) + 0.001F * dim);
                }
            }
        }
        auto const expected_context =
            reference_attention(query, expected_k, expected_v, current_k, current_v,
                                step.history_length, step.query_rows);

        size_t const active_query_elements = static_cast<size_t>(kHq) * step.query_rows * kD;
        size_t const active_current_elements = static_cast<size_t>(step.query_rows) * row_elements;
        cudaMemcpyAsync(device_query, query.data(), active_query_elements * sizeof(uint16_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_current_k_input, current_k.data(),
                        active_current_elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_current_v_input, current_v.data(),
                        active_current_elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_history_length, &step.history_length, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_staged_k, staged_k.data(), staged_bytes, cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(device_staged_v, staged_v.data(), staged_bytes, cudaMemcpyHostToDevice,
                        stream);

        check(context->enqueueV3(stream), step.name);
        cudaMemcpyAsync(actual_cache_k.data(), device_cache_k, cache_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(actual_cache_v.data(), device_cache_v, cache_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(staged_k.data(), device_staged_k, staged_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(staged_v.data(), device_staged_v, staged_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(actual_context.data(), device_context,
                        active_query_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);
        cuda_ok(cudaStreamSynchronize(stream), "synchronize segmented enqueue");

        check(actual_cache_k == expected_k, "attention does not mutate K history");
        check(actual_cache_v == expected_v, "attention does not mutate V history");
        check(std::equal(current_k.begin(),
                         current_k.begin() + static_cast<std::ptrdiff_t>(active_current_elements),
                         staged_k.begin()),
              "exact-Sq K staging matches current rows");
        check(std::equal(current_v.begin(),
                         current_v.begin() + static_cast<std::ptrdiff_t>(active_current_elements),
                         staged_v.begin()),
              "exact-Sq V staging matches current rows");
        check(std::all_of(staged_k.begin() + static_cast<std::ptrdiff_t>(active_current_elements),
                          staged_k.end(), [](uint16_t value) { return value == kGuard; }),
              "K staging does not write profile-MAX tail");
        check(std::all_of(staged_v.begin() + static_cast<std::ptrdiff_t>(active_current_elements),
                          staged_v.end(), [](uint16_t value) { return value == kGuard; }),
              "V staging does not write profile-MAX tail");

        float max_error = 0.0F;
        for (size_t index = 0; index < active_query_elements; ++index) {
            max_error = std::max(
                max_error, std::abs(from_bf16(actual_context[index]) - expected_context[index]));
        }
        check(max_error <= 0.015F, "segmented context matches full reference");

        if (step.append) {
            size_t const append_offset =
                static_cast<size_t>(step.history_length) * row_elements * sizeof(uint16_t);
            size_t const append_bytes = active_current_elements * sizeof(uint16_t);
            cudaMemcpyAsync(static_cast<std::uint8_t*>(device_cache_k) + append_offset,
                            device_staged_k, append_bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(static_cast<std::uint8_t*>(device_cache_v) + append_offset,
                            device_staged_v, append_bytes, cudaMemcpyDeviceToDevice, stream);
            std::copy_n(current_k.begin(), active_current_elements,
                        expected_k.begin() +
                            static_cast<std::ptrdiff_t>(step.history_length * row_elements));
            std::copy_n(current_v.begin(), active_current_elements,
                        expected_v.begin() +
                            static_cast<std::ptrdiff_t>(step.history_length * row_elements));
            cudaMemcpyAsync(actual_cache_k.data(), device_cache_k, cache_bytes,
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(actual_cache_v.data(), device_cache_v, cache_bytes,
                            cudaMemcpyDeviceToHost, stream);
            cuda_ok(cudaStreamSynchronize(stream), "synchronize O(Sq) runtime append");
            check(actual_cache_k == expected_k, "runtime appends only staged K rows");
            check(actual_cache_v == expected_v, "runtime appends only staged V rows");
        }
    }

    // The host runtime is the primary admission gate, but the device scalar
    // validation must remain memory-safe if a caller bypasses it. T==1 is
    // reserved exclusively for H==0; non-cold requests require H>0 and T>=2.
    struct InvalidStep {
        int32_t history_rows;
        int32_t history_length;
        char const* name;
    };
    InvalidStep const invalid_steps[] = {
        {2, 0, "reject H=0 with T>1"},
        {1, 1, "reject H>0 with T=1 sentinel"},
    };
    for (auto const& invalid : invalid_steps) {
        check(
            context->setInputShape("history_k", nvinfer1::Dims2{invalid.history_rows, kHkv * kD}) &&
                context->setInputShape("history_v",
                                       nvinfer1::Dims2{invalid.history_rows, kHkv * kD}) &&
                context->setInputShape("query", nvinfer1::Dims4{1, kHq, 1, kD}) &&
                context->setInputShape("current_k_rows_input", nvinfer1::Dims2{1, kHkv * kD}) &&
                context->setInputShape("current_v_rows_input", nvinfer1::Dims2{1, kHkv * kD}),
            invalid.name);
        std::fill(actual_context.begin(), actual_context.end(), kGuard);
        cudaMemcpyAsync(device_context, actual_context.data(),
                        static_cast<size_t>(kHq) * kD * sizeof(uint16_t), cudaMemcpyHostToDevice,
                        stream);
        cudaMemcpyAsync(device_history_length, &invalid.history_length, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        check(context->enqueueV3(stream), invalid.name);
        cudaMemcpyAsync(actual_context.data(), device_context,
                        static_cast<size_t>(kHq) * kD * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(actual_cache_k.data(), device_cache_k, cache_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(actual_cache_v.data(), device_cache_v, cache_bytes, cudaMemcpyDeviceToHost,
                        stream);
        cuda_ok(cudaStreamSynchronize(stream), "synchronize invalid scalar defense");
        check(std::all_of(actual_context.begin(),
                          actual_context.begin() +
                              static_cast<std::ptrdiff_t>(static_cast<size_t>(kHq) * kD),
                          [](uint16_t value) { return value == 0U; }),
              "invalid H/T convention produces zero context");
        check(actual_cache_k == expected_k, "invalid H/T does not mutate K history");
        check(actual_cache_v == expected_v, "invalid H/T does not mutate V history");
    }

    check(std::all_of(actual_cache_k.begin() + cache_elements, actual_cache_k.end(),
                      [](uint16_t value) { return value == kGuard; }),
          "persistent K red zone intact");
    check(std::all_of(actual_cache_v.begin() + cache_elements, actual_cache_v.end(),
                      [](uint16_t value) { return value == kGuard; }),
          "persistent V red zone intact");

    // Small smoke microbenchmark. It is not a release threshold; it records
    // one history SDPA plus the fused current-token merge and exact-Sq staging.
    check(context->setInputShape("history_k", nvinfer1::Dims2{5, kHkv * kD}) &&
              context->setInputShape("history_v", nvinfer1::Dims2{5, kHkv * kD}) &&
              context->setInputShape("query", nvinfer1::Dims4{1, kHq, 1, kD}) &&
              context->setInputShape("current_k_rows_input", nvinfer1::Dims2{1, kHkv * kD}) &&
              context->setInputShape("current_v_rows_input", nvinfer1::Dims2{1, kHkv * kD}),
          "set segmented decode benchmark shape");
    int32_t const benchmark_history = 5;
    cudaMemcpyAsync(device_history_length, &benchmark_history, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cuda_ok(cudaEventCreate(&start), "create benchmark start") &&
        cuda_ok(cudaEventCreate(&stop), "create benchmark stop")) {
        for (int warmup = 0; warmup < 5; ++warmup) {
            check(context->enqueueV3(stream), "segmented benchmark warmup");
        }
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < 50; ++iteration) {
            check(context->enqueueV3(stream), "segmented benchmark iteration");
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0F;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        std::cerr << "segmented_decode_avg_us=" << elapsed_ms * 1000.0F / 50.0F << '\n';
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
    }

    cudaFree(device_context);
    cudaFree(device_staged_v);
    cudaFree(device_staged_k);
    cudaFree(device_history_length);
    cudaFree(device_current_v_input);
    cudaFree(device_current_k_input);
    cudaFree(device_query);
    cudaFree(device_cache_v);
    cudaFree(device_cache_k);
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    trtmc_runtime_kv_plugin_force_link();
    check((trtmc_runtime_kv_plugin_capabilities() &
           trtmc::runtime_kv::kRuntimeKvCapabilityCudnnSdpa) != 0,
          "qualified cuDNN SDPA capability exported");

    TestLogger logger;
    auto runtime_engine = build_engine(logger);
    auto& engine = runtime_engine.engine;
    check(engine != nullptr, "deserialize segmented dynamic-history engine");
    if (engine) {
        check(engine->getAliasedInputTensor("context") == nullptr, "context has no engine alias");
        check(engine->getAliasedInputTensor("current_k_rows") == nullptr &&
                  engine->getAliasedInputTensor("current_v_rows") == nullptr,
              "staging outputs have no cache alias");
        run_sequence(*engine);
    }

    if (failures != 0) {
        std::cerr << failures << " segmented attention test(s) failed\n";
        return 1;
    }
    std::cerr << "Segmented dynamic-history attention spike passed\n";
    return 0;
}
