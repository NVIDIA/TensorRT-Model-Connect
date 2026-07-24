/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression test for the runtime plugin registration path. This executable
// dlopens the TensorRT backend, but never links or dlopens trtmc_trt_plugins
// directly. The backend's DT_NEEDED edge must load the common plugin DSO and
// run its static registrars before engine deserialization.

#include "runtime/backend/runtime_memory_backend.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_backend.h"

#include <NvInfer.h>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <string>

namespace {

constexpr char kPluginName[] = "NativeContiguousAttention";
constexpr char kPluginVersion[] = "2";

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

TrtPtr<nvinfer1::IHostMemory> build_plugin_plan(TestLogger& logger) {
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
    config->setMemoryPoolLimit(
        // NativeContiguousAttention reserves a bounded 1 MiB cuDNN plan
        // workspace plus aligned sequence-length/query scratch. Leave enough
        // builder workspace for that declared runtime contract; this test is
        // about transitive registration/deserialization, not tactic-memory
        // starvation.
        nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 24);
    constexpr int32_t kNumQueryHeads = 1;
    constexpr int32_t kNumKvHeads = 1;
    constexpr int32_t kHeadDim = 4;
    constexpr int32_t kCacheRows = 2;
    constexpr int32_t kQueryRows = 1;

    auto* cache_k = network->addInput("cache_k", nvinfer1::DataType::kBF16,
                                      nvinfer1::Dims2{kCacheRows, kNumKvHeads * kHeadDim});
    auto* cache_v = network->addInput("cache_v", nvinfer1::DataType::kBF16,
                                      nvinfer1::Dims2{kCacheRows, kNumKvHeads * kHeadDim});
    auto* new_q = network->addInput("new_q", nvinfer1::DataType::kBF16,
                                    nvinfer1::Dims4{1, kNumQueryHeads, kQueryRows, kHeadDim});
    auto* new_k = network->addInput("new_k", nvinfer1::DataType::kBF16,
                                    nvinfer1::Dims4{1, kNumKvHeads, kQueryRows, kHeadDim});
    auto* new_v = network->addInput("new_v", nvinfer1::DataType::kBF16,
                                    nvinfer1::Dims4{1, kNumKvHeads, kQueryRows, kHeadDim});
    auto* history_length =
        network->addInput("history_length", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    if (!cache_k || !cache_v || !new_q || !new_k || !new_v || !history_length) {
        return {};
    }

    auto* creator_interface = getPluginRegistry()->getCreator(kPluginName, kPluginVersion, "");
    if (!creator_interface) {
        std::cerr << "NativeContiguousAttention creator was not registered "
                     "through the backend dependency\n";
        return {};
    }
    auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(creator_interface);

    int32_t abi_version = 2;
    int32_t num_query_heads = kNumQueryHeads;
    int32_t num_kv_heads = kNumKvHeads;
    int32_t head_dim = kHeadDim;
    int32_t chunk_limit = kQueryRows;
    nvinfer1::PluginField fields[] = {
        {"abi_version", &abi_version, nvinfer1::PluginFieldType::kINT32, 1},
        {"num_query_heads", &num_query_heads, nvinfer1::PluginFieldType::kINT32, 1},
        {"num_kv_heads", &num_kv_heads, nvinfer1::PluginFieldType::kINT32, 1},
        {"head_dim", &head_dim, nvinfer1::PluginFieldType::kINT32, 1},
        {"chunk_limit", &chunk_limit, nvinfer1::PluginFieldType::kINT32, 1},
    };
    nvinfer1::PluginFieldCollection field_collection{
        static_cast<int32_t>(sizeof(fields) / sizeof(fields[0])), fields};
    TrtPtr<nvinfer1::IPluginV3> plugin{creator->createPlugin(
        "runtime_registration_test", &field_collection, nvinfer1::TensorRTPhase::kBUILD)};
    if (!plugin) {
        return {};
    }

    nvinfer1::ITensor* plugin_inputs[] = {cache_k, cache_v, new_q, new_k, new_v, history_length};
    auto* layer = network->addPluginV3(
        plugin_inputs, static_cast<int32_t>(sizeof(plugin_inputs) / sizeof(plugin_inputs[0])),
        nullptr, 0, *plugin);
    if (!layer) {
        return {};
    }
    auto* context = layer->getOutput(0);
    context->setName("context");
    network->markOutput(*context);

    return TrtPtr<nvinfer1::IHostMemory>{builder->buildSerializedNetwork(*network, *config)};
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " /path/to/libtrtmc_backend_trt.so\n";
        return 1;
    }
    // Keep the product core as a real process dependency. Its direct
    // CUDA/NVRTC DT_NEEDED edges must be loaded before the dlopened backend
    // and its Torch-adjacent dependencies.
    const char* core_version = trtmc_version();
    if (core_version == nullptr || *core_version == '\0') {
        std::cerr << "FAIL: product core did not load\n";
        return 1;
    }

    void* backend_handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!backend_handle) {
        std::cerr << "FAIL: could not dlopen backend: " << dlerror() << '\n';
        return 1;
    }
    auto query_backend_abi = reinterpret_cast<trtmc::BackendDsoAbiQueryFnV2>(
        dlsym(backend_handle, trtmc::kBackendDsoAbiQuerySymbolV2));
    auto create_backend =
        reinterpret_cast<trtmc::IBackend* (*)()>(dlsym(backend_handle, "trtmc_create_backend"));
    auto destroy_backend = reinterpret_cast<void (*)(trtmc::IBackend*)>(
        dlsym(backend_handle, "trtmc_destroy_backend"));
    auto runtime_stack = reinterpret_cast<const char* (*)()>(
        dlsym(backend_handle, "trtmc_backend_runtime_memory_stack_json_v1"));
    if (!query_backend_abi || !create_backend || !destroy_backend || !runtime_stack) {
        std::cerr << "FAIL: backend C ABI symbols are missing\n";
        dlclose(backend_handle);
        return 1;
    }
    trtmc::BackendDsoAbiContractV2 backend_abi{};
    const auto expected_backend_abi = trtmc::make_runtime_memory_backend_dso_abi_contract_v2(
        trtmc::kBackendDsoCapabilityRuntimeMemoryV2);
    if (query_backend_abi(&backend_abi, sizeof(backend_abi)) != 0 ||
        backend_abi.struct_size != expected_backend_abi.struct_size ||
        backend_abi.contract_version != expected_backend_abi.contract_version ||
        backend_abi.interface_fingerprint != expected_backend_abi.interface_fingerprint ||
        backend_abi.runtime_memory_layout_fingerprint !=
            expected_backend_abi.runtime_memory_layout_fingerprint ||
        backend_abi.runtime_memory_api_version != trtmc::kRuntimeMemoryBackendApiVersionCurrent ||
        backend_abi.capability_flags != trtmc::kBackendDsoCapabilityRuntimeMemoryV2) {
        std::cerr << "FAIL: backend/core ABI contract is incompatible\n";
        dlclose(backend_handle);
        return 1;
    }
    const char* stack_text = runtime_stack();
    const std::string stack = stack_text != nullptr ? stack_text : "";
    for (const char* field :
         {"\"sm\":\"sm", "\"tensorrt\":\"", "\"cuda_runtime\":\"", "\"cudnn_backend\":\"",
          "\"cudnn_frontend_revision\":\"", "\"nvrtc\":\"", "\"driver\":\""}) {
        if (stack.find(field) == std::string::npos ||
            stack.find("\":\"unavailable\"") != std::string::npos) {
            std::cerr << "FAIL: backend runtime-stack evidence is incomplete: " << stack << '\n';
            dlclose(backend_handle);
            return 1;
        }
    }
    if (stack.find("\"cuda_runtime\":\"13.3\"") == std::string::npos ||
        stack.find("\"nvrtc\":\"13.3\"") == std::string::npos) {
        std::cerr << "FAIL: backend did not inherit the core CUDA 13.3/NVRTC "
                     "13.3 coherent stack: "
                  << stack << '\n';
        dlclose(backend_handle);
        return 1;
    }

    TestLogger logger;
    auto plan = build_plugin_plan(logger);
    if (!plan) {
        std::cerr << "FAIL: could not build plugin plan\n";
        return 1;
    }

    std::unique_ptr<trtmc::IBackend, void (*)(trtmc::IBackend*)> backend{create_backend(),
                                                                         destroy_backend};
    if (!backend) {
        std::cerr << "FAIL: could not create TensorRT backend\n";
        return 1;
    }

    auto module = backend->create_module(plan->data(), plan->size(), trtmc::ModuleCreateOptions{});
    if (!module) {
        std::cerr << "FAIL: backend did not deserialize plugin plan\n";
        return 1;
    }

    module.reset();
    backend.reset();
    // Keep the dependency chain loaded until process exit because TensorRT's
    // process-wide registry retains plugin creator pointers.
    std::cout << "PASS: backend dependency registered "
                 "NativeContiguousAttention and deserialized its plan\n";
    return 0;
}
