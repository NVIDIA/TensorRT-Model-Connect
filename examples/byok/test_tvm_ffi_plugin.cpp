/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// test_tvm_ffi_plugin.cpp — Full round-trip test for TVM-FFI kernel plugin
// =============================================================================
//
// Intent:
//   Validates TvmFfiKernelPlugin (IPluginV2DynamicExt) end-to-end: register
//   an external identity kernel DSO via the public BYOK API, build a TRT engine, run
//   inference, verify output, and test serialize/deserialize round-trip.
//
// Preconditions:
//   - TRTMC_HAS_TRT=1 and TRTMC_HAS_TVM_FFI=1
//   - GPU with CUDA runtime available
//
// Postconditions:
//   - Input [1,2,3,4] produces the same output
//   - Serialized engine produces identical results after deserialization
//
// Trace IDs: ARCH-TVM-FFI-001, UD-TVM-FFI-PLUGIN-001, UT-TVM-FFI-ROUNDTRIP-001
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include "trtmc/byok.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

extern "C" void tvm_ffi_plugin_force_link();

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class TestLogger : public nvinfer1::ILogger {
  public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR)
            std::cerr << "[TRT] " << msg << '\n';
    }
};

// ---------------------------------------------------------------------------
// Build engine with TvmFfiKernel plugin
// ---------------------------------------------------------------------------

static nvinfer1::ICudaEngine* build_engine(TestLogger& logger) {
    auto* builder = nvinfer1::createInferBuilder(logger);
    auto* network = builder->createNetworkV2(0);
    auto* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 26);

    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    std::string kn = "example.identity_copy";
    std::string ss =
        R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":"same_as_input_0","dtype":"float32"}],"workspace_bytes":0})";

    nvinfer1::PluginField fields[] = {
        {"kernel_name", kn.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(kn.size())},
        {"shape_spec", ss.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(ss.size())},
    };
    nvinfer1::PluginFieldCollection fc{2, fields};

    auto* registry = getPluginRegistry();
    // TRT 11 removed IPluginRegistry::getPluginCreator and replaced it with
    // getCreator returning IPluginCreatorInterface*; downcast to the still-
    // supported (deprecated) IPluginCreator so createPlugin() is callable.
    auto* creator =
        static_cast<nvinfer1::IPluginCreator*>(registry->getCreator("TvmFfiKernel", "1", ""));
    check(creator != nullptr, "creator found");
    if (!creator) {
        delete config;
        delete network;
        delete builder;
        return nullptr;
    }

    auto* plugin = creator->createPlugin("identity_copy", &fc);
    auto* layer = network->addPluginV2(&input, 1, *plugin);
    auto* output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    auto* plan = builder->buildSerializedNetwork(*network, *config);
    check(plan != nullptr, "engine built");
    if (!plan) {
        delete config;
        delete network;
        delete builder;
        return nullptr;
    }

    auto* runtime = nvinfer1::createInferRuntime(logger);
    auto* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    delete plan;
    delete config;
    delete network;
    delete builder;
    delete runtime;
    return engine;
}

static void test_declared_input_count_mismatch_is_rejected() {
    TestLogger logger;
    auto* builder = nvinfer1::createInferBuilder(logger);
    auto* network = builder->createNetworkV2(0);
    auto* config = builder->createBuilderConfig();
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    const std::string kernel_name = "example.identity_copy";
    const std::string shape_spec =
        R"({"num_inputs":2,"num_outputs":1,"outputs":[{"dims":"same_as_input_0","dtype":"float32"}],"workspace_bytes":0})";
    nvinfer1::PluginField fields[] = {
        {"kernel_name", kernel_name.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(kernel_name.size())},
        {"shape_spec", shape_spec.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(shape_spec.size())},
    };
    nvinfer1::PluginFieldCollection collection{2, fields};
    auto* creator = static_cast<nvinfer1::IPluginCreator*>(
        getPluginRegistry()->getCreator("TvmFfiKernel", "1", ""));
    auto* plugin = creator->createPlugin("invalid_count", &collection);
    auto* layer = network->addPluginV2(&input, 1, *plugin);
    network->markOutput(*layer->getOutput(0));
    auto* plan = builder->buildSerializedNetwork(*network, *config);
    check(plan == nullptr, "declared input count mismatch rejected");

    delete plan;
    delete config;
    delete network;
    delete builder;
}

// ---------------------------------------------------------------------------
// Run engine and verify output
// ---------------------------------------------------------------------------

static void run_and_verify(nvinfer1::ICudaEngine* engine, const char* name) {
    auto* ctx = engine->createExecutionContext();
    check(ctx != nullptr, (std::string(name) + " context").c_str());
    if (!ctx)
        return;

    float h_in[] = {1, 2, 3, 4}, h_out[4] = {};
    void *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, 16);
    cudaMalloc(&d_out, 16);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_in, h_in, 16, cudaMemcpyHostToDevice, stream);

    ctx->setTensorAddress("input", d_in);
    ctx->setTensorAddress("output", d_out);
    check(ctx->enqueueV3(stream), (std::string(name) + " enqueue").c_str());

    cudaMemcpyAsync(h_out, d_out, 16, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < 4; ++i)
        check(std::abs(h_out[i] - h_in[i]) < 1e-5f,
              (std::string(name) + " output[" + std::to_string(i) + "]").c_str());

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    delete ctx;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_roundtrip() {
    TestLogger logger;
    auto* engine = build_engine(logger);
    if (engine) {
        run_and_verify(engine, "roundtrip");
        delete engine;
    }
}

static void test_serialize_deserialize() {
    TestLogger logger;
    auto* engine = build_engine(logger);
    if (!engine)
        return;

    auto* plan = engine->serialize();
    check(plan != nullptr, "serialize");
    delete engine;
    if (!plan)
        return;

    auto* runtime = nvinfer1::createInferRuntime(logger);
    auto* engine2 = runtime->deserializeCudaEngine(plan->data(), plan->size());
    check(engine2 != nullptr, "deserialize");
    delete plan;
    delete runtime;
    if (!engine2)
        return;

    run_and_verify(engine2, "serde");
    delete engine2;
}

#endif

int main(int argc, char** argv) {
#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "test_byok_tvm_ffi: no CUDA device; skipping\n";
        return 77;
    }
    tvm_ffi_plugin_force_link();
    if (argc != 2) {
        std::cerr << "usage: test_byok_tvm_ffi IDENTITY_KERNEL.so\n";
        return 2;
    }
    trtmc::load_byok_kernel(argv[1], "run", "example.identity_copy");

    test_declared_input_count_mismatch_is_rejected();
    test_roundtrip();
    test_serialize_deserialize();

    if (failures > 0) {
        std::cerr << failures << " FAILED\n";
        return 1;
    }
    std::cerr << "BYOK identity kernel round-trip passed.\n";
#else
    std::cerr << "test_tvm_ffi_plugin: skipping (no TRT/TVM-FFI)\n";
#endif
    return 0;
}
