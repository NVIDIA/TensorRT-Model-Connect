/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-ENG-CPP-02
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         TrtModule forward, async forward, introspection, device_ptr, bind_external,
//                 move assignment operator=, keep_alive, forward_device (CPU and D2D paths)
// Preconditions:  TRT + CUDA GPU available, identity engine built in-process
// Postconditions: Forward produces correct output, introspection matches engine, bindings work;
//                 move assignment transfers ownership correctly; keep_alive stores shared_ptr;
//                 forward_device returns DeviceTensorMap; D2D copy path exercised
// =============================================================================

// =============================================================================
// Test suite: TrtModule — model.forward() abstraction for TensorRT
// =============================================================================
//
// Builds a tiny identity TRT engine (input → output copy), then validates:
// - forward() with CPU tensors (H2D + execute + D2H)
// - forward_async() + sync()
// - input_info() / output_info() introspection
// - has_input() / has_output()
// - device_ptr() access
// - bind_external() for KvCache-style external buffers
//
// Requires TRT + CUDA GPU. Skips gracefully without TRT.
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "trtmc/runtime/tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <NvInfer.h>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Process-wide logger (TRT requires a single logger for all objects).
static trtmc::TrtLogger g_logger;

// Build a tiny TRT engine: identity mapping input[4] → output[4] (float32)
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Single input: "x" [4] float32
    auto* inp = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});
    if (!inp)
        return nullptr;

    // Identity layer
    auto* id_layer = network->addIdentity(*inp);
    if (!id_layer)
        return nullptr;
    auto* out = id_layer->getOutput(0);
    out->setName("y");
    network->markOutput(*out);

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!runtime)
        return nullptr;

    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_dynamic_identity_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    uint32_t flags = 0;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network =
        trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* inp = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
    if (!inp)
        return nullptr;

    auto profile = builder->createOptimizationProfile();
    if (!profile)
        return nullptr;
    profile->setDimensions("x", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 4});
    profile->setDimensions("x", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{2, 4});
    profile->setDimensions("x", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{3, 4});
    config->addOptimizationProfile(profile);

    auto* id_layer = network->addIdentity(*inp);
    if (!id_layer)
        return nullptr;
    auto* out = id_layer->getOutput(0);
    out->setName("y");
    network->markOutput(*out);

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!runtime)
        return nullptr;

    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

#if NV_TENSORRT_MAJOR >= 11
// Build a minimal native TensorRT KV-cache update engine. TensorRT reports
// "present" as an alias of "cache", which exercises the same runtime contract
// used by split prefill/decode model engines without depending on a model
// family implementation.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_kv_cache_alias_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
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
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!runtime)
        return nullptr;
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}
#endif

static void test_forward_cpu() {
    auto engine = build_identity_engine();
    check(engine != nullptr, "engine built");
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    check(module.ok(), "module is ok");

    // Create input tensor
    float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    trtmc::Tensor input_tensor;
    input_tensor.data = input_data;
    input_tensor.shape = {4};
    input_tensor.dtype = trtmc::DType::kFloat32;

    trtmc::TensorMap inputs;
    inputs["x"] = input_tensor;

    // Forward pass
    auto outputs = module.forward(inputs);

    check(outputs.count("y") == 1, "output 'y' exists");
    if (outputs.count("y")) {
        auto& y = outputs["y"];
        check(y.shape.size() == 1, "output shape has 1 dim");
        check(y.shape[0] == 4, "output shape[0] = 4");
        auto* out_data = static_cast<float*>(y.data);
        check(out_data[0] == 1.0f, "output[0] = 1.0");
        check(out_data[1] == 2.0f, "output[1] = 2.0");
        check(out_data[2] == 3.0f, "output[2] = 3.0");
        check(out_data[3] == 4.0f, "output[3] = 4.0");
    }

    cudaStreamDestroy(stream);
}

static void test_forward_async() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    float input_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    trtmc::Tensor input_tensor;
    input_tensor.data = input_data;
    input_tensor.shape = {4};
    input_tensor.dtype = trtmc::DType::kFloat32;

    trtmc::TensorMap inputs;
    inputs["x"] = input_tensor;

    // Async forward
    module.forward_async(inputs);
    module.sync();

    // Manual download from device_ptr
    auto* d_ptr = module.device_ptr("y");
    check(d_ptr != nullptr, "output device_ptr is not null");

    float result[4] = {0};
    cudaMemcpy(result, d_ptr, 16, cudaMemcpyDeviceToHost);
    check(result[0] == 10.0f, "async output[0] = 10.0");
    check(result[3] == 40.0f, "async output[3] = 40.0");

    cudaStreamDestroy(stream);
}

static void test_introspection() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    auto ins = module.input_info();
    check(ins.size() == 1, "1 input");
    if (!ins.empty()) {
        check(ins[0].name == "x", "input name = 'x'");
        check(ins[0].shape[0] == 4, "input shape[0] = 4");
        check(ins[0].is_input == true, "is_input = true");
    }

    auto outs = module.output_info();
    check(outs.size() == 1, "1 output");
    if (!outs.empty()) {
        check(outs[0].name == "y", "output name = 'y'");
        check(outs[0].is_input == false, "is_input = false");
    }

    check(module.has_input("x"), "has_input('x')");
    check(!module.has_input("y"), "!has_input('y')");
    check(module.has_output("y"), "has_output('y')");
    check(!module.has_output("x"), "!has_output('x')");

    cudaStreamDestroy(stream);
}

static void test_device_ptr() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    check(module.device_ptr("x") != nullptr, "input device_ptr not null");
    check(module.device_ptr("y") != nullptr, "output device_ptr not null");
    check(module.device_ptr("nonexistent") == nullptr, "nonexistent returns null");

    cudaStreamDestroy(stream);
}

// NOTE: the pre-rebase TrtModule constructor accepted an `external_inputs`
// list that skipped allocation for those inputs. The post-abstraction
// TrtModuleImpl always allocates and lets `bind_external` swap the pointer
// (which frees the original buffer). `test_bind_external` below covers the
// post-rebase path; the explicit "skip at construction" test was removed
// when its premise no longer existed.

static void test_bind_external() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    // Allocate external buffer
    void* ext_ptr = nullptr;
    cudaMalloc(&ext_ptr, 16);
    check(ext_ptr != nullptr, "external alloc ok");

    // Upload data to external buffer
    float ext_data[4] = {100.0f, 200.0f, 300.0f, 400.0f};
    cudaMemcpy(ext_ptr, ext_data, 16, cudaMemcpyHostToDevice);

    // Bind external buffer as input
    auto* old_ptr = module.device_ptr("x");
    module.bind_external("x", ext_ptr);
    check(module.device_ptr("x") == ext_ptr, "device_ptr updated to external");
    check(module.device_ptr("x") != old_ptr, "device_ptr changed from original");

    // Forward should use the external buffer
    // We don't pass "x" in inputs — it's already bound
    trtmc::TensorMap empty_inputs;
    module.forward_async(empty_inputs);
    module.sync();

    float result[4] = {0};
    cudaMemcpy(result, module.device_ptr("y"), 16, cudaMemcpyDeviceToHost);
    check(result[0] == 100.0f, "external bind output[0] = 100.0");
    check(result[3] == 400.0f, "external bind output[3] = 400.0");

    cudaFree(ext_ptr);
    cudaStreamDestroy(stream);
}

#if NV_TENSORRT_MAJOR >= 11
static trtmc::Tensor make_host_tensor(void* data, std::vector<int64_t> shape, trtmc::DType dtype) {
    trtmc::Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

static void test_native_kv_cache_alias_binding() {
    auto engine = build_kv_cache_alias_engine();
    check(engine != nullptr, "KV alias engine built");
    if (!engine)
        return;
    const char* aliased_input = engine->getAliasedInputTensor("present");
    check(aliased_input != nullptr && std::strcmp(aliased_input, "cache") == 0,
          "TRT reports present aliases cache");

    cudaStream_t stream;
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
        check(module.ok(), "KV alias module is ok");

        // The aliased state is intentionally lazy: model runtimes can bind
        // their one persistent allocation before the first execution.
        check(module.device_ptr("cache") == nullptr, "KV alias input allocation is lazy");
        check(module.device_ptr("present") == nullptr, "KV alias output allocation is lazy");

        float cache[4] = {1.0F, 2.0F, 3.0F, 4.0F};
        float update[2] = {5.0F, 6.0F};
        int32_t write_index[1] = {1};
        auto outputs = module.forward(
            {{"cache", make_host_tensor(cache, {1, 1, 4, 1}, trtmc::DType::kFloat32)},
             {"update", make_host_tensor(update, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        check(module.device_ptr("cache") != nullptr, "KV alias input allocated on first use");
        check(module.device_ptr("cache") == module.device_ptr("present"),
              "KV alias input and output share internal address");
        check(outputs.count("present") == 1, "internally-owned alias is returned by host forward");
        if (outputs.count("present") != 0) {
            const auto* values = static_cast<const float*>(outputs.at("present").data);
            check(values[0] == 1.0F, "internal alias preserves cache[0]");
            check(values[1] == 5.0F, "internal alias updates cache[1]");
            check(values[2] == 6.0F, "internal alias updates cache[2]");
            check(values[3] == 4.0F, "internal alias preserves cache[3]");
        }

        float external_cache[4] = {10.0F, 20.0F, 30.0F, 40.0F};
        cudaMemcpy(external_from_input, external_cache, sizeof(external_cache),
                   cudaMemcpyHostToDevice);
        module.bind_external("cache", external_from_input);
        check(module.device_ptr("cache") == external_from_input,
              "external input binding updates alias input");
        check(module.device_ptr("present") == external_from_input,
              "external input binding cascades to alias output");

        float update_from_input[2] = {7.0F, 8.0F};
        write_index[0] = 1;
        outputs = module.forward(
            {{"update", make_host_tensor(update_from_input, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        check(outputs.count("present") == 0,
              "externally-bound alias stays device-only in host forward");
        cudaMemcpy(external_cache, external_from_input, sizeof(external_cache),
                   cudaMemcpyDeviceToHost);
        check(external_cache[0] == 10.0F, "external input alias preserves cache[0]");
        check(external_cache[1] == 7.0F, "external input alias updates cache[1]");
        check(external_cache[2] == 8.0F, "external input alias updates cache[2]");
        check(external_cache[3] == 40.0F, "external input alias preserves cache[3]");

        float second_cache[4] = {1.0F, 2.0F, 3.0F, 4.0F};
        cudaMemcpy(external_from_output, second_cache, sizeof(second_cache),
                   cudaMemcpyHostToDevice);
        module.bind_external("present", external_from_output);
        check(module.device_ptr("present") == external_from_output,
              "external output binding updates alias output");
        check(module.device_ptr("cache") == external_from_output,
              "external output binding cascades to alias input");

        float update_from_output[2] = {11.0F, 12.0F};
        write_index[0] = 2;
        module.forward_async(
            {{"update", make_host_tensor(update_from_output, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        module.sync();
        cudaMemcpy(second_cache, external_from_output, sizeof(second_cache),
                   cudaMemcpyDeviceToHost);
        check(second_cache[0] == 1.0F, "external output alias preserves cache[0]");
        check(second_cache[1] == 2.0F, "external output alias preserves cache[1]");
        check(second_cache[2] == 11.0F, "external output alias updates cache[2]");
        check(second_cache[3] == 12.0F, "external output alias updates cache[3]");
    }

    {
        float cache[4] = {21.0F, 22.0F, 23.0F, 24.0F};
        cudaMemcpy(external_prebound, cache, sizeof(cache), cudaMemcpyHostToDevice);
        std::vector<trtmc::ModuleExternalBinding> bindings{
            {"cache", external_prebound, sizeof(cache)}};
        auto* ctx = engine->createExecutionContext();
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(module.ok(), "alias input prebinding creates module");
        check(module.device_ptr("cache") == external_prebound,
              "input prebinding binds alias input");
        check(module.device_ptr("present") == external_prebound,
              "input prebinding cascades to alias output");

        float update[2] = {31.0F, 32.0F};
        int32_t write_index[1] = {0};
        auto outputs = module.forward(
            {{"update", make_host_tensor(update, {1, 1, 2, 1}, trtmc::DType::kFloat32)},
             {"write_indices", make_host_tensor(write_index, {1}, trtmc::DType::kInt32)}});
        check(outputs.count("present") == 0, "prebound alias remains device-only");
        cudaMemcpy(cache, external_prebound, sizeof(cache), cudaMemcpyDeviceToHost);
        check(cache[0] == 31.0F, "prebound alias updates cache[0]");
        check(cache[1] == 32.0F, "prebound alias updates cache[1]");
        check(cache[2] == 23.0F, "prebound alias preserves cache[2]");
        check(cache[3] == 24.0F, "prebound alias preserves cache[3]");
    }

    cudaFree(external_from_input);
    cudaFree(external_from_output);
    cudaFree(external_prebound);
    cudaStreamDestroy(stream);
}
#endif

static void test_unique_ptr_ownership() {
    // Modules now live behind unique_ptr<ITrtModule> — verify ownership transfer
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(), ctx, stream);
    check(module->ok(), "module is ok via unique_ptr");

    float data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    trtmc::Tensor t;
    t.data = data;
    t.shape = {4};
    t.dtype = trtmc::DType::kFloat32;
    auto out = module->forward({{"x", t}});
    check(out.count("y") == 1, "unique_ptr module forward works");

    // Transfer ownership
    std::unique_ptr<trtmc::ITrtModule> base = std::move(module);
    check(base->ok(), "ITrtModule base ptr works after move");

    cudaStreamDestroy(stream);
}

static void test_keep_alive() {
    // Covers TrtModule::keep_alive() — stores a shared_ptr to prevent resource release
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    // keep_alive with a trivial shared_ptr<void> resource
    module.keep_alive(std::make_shared<int>(42));
    check(module.ok(), "module ok after keep_alive");

    // Module still functions correctly after keep_alive
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    trtmc::Tensor t;
    t.data = data;
    t.shape = {4};
    t.dtype = trtmc::DType::kFloat32;
    auto out = module.forward({{"x", t}});
    check(out.count("y") == 1, "keep_alive: forward still works");

    cudaStreamDestroy(stream);
}

static void test_forward_device() {
    // Covers TrtModule::forward_device() with empty inputs
    // Exercises: forward_device_async (enqueue only), sync, output DeviceTensorMap
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    // forward_device with empty inputs: runs inference on pre-zeroed buffers,
    // returns a DeviceTensorMap of name->nullptr
    trtmc::DeviceTensorMap empty_inputs;
    auto out = module.forward_device(empty_inputs);

    // Should contain output "y" mapped to nullptr
    check(out.count("y") == 1, "forward_device: 'y' in output map");
    check(out["y"] == nullptr, "forward_device: output ptr is nullptr");

    // device_ptr("y") is still a valid GPU buffer
    check(module.device_ptr("y") != nullptr, "forward_device: device_ptr('y') valid");

    cudaStreamDestroy(stream);
}

static void test_profile_idx_default() {
    // Verify that explicit profile_idx=0 works identically to the default constructor
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0);
    check(module.ok(), "profile_idx=0: module is ok");
    check(module.profile_idx() == 0, "profile_idx=0: accessor returns 0");

    float input_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    trtmc::Tensor input_tensor;
    input_tensor.data = input_data;
    input_tensor.shape = {4};
    input_tensor.dtype = trtmc::DType::kFloat32;
    trtmc::TensorMap inputs;
    inputs["x"] = input_tensor;

    auto outputs = module.forward(inputs);
    check(outputs.count("y") == 1, "profile_idx=0: output exists");
    if (outputs.count("y")) {
        auto* out = static_cast<float*>(outputs["y"].data);
        check(out[0] == 5.0f, "profile_idx=0: output[0] = 5.0");
        check(out[3] == 8.0f, "profile_idx=0: output[3] = 8.0");
    }

    cudaStreamDestroy(stream);
}

static void test_profile_idx_invalid() {
    // Verify that an invalid profile index (engine has only 1 profile) fails gracefully
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Identity engine has 0 optimization profiles (static shapes), so profile 1 should fail
    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 1);
    check(!module.ok(), "profile_idx=1 on static engine: module should not be ok");

    cudaStreamDestroy(stream);
}

static void test_forward_device_with_input() {
    // Covers TrtModule::forward_device_async() body — D2D copy from DeviceTensor
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    // Create a DeviceTensor for input "x", upload data
    trtmc::DeviceTensor dt({4}, trtmc::DType::kFloat32, stream);
    check(dt.ok(), "DeviceTensor allocated");
    float host_data[4] = {7.0f, 8.0f, 9.0f, 10.0f};
    dt.copy_from_host(host_data);

    // forward_device_async with DeviceTensor input covers D2D copy path (lines 220-228)
    trtmc::DeviceTensorMap inputs;
    inputs["x"] = &dt;
    module.forward_device_async(inputs);
    module.sync();

    // Read output back from device_ptr
    float result[4] = {0};
    cudaMemcpy(result, module.device_ptr("y"), 16, cudaMemcpyDeviceToHost);
    check(result[0] == 7.0f, "forward_device_async D2D: output[0] = 7.0");
    check(result[3] == 10.0f, "forward_device_async D2D: output[3] = 10.0");

    cudaStreamDestroy(stream);
}

int main() {
    test_forward_cpu();
    test_forward_async();
    test_introspection();
    test_device_ptr();
    test_bind_external();
#if NV_TENSORRT_MAJOR >= 11
    test_native_kv_cache_alias_binding();
#endif
    test_unique_ptr_ownership();
    test_keep_alive();
    test_forward_device();
    test_forward_device_with_input();
    test_profile_idx_default();
    test_profile_idx_invalid();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
    }
    return failures;
}
