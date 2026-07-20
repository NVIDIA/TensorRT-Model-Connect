/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-WAN22-TRT-CPP-01
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         Wan2.2-owned TRT backend ABI, dtype, external-binding, dynamic-shape,
//                 CUDA-graph, host-input, and direct-device-input contracts
// Preconditions:  TRT + CUDA GPU available, identity engine built in-process
// Postconditions: Unsupported inputs fail closed; binding/shape failures are transactional;
//                 multi-profile and CUDA-graph execution remain correct; host and device input
//                 paths enforce the model runtime contract
// =============================================================================

// =============================================================================
// Test suite: Wan2.2-specific TrtModule contracts
// =============================================================================
//
// Builds small in-process TRT engines and validates the model-owned edge cases
// relocated from the shared TrtModule suite.  General forward, introspection,
// move, and keep-alive behavior remains covered by test_trt_module.
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
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_backend_factory_symbols_are_versioned() {
    dlerror();
    void* legacy_factory = dlsym(RTLD_DEFAULT, "trtmc_create_backend");
    const char* legacy_error = dlerror();
    check(legacy_factory == nullptr && legacy_error != nullptr,
          "TRT backend does not export the legacy factory");

    dlerror();
    void* v2_factory = dlsym(RTLD_DEFAULT, "trtmc_create_backend_v2");
    const char* v2_error = dlerror();
    check(v2_factory != nullptr && v2_error == nullptr, "TRT backend exports the v2 factory");
}

template <typename Function>
static bool throws_exception(Function&& function) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return true;
    }
    return false;
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

// Build an engine whose I/O dtype exists in TensorRT but is intentionally not
// represented by trtmc::DType. TrtModule must reject it rather than allocating
// FP32-sized buffers for one-byte BOOL elements.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_bool_identity_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* input = network->addInput("x", nvinfer1::DataType::kBOOL, nvinfer1::Dims{1, {4}});
    if (!input)
        return nullptr;
    auto* identity = network->addIdentity(*input);
    if (!identity)
        return nullptr;
    auto* output = identity->getOutput(0);
    output->setName("y");
    network->markOutput(*output);

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

static trtmc::TrtUniquePtr<nvinfer1::IHostMemory> build_two_profile_identity_plan() {
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
    auto* input = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 4});
    if (!input)
        return nullptr;
    auto* identity = network->addIdentity(*input);
    if (!identity)
        return nullptr;
    auto* output = identity->getOutput(0);
    output->setName("y");
    network->markOutput(*output);

    for (int32_t profile_index = 0; profile_index < 2; ++profile_index) {
        auto* profile = builder->createOptimizationProfile();
        if (!profile)
            return nullptr;
        const int32_t opt_batch = profile_index == 0 ? 2 : 4;
        const int32_t max_batch = profile_index == 0 ? 3 : 6;
        profile->setDimensions("x", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 4});
        profile->setDimensions("x", nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims2{opt_batch, 4});
        profile->setDimensions("x", nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims2{max_batch, 4});
        if (config->addOptimizationProfile(profile) < 0)
            return nullptr;
    }
    return trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
}

static void test_unsupported_dtypes_fail_closed() {
    check(trtmc::DType::kUnsupported != trtmc::DType::kFloat32,
          "unsupported dtype is distinct from FP32");
    check(throws_exception([]() { (void)trtmc::dtype_size(trtmc::DType::kUnsupported); }),
          "unsupported dtype has no silent element size");

    uint8_t data[4] = {};
    trtmc::Tensor unsupported{data, {4}, trtmc::DType::kUnsupported};
    check(throws_exception([&unsupported]() { (void)unsupported.nbytes(); }),
          "unsupported tensor cannot compute a byte size");

    auto engine = build_bool_identity_engine();
    check(engine != nullptr, "BOOL identity engine built");
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    check(!module.ok(), "unsupported TensorRT dtype rejects module initialization");
    check(module.tensor_dtype("x") == trtmc::DType::kUnsupported,
          "rejected module does not report unsupported input as FP32");
    check(module.tensor_dtype("missing") == trtmc::DType::kUnsupported,
          "unknown tensor dtype does not default to FP32");
    cudaStreamDestroy(stream);
}

static void test_initial_external_bindings_skip_only_selected_buffers() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* external_x = nullptr;
    void* external_y = nullptr;
    void* managed_x = nullptr;
    cudaMalloc(&external_x, 4 * sizeof(float));
    cudaMalloc(&external_y, 4 * sizeof(float));
    cudaMallocManaged(&managed_x, 4 * sizeof(float));

    {
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"y", external_y, 4 * sizeof(float)}};
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(module.ok(), "output-prebound module is ok");
        check(module.owns_device_buffer("x"), "unbound input retains owned allocation");
        check(!module.owns_device_buffer("y"), "prebound output skips owned allocation");
        check(module.device_ptr("y") == external_y, "prebound output address is installed");
        check(!module.has_host_output_staging("y"), "prebound output skips host staging");

        float input[4] = {111.0F, 112.0F, 113.0F, 114.0F};
        trtmc::Tensor tensor{input, {4}, trtmc::DType::kFloat32};
        module.forward_async({{"x", tensor}});
        module.sync();
        float result[4] = {0.0F};
        cudaMemcpy(result, external_y, sizeof(result), cudaMemcpyDeviceToHost);
        check(result[0] == 111.0F && result[3] == 114.0F,
              "output-prebound module executes with caller buffer");
    }

    {
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"x", external_x, 4 * sizeof(float)}};
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(module.ok(), "input-prebound module is ok");
        check(!module.owns_device_buffer("x"), "prebound input skips owned allocation");
        check(module.device_ptr("x") == external_x, "prebound input address is installed");
        check(module.owns_device_buffer("y"), "unbound output retains owned allocation");
        check(module.has_host_output_staging("y"), "unbound output retains host staging");

        float input[4] = {121.0F, 122.0F, 123.0F, 124.0F};
        cudaMemcpyAsync(external_x, input, sizeof(input), cudaMemcpyHostToDevice, stream);
        const auto outputs = module.forward({});
        const auto* result = static_cast<const float*>(outputs.at("y").data);
        check(result[0] == 121.0F && result[3] == 124.0F,
              "input-prebound module executes without host recopy");
    }

    {
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"x", managed_x, 4 * sizeof(float)}};
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(module.ok(), "managed-memory prebinding is accepted");
        check(module.device_ptr("x") == managed_x, "managed-memory address is installed");
    }

    {
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"y", external_y, 4 * sizeof(float) - 1}};
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(!module.ok(), "undersized initial external binding is rejected");
    }

    {
        float host_buffer[4] = {0.0F};
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"x", host_buffer, sizeof(host_buffer)}};
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(!module.ok(), "host pointer initial external binding is rejected");
    }

    {
        auto* ctx = engine->createExecutionContext();
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
        check(module.ok(), "default module remains ok");
        check(module.owns_device_buffer("x") && module.owns_device_buffer("y"),
              "default module retains historical owned allocations");
        check(module.has_host_output_staging("y"),
              "default module retains historical output staging");
    }

    cudaFree(external_x);
    cudaFree(external_y);
    cudaFree(managed_x);
    cudaStreamDestroy(stream);
}

static void test_mapped_host_tensor_addresses_and_runtime_policy() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream = nullptr;
    void* host_input_allocation = nullptr;
    void* host_output_allocation = nullptr;
    void* device_input_base = nullptr;
    void* device_output_base = nullptr;
    check(cudaStreamCreate(&stream) == cudaSuccess, "mapped-host probe stream created");
    check(cudaHostAlloc(&host_input_allocation, 512, cudaHostAllocMapped) == cudaSuccess,
          "mapped-host input allocated");
    check(cudaHostAlloc(&host_output_allocation, 512, cudaHostAllocMapped) == cudaSuccess,
          "mapped-host output allocated");
    if (!stream || !host_input_allocation || !host_output_allocation)
        return;
    check(cudaHostGetDevicePointer(&device_input_base, host_input_allocation, 0) == cudaSuccess,
          "mapped-host input device alias obtained");
    check(cudaHostGetDevicePointer(&device_output_base, host_output_allocation, 0) == cudaSuccess,
          "mapped-host output device alias obtained");
    if (!device_input_base || !device_output_base)
        return;

    // Exercise an aligned interior address, matching the Wan cache-bank layout
    // rather than only proving that an allocation base can be bound.
    auto* device_input = static_cast<unsigned char*>(device_input_base) + 256;
    auto* device_output = static_cast<unsigned char*>(device_output_base) + 256;
    auto* host_input =
        reinterpret_cast<float*>(static_cast<unsigned char*>(host_input_allocation) + 256);
    auto* host_output =
        reinterpret_cast<float*>(static_cast<unsigned char*>(host_output_allocation) + 256);
    host_input[0] = 201.0F;
    host_input[1] = 202.0F;
    host_input[2] = 203.0F;
    host_input[3] = 204.0F;

    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, device_input) == cudaSuccess,
          "mapped-host interior alias has CUDA pointer attributes");
#if CUDART_VERSION >= 10000
    check(attributes.type == cudaMemoryTypeHost,
          "mapped-host interior alias is classified as host memory");
    check(attributes.devicePointer == device_input,
          "mapped-host interior alias round-trips through pointer attributes");
#endif

    check(cudaMemsetAsync(device_output, 0, 4 * sizeof(float), stream) == cudaSuccess,
          "mapped-host device memset succeeds");
    check(cudaMemcpyAsync(device_output, device_input, 4 * sizeof(float), cudaMemcpyDeviceToDevice,
                          stream) == cudaSuccess,
          "mapped-host device-to-device copy succeeds");
    check(cudaStreamSynchronize(stream) == cudaSuccess,
          "mapped-host device operations synchronize");
    check(host_output[0] == 201.0F && host_output[3] == 204.0F,
          "mapped-host device-to-device copy is correct");

    // This direct TensorRT probe is intentionally allowed on GB300: it proves
    // setTensorAddress/enqueueV3 support independently of Model Connect's
    // stricter policy, which only enables mapped host bindings on iGPUs.
    {
        auto* context = engine->createExecutionContext();
        check(context != nullptr, "mapped-host direct TRT context created");
        if (context) {
            check(context->setTensorAddress("x", device_input),
                  "TensorRT accepts a mapped-host input address");
            check(context->setTensorAddress("y", device_output),
                  "TensorRT accepts a mapped-host output address");
            check(cudaMemsetAsync(device_output, 0, 4 * sizeof(float), stream) == cudaSuccess,
                  "mapped-host TensorRT output reset succeeds");
            check(context->enqueueV3(stream), "TensorRT executes with mapped-host addresses");
            check(cudaStreamSynchronize(stream) == cudaSuccess,
                  "mapped-host TensorRT execution synchronizes");
            check(host_output[0] == 201.0F && host_output[3] == 204.0F,
                  "mapped-host TensorRT output is correct");
            delete context;
        }
    }

    // Model Connect must reject this optimization on a discrete GPU while
    // accepting the exact CUDA alias on an integrated, host-mapping-capable
    // device. Plain pageable host pointers remain covered by the negative test
    // below, and undersized capacities by the construction tests above.
    int device = 0;
    int integrated = 0;
    int can_map_host_memory = 0;
    check(cudaGetDevice(&device) == cudaSuccess, "mapped-host policy device queried");
    check(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) == cudaSuccess,
          "mapped-host policy integrated attribute queried");
    check(cudaDeviceGetAttribute(&can_map_host_memory, cudaDevAttrCanMapHostMemory, device) ==
              cudaSuccess,
          "mapped-host policy mapping attribute queried");
    {
        auto* context = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"x", device_input, 4 * sizeof(float)}, {"y", device_output, 4 * sizeof(float)}};
        trtmc::TrtModuleImpl module(engine.get(), context, stream, 0, nullptr, bindings);
        if (integrated != 0 && can_map_host_memory != 0) {
            check(module.ok(), "integrated GPU accepts mapped-host external bindings");
            if (module.ok()) {
                module.forward_device_async({});
                module.sync();
                check(host_output[0] == 201.0F && host_output[3] == 204.0F,
                      "integrated mapped-host module output is correct");
            }
        } else {
            check(!module.ok(), "discrete GPU rejects mapped-host external bindings");
        }
    }

    cudaFreeHost(host_input_allocation);
    cudaFreeHost(host_output_allocation);
    cudaStreamDestroy(stream);
}

static void test_invalid_external_bindings_are_rejected() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    auto* const original = module.device_ptr("x");

    check(throws_exception([&module]() { module.bind_external("x", nullptr); }),
          "null external binding throws");
    check(module.device_ptr("x") == original, "null external binding preserves owned buffer");
    check(throws_exception([&module, original]() { module.bind_external("missing", original); }),
          "unknown external binding throws");
    float host_external[4] = {0.0F};
    check(
        throws_exception([&module, &host_external]() { module.bind_external("x", host_external); }),
        "host external binding throws");
    check(module.device_ptr("x") == original, "host external rejection preserves owned buffer");

    float input_data[4] = {11.0f, 12.0f, 13.0f, 14.0f};
    trtmc::Tensor input{input_data, {4}, trtmc::DType::kFloat32};
    const auto outputs = module.forward({{"x", input}});
    check(outputs.count("y") == 1, "module remains usable after rejected external bindings");

    cudaStreamDestroy(stream);
}

static void test_missing_and_invalid_host_inputs_are_rejected() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    check(throws_exception([&module]() { module.forward_async({}); }),
          "missing required host input throws");

    trtmc::Tensor null_input{nullptr, {4}, trtmc::DType::kFloat32};
    check(throws_exception([&module, &null_input]() { module.forward_async({{"x", null_input}}); }),
          "null host input throws");

    float input_data[4] = {21.0f, 22.0f, 23.0f, 24.0f};
    trtmc::Tensor wrong_dtype{input_data, {4}, trtmc::DType::kInt32};
    check(
        throws_exception([&module, &wrong_dtype]() { module.forward_async({{"x", wrong_dtype}}); }),
        "host input dtype mismatch throws");

    trtmc::Tensor wrong_shape{input_data, {2}, trtmc::DType::kFloat32};
    check(
        throws_exception([&module, &wrong_shape]() { module.forward_async({{"x", wrong_shape}}); }),
        "static host input shape mismatch throws");

    trtmc::Tensor valid{input_data, {4}, trtmc::DType::kFloat32};
    trtmc::TensorMap unknown{{"x", valid}, {"unknown", valid}};
    check(throws_exception([&module, &unknown]() { module.forward_async(unknown); }),
          "unknown host input throws");

    const auto outputs = module.forward({{"x", valid}});
    check(outputs.count("y") == 1, "module remains usable after rejected host inputs");
    if (outputs.count("y")) {
        const auto* result = static_cast<const float*>(outputs.at("y").data);
        check(result[0] == 21.0f && result[3] == 24.0f,
              "valid forward after rejected inputs is not stale");
    }

    cudaStreamDestroy(stream);
}

static void test_dynamic_input_shapes_are_checked() {
    auto engine = build_dynamic_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    check(module.ok(), "dynamic module is ok");

    float first_data[4] = {41.0f, 42.0f, 43.0f, 44.0f};
    trtmc::Tensor first{first_data, {1, 4}, trtmc::DType::kFloat32};
    auto outputs = module.forward({{"x", first}});
    check(outputs.at("y").shape == std::vector<int64_t>({1, 4}),
          "dynamic input accepts in-profile shape");

    check(throws_exception(
              [&module]() { module.bind_external("x", nullptr, std::vector<int64_t>{3, 4}); }),
          "null shaped external binding throws");
    check(module.tensor_shape("x") == std::vector<int64_t>({1, 4}),
          "rejected shaped external binding preserves input shape");

    float invalid_data[16] = {0};
    trtmc::Tensor invalid{invalid_data, {4, 4}, trtmc::DType::kFloat32};
    check(throws_exception([&module, &invalid]() { module.forward_async({{"x", invalid}}); }),
          "dynamic input rejects out-of-profile shape");

    float final_data[12] = {51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f,
                            57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f};
    trtmc::Tensor final{final_data, {3, 4}, trtmc::DType::kFloat32};
    outputs = module.forward({{"x", final}});
    check(outputs.at("y").shape == std::vector<int64_t>({3, 4}),
          "dynamic module remains usable after rejected shape");
    const auto* result = static_cast<const float*>(outputs.at("y").data);
    check(result[0] == 51.0f && result[11] == 62.0f,
          "dynamic valid retry does not return stale data");

    cudaStreamDestroy(stream);
}

static void test_same_external_pointer_shape_failure_is_transactional() {
    auto engine = build_dynamic_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* external = nullptr;
    cudaMalloc(&external, 12 * sizeof(float));
    {
        auto* ctx = engine->createExecutionContext();
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
        module.bind_external("x", external, {1, 4});

        check(throws_exception(
                  [&module, external]() { module.bind_external("x", external, {4, 4}); }),
              "same external pointer rejects out-of-profile shape");
        check(module.ok(), "same external pointer shape rollback keeps module usable");
        check(module.tensor_shape("x") == std::vector<int64_t>({1, 4}),
              "same external pointer shape rollback preserves tracked shape");

        float final_data[12] = {101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f,
                                107.0f, 108.0f, 109.0f, 110.0f, 111.0f, 112.0f};
        trtmc::Tensor final{final_data, {3, 4}, trtmc::DType::kFloat32};
        const auto outputs = module.forward({{"x", final}});
        const auto* result = static_cast<const float*>(outputs.at("y").data);
        check(result[0] == 101.0f && result[11] == 112.0f,
              "same external pointer valid retry produces current data");
    }
    cudaFree(external);
    cudaStreamDestroy(stream);
}

static void test_cuda_graph_execution_remains_correct() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    module.enable_cuda_graph();

    float first_data[4] = {71.0f, 72.0f, 73.0f, 74.0f};
    trtmc::Tensor first{first_data, {4}, trtmc::DType::kFloat32};
    auto outputs = module.forward({{"x", first}});
    const auto* first_result = static_cast<const float*>(outputs.at("y").data);
    check(first_result[0] == 71.0f && first_result[3] == 74.0f,
          "CUDA graph first execution is correct");

    float second_data[4] = {81.0f, 82.0f, 83.0f, 84.0f};
    trtmc::Tensor second{second_data, {4}, trtmc::DType::kFloat32};
    outputs = module.forward({{"x", second}});
    const auto* second_result = static_cast<const float*>(outputs.at("y").data);
    check(second_result[0] == 81.0f && second_result[3] == 84.0f,
          "CUDA graph replay uses current input");

    cudaStreamDestroy(stream);
}

static void test_external_bindings_reject_multiple_live_profiles() {
    auto plan = build_two_profile_identity_plan();
    if (!plan)
        return;
    std::unique_ptr<trtmc::IBackend, void (*)(trtmc::IBackend*)> backend(trtmc_create_backend_v2(),
                                                                         trtmc_destroy_backend_v2);
    check(backend != nullptr, "backend for profile prebinding policy is available");
    if (!backend)
        return;

    void* external = nullptr;
    cudaMalloc(&external, 24 * sizeof(float));
    trtmc::ModuleCreateOptions options;
    options.external_bindings.push_back({"x", external, 24 * sizeof(float)});
    auto rejects_aliasing = [](auto&& operation) {
        try {
            operation();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find("multiple live TRT profile modules") !=
                   std::string::npos;
        }
        return false;
    };
    check(rejects_aliasing([&]() {
              (void)backend->create_dual_profile_modules(plan->data(), plan->size(), options);
          }),
          "dual-profile API rejects one aliased external binding set");
    check(rejects_aliasing([&]() {
              (void)backend->create_profile_modules(plan->data(), plan->size(), options, {0, 1});
          }),
          "profile-list API rejects one aliased external binding set");

    trtmc::ModuleCreateOptions lane_options = options;
    lane_options.optimization_profile = 0;
    bool lane_binding_was_forwarded = false;
    try {
        (void)backend->create_context_modules(plan->data(), plan->size(), {lane_options});
    } catch (const std::runtime_error& error) {
        // This test engine has dynamic input x, for which initial external
        // prebinding is intentionally rejected by TrtModuleImpl. Observing
        // that rejection proves the per-lane binding reached the module.
        lane_binding_was_forwarded =
            std::string(error.what()).find("TrtModuleImpl creation failed") != std::string::npos;
    }
    check(lane_binding_was_forwarded, "context API forwards the lane external binding");
    cudaFree(external);
}

static void test_forward_device_with_direct_buffer_population() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    float input_data[4] = {91.0f, 92.0f, 93.0f, 94.0f};
    cudaMemcpyAsync(module.device_ptr("x"), input_data, sizeof(input_data), cudaMemcpyHostToDevice,
                    stream);
    module.forward_device_async({});
    module.sync();

    float result[4] = {0};
    cudaMemcpy(result, module.device_ptr("y"), sizeof(result), cudaMemcpyDeviceToHost);
    check(result[0] == 91.0f && result[3] == 94.0f,
          "direct device_ptr population permits an empty device input map");

    cudaStreamDestroy(stream);
}

static void test_forward_device_with_supplied_input_contract() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);

    trtmc::DeviceTensor input({4}, trtmc::DType::kFloat32, stream);
    float host_data[4] = {31.0f, 32.0f, 33.0f, 34.0f};
    check(input.copy_from_host(host_data), "forward_device: input upload succeeds");
    trtmc::DeviceTensorMap null_input{{"x", nullptr}};
    check(throws_exception([&module, &null_input]() { module.forward_device(null_input); }),
          "forward_device: supplied null input throws");

    trtmc::DeviceTensorMap inputs{{"x", &input}};
    (void)module.forward_device(inputs);
    float result[4] = {0};
    cudaMemcpy(result, module.device_ptr("y"), sizeof(result), cudaMemcpyDeviceToHost);
    check(result[0] == 31.0f && result[3] == 34.0f,
          "forward_device: valid input produces current output");

    cudaStreamDestroy(stream);
}

int main() {
    test_backend_factory_symbols_are_versioned();
    test_unsupported_dtypes_fail_closed();
    test_initial_external_bindings_skip_only_selected_buffers();
    test_mapped_host_tensor_addresses_and_runtime_policy();
    test_invalid_external_bindings_are_rejected();
    test_missing_and_invalid_host_inputs_are_rejected();
    test_dynamic_input_shapes_are_checked();
    test_same_external_pointer_shape_failure_is_transactional();
    test_cuda_graph_execution_remains_correct();
    test_external_bindings_reject_multiple_live_profiles();
    test_forward_device_with_direct_buffer_population();
    test_forward_device_with_supplied_input_contract();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All Wan2.2 TRT module contract tests passed\n";
    return 0;
}
