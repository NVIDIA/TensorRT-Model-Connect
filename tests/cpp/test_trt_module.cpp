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

#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "trtmc/runtime/tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <NvInfer.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
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

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static void fill_host_output_staging(TrtModuleImpl& module, const std::string& name,
                                         uint8_t value) {
        auto found = module.host_output_staging_.find(name);
        if (found != module.host_output_staging_.end())
            std::fill(found->second.begin(), found->second.end(), value);
    }

    static bool host_output_staging_is(const TrtModuleImpl& module, const std::string& name,
                                       uint8_t value) {
        const auto found = module.host_output_staging_.find(name);
        return found != module.host_output_staging_.end() && !found->second.empty() &&
               std::all_of(found->second.begin(), found->second.end(),
                           [value](uint8_t byte) { return byte == value; });
    }

    static void inject_execution_failure(TrtModuleImpl& module, const std::string& operation) {
        module.require_execution_success(false, operation);
    }
};

} // namespace trtmc

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

// Build a two-output engine so selective host download can prove that one
// output is copied while another remains device-only in the same forward.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_two_output_identity_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* input = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});
    if (!input)
        return nullptr;

    auto* selected_layer = network->addIdentity(*input);
    auto* device_only_layer = network->addIdentity(*input);
    if (!selected_layer || !device_only_layer)
        return nullptr;
    auto* selected_output = selected_layer->getOutput(0);
    auto* device_only_output = device_only_layer->getOutput(0);
    selected_output->setName("y");
    device_only_output->setName("z");
    network->markOutput(*selected_output);
    network->markOutput(*device_only_output);

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

static trtmc::TrtUniquePtr<nvinfer1::IHostMemory> build_dynamic_identity_plan() {
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

    return trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
}

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

static void test_forward_selected_host_outputs() {
    auto engine = build_two_output_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::RuntimeMemoryTrtModuleImpl module(engine.get(), ctx, stream);
    trtmc::IRuntimeMemoryModuleV1& module_interface = module;

    constexpr uint8_t kStagingSentinel = 0xA5;
    trtmc::TrtModuleImplTestPeer::fill_host_output_staging(module, "z", kStagingSentinel);
    float selected_input[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    trtmc::Tensor selected_tensor{selected_input, {4}, trtmc::DType::kFloat32};
    auto selected = module_interface.forward_selected({{"x", selected_tensor}}, {"y"});
    check(selected.count("y") == 1, "forward_selected: requested host output exists");
    check(selected.count("z") == 0, "forward_selected: unrequested host output omitted");
    check(trtmc::TrtModuleImplTestPeer::host_output_staging_is(module, "z", kStagingSentinel),
          "forward_selected: non-empty selection skips other D2H staging writes");
    if (selected.count("y")) {
        const auto* output = static_cast<const float*>(selected.at("y").data);
        check(output[0] == 2.0f, "forward_selected: requested output[0] = 2.0");
        check(output[3] == 8.0f, "forward_selected: requested output[3] = 8.0");
    }
    float unselected_device_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpy(unselected_device_result, module.device_ptr("z"), sizeof(unselected_device_result),
               cudaMemcpyDeviceToHost);
    check(unselected_device_result[0] == 2.0f,
          "forward_selected: unrequested output remains on device");
    check(unselected_device_result[3] == 8.0f,
          "forward_selected: unrequested device output is complete");

    trtmc::TrtModuleImplTestPeer::fill_host_output_staging(module, "y", kStagingSentinel);

    float device_only_input[4] = {3.0f, 6.0f, 9.0f, 12.0f};
    trtmc::Tensor device_only_tensor{device_only_input, {4}, trtmc::DType::kFloat32};
    auto device_only = module_interface.forward_selected({{"x", device_only_tensor}}, {});
    check(device_only.empty(), "forward_selected: unrequested host outputs omitted");
    check(trtmc::TrtModuleImplTestPeer::host_output_staging_is(module, "y", kStagingSentinel),
          "forward_selected: unrequested output skips D2H staging write");

    float device_result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpy(device_result, module.device_ptr("y"), sizeof(device_result),
               cudaMemcpyDeviceToHost);
    check(device_result[0] == 3.0f, "forward_selected: skipped output remains on device");
    check(device_result[3] == 12.0f, "forward_selected: device output is complete");

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

static std::shared_ptr<void> allocate_cuda_owner(std::size_t bytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess)
        return {};
    return std::shared_ptr<void>(pointer, [](void* allocation) { cudaFree(allocation); });
}

static trtmc::RuntimeMemoryBindingV1
make_runtime_binding(const char* name, const std::shared_ptr<void>& owner, int32_t device,
                     uint64_t valid_tokens, uint64_t bound_tokens, uint64_t capacity_tokens) {
    trtmc::RuntimeMemoryBindingV1 binding;
    binding.name = name;
    binding.pointer = owner.get();
    binding.capacity_bytes = static_cast<std::size_t>(capacity_tokens) * 4U * sizeof(float);
    binding.shape = {static_cast<int64_t>(bound_tokens), 4};
    binding.dtype = trtmc::DType::kFloat32;
    binding.device = device;
    binding.lifetime = owner;
    binding.valid_tokens = valid_tokens;
    binding.bound_tokens = bound_tokens;
    binding.capacity_tokens = capacity_tokens;
    binding.sequence_axis = 0;
    return binding;
}

static trtmc::RuntimeMemoryShapeV1 make_runtime_shape(const char* name, uint64_t valid_tokens,
                                                      uint64_t bound_tokens,
                                                      uint64_t capacity_tokens) {
    trtmc::RuntimeMemoryShapeV1 shape;
    shape.name = name;
    shape.shape = {static_cast<int64_t>(bound_tokens), 4};
    shape.dtype = trtmc::DType::kFloat32;
    shape.valid_tokens = valid_tokens;
    shape.bound_tokens = bound_tokens;
    shape.capacity_tokens = capacity_tokens;
    shape.sequence_axis = 0;
    return shape;
}

static void test_runtime_memory_backend_v1() {
    auto plan = build_dynamic_identity_plan();
    if (!plan)
        return;

    std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
        trtmc_create_backend(), trtmc_destroy_backend);
    check(backend != nullptr, "runtime memory: standard backend created");
    if (!backend)
        return;

    auto* runtime_backend = dynamic_cast<trtmc::IRuntimeMemoryBackendV1*>(backend.get());
    check(runtime_backend != nullptr, "runtime memory: backend capability is discoverable");
    if (!runtime_backend)
        return;
    check(runtime_backend->runtime_memory_api_version() == trtmc::kRuntimeMemoryBackendApiVersionV1,
          "runtime memory: backend API version is v1");

    auto legacy_module = backend->create_module(plan->data(), plan->size(), {});
    check(dynamic_cast<trtmc::IRuntimeMemoryModuleV1*>(legacy_module.get()) == nullptr,
          "runtime memory: legacy and RTX-shared module type has no dynamic capability");
    check(dynamic_cast<trtmc::IRuntimeMemoryEngineIntrospectionV1*>(legacy_module.get()) == nullptr,
          "runtime memory: engine accounting is private to qualified modules");

    trtmc::RuntimeMemoryModuleOptionsV1 invalid_alias_options;
    trtmc::RuntimeMemoryAliasPairV1 invalid_alias;
    invalid_alias.input_name = "x";
    invalid_alias.output_name = "y";
    invalid_alias_options.alias_pairs.push_back(invalid_alias);
    bool graph_alias_rejected = false;
    try {
        (void)runtime_backend->create_profile_modules_runtime_memory(plan->data(), plan->size(), {},
                                                                     {0}, invalid_alias_options);
    } catch (const std::exception&) {
        graph_alias_rejected = true;
    }
    check(graph_alias_rejected,
          "runtime memory: declared alias must match deserialized engine metadata");

    trtmc::RuntimeMemoryModuleOptionsV1 memory_options;
    memory_options.deferred_tensor_names = {"x", "y"};
    auto profile_modules = runtime_backend->create_profile_modules_runtime_memory(
        plan->data(), plan->size(), {}, {0, 0}, memory_options);
    check(profile_modules.modules.size() == 2,
          "runtime memory: two USER_MANAGED contexts share one engine");
    if (profile_modules.modules.size() != 2)
        return;

    int32_t device = -1;
    cudaGetDevice(&device);
    std::vector<trtmc::ITrtModule*> modules;
    modules.reserve(2);
    std::uintptr_t shared_engine_identity = 0;

    for (auto& profile_module : profile_modules.modules) {
        auto* module = profile_module.module.get();
        auto* runtime_module = dynamic_cast<trtmc::IRuntimeMemoryModuleV1*>(module);
        auto* engine_introspection =
            dynamic_cast<trtmc::IRuntimeMemoryEngineIntrospectionV1*>(module);
        check(runtime_module != nullptr, "runtime memory: module capability is discoverable");
        check(engine_introspection != nullptr,
              "runtime memory: qualified module exposes engine accounting");
        if (!runtime_module || !engine_introspection)
            return;
        const auto engine_stats = engine_introspection->runtime_memory_engine_stats();
        check(engine_stats.struct_size == sizeof(trtmc::RuntimeMemoryEngineStatsV1),
              "runtime memory: engine accounting struct size is v1");
        check(engine_stats.api_version == trtmc::kRuntimeMemoryBackendApiVersionV1,
              "runtime memory: engine accounting API version is v1");
        check(engine_stats.engine_identity != 0,
              "runtime memory: engine accounting has a dedupe identity");
        check(engine_stats.total_weight_bytes_available,
              "runtime memory: TensorRT total-weight engine stat is available");
        check(engine_stats.device_output_bytes == 0,
              "runtime memory: deferred output has no backend device allocation");
        check(engine_stats.host_output_staging_bytes == 0,
              "runtime memory: deferred output has no backend host staging allocation");
        if (shared_engine_identity == 0)
            shared_engine_identity = engine_stats.engine_identity;
        else
            check(engine_stats.engine_identity == shared_engine_identity,
                  "runtime memory: profile contexts report one shared engine identity");
        check(module->device_ptr("x") == nullptr,
              "runtime memory: deferred input is not profile-MAX allocated");
        check(module->device_ptr("y") == nullptr,
              "runtime memory: deferred output is not profile-MAX allocated");

        auto invalid_ordering = make_runtime_shape("x", 3, 2, 3);
        bool ordering_rejected = false;
        try {
            runtime_module->set_runtime_binding_shape(invalid_ordering);
        } catch (const std::invalid_argument&) {
            ordering_rejected = true;
        }
        check(ordering_rejected, "runtime memory: valid <= bound <= capacity is enforced");

        for (const auto [history, bound] : {std::pair<std::uint64_t, std::uint64_t>{0, 2},
                                            std::pair<std::uint64_t, std::uint64_t>{1, 1}}) {
            bool sentinel_rejected = false;
            try {
                runtime_module->set_runtime_binding_shape(
                    make_runtime_shape("x", history, bound, 3));
            } catch (const std::invalid_argument&) {
                sentinel_rejected = true;
            }
            check(sentinel_rejected, "runtime memory: cold history has a unique T=1 sentinel");
        }

        runtime_module->set_runtime_binding_shape(make_runtime_shape("x", 2, 2, 3));
        runtime_module->set_runtime_binding_shape(make_runtime_shape("y", 2, 2, 3));
        check(!runtime_module->runtime_memory_ready(),
              "runtime memory: shape planning does not imply an allocated buffer");
        modules.push_back(module);
    }

    const auto shared_requirement = runtime_backend->shared_context_memory_requirement(modules);
    check(shared_requirement.device == device,
          "runtime memory: shared context requirement reports CUDA device");

    trtmc::RuntimeMemoryContextBlockV1 context_block;
    context_block.capacity_bytes = shared_requirement.capacity_bytes;
    context_block.alignment = shared_requirement.alignment;
    context_block.device = shared_requirement.device;
    if (context_block.capacity_bytes > 0) {
        context_block.lifetime = allocate_cuda_owner(context_block.capacity_bytes);
        check(context_block.lifetime != nullptr,
              "runtime memory: shared actual-shape context block allocated");
        if (!context_block.lifetime)
            return;
        context_block.pointer = context_block.lifetime.get();
    }
    runtime_backend->bind_shared_context_memory(modules, context_block);

    std::vector<std::shared_ptr<void>> input_owners;
    std::vector<std::shared_ptr<void>> output_owners;
    input_owners.reserve(2);
    output_owners.reserve(2);
    for (std::size_t lane = 0; lane < modules.size(); ++lane) {
        auto* runtime_module = dynamic_cast<trtmc::IRuntimeMemoryModuleV1*>(modules[lane]);
        check(!runtime_module->runtime_memory_ready(),
              "runtime memory: context can be sized before allocating KV buffers");
        bool premature_enqueue_rejected = false;
        try {
            modules[lane]->forward_async({});
        } catch (const std::logic_error&) {
            premature_enqueue_rejected = true;
        }
        check(premature_enqueue_rejected,
              "runtime memory: enqueue before final pointer bind is rejected");

        auto input_owner = allocate_cuda_owner(3U * 4U * sizeof(float));
        auto output_owner = allocate_cuda_owner(3U * 4U * sizeof(float));
        check(input_owner != nullptr && output_owner != nullptr,
              "runtime memory: external CUDA buffers allocated after O(r) query");
        if (!input_owner || !output_owner)
            return;
        auto input_binding = make_runtime_binding("x", input_owner, device, 2, 2, 3);
        auto undersized = input_binding;
        undersized.capacity_bytes = 2U * 4U * sizeof(float);
        bool capacity_rejected = false;
        try {
            runtime_module->bind_runtime_memory(undersized);
        } catch (const std::invalid_argument&) {
            capacity_rejected = true;
        }
        check(capacity_rejected, "runtime memory: physical R capacity is enforced");

        runtime_module->bind_runtime_memory(input_binding);
        runtime_module->bind_runtime_memory(
            make_runtime_binding("y", output_owner, device, 2, 2, 3));
        check(modules[lane]->device_ptr("x") == input_owner.get(),
              "runtime memory: input descriptor installs the caller buffer");
        check(modules[lane]->device_ptr("y") == output_owner.get(),
              "runtime memory: output descriptor installs the caller buffer");
        check(runtime_module->runtime_memory_ready(),
              "runtime memory: descriptor and context phases complete");

        float values[8];
        for (std::size_t index = 0; index < 8; ++index)
            values[index] = static_cast<float>(lane * 10 + index + 1);
        cudaMemcpy(input_owner.get(), values, sizeof(values), cudaMemcpyHostToDevice);
        modules[lane]->forward_async({});
        modules[lane]->sync();

        float result[8] = {};
        cudaMemcpy(result, output_owner.get(), sizeof(result), cudaMemcpyDeviceToHost);
        check(result[0] == values[0],
              "runtime memory: shared context block executes serial lane first value");
        check(result[7] == values[7],
              "runtime memory: shared context block executes serial lane last value");
        input_owners.push_back(std::move(input_owner));
        output_owners.push_back(std::move(output_owner));
    }

    trtmc::RuntimeMemoryModuleOptionsV1 output_only_options;
    output_only_options.deferred_tensor_names = {"y"};
    auto candidate_module = runtime_backend->create_module_runtime_memory(
        plan->data(), plan->size(), {}, output_only_options);
    auto* candidate_runtime = dynamic_cast<trtmc::IRuntimeMemoryModuleV1*>(candidate_module.get());
    check(candidate_runtime != nullptr,
          "runtime memory: candidate module exposes private shape API");
    if (!candidate_runtime)
        return;

    trtmc::RuntimeInputShapeV1 ordinary_input_shape;
    ordinary_input_shape.name = "x";
    ordinary_input_shape.shape = {2, 4};
    candidate_runtime->set_runtime_input_shape(ordinary_input_shape);
    candidate_runtime->set_runtime_binding_shape(make_runtime_shape("y", 2, 2, 3));
    std::vector<trtmc::ITrtModule*> candidate_modules{candidate_module.get()};
    const auto candidate_requirement =
        runtime_backend->shared_context_memory_requirement(candidate_modules);
    trtmc::RuntimeMemoryContextBlockV1 candidate_block;
    candidate_block.capacity_bytes = candidate_requirement.capacity_bytes;
    candidate_block.alignment = candidate_requirement.alignment;
    candidate_block.device = candidate_requirement.device;
    if (candidate_block.capacity_bytes > 0) {
        candidate_block.lifetime = allocate_cuda_owner(candidate_block.capacity_bytes);
        candidate_block.pointer = candidate_block.lifetime.get();
    }
    runtime_backend->bind_shared_context_memory(candidate_modules, candidate_block);

    auto candidate_output = allocate_cuda_owner(3U * 4U * sizeof(float));
    check(candidate_output != nullptr,
          "runtime memory: candidate output allocation succeeds after O(r) query");
    if (!candidate_output)
        return;
    candidate_runtime->bind_runtime_memory(
        make_runtime_binding("y", candidate_output, device, 2, 2, 3));
    float candidate_values[8] = {21.0F, 22.0F, 23.0F, 24.0F, 25.0F, 26.0F, 27.0F, 28.0F};
    trtmc::Tensor candidate_input{candidate_values, {2, 4}, trtmc::DType::kFloat32};
    candidate_module->forward_async({{"x", candidate_input}});
    candidate_module->sync();
    float candidate_result[8] = {};
    cudaMemcpy(candidate_result, candidate_output.get(), sizeof(candidate_result),
               cudaMemcpyDeviceToHost);
    check(candidate_result[0] == candidate_values[0],
          "runtime memory: ordinary input candidate shape executes without planner copy");
    check(candidate_result[7] == candidate_values[7],
          "runtime memory: ordinary input candidate shape covers full Sq");

    ordinary_input_shape.shape = {3, 4};
    candidate_runtime->set_runtime_input_shape(ordinary_input_shape);
    check(!candidate_runtime->runtime_memory_ready(),
          "runtime memory: input shape change invalidates prior output generation");
    bool stale_output_rejected = false;
    try {
        (void)candidate_runtime->context_memory_requirement();
    } catch (const std::logic_error&) {
        stale_output_rejected = true;
    }
    check(stale_output_rejected, "runtime memory: stale deferred output blocks context re-query");

    candidate_runtime->set_runtime_binding_shape(make_runtime_shape("y", 3, 3, 3));
    const auto resized_requirement = candidate_runtime->context_memory_requirement();
    trtmc::RuntimeMemoryContextBlockV1 resized_block;
    resized_block.capacity_bytes = resized_requirement.capacity_bytes;
    resized_block.alignment = resized_requirement.alignment;
    resized_block.device = resized_requirement.device;
    if (resized_block.capacity_bytes > 0) {
        resized_block.lifetime = allocate_cuda_owner(resized_block.capacity_bytes);
        resized_block.pointer = resized_block.lifetime.get();
    }
    candidate_runtime->bind_context_memory(resized_block);
    check(!candidate_runtime->runtime_memory_ready(),
          "runtime memory: replanned output still requires a generation-matched binding");
    auto resized_output = allocate_cuda_owner(3U * 4U * sizeof(float));
    check(resized_output != nullptr, "runtime memory: resized deferred output allocation succeeds");
    if (!resized_output)
        return;
    candidate_runtime->bind_runtime_memory(
        make_runtime_binding("y", resized_output, device, 3, 3, 3));
    check(candidate_runtime->runtime_memory_ready(),
          "runtime memory: output replan and rebind complete the new generation");
}

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

static void test_execution_failure_poison_is_sticky() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    check(module.ok(), "execution poison: module starts valid");

    bool injected_throw = false;
    try {
        trtmc::TrtModuleImplTestPeer::inject_execution_failure(module, "injected enqueue failure");
    } catch (const std::runtime_error& error) {
        injected_throw =
            std::string(error.what()).find("injected enqueue failure") != std::string::npos;
    }
    check(injected_throw, "execution poison: triggering failure throws");
    check(!module.ok(), "execution poison: failure invalidates module");

    bool sticky_throw = false;
    try {
        module.sync();
    } catch (const std::runtime_error& error) {
        sticky_throw =
            std::string(error.what()).find("injected enqueue failure") != std::string::npos;
    }
    check(sticky_throw, "execution poison: later calls fail closed");
    cudaStreamDestroy(stream);
}

int main() {
    test_forward_cpu();
    test_forward_async();
    test_forward_selected_host_outputs();
    test_introspection();
    test_device_ptr();
    test_runtime_memory_backend_v1();
    test_bind_external();
    test_unique_ptr_ownership();
    test_keep_alive();
    test_forward_device();
    test_forward_device_with_input();
    test_profile_idx_default();
    test_profile_idx_invalid();
    test_execution_failure_poison_is_sticky();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
    }
    return failures;
}
