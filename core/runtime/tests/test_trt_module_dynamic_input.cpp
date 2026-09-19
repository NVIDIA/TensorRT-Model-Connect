/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "runtime/tensorrt/trt_module_impl.h"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static std::size_t pairs(const TrtModuleImpl& module) {
        return module.timing_events_.size() + module.free_timing_events_.size();
    }
    static std::size_t pending(const TrtModuleImpl& module) { return module.timing_events_.size(); }
    static std::uint64_t completed(const TrtModuleImpl& module) { return module.timing_launches_; }
    static cudaEvent_t oldest_start(const TrtModuleImpl& module) {
        return module.timing_events_.front().start;
    }
    static cudaEvent_t oldest_stop(const TrtModuleImpl& module) {
        return module.timing_events_.front().stop;
    }
    static void reclaim(TrtModuleImpl& module) { module.reclaim_timing_events(); }
    static void flush(TrtModuleImpl& module) { module.flush_timing_events(); }
    static bool collects_timing(const TrtModuleImpl& module) { return module.collect_timing_; }
    static bool drains_before_destroy(const TrtModuleImpl& module) {
        return module.drain_before_destroy_;
    }
};

} // namespace trtmc

namespace {

int failures = 0;

struct TimingLog {
    std::ostringstream text;
    std::streambuf* previous{std::cerr.rdbuf(text.rdbuf())};
    ~TimingLog() {
        std::cerr.rdbuf(previous);
        std::cerr << text.str();
    }
};

void check_timing_log(const std::string& text, std::uint64_t calls, const char* label) {
    const auto prefix = std::string("[trtmc.engine_timing] label=\"") + label + "\" execute_ms=";
    const auto position = text.find(prefix);
    if (position == std::string::npos ||
        text.find(" launches=" + std::to_string(calls) + "\n", position) == std::string::npos ||
        text.find(prefix, position + 1) != std::string::npos) {
        std::cerr << "FAIL: timing log lost or duplicated launches\n";
        ++failures;
    }
}

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_dynamic_identity() {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {-1}});
    auto* identity = network->addIdentity(*input);
    identity->getOutput(0)->setName("output");
    network->markOutput(*identity->getOutput(0));

    auto* profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {2}});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {4}});
    config->addOptimizationProfile(profile);

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

void test_first_forward_allocates_unbound_dynamic_input(nvinfer1::ICudaEngine& engine,
                                                        cudaStream_t stream) {
    // Preserve coverage for existing callers that transfer a raw context.
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
    check(module.device_ptr("input") == nullptr,
          "dynamic input is not allocated during module construction");
    float values[] = {1.0F, 2.0F};
    const auto outputs =
        module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
    check(module.device_ptr("input") != nullptr,
          "first forward allocates and binds an unbound dynamic input");
    const auto found = outputs.find("output");
    check(found != outputs.end() && found->second.numel() == 2,
          "dynamic identity returns its runtime shape");
    if (found != outputs.end() && found->second.numel() == 2) {
        const auto* output = static_cast<const float*>(found->second.data);
        check(std::fabs(output[0] - 1.0F) < 1.0e-6F && std::fabs(output[1] - 2.0F) < 1.0e-6F,
              "dynamic identity returns its input values");
    }
}

void test_external_dynamic_input_stays_external(nvinfer1::ICudaEngine& engine,
                                                cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(
            &engine,
            trtmc::TrtUniquePtr<nvinfer1::IExecutionContext>(engine.createExecutionContext()),
            stream);
        check(module.device_ptr("input") == nullptr,
              "second dynamic input also starts without an allocation");
        module.bind_external("input", external, {2});
        float values[] = {3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
        check(module.device_ptr("input") == external,
              "forward preserves the pre-bound dynamic input buffer");
    }
    cudaFree(external);
}

void test_backend_managed_graph_does_not_enable_nested_capture(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    trtmc::TrtModuleImpl module(
        &engine, trtmc::TrtUniquePtr<nvinfer1::IExecutionContext>(engine.createExecutionContext()),
        stream, 0, nullptr, {}, true);
    module.enable_cuda_graph();
    check(!module.cuda_graph_active(), "backend-managed graph leaves stream capture disabled");
}

void test_timing_events_reuse_completed_pairs(nvinfer1::ICudaEngine& engine, cudaStream_t stream) {
    TimingLog log;
    constexpr int calls = 256;
    {
        trtmc::TrtModuleImpl module(
            &engine,
            trtmc::TrtUniquePtr<nvinfer1::IExecutionContext>(engine.createExecutionContext()),
            stream);
        module.set_timing_label("pooled synchronous");
        cudaEvent_t first = nullptr;
        for (int index = 0; index < calls; ++index) {
            float values[] = {static_cast<float>(index), 2.0F};
            const auto output = module.forward({{"input", {values, {2}, trtmc::DType::kFloat32}}});
            check(static_cast<const float*>(output.at("output").data)[0] == values[0],
                  "event reuse preserves inference output");
            const auto start = trtmc::TrtModuleImplTestPeer::oldest_start(module);
            if (index == 0)
                first = start;
            check(start == first && trtmc::TrtModuleImplTestPeer::pairs(module) == 1,
                  "synchronous forwards reuse one event pair");
        }
        trtmc::TrtModuleImplTestPeer::reclaim(module);
        check(trtmc::TrtModuleImplTestPeer::pending(module) == 0 &&
                  trtmc::TrtModuleImplTestPeer::completed(module) == calls,
              "all completed launches are accumulated exactly once");
        trtmc::TrtModuleImplTestPeer::flush(module);
    }
    check_timing_log(log.text.str(), calls, "pooled synchronous");
}

void CUDART_CB wait_for_gate(void* data) {
    auto* gate = static_cast<std::atomic<bool>*>(data);
    while (!gate->load(std::memory_order_acquire))
        std::this_thread::yield();
}

bool enqueue_behind_gate(trtmc::TrtModuleImpl& module, int device, int calls) {
    if (cudaSetDevice(device) != cudaSuccess)
        return false;
    try {
        for (int index = 0; index < calls; ++index)
            module.forward_device_async({});
    } catch (...) {
        return false;
    }
    return true;
}

void test_timing_events_keep_pending_pairs(nvinfer1::ICudaEngine& engine, cudaStream_t stream) {
    TimingLog log;
    constexpr int calls = 8;
    {
        trtmc::TrtModuleImpl module(
            &engine,
            trtmc::TrtUniquePtr<nvinfer1::IExecutionContext>(engine.createExecutionContext()),
            stream);
        module.set_timing_label("pooled asynchronous");
        float values[] = {3.0F, 4.0F};
        (void)module.forward({{"input", {values, {2}, trtmc::DType::kFloat32}}});
        int device = 0;
        cudaGetDevice(&device);
        std::atomic<bool> gate{false};
        check(cudaLaunchHostFunc(stream, wait_for_gate, &gate) == cudaSuccess, "queue test gate");
        auto producer =
            std::async(std::launch::async, enqueue_behind_gate, std::ref(module), device, calls);
        const bool nonblocking =
            producer.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
        if (nonblocking) {
            check(trtmc::TrtModuleImplTestPeer::pending(module) == calls &&
                      trtmc::TrtModuleImplTestPeer::pairs(module) == calls,
                  "pending launches retain distinct event pairs");
            check(cudaEventQuery(trtmc::TrtModuleImplTestPeer::oldest_stop(module)) ==
                      cudaErrorNotReady,
                  "oldest recorded stop is still in flight");
        }
        gate.store(true, std::memory_order_release);
        check(producer.get() && nonblocking, "enqueue never waits for pending timing events");
        module.sync();
        trtmc::TrtModuleImplTestPeer::reclaim(module);
        check(trtmc::TrtModuleImplTestPeer::completed(module) == calls + 1 &&
                  trtmc::TrtModuleImplTestPeer::pending(module) == 0,
              "completed asynchronous batch is fully reclaimed");
        module.forward_device_async({});
        check(trtmc::TrtModuleImplTestPeer::pairs(module) == calls,
              "reclaimed asynchronous pairs are reused without growing the pool");
        // Leave the final pair pending for the destructor to synchronize and count.
    }
    check_timing_log(log.text.str(), calls + 2, "pooled asynchronous");
}

void test_timing_events_cover_graph_replays(nvinfer1::ICudaEngine& engine, cudaStream_t stream) {
    TimingLog log;
    constexpr int calls = 10;
    {
        trtmc::TrtModuleImpl module(
            &engine,
            trtmc::TrtUniquePtr<nvinfer1::IExecutionContext>(engine.createExecutionContext()),
            stream);
        module.set_timing_label("pooled graphs");
        module.enable_cuda_graph();
        for (int index = 0; index < calls; ++index) {
            float values[] = {static_cast<float>(index), 2.0F, 3.0F};
            const std::int64_t length = index < calls / 2 ? 2 : 3;
            const auto output =
                module.forward({{"input", {values, {length}, trtmc::DType::kFloat32}}});
            check(output.at("output").numel() == static_cast<std::size_t>(length) &&
                      static_cast<const float*>(output.at("output").data)[0] == values[0],
                  "graph timing reuse preserves results across shape changes");
            check(module.cuda_graph_captured() && trtmc::TrtModuleImplTestPeer::pairs(module) == 1,
                  "graph capture and replay reuse completed timing events");
        }
    }
    check_timing_log(log.text.str(), calls, "pooled graphs");
}

trtmc::TrtUniquePtr<nvinfer1::IHostMemory> build_counter_plan(int profiles) {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto* state =
        network->addInput("state", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{4, {1, 1, 1, 8}});
    auto* delta = network->addInput("delta", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {-1}});
    auto* bias = network->addInput("bias", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {1}});
    auto* sum = network->addReduce(*delta, nvinfer1::ReduceOperation::kSUM, 1, true);
    auto* biased =
        network->addElementWise(*sum->getOutput(0), *bias, nvinfer1::ElementWiseOperation::kSUM);
    auto* view = network->addShuffle(*biased->getOutput(0));
    view->setReshapeDimensions(nvinfer1::Dims{4, {1, 1, 1, 1}});
    auto* update =
        network->addElementWise(*state, *view->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    const std::int32_t index = 0;
    auto* indices = network->addConstant(nvinfer1::Dims{1, {1}},
                                         nvinfer1::Weights{nvinfer1::DataType::kINT32, &index, 1});
    auto* write = network->addKVCacheUpdate(*state, *update->getOutput(0), *indices->getOutput(0),
                                            nvinfer1::KVCacheMode::kLINEAR);
    write->getOutput(0)->setName("present");
    network->markOutput(*write->getOutput(0));
    for (int number = 0; number < profiles; ++number) {
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions("delta", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
        profile->setDimensions("delta", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {2}});
        profile->setDimensions("delta", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {4}});
        config->addOptimizationProfile(profile);
    }
    config->setBuilderOptimizationLevel(0);
    return trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
}

void check_counter(trtmc::DeviceTensor& state, float expected) {
    std::array<float, 8> values{};
    check(state.copy_to_host(values.data()), "counter state downloads");
    check(std::all_of(values.begin(), values.end(), [&](float value) { return value == expected; }),
          "each real call increments native cache state exactly once");
}

void counter_call(trtmc::ITrtModule& module, trtmc::DeviceTensor& state, std::int64_t length,
                  float expected, bool captured) {
    float delta[]{1.0F, 0.0F, 0.0F, 0.0F};
    float bias = 0.0F;
    module.forward_async({{"delta", {delta, {length}, trtmc::DType::kFloat32}},
                          {"bias", {&bias, {1}, trtmc::DType::kFloat32}}});
    module.sync();
    check_counter(state, expected);
    check(module.cuda_graph_captured() == captured, "automatic graph preparation state is correct");
}

void test_automatic_graph_counter(trtmc::IBackend& backend, nvinfer1::IHostMemory& plan,
                                  bool prebound) {
    auto state = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    auto alternate = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    auto bias = trtmc::DeviceTensor::zeros({1}, trtmc::DType::kFloat32, nullptr);
    check(state.ok() && alternate.ok(), "stateful graph buffers allocate");
    cudaDeviceSynchronize();
    trtmc::ModuleCreateOptions options;
    options.cuda_graphs = true;
    auto module = prebound ? backend.create_module_prebound(plan.data(), plan.size(), options,
                                                            {{"bias", bias.data(), bias.nbytes()}})
                           : backend.create_module(plan.data(), plan.size(), options);
    module->bind_external("state", state.data());
    counter_call(*module, state, 1, 1.0F, false);
    module->bind_external("state", state.data());
    module->reset_execution_context();
    counter_call(*module, state, 1, 2.0F, true);
    counter_call(*module, state, 1, 3.0F, true);
    counter_call(*module, state, 3, 4.0F, false);
    counter_call(*module, state, 3, 5.0F, true);
    bool rejected = false;
    try {
        float bad_delta[]{1, 0, 0, 0, 0};
        module->forward_async({{"delta", {bad_delta, {5}, trtmc::DType::kFloat32}}});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected && !module->cuda_graph_captured(),
          "invalid shape revokes prepared graph without execution");
    check_counter(state, 5.0F);
    counter_call(*module, state, 3, 6.0F, false);
    counter_call(*module, state, 3, 7.0F, true);
    module->bind_external("state", alternate.data());
    counter_call(*module, alternate, 3, 1.0F, false);
    counter_call(*module, alternate, 3, 2.0F, true);
    check_counter(state, 7.0F);
    module->bind_external("state", state.data());
    counter_call(*module, state, 3, 8.0F, false);
    counter_call(*module, state, 3, 9.0F, true);
    check(module->device_ptr("present") == state.data(),
          "graph policy preserves native state alias");
}

void test_automatic_dual_profiles(trtmc::IBackend& backend, nvinfer1::IHostMemory& plan) {
    auto first = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    auto second = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    cudaDeviceSynchronize();
    trtmc::ModuleCreateOptions options;
    options.cuda_graphs = true;
    auto modules = backend.create_dual_profile_modules(plan.data(), plan.size(), options);
    modules.prefill->bind_external("state", first.data());
    modules.decode->bind_external("state", second.data());
    counter_call(*modules.prefill, first, 1, 1.0F, false);
    counter_call(*modules.decode, second, 2, 1.0F, false);
    counter_call(*modules.prefill, first, 1, 2.0F, true);
    counter_call(*modules.decode, second, 2, 2.0F, true);
    modules.prefill.reset();
    counter_call(*modules.decode, second, 2, 3.0F, true);
}

trtmc::TrtUniquePtr<nvinfer1::IHostMemory> build_shape_input_plan() {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto* data = network->addInput("data", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {-1}});
    auto* shape = network->addInput("shape", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* reshape = network->addShuffle(*data);
    reshape->setInput(1, *shape);
    reshape->getOutput(0)->setName("output");
    network->markOutput(*reshape->getOutput(0));
    auto* profile = builder->createOptimizationProfile();
    const std::array<nvinfer1::OptProfileSelector, 3> selectors{nvinfer1::OptProfileSelector::kMIN,
                                                                nvinfer1::OptProfileSelector::kOPT,
                                                                nvinfer1::OptProfileSelector::kMAX};
    const std::array<std::int64_t, 3> counts{1, 2, 4};
    for (std::size_t index = 0; index < counts.size(); ++index) {
        profile->setDimensions("data", selectors[index], nvinfer1::Dims{1, {counts[index]}});
        profile->setShapeValuesV2("shape", selectors[index], &counts[index], 1);
    }
    config->addOptimizationProfile(profile);
    config->setBuilderOptimizationLevel(0);
    return trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
}

void test_automatic_graph_rejects_shape_values(trtmc::IBackend& backend) {
    auto plan = build_shape_input_plan();
    check(bool(plan), "shape-input rejection fixture builds");
    if (!plan)
        return;
    trtmc::ModuleCreateOptions options;
    options.cuda_graphs = true;
    bool rejected = false;
    try {
        (void)backend.create_module(plan->data(), plan->size(), options);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("shape-inference input") != std::string::npos;
    }
    check(rejected, "automatic graph policy rejects untracked shape-input values");
}

void test_automatic_graph_policy() {
    auto plan = build_counter_plan(2);
    check(bool(plan), "stateful native KV counter plan builds");
    if (!plan)
        return;
    std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
        trtmc_create_backend(), trtmc_destroy_backend);
    check(bool(backend), "standard backend creates for graph policy");
    if (!backend)
        return;
    test_automatic_graph_counter(*backend, *plan, false);
    test_automatic_graph_counter(*backend, *plan, true);
    test_automatic_dual_profiles(*backend, *plan);
    test_automatic_graph_rejects_shape_values(*backend);
}

void check_untimed_module(trtmc::ITrtModule& value) {
    auto* module = dynamic_cast<trtmc::TrtModuleImpl*>(&value);
    check(module != nullptr, "standard backend exposes the expected implementation");
    if (!module)
        return;
    check(!trtmc::TrtModuleImplTestPeer::collects_timing(*module) &&
              trtmc::TrtModuleImplTestPeer::pairs(*module) == 0 &&
              trtmc::TrtModuleImplTestPeer::completed(*module) == 0,
          "disabled module creates no timing pairs or aggregate samples");
    check(trtmc::TrtModuleImplTestPeer::drains_before_destroy(*module),
          "disabled timing retains explicit asynchronous lifetime protection");
}

void test_untimed_counter_modes(trtmc::IBackend& backend, nvinfer1::IHostMemory& plan,
                                bool prebound, int graph_mode) {
    TimingLog log;
    auto state = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    auto alternate = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    auto bias = trtmc::DeviceTensor::zeros({1}, trtmc::DType::kFloat32, nullptr);
    check(state.ok() && alternate.ok() && bias.ok(), "untimed counter allocations succeed");
    cudaDeviceSynchronize();
    trtmc::ModuleCreateOptions options;
    options.collect_timing = false;
    options.cuda_graphs = graph_mode == 2;
    auto module = prebound ? backend.create_module_prebound(plan.data(), plan.size(), options,
                                                            {{"bias", bias.data(), bias.nbytes()}})
                           : backend.create_module(plan.data(), plan.size(), options);
    check(module && module->ok(), "untimed ordinary/prebound module creates");
    if (!module || !module->ok())
        return;
    module->bind_external("state", state.data());
    if (graph_mode == 1)
        module->enable_cuda_graph();
    counter_call(*module, state, 1, 1.0F, graph_mode == 1);
    counter_call(*module, state, 1, 2.0F, graph_mode != 0);
    counter_call(*module, state, 3, 3.0F, graph_mode == 1);
    counter_call(*module, state, 3, 4.0F, graph_mode != 0);
    bool rejected = false;
    try {
        float invalid[]{1, 0, 0, 0, 0};
        module->forward_async({{"delta", {invalid, {5}, trtmc::DType::kFloat32}}});
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected && !module->cuda_graph_captured(),
          "untimed invalid shape fails and invalidates capture");
    module->sync();
    check_counter(state, 4.0F);
    counter_call(*module, state, 3, 5.0F, graph_mode == 1);
    counter_call(*module, state, 3, 6.0F, graph_mode != 0);
    module->bind_external("state", alternate.data());
    counter_call(*module, alternate, 3, 1.0F, graph_mode == 1);
    counter_call(*module, alternate, 3, 2.0F, graph_mode != 0);
    check_counter(state, 6.0F);
    check_untimed_module(*module);
    module.reset();
    check(log.text.str().find("[trtmc.engine_timing]") == std::string::npos,
          "disabled ordinary/prebound module emits no aggregate timing log");
}

void test_untimed_dual_profiles(trtmc::IBackend& backend, nvinfer1::IHostMemory& plan,
                                int graph_mode) {
    TimingLog log;
    auto first = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    auto second = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    cudaDeviceSynchronize();
    trtmc::ModuleCreateOptions options;
    options.collect_timing = false;
    options.cuda_graphs = graph_mode == 2;
    auto modules = backend.create_dual_profile_modules(plan.data(), plan.size(), options);
    check(modules.prefill && modules.decode, "untimed dual modules create");
    if (!modules.prefill || !modules.decode)
        return;
    if (graph_mode == 1) {
        modules.prefill->enable_cuda_graph();
        modules.decode->enable_cuda_graph();
    }
    modules.prefill->bind_external("state", first.data());
    modules.decode->bind_external("state", second.data());
    counter_call(*modules.prefill, first, 1, 1.0F, graph_mode == 1);
    counter_call(*modules.decode, second, 2, 1.0F, graph_mode == 1);
    counter_call(*modules.prefill, first, 1, 2.0F, graph_mode != 0);
    counter_call(*modules.decode, second, 2, 2.0F, graph_mode != 0);
    check_untimed_module(*modules.prefill);
    check_untimed_module(*modules.decode);
    modules.prefill.reset();
    counter_call(*modules.decode, second, 2, 3.0F, graph_mode != 0);
    check_counter(first, 2.0F);
    modules.decode.reset();
    check(log.text.str().find("[trtmc.engine_timing]") == std::string::npos,
          "disabled dual contexts emit no aggregate timing logs");
}

void test_untimed_pending_destruction(trtmc::IBackend& backend, nvinfer1::IHostMemory& plan) {
    TimingLog log;
    auto state = trtmc::DeviceTensor::zeros({1, 1, 1, 8}, trtmc::DType::kFloat32, nullptr);
    cudaDeviceSynchronize();
    trtmc::ModuleCreateOptions options;
    options.collect_timing = false;
    auto module = backend.create_module(plan.data(), plan.size(), options);
    check(module && module->ok(), "untimed destructor fixture creates");
    if (!module || !module->ok())
        return;
    module->bind_external("state", state.data());
    counter_call(*module, state, 1, 1.0F, false);
    check_untimed_module(*module);
    auto* concrete = dynamic_cast<trtmc::TrtModuleImpl*>(module.get());
    check(concrete != nullptr, "untimed destructor has concrete module");
    if (!concrete)
        return;
    int device = 0;
    check(cudaGetDevice(&device) == cudaSuccess, "untimed destructor gets device");
    std::atomic<bool> gate{false};
    check(cudaLaunchHostFunc(module->stream(), wait_for_gate, &gate) == cudaSuccess,
          "untimed destructor queues blocked stream work");
    auto producer =
        std::async(std::launch::async, enqueue_behind_gate, std::ref(*concrete), device, 1);
    const bool nonblocking =
        producer.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
    if (!nonblocking)
        gate.store(true, std::memory_order_release);
    const bool enqueued = producer.get();
    check(nonblocking && enqueued, "untimed enqueue remains asynchronous behind the gate");
    check_untimed_module(*module);
    std::promise<void> started;
    auto entered = started.get_future();
    auto destroy =
        std::async(std::launch::async, [owned = std::move(module), device, &started]() mutable {
            const bool selected = cudaSetDevice(device) == cudaSuccess;
            started.set_value();
            owned.reset();
            return selected;
        });
    entered.wait();
    const bool waited =
        destroy.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    gate.store(true, std::memory_order_release);
    check(destroy.get() && waited,
          "untimed destruction drains pending inference before freeing state");
    check_counter(state, 2.0F);
    check(log.text.str().find("[trtmc.engine_timing]") == std::string::npos,
          "untimed destructor does not manufacture timing samples");
}

void test_untimed_collection_policy() {
    check(trtmc::ModuleCreateOptions{}.collect_timing,
          "default module options preserve legacy timing collection");
    auto plan = build_counter_plan(2);
    check(bool(plan), "untimed stateful native counter plan builds");
    if (!plan)
        return;
    std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
        trtmc_create_backend(), trtmc_destroy_backend);
    check(bool(backend), "untimed standard backend creates");
    if (!backend)
        return;
    for (const bool prebound : {false, true})
        for (const int graph_mode : {0, 1, 2})
            test_untimed_counter_modes(*backend, *plan, prebound, graph_mode);
    for (const int graph_mode : {0, 1, 2})
        test_untimed_dual_profiles(*backend, *plan, graph_mode);
    test_untimed_pending_destruction(*backend, *plan);
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    auto engine = build_dynamic_identity();
    if (!engine)
        return 1;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;
    test_first_forward_allocates_unbound_dynamic_input(*engine, stream);
    test_external_dynamic_input_stays_external(*engine, stream);
    test_backend_managed_graph_does_not_enable_nested_capture(*engine, stream);
    test_timing_events_reuse_completed_pairs(*engine, stream);
    test_timing_events_keep_pending_pairs(*engine, stream);
    test_timing_events_cover_graph_replays(*engine, stream);
    test_automatic_graph_policy();
    test_untimed_collection_policy();
    cudaStreamDestroy(stream);
    return failures;
}
