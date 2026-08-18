/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SAM2-HOI-CPP-05
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SAM2-HOI-05
// Intent:         SAM2 HOI benchmark timing policy at TensorRT module construction
// Preconditions:  TRT + CUDA GPU available, identity engine built in-process
// Postconditions: Timing defaults on, accepts exact 0/1, snapshots construction state,
//                 retains no events when disabled, and rejects invalid configuration
// =============================================================================

#include "../../test_helpers.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"

#include <NvInfer.h>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static bool timing_enabled(const TrtModuleImpl& module) { return module.timing_enabled_; }
    static std::size_t timing_event_count(const TrtModuleImpl& module) {
        return module.timing_events_.size();
    }
};

} // namespace trtmc

namespace {

int g_failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++g_failures;
    }
}

trtmc::TrtLogger g_logger;

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* input = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});
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

void test_engine_timing_environment() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        trtmc_test::EnvVarGuard timing("TRTMC_ENGINE_TIMING");
        trtmc::TrtModuleImpl module(engine.get(), engine->createExecutionContext(), stream);
        check(module.ok(), "engine timing default: module is ok");
        check(trtmc::TrtModuleImplTestPeer::timing_enabled(module),
              "engine timing default: enabled");
    }
    {
        trtmc_test::EnvVarGuard timing("TRTMC_ENGINE_TIMING", "1");
        trtmc::TrtModuleImpl module(engine.get(), engine->createExecutionContext(), stream);
        check(module.ok(), "engine timing explicit on: module is ok");
        check(trtmc::TrtModuleImplTestPeer::timing_enabled(module),
              "engine timing explicit on: enabled");
        module.forward_async({});
        module.sync();
        check(trtmc::TrtModuleImplTestPeer::timing_event_count(module) == 1,
              "engine timing explicit on: event retained");
    }
    {
        trtmc_test::EnvVarGuard timing("TRTMC_ENGINE_TIMING", "0");
        trtmc::TrtModuleImpl module(engine.get(), engine->createExecutionContext(), stream);
        check(module.ok(), "engine timing off: module is ok");
        check(!trtmc::TrtModuleImplTestPeer::timing_enabled(module), "engine timing off: disabled");
        setenv("TRTMC_ENGINE_TIMING", "1", 1);
        check(!trtmc::TrtModuleImplTestPeer::timing_enabled(module),
              "engine timing off: construction snapshot is stable");
        module.forward_async({});
        module.sync();
        check(trtmc::TrtModuleImplTestPeer::timing_event_count(module) == 0,
              "engine timing off: no event retained");
    }
    {
        trtmc_test::EnvVarGuard timing("TRTMC_ENGINE_TIMING", "invalid");
        trtmc::TrtModuleImpl module(engine.get(), engine->createExecutionContext(), stream);
        check(!module.ok(), "engine timing invalid: module creation fails closed");
    }

    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_engine_timing_environment();

    if (g_failures > 0)
        std::cerr << g_failures << " test(s) FAILED\n";
    return g_failures;
}
