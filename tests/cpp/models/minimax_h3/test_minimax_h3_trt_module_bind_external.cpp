/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static nvinfer1::IExecutionContext* release_execution_context(TrtModuleImpl& module) {
        return std::exchange(module.ctx_, nullptr);
    }

    static bool binding_is_external(const TrtModuleImpl& module, const std::string& name) {
        return module.buffers_.at(name).is_external;
    }
};

} // namespace trtmc

namespace {

int failures = 0;
trtmc::TrtLogger g_logger;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

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

void test_bind_external_failure_preserves_owned_buffer() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    auto* const old_ptr = module.device_ptr("x");
    check(old_ptr != nullptr, "failed external bind: original buffer exists");

    void* ext_ptr = nullptr;
    cudaMalloc(&ext_ptr, 16);
    check(ext_ptr != nullptr, "failed external bind: external alloc ok");

    // A missing execution context makes the TensorRT address update fail.
    // The binding operation must not publish ext_ptr or free old_ptr.
    auto* detached_ctx = trtmc::TrtModuleImplTestPeer::release_execution_context(module);
    bool rejected = false;
    try {
        module.bind_external("x", ext_ptr);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "failed external bind: rejection is surfaced");
    check(module.device_ptr("x") == old_ptr,
          "failed external bind: original buffer remains published");
    check(cudaMemset(old_ptr, 0, 16) == cudaSuccess,
          "failed external bind: original owned buffer remains live");

    delete detached_ctx;
    cudaFree(ext_ptr);
    cudaStreamDestroy(stream);
}

void test_bind_external_same_owned_pointer_preserves_ownership() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    {
        auto* ctx = engine->createExecutionContext();
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
        auto* const owned_ptr = module.device_ptr("x");
        check(owned_ptr != nullptr, "same-pointer external bind: original buffer exists");
        check(!trtmc::TrtModuleImplTestPeer::binding_is_external(module, "x"),
              "same-pointer external bind: original buffer is owned");

        module.bind_external("x", owned_ptr);

        check(module.device_ptr("x") == owned_ptr,
              "same-pointer external bind: device pointer is unchanged");
        check(!trtmc::TrtModuleImplTestPeer::binding_is_external(module, "x"),
              "same-pointer external bind: ownership is preserved");
    }
    cudaStreamDestroy(stream);
}

void test_constructor_prebinding_avoids_owned_io_buffers() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* input = nullptr;
    void* output = nullptr;
    cudaMalloc(&input, 16);
    cudaMalloc(&output, 16);
    {
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> bindings = {
            {"x", input, 16},
            {"y", output, 16},
        };
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, bindings);
        check(module.ok(), "constructor prebinding: module is valid");
        check(module.device_ptr("x") == input && module.device_ptr("y") == output,
              "constructor prebinding: exact external addresses are used");
        check(trtmc::TrtModuleImplTestPeer::binding_is_external(module, "x") &&
                  trtmc::TrtModuleImplTestPeer::binding_is_external(module, "y"),
              "constructor prebinding: no owned I/O buffers are allocated");
    }
    check(cudaMemset(input, 0, 16) == cudaSuccess && cudaMemset(output, 0, 16) == cudaSuccess,
          "constructor prebinding: module destruction preserves caller buffers");
    cudaFree(input);
    cudaFree(output);
    cudaStreamDestroy(stream);
}

void test_constructor_failure_preserves_external_buffers() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* input = nullptr;
    void* output = nullptr;
    cudaMalloc(&input, 16);
    cudaMalloc(&output, 16);
    {
        auto* ctx = engine->createExecutionContext();
        const std::vector<trtmc::ModuleExternalBinding> invalid_bindings = {
            {"x", input, 1}, // smaller than the engine tensor
            {"y", output, 16},
        };
        trtmc::TrtModuleImpl module(engine.get(), ctx, stream, 0, nullptr, invalid_bindings);
        check(!module.ok(), "constructor failure: invalid capacity is rejected");
    }
    check(cudaMemset(input, 0, 16) == cudaSuccess && cudaMemset(output, 0, 16) == cudaSuccess,
          "constructor failure: caller buffers remain live");
    cudaFree(input);
    cudaFree(output);
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_bind_external_failure_preserves_owned_buffer();
    test_bind_external_same_owned_pointer_preserves_ownership();
    test_constructor_prebinding_avoids_owned_io_buffers();
    test_constructor_failure_preserves_external_buffers();
    return failures == 0 ? 0 : 1;
}
