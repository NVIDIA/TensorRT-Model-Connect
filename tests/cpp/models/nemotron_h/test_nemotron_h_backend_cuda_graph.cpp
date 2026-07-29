/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-REC-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-REC-01
// Intent:         TRT backend honors the Nemotron-H CUDA Graph module option
// Preconditions:  TRT + CUDA GPU available, identity engine buildable
// Postconditions: A requested CUDA Graph module is active after backend creation
// =============================================================================

#include "runtime/backend/trt_logger.h"
#include "trtmc/runtime/trt_backend.h"

#include <NvInfer.h>
#include <iostream>

static int failures = 0;
static trtmc::TrtLogger g_logger;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtmc::TrtUniquePtr<nvinfer1::IHostMemory> build_identity_plan() {
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

    return trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
}

static void test_backend_honors_cuda_graph_option() {
    auto plan = build_identity_plan();
    check(plan != nullptr, "backend_option: identity plan built");
    if (!plan)
        return;

    auto* backend = trtmc_create_backend();
    check(backend != nullptr, "backend_option: backend created");
    if (!backend)
        return;

    trtmc::ModuleCreateOptions options;
    options.cuda_graphs = true;
    auto module = backend->create_module(plan->data(), plan->size(), options);
    check(module != nullptr, "backend_option: module created");
    if (module)
        check(module->cuda_graph_active(), "backend_option: CUDA graph option honored");

    module.reset();
    trtmc_destroy_backend(backend);
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 0;
    }

    test_backend_honors_cuda_graph_option();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
