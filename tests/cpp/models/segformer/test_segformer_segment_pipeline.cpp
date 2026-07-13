/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-01-SEGMENT
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SegmentPipeline construction, class-map parsing, 4D logits,
//                 and single-output mask branch coverage
// Preconditions:  TRT headers and CUDA available
// Postconditions: SegmentPipeline constructs with mock engines and exposes
//                 expected mask outputs
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/models/segformer/segment_pipeline.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

static trtmc::SegformerPreprocessConfig make_test_preprocess_config() {
    trtmc::SegformerPreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    config.output_h = 4;
    config.output_w = 4;
    return config;
}

// Mock: pixel_values[3,4,4] float -> output_mask[1,16] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[16];
    for (int i = 0; i < 16; ++i)
        cv[i] = static_cast<float>(i);
    auto* cst = n->addConstant(nvinfer1::Dims{1, {16}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 16});
    cst->getOutput(0)->setName("output_mask");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> mask[1] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_mask_output() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[1] = {1.0f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {1}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 1});
    cst->getOutput(0)->setName("mask");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> logits[1,2,4,4] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_4d() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[32];
    for (int i = 0; i < 32; ++i)
        cv[i] = static_cast<float>(i % 2);
    auto* cst = n->addConstant(nvinfer1::Dims{4, {1, 2, 4, 4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 32});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> logits[2,4,4] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_2d() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[32];
    for (int i = 0; i < 32; ++i)
        cv[i] = static_cast<float>(i % 2);
    auto* cst = n->addConstant(nvinfer1::Dims{3, {2, 4, 4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 32});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_segment_pipeline() {
    auto engine = build_segment_engine();
    if (!engine) {
        std::cerr << "SKIP segment\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    check(std::string(pipeline.pipeline_type()) == "SegmentPipeline", "segment name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "segment output has values");

    cudaStreamDestroy(stream);
}

static void test_segment_with_class_output() {
    auto engine = build_segment_engine_2d();
    if (!engine) {
        std::cerr << "SKIP segment_2d\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(result.mask.size() == 16, "segment 2d: mask has 16 entries");
    check(result.height == 4, "segment 2d: height = 4");
    check(result.width == 4, "segment 2d: width = 4");

    cudaStreamDestroy(stream);
}

static void test_segment_validates() {
    bool threw = false;
    try {
        trtmc::SegmentPipeline pipeline(nullptr);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "segment: null model throws");
}

static void test_segment_4d_output() {
    auto engine = build_segment_engine_4d();
    if (!engine) {
        std::cerr << "SKIP segment_4d\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(result.mask.size() == 16, "segment 4d: mask has 16 entries");
    check(result.height == 4, "segment 4d: height = 4");
    check(result.width == 4, "segment 4d: width = 4");

    cudaStreamDestroy(stream);
}

static void test_segment_mask_named_output() {
    auto engine = build_segment_engine_mask_output();
    if (!engine) {
        std::cerr << "SKIP segment_mask\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(result.mask.size() == 1, "segment mask: 1 value in mask");

    cudaStreamDestroy(stream);
}

int main() {
    test_segment_pipeline();
    test_segment_with_class_output();
    test_segment_validates();
    test_segment_4d_output();
    test_segment_mask_named_output();

    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
