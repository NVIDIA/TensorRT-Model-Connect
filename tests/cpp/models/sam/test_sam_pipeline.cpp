/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-01-SAM
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SamPipeline construction and constructor validation
// Preconditions:  TRT headers and CUDA available
// Postconditions: SamPipeline constructs with mock engines and exposes
//                 prompted segmentation output through the segment facade
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/models/sam/sam_pipeline.h"

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

// Mock: pixel_values[1,3,4,4] float -> image_embeddings[1,4] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_sam_encoder_engine() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{4, {1, 3, 4, 4}});

    float cv[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    auto* cst = n->addConstant(nvinfer1::Dims{2, {1, 4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 4});
    cst->getOutput(0)->setName("image_embeddings");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: image_embeddings + sparse_prompt_embeddings -> masks[1,1,4,4], iou_scores[1].
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_sam_decoder_engine() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* image_embeddings =
        n->addInput("image_embeddings", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {1, 4}});
    auto* sparse_prompt = n->addInput("sparse_prompt_embeddings", nvinfer1::DataType::kFLOAT,
                                      nvinfer1::Dims{2, {2, 2}});

    float masks[16];
    for (int i = 0; i < 16; ++i)
        masks[i] = static_cast<float>(i);
    auto* mask_cst = n->addConstant(nvinfer1::Dims{4, {1, 1, 4, 4}},
                                    nvinfer1::Weights{nvinfer1::DataType::kFLOAT, masks, 16});
    mask_cst->getOutput(0)->setName("masks");
    n->markOutput(*mask_cst->getOutput(0));

    float iou[1] = {0.9f};
    auto* iou_cst = n->addConstant(nvinfer1::Dims{1, {1}},
                                   nvinfer1::Weights{nvinfer1::DataType::kFLOAT, iou, 1});
    iou_cst->getOutput(0)->setName("iou_scores");
    n->markOutput(*iou_cst->getOutput(0));

    n->addIdentity(*image_embeddings)->getOutput(0)->setName("_image_embeddings");
    n->addIdentity(*sparse_prompt)->getOutput(0)->setName("_sparse_prompt");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static trtmc::SamConfig make_test_sam_config() {
    trtmc::SamConfig config;
    config.image_size = 4;
    config.image_embedding_size = 1;
    config.decoder_hidden_size = 2;
    config.num_mask_outputs = 1;
    config.num_multimask_outputs = 1;
    return config;
}

static void test_sam_pipeline() {
    auto enc_engine = build_sam_encoder_engine();
    auto dec_engine = build_sam_decoder_engine();
    if (!enc_engine || !dec_engine) {
        std::cerr << "SKIP sam\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto enc_mod = std::make_unique<trtmc::TrtModuleImpl>(
        enc_engine.get(), enc_engine->createExecutionContext(), stream);
    auto dec_mod = std::make_unique<trtmc::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);
    trtmc::SamPipeline pipeline(std::move(enc_mod), std::move(dec_mod), make_test_sam_config());

    check(std::string(pipeline.pipeline_type()) == "SamPipeline", "sam name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "sam produces output");

    cudaStreamDestroy(stream);
}

static void test_sam_validates() {
    bool threw = false;
    try {
        trtmc::SamPipeline pipeline(nullptr, nullptr, make_test_sam_config());
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "sam: null encoder throws");
}

int main() {
    test_sam_pipeline();
    test_sam_validates();

    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
