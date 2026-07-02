/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/backend/trt_logger.h"

#include <NvInfer.h>
#include <cstdint>
#include <vector>

namespace trtmc::test {

inline TrtLogger g_logger;

inline TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_encoder(int32_t mel_bins, int32_t mel_len,
                                                              int32_t enc_out_size) {
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder) {
        return nullptr;
    }
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* mel_inp = network->addInput("mel_features", nvinfer1::DataType::kFLOAT,
                                      nvinfer1::Dims{2, {mel_bins, mel_len}});

    std::vector<float> enc_const(enc_out_size, 0.0F);
    auto* const_enc = network->addConstant(
        nvinfer1::Dims{1, {enc_out_size}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, enc_const.data(), enc_out_size});
    if (!const_enc) {
        return nullptr;
    }

    auto* enc_out = const_enc->getOutput(0);
    enc_out->setName("encoder_output");
    network->markOutput(*enc_out);

    auto* id_mel = network->addIdentity(*mel_inp);
    id_mel->getOutput(0)->setName("_unused_mel");

    auto plan =
        TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        return nullptr;
    }
    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

inline TrtUniquePtr<nvinfer1::ICudaEngine>
build_mock_step_engine(int32_t mask_size, int32_t vocab_size,
                       const std::vector<float>& const_logits) {
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder) {
        return nullptr;
    }
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* token_inp =
        network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask_inp = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT,
                                       nvinfer1::Dims{1, {mask_size}});

    auto* const_w = network->addConstant(
        nvinfer1::Dims{1, {vocab_size}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits.data(), vocab_size});
    if (!const_w) {
        return nullptr;
    }

    auto* out = const_w->getOutput(0);
    out->setName("logits");
    network->markOutput(*out);

    auto* id_tok = network->addIdentity(*token_inp);
    id_tok->getOutput(0)->setName("_unused_token");
    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_unused_mask");

    auto plan =
        TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        return nullptr;
    }
    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

inline TrtUniquePtr<nvinfer1::ICudaEngine>
build_mock_mask_only_engine(int32_t mask_size, int32_t vocab_size,
                            const std::vector<float>& const_logits) {
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder) {
        return nullptr;
    }
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* mask_inp = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT,
                                       nvinfer1::Dims{1, {mask_size}});

    auto* const_w = network->addConstant(
        nvinfer1::Dims{1, {vocab_size}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits.data(), vocab_size});
    if (!const_w) {
        return nullptr;
    }

    auto* out = const_w->getOutput(0);
    out->setName("logits");
    network->markOutput(*out);

    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_unused_mask");

    auto plan =
        TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        return nullptr;
    }
    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

} // namespace trtmc::test
