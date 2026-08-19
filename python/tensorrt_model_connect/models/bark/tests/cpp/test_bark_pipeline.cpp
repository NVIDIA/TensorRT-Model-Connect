/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "bark_config.h"
#include "kv_cache.h"
#include "pipeline.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_bark_generate_audio() {
    trtmc::BarkConfig bcfg;
    bcfg.hidden_size = 4;
    bcfg.text_pad_token = 5;
    bcfg.semantic_pad_token = 3;
    bcfg.semantic_infer_token = 4;
    bcfg.semantic_input_vocab = 6;
    bcfg.semantic_output_vocab = 10048;
    bcfg.semantic_vocab_size = 4;
    bcfg.n_coarse_codebooks = 2;
    bcfg.codebook_size = 4;
    bcfg.coarse_semantic_pad_token = 10;
    bcfg.coarse_infer_token = 9;
    bcfg.max_coarse_input_length = 4;
    bcfg.max_coarse_history = 4;
    bcfg.sliding_window_len = 10;
    bcfg.greedy = true;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);

    check(sem_cache->ok(), "bark semantic cache ok");
    check(coarse_cache->ok(), "bark coarse cache ok");

    const std::vector<float> sem_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    auto sem_engine = trtmc::test::build_mock_mask_only_engine(513, 5, sem_logits);
    if (!sem_engine) {
        std::cerr << "WARNING: Could not build mock semantic engine, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    const std::vector<float> coarse_logits(12, 0.1F);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!coarse_engine) {
        std::cerr << "WARNING: Could not build mock coarse engine, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);

    std::vector<float> sem_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);

    trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                 std::move(coarse_cache), sem_embed, coarse_embed, bcfg, stream);

    check(std::string(pipeline.pipeline_type()) == "BarkPipeline", "bark pipeline_type");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    auto out = pipeline.generate_audio("", gen_cfg);

    check(out.num_samples > 0, "bark generate_audio produces samples");
    check(out.sample_rate == 24000, "bark generate_audio sample_rate");

    cudaStreamDestroy(stream);
}

void test_bark_constructor_validates_semantic() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);
    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);

    const std::vector<float> coarse_logits(12, 0.1F);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);

    std::vector<float> sem_embed(24, 0.1F);
    std::vector<float> coarse_embed(44, 0.1F);

    bool threw = false;
    try {
        trtmc::BarkPipeline pipeline(nullptr, std::move(coarse), std::move(sem_cache),
                                     std::move(coarse_cache), sem_embed, coarse_embed,
                                     trtmc::BarkConfig{}, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects null semantic module");

    cudaStreamDestroy(stream);
}

void test_bark_constructor_validates_embed() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const std::vector<float> sem_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    auto sem_engine = trtmc::test::build_mock_mask_only_engine(513, 5, sem_logits);
    const std::vector<float> coarse_logits(12, 0.1F);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);

    if (!sem_engine || !coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);
    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);

    bool threw = false;
    try {
        std::vector<float> empty_embed;
        std::vector<float> coarse_embed(44, 0.1F);
        trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                     std::move(coarse_cache), empty_embed, coarse_embed,
                                     trtmc::BarkConfig{}, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects empty semantic embed");

    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_bark_generate_audio();
    test_bark_constructor_validates_semantic();
    test_bark_constructor_validates_embed();
    if (failures > 0) {
        std::cerr << failures << " bark pipeline test(s) FAILED\n";
    }
    return failures;
}
