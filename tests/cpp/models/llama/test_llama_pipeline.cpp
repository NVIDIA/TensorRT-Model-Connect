/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DEC-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-TRT-DEC-01
// Intent:         LlamaTextGenerationPipeline prefill/decode loop, argmax selection, EOS stopping
// Preconditions:  TRT + CUDA GPU available, identity engine built in-process
// Postconditions: Pipeline generates correct tokens, stops at EOS, respects max_new_tokens
// =============================================================================

// =============================================================================
// Test suite: Llama-owned LlamaTextGenerationPipeline copy
// =============================================================================
//
// Tests the LlamaTextGenerationPipeline using a tiny TRT identity engine.
// The identity engine maps token_id[1] → logits[4] (just copies input to output).
// This validates the prefill→decode loop, argmax, and EOS stopping.
//
// For full E2E validation with real models, see tests/test_e2e.py.
// =============================================================================

#include "runtime/models/llama/kv_cache.h"
#include "runtime/models/llama/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"
// pipeline_interface.h was removed; GenerateConfig is in trtmc/pipeline.h
// (already included transitively via runtime/models/llama/pipeline.h)

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

class MockTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        (void)text;
        return {9};
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string out;
        for (int32_t id : ids) {
            out += token_for_id(id);
        }
        return out;
    }

    int32_t id_for_token(std::string_view token) const override {
        if (token == "\\boxed{")
            return 1;
        if (token == "70")
            return 2;
        if (token == "}")
            return 3;
        if (token == " extra")
            return 4;
        return 0;
    }

    std::string token_for_id(int32_t id) const override {
        switch (id) {
        case 1:
            return "\\boxed{";
        case 2:
            return "70";
        case 3:
            return "}";
        case 4:
            return " extra";
        default:
            return "";
        }
    }
};

class SequenceSampler final : public trtmc::LlamaISampler {
  public:
    explicit SequenceSampler(std::vector<int32_t> tokens) : tokens_(std::move(tokens)) {}

    trtmc::LlamaSampleResult sample(const float* logits, int32_t vocab_size,
                                    const trtmc::LlamaSamplingParams& params) override {
        (void)logits;
        (void)vocab_size;
        trtmc::LlamaSampleResult result;
        const std::size_t idx = cursor_ < tokens_.size() ? cursor_ : (tokens_.size() - 1);
        result.token_id = tokens_[idx];
        result.is_eos = (result.token_id == params.eos_token_id);
        if (cursor_ < tokens_.size())
            ++cursor_;
        return result;
    }

    trtmc::LlamaLogitsLocation logits_location() const override {
        return trtmc::LlamaLogitsLocation::HOST;
    }
    const char* sampler_type() const override { return "sequence"; }
    void reset() override { cursor_ = 0; }

  private:
    std::vector<int32_t> tokens_;
    std::size_t cursor_{0};
};

// Build a tiny decoder-like engine:
// Inputs:  token_id [1] int32, attention_mask [8] float32
// Outputs: logits [4] float32
// The engine produces fixed logits [0.1, 0.2, 0.9, 0.3] regardless of input
// (identity on a constant), so argmax always returns 2.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Inputs
    auto* token_inp =
        network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask_inp =
        network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});

    // Constant logits: [0.1, 0.2, 0.9, 0.3] — argmax = index 2
    float const_logits[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* const_w = network->addConstant(
        nvinfer1::Dims{1, {4}}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits, 4});
    if (!const_w)
        return nullptr;

    auto* out = const_w->getOutput(0);
    out->setName("logits");
    network->markOutput(*out);

    // Need to "use" the inputs so TRT doesn't optimize them away
    // Add identity on token_id and mask (mark as outputs too, then unmark)
    // Actually, for a proper test engine, just mark them as used via identity
    auto* id_token = network->addIdentity(*token_inp);
    id_token->getOutput(0)->setName("_unused_token");

    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_unused_mask");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_pipeline_construction() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping test\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::LlamaKvCache>(1, 8, 4, stream);

    trtmc::LlamaTextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 0;
    cfg.id_eos = 2; // argmax will always hit this!
    cfg.has_position_input = false;

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(cache), cfg, stream);

    check(std::string(pipeline.pipeline_type()) == "LlamaTextGenerationPipeline", "pipeline name");

    cudaStreamDestroy(stream);
}

static void test_generate_stops_at_eos() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping test\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::LlamaKvCache>(1, 8, 4, stream);

    trtmc::LlamaTextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 0;
    cfg.id_eos = 2; // argmax of [0.1, 0.2, 0.9, 0.3] = 2 = eos
    cfg.has_position_input = false;

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(cache), cfg, stream);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 10;

    auto result = pipeline.generate_ids({1}, gen_cfg);

    // Input [1] + one generated token (eos=2) → should stop immediately
    check(result.token_ids.size() == 2, "output has 2 tokens (input + eos)");
    check(result.token_ids[0] == 1, "first token is input");
    check(result.token_ids[1] == 2, "second token is eos (argmax=2)");

    cudaStreamDestroy(stream);
}

static void test_generate_max_tokens() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping test\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::LlamaKvCache>(1, 8, 4, stream);

    trtmc::LlamaTextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 0;
    cfg.id_eos = 99; // EOS token that argmax will never produce
    cfg.has_position_input = false;

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(cache), cfg, stream);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 3;

    auto result = pipeline.generate_ids({1}, gen_cfg);

    // Input [1] + 3 generated tokens (all argmax=2, never hits eos=99)
    check(result.token_ids.size() == 4, "output has 4 tokens (input + 3 generated)");
    check(result.token_ids[0] == 1, "first = input");
    check(result.token_ids[1] == 2, "gen 1 = argmax(2)");
    check(result.token_ids[2] == 2, "gen 2 = argmax(2)");
    check(result.token_ids[3] == 2, "gen 3 = argmax(2)");

    cudaStreamDestroy(stream);
}

static void test_argmax() {
    std::vector<float> logits = {0.1f, 0.5f, 0.3f, 0.8f, 0.2f};
    int32_t result = trtmc::LlamaTextGenerationPipeline::argmax(logits);
    check(result == 3, "argmax of [0.1, 0.5, 0.3, 0.8, 0.2] = 3");

    std::vector<float> single = {42.0f};
    check(trtmc::LlamaTextGenerationPipeline::argmax(single) == 0, "argmax of single = 0");

    std::vector<float> empty;
    check(trtmc::LlamaTextGenerationPipeline::argmax(empty) == 0, "argmax of empty = 0");
}

static void test_zero_max_tokens() {
    auto engine = build_mock_decoder();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::LlamaKvCache>(1, 8, 4, stream);

    trtmc::LlamaTextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(cache), cfg, stream);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 0;

    auto result = pipeline.generate_ids({1, 2, 3}, gen_cfg);
    check(result.token_ids.size() == 3, "zero max_new_tokens returns input unchanged");

    cudaStreamDestroy(stream);
}

static void test_kv_reset_is_logical_and_masks_stale_rows() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtmc::LlamaKvCache cache(1, 8, 4, stream);
    std::vector<float> stale_k(32, 3.25F);
    std::vector<float> stale_v(32, -7.5F);
    check(cache.cache_k(0).copy_from_host(stale_k.data()), "upload stale K cache rows");
    check(cache.cache_v(0).copy_from_host(stale_v.data()), "upload stale V cache rows");
    cache.set_position(5);

    cache.reset();

    std::vector<float> actual_k(stale_k.size());
    std::vector<float> actual_v(stale_v.size());
    check(cache.cache_k(0).copy_to_host(actual_k.data()), "download stale K cache rows");
    check(cache.cache_v(0).copy_to_host(actual_v.data()), "download stale V cache rows");
    check(actual_k == stale_k, "logical reset preserves allocated K cache storage");
    check(actual_v == stale_v, "logical reset preserves allocated V cache storage");
    check(cache.position() == 0, "logical reset clears the visible cache length");

    trtmc::TensorMap inputs;
    cache.prepare_step(inputs);
    const auto mask_it = inputs.find("attention_mask");
    check(mask_it != inputs.end(), "logical reset creates an attention mask");
    if (mask_it != inputs.end()) {
        const auto& mask = mask_it->second;
        check(mask.shape == std::vector<int64_t>{9}, "logical reset mask covers cache and token");
        const auto* values = static_cast<const float*>(mask.data);
        bool stale_rows_hidden = values != nullptr;
        for (int32_t i = 0; stale_rows_hidden && i < 8; ++i)
            stale_rows_hidden = values[i] < -1000.0F;
        check(stale_rows_hidden, "logical reset masks every stale cache row");
        check(values != nullptr && values[8] == 0.0F,
              "logical reset keeps the current token visible");
    }

    cudaStreamDestroy(stream);
}

static void test_stop_on_boxed_answer() {
    auto engine = build_mock_decoder();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::LlamaKvCache>(1, 8, 4, stream);
    auto tokenizer = std::make_shared<MockTokenizer>();
    auto sampler = std::make_unique<SequenceSampler>(std::vector<int32_t>{1, 2, 3, 4});

    trtmc::LlamaTextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.has_position_input = false;

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(cache), cfg, stream,
                                                tokenizer, "mock", std::move(sampler));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 10;
    gen_cfg.stop_on_boxed_answer = true;
    gen_cfg.stop_check_interval = 1;

    auto result = pipeline.generate_ids({9}, gen_cfg);
    check(result.token_ids.size() == 4, "boxed-answer stop truncates generation");
    check(result.token_ids[1] == 1, "boxed stop token 1");
    check(result.token_ids[2] == 2, "boxed stop token 2");
    check(result.token_ids[3] == 3, "boxed stop token 3");

    cudaStreamDestroy(stream);
}

int main() {
    test_argmax();
    test_pipeline_construction();
    test_generate_stops_at_eos();
    test_generate_max_tokens();
    test_zero_max_tokens();
    test_kv_reset_is_logical_and_masks_stale_rows();
    test_stop_on_boxed_answer();

    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
