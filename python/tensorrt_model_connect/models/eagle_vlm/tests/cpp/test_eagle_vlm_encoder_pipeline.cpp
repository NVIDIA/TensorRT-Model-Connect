/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         EncoderPipeline construction, type checks, embed/encode/rerank
//                 paths, int32 mask branch, constructor validation,
//                 no-tokenizer throws, and score-named output
// Preconditions:  TRT headers and CUDA available
// Postconditions: Pipelines construct with mock engines and expose correct interfaces;
//                 embed/encode/rerank methods return non-empty results;
//                 rerank uses the checkpoint prompt template, separator tokens,
//                 and pooling contract;
//                 invalid inputs and score shapes are rejected with std::exception;
//                 score output name, 4D logits shape, and size==1 output branch covered
// =============================================================================

// =============================================================================
// Test suite: EncoderPipeline
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <NvInfer.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

// ---------------------------------------------------------------------------
// Inline FixedTokenizer — encodes any string as {1,2,3,4}
// ---------------------------------------------------------------------------
class FixedTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        last_encoded_text = text;
        return {1, 2, 3, 4};
    }
    std::string decode(const std::vector<int32_t>&) const override { return "test"; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }

    mutable std::string last_encoded_text;
};

class SeparatorTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override {
        return {128000, 10, 720, 720, 20};
    }
    std::string decode(const std::vector<int32_t>&) const override { return "test"; }
    int32_t id_for_token(std::string_view token) const override {
        if (token == "ĠĊ")
            return 720;
        if (token == "ĠĊĠĊ")
            return 33006;
        return -1;
    }
    std::string token_for_id(int32_t) const override { return ""; }
};

class FakeScoreModule final : public trtmc::ITrtModule {
  public:
    FakeScoreModule(std::vector<float> scores, std::vector<int64_t> shape)
        : scores_(std::move(scores)), shape_(std::move(shape)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto input = inputs.find("input_ids");
        if (input != inputs.end() && input->second.data && input->second.shape.size() == 1) {
            const auto size = static_cast<std::size_t>(input->second.shape.front());
            const auto* ids = static_cast<const int32_t*>(input->second.data);
            input_ids_.assign(ids, ids + size);
        }
        return {{"score", trtmc::Tensor{scores_.data(), shape_, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {
            {"input_ids", {4}, trtmc::DType::kInt32, true},
            {"attention_mask", {4}, trtmc::DType::kInt32, true},
        };
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"score", shape_, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "input_ids" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override { return name == "score"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return shape_; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {4};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
    const std::vector<int32_t>& input_ids() const { return input_ids_; }

  private:
    std::vector<float> scores_;
    std::vector<int64_t> shape_;
    std::vector<int32_t> input_ids_;
};

// ---------------------------------------------------------------------------
// Engine builders
// ---------------------------------------------------------------------------

// Mock: input_ids[4] int32 + attention_mask[4] float32 -> output_embeddings[8] float (flat)
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    float cv[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {8}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("output_embeddings");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: input_ids[4] int32 + attention_mask[4] float32 -> output_hidden[4,2] float (2D)
// infer_output_hidden_dim returns 2 (last axis of [4,2])
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine_2d() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    // Output shape [4, 2] so infer_output_hidden_dim returns 2
    float cv[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    auto* cst = n->addConstant(nvinfer1::Dims{2, {4, 2}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("output_hidden");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: input_ids[4] int32 + attention_mask[4] int32 -> output_hidden[8] float
// Used to cover the int32 mask branch in encode_ids()
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine_int32_mask() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // INT32 attention_mask to exercise engine_mask_is_int32() == true branch
    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});

    float cv[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {8}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("output_hidden");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: input_ids[4] int32 (no attention_mask) -> score[1] float
// Covers: engine_mask_is_int32() return false (line 52) and name.find("score") (line 164)
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_encoder_engine_score_output() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Only input_ids — NO attention_mask — so engine_mask_is_int32 returns false (line 52)
    auto* ids = n->addInput("input_ids", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {4}});

    float cv[1] = {0.5f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {1}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 1});
    cst->getOutput(0)->setName("score"); // name.find("score") covers line 164
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*ids)->getOutput(0)->setName("_i");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_encoder_pipeline() {
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::EncoderPipeline pipeline(std::move(module), "embedding");

    check(std::string(pipeline.pipeline_type()) == "EncoderPipeline", "encoder name");

    auto result = pipeline.encode_ids({1, 2, 3, 0});
    check(result.dim == 8, "encoder output dim = 8");
    check(result.data.size() == 8, "embedding has 8 floats");

    cudaStreamDestroy(stream);
}

static void test_encoder_embed_mode() {
    auto engine = build_encoder_engine_2d();
    if (!engine) {
        std::cerr << "SKIP encoder_embed\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    trtmc::EncoderPipeline pipeline(std::move(module), "embedding", tokenizer);

    // embed() calls tokenizer->encode() -> {1,2,3,4}, then encode_ids, then
    // mean_pool_and_normalize(data, 4, 2) since mode_=="embedding" and raw.dim(8) >= 4*2
    auto result = pipeline.embed("hello");
    check(!result.data.empty(), "embed: result has data");
    check(result.dim == 2, "embed: hidden dim = 2 after mean-pool");

    cudaStreamDestroy(stream);
}

static void test_encoder_encode_mode() {
    auto engine = build_encoder_engine_2d();
    if (!engine) {
        std::cerr << "SKIP encoder_encode\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    trtmc::EncoderPipeline pipeline(std::move(module), "encode", tokenizer);

    // encode() extracts CLS token: first hidden_dim(=2) values from raw output
    auto result = pipeline.encode("hello");
    check(result.dim == 2, "encode: CLS dim = 2");
    check(result.data.size() == 2, "encode: data size = 2");

    cudaStreamDestroy(stream);
}

static void test_encoder_rerank() {
    auto tokenizer = std::make_shared<FixedTokenizer>();
    auto module = std::make_unique<FakeScoreModule>(std::vector<float>{0.1f, 0.2f, 0.3f, 0.9f},
                                                    std::vector<int64_t>{4, 1});
    trtmc::EncoderPipeline pipeline(std::move(module), "reranking", tokenizer, "", "avg");

    // The pinned Nemotron reranker uses average pooling over valid positions.
    float score = pipeline.rerank("query", "doc");
    check(std::abs(score - 0.375f) < 1e-6f, "rerank: averages valid token scores");
    check(tokenizer->last_encoded_text == "question:query \n \n passage:doc",
          "rerank: uses the checkpoint prompt template");
}

static void test_encoder_rerank_canonicalizes_checkpoint_separator_tokens() {
    auto tokenizer = std::make_shared<SeparatorTokenizer>();
    auto module = std::make_unique<FakeScoreModule>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
                                                    std::vector<int64_t>{4, 1});
    auto* module_ptr = module.get();
    trtmc::EncoderPipeline pipeline(std::move(module), "reranking", tokenizer, "", "avg");

    const float score = pipeline.rerank("query", "doc");
    check(std::abs(score - 2.5f) < 1e-6f, "rerank separator: pools the canonical token sequence");
    check(module_ptr->input_ids() == std::vector<int32_t>({128000, 10, 33006, 20}),
          "rerank separator: merges the native separator pieces like HF");
}

static void test_encoder_rerank_preserves_last_and_scalar_outputs() {
    auto tokenizer = std::make_shared<FixedTokenizer>();
    auto last_module = std::make_unique<FakeScoreModule>(std::vector<float>{0.1f, 0.2f, 0.3f, 0.9f},
                                                         std::vector<int64_t>{4, 1});
    trtmc::EncoderPipeline last_pipeline(std::move(last_module), "reranking", tokenizer, "",
                                         "last");
    check(last_pipeline.rerank("query", "doc") == 0.9f, "rerank: preserves last-token pooling");

    auto scalar_module =
        std::make_unique<FakeScoreModule>(std::vector<float>{0.7f}, std::vector<int64_t>{1});
    trtmc::EncoderPipeline scalar_pipeline(std::move(scalar_module), "reranking", tokenizer, "",
                                           "avg");
    check(scalar_pipeline.rerank("query", "doc") == 0.7f,
          "rerank: preserves an engine-reduced scalar");
}

static void test_encoder_rerank_rejects_invalid_score_shape() {
    auto tokenizer = std::make_shared<FixedTokenizer>();
    auto module = std::make_unique<FakeScoreModule>(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                    std::vector<int64_t>{2, 2});
    trtmc::EncoderPipeline pipeline(std::move(module), "reranking", tokenizer);

    bool threw = false;
    try {
        pipeline.rerank("query", "doc");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "rerank: invalid score shape throws");

    auto short_module = std::make_unique<FakeScoreModule>(std::vector<float>{0.1f, 0.2f, 0.3f},
                                                          std::vector<int64_t>{3, 1});
    trtmc::EncoderPipeline short_pipeline(std::move(short_module), "reranking", tokenizer, "",
                                          "avg");
    threw = false;
    try {
        short_pipeline.rerank("query", "doc");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "rerank: score output shorter than the token sequence throws");

    auto unsupported_module = std::make_unique<FakeScoreModule>(
        std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, std::vector<int64_t>{4, 1});
    trtmc::EncoderPipeline unsupported_pipeline(std::move(unsupported_module), "reranking",
                                                tokenizer, "", "median");
    threw = false;
    try {
        unsupported_pipeline.rerank("query", "doc");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "rerank: unsupported pooling mode throws");
}

static void test_encoder_int32_mask() {
    auto engine = build_encoder_engine_int32_mask();
    if (!engine) {
        std::cerr << "SKIP encoder_int32_mask\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto tokenizer = std::make_shared<FixedTokenizer>();
    // mode="embedding" with int32 mask covers engine_mask_is_int32() == true path
    trtmc::EncoderPipeline pipeline(std::move(module), "embedding", tokenizer);

    auto result = pipeline.embed("hello");
    check(!result.data.empty(), "int32_mask: result has data");

    cudaStreamDestroy(stream);
}

static void test_encoder_validates() {
    // Null encoder -> constructor throws std::exception
    bool threw = false;
    try {
        trtmc::EncoderPipeline pipeline(nullptr, "embedding");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "encoder: null encoder throws");
}

static void test_encoder_score_output() {
    // Engine with output "score" and no attention_mask input covers:
    //   engine_mask_is_int32() return false path (line 52)
    //   name.find("score") evaluation in encode_ids (line 164)
    auto engine = build_encoder_engine_score_output();
    if (!engine) {
        std::cerr << "SKIP encoder_score\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::EncoderPipeline pipeline(std::move(module), "embedding");

    auto result = pipeline.encode_ids({1, 2, 3, 4});
    check(!result.data.empty(), "score output: data not empty");
    check(result.dim == 1, "score output: dim=1");

    cudaStreamDestroy(stream);
}

static void test_encoder_no_tokenizer_embed() {
    // embed() without tokenizer covers line 75 throw
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder_no_tok_embed\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::EncoderPipeline pipeline(std::move(module), "embedding"); // no tokenizer

    bool threw = false;
    try {
        pipeline.embed("hello");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "embed: no tokenizer throws");

    cudaStreamDestroy(stream);
}

static void test_encoder_no_tokenizer_encode() {
    // encode() without tokenizer covers line 98 throw
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder_no_tok_encode\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::EncoderPipeline pipeline(std::move(module), "encode"); // no tokenizer

    bool threw = false;
    try {
        pipeline.encode("hello");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "encode: no tokenizer throws");

    cudaStreamDestroy(stream);
}

static void test_encoder_no_tokenizer_rerank() {
    // rerank() without tokenizer covers line 117 throw
    auto engine = build_encoder_engine();
    if (!engine) {
        std::cerr << "SKIP encoder_no_tok_rerank\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::EncoderPipeline pipeline(std::move(module), "rerank"); // no tokenizer

    bool threw = false;
    try {
        pipeline.rerank("q", "d");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "rerank: no tokenizer throws");

    cudaStreamDestroy(stream);
}

int main() {
    test_encoder_pipeline();
    test_encoder_embed_mode();
    test_encoder_encode_mode();
    test_encoder_rerank();
    test_encoder_rerank_canonicalizes_checkpoint_separator_tokens();
    test_encoder_rerank_preserves_last_and_scalar_outputs();
    test_encoder_rerank_rejects_invalid_score_shape();
    test_encoder_int32_mask();
    test_encoder_validates();
    test_encoder_score_output();
    test_encoder_no_tokenizer_embed();
    test_encoder_no_tokenizer_encode();
    test_encoder_no_tokenizer_rerank();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
