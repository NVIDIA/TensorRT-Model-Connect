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

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static void set_execution_context_name(TrtModuleImpl& module, const char* name) {
        module.ctx_->setName(name);
    }

    static std::string execution_context_name(const TrtModuleImpl& module) {
        return module.ctx_->getName();
    }
};

} // namespace trtmc

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

class AdmissionOnlyRuntimeState final : public trtmc::LlamaInferenceState {
  public:
    void reset() override { ++reset_calls; }
    void bind_to(trtmc::TrtModule& module) override {
        (void)module;
        ++bind_calls;
    }
    void prepare_step(trtmc::TensorMap& inputs, int32_t seq_len) override {
        (void)inputs;
        (void)seq_len;
        ++prepare_calls;
    }
    void advance(int32_t n_tokens) override {
        position_ += n_tokens;
        ++advance_calls;
    }
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return 6; }
    bool runtime_owned_kv() const override { return true; }
    int32_t prefill_chunk_limit() const override { return 4; }
    std::uint64_t runtime_kv_capacity_tokens() const override { return 6; }
    std::string runtime_memory_receipt_json() const override { return R"({"kv_allocation_id":1})"; }
    int32_t num_layers() const override { return 1; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "qualification-test"; }
    bool ok() const override { return true; }

    int reset_calls{0};
    int bind_calls{0};
    int prepare_calls{0};
    int advance_calls{0};

  private:
    int32_t position_{0};
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
    gen_cfg.max_new_tokens = 7;

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

static void test_compacting_cache_uses_logical_sequence_limit() {
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
    cfg.id_eos = 99;
    cfg.has_position_input = false;
    cfg.max_sequence_length = 12;
    cfg.kv_cache_compaction = true;

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(cache), cfg, stream);

    trtmc::GenerateConfig accepted;
    accepted.max_new_tokens = 2;
    auto result = pipeline.generate_ids(std::vector<int32_t>(9, 1), accepted);
    check(result.token_ids.size() == 11,
          "compacting cache allows logical sequence beyond physical KV rows");

    trtmc::GenerateConfig rejected;
    rejected.max_new_tokens = 4;
    bool limit_enforced = false;
    try {
        (void)pipeline.generate_ids(std::vector<int32_t>(9, 1), rejected);
    } catch (const std::runtime_error& error) {
        limit_enforced = std::string(error.what()).find("exceeds runtime max sequence length 12") !=
                         std::string::npos;
    }
    check(limit_enforced, "compacting cache still enforces logical sequence limit");

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

static void test_qualification_m_plus_one_rejects_before_attention() {
    auto engine = build_mock_decoder();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    auto state = std::make_unique<AdmissionOnlyRuntimeState>();
    auto* state_observer = state.get();

    trtmc::LlamaTextGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;
    cfg.max_sequence_length = 6;
    cfg.runtime_sequence_admission = trtmc::RuntimeSequenceAdmissionContext{
        /*model_context_limit=*/8,
        /*runtime_kv_capacity_tokens=*/6,
        /*request_context_limit=*/0,
        /*kv_bytes_per_token=*/16,
        /*kv_budget_bytes=*/96,
        /*kv_reserved_bytes=*/96,
    };

    trtmc::LlamaTextGenerationPipeline pipeline(std::move(module), std::move(state), cfg, stream);
    auto* qualification = dynamic_cast<trtmc::IRuntimeMemoryQualificationV1*>(&pipeline);
    check(qualification != nullptr, "qualification V1 is independently discoverable");
    check(qualification != nullptr && qualification->runtime_memory_qualification_api_version() ==
                                          trtmc::kRuntimeMemoryQualificationApiVersionV1,
          "qualification V1 reports the expected version");
    auto* introspection = dynamic_cast<trtmc::IRuntimeMemoryIntrospectionV1*>(&pipeline);
    check(introspection != nullptr,
          "runtime-memory introspection RTTI crosses the Llama model DSO boundary");
    check(introspection != nullptr && introspection->runtime_memory_api_version() == 1 &&
              introspection->runtime_kv_capacity_tokens() == 6,
          "cross-DSO introspection reports the model-owned runtime state");
    const char* c_abi_receipt = trtmc_pipeline_runtime_memory_receipt_json(&pipeline);
    check(c_abi_receipt != nullptr &&
              std::string(c_abi_receipt).find("\"kv_allocation_id\":1") != std::string::npos,
          "core C ABI introspects a pipeline implemented by the Llama model DSO");

    trtmc::RuntimeMemoryQualificationRequestV1 request;
    request.input_ids.assign(9, 1);
    request.max_new_tokens = 0;
    bool typed_rejection = false;
    try {
        (void)qualification->qualify_runtime_memory(request);
    } catch (const trtmc::RuntimeMemoryQualificationAdmissionError& error) {
        typed_rejection = std::string(error.what()).find("semantic model context limit exceeded") !=
                          std::string::npos;
    }
    check(typed_rejection, "M+1 uses the typed qualification admission error");
    check(state_observer->reset_calls == 0, "M+1 rejects before resetting runtime state");
    check(state_observer->bind_calls == 0, "M+1 rejects before binding an attention engine");
    check(state_observer->prepare_calls == 0,
          "M+1 rejects before preparing an attention invocation");
    check(state_observer->advance_calls == 0, "M+1 rejects before advancing KV state");

    trtmc::GenerateConfig generation_request;
    generation_request.max_new_tokens = 0;
    bool resource_rejection = false;
    try {
        (void)pipeline.generate_ids(std::vector<int32_t>(7, 1), generation_request);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        resource_rejection =
            message.find("runtime KV resource capacity exceeded") != std::string::npos &&
            message.find("required_kv_bytes=112") != std::string::npos &&
            message.find("kv_budget_bytes=96") != std::string::npos;
    }
    check(resource_rejection, "normal Llama generation reports physical runtime KV exhaustion");
    check(state_observer->reset_calls == 0,
          "resource rejection occurs before resetting runtime state");
    check(state_observer->bind_calls == 0,
          "resource rejection occurs before binding an attention engine");
    check(state_observer->prepare_calls == 0,
          "resource rejection occurs before preparing an attention invocation");
    check(state_observer->advance_calls == 0,
          "resource rejection occurs before advancing KV state");

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

static void test_generation_reset_reuses_execution_context() {
    auto engine = build_mock_decoder();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto* ctx = engine->createExecutionContext();
    trtmc::TrtModuleImpl module(engine.get(), ctx, stream);
    trtmc::TrtModuleImplTestPeer::set_execution_context_name(module, "generation-context");

    module.reset_execution_context();

    check(trtmc::TrtModuleImplTestPeer::execution_context_name(module) == "generation-context",
          "generation reset reuses the loaded execution context");

    int32_t token_id = 7;
    std::vector<float> attention_mask(8, 0.0F);
    trtmc::TensorMap inputs;
    inputs["token_id"] = trtmc::Tensor{&token_id, {1}, trtmc::DType::kInt32};
    inputs["attention_mask"] = trtmc::Tensor{attention_mask.data(), {8}, trtmc::DType::kFloat32};
    auto outputs = module.forward(inputs);
    check(outputs.count("logits") == 1, "reused execution context remains executable");

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
    gen_cfg.max_new_tokens = 7;
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
    test_compacting_cache_uses_logical_sequence_limit();
    test_zero_max_tokens();
    test_qualification_m_plus_one_rejects_before_attention();
    test_kv_reset_is_logical_and_masks_stale_rows();
    test_generation_reset_reuses_execution_context();
    test_stop_on_boxed_answer();

    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
