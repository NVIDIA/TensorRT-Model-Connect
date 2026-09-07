/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/runtime/primitives/trt_common.h"
#include "core/runtime/tensorrt/trt_module_impl.h"
#include "families/smollm3/runtime/kv_cache.h"
#include "families/smollm3/runtime/pipeline.h"
#include "families/smollm3/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"

#include <NvInfer.h>
#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class StreamFixture {
  public:
    StreamFixture() { cudaStreamCreate(&stream); }
    ~StreamFixture() { cudaStreamDestroy(stream); }
    cudaStream_t stream{nullptr};
};

class MockTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string&) const override { return {9}; }

    std::string decode(const std::vector<std::int32_t>& ids) const override {
        std::string text;
        for (const std::int32_t id : ids)
            text += token_for_id(id);
        return text;
    }

    std::int32_t id_for_token(std::string_view token) const override {
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

    std::string token_for_id(std::int32_t id) const override {
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

class FixedSmolLM3Module final : public trtmc::ITrtModule {
  public:
    FixedSmolLM3Module(cudaStream_t stream, bool prefill, std::vector<std::int32_t> output_tokens,
                       std::int32_t capacity = 16)
        : stream_(stream), prefill_(prefill), output_tokens_(std::move(output_tokens)),
          capacity_(capacity), present_k_(std::vector<std::int64_t>{prefill ? capacity : 1, 4},
                                          trtmc::DType::kFloat32, stream),
          present_v_(std::vector<std::int64_t>{prefill ? capacity : 1, 4}, trtmc::DType::kFloat32,
                     stream) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto rows = static_cast<std::int32_t>(inputs.at("token_id").numel());
        const auto index = std::min(cursor_, output_tokens_.size() - 1);
        const std::int32_t token = output_tokens_[index];
        if (cursor_ < output_tokens_.size())
            ++cursor_;
        logits_.assign(static_cast<std::size_t>(rows) * 4, -10.0F);
        for (std::int32_t row = 0; row < rows; ++row)
            logits_[static_cast<std::size_t>(row) * 4 + token] = 10.0F;
        return {{"logits", trtmc::Tensor{logits_.data(), {rows, 4}, trtmc::DType::kFloat32}}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }

    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "attention_mask" || name == "cache_k_0" ||
               name == "cache_v_0";
    }

    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == "token_id" ? trtmc::DType::kInt32 : trtmc::DType::kFloat32;
    }

    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {capacity_, 4};
        if (name == "present_k_0" || name == "present_v_0")
            return {prefill_ ? capacity_ : 1, 4};
        if (name == "attention_mask")
            return {1, capacity_ + 1};
        return {};
    }

    std::vector<std::int64_t>
    input_profile_shape(const std::string& name, std::int32_t,
                        trtmc::ProfileShapeSelector selector) const override {
        if (name == "token_id")
            return {selector == trtmc::ProfileShapeSelector::kMin ? 1 : capacity_};
        return tensor_shape(name);
    }

    std::int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override {
        const auto binding = bindings_.find(name);
        if (binding != bindings_.end())
            return binding->second;
        if (name == "present_k_0")
            return const_cast<void*>(present_k_.data());
        if (name == "present_v_0")
            return const_cast<void*>(present_v_.data());
        return nullptr;
    }

    void bind_external(const std::string& name, void* pointer) override {
        bindings_[name] = pointer;
    }

    void bind_external(const std::string& name, void* pointer,
                       const std::vector<std::int64_t>&) override {
        bind_external(name, pointer);
    }

    std::int32_t input_rank(const std::string& name) const override {
        return static_cast<std::int32_t>(tensor_shape(name).size());
    }

    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override { cursor_ = 0; }
    void set_timing_label(std::string) override {}
    bool ok() const override {
        return !output_tokens_.empty() && present_k_.ok() && present_v_.ok();
    }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    cudaStream_t stream_{nullptr};
    bool prefill_{false};
    std::vector<std::int32_t> output_tokens_;
    std::size_t cursor_{0};
    std::int32_t capacity_{0};
    mutable std::unordered_map<std::string, void*> bindings_;
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_;
};

trtmc::SmolLM3TextGenConfig make_config() {
    trtmc::SmolLM3TextGenConfig config;
    config.vocab_size = 4;
    config.id_bos = 0;
    config.id_eos = 2;
    config.prefill_max_length = 16;
    config.num_layers = 1;
    return config;
}

std::unique_ptr<trtmc::SmolLM3TextGenerationPipeline>
make_pipeline(cudaStream_t stream, trtmc::SmolLM3TextGenConfig config,
              std::vector<std::int32_t> prefill_tokens = {2},
              std::vector<std::int32_t> decode_tokens = {2},
              std::shared_ptr<trtmc::ITokenizer> tokenizer = std::make_shared<MockTokenizer>()) {
    auto decoder =
        std::make_unique<FixedSmolLM3Module>(stream, false, std::move(decode_tokens), 16);
    auto prefill =
        std::make_unique<FixedSmolLM3Module>(stream, true, std::move(prefill_tokens), 16);
    auto cache = std::make_unique<trtmc::SmolLM3KvCache>(1, 16, 4, stream);
    return std::make_unique<trtmc::SmolLM3TextGenerationPipeline>(
        std::move(decoder), std::move(cache), std::move(config), std::move(tokenizer),
        std::move(prefill));
}

void test_pipeline_construction() {
    StreamFixture fixture;
    auto pipeline = make_pipeline(fixture.stream, make_config());
    check(std::string(pipeline->task()) == trtmc::ITextGeneration::kTask,
          "pipeline exposes text-generation Task API");
    check(pipeline->default_max_new_tokens() == 128, "pipeline default token limit");
}

void test_generate_stops_at_eos() {
    StreamFixture fixture;
    auto pipeline = make_pipeline(fixture.stream, make_config());
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 10;
    const auto result = pipeline->generate_ids({1}, request);
    check(result.token_ids == std::vector<std::int32_t>({1, 2}),
          "generation stops at configured EOS");
}

void test_generate_stops_at_any_default_eos() {
    StreamFixture fixture;
    auto config = make_config();
    config.id_eos = 1;
    config.id_eos_ids = {1, 2};
    auto pipeline = make_pipeline(fixture.stream, config);
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 10;
    check(pipeline->generate_ids({1}, request).token_ids == std::vector<std::int32_t>({1, 2}),
          "generation stops at any default EOS");
}

void test_explicit_eos_override_replaces_default_set() {
    StreamFixture fixture;
    auto config = make_config();
    config.id_eos = 1;
    config.id_eos_ids = {1, 2};
    auto pipeline = make_pipeline(fixture.stream, config);
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 3;
    request.eos_token_id = 3;
    check(pipeline->generate_ids({1}, request).token_ids == std::vector<std::int32_t>({1, 2, 2, 2}),
          "explicit EOS replaces the default set");
}

void test_generate_max_tokens() {
    StreamFixture fixture;
    auto config = make_config();
    config.id_eos = 99;
    auto pipeline = make_pipeline(fixture.stream, config);
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 3;
    check(pipeline->generate_ids({1}, request).token_ids == std::vector<std::int32_t>({1, 2, 2, 2}),
          "generation respects max_new_tokens");
}

void test_argmax() {
    trtmc::SmolLM3SamplingParams params;
    auto sampler = trtmc::create_smollm3_sampler(params);
    const std::vector<float> logits = {0.1F, 0.5F, 0.3F, 0.8F, 0.2F};
    check(
        sampler->sample(logits.data(), static_cast<std::int32_t>(logits.size()), params).token_id ==
            3,
        "argmax selects the largest logit");
    const std::vector<float> single = {42.0F};
    check(sampler->sample(single.data(), 1, params).token_id == 0, "argmax selects the only token");
    check(sampler->sample(nullptr, 0, params).token_id == 0, "empty argmax returns token zero");
}

void test_min_p_uses_the_full_distribution_without_top_k() {
    trtmc::SmolLM3SamplingParams params;
    params.min_p = 0.5F;
    params.seed = 2;
    const std::vector<float> logits = {1.0F, 0.9F, 0.8F};
    auto sampler = trtmc::create_smollm3_sampler(params);
    check(
        sampler->sample(logits.data(), static_cast<std::int32_t>(logits.size()), params).token_id ==
            1,
        "min-p with default top-k samples beyond the argmax");
}

void test_zero_max_tokens() {
    StreamFixture fixture;
    auto pipeline = make_pipeline(fixture.stream, make_config());
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 0;
    check(pipeline->generate_ids({1, 2, 3}, request).token_ids ==
              std::vector<std::int32_t>({1, 2, 3}),
          "zero max_new_tokens returns the prompt unchanged");
}

void test_kv_reset_is_logical_and_masks_stale_rows() {
    StreamFixture fixture;
    FixedSmolLM3Module decoder(fixture.stream, false, {2}, 8);
    trtmc::SmolLM3KvCache cache(1, 8, 4, fixture.stream);
    cache.bind_to(decoder);
    std::vector<float> stale_k(32, 3.25F);
    std::vector<float> stale_v(32, -7.5F);
    check(cache.cache_k(0).copy_from_host(stale_k.data()), "upload stale K rows");
    check(cache.cache_v(0).copy_from_host(stale_v.data()), "upload stale V rows");
    cache.set_position(5);
    cache.reset();

    std::vector<float> actual_k(stale_k.size());
    std::vector<float> actual_v(stale_v.size());
    check(cache.cache_k(0).copy_to_host(actual_k.data()), "download stale K rows");
    check(cache.cache_v(0).copy_to_host(actual_v.data()), "download stale V rows");
    check(actual_k == stale_k && actual_v == stale_v,
          "logical reset preserves allocated KV storage");
    check(cache.position() == 0, "logical reset clears visible cache length");

    trtmc::TensorMap inputs;
    cache.prepare_step(inputs);
    const auto& mask = inputs.at("attention_mask");
    check(mask.shape == std::vector<std::int64_t>({1, 9}),
          "logical reset mask matches current rank-two decoder contract");
    const auto* values = static_cast<const float*>(mask.data);
    check(std::all_of(values, values + 8, [](float value) { return value < -1000.0F; }),
          "logical reset masks stale cache rows");
    check(values[8] == 0.0F, "logical reset keeps the current token visible");
}

trtmc::TrtLogger logger;

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_context_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);
    auto* token = network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask =
        network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});
    const float values[4] = {0.1F, 0.2F, 0.9F, 0.3F};
    auto* logits = network->addConstant(nvinfer1::Dims{1, {4}},
                                        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, values, 4});
    logits->getOutput(0)->setName("logits");
    network->markOutput(*logits->getOutput(0));
    network->addIdentity(*token)->getOutput(0)->setName("_token");
    network->addIdentity(*mask)->getOutput(0)->setName("_mask");
    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

void test_generation_reset_reuses_execution_context() {
    StreamFixture fixture;
    auto engine = build_context_engine();
    check(engine != nullptr, "build context-reset engine");
    if (!engine)
        return;
    trtmc::TrtModuleImpl module(engine.get(), engine->createExecutionContext(), fixture.stream);
    trtmc::TrtModuleImplTestPeer::set_execution_context_name(module, "generation-context");
    module.reset_execution_context();
    check(trtmc::TrtModuleImplTestPeer::execution_context_name(module) == "generation-context",
          "generation reset reuses the loaded execution context");

    std::int32_t token_id = 7;
    std::vector<float> attention_mask(8, 0.0F);
    trtmc::TensorMap inputs;
    inputs["token_id"] = trtmc::Tensor{&token_id, {1}, trtmc::DType::kInt32};
    inputs["attention_mask"] = trtmc::Tensor{attention_mask.data(), {8}, trtmc::DType::kFloat32};
    check(module.forward(inputs).count("logits") == 1,
          "reused execution context remains executable");
}

void test_stop_on_boxed_answer() {
    StreamFixture fixture;
    auto config = make_config();
    config.id_eos = 99;
    auto pipeline = make_pipeline(fixture.stream, config, {1}, {2, 3, 4});
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 10;
    request.stop_on_boxed_answer = true;
    request.stop_check_interval = 1;
    check(pipeline->generate_ids({9}, request).token_ids == std::vector<std::int32_t>({9, 1, 2, 3}),
          "boxed-answer stop truncates after the closing brace");
}

} // namespace

int main() {
    test_argmax();
    test_min_p_uses_the_full_distribution_without_top_k();
    test_pipeline_construction();
    test_generate_stops_at_eos();
    test_generate_stops_at_any_default_eos();
    test_explicit_eos_override_replaces_default_set();
    test_generate_max_tokens();
    test_zero_max_tokens();
    test_kv_reset_is_logical_and_masks_stale_rows();
    test_generation_reset_reuses_execution_context();
    test_stop_on_boxed_answer();
    if (failures != 0)
        std::cerr << failures << " SmolLM3 pipeline test(s) failed\n";
    return failures;
}
