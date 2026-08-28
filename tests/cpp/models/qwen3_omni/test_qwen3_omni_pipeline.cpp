/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/qwen3_omni/kv_cache.h"
#include "runtime/models/qwen3_omni/omni_config.h"
#include "runtime/models/qwen3_omni/omni_thinker_plan.h"
#include "runtime/models/qwen3_omni/pipeline.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class OmniFixedTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        encoded_text = text;
        return {1, 2};
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        decoded_ids = ids;
        return ids.empty() ? "" : "native thinker response";
    }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }

    mutable std::string encoded_text;
    mutable std::vector<int32_t> decoded_ids;
};

struct CountingOmniStats {
    int32_t launches{0};
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
};

class CountingOmniModule final : public trtmc::ITrtModule {
  public:
    CountingOmniModule(std::shared_ptr<CountingOmniStats> stats, bool prefill, cudaStream_t stream)
        : stats_(std::move(stats)), prefill_(prefill), stream_(stream),
          device_logits_({1, 4}, trtmc::DType::kFloat32, stream) {
        device_logits_.copy_from_host(host_logits_.data());
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        record(inputs);
        return {{"logits", trtmc::Tensor{host_logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { record(inputs); }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return prefill_ ? 0 : 1; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override { return name == "logits"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return prefill_ ? 2 : 1; }
    void* device_ptr(const std::string& name) const override {
        return name == "logits" ? const_cast<void*>(device_logits_.data()) : nullptr;
    }
    void bind_external(const std::string&, void*) override {}
    int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return prefill_; }
    bool ok() const override { return device_logits_.ok(); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    void record(const trtmc::TensorMap& inputs) {
        ++stats_->launches;
        stats_->shapes.clear();
        for (const auto& [name, tensor] : inputs)
            stats_->shapes[name] = tensor.shape;
    }

    std::shared_ptr<CountingOmniStats> stats_;
    bool prefill_{false};
    cudaStream_t stream_{nullptr};
    std::vector<float> host_logits_{0.1F, 0.9F, 0.9F, 0.3F};
    mutable trtmc::DeviceTensor device_logits_;
};

void test_omni_pipeline_construction() {
    const std::vector<float> thinker_logits = {1.0F, 0.1F, 0.1F, 0.1F};
    auto thinker_engine = trtmc::test::build_mock_step_engine(9, 4, thinker_logits);
    if (!thinker_engine) {
        std::cerr << "WARNING: Could not build thinker engine for OmniPipeline, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto thinker = std::make_unique<trtmc::TrtModuleImpl>(
        thinker_engine.get(), thinker_engine->createExecutionContext(), stream);
    auto thinker_cache = std::make_unique<trtmc::Qwen3OmniKvCache>(0, 8, 0, stream);

    check(thinker->ok(), "omni thinker module ok");
    check(thinker_cache->ok(), "omni thinker cache ok");

    trtmc::OmniConfig cfg;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, cfg, stream,
                                 nullptr, "test-omni");

    check(std::string(pipeline.pipeline_type()) == "OmniPipeline", "OmniPipeline: pipeline_type");
    check(std::string(pipeline.model_id()) == "test-omni", "OmniPipeline: model_id");

    bool missing_tokenizer_failed_closed = false;
    try {
        (void)pipeline.generate("hello");
    } catch (const std::runtime_error& error) {
        missing_tokenizer_failed_closed =
            std::string(error.what()).find("native tokenizer is required") != std::string::npos;
    }
    check(missing_tokenizer_failed_closed,
          "omni text generation fails closed without a native tokenizer");

    cudaStreamDestroy(stream);
}

void test_omni_generate_audio_fails_without_native_talker() {
    const std::vector<float> thinker_logits = {0.1F, 0.1F, 0.1F, 1.0F};
    auto thinker_engine = trtmc::test::build_mock_step_engine(9, 4, thinker_logits);
    if (!thinker_engine) {
        std::cerr << "WARNING: Could not build thinker engine for omni_generate, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto thinker = std::make_unique<trtmc::TrtModuleImpl>(
        thinker_engine.get(), thinker_engine->createExecutionContext(), stream);
    auto thinker_cache = std::make_unique<trtmc::Qwen3OmniKvCache>(0, 8, 0, stream);

    trtmc::OmniConfig cfg;
    cfg.thinker_eos_token_id = 3;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, cfg, stream,
                                 std::make_shared<OmniFixedTokenizer>(), "test-omni-gen");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;

    bool threw = false;
    try {
        (void)pipeline.generate_audio("hello", gen_cfg);
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("native Qwen3-Omni Talker is unavailable") !=
                std::string::npos;
    }
    check(threw, "omni generate_audio fails closed without a native Talker");

    cudaStreamDestroy(stream);
}

void test_omni_generate_text_uses_native_thinker_and_tokenizer() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto thinker = std::make_unique<CountingOmniModule>(decode_stats, false, stream);
    if (!thinker->ok()) {
        std::cerr << "WARNING: Could not allocate Omni test buffers, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto thinker_cache = std::make_unique<trtmc::Qwen3OmniKvCache>(0, 8, 0, stream);
    auto tokenizer = std::make_shared<OmniFixedTokenizer>();
    trtmc::OmniConfig cfg;
    cfg.thinker_vocab_size = 4;
    cfg.thinker_eos_token_id = 99;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, cfg, stream,
                                 tokenizer, "test-omni-text");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    const auto result = pipeline.generate("Reply briefly.", gen_cfg);

    check(result.text == "native thinker response", "omni text generation decodes Thinker IDs");
    check(result.token_ids == std::vector<int32_t>({1, 1}),
          "omni text generation returns generated Thinker IDs");
    check(tokenizer->decoded_ids == result.token_ids,
          "omni text generation decodes only generated IDs");
    check(tokenizer->encoded_text == trtmc::format_omni_chat_prompt("Reply briefly."),
          "omni text generation applies the official chat prompt");
    check(decode_stats->launches == 3,
          "omni text generation runs two prompt steps and one decode step");

    cudaStreamDestroy(stream);
}

void test_omni_generate_text_honors_request_and_configured_eos() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto thinker = std::make_unique<CountingOmniModule>(decode_stats, false, stream);
    if (!thinker->ok()) {
        std::cerr << "WARNING: Could not allocate Omni test buffers, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto thinker_cache = std::make_unique<trtmc::Qwen3OmniKvCache>(0, 8, 0, stream);
    auto tokenizer = std::make_shared<OmniFixedTokenizer>();
    trtmc::OmniConfig cfg;
    cfg.thinker_vocab_size = 4;
    cfg.thinker_eos_token_id = 1;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, cfg, stream,
                                 tokenizer, "test-omni-eos");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    gen_cfg.eos_token_id = 99;
    const auto overridden = pipeline.generate("Use the request EOS.", gen_cfg);

    check(overridden.token_ids == std::vector<int32_t>({1, 1}),
          "omni text generation lets request EOS override the configured EOS");

    decode_stats->launches = 0;
    gen_cfg.eos_token_id = -1;
    const auto configured = pipeline.generate("Use the configured EOS.", gen_cfg);

    check(configured.text.empty(),
          "omni text generation excludes configured EOS from decoded text");
    check(configured.token_ids.empty(),
          "omni text generation excludes configured EOS from token IDs");
    check(tokenizer->decoded_ids.empty(), "omni text generation decodes no EOS token");
    check(decode_stats->launches == 2,
          "omni text generation stops after prompt when its first output is configured EOS");

    cudaStreamDestroy(stream);
}

void test_omni_generate_text_rejects_cache_overflow_before_launch() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto thinker = std::make_unique<CountingOmniModule>(decode_stats, false, stream);
    if (!thinker->ok()) {
        std::cerr << "WARNING: Could not allocate Omni test buffers, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto thinker_cache = std::make_unique<trtmc::Qwen3OmniKvCache>(0, 3, 0, stream);
    auto tokenizer = std::make_shared<OmniFixedTokenizer>();
    trtmc::OmniConfig cfg;
    cfg.thinker_vocab_size = 4;
    cfg.thinker_eos_token_id = 99;
    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, cfg, stream,
                                 tokenizer, "test-omni-capacity");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    bool threw = false;
    try {
        (void)pipeline.generate("Too long.", gen_cfg);
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("KV cache capacity") != std::string::npos;
    }

    check(threw, "omni text generation rejects prompt plus output beyond cache capacity");
    check(decode_stats->launches == 0,
          "omni text generation rejects cache overflow before a Thinker launch");

    cudaStreamDestroy(stream);
}

void test_omni_validates_thinker() {
    bool threw = false;
    try {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        trtmc::OmniConfig cfg;
        trtmc::OmniPipeline p(nullptr, nullptr, nullptr, cfg, stream, nullptr, "x");
        check(false, "null thinker should throw");
        cudaStreamDestroy(stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "omni: null thinker throws");
}

void test_omni_batched_prefill_and_device_argmax() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingOmniStats>();
    auto prefill_stats = std::make_shared<CountingOmniStats>();
    auto decoder = std::make_unique<CountingOmniModule>(decode_stats, false, stream);
    auto prefill = std::make_unique<CountingOmniModule>(prefill_stats, true, stream);
    if (!decoder->ok() || !prefill->ok()) {
        std::cerr << "WARNING: Could not allocate Omni test buffers, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto cache = std::make_unique<trtmc::Qwen3OmniKvCache>(0, 8, 0, stream);
    trtmc::OmniConfig cfg;
    cfg.thinker_hidden_size = 4;
    cfg.thinker_vocab_size = 4;
    cfg.thinker_eos_token_id = 99;

    {
        trtmc::OmniPipeline pipeline(std::move(decoder), std::move(cache), nullptr, cfg, stream,
                                     nullptr, "test-prefill", std::move(prefill));
        const auto output = pipeline.generate_thinker_ids({1, 2, 3}, 2);
        const auto& stats = pipeline.thinker_run_stats();

        check(output == std::vector<int32_t>({1, 1}),
              "omni prefill: token output and lowest-index tie break preserved");
        check(prefill_stats->launches == 1, "omni prefill: one prompt launch");
        check(decode_stats->launches == 1, "omni prefill: decode only generated token");
        check(prefill_stats->shapes["token_id"] == std::vector<int64_t>({3}),
              "omni prefill: token shape");
        check(prefill_stats->shapes["attention_mask"] == std::vector<int64_t>({3, 11}),
              "omni prefill: causal mask shape");
        check(prefill_stats->shapes.count("input_embed") == 0,
              "omni prefill: text-only contract has no external embedding input");
        check(prefill_stats->shapes.count("use_input_embed") == 0,
              "omni prefill: text-only contract has no embedding selector input");
        check(stats.prompt_tokens == 3 && stats.prefill_launches == 1 && stats.decode_launches == 1,
              "omni prefill: deterministic launch counters");
        check(stats.full_logits_d2h == 0, "omni prefill: no full-logits D2H");
    }

    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_omni_pipeline_construction();
    test_omni_generate_text_uses_native_thinker_and_tokenizer();
    test_omni_generate_text_honors_request_and_configured_eos();
    test_omni_generate_text_rejects_cache_overflow_before_launch();
    test_omni_generate_audio_fails_without_native_talker();
    test_omni_validates_thinker();
    test_omni_batched_prefill_and_device_argmax();
    if (failures > 0) {
        std::cerr << failures << " omni pipeline test(s) FAILED\n";
    }
    return failures;
}
