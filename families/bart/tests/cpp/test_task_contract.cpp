/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bart/runtime/pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// CPU CUDA stubs so tests run without a GPU.
extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (size == 0) {
        *devPtr = nullptr;
        return cudaSuccess;
    }
    *devPtr = std::malloc(size);
    return *devPtr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFree(void* devPtr) {
    std::free(devPtr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void)kind;
    if (count > 0 && dst && src) {
        std::memcpy(dst, src, count);
    }
    return cudaSuccess;
}

} // extern "C"

namespace {

constexpr int32_t kHiddenSize = 32;
constexpr int32_t kDecoderLayers = 2;
constexpr int32_t kMaxSourceLength = 64;
constexpr int32_t kDecoderStartTokenId = 2;
constexpr int32_t kEosTokenId = 2;
constexpr int32_t kBosTokenId = 0;
constexpr int32_t kPadTokenId = 1;
constexpr int32_t kVocabSize = 100;

class MockTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string& text) const override {
        ++encode_calls;
        std::vector<std::int32_t> ids;
        ids.reserve(text.size());
        for (const unsigned char byte : text) {
            ids.push_back(static_cast<std::int32_t>(byte % 90 + 3));
        }
        return ids;
    }

    std::string decode(const std::vector<std::int32_t>& ids) const override {
        ++decode_calls;
        std::string text;
        text.reserve(ids.size());
        for (const auto id : ids) {
            text.push_back(static_cast<char>('a' + (id % 26)));
        }
        return text;
    }

    std::int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(std::int32_t) const override { return {}; }

    mutable int encode_calls{0};
    mutable int decode_calls{0};
};

class MockEncoderModule final : public trtmc::ITrtModule {
  public:
    MockEncoderModule() {
        enc_out_buffer_.assign(static_cast<std::size_t>(kMaxSourceLength) * kHiddenSize, 0.5f);
    }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}

    void forward_async(const trtmc::TensorMap& inputs) override {
        ++calls;
        if (inputs.find("input_ids") != inputs.end()) {
            const auto& tensor = inputs.at("input_ids");
            last_input_numel = static_cast<int32_t>(tensor.numel());
        }
    }

    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return name == "attention_mask"; }
    bool has_output(const std::string& name) const override { return name == "encoder_output"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override {
        return {kMaxSourceLength, kHiddenSize};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) override {
        if (name == "encoder_output")
            return enc_out_buffer_.data();
        return nullptr;
    }

    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return is_ok; }
    void keep_alive(std::shared_ptr<void>) override {}

    bool is_ok{true};
    int calls{0};
    int32_t last_input_numel{0};
    std::vector<float> enc_out_buffer_;
};

class MockDecoderModule final : public trtmc::ITrtModule {
  public:
    MockDecoderModule() { logits_buffer_.assign(kVocabSize, 0.0f); }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        ++calls;
        std::fill(logits_buffer_.begin(), logits_buffer_.end(), 0.0f);
        if (calls <= stop_at_step) {
            logits_buffer_[static_cast<std::size_t>(emit_token)] = 10.0f;
        } else {
            logits_buffer_[static_cast<std::size_t>(kEosTokenId)] = 10.0f;
        }

        trtmc::Tensor logits_tensor;
        logits_tensor.data = logits_buffer_.data();
        logits_tensor.shape = {kVocabSize};
        logits_tensor.dtype = trtmc::DType::kFloat32;
        return {{"logits", logits_tensor}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "cross_attention_mask";
    }
    bool has_output(const std::string& name) const override { return name == "logits"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {kVocabSize}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return is_ok; }
    void keep_alive(std::shared_ptr<void>) override {}

    bool is_ok{true};
    int calls{0};
    int stop_at_step{3};
    int32_t emit_token{15};
    std::vector<float> logits_buffer_;
};

class MockBartInferenceState final : public trtmc::BartInferenceState {
  public:
    void reset() override {
        ++reset_calls;
        position_ = 0;
    }

    void bind_to(trtmc::ITrtModule&) override { ++bind_calls; }
    void prepare_step(trtmc::TensorMap&, int32_t) override { ++prepare_calls; }

    void advance(int32_t n_tokens = 1) override {
        ++advance_calls;
        position_ += n_tokens;
    }

    int32_t position() const override { return position_; }
    int32_t max_length() const override { return 128; }
    int32_t num_layers() const override { return kDecoderLayers; }
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "mock_state"; }
    bool ok() const override { return is_ok; }

    bool is_ok{true};
    int reset_calls{0};
    int bind_calls{0};
    int prepare_calls{0};
    int advance_calls{0};
    int32_t position_{0};
};

std::unique_ptr<trtmc::BartPipeline> make_pipeline(MockEncoderModule** out_enc = nullptr,
                                                   MockDecoderModule** out_dec = nullptr,
                                                   MockBartInferenceState** out_state = nullptr,
                                                   MockTokenizer** out_tok = nullptr) {
    auto enc = std::make_unique<MockEncoderModule>();
    auto dec = std::make_unique<MockDecoderModule>();
    auto state = std::make_unique<MockBartInferenceState>();
    auto tok = std::make_shared<MockTokenizer>();

    if (out_enc)
        *out_enc = enc.get();
    if (out_dec)
        *out_dec = dec.get();
    if (out_state)
        *out_state = state.get();
    if (out_tok)
        *out_tok = tok.get();

    return std::make_unique<trtmc::BartPipeline>(std::move(enc), std::move(dec), std::move(state),
                                                 kHiddenSize, kDecoderLayers, kMaxSourceLength,
                                                 kDecoderStartTokenId, kEosTokenId, kBosTokenId,
                                                 kPadTokenId, kVocabSize, nullptr, std::move(tok));
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("contract assertion failed: ") + message);
}

template <typename Exception, typename Fn>
void rejects(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string("expected exception not thrown: ") + message);
}

void test_task_bindings_and_primary_task() {
    auto pipeline = make_pipeline();
    require(std::string(pipeline->task()) == "conditional_text_generation",
            "primary task must be conditional_text_generation");

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 1, "pipeline must bind exactly 1 task");
    require(bindings[0].task == trtmc::internal::IConditionalTextGeneration::kTask,
            "bound task must be conditional_text_generation");
    require(bindings[0].config_fields.size() == 3, "bound task must expose 3 config fields");
    require(bindings[0].config_fields[0].name == "max_new_tokens",
            "field name must be max_new_tokens");
    require(bindings[0].config_fields[0].kind == trtmc::internal::ConfigKind::I64,
            "field kind must be I64");
    require(bindings[0].config_fields[1].name == "temperature", "field name must be temperature");
    require(bindings[0].config_fields[1].kind == trtmc::internal::ConfigKind::F64,
            "field kind must be F64");
    require(bindings[0].config_fields[2].name == "top_k", "field name must be top_k");
    require(bindings[0].config_fields[2].kind == trtmc::internal::ConfigKind::I64,
            "field kind must be I64");
}

void test_conditional_text_generation_text_input() {
    MockEncoderModule* enc = nullptr;
    MockDecoderModule* dec = nullptr;
    MockBartInferenceState* state = nullptr;
    MockTokenizer* tok = nullptr;

    auto pipeline = make_pipeline(&enc, &dec, &state, &tok);
    dec->stop_at_step = 3;
    dec->emit_token = 25;

    const trtmc::internal::ConditionalTextGenerationRequest req{
        trtmc::internal::TextSource{"Translate this sentence"}};
    const auto result = pipeline->run(req, {});

    require(enc->calls == 1, "encoder must be run once");
    require(dec->calls == 4, "decoder must run 4 steps (3 valid + 1 EOS)");
    require(state->reset_calls == 1, "state reset must be called");
    require(state->bind_calls == 1, "state bind_to must be called");
    require(tok->encode_calls == 1, "tokenizer encode must be called for text input");
    require(tok->decode_calls == 1, "tokenizer decode must be called for output tokens");
    require(result.token_ids.size() == 3, "must produce 3 tokens before EOS");
    require(result.token_ids[0] == 25 && result.token_ids[1] == 25 && result.token_ids[2] == 25,
            "output token ids must match decoder emit");
    require(!result.text.empty(), "result text must be non-empty");
}

void test_conditional_text_generation_token_ids_input() {
    MockEncoderModule* enc = nullptr;
    MockDecoderModule* dec = nullptr;
    MockTokenizer* tok = nullptr;

    auto pipeline = make_pipeline(&enc, &dec, nullptr, &tok);
    dec->stop_at_step = 2;
    dec->emit_token = 18;

    const std::vector<int32_t> ids = {10, 11, 12};
    const trtmc::internal::ConditionalTextGenerationRequest req{
        trtmc::internal::TextSource{trtmc::Span<const int32_t>{ids.data(), ids.size()}}};
    const auto result = pipeline->run(req, {});

    require(tok->encode_calls == 0, "token ids input must not invoke tokenizer encode");
    require(result.token_ids.size() == 2, "must generate 2 tokens");
    require(result.token_ids[0] == 18 && result.token_ids[1] == 18,
            "generated token ids must match decoder emit");
}

void test_max_new_tokens_config() {
    MockDecoderModule* dec = nullptr;
    auto pipeline = make_pipeline(nullptr, &dec, nullptr, nullptr);
    dec->stop_at_step = 100; // Does not emit EOS early
    dec->emit_token = 12;

    const int64_t custom_max = 5;
    const trtmc::internal::ConfigEntry entries[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{custom_max}}};
    const trtmc::internal::ConfigView cfg{entries, 1};

    const trtmc::internal::ConditionalTextGenerationRequest req{
        trtmc::internal::TextSource{"Test prompt"}};
    const auto result = pipeline->run(req, cfg);

    require(result.token_ids.size() == 5, "generation must stop at custom max_new_tokens limit");
}

void test_early_eos_stopping() {
    MockDecoderModule* dec = nullptr;
    auto pipeline = make_pipeline(nullptr, &dec, nullptr, nullptr);
    dec->stop_at_step = 1; // Emits EOS on step 2
    dec->emit_token = 30;

    const trtmc::internal::ConditionalTextGenerationRequest req{
        trtmc::internal::TextSource{"Another prompt"}};
    const auto result = pipeline->run(req, {});

    require(result.token_ids.size() == 1, "generation must stop immediately after EOS token");
    require(result.token_ids[0] == 30, "first token must match");
}

void test_invalid_inputs() {
    auto pipeline = make_pipeline();

    // Empty text
    rejects<std::invalid_argument>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{trtmc::internal::TextSource{""}},
                {});
        },
        "empty text input must throw invalid_argument");

    // Empty token ids span
    const std::vector<int32_t> empty_ids;
    rejects<std::invalid_argument>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{
                    trtmc::Span<const int32_t>{empty_ids.data(), empty_ids.size()}},
                {});
        },
        "empty token span must throw invalid_argument");

    // Negative token id
    const std::vector<int32_t> neg_ids = {5, -1, 10};
    rejects<std::invalid_argument>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{
                    trtmc::Span<const int32_t>{neg_ids.data(), neg_ids.size()}},
                {});
        },
        "negative token id must throw invalid_argument");

    // Token id >= vocab_size
    const std::vector<int32_t> oob_ids = {5, kVocabSize, 10};
    rejects<std::invalid_argument>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{
                    trtmc::Span<const int32_t>{oob_ids.data(), oob_ids.size()}},
                {});
        },
        "out of bounds token id must throw invalid_argument");

    // Input sequence exceeding max_source_length
    const std::vector<int32_t> overlong_ids(static_cast<std::size_t>(kMaxSourceLength) + 1, 10);
    rejects<std::invalid_argument>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{
                    trtmc::Span<const int32_t>{overlong_ids.data(), overlong_ids.size()}},
                {});
        },
        "input exceeding max_source_length must throw invalid_argument");
}

void test_invalid_config() {
    auto pipeline = make_pipeline();

    // Unknown config field
    const trtmc::internal::ConfigEntry unknown[] = {
        {"unknown_param", trtmc::internal::ConfigValue{int64_t{42}}}};
    rejects<trtmc::internal::ConfigError>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{trtmc::internal::TextSource{"a"}},
                {unknown, 1});
        },
        "unknown config field must throw ConfigError");

    // Config type mismatch (double instead of I64)
    const trtmc::internal::ConfigEntry bad_type[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{double{128.0}}}};
    rejects<trtmc::internal::ConfigError>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{trtmc::internal::TextSource{"a"}},
                {bad_type, 1});
        },
        "config type mismatch must throw ConfigError");

    // Non-positive max_new_tokens (0)
    const trtmc::internal::ConfigEntry zero_tokens[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{int64_t{0}}}};
    rejects<trtmc::internal::ConfigError>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{trtmc::internal::TextSource{"a"}},
                {zero_tokens, 1});
        },
        "zero max_new_tokens must throw ConfigError");

    // Negative max_new_tokens (-5)
    const trtmc::internal::ConfigEntry neg_tokens[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{int64_t{-5}}}};
    rejects<trtmc::internal::ConfigError>(
        [&] {
            pipeline->run(
                trtmc::internal::ConditionalTextGenerationRequest{trtmc::internal::TextSource{"a"}},
                {neg_tokens, 1});
        },
        "negative max_new_tokens must throw ConfigError");
}

void test_constructor_validates_dependencies() {
    // Null encoder
    rejects<std::runtime_error>(
        [&] {
            trtmc::BartPipeline(nullptr, std::make_unique<MockDecoderModule>(),
                                std::make_unique<MockBartInferenceState>(), kHiddenSize,
                                kDecoderLayers, kMaxSourceLength, kDecoderStartTokenId, kEosTokenId,
                                kBosTokenId, kPadTokenId, kVocabSize, nullptr,
                                std::make_shared<MockTokenizer>());
        },
        "null encoder must throw runtime_error");

    // Null decoder
    rejects<std::runtime_error>(
        [&] {
            trtmc::BartPipeline(std::make_unique<MockEncoderModule>(), nullptr,
                                std::make_unique<MockBartInferenceState>(), kHiddenSize,
                                kDecoderLayers, kMaxSourceLength, kDecoderStartTokenId, kEosTokenId,
                                kBosTokenId, kPadTokenId, kVocabSize, nullptr,
                                std::make_shared<MockTokenizer>());
        },
        "null decoder must throw runtime_error");

    // Null state
    rejects<std::runtime_error>(
        [&] {
            trtmc::BartPipeline(std::make_unique<MockEncoderModule>(),
                                std::make_unique<MockDecoderModule>(), nullptr, kHiddenSize,
                                kDecoderLayers, kMaxSourceLength, kDecoderStartTokenId, kEosTokenId,
                                kBosTokenId, kPadTokenId, kVocabSize, nullptr,
                                std::make_shared<MockTokenizer>());
        },
        "null state must throw runtime_error");

    // Null tokenizer
    rejects<std::runtime_error>(
        [&] {
            trtmc::BartPipeline(std::make_unique<MockEncoderModule>(),
                                std::make_unique<MockDecoderModule>(),
                                std::make_unique<MockBartInferenceState>(), kHiddenSize,
                                kDecoderLayers, kMaxSourceLength, kDecoderStartTokenId, kEosTokenId,
                                kBosTokenId, kPadTokenId, kVocabSize, nullptr, nullptr);
        },
        "null tokenizer must throw runtime_error");

    // Non-positive vocab size
    rejects<std::runtime_error>(
        [&] {
            trtmc::BartPipeline(
                std::make_unique<MockEncoderModule>(), std::make_unique<MockDecoderModule>(),
                std::make_unique<MockBartInferenceState>(), kHiddenSize, kDecoderLayers,
                kMaxSourceLength, kDecoderStartTokenId, kEosTokenId, kBosTokenId, kPadTokenId, 0,
                nullptr, std::make_shared<MockTokenizer>());
        },
        "zero vocab size must throw runtime_error");
}

void test_legacy_generate() {
    auto pipeline = make_pipeline();
    trtmc::TextGenerationConfig cfg;
    cfg.max_new_tokens = 4;
    const auto res = pipeline->generate("Legacy prompt", cfg);
    require(!res.token_ids.empty(), "legacy generate must produce tokens");
    require(!res.text.empty(), "legacy generate must decode text");
}

void test_result_lifetime() {
    trtmc::internal::TextResult saved_result;
    {
        auto pipeline = make_pipeline();
        saved_result = pipeline->run(
            trtmc::internal::ConditionalTextGenerationRequest{trtmc::internal::TextSource{"Test"}},
            {});
    }
    // Result remains fully valid after pipeline is destroyed
    require(!saved_result.text.empty(), "result text must persist after pipeline destruction");
    require(!saved_result.token_ids.empty(),
            "result token_ids must persist after pipeline destruction");
}

} // namespace

int main() {
    try {
        test_task_bindings_and_primary_task();
        test_conditional_text_generation_text_input();
        test_conditional_text_generation_token_ids_input();
        test_max_new_tokens_config();
        test_early_eos_stopping();
        test_invalid_inputs();
        test_invalid_config();
        test_constructor_validates_dependencies();
        test_legacy_generate();
        test_result_lifetime();
        std::cout << "All BART Task contract tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
