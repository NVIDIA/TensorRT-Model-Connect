/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gpt2/runtime/inference_state.h"
#include "families/gpt2/runtime/pipeline.h"
#include "families/gpt2/runtime/tokenizer.h"
#include "trtmc/internal/config.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_module.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string& text) const override {
        std::vector<std::int32_t> ids;
        for (const unsigned char c : text) {
            ids.push_back(static_cast<std::int32_t>(c));
        }
        return ids;
    }
    std::string decode(const std::vector<std::int32_t>& ids) const override {
        std::string s;
        for (const auto id : ids) {
            s.push_back(static_cast<char>(id));
        }
        return s;
    }
    std::int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(std::int32_t) const override { return {}; }
};

class FakeTrtModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
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
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return false; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

class FakeInferenceState final : public trtmc::Gpt2InferenceState {
  public:
    void reset() override {}
    void bind_to(trtmc::ITrtModule&) override {}
    void prepare_step(trtmc::TensorMap&, int32_t) override {}
    void advance(int32_t) override {}
    int32_t position() const override { return 0; }
    int32_t max_length() const override { return 1024; }
    int32_t num_layers() const override { return 12; }
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "fake"; }
    bool ok() const override { return true; }
};

std::unique_ptr<trtmc::Gpt2TextGenerationPipeline> make_pipeline() {
    trtmc::Gpt2TextGenConfig config;
    config.vocab_size = 50257;
    config.id_bos = 50256;
    config.id_eos = 50256;
    config.num_layers = 12;
    config.prefill_max_length = 1024;

    auto decoder = std::make_unique<FakeTrtModule>();
    auto state = std::make_unique<FakeInferenceState>();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto prefill = std::make_unique<FakeTrtModule>();

    return std::make_unique<trtmc::Gpt2TextGenerationPipeline>(
        std::move(decoder), std::move(state), config, tokenizer, std::move(prefill));
}

void test_task_bindings_and_metadata() {
    auto pipeline = make_pipeline();

    if (std::string_view(pipeline->task()) != trtmc::internal::ITextContinuation::kTask) {
        throw std::runtime_error("pipeline->task() mismatch");
    }

    auto bindings = pipeline->task_bindings();
    if (bindings.size() != 1) {
        throw std::runtime_error("expected exactly 1 task binding");
    }

    const auto& binding = bindings[0];
    const auto expected_key = trtmc::internal::contract_key<trtmc::internal::ITextContinuation>();
    if (binding.key.id != expected_key.id) {
        throw std::runtime_error("binding.key.id mismatch");
    }
    if (binding.key.major != expected_key.major || binding.key.minor != expected_key.minor) {
        throw std::runtime_error("binding version mismatch");
    }
    if (binding.implementation !=
        static_cast<const trtmc::internal::ITextContinuation*>(pipeline.get())) {
        throw std::runtime_error("binding.implementation pointer mismatch");
    }

    const std::vector<std::string> expected_fields = {
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "seed",
        "eos_token_id",
        "use_chat_template",
        "enable_thinking",
        "stop_on_boxed_answer",
        "stop_check_interval",
    };

    for (const auto& expected : expected_fields) {
        bool found = false;
        for (const auto& field : binding.fields) {
            if (field.name == expected) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("missing expected config field: " + expected);
        }
    }
}

void test_unknown_config_rejected() {
    auto pipeline = make_pipeline();
    trtmc::internal::ConfigEntry entries[] = {
        {"invalid_key", trtmc::internal::ConfigValue{static_cast<std::int64_t>(123)}},
    };
    trtmc::internal::ConfigView bad_cfg{entries, 1};
    trtmc::internal::TextContinuationRequest req{std::string_view("Hello")};

    bool caught = false;
    try {
        (void)pipeline->run(req, bad_cfg);
    } catch (const trtmc::internal::ConfigError&) {
        caught = true;
    }
    if (!caught) {
        throw std::runtime_error("expected ConfigError for unknown config key");
    }
}

void test_invalid_config_values_rejected() {
    auto pipeline = make_pipeline();
    trtmc::internal::TextContinuationRequest req{std::string_view("Hello")};

    auto check_rejected = [&](const char* name, trtmc::internal::ConfigValue val) {
        trtmc::internal::ConfigEntry entries[] = {{name, val}};
        trtmc::internal::ConfigView cfg{entries, 1};
        bool caught = false;
        try {
            (void)pipeline->run(req, cfg);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            throw std::runtime_error(std::string("expected invalid_argument for ") + name);
        }
    };

    check_rejected("max_new_tokens", trtmc::internal::ConfigValue{static_cast<std::int64_t>(0)});
    check_rejected("max_new_tokens", trtmc::internal::ConfigValue{static_cast<std::int64_t>(-1)});
    check_rejected("temperature", trtmc::internal::ConfigValue{-0.5});
    check_rejected("top_k", trtmc::internal::ConfigValue{static_cast<std::int64_t>(-1)});
    check_rejected("top_p", trtmc::internal::ConfigValue{-0.1});
    check_rejected("top_p", trtmc::internal::ConfigValue{1.5});
    check_rejected("min_p", trtmc::internal::ConfigValue{-0.01});
    check_rejected("min_p", trtmc::internal::ConfigValue{1.01});
    check_rejected("eos_token_id", trtmc::internal::ConfigValue{static_cast<std::int64_t>(-2)});
    check_rejected("eos_token_id", trtmc::internal::ConfigValue{static_cast<std::int64_t>(60000)});
    check_rejected("use_chat_template", trtmc::internal::ConfigValue{true});
    check_rejected("stop_check_interval",
                   trtmc::internal::ConfigValue{static_cast<std::int64_t>(0)});
}

void test_invalid_inputs_rejected() {
    auto pipeline = make_pipeline();

    // 1. Empty string prompt
    {
        trtmc::internal::TextContinuationRequest req{std::string_view("")};
        bool caught = false;
        try {
            (void)pipeline->run(req, {});
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            throw std::runtime_error("expected invalid_argument for empty string prompt");
        }
    }

    // 2. Empty token IDs
    {
        trtmc::internal::Span<const std::int32_t> empty_span{nullptr, 0};
        trtmc::internal::TextContinuationRequest req{empty_span};
        bool caught = false;
        try {
            (void)pipeline->run(req, {});
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            throw std::runtime_error("expected invalid_argument for empty token IDs");
        }
    }

    // 3. Token ID out of bounds
    {
        const std::int32_t bad_tokens[] = {60000};
        trtmc::internal::Span<const std::int32_t> span{bad_tokens, 1};
        trtmc::internal::TextContinuationRequest req{span};
        bool caught = false;
        try {
            (void)pipeline->run(req, {});
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            throw std::runtime_error("expected invalid_argument for out-of-vocab token ID");
        }
    }

    // 4. use_chat_template with raw token IDs
    {
        trtmc::internal::ConfigEntry entries[] = {
            {"use_chat_template", trtmc::internal::ConfigValue{true}},
        };
        trtmc::internal::ConfigView cfg{entries, 1};
        const std::int32_t tokens[] = {1, 2, 3};
        trtmc::internal::Span<const std::int32_t> span{tokens, 3};
        trtmc::internal::TextContinuationRequest req{span};
        bool caught = false;
        try {
            (void)pipeline->run(req, cfg);
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            throw std::runtime_error(
                "expected invalid_argument for use_chat_template with token IDs");
        }
    }

    // 5. Sequence length exceeding capacity
    {
        std::vector<std::int32_t> long_tokens(2048, 10);
        trtmc::internal::Span<const std::int32_t> span{long_tokens.data(), long_tokens.size()};
        trtmc::internal::TextContinuationRequest req{span};
        bool caught = false;
        try {
            (void)pipeline->run(req, {});
        } catch (const std::invalid_argument&) {
            caught = true;
        }
        if (!caught) {
            throw std::runtime_error(
                "expected invalid_argument for prompt length exceeding max sequence length");
        }
    }
}

} // namespace

int main() {
    try {
        test_task_bindings_and_metadata();
        test_unknown_config_rejected();
        test_invalid_config_values_rejected();
        test_invalid_inputs_rejected();
        std::cout << "gpt2_task_contract: all tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpt2_task_contract failure: " << error.what() << '\n';
        return 1;
    }
}
