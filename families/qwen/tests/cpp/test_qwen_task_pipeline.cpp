/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/pipeline.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
class FixtureModule final : public trtmc::ITrtModule {
  public:
    int64_t capacity = 4;
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        throw std::runtime_error("zero-token request executed an engine");
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
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {capacity};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};
class FixtureTokenizer final : public trtmc::ITokenizer {
  public:
    mutable std::string last_text;
    mutable int encode_calls = 0;
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        ++encode_calls;
        return {42};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

class FixtureState final : public trtmc::QwenInferenceState {
  public:
    void reset() override { throw std::runtime_error("zero-token request reset the cache"); }
    void bind_to(trtmc::ITrtModule&) override {}
    void prepare_step(trtmc::TensorMap&, int32_t) override {}
    void advance(int32_t) override {}
    int32_t position() const override { return 0; }
    int32_t max_length() const override { return 8; }
    int32_t num_layers() const override { return 1; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "fixture"; }
    bool ok() const override { return true; }
};
void require(bool value) {
    if (!value)
        throw std::runtime_error("Qwen Task pipeline contract failed");
}
template <class Error, class Function>
void rejects(Function call) {
    try {
        call();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("invalid Qwen request was accepted");
}
} // namespace
int main() {
    using namespace trtmc;
    using namespace trtmc::internal;
    try {
        auto tokenizer = std::make_shared<FixtureTokenizer>();
        QwenTextGenConfig settings;
        settings.vocab_size = 100;
        settings.chat_template_format = "chatml";
        auto make = [&](std::shared_ptr<void> owner = nullptr) {
            return std::make_unique<QwenTextGenerationPipeline>(
                std::make_unique<FixtureModule>(), std::make_unique<FixtureState>(), settings,
                tokenizer, std::make_unique<FixtureModule>(), owner);
        };
        auto model = make();
        const auto bindings = model->task_bindings();
        require(bindings.size() == 2);
        require(bindings[0].key.id == ITextContinuation::kTask);
        require(bindings[1].key.id == IStreamingTextContinuation::kTask);
        const ConfigEntry zero[] = {{"max_new_tokens", std::int64_t{0}}};
        const auto text = model->run({std::string_view{"hello"}}, zero);
        require(text.text.empty() && text.token_ids.empty());
        require(tokenizer->last_text == "hello" && tokenizer->encode_calls == 1);
        std::int32_t tokens[] = {3, 1, 4};
        const auto ids = model->run({Span<const std::int32_t>{tokens}}, zero);
        require(ids.text.empty() && ids.token_ids.empty() && tokenizer->encode_calls == 1);
        const ConfigEntry system[] = {{"max_new_tokens", std::int64_t{0}},
                                      {"use_chat_template", true},
                                      {"system_prompt", std::string_view{"system"}}};
        (void)model->run({std::string_view{"user"}}, system);
        require(tokenizer->last_text.find("<|im_start|>system\nsystem<|im_end|>") == 0);
        rejects<std::invalid_argument>(
            [&] { model->run({Span<const std::int32_t>{tokens}}, system); });
        for (const std::int32_t bad : {-1, 100}) {
            const TextContinuationRequest request{Span<const std::int32_t>{&bad, 1}};
            rejects<std::invalid_argument>([&] { model->run(request, zero); });
            rejects<std::invalid_argument>([&] { model->start(request, zero); });
        }
        const TextContinuationRequest missing{Span<const std::int32_t>{nullptr, 1}};
        rejects<std::invalid_argument>([&] { model->run(missing, zero); });
        rejects<std::invalid_argument>([&] { model->start(missing, zero); });
        const ConfigEntry unknown[] = {{"unknown", true}};
        rejects<ConfigError>([&] { model->run({std::string_view{"hello"}}, unknown); });
        (void)model->run({std::string_view{"hello"}}, zero);
        auto tp = make(std::make_shared<int>(1));
        require(tp->task_bindings().size() == 1);
        rejects<UnsupportedTask>([&] { tp->start({std::string_view{"hello"}}, zero); });
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
