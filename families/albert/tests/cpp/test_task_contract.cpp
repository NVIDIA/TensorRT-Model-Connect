/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/albert/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string& text) const override {
        ++calls;
        std::vector<std::int32_t> ids;
        ids.reserve(text.size());
        for (const unsigned char byte : text) {
            ids.push_back(static_cast<std::int32_t>(byte));
        }
        return ids;
    }
    std::string decode(const std::vector<std::int32_t>&) const override {
        return {};
    }
    std::int32_t id_for_token(std::string_view) const override {
        return 0;
    }
    std::string token_for_id(std::int32_t) const override {
        return {};
    }

    mutable int calls{0};
};

class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& ids_tensor = inputs.at("input_ids");
        last_input_length = static_cast<std::int64_t>(ids_tensor.numel());
        const auto* ids = static_cast<const std::int32_t*>(ids_tensor.data);
        last_input_ids.assign(ids, ids + last_input_length);

        hidden_buffer.assign(static_cast<std::size_t>(last_input_length) * hidden, 0.0F);
        for (std::int64_t i = 0; i < last_input_length; ++i) {
            for (std::int32_t h = 0; h < hidden; ++h) {
                hidden_buffer[static_cast<std::size_t>(i) * hidden + h] =
                    static_cast<float>(2 * i + h + 1);
            }
        }
        return {{"hidden_states",
                 {hidden_buffer.data(),
                  {last_input_length, static_cast<std::int64_t>(hidden)},
                  trtmc::DType::kFloat32}}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override {
        return nullptr;
    }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override {
        return false;
    }
    bool cuda_graph_captured() const override {
        return false;
    }
    int32_t profile_idx() const override {
        return 0;
    }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{"input_ids", {-1}, trtmc::DType::kInt32, true},
                {"attention_mask", {-1}, trtmc::DType::kInt32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"hidden_states", {-1, hidden}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "input_ids" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override {
        return name == "hidden_states";
    }
    trtmc::DType tensor_dtype(const std::string&) const override {
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string&) const override {
        return {-1, hidden};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {512};
    }
    int32_t optimization_profile_count() const override {
        return 1;
    }
    void* device_ptr(const std::string&) const override {
        return nullptr;
    }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override {
        return 1;
    }
    bool input_is_dynamic(const std::string&) const override {
        return true;
    }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override {
        return true;
    }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    std::int32_t hidden{2};
    std::int64_t last_input_length{0};
    std::vector<std::int32_t> last_input_ids;
    std::vector<float> hidden_buffer;
};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(float actual, float expected, float epsilon = 1e-5F) {
    return std::fabs(actual - expected) <= epsilon;
}

template <class Error, class Function>
void rejects(Function function, const char* message) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

std::unique_ptr<trtmc::EncoderPipeline>
make_pipeline(std::string mode, RecordingModule** module_out, FakeTokenizer** tokenizer_out) {
    auto module = std::make_unique<RecordingModule>();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    *module_out = module.get();
    *tokenizer_out = tokenizer.get();
    return std::make_unique<trtmc::EncoderPipeline>(std::move(module), std::move(mode),
                                                    std::move(tokenizer));
}

void test_task_bindings_and_primary_task() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);

    require(std::string(pipeline->task()) == "text_to_pooled_features",
            "primary task must match constructor specification");

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 3, "ALBERT pipeline must bind 3 tasks");

    bool has_pooled = false;
    bool has_token = false;
    bool has_embed = false;
    for (const auto& binding : bindings) {
        if (binding.key.id == trtmc::internal::ITextToPooledFeatures::kTask) {
            has_pooled = true;
        }
        if (binding.key.id == trtmc::internal::ITextToTokenFeatures::kTask) {
            has_token = true;
        }
        if (binding.key.id == trtmc::internal::ITextToEmbedding::kTask) {
            has_embed = true;
        }
    }
    require(has_pooled && has_token && has_embed,
            "bindings must include text_to_pooled_features, text_to_token_features, and text_to_embedding");
}

void test_pooled_features_cls_extraction() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);

    const auto result = pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {});
    require(tokenizer->calls == 1, "text prompt must invoke tokenizer");
    require(module->calls == 1, "engine must execute forward once");
    require(result.pooling == "cls", "pooled features pooling must be 'cls'");
    require(result.normalization == "none", "pooled features normalization must be 'none'");
    require(result.values.size() == 2 && near(result.values[0], 1.0F) &&
                near(result.values[1], 2.0F),
            "pooled features values must match CLS row {1.0, 2.0}");

    const std::vector<std::int32_t> ids{10, 20, 30};
    const trtmc::internal::TextToPooledFeaturesRequest req_ids{
        trtmc::Span<const std::int32_t>{ids.data(), ids.size()}};
    const auto result_ids = pipeline->run(req_ids, {});
    require(tokenizer->calls == 1, "token ID request must bypass tokenizer");
    require(result_ids.values.size() == 2 && near(result_ids.values[0], 1.0F),
            "token ID pooled features must extract CLS row");
}

void test_token_features_extraction() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToTokenFeatures::kTask),
                                  &module, &tokenizer);

    const auto result = pipeline->run(trtmc::internal::TextToTokenFeaturesRequest{"xyz"}, {});
    require(tokenizer->calls == 1, "text prompt must invoke tokenizer");
    require(result.features.rows == 3, "token features rows must equal token count (3)");
    require(result.features.columns == 2, "token features columns must equal hidden dim (2)");
    require(result.features.values.size() == 6, "total values must equal rows * cols");
    require(result.tokens.size() == 3, "token metadata count must equal token count");
    require(result.tokens[0].token_id == static_cast<std::int32_t>('x') &&
                result.tokens[1].token_id == static_cast<std::int32_t>('y') &&
                result.tokens[2].token_id == static_cast<std::int32_t>('z'),
            "token metadata must record exact input IDs");
}

void test_embedding_mean_pools_and_normalizes() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline =
        make_pipeline(std::string(trtmc::internal::ITextToEmbedding::kTask), &module, &tokenizer);

    const auto result = pipeline->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {});
    const float expected_norm = std::sqrt(13.0F);
    require(result.pooling == "mean", "embedding pooling must be 'mean'");
    require(result.normalization == "l2", "embedding normalization must be 'l2'");
    require(result.values.size() == 2, "embedding dimension must match hidden dimension");
    require(near(result.values[0], 2.0F / expected_norm) &&
                near(result.values[1], 3.0F / expected_norm),
            "embedding must be mean-pooled and L2 normalized");
}

void test_missing_tokenizer_and_invalid_inputs() {
    auto no_tok_pipeline =
        std::make_unique<trtmc::EncoderPipeline>(std::make_unique<RecordingModule>(),
                                                 "text_to_pooled_features", nullptr);
    rejects<std::runtime_error>(
        [&] { no_tok_pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{"text"}, {}); },
        "missing tokenizer must reject plain-text request");

    rejects<std::runtime_error>(
        [&] { no_tok_pipeline->run(trtmc::internal::TextToEmbeddingRequest{"text"}, {}); },
        "missing tokenizer must reject text embedding request");
}

void test_constructor_validates_dependencies() {
    rejects<std::runtime_error>(
        [&] {
            trtmc::EncoderPipeline(nullptr, "text_to_pooled_features",
                                   std::make_shared<FakeTokenizer>());
        },
        "null encoder module must be rejected by constructor");
}

void test_result_lifetime() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);

    trtmc::internal::PooledFeaturesResult result;
    {
        std::string temp_prompt = "temporary_prompt_string";
        result = pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{temp_prompt}, {});
    }
    require(result.values.size() == 2, "result data must outlive the request scope");
    require(near(result.values[0], 1.0F), "result values must remain valid after request destroyed");
}

} // namespace

int main() {
    try {
        test_task_bindings_and_primary_task();
        test_pooled_features_cls_extraction();
        test_token_features_extraction();
        test_embedding_mean_pools_and_normalizes();
        test_missing_tokenizer_and_invalid_inputs();
        test_constructor_validates_dependencies();
        test_result_lifetime();
        std::cout << "albert_task_contract: all tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "albert_task_contract failure: " << error.what() << '\n';
        return 1;
    }
}
