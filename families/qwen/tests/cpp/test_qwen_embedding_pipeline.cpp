/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/embedding_pipeline.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// Protocol fixture: executes the real pipeline without CUDA or TensorRT.
class FixtureModule final : public trtmc::ITrtModule {
  public:
    std::vector<float> hidden = std::vector<float>(2048, 0.0F);
    int64_t capacity = 4;
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        if (inputs.at("token_id").shape != std::vector<int64_t>{2})
            throw std::runtime_error("EOS not appended");
        auto* ids = static_cast<const int32_t*>(inputs.at("token_id").data);
        if (ids[0] != 42 || ids[1] != 151643)
            throw std::runtime_error("unexpected tokens");
        hidden[1024] = 3;
        hidden[1025] = 4;
        trtmc::Tensor out;
        out.data = hidden.data();
        out.shape = {2, 1024};
        out.dtype = trtmc::DType::kFloat32;
        return {{"hidden_states", out}};
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
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        return {42};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

void require_close(float actual, float expected) {
    if (std::abs(actual - expected) > 1.0e-6F)
        throw std::runtime_error("unexpected normalized embedding value");
}

void test_last_token_pool_handles_right_padding() {
    const std::vector<float> hidden{
        1.0F, 0.0F,  0.0F, 2.0F,  3.0F, 4.0F,  9.0F, 9.0F,
        5.0F, 12.0F, 8.0F, 15.0F, 7.0F, 24.0F, 9.0F, 40.0F,
    };
    const std::vector<int32_t> mask{1, 1, 1, 0, 1, 1, 1, 0};

    const auto pooled = trtmc::qwen_last_token_pool_and_normalize(hidden, mask, 2, 4, 2);

    require_close(pooled[0], 0.6F);
    require_close(pooled[1], 0.8F);
    require_close(pooled[2], 0.28F);
    require_close(pooled[3], 0.96F);
}

void test_last_token_pool_handles_left_and_mixed_padding() {
    const std::vector<float> hidden{
        99.0F, 99.0F, 3.0F, 4.0F, 5.0F, 12.0F, 8.0F, 15.0F, 7.0F, 24.0F, 99.0F, 99.0F,
    };
    const std::vector<int32_t> mask{0, 1, 1, 1, 1, 0};

    const auto pooled = trtmc::qwen_last_token_pool_and_normalize(hidden, mask, 2, 3, 2);

    require_close(pooled[0], 5.0F / 13.0F);
    require_close(pooled[1], 12.0F / 13.0F);
    require_close(pooled[2], 7.0F / 25.0F);
    require_close(pooled[3], 24.0F / 25.0F);
}

void test_last_token_pool_rejects_empty_rows() {
    bool threw = false;
    try {
        (void)trtmc::qwen_last_token_pool_and_normalize(std::vector<float>(4, 0.0F),
                                                        std::vector<int32_t>{0, 0}, 1, 2, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw)
        throw std::runtime_error("empty attention-mask row was accepted");
}

void test_public_task_discovery_and_execution() {
    auto module = std::make_unique<FixtureModule>();
    auto* engine = module.get();
    auto tokenizer = std::make_shared<FixtureTokenizer>();
    trtmc::QwenEmbeddingPipeline pipeline(std::move(module), tokenizer, 151643,
                                          "Qwen/Qwen3-Embedding-0.6B");
    trtmc::ITask* task = &pipeline;
    auto* model = dynamic_cast<trtmc::internal::IModel*>(task);
    if (!model)
        throw std::runtime_error("public SDK requires IModel");
    const auto bindings = model->task_bindings();
    if (bindings.size() != 1 || bindings[0].key.id != "text_to_embedding")
        throw std::runtime_error("missing SDK task binding");
    auto* embedding = static_cast<trtmc::internal::ITextToEmbedding*>(bindings[0].implementation);
    const auto result = embedding->run({"question", trtmc::internal::EmbeddingRole::Query}, {});
    require_close(result.values[0], 0.6F);
    require_close(result.values[1], 0.8F);
    if (result.values.size() != 1024 || result.pooling != "last_token" ||
        result.normalization != "l2")
        throw std::runtime_error("semantic metadata mismatch");
    if (tokenizer->last_text != "Instruct: Given a web search query, retrieve relevant passages "
                                "that answer the query\nQuery:question")
        throw std::runtime_error("query format mismatch");
    (void)embedding->run({"document", trtmc::internal::EmbeddingRole::Document}, {});
    if (tokenizer->last_text != "document")
        throw std::runtime_error("document unexpectedly formatted");
    engine->capacity = 1;
    bool rejected = false;
    try {
        (void)embedding->run({"too long"}, {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("oversized input accepted");
}

} // namespace

int main() {
    try {
        test_public_task_discovery_and_execution();
        test_last_token_pool_handles_right_padding();
        test_last_token_pool_handles_left_and_mixed_padding();
        test_last_token_pool_rejects_empty_rows();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
