/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/embeddinggemma2/runtime/pipeline.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {
class FixtureModule final : public trtmc::ITrtModule {
  public:
    std::vector<float> hidden = std::vector<float>{3, 0, 0, 4};
    int64_t capacity = 4;
    bool bad_shape = false;
    bool bad_dtype = false;
    bool no_output = false;
    int calls = 0;
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        if (inputs.at("token_id").shape != std::vector<int64_t>{2})
            throw std::runtime_error("EOS not appended");
        auto* ids = static_cast<const int32_t*>(inputs.at("token_id").data);
        if (ids[0] != 2 || ids[1] != 1)
            throw std::runtime_error("unexpected tokens");
        const auto& positions = inputs.at("position_id");
        const auto* values = static_cast<const int32_t*>(positions.data);
        if (positions.shape != std::vector<int64_t>{2} || positions.dtype != trtmc::DType::kInt32 ||
            values[0] != 0 || values[1] != 1)
            throw std::runtime_error("unexpected position IDs");

        trtmc::Tensor out;
        out.data = hidden.data();
        out.shape = bad_shape ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{2, 2};
        out.dtype = bad_dtype ? trtmc::DType::kInt32 : trtmc::DType::kFloat32;
        if (no_output)
            return {};
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
    std::vector<int32_t> ids{2, 1};
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        return ids;
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("EmbeddingGemma2 pipeline contract failed");
}
template <class F>
void rejects(F action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected);
}
void test_pipeline() {
    auto module = std::make_unique<FixtureModule>();
    auto* engine = module.get();
    auto tokenizer = std::make_shared<FixtureTokenizer>();
    trtmc::EmbeddingGemma2Pipeline pipeline(std::move(module), tokenizer, 2, 16, 4, {14},
                                            "fixture");
    auto bindings = pipeline.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "text_to_embedding");
    auto* task = static_cast<trtmc::internal::ITextToEmbedding*>(bindings[0].implementation);
    auto result = task->run({"question", trtmc::internal::EmbeddingRole::Query}, {});
    require(result.values.size() == 2 && std::abs(result.values[0] - 0.6F) < 1e-6F &&
            std::abs(result.values[1] - 0.8F) < 1e-6F);
    require(result.pooling == "mean" && result.normalization == "l2" &&
            result.embedding_space == "fixture");
    require(tokenizer->last_text == "task: search result | query: question");
    (void)task->run({"document", trtmc::internal::EmbeddingRole::Document}, {});
    require(tokenizer->last_text == "title: none | text: document");
    (void)task->run({"plain"}, {});
    require(tokenizer->last_text == "plain");
    engine->hidden = {0, 4, 3, 0};
    (void)pipeline.embed("next");
    require(std::abs(result.values[0] - 0.6F) < 1e-6F);
    for (const auto& invalid :
         {std::vector<int32_t>{}, std::vector<int32_t>{2, 14}, std::vector<int32_t>{2, 16},
          std::vector<int32_t>{2, -1}, std::vector<int32_t>{2, 2, 2, 2, 1}}) {
        tokenizer->ids = invalid;
        const auto calls = engine->calls;
        rejects([&] { (void)pipeline.embed("invalid"); });
        require(engine->calls == calls);
    }
    tokenizer->ids = {2, 1};
    engine->hidden = {0, 0, 0, 0};
    rejects([&] { (void)pipeline.embed("zero"); });
    engine->hidden[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)pipeline.embed("infinite"); });
    engine->hidden = {3, 0, 0, 4};
    engine->bad_shape = true;
    rejects([&] { (void)pipeline.embed("shape"); });
    engine->bad_shape = false;
    engine->bad_dtype = true;
    rejects([&] { (void)pipeline.embed("dtype"); });
    engine->bad_dtype = false;
    engine->no_output = true;
    rejects([&] { (void)pipeline.embed("missing"); });
    rejects(
        [&] { (void)task->run({"invalid", static_cast<trtmc::internal::EmbeddingRole>(99)}, {}); });
    const trtmc::internal::ConfigEntry option{"unknown", int64_t{1}};
    rejects([&] { (void)task->run({"invalid"}, {&option, 1}); });
    rejects([&] {
        trtmc::EmbeddingGemma2Pipeline bad(std::make_unique<FixtureModule>(), tokenizer, 2, 16, 3,
                                           {}, "fixture");
    });
}
} // namespace

int main() {
    try {
        test_pipeline();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
