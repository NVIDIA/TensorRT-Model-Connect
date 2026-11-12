/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract of the ministral3 pipeline with a scripted engine: Task
// bindings, role prompts, pooling, token mapping, input validation and engine
// faults. It establishes no GPU or model parity.

#include "families/ministral3/runtime/pipeline.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace trtmc;
using namespace trtmc::internal;
using trtmc::ministral3::EncoderPipeline;
using trtmc::ministral3::EncoderSettings;
using trtmc::ministral3::TekkenTokenizer;

constexpr std::int64_t kMaxLength = 6;

void check(bool condition, const char* name) {
    if (!condition)
        throw std::runtime_error(std::string("ministral3 task contract failed: ") + name);
}

template <class Error, class Function>
void rejects(Function&& function, const char* name) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(std::string("expected failure was not raised: ") + name);
}

std::shared_ptr<const TekkenTokenizer> tokenizer() {
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" /
                      "tekken_tokenizer.json";
    std::ifstream input(path);
    const auto data = nlohmann::json::parse(input).at("tokenizer").dump();
    return TekkenTokenizer::load(data.data(), data.size());
}

class Engine final : public ITrtModule {
  public:
    int fault{0};
    std::int64_t profile_max{kMaxLength};
    std::vector<std::vector<std::int32_t>> calls;
    std::vector<float> states;

    TensorMap forward(const TensorMap& input) override {
        const auto& ids = input.at("input_ids");
        check(ids.dtype == DType::kInt32 && ids.shape.size() == 1, "engine input");
        const auto* data = static_cast<const std::int32_t*>(ids.data);
        calls.emplace_back(data, data + ids.numel());
        if (fault == 1)
            throw std::runtime_error("engine failure");
        states.clear();
        for (const auto id : calls.back()) {
            states.push_back(static_cast<float>(id));
            states.push_back(fault == 7 ? std::numeric_limits<float>::quiet_NaN()
                                        : 2.0f * static_cast<float>(id));
        }
        if (fault == 2)
            return {};
        Tensor output{states.data(), {ids.shape[0], 2}, DType::kFloat32};
        if (fault == 3)
            output.dtype = DType::kFloat16;
        if (fault == 4)
            output.shape[0] += 1;
        if (fault == 5)
            output.shape[1] = 3;
        if (fault == 6)
            output.data = nullptr;
        return {{"hidden_states", output}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string& name) const override {
        return name == "input_ids" ? DType::kInt32 : DType::kFloat32;
    }
    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        return name == "hidden_states" ? std::vector<std::int64_t>{-1, 2}
                                       : std::vector<std::int64_t>{-1};
    }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  ProfileShapeSelector) const override {
        return {profile_max};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

EncoderSettings settings(std::string task, std::int64_t vocab) {
    return {std::move(task), 2, vocab, kMaxLength, "q: ", "d: "};
}

std::vector<std::int32_t> ids_of(const TekkenTokenizer& tokenizer, const std::string& text) {
    std::vector<std::int32_t> ids;
    for (const auto& token : tokenizer.encode(text))
        ids.push_back(token.id);
    return ids;
}

std::vector<float> mean_of(const std::vector<std::int32_t>& ids) {
    double sum = 0.0;
    for (const auto id : ids)
        sum += id;
    const auto mean = static_cast<float>(sum / static_cast<double>(ids.size()));
    return {mean, 2.0f * mean};
}

void construction_contract() {
    const auto tok = tokenizer();
    const auto vocab = tok->vocab_size();
    rejects<UnsupportedTask>(
        [&] {
            EncoderPipeline(std::make_unique<Engine>(), tok, settings("text_continuation", vocab));
        },
        "unknown task");
    rejects<std::invalid_argument>(
        [&] {
            EncoderPipeline(std::make_unique<Engine>(), tok,
                            settings("text_to_embedding", vocab + 1));
        },
        "vocabulary mismatch");
    auto wide = settings("text_to_embedding", vocab);
    wide.hidden_size = 3;
    rejects<std::invalid_argument>([&] { EncoderPipeline(std::make_unique<Engine>(), tok, wide); },
                                   "hidden mismatch");
    auto engine = std::make_unique<Engine>();
    engine->profile_max = kMaxLength + 1;
    rejects<std::invalid_argument>(
        [&] { EncoderPipeline(std::move(engine), tok, settings("text_to_embedding", vocab)); },
        "profile mismatch");
    rejects<std::invalid_argument>(
        [&] { EncoderPipeline(nullptr, tok, settings("text_to_embedding", vocab)); }, "no engine");
}

void task_contract(const char* primary) {
    const auto tok = tokenizer();
    auto engine = std::make_unique<Engine>();
    auto* observed = engine.get();
    EncoderPipeline model(std::move(engine), tok, settings(primary, tok->vocab_size()));
    check(std::string(model.task()) == primary, "primary task");
    const auto bindings = model.task_bindings();
    check(bindings.size() == 3 && bindings[0].key.id == ITextToEmbedding::kTask &&
              bindings[1].key.id == ITextToPooledFeatures::kTask &&
              bindings[2].key.id == ITextToTokenFeatures::kTask,
          "task bindings");
    auto* embedding = static_cast<ITextToEmbedding*>(bindings[0].implementation);
    auto* pooled = static_cast<ITextToPooledFeatures*>(bindings[1].implementation);
    auto* token = static_cast<ITextToTokenFeatures*>(bindings[2].implementation);

    const std::pair<EmbeddingRole, const char*> roles[] = {{EmbeddingRole::Default, "hi"},
                                                           {EmbeddingRole::Query, "q: hi"},
                                                           {EmbeddingRole::Document, "d: hi"}};
    for (const auto& [role, prompted] : roles) {
        const auto result = embedding->run({"hi", role}, {});
        const auto expected_ids = ids_of(*tok, prompted);
        check(observed->calls.back() == expected_ids, "role prompt");
        const auto mean = mean_of(expected_ids);
        const auto norm = std::sqrt(mean[0] * mean[0] + mean[1] * mean[1]);
        check(result.values.size() == 2 && std::fabs(result.values[0] - mean[0] / norm) < 1e-6f &&
                  std::fabs(result.values[1] - mean[1] / norm) < 1e-6f,
              "normalized mean");
        check(result.pooling == "mean" && result.normalization == "l2" &&
                  result.embedding_space.empty(),
              "embedding metadata");
    }

    const auto text_ids = ids_of(*tok, "the hello");
    const auto features = pooled->run({std::string_view("the hello")}, {});
    check(features.values == mean_of(text_ids) && features.pooling == "mean" &&
              features.normalization == "none",
          "pooled features");
    std::vector<std::int32_t> ids{5, 9};
    check(pooled->run({Span<const std::int32_t>(ids.data(), ids.size())}, {}).values ==
              mean_of(ids),
          "pooled token ids");

    const auto rows = token->run({std::string_view("the hello")}, {});
    const auto spans = tok->encode("the hello");
    check(rows.features.rows == spans.size() && rows.features.columns == 2 &&
              rows.tokens.size() == spans.size(),
          "token feature shape");
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const auto& item = rows.tokens[index];
        check(item.token_id == spans[index].id && item.input_index == 0 &&
                  item.token_index == index && item.has_byte_offsets &&
                  item.byte_begin == spans[index].byte_begin &&
                  item.byte_end == spans[index].byte_end &&
                  rows.features.values[2 * index] == static_cast<float>(spans[index].id),
              "token mapping");
    }
    const auto from_ids = token->run({Span<const std::int32_t>(ids.data(), ids.size())}, {});
    check(from_ids.tokens.size() == 2 && !from_ids.tokens[0].has_byte_offsets &&
              from_ids.tokens[1].token_id == 9,
          "token ids without offsets");

    const auto before = observed->calls.size();
    rejects<std::invalid_argument>([&] { pooled->run({std::string_view("")}, {}); }, "empty");
    rejects<std::invalid_argument>([&] { pooled->run({std::string_view("abcdefgh")}, {}); },
                                   "too long");
    rejects<std::invalid_argument>([&] { token->run({std::string_view("a\xC3")}, {}); },
                                   "invalid UTF-8");
    for (const auto id : {std::int32_t{-1}, static_cast<std::int32_t>(tok->vocab_size())}) {
        ids = {id};
        rejects<std::invalid_argument>(
            [&] { pooled->run({Span<const std::int32_t>(ids.data(), ids.size())}, {}); },
            "token id range");
    }
    rejects<std::invalid_argument>([&] { token->run({Span<const std::int32_t>{}}, {}); },
                                   "empty ids");
    rejects<std::invalid_argument>(
        [&] { embedding->run({"hi", static_cast<EmbeddingRole>(9)}, {}); }, "unknown role");
    check(observed->calls.size() == before, "rejected inputs never reach the engine");
    ConfigEntry config{"unknown", std::int64_t(1)};
    rejects<ConfigError>([&] { embedding->run({"hi", EmbeddingRole::Query}, {&config, 1}); },
                         "configuration");
    for (int fault = 1; fault <= 7; ++fault) {
        observed->fault = fault;
        rejects<std::runtime_error>([&] { pooled->run({std::string_view("hi")}, {}); }, "fault");
    }
}

} // namespace

int main() {
    try {
        construction_contract();
        for (const auto* primary :
             {"text_to_embedding", "text_to_pooled_features", "text_to_token_features"})
            task_contract(primary);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ministral3 task contract passed\n";
    return 0;
}
