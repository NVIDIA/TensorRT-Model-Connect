/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/modernbert/runtime/pipeline.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace trtmc;
using namespace trtmc::internal;

void check(bool condition) {
    if (!condition)
        throw std::runtime_error("ModernBERT task contract failed");
}

template <class Error, class Function>
void rejects(Function&& function) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("expected failure was not propagated");
}

class Tokenizer final : public ITokenizer {
  public:
    mutable std::vector<std::string> inputs;
    std::vector<std::int32_t> encode(const std::string& text) const override {
        inputs.push_back(text);
        if (text == "empty")
            return {};
        if (text == "oversize")
            return {1, 2, 3, 4, 5};
        if (text == "invalid")
            return {100};
        return text == "short" ? std::vector<std::int32_t>{3} : std::vector<std::int32_t>{1, 2};
    }
    std::string decode(const std::vector<std::int32_t>&) const override { return {}; }
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(std::int32_t) const override { return {}; }
};

class Engine final : public ITrtModule {
  public:
    DType mask_dtype{DType::kInt32};
    int calls{0};
    int fault{0};
    bool dynamic{true};
    std::vector<float> states;
    std::vector<std::int64_t> lengths;
    std::vector<std::vector<std::int32_t>> input_ids;
    std::vector<std::vector<float>> masks;
    TensorMap forward(const TensorMap& input) override {
        ++calls;
        if (fault == 1)
            throw std::runtime_error("engine failure");
        const auto& ids = input.at("input_ids");
        const auto& mask = input.at("attention_mask");
        check(ids.dtype == DType::kInt32 && ids.shape == mask.shape && mask.dtype == mask_dtype);
        lengths.push_back(ids.shape.at(0));
        input_ids.emplace_back(static_cast<std::int32_t*>(ids.data),
                               static_cast<std::int32_t*>(ids.data) + ids.numel());
        masks.emplace_back();
        states.clear();
        for (std::size_t i = 0; i < ids.numel(); ++i) {
            const auto valid = mask_dtype == DType::kInt32
                                   ? static_cast<float>(static_cast<std::int32_t*>(mask.data)[i])
                                   : static_cast<float*>(mask.data)[i];
            check(valid == 0.0f || valid == 1.0f);
            masks.back().push_back(valid);
            const auto value = static_cast<float>(static_cast<std::int32_t*>(ids.data)[i]);
            check(valid != 0.0f || value == 0.0f);
            states.push_back(valid == 0.0f ? 1000.0f : value);
            states.push_back(valid == 0.0f ? 500.0f : value * 2.0f);
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
        return name == "input_ids"        ? DType::kInt32
               : name == "attention_mask" ? mask_dtype
                                          : DType::kFloat32;
    }
    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        const std::int64_t length = dynamic ? -1 : 4;
        return name == "hidden_states" ? std::vector<std::int64_t>{length, 2}
                                       : std::vector<std::int64_t>{length};
    }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  ProfileShapeSelector) const override {
        return {4};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return dynamic; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

void pooled_contract(DType mask_dtype, bool dynamic) {
    auto engine = std::make_unique<Engine>();
    auto* observed = engine.get();
    observed->mask_dtype = mask_dtype;
    observed->dynamic = dynamic;
    auto tokenizer = std::make_shared<Tokenizer>();
    modernbert::EncoderPipeline model(std::move(engine), "text_to_pooled_features", tokenizer, 100,
                                      4);
    auto bindings = model.task_bindings();
    check(bindings.size() == 1 && bindings[0].key.id == ITextToPooledFeatures::kTask);
    auto* task = static_cast<ITextToPooledFeatures*>(bindings[0].implementation);
    const auto first = task->run({std::string_view("long")}, {});
    const auto second = task->run({std::string_view("short")}, {});
    check(first.values == std::vector<float>({1, 2}) && first.pooling == "cls" &&
          first.normalization == "none");
    check(second.values == std::vector<float>({3, 6}));
    check(observed->lengths ==
          (dynamic ? std::vector<std::int64_t>{2, 1} : std::vector<std::int64_t>{4, 4}));
    if (!dynamic) {
        check(observed->input_ids[0] == std::vector<std::int32_t>({1, 2, 0, 0}));
        check(observed->input_ids[1] == std::vector<std::int32_t>({3, 0, 0, 0}));
        check(observed->masks[0] == std::vector<float>({1, 1, 0, 0}));
        check(observed->masks[1] == std::vector<float>({1, 0, 0, 0}));
    }
    std::vector<std::int32_t> ids{7, 8};
    check(task->run({Span<const std::int32_t>(ids.data(), ids.size())}, {}).values ==
          std::vector<float>({7, 14}));
    for (const auto text : {"empty", "oversize", "invalid"}) {
        const auto before = observed->calls;
        rejects<std::invalid_argument>([&] { task->run({std::string_view(text)}, {}); });
        check(before == observed->calls);
    }
    for (const auto id : {-1, 100}) {
        ids = {id};
        rejects<std::invalid_argument>(
            [&] { task->run({Span<const std::int32_t>(ids.data(), ids.size())}, {}); });
    }
    rejects<std::invalid_argument>([&] { task->run({Span<const std::int32_t>{}}, {}); });
    ConfigEntry config{"unknown", std::int64_t(1)};
    rejects<ConfigError>([&] { task->run({std::string_view("long")}, {&config, 1}); });
    rejects<UnsupportedTask>([&] { model.run(TextToEmbeddingRequest{"long"}, {}); });
    for (int fault = 1; fault <= 6; ++fault) {
        observed->fault = fault;
        rejects<std::runtime_error>([&] { task->run({std::string_view("long")}, {}); });
    }
}

void embedding_contract(bool dynamic) {
    auto engine = std::make_unique<Engine>();
    engine->dynamic = dynamic;
    modernbert::EncoderPipeline model(std::move(engine), "text_to_embedding",
                                      std::make_shared<Tokenizer>(), 100, 4);
    const auto bindings = model.task_bindings();
    check(bindings.size() == 1 && bindings[0].key.id == ITextToEmbedding::kTask);
    for (const auto role :
         {EmbeddingRole::Default, EmbeddingRole::Query, EmbeddingRole::Document}) {
        const auto result = model.run(TextToEmbeddingRequest{"long", role}, {});
        check(result.values.size() == 2 && result.pooling == "mean" &&
              result.normalization == "l2");
        check(std::abs(result.values[0] - 1.0f / std::sqrt(5.0f)) < 1e-6f);
        check(std::abs(result.values[1] - 2.0f / std::sqrt(5.0f)) < 1e-6f);
    }
    rejects<std::invalid_argument>(
        [&] { model.run(TextToEmbeddingRequest{"long", static_cast<EmbeddingRole>(99)}, {}); });
}

void relevance_contract() {
    auto engine = std::make_unique<Engine>();
    auto* observed = engine.get();
    auto tokenizer = std::make_shared<Tokenizer>();
    modernbert::EncoderPipeline model(std::move(engine), "text_pair_to_relevance", tokenizer, 100,
                                      4);
    const auto bindings = model.task_bindings();
    check(bindings.size() == 2 && bindings[0].key.id == ITextPairToRelevance::kTask &&
          bindings[1].key.id == ITextQueryDocumentsToRelevance::kTask);
    const auto single = model.run(TextPairToRelevanceRequest{"query", "document"}, {});
    check(single.score == 1 && single.kind == ScoreKind::Unbounded);
    check(tokenizer->inputs.back() == "question:query   passage:document");
    std::vector<std::string_view> documents{"first", "second"};
    const auto list = model.run(
        TextQueryDocumentsToRelevanceRequest{"query", {documents.data(), documents.size()}}, {});
    check(list.scores == std::vector<float>({1, 1}) && list.kind == ScoreKind::Unbounded);
    check(tokenizer->inputs[tokenizer->inputs.size() - 2] == "question:query   passage:first");
    check(tokenizer->inputs.back() == "question:query   passage:second");
    const auto calls = observed->calls;
    check(model.run(TextQueryDocumentsToRelevanceRequest{"query", {}}, {}).scores.empty());
    check(observed->calls == calls);
    ConfigEntry config{"unknown", std::int64_t(1)};
    rejects<ConfigError>(
        [&] { model.run(TextQueryDocumentsToRelevanceRequest{"query", {}}, {&config, 1}); });
}
} // namespace

int main() {
    try {
        for (const bool dynamic : {true, false}) {
            pooled_contract(trtmc::DType::kInt32, dynamic);
            pooled_contract(trtmc::DType::kFloat32, dynamic);
            embedding_contract(dynamic);
        }
        relevance_contract();
        rejects<trtmc::internal::UnsupportedTask>([] {
            trtmc::modernbert::EncoderPipeline model(std::make_unique<Engine>(), "encoding",
                                                     std::make_shared<Tokenizer>(), 100, 4);
        });
        std::cout << "ModernBERT native task contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
