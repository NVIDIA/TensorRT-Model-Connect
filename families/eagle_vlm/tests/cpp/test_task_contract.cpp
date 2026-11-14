/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/eagle_vlm/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string& text) const override {
        ++calls;
        std::vector<std::int32_t> ids;
        ids.reserve(text.size());
        for (const unsigned char byte : text)
            ids.push_back(static_cast<std::int32_t>(byte));
        return ids;
    }
    std::string decode(const std::vector<std::int32_t>&) const override { return {}; }
    // Never a real separator, so the reranking input is the plain prompt bytes.
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(std::int32_t) const override { return {}; }

    mutable int calls{0};
};

// Records every forward and answers like the two engine kinds this family
// builds. A feature engine takes rank-1 [tokens] inputs and returns [tokens,
// hidden] hidden states whose row i, column h holds 2*i + h + 1, so the first
// row is always {1, 2, ..., hidden}. A reranking engine takes rank-2
// [batch, tokens] inputs and returns one score per position as [batch, tokens,
// 1], where row r, position p holds 100*r + p.
class RecordingModule final : public trtmc::ITrtModule {
  public:
    explicit RecordingModule(bool reranking = false) : reranking_(reranking) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& ids_tensor = inputs.at("input_ids");
        last_shape = ids_tensor.shape;
        const auto* ids = static_cast<const std::int32_t*>(ids_tensor.data);
        last_input_ids.assign(ids, ids + ids_tensor.numel());
        shapes.push_back(ids_tensor.shape);

        if (reranking_) {
            const auto batch = ids_tensor.shape.at(0);
            const auto tokens = ids_tensor.shape.at(1);
            output_buffer.assign(static_cast<std::size_t>(batch * tokens), 0.0F);
            for (std::int64_t row = 0; row < batch; ++row)
                for (std::int64_t position = 0; position < tokens; ++position)
                    output_buffer[static_cast<std::size_t>(row * tokens + position)] =
                        static_cast<float>(100 * row + position);
            return {{"logits", {output_buffer.data(), {batch, tokens, 1}, trtmc::DType::kFloat32}}};
        }
        const auto tokens = ids_tensor.shape.at(0);
        output_buffer.assign(static_cast<std::size_t>(tokens) * hidden, 0.0F);
        for (std::int64_t i = 0; i < tokens; ++i)
            for (std::int32_t h = 0; h < hidden; ++h)
                output_buffer[static_cast<std::size_t>(i) * hidden + h] =
                    static_cast<float>(2 * i + h + 1);
        return {
            {"hidden_states", {output_buffer.data(), {tokens, hidden}, trtmc::DType::kFloat32}}};
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
    std::vector<trtmc::TensorInfo> input_info() const override {
        const std::vector<std::int64_t> shape =
            reranking_ ? std::vector<std::int64_t>{-1, -1} : std::vector<std::int64_t>{-1};
        return {{"input_ids", shape, trtmc::DType::kInt32, true},
                {"attention_mask", shape, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        if (reranking_)
            return {{"logits", {-1, -1, 1}, trtmc::DType::kFloat32, false}};
        return {{"hidden_states",
                 {-1, reported_hidden >= 0 ? reported_hidden : hidden},
                 trtmc::DType::kFloat32,
                 false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "input_ids" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override {
        return name == (reranking_ ? "logits" : "hidden_states");
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {capacity}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return reranking_ ? std::vector<int64_t>{max_batch, capacity}
                          : std::vector<int64_t>{capacity};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return reranking_ ? 2 : 1; }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int calls{0};
    std::int32_t hidden{2};
    // When non-negative, the hidden size the engine metadata claims.
    std::int32_t reported_hidden{-1};
    std::int64_t capacity{64};
    std::int64_t max_batch{2};
    std::vector<std::int64_t> last_shape;
    std::vector<std::vector<std::int64_t>> shapes;
    std::vector<std::int32_t> last_input_ids;
    std::vector<float> output_buffer;

  private:
    bool reranking_;
};

constexpr std::int64_t kVocabSize = 1000;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
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

std::string_view task_of(const trtmc::internal::ITextToPooledFeatures*) {
    return trtmc::internal::ITextToPooledFeatures::kTask;
}

std::unique_ptr<trtmc::EncoderPipeline> make_pipeline(std::string mode,
                                                      RecordingModule** module_out,
                                                      FakeTokenizer** tokenizer_out,
                                                      std::string pooling = "last") {
    const bool reranking = mode == trtmc::internal::ITextQueryDocumentsToRelevance::kTask;
    auto module = std::make_unique<RecordingModule>(reranking);
    auto tokenizer = std::make_shared<FakeTokenizer>();
    *module_out = module.get();
    *tokenizer_out = tokenizer.get();
    return std::make_unique<trtmc::EncoderPipeline>(
        std::move(module), std::move(mode), std::move(tokenizer), kVocabSize, std::move(pooling));
}

std::unique_ptr<trtmc::EncoderPipeline> make_pooled(RecordingModule** module,
                                                    FakeTokenizer** tokenizer) {
    return make_pipeline(std::string(task_of(nullptr)), module, tokenizer);
}

std::unique_ptr<trtmc::EncoderPipeline> make_embedding(RecordingModule** module,
                                                       FakeTokenizer** tokenizer) {
    return make_pipeline(std::string(trtmc::internal::ITextToEmbedding::kTask), module, tokenizer);
}

std::unique_ptr<trtmc::EncoderPipeline>
make_relevance(RecordingModule** module, FakeTokenizer** tokenizer, std::string pooling = "last") {
    return make_pipeline(std::string(trtmc::internal::ITextQueryDocumentsToRelevance::kTask),
                         module, tokenizer, std::move(pooling));
}

// Length of the pinned reranker prompt "question:<q> \n \n passage:<d>".
std::int64_t prompt_length(const std::string& query, const std::string& document) {
    return static_cast<std::int64_t>(std::string("question:").size() +
                                     std::string(" \n \n passage:").size() + query.size() +
                                     document.size());
}

trtmc::internal::TextToPooledFeaturesRequest ids_request(const std::vector<std::int32_t>& ids) {
    return trtmc::internal::TextToPooledFeaturesRequest{
        trtmc::Span<const std::int32_t>{ids.data(), ids.size()}};
}

void test_pooled_features_binding_and_first_token_extraction() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pooled(&module, &tokenizer);

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "text_to_pooled_features" &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0,
            "a pooled-features bundle must advertise exactly that one Task");
    require(std::string(pipeline->task()) == "text_to_pooled_features",
            "the bundle's primary task must match the binding");

    const auto result = pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {});
    require(tokenizer->calls == 1, "plain text must be tokenized exactly once");
    require(module->last_input_ids == std::vector<std::int32_t>{'a', 'b'},
            "the engine must receive the tokenizer's own ids");
    require(result.values.size() == 2 && near(result.values[0], 1.0F) &&
                near(result.values[1], 2.0F),
            "pooled features must be exactly the first-token row, not pooled or normalized");
    require(result.pooling == "first_token" && result.normalization == "none",
            "pooled-features metadata must declare the real first_token/none contract");
}

void test_pooled_features_accepts_token_ids_without_tokenizing() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pooled(&module, &tokenizer);
    const std::vector<std::int32_t> ids{11, 22, 33};
    const auto result = pipeline->run(ids_request(ids), {});
    require(tokenizer->calls == 0, "pre-tokenized ids must bypass the tokenizer entirely");
    require(module->last_input_ids == ids, "the engine must receive the supplied ids");
    require(result.values.size() == 2 && near(result.values[0], 1.0F),
            "token-id input must still extract the first-token row");
}

void test_token_ids_must_index_the_vocabulary() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pooled(&module, &tokenizer);
    for (const std::int32_t bad : {-1, static_cast<std::int32_t>(kVocabSize)}) {
        const std::vector<std::int32_t> ids{3, bad};
        rejects<std::invalid_argument>([&] { pipeline->run(ids_request(ids), {}); },
                                       "a token id outside the vocabulary must be rejected");
    }
    require(module->calls == 0, "an out-of-range id must fail before the engine runs");
    const std::vector<std::int32_t> edge{0, static_cast<std::int32_t>(kVocabSize) - 1};
    pipeline->run(ids_request(edge), {});
    require(module->calls == 1, "the first and last vocabulary ids must be accepted");
}

void test_embedding_mean_pools_and_normalizes() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_embedding(&module, &tokenizer);

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "text_to_embedding",
            "an embedding bundle must advertise exactly that one Task");

    const auto result = pipeline->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {});
    // Rows are {1,2} and {3,4}; mean is {2,3}; L2 norm is sqrt(13).
    const float expected_norm = std::sqrt(13.0F);
    require(result.values.size() == 2 && near(result.values[0], 2.0F / expected_norm) &&
                near(result.values[1], 3.0F / expected_norm),
            "embedding must be the mean-pooled, L2-normalized sentence vector");
    require(result.pooling == "mean" && result.normalization == "l2",
            "embedding metadata must declare the real mean/l2 contract");
}

void test_embedding_rejects_a_non_default_role() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_embedding(&module, &tokenizer);
    for (const auto role :
         {trtmc::internal::EmbeddingRole::Query, trtmc::internal::EmbeddingRole::Document}) {
        rejects<std::invalid_argument>(
            [&] { pipeline->run(trtmc::internal::TextToEmbeddingRequest{"ab", role}, {}); },
            "a role this checkpoint has no prefix for must not be silently ignored");
    }
    require(module->calls == 0, "a rejected role must fail before the engine runs");
}

void test_features_fail_loudly_on_an_unusable_engine_output() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto embedding = make_embedding(&module, &tokenizer);
    module->reported_hidden = 0;
    rejects<std::runtime_error>(
        [&] { embedding->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {}); },
        "an engine output with no hidden dimension must be an error, not an un-pooled result");

    RecordingModule* pooled_module = nullptr;
    FakeTokenizer* pooled_tokenizer = nullptr;
    auto pooled = make_pooled(&pooled_module, &pooled_tokenizer);
    pooled_module->reported_hidden = 0;
    rejects<std::runtime_error>(
        [&] { pooled->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {}); },
        "an engine output with no hidden dimension must be an error, not a mislabeled result");
}

void test_relevance_binds_both_tasks_and_scores_a_pair() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_relevance(&module, &tokenizer);

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 2 && bindings[0].key.id == "text_query_documents_to_relevance" &&
                bindings[1].key.id == "text_pair_to_relevance",
            "a reranking bundle must advertise the list-scoring and the pair Task");
    require(std::string(pipeline->task()) == "text_query_documents_to_relevance",
            "the bundle's primary task must be the list-scoring Task");

    const auto result = pipeline->run(trtmc::internal::TextPairToRelevanceRequest{"q", "doc"}, {});
    // "last" pooling selects the final valid position of the prompt.
    require(near(result.score, static_cast<float>(prompt_length("q", "doc") - 1)),
            "the pair score must be the last-position score of the pinned prompt");
    require(result.kind == trtmc::internal::ScoreKind::Logit, "the head emits raw logits");
    require(module->shapes.size() == 1 && module->shapes[0].size() == 2 &&
                module->shapes[0][0] == 1,
            "a pair must run as a batch of one on a batched reranking engine");
}

void test_relevance_scores_documents_in_order_with_per_row_lengths() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_relevance(&module, &tokenizer);
    const std::vector<std::string_view> documents{"a", "bbbb"};

    const auto result = pipeline->run(
        trtmc::internal::TextQueryDocumentsToRelevanceRequest{
            "q", trtmc::Span<const std::string_view>{documents.data(), documents.size()}},
        {});
    require(result.scores.size() == 2, "one score per document");
    require(near(result.scores[0], static_cast<float>(prompt_length("q", "a") - 1)),
            "row 0 must be scored at its own last valid position");
    require(near(result.scores[1], static_cast<float>(100 + prompt_length("q", "bbbb") - 1)),
            "a longer row must not be scored on another row's padding");
    require(result.kind == trtmc::internal::ScoreKind::Logit, "the head emits raw logits");
    require(module->calls == 1, "documents within the profile batch must run in one forward");
}

void test_relevance_chunks_by_the_profile_batch_and_keeps_order() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_relevance(&module, &tokenizer);
    const std::vector<std::string_view> documents{"a", "bb", "ccc", "dddd", "eeeee"};

    const auto result = pipeline->run(
        trtmc::internal::TextQueryDocumentsToRelevanceRequest{
            "q", trtmc::Span<const std::string_view>{documents.data(), documents.size()}},
        {});
    require(module->calls == 3, "five documents with a profile batch of two need three forwards");
    require(result.scores.size() == 5, "every document must be scored");
    for (std::size_t i = 0; i < documents.size(); ++i) {
        const auto row = static_cast<std::int64_t>(i % 2);
        const auto expected = 100 * row + prompt_length("q", std::string(documents[i])) - 1;
        require(near(result.scores[i], static_cast<float>(expected)),
                "scores must keep the input document order across chunks");
    }
}

void test_relevance_average_pooling_follows_the_checkpoint_contract() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_relevance(&module, &tokenizer, "avg");
    const auto result = pipeline->run(trtmc::internal::TextPairToRelevanceRequest{"q", "d"}, {});
    // Positions 0..n-1 hold 0..n-1, whose mean is (n - 1) / 2.
    require(near(result.score, static_cast<float>(prompt_length("q", "d") - 1) / 2.0F),
            "avg pooling must average the valid positions only");

    RecordingModule* bad_module = nullptr;
    FakeTokenizer* bad_tokenizer = nullptr;
    auto bad = make_relevance(&bad_module, &bad_tokenizer, "median");
    rejects<std::runtime_error>(
        [&] { bad->run(trtmc::internal::TextPairToRelevanceRequest{"q", "d"}, {}); },
        "an unknown pooling contract must be an error, not a guess");
}

void test_relevance_edge_inputs() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_relevance(&module, &tokenizer);
    const auto empty = pipeline->run(
        trtmc::internal::TextQueryDocumentsToRelevanceRequest{
            "q", trtmc::Span<const std::string_view>{}},
        {});
    require(empty.scores.empty() && module->calls == 0,
            "an empty document list must produce an empty result without running the engine");

    const std::string oversized(static_cast<std::size_t>(module->capacity), 'x');
    rejects<std::invalid_argument>(
        [&] { pipeline->run(trtmc::internal::TextPairToRelevanceRequest{"q", oversized}, {}); },
        "an input beyond the engine sequence profile must be rejected");
    require(module->calls == 0, "an oversized input must fail before the engine runs");
}

void test_each_mode_rejects_the_other_tasks() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pooled = make_pooled(&module, &tokenizer);
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { pooled->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {}); },
        "a pooled-features bundle must reject an embedding call, not silently mis-pool");
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { pooled->run(trtmc::internal::TextPairToRelevanceRequest{"q", "d"}, {}); },
        "a pooled-features bundle must reject a relevance call");

    RecordingModule* embedding_module = nullptr;
    FakeTokenizer* embedding_tokenizer = nullptr;
    auto embedding = make_embedding(&embedding_module, &embedding_tokenizer);
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { embedding->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {}); },
        "an embedding bundle must reject a pooled-features call");
    rejects<trtmc::internal::UnsupportedTask>(
        [&] {
            embedding->run(
                trtmc::internal::TextQueryDocumentsToRelevanceRequest{
                    "q", trtmc::Span<const std::string_view>{}},
                {});
        },
        "an embedding bundle must reject a list-scoring call");
    require(embedding_module->calls == 0, "a rejected Task must fail before the engine runs");

    RecordingModule* relevance_module = nullptr;
    FakeTokenizer* relevance_tokenizer = nullptr;
    auto relevance = make_relevance(&relevance_module, &relevance_tokenizer);
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { relevance->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {}); },
        "a reranking bundle must reject an embedding call");
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { relevance->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {}); },
        "a reranking bundle must reject a pooled-features call");
    require(relevance_module->calls == 0, "a rejected Task must fail before the engine runs");
}

void test_unsupported_config_and_input_limits() {
    const trtmc::internal::ConfigEntry unsupported{"max_length", std::int64_t{16}};
    const trtmc::internal::ConfigView config{&unsupported, 1};

    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pooled = make_pooled(&module, &tokenizer);
    rejects<trtmc::internal::ConfigError>(
        [&] { pooled->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, config); },
        "this family has no runtime configuration; an unknown option must fail");
    require(module->calls == 0, "a rejected config must fail before the engine ever runs");

    RecordingModule* embedding_module = nullptr;
    FakeTokenizer* embedding_tokenizer = nullptr;
    auto embedding = make_embedding(&embedding_module, &embedding_tokenizer);
    rejects<trtmc::internal::ConfigError>(
        [&] { embedding->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, config); },
        "an embedding call must reject unknown configuration");

    RecordingModule* relevance_module = nullptr;
    FakeTokenizer* relevance_tokenizer = nullptr;
    auto relevance = make_relevance(&relevance_module, &relevance_tokenizer);
    rejects<trtmc::internal::ConfigError>(
        [&] { relevance->run(trtmc::internal::TextPairToRelevanceRequest{"q", "d"}, config); },
        "a pair call must reject unknown configuration");
    rejects<trtmc::internal::ConfigError>(
        [&] {
            relevance->run(
                trtmc::internal::TextQueryDocumentsToRelevanceRequest{
                    "q", trtmc::Span<const std::string_view>{}},
                config);
        },
        "a list-scoring call must reject unknown configuration");
    require(relevance_module->calls == 0, "a rejected config must fail before the engine runs");

    const std::vector<std::int32_t> oversized(static_cast<std::size_t>(module->capacity) + 1, 7);
    rejects<std::invalid_argument>([&] { pooled->run(ids_request(oversized), {}); },
                                   "a sequence beyond the engine capacity must be rejected");
    require(module->calls == 0, "an oversized input must fail before the engine runs");

    const std::vector<std::int32_t> empty_ids;
    rejects<std::invalid_argument>([&] { pooled->run(ids_request(empty_ids), {}); },
                                   "an empty token sequence must be rejected, not silently run");
}

void test_constructor_validates_its_dependencies() {
    const std::string mode(trtmc::internal::ITextToPooledFeatures::kTask);
    rejects<std::runtime_error>(
        [&] {
            trtmc::EncoderPipeline(nullptr, mode, std::make_shared<FakeTokenizer>(), kVocabSize);
        },
        "a null engine module must be rejected");
    rejects<std::runtime_error>(
        [&] {
            trtmc::EncoderPipeline(std::make_unique<RecordingModule>(), mode, nullptr, kVocabSize);
        },
        "a missing tokenizer must be rejected");
    rejects<std::runtime_error>(
        [&] {
            trtmc::EncoderPipeline(std::make_unique<RecordingModule>(), mode,
                                   std::make_shared<FakeTokenizer>(), 0);
        },
        "a non-positive vocabulary size must be rejected");
}

} // namespace

int main() {
    try {
        test_pooled_features_binding_and_first_token_extraction();
        test_pooled_features_accepts_token_ids_without_tokenizing();
        test_token_ids_must_index_the_vocabulary();
        test_embedding_mean_pools_and_normalizes();
        test_embedding_rejects_a_non_default_role();
        test_features_fail_loudly_on_an_unusable_engine_output();
        test_relevance_binds_both_tasks_and_scores_a_pair();
        test_relevance_scores_documents_in_order_with_per_row_lengths();
        test_relevance_chunks_by_the_profile_batch_and_keeps_order();
        test_relevance_average_pooling_follows_the_checkpoint_contract();
        test_relevance_edge_inputs();
        test_each_mode_rejects_the_other_tasks();
        test_unsupported_config_and_input_limits();
        test_constructor_validates_its_dependencies();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
