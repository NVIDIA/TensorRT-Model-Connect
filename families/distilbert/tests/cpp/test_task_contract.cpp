/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/distilbert/runtime/pipeline.h"

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
    std::int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(std::int32_t) const override { return {}; }

    mutable int calls{0};
};

// Returns n rows of `hidden` values; row i, column h holds 2*i + h + 1, so
// the first row (the CLS token) is always {1, 2, ..., hidden}.
class RecordingModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& ids_tensor = inputs.at("input_ids");
        const auto& mask_tensor = inputs.at("attention_mask");
        last_input_length = static_cast<std::int64_t>(ids_tensor.numel());
        const auto* ids = static_cast<const std::int32_t*>(ids_tensor.data);
        last_input_ids.assign(ids, ids + last_input_length);

        // Like the real runtime: copy only the bytes supplied into persistent
        // fixed-length device buffers, leaving whatever an earlier call wrote
        // beyond them.
        std::memcpy(buffer_ids.data(), ids_tensor.data,
                    std::min(ids_tensor.nbytes(), buffer_ids.size() * sizeof(std::int32_t)));
        std::memcpy(buffer_mask.data(), mask_tensor.data,
                    std::min(mask_tensor.nbytes(), buffer_mask.size() * sizeof(float)));
        valid_seen = 0;
        for (const float value : buffer_mask)
            valid_seen += value > 0.0F ? 1 : 0;

        hidden_buffer.assign(static_cast<std::size_t>(capacity) * hidden, 0.0F);
        for (std::int64_t i = 0; i < capacity; ++i)
            for (std::int32_t h = 0; h < hidden; ++h)
                hidden_buffer[static_cast<std::size_t>(i) * hidden + h] =
                    static_cast<float>(2 * i + h + 1);
        return {
            {"hidden_states", {hidden_buffer.data(), {capacity, hidden}, trtmc::DType::kFloat32}}};
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
        return {{"input_ids", {-1}, trtmc::DType::kInt32, true},
                {"attention_mask", {-1}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"hidden_states", {-1, hidden}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "input_ids" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override { return name == "hidden_states"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {capacity}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {capacity};
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

    int calls{0};
    std::int32_t hidden{2};
    std::int64_t capacity{512};
    std::int64_t last_input_length{0};
    std::int64_t valid_seen{0};
    std::vector<std::int32_t> buffer_ids = std::vector<std::int32_t>(512, 0);
    std::vector<float> buffer_mask = std::vector<float>(512, 0.0F);
    std::vector<std::int32_t> last_input_ids;
    std::vector<float> hidden_buffer;
};

// True when the engine received exactly `expected` followed by zero padding to
// the engine's fixed input length.
bool sent_ids(const RecordingModule& module, const std::vector<std::int32_t>& expected) {
    if (module.last_input_length != module.capacity ||
        module.last_input_ids.size() != static_cast<std::size_t>(module.capacity))
        return false;
    for (std::size_t i = 0; i < module.last_input_ids.size(); ++i) {
        const std::int32_t want = i < expected.size() ? expected[i] : 0;
        if (module.last_input_ids[i] != want)
            return false;
    }
    return true;
}

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

std::unique_ptr<trtmc::EncoderPipeline>
make_pipeline(std::string mode, RecordingModule** module_out, FakeTokenizer** tokenizer_out) {
    auto module = std::make_unique<RecordingModule>();
    auto tokenizer = std::make_shared<FakeTokenizer>();
    *module_out = module.get();
    *tokenizer_out = tokenizer.get();
    return std::make_unique<trtmc::EncoderPipeline>(std::move(module), std::move(mode),
                                                    std::move(tokenizer));
}

void test_pooled_features_binding_and_cls_extraction() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == "text_to_pooled_features" &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0,
            "a pooled-features bundle must advertise exactly that one Task");
    require(std::string(pipeline->task()) == "text_to_pooled_features",
            "the bundle's primary task must match the binding");

    const auto result = pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {});
    require(tokenizer->calls == 1, "plain text must be tokenized exactly once");
    require(sent_ids(*module, {'a', 'b'}),
            "the engine must receive the tokenizer's own ids, zero padded to its length");
    require(result.values.size() == 2 && near(result.values[0], 1.0F) &&
                near(result.values[1], 2.0F),
            "pooled features must be exactly the CLS row, not pooled or normalized");
    require(result.pooling == "cls" && result.normalization == "none",
            "pooled-features metadata must declare the real cls/none contract");
}

void test_pooled_features_accepts_token_ids_without_tokenizing() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);
    const std::vector<std::int32_t> ids{11, 22, 33};
    const trtmc::internal::TextToPooledFeaturesRequest request{
        trtmc::Span<const std::int32_t>{ids.data(), ids.size()}};
    const auto result = pipeline->run(request, {});
    require(tokenizer->calls == 0, "pre-tokenized ids must bypass the tokenizer entirely");
    require(sent_ids(*module, ids), "the engine must receive the supplied ids, zero padded");
    require(result.values.size() == 2 && near(result.values[0], 1.0F) &&
                near(result.values[1], 2.0F),
            "token-id input must still extract the CLS row");
}

void test_embedding_mean_pools_and_normalizes() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline =
        make_pipeline(std::string(trtmc::internal::ITextToEmbedding::kTask), &module, &tokenizer);

    const auto result = pipeline->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {});
    // Rows are {1,2} and {3,4}; mean is {2,3}; L2 norm is sqrt(13).
    const float expected_norm = std::sqrt(13.0F);
    require(result.values.size() == 2 && near(result.values[0], 2.0F / expected_norm) &&
                near(result.values[1], 3.0F / expected_norm),
            "embedding must be the mean-pooled, L2-normalized sentence vector");
    require(result.pooling == "mean" && result.normalization == "l2",
            "embedding metadata must declare the real mean/l2 contract");
}

void test_relevance_single_pair_and_batch() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextPairToRelevance::kTask),
                                  &module, &tokenizer);

    const auto bindings = pipeline->task_bindings();
    require(bindings.size() == 2, "a reranking bundle must also expose batched document scoring");
    bool has_pair = false, has_batch = false;
    for (const auto& binding : bindings) {
        has_pair = has_pair || binding.key.id == "text_pair_to_relevance";
        has_batch = has_batch || binding.key.id == "text_query_documents_to_relevance";
    }
    require(has_pair && has_batch, "both the pair and the batch relevance Tasks must be bound");

    const auto single = pipeline->run(
        trtmc::internal::TextPairToRelevanceRequest{"what is the capital of France?", "Paris."},
        {});
    require(near(single.score, 1.0F), "the relevance score must be the engine's own first output");
    require(single.kind == trtmc::internal::ScoreKind::Unbounded,
            "an unnormalized cross-encoder score must be reported as Unbounded");

    const std::vector<std::string_view> documents{"Paris.", "a river.", "a mountain."};
    const auto batch = pipeline->run(
        trtmc::internal::TextQueryDocumentsToRelevanceRequest{
            "what is the capital of France?",
            trtmc::Span<const std::string_view>{documents.data(), documents.size()}},
        {});
    require(batch.scores.size() == 3, "one score must be returned per input document, in order");
    for (const auto score : batch.scores)
        require(near(score, 1.0F), "batch scoring must reuse the exact same pair-scoring logic");
}

void test_shorter_input_after_longer_sees_no_stale_tokens() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);
    const std::vector<std::int32_t> longer(30, 5);
    const std::vector<std::int32_t> shorter(5, 6);
    pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{trtmc::Span<const std::int32_t>{
                      longer.data(), longer.size()}},
                  {});
    require(module->valid_seen == 30, "the first call must expose exactly its own tokens");
    pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{trtmc::Span<const std::int32_t>{
                      shorter.data(), shorter.size()}},
                  {});
    require(module->valid_seen == 5,
            "a shorter input after a longer one must not leave stale tokens attended to");
    require(sent_ids(*module, shorter), "the shorter input must overwrite the whole padded buffer");
}

void test_each_mode_rejects_the_other_tasks() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { pipeline->run(trtmc::internal::TextToEmbeddingRequest{"ab"}, {}); },
        "a pooled-features bundle must reject an embedding call, not silently mis-pool");
    rejects<trtmc::internal::UnsupportedTask>(
        [&] { pipeline->run(trtmc::internal::TextPairToRelevanceRequest{"q", "d"}, {}); },
        "a pooled-features bundle must reject a relevance call");
}

void test_unsupported_config_and_empty_input() {
    RecordingModule* module = nullptr;
    FakeTokenizer* tokenizer = nullptr;
    auto pipeline = make_pipeline(std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                  &module, &tokenizer);
    const trtmc::internal::ConfigEntry unsupported{"max_length", std::int64_t{16}};
    rejects<trtmc::internal::ConfigError>(
        [&] {
            pipeline->run(trtmc::internal::TextToPooledFeaturesRequest{"ab"}, {&unsupported, 1});
        },
        "this family has no runtime configuration; an unknown option must fail");
    require(module->calls == 0, "a rejected config must fail before the engine ever runs");

    const std::vector<std::int32_t> oversized(static_cast<std::size_t>(module->capacity) + 1, 7);
    const trtmc::internal::TextToPooledFeaturesRequest oversized_request{
        trtmc::Span<const std::int32_t>{oversized.data(), oversized.size()}};
    rejects<std::invalid_argument>([&] { pipeline->run(oversized_request, {}); },
                                   "a sequence beyond the engine capacity must be rejected");
    require(module->calls == 0, "an oversized input must fail before the engine runs");

    const std::vector<std::int32_t> empty_ids;
    const trtmc::internal::TextToPooledFeaturesRequest empty_request{
        trtmc::Span<const std::int32_t>{empty_ids.data(), empty_ids.size()}};
    rejects<std::invalid_argument>([&] { pipeline->run(empty_request, {}); },
                                   "an empty token sequence must be rejected, not silently run");
}

void test_constructor_validates_its_dependencies() {
    rejects<std::runtime_error>(
        [&] {
            trtmc::EncoderPipeline(nullptr,
                                   std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                   std::make_shared<FakeTokenizer>());
        },
        "a null engine module must be rejected");
    rejects<std::runtime_error>(
        [&] {
            trtmc::EncoderPipeline(std::make_unique<RecordingModule>(),
                                   std::string(trtmc::internal::ITextToPooledFeatures::kTask),
                                   nullptr);
        },
        "a missing tokenizer must be rejected");
}

} // namespace

int main() {
    try {
        test_pooled_features_binding_and_cls_extraction();
        test_pooled_features_accepts_token_ids_without_tokenizing();
        test_embedding_mean_pools_and_normalizes();
        test_relevance_single_pair_and_batch();
        test_shorter_input_after_longer_sees_no_stale_tokens();
        test_each_mode_rejects_the_other_tasks();
        test_unsupported_config_and_empty_input();
        test_constructor_validates_its_dependencies();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
