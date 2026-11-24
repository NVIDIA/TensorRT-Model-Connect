/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU contract of the MobileBERT runtime: the WordPiece tokenizer on a small
// tokenizer.json, and both Tasks of the pipeline against a recording engine.

#include "families/mobilebert/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// ids: 0 [PAD] 1 [UNK] 2 [CLS] 3 [SEP] 4 [MASK] 5 the 6 cafe 7 un 8 ##aff 9 ##able
// 10 , 11 ! 12 hello 13 world 14 中 15 文 16 naive 17 a 18 ##b 19 [ 20 ] 21 mask 22 x
const char* kTokenizerJson = R"json({
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true},
    {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true},
    {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true},
    {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true},
    {"id": 4, "content": "[MASK]", "single_word": false, "lstrip": false, "rstrip": false,
     "normalized": false, "special": true}
  ],
  "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                 "strip_accents": null, "lowercase": true},
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 10,
            "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "the": 5,
                      "cafe": 6, "un": 7, "##aff": 8, "##able": 9, ",": 10, "!": 11,
                      "hello": 12, "world": 13, "中": 14, "文": 15, "naive": 16,
                      "a": 17, "##b": 18, "[": 19, "]": 20, "mask": 21, "x": 22}}
})json";

constexpr std::int32_t kVocab = 23;
constexpr std::int32_t kWindow = 8;
constexpr std::size_t kNoLimit = std::numeric_limits<std::size_t>::max();

void require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}

template <class Error, class Function>
void rejects(Function function, const std::string& message) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

// The message of the invalid_argument the function throws.
template <class Function>
std::string refusal(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    throw std::runtime_error(message);
}

std::shared_ptr<trtmc::IMobileBertTokenizer> tokenizer() {
    return trtmc::CreateMobileBertTokenizer(kTokenizerJson, std::strlen(kTokenizerJson));
}

// Keeps fixed-length input buffers across calls the way the runtime does, and
// answers with logits[row][id] = 1000 * row + id and next-sentence logits
// {7.5, -2.5}.
class RecordingEngine final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        for (auto* buffer : {&ids, &types, &mask}) {
            const char* name = buffer == &ids     ? "input_ids"
                               : buffer == &types ? "token_type_ids"
                                                  : "attention_mask";
            const auto& tensor = inputs.at(name);
            if (tensor.dtype != trtmc::DType::kInt32)
                throw std::runtime_error("engine inputs must be int32");
            std::memcpy(buffer->data(), tensor.data,
                        std::min(tensor.nbytes(), buffer->size() * sizeof(std::int32_t)));
        }
        logits.assign(static_cast<std::size_t>(kWindow) * kVocab, 0.0F);
        for (std::int32_t row = 0; row < kWindow; ++row)
            for (std::int32_t id = 0; id < kVocab; ++id)
                logits[static_cast<std::size_t>(row * kVocab + id)] =
                    static_cast<float>(1000 * row + id);
        relation = {7.5F, -2.5F};
        trtmc::TensorMap outputs;
        if (!drop_relation)
            outputs["seq_relationship_logits"] = {relation.data(), {2}, trtmc::DType::kFloat32};
        outputs["prediction_logits"] = {logits.data(), {kWindow, kVocab}, trtmc::DType::kFloat32};
        return outputs;
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
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kInt32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {kWindow}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {kWindow};
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
    bool drop_relation{false};
    std::vector<std::int32_t> ids = std::vector<std::int32_t>(kWindow, -7);
    std::vector<std::int32_t> types = std::vector<std::int32_t>(kWindow, -7);
    std::vector<std::int32_t> mask = std::vector<std::int32_t>(kWindow, -7);
    std::vector<float> logits;
    std::vector<float> relation;
};

trtmc::MobileBertRuntimeConfig runtime_config() {
    trtmc::MobileBertRuntimeConfig config;
    config.max_sequence_length = kWindow;
    config.vocab_size = kVocab;
    config.type_vocab_size = 2;
    config.pad_token_id = 0;
    config.unk_token_id = 1;
    config.cls_token_id = 2;
    config.sep_token_id = 3;
    config.mask_token_id = 4;
    config.vocabulary_id = "mobilebert";
    config.relation_labels = {"next", "not_next"};
    config.relation_vocabulary_id = "mobilebert.next_sentence";
    return config;
}

std::unique_ptr<trtmc::MobileBertPipeline>
pipeline(RecordingEngine** engine_out, std::string task = "masked_text_to_token_scores") {
    auto engine = std::make_unique<RecordingEngine>();
    *engine_out = engine.get();
    return std::make_unique<trtmc::MobileBertPipeline>(std::move(engine), tokenizer(),
                                                       runtime_config(), std::move(task));
}

std::vector<std::int32_t> padded(std::vector<std::int32_t> values, std::int32_t fill) {
    values.resize(kWindow, fill);
    return values;
}

trtmc::internal::VocabularyScoresResult masked(trtmc::MobileBertPipeline& model,
                                               std::string_view text) {
    return model.run(trtmc::internal::MaskedTextToTokenScoresRequest{text}, {});
}

trtmc::internal::VocabularyScoresResult masked_ids(trtmc::MobileBertPipeline& model,
                                                   const std::vector<std::int32_t>& ids) {
    return model.run(
        trtmc::internal::MaskedTextToTokenScoresRequest{
            trtmc::Span<const std::int32_t>{ids.data(), ids.size()}},
        {});
}

void test_tokenizer() {
    const auto words = tokenizer();
    const std::vector<std::pair<std::string, std::vector<std::int32_t>>> cases = {
        {"Hello, World!", {12, 10, 13, 11}},
        {"Caf\xc3\xa9", {6}},     // accent stripped, lowercased
        {"na\xc3\xafve", {16}},   // diaeresis stripped
        {"unaffable", {7, 8, 9}}, // greedy longest-match pieces
        {"ab", {17, 18}},
        {"xyz", {1}},                              // one missing piece: whole word unknown
        {"the [MASK] world", {5, 4, 13}},          // raw special token, before lowercasing
        {"the [mask] world", {5, 19, 21, 20, 13}}, // not the special token
        {"[MASK][MASK]!", {4, 4, 11}},
        {"\xe4\xb8\xad\xe6\x96\x87", {14, 15}},     // CJK characters are words of their own
        {"\tthe\x07\nworld\xc2\xa0!", {5, 13, 11}}, // control dropped, odd whitespace splits
        {"abbbbbbbbbbb", {1}},                      // longer than max_input_chars_per_word
        {"", {}},
    };
    for (const auto& [text, expected] : cases)
        require(words->encode(text, kNoLimit) == expected, "tokenizer mismatch on '" + text + "'");
    require(words->size() == static_cast<std::size_t>(kVocab), "tokenizer size");
    require(words->id_for_token("[MASK]") == 4 && words->id_for_token("missing") == -1,
            "tokenizer token lookup");

    // The published checkpoint's tokenizer.json predates the model "type" field;
    // WordPiece is then recognised by its fields, and any other type is refused.
    const std::string typed = "\"type\": \"WordPiece\", ";
    std::string untyped = kTokenizerJson;
    untyped.erase(untyped.find(typed), typed.size());
    require(trtmc::CreateMobileBertTokenizer(untyped.data(), untyped.size())
                    ->encode("Café", kNoLimit) == std::vector<std::int32_t>{6},
            "an untyped WordPiece model is read like a typed one");
    std::string other = kTokenizerJson;
    other.replace(other.find(typed), typed.size(), "\"type\": \"BPE\", ");
    rejects<std::runtime_error>(
        [&] { trtmc::CreateMobileBertTokenizer(other.data(), other.size()); },
        "a model of another type is refused");

    // Malformed UTF-8 has no reference behaviour and is refused, not dropped.
    for (const auto& bad :
         {std::string("a\xff"), std::string("\xc0\xaf"), std::string("\xed\xa0\x80"),
          std::string("\xe4\xb8"), std::string("\xf4\x90\x80\x80")})
        rejects<std::invalid_argument>([&] { words->encode(bad, kNoLimit); },
                                       "malformed UTF-8 must be refused");

    std::string normalized_special = kTokenizerJson;
    const auto at = normalized_special.find("\"normalized\": false");
    normalized_special.replace(at, std::strlen("\"normalized\": false"), "\"normalized\": true");
    rejects<std::runtime_error>(
        [&] {
            trtmc::CreateMobileBertTokenizer(normalized_special.data(), normalized_special.size());
        },
        "an added token matched after normalisation is not implemented and must be refused");

    // An added token keeps its vocabulary id, inside the vocabulary; any other id
    // is refused before it can size a table (2^32 + 4 must not wrap to 4).
    const std::string mask_entry = "\"id\": 4, \"content\": \"[MASK]\"";
    for (const char* id : {"2147483647", "4294967300", "23", "5", "-1", "4.5"}) {
        std::string json = kTokenizerJson;
        json.replace(json.find(mask_entry), mask_entry.size(),
                     std::string("\"id\": ") + id + ", \"content\": \"[MASK]\"");
        rejects<std::runtime_error>(
            [&] { trtmc::CreateMobileBertTokenizer(json.data(), json.size()); },
            std::string("an added token with id ") + id + " must be refused");
    }
}

void test_tokenizer_reads_only_what_fits() {
    const auto words = tokenizer();
    require(words->encode("the [MASK] world", 3) == std::vector<std::int32_t>({5, 4, 13}),
            "a text that fits the limit is encoded whole");
    // Past the limit the text is read only to the first id beyond it, so the rest,
    // here malformed, is never read.
    const std::string tail = "the [MASK] world hello \xff";
    rejects<std::invalid_argument>([&] { words->encode(tail, kNoLimit); }, "the tail is malformed");
    require(words->encode(tail, 3) == std::vector<std::int32_t>({5, 4, 13, 12}),
            "encoding stops at the first id past the limit");
    // A long run of added tokens costs one pass, and stops at the limit.
    std::string masks;
    for (int i = 0; i < 100000; ++i)
        masks += "[MASK]";
    require(words->encode(masks, kNoLimit) == std::vector<std::int32_t>(100000, 4),
            "every added token of a long run");
    require(words->encode(masks + "\xff", 6).size() == 7, "a long run stops at the limit");
}

void test_bindings() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine, "text_pair_to_pretraining_relation_scores");
    const auto bindings = model->task_bindings();
    require(bindings.size() == 2 && bindings[0].key.id == "masked_text_to_token_scores" &&
                bindings[1].key.id == "text_pair_to_pretraining_relation_scores",
            "a bundle serves both pretraining heads");
    require(std::string(model->task()) == "text_pair_to_pretraining_relation_scores",
            "the primary task is the bundle's");
}

void test_masked_positions() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    const auto result = masked(*model, "the [MASK] world");
    require(engine->ids == padded({2, 5, 4, 13, 3}, 0), "framed ids padded with [PAD]");
    require(engine->types == padded({}, 0), "single text is segment 0");
    require(engine->mask == padded({1, 1, 1, 1, 1}, 0), "attention mask covers the real tokens");
    require(result.logits.rows == 1 && result.logits.columns == kVocab &&
                result.logits.values.size() == static_cast<std::size_t>(kVocab),
            "one [MASK], one row");
    require(result.logits.values[0] == 2000.0F && result.logits.values[kVocab - 1] == 2022.0F,
            "the row is the [MASK] position's");
    require(result.positions.size() == 1 && result.positions[0].token_id == 4 &&
                result.positions[0].token_index == 2 && result.positions[0].input_index == 0 &&
                !result.positions[0].has_byte_offsets,
            "position metadata of the [MASK]");
    require(result.vocabulary_id == "mobilebert", "vocabulary identity");
}

void test_unmasked_text_returns_every_position() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    const auto result = masked(*model, "hello world");
    require(result.logits.rows == 4 && result.positions.size() == 4, "every real position");
    for (std::size_t row = 0; row < 4; ++row)
        require(result.logits.values[row * kVocab] == static_cast<float>(1000 * row),
                "rows in sequence order");
    require(result.positions[0].input_index == -1 && result.positions[1].input_index == 0 &&
                result.positions[2].input_index == 0 && result.positions[3].input_index == -1,
            "[CLS] and [SEP] added by the family are inserted tokens");
}

void test_caller_ids() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    const auto result = masked_ids(*model, {2, 4, 22, 4, 3});
    require(engine->ids == padded({2, 4, 22, 4, 3}, 0), "caller ids are sent as given");
    require(result.logits.rows == 2 && result.positions[0].token_index == 1 &&
                result.positions[1].token_index == 3,
            "both masks are selected");
    require(result.positions[0].input_index == 0 && result.positions[1].input_index == 0,
            "caller ids are all input tokens");
    const int before = engine->calls;
    rejects<std::invalid_argument>([&] { masked_ids(*model, {2, kVocab, 3}); },
                                   "an id outside the vocabulary is refused");
    rejects<std::invalid_argument>([&] { masked_ids(*model, {2, -1, 3}); },
                                   "a negative id is refused");
    rejects<std::invalid_argument>([&] { masked_ids(*model, {}); }, "no ids is refused");
    require(engine->calls == before, "refused ids never reach the engine");
}

void test_window_is_not_truncated() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    masked(*model, "the the the the the [MASK]"); // 8 tokens framed: fits exactly
    rejects<std::invalid_argument>([&] { masked(*model, "the the the the the the [MASK]"); },
                                   "an input longer than the window is refused");
    rejects<std::invalid_argument>(
        [&] {
            model->run(trtmc::internal::TextPairToPretrainingRelationScoresRequest{"the the the",
                                                                                   "the the the"},
                       {});
        },
        "a pair longer than the window is refused");
    // Text is read only until it overflows the window: these are refused for their
    // length, and their malformed tails are never read.
    const auto too_long = [](const std::string& message) {
        return message.find("longer than the engine window") != std::string::npos;
    };
    const auto pair = [&](std::string_view first, std::string_view second) {
        model->run(trtmc::internal::TextPairToPretrainingRelationScoresRequest{first, second}, {});
    };
    require(too_long(refusal([&] { masked(*model, "the the the the the the the [MASK] \xff"); },
                             "an over-long text is refused")),
            "an over-long text is refused before its tail is read");
    require(too_long(refusal([&] { pair("the the the the the the", "\xff"); },
                             "an over-long first text is refused")),
            "the second text is not read once the first overflows");
    require(too_long(refusal([&] { pair("the", "the the the the the \xff"); },
                             "an over-long second text is refused")),
            "an over-long second text is refused before its tail is read");
    require(engine->calls == 1, "refused inputs never reach the engine");
}

void test_shorter_after_longer_has_no_stale_tokens() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    masked(*model, "hello hello hello [MASK] world");
    masked(*model, "[MASK]");
    require(engine->ids == padded({2, 4, 3}, 0) && engine->mask == padded({1, 1, 1}, 0),
            "every call overwrites the whole window");
}

void test_next_sentence() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    const auto result = model->run(
        trtmc::internal::TextPairToPretrainingRelationScoresRequest{"Hello", "world!"}, {});
    require(engine->ids == padded({2, 12, 3, 13, 11, 3}, 0), "[CLS] A [SEP] B [SEP]");
    require(engine->types == padded({0, 0, 0, 1, 1, 1}, 0), "segment 1 starts after first [SEP]");
    require(engine->mask == padded({1, 1, 1, 1, 1, 1}, 0), "pair attention mask");
    require(result.scores == std::vector<float>({7.5F, -2.5F}), "the two head logits in order");
    require(result.labels == std::vector<std::string>({"next", "not_next"}),
            "index 0 means the second text continues the first");
    require(result.kind == trtmc::internal::ScoreKind::Logit, "scores are logits");
    require(result.vocabulary_id == "mobilebert.next_sentence", "relation label space");
}

void test_runtime_config_and_outputs_are_checked() {
    RecordingEngine* engine = nullptr;
    auto model = pipeline(&engine);
    const trtmc::internal::ConfigEntry entry{"top_k", std::int64_t{5}};
    rejects<trtmc::internal::ConfigError>(
        [&] {
            model->run(trtmc::internal::MaskedTextToTokenScoresRequest{std::string_view("x")},
                       trtmc::internal::ConfigView{&entry, 1});
        },
        "there is no runtime configuration");
    engine->drop_relation = true;
    rejects<std::runtime_error>([&] { masked(*model, "x"); },
                                "a missing engine output is an error");

    auto broken = runtime_config();
    broken.relation_labels = {"next"};
    rejects<std::runtime_error>(
        [&] {
            trtmc::MobileBertPipeline(std::make_unique<RecordingEngine>(), tokenizer(), broken,
                                      "masked_text_to_token_scores");
        },
        "the relation head needs one label per score");
}

} // namespace

int main() {
    try {
        test_tokenizer();
        test_tokenizer_reads_only_what_fits();
        test_bindings();
        test_masked_positions();
        test_unmasked_text_returns_every_position();
        test_caller_ids();
        test_window_is_not_truncated();
        test_shorter_after_longer_has_no_stale_tokens();
        test_next_sentence();
        test_runtime_config_and_outputs_are_checked();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "mobilebert task contract: all checks passed\n";
    return 0;
}
