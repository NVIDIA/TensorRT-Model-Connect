/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side text runtime of the openai_gpt family: the tokenizer.json contract, encoding,
// decoding and token sampling. The expected ids of the synthetic tokenizer below were read
// from the tokenizers library (0.22.2) on the same tokenizer.json.
//
// With `--encode TOKENIZER_JSON REQUESTS_JSONL` it instead answers one request per line,
// {"encode": text} with {"ids": [...]} and {"decode": [ids]} with {"text": ...}, so that
// test_tokenizer_parity.py can compare the native tokenizer with the reference library.

#include "families/openai_gpt/runtime/engine_contract.h"
#include "families/openai_gpt/runtime/sampler.h"
#include "families/openai_gpt/runtime/text_config.h"
#include "families/openai_gpt/runtime/tokenizer.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using trtmc::openai_gpt::parse_text_config;
using trtmc::openai_gpt::Sampler;
using trtmc::openai_gpt::SamplingParams;
using trtmc::openai_gpt::Tokenizer;
using Ids = std::vector<std::int32_t>;

int failures = 0;

void check(bool passed, const std::string& what) {
    if (!passed) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

template <typename Error>
void expect_throw(const std::function<void()>& action, const std::string& what) {
    try {
        action();
    } catch (const Error&) {
        return;
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: " << what << ": unexpected exception: " << error.what() << '\n';
        return;
    }
    ++failures;
    std::cerr << "FAIL: " << what << ": no exception\n";
}

// A small tokenizer.json of the checkpoint's shape: uncased BertNormalizer,
// BertPreTokenizer, BPE with the end-of-word suffix, BPEDecoder.
const char* const kTokenizer = R"json({"version":"1.0","truncation":null,"padding":null,
"added_tokens":[{"id":0,"special":true,"content":"<unk>","single_word":false,"lstrip":false,
"rstrip":false,"normalized":false}],"normalizer":{"type":"BertNormalizer","clean_text":true,
"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":null,
"decoder":{"type":"BPEDecoder","suffix":"</w>"},"model":{"dropout":null,"unk_token":"<unk>",
"continuing_subword_prefix":null,"end_of_word_suffix":"</w>","fuse_unk":false,
"vocab":{"<unk>":0,"h":1,"e":2,"l":3,"o":4,"w":5,"r":6,"d":7,"i":8,"n":9,"g":10,"a":11,"b":12,
"c":13,"t":14,"s":15,"h</w>":16,"e</w>":17,"l</w>":18,"o</w>":19,"w</w>":20,"r</w>":21,
"d</w>":22,"i</w>":23,"n</w>":24,"g</w>":25,"a</w>":26,"b</w>":27,"c</w>":28,"t</w>":29,
"s</w>":30,"!</w>":31,",</w>":32,".</w>":33,"aa":34,"aa</w>":35,"aaa</w>":36,"he":37,"ll":38,
"hell":39,"hello</w>":40,"wo":41,"rl":42,"worl":43,"world</w>":44,"in":45,"ing</w>":46,
"th":47,"the</w>":48},
"merges":["a a","a a</w>","aa a</w>","h e","l l","he ll","hell o</w>","w o","r l","wo rl",
"worl d</w>","i n","in g</w>","t h","th e</w>"]}})json";

void test_encode() {
    const Tokenizer tokenizer(kTokenizer);
    check(tokenizer.vocab_size() == 49, "vocabulary size");
    const std::vector<std::pair<std::string, Ids>> cases = {
        {"Hello, World!", {40, 32, 44, 31}},
        // Accents are stripped and text is lowercased.
        {"H\xC3\x89LLO world", {40, 44}},
        // "t h" ranks after "h e", so "theend" keeps "he" and "th" never forms.
        {"tab\tthe\x07"
         "end.",
         {14, 11, 27, 14, 37, 2, 9, 22, 33}},
        {"the thing", {48, 47, 46}},
        // Lowest rank first, leftmost first among equal ranks; stale entries are skipped.
        {"aaa aaaa a", {36, 34, 35, 26}},
        // A combining acute accent is removed.
        {"a\xCC\x81"
         "b",
         {11, 27}},
        // A CJK ideograph is a word of its own; outside the vocabulary it is the unknown token.
        {"\xE4\xB8\xAD hello", {0, 40}},
        // The added token matches in the raw text and splits the words around it.
        {"hello<unk>world", {40, 0, 44}},
        {"", {}},
        {"  \t\n ", {}},
        {"\xC3\xA9\xC3\xA8", {2, 17}},
        {"sing,sing", {15, 46, 32, 15, 46}},
    };
    for (const auto& [text, expected] : cases)
        check(tokenizer.encode(text, 64) == expected, "encode '" + text + "'");
}

void test_token_budget() {
    const Tokenizer tokenizer(kTokenizer);
    check(tokenizer.encode("Hello, World!", 4).size() == 4, "exactly fitting text");
    expect_throw<std::invalid_argument>([&] { tokenizer.encode("Hello, World!", 3); },
                                        "text over the token budget");
    expect_throw<std::invalid_argument>([&] { tokenizer.encode("hello<unk>", 1); },
                                        "added token over the token budget");
    // A 1000-character word needs more than 50 tokens, whatever it merges to.
    expect_throw<std::invalid_argument>([&] { tokenizer.encode(std::string(1000, 'x'), 50); },
                                        "word over the token budget");
    check(tokenizer.encode(std::string(30, 'x'), 30) == Ids(30, 0), "unknown characters");
}

void test_invalid_text() {
    const Tokenizer tokenizer(kTokenizer);
    for (const std::string& text : {std::string("\xC3"), std::string("a\xFF"),
                                    std::string("\xED\xA0\x80"), std::string("\xC0\xAF")})
        expect_throw<std::invalid_argument>([&] { tokenizer.encode(text, 64); },
                                            "invalid UTF-8 is refused");
}

void test_decode() {
    const Tokenizer tokenizer(kTokenizer);
    const std::vector<std::pair<Ids, std::string>> cases = {
        {{40, 44}, "hello world"},
        {{37, 38, 19}, "hello"},
        // Special tokens are skipped before the decoder, so the last kept token ends the text.
        {{40, 0, 31}, "hello !"},
        {{40, 0}, "hello"},
        {{0}, ""},
        {{}, ""},
        // Ids without a token are skipped, as the reference does.
        {{40, 999}, "hello"},
        {{40, -1}, "hello"},
    };
    for (const auto& [ids, expected] : cases)
        check(tokenizer.decode(ids) == expected, "decode to '" + expected + "'");
}

std::string mutated(const std::function<void(nlohmann::json&)>& change) {
    auto json = nlohmann::json::parse(kTokenizer);
    change(json);
    return json.dump();
}

void test_contract() {
    const std::vector<std::pair<std::string, std::string>> refused = {
        {"not JSON", "{"},
        {"repeated key", std::string(kTokenizer).replace(1, 0, "\"padding\":null,")},
        {"truncation", mutated([](auto& j) { j["truncation"] = {{"max_length", 8}}; })},
        {"cased normalizer", mutated([](auto& j) { j["normalizer"]["lowercase"] = false; })},
        {"accents kept", mutated([](auto& j) { j["normalizer"]["strip_accents"] = false; })},
        {"other normalizer", mutated([](auto& j) { j["normalizer"] = {{"type", "NFC"}}; })},
        {"other pre-tokenizer",
         mutated([](auto& j) { j["pre_tokenizer"] = {{"type", "Whitespace"}}; })},
        {"post-processor",
         mutated([](auto& j) { j["post_processor"] = {{"type", "TemplateProcessing"}}; })},
        {"decoder suffix", mutated([](auto& j) { j["decoder"]["suffix"] = "</x>"; })},
        {"WordPiece model", mutated([](auto& j) { j["model"]["type"] = "WordPiece"; })},
        {"fused unknown", mutated([](auto& j) { j["model"]["fuse_unk"] = true; })},
        {"byte fallback", mutated([](auto& j) { j["model"]["byte_fallback"] = true; })},
        {"dropout", mutated([](auto& j) { j["model"]["dropout"] = 0.1; })},
        {"subword prefix",
         mutated([](auto& j) { j["model"]["continuing_subword_prefix"] = "##"; })},
        {"empty suffix", mutated([](auto& j) {
             j["model"]["end_of_word_suffix"] = "";
             j["decoder"]["suffix"] = "";
         })},
        {"unknown token", mutated([](auto& j) { j["model"]["unk_token"] = "<missing>"; })},
        {"id gap", mutated([](auto& j) { j["model"]["vocab"]["the</w>"] = 49; })},
        {"repeated id", mutated([](auto& j) { j["model"]["vocab"]["the</w>"] = 47; })},
        {"fractional id", mutated([](auto& j) { j["model"]["vocab"]["the</w>"] = 48.5; })},
        {"empty token", mutated([](auto& j) {
             j["model"]["vocab"].erase("the</w>");
             j["model"]["vocab"][""] = 48;
             j["model"]["merges"].erase(14);
         })},
        // U+1D165 is a combining mark that survives normalization and can be reordered.
        {"reorderable mark", mutated([](auto& j) {
             j["model"]["vocab"].erase("the</w>");
             j["model"]["vocab"]["x\xF0\x9D\x85\xA5"] = 48;
             j["model"]["merges"].erase(14);
         })},
        {"merge outside the vocabulary",
         mutated([](auto& j) { j["model"]["merges"].push_back("x y"); })},
        {"merge result outside the vocabulary",
         mutated([](auto& j) { j["model"]["merges"].push_back("b c"); })},
        {"repeated merge", mutated([](auto& j) { j["model"]["merges"].push_back("a a"); })},
        {"merge without a pair", mutated([](auto& j) { j["model"]["merges"].push_back("aa"); })},
        {"merge with two spaces", mutated([](auto& j) { j["model"]["merges"].push_back("a  a"); })},
        {"added token outside the vocabulary",
         mutated([](auto& j) { j["added_tokens"][0]["content"] = "<pad>"; })},
        {"added token id", mutated([](auto& j) { j["added_tokens"][0]["id"] = 1; })},
        {"normalized added token",
         mutated([](auto& j) { j["added_tokens"][0]["normalized"] = true; })},
        {"stripped added token", mutated([](auto& j) { j["added_tokens"][0]["lstrip"] = true; })},
        {"non-boolean option", mutated([](auto& j) { j["model"]["fuse_unk"] = 0; })},
        {"missing vocabulary", mutated([](auto& j) { j["model"].erase("vocab"); })},
        {"normalizer without lowercase",
         mutated([](auto& j) { j["normalizer"].erase("lowercase"); })},
        {"added token without special",
         mutated([](auto& j) { j["added_tokens"][0].erase("special"); })},
    };
    for (const auto& [what, json] : refused)
        expect_throw<std::runtime_error>([&] { Tokenizer tokenizer(json); }, "refuses " + what);
    // The merge list may also hold [left, right] pairs.
    const Tokenizer pairs(
        mutated([](auto& j) { j["model"]["merges"][0] = nlohmann::json::array({"a", "a"}); }));
    check(pairs.encode("aaa", 8) == Ids{36}, "merge pairs as lists");
}

void test_sampler() {
    const std::vector<float> ties{1.0F, 3.0F, 3.0F, 2.0F};
    check(Sampler(SamplingParams{}).sample(ties) == 1, "greedy takes the first maximum");
    check(Sampler(SamplingParams{0.0, 50, 1.0, 0.0, 3}).sample(ties) == 1,
          "temperature 0 is greedy");
    for (const float bad :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        Sampler sampler(SamplingParams{});
        expect_throw<std::runtime_error>([&] { sampler.sample({0.0F, bad}); },
                                         "a non-finite logit is refused");
    }
    expect_throw<std::invalid_argument>([] { Sampler(SamplingParams{}).sample({}); }, "no logits");
    for (const auto& params :
         {SamplingParams{-1.0, 1, 1.0, 0.0, -1}, SamplingParams{1.0, -1, 1.0, 0.0, -1},
          SamplingParams{1.0, 1, 1.5, 0.0, -1}, SamplingParams{1.0, 1, 1.0, 2.0, -1},
          SamplingParams{std::numeric_limits<double>::quiet_NaN(), 0, 1.0, 0.0, -1}})
        expect_throw<std::invalid_argument>([&] { Sampler sampler(params); },
                                            "out-of-range sampling parameters");

    const auto draws = [](const SamplingParams& params, const std::vector<float>& logits,
                          int count) {
        Sampler sampler(params);
        std::vector<int> histogram(logits.size(), 0);
        for (int i = 0; i < count; ++i)
            ++histogram[static_cast<std::size_t>(sampler.sample(logits))];
        return histogram;
    };
    // Probabilities 0.5, 0.3, 0.2 and a negligible fourth token.
    const std::vector<float> logits{std::log(0.5F), std::log(0.3F), std::log(0.2F), -40.0F};
    const auto top_k = draws(SamplingParams{1.0, 2, 1.0, 0.0, 7}, logits, 2000);
    check(top_k[2] == 0 && top_k[3] == 0 && top_k[0] > 0 && top_k[1] > 0, "top_k keeps two");
    const auto top_p = draws(SamplingParams{1.0, 0, 0.7, 0.0, 7}, logits, 2000);
    check(top_p[2] == 0 && top_p[3] == 0 && top_p[1] > 0, "top_p 0.7 keeps 0.5 and 0.3");
    const auto narrow = draws(SamplingParams{1.0, 0, 0.45, 0.0, 7}, logits, 200);
    check(narrow[0] == 200, "top_p 0.45 keeps the most likely token only");
    const auto min_p = draws(SamplingParams{1.0, 0, 1.0, 0.5, 7}, logits, 2000);
    check(min_p[2] == 0 && min_p[3] == 0 && min_p[1] > 0, "min_p 0.5 keeps ratios >= 0.5");
    const auto full = draws(SamplingParams{1.0, 0, 1.0, 0.0, 7}, logits, 20000);
    check(std::abs(full[0] / 20000.0 - 0.5) < 0.02 && std::abs(full[1] / 20000.0 - 0.3) < 0.02 &&
              std::abs(full[2] / 20000.0 - 0.2) < 0.02,
          "sampling follows the probabilities");

    const std::vector<float> flat(64, 0.0F);
    const auto sequence = [&](std::int64_t seed) {
        Sampler sampler(SamplingParams{1.0, 0, 1.0, 0.0, seed});
        Ids ids;
        for (int i = 0; i < 32; ++i)
            ids.push_back(sampler.sample(flat));
        return ids;
    };
    // A tiny positive temperature stays a sampling temperature: only the maximum keeps weight.
    const auto tiny = draws(SamplingParams{1e-100, 0, 1.0, 0.0, 7}, {0.0F, 1.0F, 0.5F}, 50);
    check(tiny[1] == 50, "a tiny temperature samples the most likely token");
    check(sequence(1) == sequence(1), "a seed repeats its draws");
    check(sequence(1) != sequence(2), "different seeds draw differently");
    check(sequence(-1) == sequence(-5), "negative seeds select one fixed default");
}

void test_text_config() {
    using trtmc::internal::ConfigEntry;
    using trtmc::internal::ConfigError;
    using trtmc::internal::ConfigValue;
    const auto parse = [](std::vector<ConfigEntry> entries, std::int32_t capacity = 512) {
        return parse_text_config(trtmc::Span<const ConfigEntry>(entries.data(), entries.size()),
                                 capacity);
    };
    const auto defaults = parse({});
    check(defaults.max_new_tokens == 128 && defaults.sampling.top_k == 1 &&
              defaults.sampling.temperature == 1.0 && defaults.sampling.seed == -1,
          "default text config");
    check(parse({{"max_new_tokens", ConfigValue{std::int64_t{512}}}}).max_new_tokens == 512,
          "max_new_tokens may fill the cache");
    // A positive temperature too small for float32 is kept, not rounded to greedy zero.
    check(parse({{"temperature", ConfigValue{1e-100}}}).sampling.temperature == 1e-100,
          "tiny temperature is kept");
    const std::vector<std::pair<std::string, ConfigEntry>> refused = {
        {"no new tokens", {"max_new_tokens", ConfigValue{std::int64_t{0}}}},
        {"more new tokens than cache rows", {"max_new_tokens", ConfigValue{std::int64_t{513}}}},
        {"negative temperature", {"temperature", ConfigValue{-0.5}}},
        {"infinite temperature",
         {"temperature", ConfigValue{std::numeric_limits<double>::infinity()}}},
        {"NaN top_p", {"top_p", ConfigValue{std::numeric_limits<double>::quiet_NaN()}}},
        {"top_p above one", {"top_p", ConfigValue{1.5}}},
        {"negative min_p", {"min_p", ConfigValue{-0.1}}},
        {"negative top_k", {"top_k", ConfigValue{std::int64_t{-1}}}},
        {"top_k past int32", {"top_k", ConfigValue{std::int64_t{1} << 40}}},
        {"unknown field", {"repetition_penalty", ConfigValue{1.0}}},
        {"wrong type", {"max_new_tokens", ConfigValue{2.0}}},
    };
    for (const auto& [what, entry] : refused)
        expect_throw<ConfigError>([&] { parse({entry}); }, "text config refuses " + what);
    expect_throw<ConfigError>(
        [&] {
            parse({{"seed", ConfigValue{std::int64_t{1}}}, {"seed", ConfigValue{std::int64_t{2}}}});
        },
        "text config refuses a repeated field");
}

// A module that answers only the introspection the engine contract reads: tensor names,
// dtypes and shapes, and one profile's minimum and maximum shapes.
class FakeModule final : public trtmc::ITrtModule {
  public:
    std::map<std::string, trtmc::TensorInfo> tensors;
    std::map<std::string, std::pair<std::vector<int64_t>, std::vector<int64_t>>> profile;
    std::int32_t profiles{2};

    trtmc::TensorMap forward(const trtmc::TensorMap&) override { throw std::logic_error("unused"); }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_async(const trtmc::TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(false); }
    bool has_input(const std::string& name) const override {
        return tensors.count(name) != 0 && tensors.at(name).is_input;
    }
    bool has_output(const std::string& name) const override {
        return tensors.count(name) != 0 && !tensors.at(name).is_input;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return tensors.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return tensors.at(name).shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, std::int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        const auto& range = profile.at(name);
        return selector == trtmc::ProfileShapeSelector::kMin ? range.first : range.second;
    }
    std::int32_t optimization_profile_count() const override { return profiles; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    std::int32_t input_rank(const std::string& name) const override {
        return static_cast<std::int32_t>(tensors.at(name).shape.size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        return profile.count(name) != 0;
    }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<trtmc::TensorInfo> info(bool inputs) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, tensor] : tensors) {
            if (tensor.is_input == inputs)
                result.push_back(tensor);
        }
        return result;
    }
};

// An engine as the builder makes it: 2 layers, 2 heads of 4, 16 cache rows, 10 logits.
FakeModule fake_engine(std::int64_t max_tokens) {
    using trtmc::DType;
    FakeModule module;
    const auto add = [&](const std::string& name, std::vector<int64_t> shape, DType dtype,
                         bool input) {
        module.tensors[name] = {name, std::move(shape), dtype, input};
    };
    add("token_id", {-1}, DType::kInt32, true);
    add("position_id", {-1}, DType::kInt32, true);
    add("cache_write_indices", {1}, DType::kInt32, true);
    add("key_value_lengths", {1}, DType::kInt32, true);
    add("key_positions", {-1}, DType::kInt32, true);
    add("logits", {1, 10}, DType::kFloat32, false);
    for (int layer = 0; layer < 2; ++layer) {
        const auto suffix = "_" + std::to_string(layer);
        add("cache_k" + suffix, {1, 2, 16, 4}, DType::kFloat32, true);
        add("cache_v" + suffix, {1, 2, 16, 4}, DType::kFloat32, true);
        add("present_k" + suffix, {1, 2, 16, 4}, DType::kFloat32, false);
        add("present_v" + suffix, {1, 2, 16, 4}, DType::kFloat32, false);
    }
    module.profile["token_id"] = {{1}, {max_tokens}};
    module.profile["position_id"] = {{1}, {max_tokens}};
    module.profile["key_positions"] = {{1}, {16}};
    return module;
}

void test_engine_contract() {
    using trtmc::openai_gpt::EngineShape;
    using trtmc::openai_gpt::validate_engine;
    const EngineShape shape{10, 2, 2, 4, 16};
    for (const std::int64_t max_tokens : {std::int64_t{16}, std::int64_t{1}}) {
        auto module = fake_engine(max_tokens);
        try {
            validate_engine(shape, module, max_tokens);
        } catch (const std::exception& error) {
            check(false, std::string("the built engine is accepted: ") + error.what());
        }
    }
    const std::vector<std::pair<std::string, std::function<void(FakeModule&, EngineShape&)>>>
        refused = {
            // Refused from the tensor counts, before a name is built for each declared layer.
            {"two billion declared layers",
             [](FakeModule&, EngineShape& s) { s.num_layers = std::numeric_limits<int>::max(); }},
            {"another layer count", [](FakeModule&, EngineShape& s) { s.num_layers = 3; }},
            {"a short-request profile",
             [](FakeModule& m, EngineShape&) { m.profile["key_positions"].first = {8}; }},
            {"fewer cache rows read",
             [](FakeModule& m, EngineShape&) { m.profile["key_positions"].second = {8}; }},
            {"a smaller prefill",
             [](FakeModule& m, EngineShape&) { m.profile["token_id"].second = {8}; }},
            {"one profile", [](FakeModule& m, EngineShape&) { m.profiles = 1; }},
            {"another vocabulary", [](FakeModule&, EngineShape& s) { s.vocab_size = 11; }},
            {"another head width", [](FakeModule&, EngineShape& s) { s.head_dim = 8; }},
            {"a half-precision present cache",
             [](FakeModule& m, EngineShape&) {
                 m.tensors["present_k_0"].dtype = trtmc::DType::kFloat16;
             }},
            {"an int8 write index",
             [](FakeModule& m, EngineShape&) {
                 m.tensors["cache_write_indices"].dtype = trtmc::DType::kInt8;
             }},
            {"a renamed tensor",
             [](FakeModule& m, EngineShape&) {
                 auto node = m.tensors.extract("cache_v_1");
                 node.key() = node.mapped().name = "cache_x_1";
                 m.tensors.insert(std::move(node));
             }},
        };
    for (const auto& [what, change] : refused) {
        auto module = fake_engine(16);
        EngineShape changed = shape;
        change(module, changed);
        expect_throw<std::runtime_error>([&] { validate_engine(changed, module, 16); },
                                         "engine contract refuses " + what);
    }
}

int encode_requests(const char* tokenizer_path, const char* requests_path) {
    std::ifstream tokenizer_file(tokenizer_path, std::ios::binary);
    std::stringstream buffer;
    buffer << tokenizer_file.rdbuf();
    if (!tokenizer_file) {
        std::cerr << "cannot read " << tokenizer_path << '\n';
        return 2;
    }
    std::unique_ptr<Tokenizer> loaded;
    try {
        loaded = std::make_unique<Tokenizer>(buffer.str());
    } catch (const std::runtime_error& error) {
        // A tokenizer.json outside the contract: the caller checks this exit code.
        std::cerr << error.what() << '\n';
        return 3;
    }
    const Tokenizer& tokenizer = *loaded;
    std::ifstream requests(requests_path, std::ios::binary);
    if (!requests) {
        std::cerr << "cannot read " << requests_path << '\n';
        return 2;
    }
    for (std::string line; std::getline(requests, line);) {
        const auto request = nlohmann::json::parse(line);
        nlohmann::json answer;
        if (request.contains("encode")) {
            try {
                answer["ids"] = tokenizer.encode(request.at("encode").get<std::string>(),
                                                 std::numeric_limits<std::size_t>::max());
            } catch (const std::invalid_argument& error) {
                answer["error"] = error.what();
            }
        } else {
            answer["text"] = tokenizer.decode(request.at("decode").get<Ids>());
        }
        std::cout << answer.dump() << '\n';
    }
    return std::cout ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--encode")
        return encode_requests(argv[2], argv[3]);
    if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [--encode TOKENIZER_JSON REQUESTS_JSONL]\n";
        return 2;
    }
    test_encode();
    test_token_budget();
    test_invalid_text();
    test_decode();
    test_contract();
    test_sampler();
    test_text_config();
    test_engine_contract();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "openai_gpt text runtime: all checks passed\n";
    return 0;
}
