/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU tests of the Laguna text runtime: pre-tokenizer, tokenizer contract,
// chat template and sampler. Expected pre-tokenizer pieces and chat renders
// were produced by the Hugging Face tokenizer and chat template of
// poolside/Laguna-XS.2 at revision a397bde04501591d0ac3d74a5d9d0a4ec298ffb8.

#include "families/laguna/runtime/chat_template.h"
#include "families/laguna/runtime/sampler.h"
#include "families/laguna/runtime/tokenizer.h"

#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

void check_pieces(const std::string& text, const std::vector<std::string>& expected,
                  const char* name) {
    check(trtmc::laguna::pre_tokenize(text) == expected, name);
}

void test_pre_tokenizer() {
    check_pieces("def f(x):\n    return x * 2\n\n\nprint(f(21))\n",
                 {"def", " f", "(x", "):", "\n", "   ", " return", " x", " *", " ", "2", "\n\n\n",
                  "print", "(f", "(", "2", "1", "))", "\n"},
                 "code with indentation and blank lines");
    check_pieces("a  \n  b", {"a", "  ", "\n", " ", " b"}, "spaces around a newline");
    check_pieces(" \n \n ", {" ", "\n", " ", "\n", " "}, "newline runs start pieces");
    check_pieces("it's IT'S we'RE they've I'M you'll he'd 'hello ''' ''s",
                 {"it", "'s", " IT", "'S", " we", "'RE", " they", "'ve", " I", "'M", " you", "'ll",
                  " he", "'d", " '", "hello", " '''", " ''", "s"},
                 "case-insensitive contractions");
    check_pieces("x=12345, y=3.14e-10; z=-0x1F;",
                 {"x", "=", "1", "2", "3", "4", "5",  ",",  " y", "=", "3", ".", "1",
                  "4", "e", "-", "1", "0", ";", " z", "=-", "0",  "x", "1", "F", ";"},
                 "single-digit numbers");
    check_pieces("line1\r\nline2\r\n\r\nline3\rline4",
                 {"line", "1", "\r\n", "line", "2", "\r\n\r\n", "line", "3", "\r", "line", "4"},
                 "CRLF runs and a lone CR");
}

// A minimal tokenizer.json with Laguna's exact pre-tokenizer contract.
nlohmann::json tiny_tokenizer() {
    nlohmann::json vocab = nlohmann::json::object();
    int next = 0;
    vocab["\xE3\x80\x88|EOS|\xE3\x80\x89"] = next++; // 〈|EOS|〉
    vocab["<think>"] = next++;
    for (const char* symbol : {"h", "e", "l", "o", "\xC4\xA0" /* Ġ */, "w", "r", "d", "he", "hel",
                               "hell", "hello", "\xC4\xA0w", "\xC4\xA0wo", "o\xC4\xA0"})
        vocab[symbol] = next++;
    nlohmann::json merges = nlohmann::json::array();
    for (const auto& [a, b] : std::vector<std::pair<std::string, std::string>>{{"h", "e"},
                                                                               {"he", "l"},
                                                                               {"hel", "l"},
                                                                               {"hell", "o"},
                                                                               {"\xC4\xA0", "w"},
                                                                               {"\xC4\xA0w", "o"}})
        merges.push_back({a, b});
    const auto split = [](const char* regex, const char* behavior) {
        return nlohmann::json{{"type", "Split"},
                              {"pattern", {{"Regex", regex}}},
                              {"behavior", behavior},
                              {"invert", false}};
    };
    const auto added = [](const char* content, int id, bool special) {
        return nlohmann::json{{"id", id},          {"content", content}, {"single_word", false},
                              {"lstrip", false},   {"rstrip", false},    {"normalized", false},
                              {"special", special}};
    };
    return {
        {"added_tokens",
         {added("\xE3\x80\x88|EOS|\xE3\x80\x89", 0, true), added("<think>", 1, false)}},
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {split("(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext"),
            split("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
                  "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                  "Isolated"),
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", true},
             {"use_regex", false}}}}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single",
           {{{"SpecialToken", {{"id", "\xE3\x80\x88|EOS|\xE3\x80\x89"}, {"type_id", 0}}}},
            {{"Sequence", {{"id", "A"}, {"type_id", 0}}}}}},
          {"special_tokens",
           {{"\xE3\x80\x88|EOS|\xE3\x80\x89",
             {{"id", "\xE3\x80\x88|EOS|\xE3\x80\x89"},
              {"ids", {0}},
              {"tokens", {"\xE3\x80\x88|EOS|\xE3\x80\x89"}}}}}}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"model",
         {{"type", "BPE"},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"continuing_subword_prefix", ""},
          {"end_of_word_suffix", ""},
          {"vocab", vocab},
          {"merges", merges}}},
    };
}

void test_tokenizer() {
    const auto json = tiny_tokenizer();
    const auto tokenizer = trtmc::laguna::Tokenizer::create(json.dump());
    const auto id = [&](const char* token) { return tokenizer->id_for_token(token); };
    check(tokenizer->encode("hello world", false) ==
              std::vector<std::int32_t>{id("hello"), id("\xC4\xA0wo"), id("r"), id("l"), id("d")},
          "BPE merges by rank");
    check(tokenizer->encode("hello", true) == std::vector<std::int32_t>{0, id("hello")},
          "post-processor prefixes the special token");
    check(tokenizer->encode("he<think>he", false) ==
              std::vector<std::int32_t>{id("he"), 1, id("he")},
          "added tokens split the text before pre-tokenization");
    check(tokenizer->decode({0, id("hello"), 1, id("\xC4\xA0wo")}) == "hello<think> wo",
          "decode skips special tokens only");

    auto broken = json;
    broken["normalizer"] = {{"type", "NFC"}};
    bool rejected = false;
    try {
        trtmc::laguna::Tokenizer::create(broken.dump());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "a different tokenizer contract is rejected");

    const auto rejects = [](const nlohmann::json& candidate) {
        try {
            trtmc::laguna::Tokenizer::create(candidate.dump());
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    // An empty added token would match at every position without advancing encode().
    auto empty_added = json;
    empty_added["added_tokens"].push_back({{"id", 17},
                                           {"content", ""},
                                           {"single_word", false},
                                           {"lstrip", false},
                                           {"rstrip", false},
                                           {"normalized", false},
                                           {"special", false}});
    check(rejects(empty_added), "an added token with empty content is rejected");
    // Token ids are integers in [0, INT32_MAX]: 1.5 would truncate to 1 and 2^32 would wrap to 0.
    auto fractional = json;
    fractional["added_tokens"][1]["id"] = 1.5;
    check(rejects(fractional), "a fractional added-token id is rejected");
    auto wide_vocab = json;
    wide_vocab["model"]["vocab"]["hello"] = 4294967296ULL;
    check(rejects(wide_vocab), "a vocab id beyond INT32_MAX is rejected");
    auto wide_special = json;
    wide_special["post_processor"]["special_tokens"]["\xE3\x80\x88|EOS|\xE3\x80\x89"]["ids"] = {
        4294967296ULL};
    check(rejects(wide_special), "a post-processor token id beyond INT32_MAX is rejected");

    // tokenizers' BPE merges one occurrence at a time: (a, b) creates (ab, a), whose lower rank
    // applies next, so "abab" is "aba b"; merging every occurrence of (a, b) first would give "ab
    // ab".
    auto ranked = json;
    ranked["model"]["vocab"] = {{"\xE3\x80\x88|EOS|\xE3\x80\x89", 0},
                                {"<think>", 1},
                                {"a", 2},
                                {"b", 3},
                                {"ab", 4},
                                {"aba", 5}};
    ranked["model"]["merges"] = nlohmann::json::array(
        {nlohmann::json::array({"ab", "a"}), nlohmann::json::array({"a", "b"})});
    check(trtmc::laguna::Tokenizer::create(ranked.dump())->encode("abab", false) ==
              std::vector<std::int32_t>{5, 3},
          "BPE reconsiders merge ranks after each merge");

    auto dropout = json;
    dropout["model"]["dropout"] = 1.0;
    check(rejects(dropout), "BPE dropout is rejected");
    dropout["model"]["dropout"] = 0.0;
    check(!rejects(dropout), "zero BPE dropout merges deterministically");
    check(tokenizer->max_id() == 16, "max_id is the largest token id");

    auto repeated_id = json;
    repeated_id["model"]["vocab"]["hello"] = 2; // the id of "h"
    check(rejects(repeated_id), "a repeated vocab id is rejected");
    auto outside_result = json;
    outside_result["model"]["merges"].push_back({"o", "w"}); // "ow" is not in the vocabulary
    check(rejects(outside_result), "a merge whose result is outside the vocabulary is rejected");
    auto outside_operand = json;
    outside_operand["model"]["merges"].push_back({"he", "llo"}); // "llo" is not in the vocabulary
    check(rejects(outside_operand), "a merge with an operand outside the vocabulary is rejected");
    auto repeated_merge = json;
    repeated_merge["model"]["merges"].push_back({"h", "e"});
    check(rejects(repeated_merge), "a repeated merge rule is rejected");
    auto taken_id = json;
    taken_id["added_tokens"][1]["id"] = 13; // the id of "hello"
    check(rejects(taken_id), "an added token with another token's id is rejected");

    // tokenizers numbers an added token outside the vocabulary from the 17 vocabulary entries on.
    const auto new_token = [](int id) {
        return nlohmann::json{{"id", id},        {"content", "<x>"}, {"single_word", false},
                              {"lstrip", false}, {"rstrip", false},  {"normalized", false},
                              {"special", false}};
    };
    auto next_added = json;
    next_added["added_tokens"].push_back(new_token(17));
    check(!rejects(next_added), "a new added token takes the next id after the vocabulary");
    auto skipping_added = json;
    skipping_added["added_tokens"].push_back(new_token(20));
    check(rejects(skipping_added), "a new added token that skips ids is rejected");
    // The JSON parser keeps the last of repeated keys; a repeated vocab entry is rejected instead.
    auto repeated_key = json.dump();
    const std::string entry = "\"hello\":13";
    const auto at = repeated_key.find(entry);
    check(at != std::string::npos, "the fixture holds the vocab entry to repeat");
    repeated_key.insert(at, entry + ",");
    bool repeated_rejected = false;
    try {
        trtmc::laguna::Tokenizer::create(repeated_key);
    } catch (const std::runtime_error&) {
        repeated_rejected = true;
    }
    check(repeated_rejected, "a repeated vocab key is rejected");

    auto two_operations = json;
    two_operations["post_processor"]["single"][0]["Sequence"] = {{"id", "A"}, {"type_id", 0}};
    check(rejects(two_operations), "a post-processor item with two operations is rejected");
    auto two_sequences = json;
    two_sequences["post_processor"]["single"].push_back(
        {{"Sequence", {{"id", "A"}, {"type_id", 0}}}});
    check(rejects(two_sequences), "a second post-processor Sequence is rejected");
    auto no_sequence = json;
    no_sequence["post_processor"]["single"].erase(1);
    check(rejects(no_sequence), "a post-processor without Sequence A is rejected");

    // "hello hello" is three tokens (hello, \u0120, hello).
    check(tokenizer->encode("hello hello", false, 3).size() == 3,
          "a prompt within the token budget encodes");
    const auto over_budget = [&](const std::string& text, std::size_t max_tokens) {
        try {
            tokenizer->encode(text, false, max_tokens);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(over_budget("hello hello", 2), "a prompt one token over the budget is rejected");
    // The longest token, \u3008|EOS|\u3009, is 11 bytes: no text over 8 x 11 bytes fits 8 tokens.
    check(tokenizer->max_text_bytes(8) == 88,
          "the text limit is the budget times the longest token");
    check(tokenizer->max_text_bytes(SIZE_MAX) == SIZE_MAX, "the text limit saturates");
    check(over_budget(std::string(1 << 20, 'h'), 8),
          "a text longer than the budget times the longest token is rejected");
}

void test_chat_template() {
    const std::string header = "\xE3\x80\x88|EOS|\xE3\x80\x89<system>\n\n";
    const std::string default_system = std::string(trtmc::laguna::kDefaultSystemMessage);
    check(trtmc::laguna::render_chat("Q", "", false) ==
              header + default_system + "\n</system>\n<user>\nQ\n</user>\n<assistant>\n</think>",
          "default system message, thinking off");
    check(trtmc::laguna::render_chat("Q", "", true) ==
              header + default_system + "\n</system>\n<user>\nQ\n</user>\n<assistant>\n<think>",
          "thinking on");
    check(trtmc::laguna::render_chat("Summarize:\n\nThe quick brown fox.\n", "You are terse.  \n",
                                     false) ==
              header + "You are terse.\n</system>\n<user>\nSummarize:\n\nThe quick brown fox.\n\n"
                       "</user>\n<assistant>\n</think>",
          "custom system message is right-stripped");
    check(trtmc::laguna::render_chat("Q", " \n", false) ==
              "\xE3\x80\x88|EOS|\xE3\x80\x89<user>\nQ\n</user>\n<assistant>\n</think>",
          "a blank system message drops the system block");
    // Python's strip() / rstrip() treat U+3000, U+00A0 and U+2028 as whitespace.
    check(trtmc::laguna::render_chat("Q", "\xE3\x80\x80\xC2\xA0", false) ==
              "\xE3\x80\x88|EOS|\xE3\x80\x89<user>\nQ\n</user>\n<assistant>\n</think>",
          "a system message of Unicode whitespace drops the system block");
    check(trtmc::laguna::render_chat("Q", "Be brief.\xE3\x80\x80\xE2\x80\xA8\n", false) ==
              header + "Be brief.\n</system>\n<user>\nQ\n</user>\n<assistant>\n</think>",
          "trailing Unicode whitespace is right-stripped");
    check(trtmc::laguna::render_chat("Q", "Caf\xC3\xA9 \xE3\x80\x89", false) ==
              header +
                  "Caf\xC3\xA9 \xE3\x80\x89\n</system>\n<user>\nQ\n</user>\n<assistant>\n</think>",
          "multi-byte characters that are not whitespace are kept");
}

void test_chat_text_bytes() {
    // The size is computed as render_chat renders, including the system prompt's trailing
    // whitespace trim.
    const std::string long_spaces(10000, ' ');
    const std::string padded = std::string("be terse") + long_spaces;
    for (const auto& [user, system, thinking] :
         std::vector<std::tuple<std::string, std::string, bool>>{
             {"hi", "", false},
             {"hi", padded, true},
             {"\xE6\x97\xA5", long_spaces, false},
             {"", "sys\n", true}})
        check(trtmc::laguna::chat_text_bytes(user, system, thinking) ==
                  trtmc::laguna::render_chat(user, system, thinking).size(),
              "chat_text_bytes is the rendered chat's size");
}

void test_sampler() {
    trtmc::laguna::SamplingParams params;
    params.eos_token_ids = {2, 24};
    const float logits[] = {0.0F, 3.0F, 1.0F, 3.0F};
    auto greedy = trtmc::laguna::create_sampler(params);
    const auto first = greedy->sample(logits, 4, params);
    check(first.token_id == 1 && !first.is_eos, "greedy takes the first maximum");
    const float eos_logits[] = {0.0F, 0.0F, 5.0F};
    check(greedy->sample(eos_logits, 3, params).is_eos, "any listed EOS id stops");
    check(trtmc::laguna::is_eos_token(params, 24) && !trtmc::laguna::is_eos_token(params, 9),
          "EOS membership");

    // Any top_p below 1 enables nucleus sampling: over uniform logits the draws are not all
    // the first maximum that the greedy path returns.
    trtmc::laguna::SamplingParams nucleus;
    nucleus.top_p = 0.999999F;
    auto sampler = trtmc::laguna::create_sampler(nucleus);
    const float uniform[] = {0.0F, 0.0F, 0.0F, 0.0F};
    bool other = false;
    for (int i = 0; i < 32; ++i)
        other = other || sampler->sample(uniform, 4, nucleus).token_id != 0;
    check(other, "top_p just below 1 samples from the nucleus");
}

} // namespace

int main() {
    test_pre_tokenizer();
    test_tokenizer();
    test_chat_template();
    test_chat_text_bytes();
    test_sampler();
    if (failures != 0) {
        std::fprintf(stderr, "%d Laguna text runtime checks failed\n", failures);
        return 1;
    }
    std::printf("Laguna text runtime checks passed\n");
    return 0;
}
