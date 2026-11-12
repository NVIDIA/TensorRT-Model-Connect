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
    test_sampler();
    if (failures != 0) {
        std::fprintf(stderr, "%d Laguna text runtime checks failed\n", failures);
        return 1;
    }
    std::printf("Laguna text runtime checks passed\n");
    return 0;
}
