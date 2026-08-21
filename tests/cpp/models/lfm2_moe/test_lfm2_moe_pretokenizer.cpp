/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2_moe/plugin_helpers.h"
#include "runtime/models/lfm2_moe/pretokenizer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

using Ids = std::vector<int32_t>;
using PieceMap = std::unordered_map<std::string, Ids>;

PieceMap pinned_piece_ids() {
    return {
        {"Hello", {36309}},
        {",", {521}},
        {" world", {2031}},
        {"!", {510}},
        {"Café", {544, 2305, 860}},
        {" déjà", {18913}},
        {" vu", {24166}},
        {".", {523}},
        {"¿Dónde", {17995, 545, 58838}},
        {" está", {7453}},
        {" París", {35820}},
        {"?", {540}},
        {"中文标点", {2377, 5467, 10257, 6338}},
        {"，测试", {1198, 57832}},
        {"123", {10293}},
        {"。", {1147}},
        {"日本語", {62506}},
        {"・テスト", {2928, 4374, 7133}},
        {"한국어", {61082, 5193}},
        {" 테스트", {54674, 33701}},
        {" ", {730}},
        {"Привет", {7245, 13328}},
        {" мир", {49182}},
        {"مرحبا", {18381, 2763, 2071, 978}},
        {"،", {3780}},
        {" العالم", {36273}},
        {"١٢٣", {659, 604, 659, 605, 659, 606}},
        {"नमस", {6355, 611, 6355, 616, 6355, 626}},
        {"्त", {34801, 607}},
        {"े", {50008}},
        {" द", {17903, 609}},
        {"ुन", {14860, 733, 6355, 611}},
        {"िय", {61840, 617}},
        {"ा", {30170}},
        {"१२३", {14860, 610, 14860, 611, 14860, 612}},
        {"Grüße", {10706, 1142, 7017}},
        {" Welt", {9808}},
        {"Bonjour", {34933, 13143}},
        {" ça", {18757}},
        {" va", {4223}},
        {" ?", {5219}},
        {"I", {550}},
        {"'M", {51251}},
        {"WE", {24435}},
        {"'RE", {516, 3277}},
        {"HE", {6347}},
        {"'LL", {516, 5092}},
        {"DON", {545, 2332}},
        {"'T", {32640}},
        {"SHE", {560, 6347}},
        {"'S", {24406}},
        {"a", {574}},
        {" b", {794}},
        {"5", {530}},
        {" \t", {46482}},
        {"trailing", {19310, 8206}},
        {"  ", {767}},
        {"def", {3663}},
        {" f", {792}},
        {"():\n", {32711}},
        {"   ", {861}},
        {" return", {1789}},
        {" x", {3053}},
        {"|", {601}},
        {" a", {768}},
        {" |", {2171}},
    };
}

class PinnedPieceTokenizer final : public trtmc::ITokenizer {
  public:
    PinnedPieceTokenizer() : pieces_(pinned_piece_ids()) {
        added_ = {{"<|startoftext|>", 1},
                  {"<|im_start|>", 6},
                  {"<|im_end|>", 7},
                  {"<|tool_list_start|>", 8},
                  {"<|tool_list_end|>", 9},
                  {"<|tool_call_start|>", 10},
                  {"<|tool_call_end|>", 11},
                  {"<|tool_response_start|>", 12},
                  {"<|tool_response_end|>", 13}};
    }

    Ids encode(const std::string& text) const override {
        if (const auto found = pieces_.find(text); found != pieces_.end())
            return found->second;
        if (const auto found = added_.find(text); found != added_.end())
            return {found->second};
        return {-1};
    }
    std::string decode(const Ids&) const override { return {}; }
    int32_t id_for_token(std::string_view token) const override {
        const auto found = added_.find(std::string(token));
        return found == added_.end() ? -1 : found->second;
    }
    std::string token_for_id(int32_t id) const override {
        for (const auto& [token, token_id] : added_) {
            if (token_id == id)
                return token;
        }
        return {};
    }

  private:
    PieceMap pieces_;
    std::unordered_map<std::string, int32_t> added_;
};

struct Fixture {
    const char* name;
    std::string text;
    std::vector<std::string> pieces;
    Ids ids;
};

std::vector<Fixture> supported_language_fixtures() {
    return {
        {"english", "Hello, world!", {"Hello", ",", " world", "!"}, {36309, 521, 2031, 510}},
        {"french",
         "Café déjà vu.",
         {"Café", " déjà", " vu", "."},
         {544, 2305, 860, 18913, 24166, 523}},
        {"spanish",
         "¿Dónde está París?",
         {"¿Dónde", " está", " París", "?"},
         {17995, 545, 58838, 7453, 35820, 540}},
        {"chinese",
         "中文标点，测试123。",
         {"中文标点", "，测试", "123", "。"},
         {2377, 5467, 10257, 6338, 1198, 57832, 10293, 1147}},
        {"japanese",
         "日本語・テスト123",
         {"日本語", "・テスト", "123"},
         {62506, 2928, 4374, 7133, 10293}},
        {"korean",
         "한국어 테스트 123",
         {"한국어", " 테스트", " ", "123"},
         {61082, 5193, 54674, 33701, 730, 10293}},
        {"russian",
         "Привет, мир! 123",
         {"Привет", ",", " мир", "!", " ", "123"},
         {7245, 13328, 521, 49182, 510, 730, 10293}},
        {"arabic",
         "مرحبا، العالم ١٢٣",
         {"مرحبا", "،", " العالم", " ", "١٢٣"},
         {18381, 2763, 2071, 978, 3780, 36273, 730, 659, 604, 659, 605, 659, 606}},
        {"hindi",
         "नमस्ते दुनिया १२३",
         {"नमस", "्त", "े", " द", "ुन", "िय", "ा", " ", "१२३"},
         {6355, 611, 6355,  616, 6355,  626, 34801, 607, 50008, 17903, 609,   14860, 733,
          6355, 611, 61840, 617, 30170, 730, 14860, 610, 14860, 611,   14860, 612}},
        {"german",
         "Grüße, Welt! 123",
         {"Grüße", ",", " Welt", "!", " ", "123"},
         {10706, 1142, 7017, 521, 9808, 510, 730, 10293}},
        {"french-question",
         "Bonjour, ça va ? 123",
         {"Bonjour", ",", " ça", " va", " ?", " ", "123"},
         {34933, 13143, 521, 18757, 4223, 5219, 730, 10293}},
    };
}

std::vector<Fixture> whitespace_run_fixtures() {
    // Pins the \s+(?!\S) alternative of the pinned Split regex: a
    // newline-free whitespace run followed by non-whitespace yields its final
    // whitespace unit to the next piece. Every id below is the byte-level BPE
    // encoding the reference Hugging Face tokenizer assigns the same text.
    return {
        {"double-space", "a  b", {"a", " ", " b"}, {574, 730, 794}},
        {"space-before-digit", "a  5", {"a", " ", " ", "5"}, {574, 730, 730, 530}},
        {"space-tab", "a \t b", {"a", " \t", " b"}, {574, 46482, 794}},
        {"trailing-run", "trailing  ", {"trailing", "  "}, {19310, 8206, 767}},
        {"indented-code",
         "def f():\n    return  x",
         {"def", " f", "():\n", "   ", " return", " ", " x"},
         {3663, 792, 32711, 861, 1789, 730, 3053}},
        {"markdown-row",
         "| a |  b |",
         {"|", " a", " |", " ", " b", " |"},
         {601, 768, 2171, 730, 794, 2171}},
    };
}

void test_pinned_metadata_detection() {
    const std::string pinned = R"({"pre_tokenizer":{"type":"Sequence","pretokenizers":[
      {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
      {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
    ]}})";
    check(trtmc::lfm2_moe_uses_pinned_split_byte_level_pretokenizer(pinned.data(), pinned.size()),
          "detect pinned Split plus ByteLevel pretokenizer");
    std::string changed = pinned;
    const auto position = changed.find("\"use_regex\":false");
    changed.replace(position, std::string("\"use_regex\":false").size(), "\"use_regex\":true");
    check(!trtmc::lfm2_moe_uses_pinned_split_byte_level_pretokenizer(changed.data(), changed.size()),
          "fail closed for a different pretokenizer contract");
}

void test_supported_language_piece_and_id_parity() {
    const std::vector<std::string> added{"<|startoftext|>", "<|im_start|>", "<|im_end|>",
                                         "<|tool_call_start|>"};
    for (const auto& fixture : supported_language_fixtures()) {
        check(trtmc::lfm2_moe_split_pretokens(fixture.text) == fixture.pieces,
              std::string(fixture.name) + " pinned Split pieces");
        auto tokenizer =
            trtmc::lfm2_moe_wrap_pinned_pretokenizer(std::make_unique<PinnedPieceTokenizer>(), added);
        check(tokenizer->encode(fixture.text) == fixture.ids,
              std::string(fixture.name) + " pinned token ids");
    }
}

void test_whitespace_run_piece_and_id_parity() {
    const std::vector<std::string> added{"<|startoftext|>", "<|im_start|>", "<|im_end|>",
                                         "<|tool_call_start|>"};
    for (const auto& fixture : whitespace_run_fixtures()) {
        check(trtmc::lfm2_moe_split_pretokens(fixture.text) == fixture.pieces,
              std::string(fixture.name) + " pinned Split pieces");
        auto tokenizer =
            trtmc::lfm2_moe_wrap_pinned_pretokenizer(std::make_unique<PinnedPieceTokenizer>(), added);
        check(tokenizer->encode(fixture.text) == fixture.ids,
              std::string(fixture.name) + " pinned token ids");
    }
}

void test_case_insensitive_contractions() {
    const std::vector<Fixture> fixtures{
        {"I'M", "I'M", {"I", "'M"}, {550, 51251}},
        {"WE'RE", "WE'RE", {"WE", "'RE"}, {24435, 516, 3277}},
        {"HE'LL", "HE'LL", {"HE", "'LL"}, {6347, 516, 5092}},
        {"DON'T", "DON'T", {"DON", "'T"}, {545, 2332, 32640}},
        {"SHE'S", "SHE'S", {"SHE", "'S"}, {560, 6347, 24406}},
    };
    for (const auto& fixture : fixtures) {
        check(trtmc::lfm2_moe_split_pretokens(fixture.text) == fixture.pieces,
              fixture.text + " contraction pieces");
        auto tokenizer =
            trtmc::lfm2_moe_wrap_pinned_pretokenizer(std::make_unique<PinnedPieceTokenizer>(), {});
        check(tokenizer->encode(fixture.text) == fixture.ids,
              fixture.text + " contraction token ids");
    }
}

void test_chat_and_tool_added_tokens_are_not_split() {
    const std::vector<std::string> added{"<|startoftext|>", "<|im_start|>", "<|im_end|>",
                                         "<|tool_call_start|>"};
    auto tokenizer =
        trtmc::lfm2_moe_wrap_pinned_pretokenizer(std::make_unique<PinnedPieceTokenizer>(), added);
    const std::string raw = "<|startoftext|><|im_start|>Hello<|im_end|><|tool_call_start|>";
    check(tokenizer->encode(raw) == Ids({1, 6, 36309, 7, 10}),
          "added-token extraction preserves BOS, ChatML, and tool IDs");
}

void test_real_pinned_tokenizer_when_provided() {
    const char* path = std::getenv("TRTMC_LFM2_TOKENIZER_JSON");
    if (path == nullptr || *path == '\0')
        return;
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "open externally supplied pinned tokenizer.json");
    if (!input)
        return;
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    trtmc::BundleFile bundle;
    bundle.info.tokenizer_add_special_tokens = false;
    bundle.info.tokenizer_add_special_tokens_present = true;
    bundle.sections.push_back({"tokenizer.json", std::vector<char>(json.begin(), json.end())});
    auto tokenizer = trtmc::create_lfm2_moe_tokenizer(bundle);
    check(tokenizer->decode({1334, 542, 518, 5242}) == ":\nA) Paris",
          "real pinned tokenizer fixes the reported decoder token sequence");
    for (const auto& fixture : supported_language_fixtures()) {
        check(tokenizer->encode(fixture.text) == fixture.ids,
              std::string("real pinned tokenizer ids: ") + fixture.name);
    }

    const std::string chat =
        "<|startoftext|><|im_start|>user\nWhat is the capital of France? Answer in one word."
        "<|im_end|>\n<|im_start|>assistant\n";
    const Ids expected_chat{1,     6,   6423, 708,  3493, 856, 779, 5706, 803,   4481, 540,
                            42275, 797, 1235, 2359, 523,  7,   708, 6,    64015, 708};
    check(tokenizer->encode(chat) == expected_chat,
          "real pinned tokenizer preserves exact chat prompt IDs");

    const std::string tool_prompt = "<|startoftext|><|im_start|>Hello<|im_end|><|tool_call_start|>";
    check(tokenizer->encode(tool_prompt) == Ids({1, 6, 36309, 7, 10}),
          "real pinned tokenizer preserves raw ChatML and tool token IDs");

    const std::vector<std::pair<std::string, Ids>> contractions{
        {"I'M", {550, 51251}},         {"WE'RE", {24435, 516, 3277}}, {"HE'LL", {6347, 516, 5092}},
        {"DON'T", {545, 2332, 32640}}, {"SHE'S", {560, 6347, 24406}},
    };
    for (const auto& [text, ids] : contractions)
        check(tokenizer->encode(text) == ids, "real pinned contraction IDs: " + text);
}

} // namespace

int main() {
    test_pinned_metadata_detection();
    test_supported_language_piece_and_id_parity();
    test_whitespace_run_piece_and_id_parity();
    test_case_insensitive_contractions();
    test_chat_and_tool_added_tokens_are_not_split();
    test_real_pinned_tokenizer_when_provided();
    return failures;
}
