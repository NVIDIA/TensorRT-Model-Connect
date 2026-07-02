/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Golden correctness tests for SentencePiece Unigram tokenizer.
//
// Two modes:
//   1. Built-in: minimal vocab with hand-verified golden vectors
//   2. Real model: TOKENIZER_JSON + GOLDEN_VECTORS env vars
//
// Generate golden vectors:
//   python3 tests/tools/generate_unigram_golden.py \
//       --model FacebookAI/xlm-roberta-base \
//       --output tests/data/xlm_roberta_golden.txt
//
// Trace: ARCH-TOK-UNI, UD-TOK-UNI-02

#include "trtmc/tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;

void check(bool condition, const std::string& name)
{
    if (!condition) { std::cerr << "FAIL: " << name << std::endl; failures++; }
    else { std::cerr << "PASS: " << name << std::endl; }
}

void check_ids(const std::vector<int32_t>& actual,
               const std::vector<int32_t>& expected,
               const std::string& label)
{
    if (actual == expected) {
        std::cerr << "PASS: " << label << " (" << actual.size() << " tokens)\n";
        return;
    }
    std::cerr << "FAIL: " << label << "\n  expected [";
    for (size_t i = 0; i < expected.size(); ++i)
        std::cerr << (i ? "," : "") << expected[i];
    std::cerr << "]\n  actual   [";
    for (size_t i = 0; i < actual.size(); ++i)
        std::cerr << (i ? "," : "") << actual[i];
    std::cerr << "]\n";
    failures++;
}

std::string unescape(const std::string& s)
{
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case 't': out.push_back('\t'); ++i; break;
                case 'n': out.push_back('\n'); ++i; break;
                case 'r': out.push_back('\r'); ++i; break;
                case '\\': out.push_back('\\'); ++i; break;
                default: out.push_back(s[i]); break;
            }
        } else { out.push_back(s[i]); }
    }
    return out;
}

int parse_golden_line(const std::string& line, std::string& text, std::vector<int32_t>& ids)
{
    auto tab = line.find('\t');
    if (tab == std::string::npos) return 0;
    text = unescape(line.substr(0, tab));
    ids.clear();
    std::string s = line.substr(tab + 1);
    if (s.empty()) return 1;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, ','))
        if (!tok.empty()) ids.push_back(std::stoi(tok));
    return 1;
}

std::string read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// ─── Built-in golden vectors ───

static const char* kSmallUnigramJson = R"({
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["\u2581the", -4.0],
      ["h", -3.0],
      ["e", -3.0],
      ["l", -3.0],
      ["o", -3.0],
      ["!", -2.5],
      ["\u2581cat", -5.0]
    ]
  },
  "pre_tokenizer": {
    "type": "Metaspace",
    "replacement": "\u2581",
    "add_prefix_space": true
  },
  "added_tokens": [
    {"id": 10, "content": "<s>", "special": true},
    {"id": 11, "content": "</s>", "special": true}
  ],
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ]
  }
})";

void run_builtin_golden_tests()
{
    std::cerr << "=== Built-in Golden Vectors (Unigram) ===\n";

    std::string json(kSmallUnigramJson);

    // Without special tokens
    {
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);
        check_ids(tok->encode("hello"), {1}, "golden_hello");
        check_ids(tok->encode("hello world"), {1, 2}, "golden_hello_world");
        check_ids(tok->encode("the cat"), {3, 9}, "golden_the_cat");
        check_ids(tok->encode(""), {}, "golden_empty");

        check(tok->decode({1}) == "hello", "golden_decode_hello");
        check(tok->decode({1, 2}) == "hello world", "golden_decode_hello_world");
    }

    // With special tokens
    {
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), true);
        check_ids(tok->encode("hello"), {10, 1, 11}, "golden_hello_special");
        check_ids(tok->encode(""), {10, 11}, "golden_empty_special");
    }
}

// ─── File-based golden tests ───

int run_file_golden_tests(const std::string& json_data, const std::string& golden_path,
                          bool add_special)
{
    std::cerr << "\n=== File Golden: " << golden_path << " ===\n";

    auto tok = trtmc::CreateUnigramTokenizer(json_data.data(), json_data.size(), add_special);
    if (!tok) { std::cerr << "ERROR: failed to create tokenizer\n"; return 1; }

    std::ifstream f(golden_path);
    if (!f) { std::cerr << "ERROR: cannot open " << golden_path << "\n"; return 1; }

    int line_num = 0, pass = 0;
    std::string line;
    while (std::getline(f, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#') continue;
        std::string text;
        std::vector<int32_t> expected;
        if (!parse_golden_line(line, text, expected)) continue;

        auto actual = tok->encode(text);
        std::string label = "golden_line_" + std::to_string(line_num);
        if (actual == expected) {
            ++pass;
            std::cerr << "PASS: " << label << " (" << actual.size() << " tokens)\n";
        } else {
            check_ids(actual, expected, label);
            std::cerr << "  input: \"" << text.substr(0, 60) << "\"\n";
        }
    }
    std::cerr << "\nFile golden: " << pass << "/" << line_num << " passed\n";
    return 0;
}

int main()
{
    std::cerr << "Unigram Tokenizer Golden Correctness Tests\n\n";

    run_builtin_golden_tests();

    const char* json_path = std::getenv("TOKENIZER_JSON");
    const char* golden_path = std::getenv("GOLDEN_VECTORS");
    const char* add_special_env = std::getenv("ADD_SPECIAL_TOKENS");
    bool add_special = true;
    if (add_special_env && std::string(add_special_env) == "0") add_special = false;

    if (json_path && golden_path) {
        std::string json_data = read_file(json_path);
        if (json_data.empty()) { std::cerr << "ERROR: cannot read " << json_path << "\n"; return 1; }
        run_file_golden_tests(json_data, golden_path, add_special);
    }

    if (failures > 0) { std::cerr << "\n" << failures << " test(s) FAILED\n"; return 1; }
    std::cerr << "\nAll Unigram golden tests passed!\n";
    return 0;
}
