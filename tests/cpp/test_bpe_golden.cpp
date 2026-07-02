/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Golden correctness tests for BPE tokenizer (and auto-detect fallback).
//
// Verifies encode() output matches HuggingFace reference token IDs.
//
// Set TOKENIZER_JSON + GOLDEN_VECTORS env vars to test real models.
// Set ADD_SPECIAL_TOKENS=1 to enable special token insertion (default: off).
//
// Generate golden vectors with:
//   python3 tests/tools/generate_bpe_golden.py \
//       --model example-org/bpe-decoder \
//       --output tests/data/newline_aware_golden.txt
//
// Trace: ARCH-TOK-BPE, UD-TOK-BPE-02
// Intent: Verify native tokenizer produces identical token sequences to HF.
// Preconditions: TOKENIZER_JSON and GOLDEN_VECTORS env vars set.
// Postconditions: All golden vectors match (exit 0).

#include "trtmc/tokenizer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;

void check_ids(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
               const std::string& label) {
    if (actual == expected) {
        std::cerr << "PASS: " << label << " (" << actual.size() << " tokens)\n";
        return;
    }
    std::cerr << "FAIL: " << label << "\n";
    std::cerr << "  expected [";
    for (size_t i = 0; i < expected.size(); ++i)
        std::cerr << (i ? "," : "") << expected[i];
    std::cerr << "]\n  actual   [";
    for (size_t i = 0; i < actual.size(); ++i)
        std::cerr << (i ? "," : "") << actual[i];
    std::cerr << "]\n";
    size_t min_len = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < min_len; ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "  first mismatch at index " << i << ": expected " << expected[i]
                      << " got " << actual[i] << "\n";
            break;
        }
    }
    if (actual.size() != expected.size())
        std::cerr << "  length: expected " << expected.size() << " got " << actual.size() << "\n";
    failures++;
}

std::string unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
            case 't':
                out.push_back('\t');
                ++i;
                break;
            case 'n':
                out.push_back('\n');
                ++i;
                break;
            case 'r':
                out.push_back('\r');
                ++i;
                break;
            case '\\':
                out.push_back('\\');
                ++i;
                break;
            default:
                out.push_back(s[i]);
                break;
            }
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

int parse_golden_line(const std::string& line, std::string& out_text,
                      std::vector<int32_t>& out_ids) {
    auto tab = line.find('\t');
    if (tab == std::string::npos)
        return 0;
    out_text = unescape(line.substr(0, tab));
    out_ids.clear();
    std::string ids_str = line.substr(tab + 1);
    if (ids_str.empty())
        return 1;
    std::istringstream iss(ids_str);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        if (!tok.empty())
            out_ids.push_back(std::stoi(tok));
    }
    return 1;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return "";
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Try BPE → WordPiece → Unigram (mirrors try_create_native_tokenizer)
std::unique_ptr<trtmc::ITokenizer> create_tokenizer(const char* data, size_t size,
                                                    bool add_special) {
    try {
        auto tok = trtmc::CreateBpeTokenizer(data, size, add_special);
        if (tok) {
            std::cerr << "  [type: BPE]\n";
            return tok;
        }
    } catch (...) {
    }

    try {
        auto tok = trtmc::CreateWordPieceTokenizer(data, size, add_special);
        if (tok) {
            std::cerr << "  [type: WordPiece]\n";
            return tok;
        }
    } catch (...) {
    }

    try {
        auto tok = trtmc::CreateUnigramTokenizer(data, size, add_special);
        if (tok) {
            std::cerr << "  [type: Unigram]\n";
            return tok;
        }
    } catch (...) {
    }

    return nullptr;
}

int main() {
    std::cerr << "Native Tokenizer Golden Correctness Tests\n\n";

    const char* json_path = std::getenv("TOKENIZER_JSON");
    const char* golden_path = std::getenv("GOLDEN_VECTORS");
    const char* add_special_env = std::getenv("ADD_SPECIAL_TOKENS");
    bool add_special = add_special_env && std::string(add_special_env) == "1";

    if (!json_path || !golden_path) {
        std::cerr << "No TOKENIZER_JSON/GOLDEN_VECTORS set, skipping file-based tests.\n";
        return 0;
    }

    std::string json_data = read_file(json_path);
    if (json_data.empty()) {
        std::cerr << "ERROR: cannot read " << json_path << "\n";
        return 1;
    }

    std::cerr << "Tokenizer: " << json_path << "\n";
    auto tok = create_tokenizer(json_data.data(), json_data.size(), add_special);
    if (!tok) {
        std::cerr << "ERROR: no native tokenizer matched\n";
        return 1;
    }

    std::ifstream golden_file(golden_path);
    if (!golden_file) {
        std::cerr << "ERROR: cannot open " << golden_path << "\n";
        return 1;
    }

    int line_num = 0;
    int pass = 0;
    int total = 0;
    std::string line;
    while (std::getline(golden_file, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#')
            continue;

        std::string text;
        std::vector<int32_t> expected_ids;
        if (!parse_golden_line(line, text, expected_ids))
            continue;

        ++total;
        auto actual_ids = tok->encode(text);
        std::string label = "golden_line_" + std::to_string(line_num);

        if (actual_ids == expected_ids) {
            ++pass;
            std::cerr << "PASS: " << label << " (" << actual_ids.size() << " tokens)\n";
        } else {
            check_ids(actual_ids, expected_ids, label);
            std::string display = text.substr(0, 60);
            if (text.size() > 60)
                display += "...";
            std::cerr << "  input: \"" << display << "\"\n";
        }
    }

    std::cerr << "\nResult: " << pass << "/" << total << " passed\n";

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All golden tests passed!\n";
    return 0;
}
