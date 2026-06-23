// Golden correctness tests for WordPiece tokenizer.
//
// Verifies encode() output matches HuggingFace reference token IDs.
//
// Two modes:
//   1. Built-in: runs with embedded small vocab and hand-verified golden vectors
//   2. Real model: set TOKENIZER_JSON + GOLDEN_VECTORS to test against HF output
//
// Generate golden vectors with:
//   python3 tests/tools/generate_wordpiece_golden.py \
//       --model example-org/wordpiece-encoder \
//       --output tests/data/wordpiece_encoder_golden.txt
//
// Golden vector file format (one test per line):
//   <text>\t<id1>,<id2>,<id3>,...
//
// Trace: ARCH-TOK-WP, UD-TOK-WP-02
// Intent: Verify WordPiece tokenizer produces identical token sequences to HF.
// Preconditions: For file mode, TOKENIZER_JSON and GOLDEN_VECTORS env vars set.
// Postconditions: All golden vectors match (exit 0).

#include "trtmc/tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        failures++;
    } else {
        std::cerr << "PASS: " << name << std::endl;
    }
}

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
    if (actual.size() != expected.size()) {
        std::cerr << "  length mismatch: expected " << expected.size() << " got " << actual.size()
                  << "\n";
    }
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

// ─── Built-in golden vectors ───

static const char* kSmallWordPieceJson = R"({
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
      "hello": 4, "world": 5, "un": 6, "##aff": 7, "##able": 8,
      "the": 9, "!": 10, ",": 11, ".": 12,
      "test": 13, "##ing": 14, "cat": 15, "##s": 16
    }
  },
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "lowercase": true
  },
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "special": true},
    {"id": 1, "content": "[UNK]", "special": true},
    {"id": 2, "content": "[CLS]", "special": true},
    {"id": 3, "content": "[SEP]", "special": true}
  ]
})";

void run_builtin_golden_tests() {
    std::cerr << "=== Built-in Golden Vectors (WordPiece) ===\n";

    std::string json(kSmallWordPieceJson);

    // Without special tokens
    {
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // "hello" → [4]
        check_ids(tok->encode("hello"), {4}, "golden_hello");

        // "hello world" → [4, 5]
        check_ids(tok->encode("hello world"), {4, 5}, "golden_hello_world");

        // "unaffable" → "un" + "##aff" + "##able" = [6, 7, 8]
        check_ids(tok->encode("unaffable"), {6, 7, 8}, "golden_subword");

        // "HELLO" → lowercased to "hello" → [4]
        check_ids(tok->encode("HELLO"), {4}, "golden_uppercase");

        // "testing" → "test" + "##ing" = [13, 14]
        check_ids(tok->encode("testing"), {13, 14}, "golden_testing");

        // "cats" → "cat" + "##s" = [15, 16]
        check_ids(tok->encode("cats"), {15, 16}, "golden_cats");

        // "hello!" → [4, 10]
        check_ids(tok->encode("hello!"), {4, 10}, "golden_punct");

        // Empty → []
        check_ids(tok->encode(""), {}, "golden_empty");

        // OOV → [1] (UNK)
        check_ids(tok->encode("xyz"), {1}, "golden_oov");

        // Round-trip checks
        for (const auto& text : {"hello", "hello world", "unaffable", "testing"}) {
            auto decoded = tok->decode(tok->encode(text));
            check(decoded == text, std::string("golden_roundtrip_") + text);
        }
    }

    // With special tokens
    {
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), true);

        // [CLS] + tokens + [SEP]
        check_ids(tok->encode("hello"), {2, 4, 3}, "golden_hello_special");
        check_ids(tok->encode("hello world"), {2, 4, 5, 3}, "golden_hello_world_special");
        check_ids(tok->encode(""), {2, 3}, "golden_empty_special");
    }
}

// ─── File-based golden tests ───

int run_file_golden_tests(const std::string& json_data, const std::string& golden_path,
                          bool add_special) {
    std::cerr << "\n=== File Golden Vectors: " << golden_path << " ===\n";

    auto tok = trtmc::CreateWordPieceTokenizer(json_data.data(), json_data.size(), add_special);
    if (!tok) {
        std::cerr << "ERROR: failed to create tokenizer\n";
        return 1;
    }

    std::ifstream golden_file(golden_path);
    if (!golden_file) {
        std::cerr << "ERROR: cannot open " << golden_path << "\n";
        return 1;
    }

    int line_num = 0;
    int pass = 0;
    std::string line;
    while (std::getline(golden_file, line)) {
        ++line_num;
        if (line.empty() || line[0] == '#')
            continue;

        std::string text;
        std::vector<int32_t> expected_ids;
        if (!parse_golden_line(line, text, expected_ids)) {
            std::cerr << "WARNING: skipping malformed line " << line_num << "\n";
            continue;
        }

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

    std::cerr << "\nFile golden: " << pass << "/" << line_num << " passed\n";
    return 0;
}

int main() {
    std::cerr << "WordPiece Tokenizer Golden Correctness Tests\n\n";

    run_builtin_golden_tests();

    // File-based tests
    const char* json_path = std::getenv("TOKENIZER_JSON");
    const char* golden_path = std::getenv("GOLDEN_VECTORS");
    // Default: add_special_tokens = true for WordPiece file golden tests
    const char* add_special_env = std::getenv("ADD_SPECIAL_TOKENS");
    bool add_special = true;
    if (add_special_env && std::string(add_special_env) == "0") {
        add_special = false;
    }

    if (json_path && golden_path) {
        std::string json_data = read_file(json_path);
        if (json_data.empty()) {
            std::cerr << "ERROR: cannot read " << json_path << "\n";
            return 1;
        }
        run_file_golden_tests(json_data, golden_path, add_special);
    } else if (json_path || golden_path) {
        std::cerr << "\nNote: set both TOKENIZER_JSON and GOLDEN_VECTORS for real model tests\n";
    }

    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "\nAll WordPiece golden tests passed!\n";
    return 0;
}
