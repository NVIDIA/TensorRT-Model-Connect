/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for WordPiece tokenizer.
//
// Tests: JSON parsing, BertNormalizer, BertPreTokenizer, encoding (greedy
// longest-match), decoding, token lookup, special tokens, edge cases.
//
// Trace: ARCH-TOK-WP, UD-TOK-WP-01
// Intent: Verify WordPiece tokenizer correctness against known expected values.
// Preconditions: None (self-contained JSON test data).
// Postconditions: All checks pass (exit 0) or report failures (exit 1).

#include "trtmc/tokenizer.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

void check(bool condition, const std::string& name)
{
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        failures++;
    } else {
        std::cerr << "PASS: " << name << std::endl;
    }
}

void check_ids(const std::vector<int32_t>& actual,
               const std::vector<int32_t>& expected,
               const std::string& label)
{
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
            std::cerr << "  first mismatch at index " << i
                      << ": expected " << expected[i] << " got " << actual[i] << "\n";
            break;
        }
    }
    if (actual.size() != expected.size()) {
        std::cerr << "  length mismatch: expected " << expected.size()
                  << " got " << actual.size() << "\n";
    }
    failures++;
}

// ─── Minimal WordPiece tokenizer JSON (lowercase, uncased style) ───
static const char* kWordPieceJson = R"({
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[CLS]": 2,
      "[SEP]": 3,
      "[MASK]": 4,
      "hello": 5,
      "world": 6,
      "un": 7,
      "##aff": 8,
      "##able": 9,
      "the": 10,
      "quick": 11,
      "brown": 12,
      "fox": 13,
      "!": 14,
      ".": 15,
      ",": 16,
      "?": 17,
      "a": 18,
      "b": 19,
      "##c": 20,
      "##d": 21,
      "test": 22,
      "##ing": 23,
      "##s": 24,
      "##ed": 25,
      "re": 26,
      "##play": 27,
      "cat": 28,
      "##s": 24,
      "dog": 29,
      "i": 30,
      "love": 31,
      "you": 32,
      "##r": 33,
      "good": 34,
      "morning": 35
    }
  },
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": null,
    "lowercase": true
  },
  "pre_tokenizer": {
    "type": "BertPreTokenizer"
  },
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "special": true},
    {"id": 1, "content": "[UNK]", "special": true},
    {"id": 2, "content": "[CLS]", "special": true},
    {"id": 3, "content": "[SEP]", "special": true},
    {"id": 4, "content": "[MASK]", "special": true}
  ]
})";

// ─── Non-lowercase (cased) tokenizer ───
static const char* kCasedJson = R"({
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
      "Hello": 4, "hello": 5, "World": 6, "world": 7
    }
  },
  "normalizer": {
    "type": "BertNormalizer",
    "clean_text": true,
    "handle_chinese_chars": true,
    "strip_accents": false,
    "lowercase": false
  },
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "special": true},
    {"id": 1, "content": "[UNK]", "special": true},
    {"id": 2, "content": "[CLS]", "special": true},
    {"id": 3, "content": "[SEP]", "special": true}
  ]
})";

int main()
{
    std::cerr << "WordPiece Tokenizer Unit Tests\n\n";

    // ════════════════════════════════════════════════════════════
    // 1. Factory / JSON parsing
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "=== Factory & JSON Parsing ===\n";

        // Valid WordPiece JSON
        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), true);
        check(tok != nullptr, "create_from_valid_json");

        // Invalid JSON
        bool threw = false;
        try {
            trtmc::CreateWordPieceTokenizer("not json", 8, true);
        } catch (const std::exception&) { threw = true; }
        check(threw, "reject_invalid_json");

        // BPE type rejected
        const char* bpe_json = R"({"model":{"type":"BPE","vocab":{},"merges":[]}})";
        threw = false;
        try {
            trtmc::CreateWordPieceTokenizer(bpe_json, std::strlen(bpe_json), true);
        } catch (const std::exception&) { threw = true; }
        check(threw, "reject_bpe_type");

        // Missing vocab
        const char* no_vocab = R"({"model":{"type":"WordPiece"}})";
        threw = false;
        try {
            trtmc::CreateWordPieceTokenizer(no_vocab, std::strlen(no_vocab), true);
        } catch (const std::exception&) { threw = true; }
        check(threw, "reject_missing_vocab");

        // Missing model.type
        const char* no_type = R"({"model":{"vocab":{}}})";
        threw = false;
        try {
            trtmc::CreateWordPieceTokenizer(no_type, std::strlen(no_type), true);
        } catch (const std::exception&) { threw = true; }
        check(threw, "reject_missing_type");
    }

    // ════════════════════════════════════════════════════════════
    // 2. Token/ID lookup
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Token/ID Lookup ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        check(tok->id_for_token("hello") == 5, "id_for_token_hello");
        check(tok->id_for_token("[UNK]") == 1, "id_for_token_unk");
        check(tok->id_for_token("nonexistent") == -1, "id_for_token_missing");
        check(tok->id_for_token("##able") == 9, "id_for_token_subword");

        check(tok->token_for_id(5) == "hello", "token_for_id_5");
        check(tok->token_for_id(2) == "[CLS]", "token_for_id_cls");
        check(tok->token_for_id(-1) == "", "token_for_id_negative");
        check(tok->token_for_id(99999) == "", "token_for_id_oob");
    }

    // ════════════════════════════════════════════════════════════
    // 3. Encoding - basic
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (basic) ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Simple single word
        check_ids(tok->encode("hello"), {5}, "encode_hello");

        // Two words
        check_ids(tok->encode("hello world"), {5, 6}, "encode_hello_world");

        // Subword splitting: "unaffable" → "un" + "##aff" + "##able"
        check_ids(tok->encode("unaffable"), {7, 8, 9}, "encode_subword");

        // Word with suffix: "testing" → "test" + "##ing"
        check_ids(tok->encode("testing"), {22, 23}, "encode_testing");

        // OOV word → [UNK]
        check_ids(tok->encode("xyzzy"), {1}, "encode_oov");

        // Empty string (no special tokens)
        check_ids(tok->encode(""), {}, "encode_empty_no_special");

        // Multiple words with mix of known/unknown
        check_ids(tok->encode("hello xyzzy world"), {5, 1, 6}, "encode_mixed_oov");
    }

    // ════════════════════════════════════════════════════════════
    // 4. Encoding - special tokens
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (special tokens) ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), true);

        // With special tokens: [CLS] + tokens + [SEP]
        check_ids(tok->encode("hello"), {2, 5, 3}, "encode_hello_with_special");

        check_ids(tok->encode("hello world"), {2, 5, 6, 3}, "encode_hello_world_with_special");

        // Empty with special tokens
        check_ids(tok->encode(""), {2, 3}, "encode_empty_with_special");
    }

    // ════════════════════════════════════════════════════════════
    // 5. Encoding - normalizer (lowercase)
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (normalizer) ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Uppercase input → lowercase by normalizer
        check_ids(tok->encode("HELLO"), {5}, "encode_uppercase");
        check_ids(tok->encode("Hello World"), {5, 6}, "encode_mixed_case");

        // Punctuation handling
        check_ids(tok->encode("hello!"), {5, 14}, "encode_with_punct");
        check_ids(tok->encode("hello, world!"), {5, 16, 6, 14}, "encode_punct_separated");

        // Multiple spaces (should be treated as single separator)
        check_ids(tok->encode("hello   world"), {5, 6}, "encode_multiple_spaces");
    }

    // ════════════════════════════════════════════════════════════
    // 6. Encoding - cased model
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (cased model) ===\n";

        std::string json(kCasedJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Case-sensitive: "Hello" != "hello"
        check_ids(tok->encode("Hello"), {4}, "encode_cased_Hello");
        check_ids(tok->encode("hello"), {5}, "encode_cased_hello");
        check_ids(tok->encode("Hello World"), {4, 6}, "encode_cased_Hello_World");

        // Unknown casing
        check_ids(tok->encode("HELLO"), {1}, "encode_cased_HELLO_oov");
    }

    // ════════════════════════════════════════════════════════════
    // 7. Encoding - max_input_chars_per_word
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (max chars per word) ===\n";

        // Create tokenizer with max_input_chars_per_word = 5
        const char* short_max_json = R"({
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 5,
                "vocab": {
                    "[UNK]": 0, "hello": 1, "hi": 2
                }
            }
        })";
        auto tok = trtmc::CreateWordPieceTokenizer(short_max_json,
            std::strlen(short_max_json), false);

        // "hello" is exactly 5 chars — should work
        check_ids(tok->encode("hello"), {1}, "max_chars_exact");

        // "helloo" is 6 chars — exceeds limit → [UNK]
        check_ids(tok->encode("helloo"), {0}, "max_chars_exceeded");

        // "hi" is 2 chars — under limit
        check_ids(tok->encode("hi"), {2}, "max_chars_under");
    }

    // ════════════════════════════════════════════════════════════
    // 8. Decoding
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Decoding ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Basic decode
        check(tok->decode({5}) == "hello", "decode_single");
        check(tok->decode({5, 6}) == "hello world", "decode_two_words");

        // Subword decode: "un" + "##aff" + "##able" → "unaffable"
        check(tok->decode({7, 8, 9}) == "unaffable", "decode_subword");

        // Decode with special tokens → they are filtered out
        check(tok->decode({2, 5, 6, 3}) == "hello world", "decode_filter_special");

        // Empty decode
        check(tok->decode({}) == "", "decode_empty");

        // Decode only special tokens → empty
        check(tok->decode({2, 3}) == "", "decode_only_special");

        // Decode with [UNK]
        check(tok->decode({5, 1, 6}) == "hello [UNK] world", "decode_with_unk");
    }

    // ════════════════════════════════════════════════════════════
    // 9. Round-trip
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Round-trip ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Round-trip for in-vocab words (normalizer lowercases, so compare lowercase)
        auto rt = [&](const std::string& input, const std::string& expected) {
            auto decoded = tok->decode(tok->encode(input));
            check(decoded == expected,
                  "roundtrip_" + input + " → '" + decoded + "'");
        };

        rt("hello", "hello");
        rt("hello world", "hello world");
        rt("testing", "testing");
        rt("unaffable", "unaffable");
        rt("HELLO", "hello"); // lowercased
    }

    // ════════════════════════════════════════════════════════════
    // 10. BertNormalizer - control characters
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== BertNormalizer (control chars) ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Tab/newline replaced with space (then split as whitespace)
        check_ids(tok->encode("hello\tworld"), {5, 6}, "norm_tab");
        check_ids(tok->encode("hello\nworld"), {5, 6}, "norm_newline");

        // Control chars stripped (0x01 is removed, joining the words)
        std::string with_ctrl = "hello";
        with_ctrl.push_back('\x01');
        with_ctrl += "world";
        // "hello\x01world" → clean_text removes \x01 → "helloworld" (one word, OOV)
        check_ids(tok->encode(with_ctrl), {1}, "norm_ctrl_stripped_oov");

        // Control char with space separation still works
        std::string with_ctrl_space = "hello";
        with_ctrl_space.push_back('\x01');
        with_ctrl_space += " world";
        check_ids(tok->encode(with_ctrl_space), {5, 6}, "norm_ctrl_with_space");
    }

    // ════════════════════════════════════════════════════════════
    // 11. BertNormalizer - Chinese characters
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== BertNormalizer (Chinese chars) ===\n";

        // Tokenizer with Chinese chars in vocab
        const char* cjk_json = R"({
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "vocab": {
                    "[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
                    "\u4f60": 3, "\u597d": 4, "hello": 5
                }
            },
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": true,
                "handle_chinese_chars": true,
                "lowercase": true
            },
            "added_tokens": [
                {"id": 0, "content": "[UNK]", "special": true},
                {"id": 1, "content": "[CLS]", "special": true},
                {"id": 2, "content": "[SEP]", "special": true}
            ]
        })";
        auto tok = trtmc::CreateWordPieceTokenizer(cjk_json, std::strlen(cjk_json), false);

        // CJK chars get spaces around them, so each becomes its own word
        // "\u4f60\u597d" → " \u4f60 " + " \u597d " → ["你", "好"]
        check_ids(tok->encode("\xe4\xbd\xa0\xe5\xa5\xbd"), {3, 4}, "cjk_two_chars");

        // Mixed CJK and ASCII
        check_ids(tok->encode("hello\xe4\xbd\xa0"), {5, 3}, "cjk_mixed");
    }

    // ════════════════════════════════════════════════════════════
    // 12. BertPreTokenizer - punctuation splitting
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== BertPreTokenizer (punctuation) ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Punctuation becomes separate tokens
        check_ids(tok->encode("hello."), {5, 15}, "pretok_trailing_dot");
        check_ids(tok->encode("hello.world"), {5, 15, 6}, "pretok_dot_between");
        check_ids(tok->encode("!hello!"), {14, 5, 14}, "pretok_surrounding_punct");
        check_ids(tok->encode("hello?world!"), {5, 17, 6, 14}, "pretok_multi_punct");
    }

    // ════════════════════════════════════════════════════════════
    // 13. Edge cases
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Edge Cases ===\n";

        std::string json(kWordPieceJson);
        auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

        // Single character that is in vocab
        check_ids(tok->encode("a"), {18}, "edge_single_char");

        // Single punctuation
        check_ids(tok->encode("!"), {14}, "edge_single_punct");

        // Only whitespace
        check_ids(tok->encode("   "), {}, "edge_only_whitespace");

        // Leading/trailing whitespace
        check_ids(tok->encode("  hello  "), {5}, "edge_surrounding_ws");

        // Multiple words with suffixes
        check_ids(tok->encode("testing cats"), {22, 23, 28, 24}, "edge_multi_suffix");
    }

    // ════════════════════════════════════════════════════════════
    // 14. No normalizer
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== No Normalizer ===\n";

        const char* no_norm_json = R"({
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "vocab": {
                    "[UNK]": 0, "Hello": 1, "hello": 2
                }
            }
        })";
        auto tok = trtmc::CreateWordPieceTokenizer(no_norm_json,
            std::strlen(no_norm_json), false);

        // Without normalizer, case is preserved
        check_ids(tok->encode("Hello"), {1}, "no_norm_Hello");
        check_ids(tok->encode("hello"), {2}, "no_norm_hello");
        check_ids(tok->encode("HELLO"), {0}, "no_norm_HELLO_oov");
    }

    // ════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════
    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "\nAll WordPiece tokenizer tests passed!\n";
    return 0;
}
