/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-TOK-EDGE-01
// Architecture:   ARCH-TOK-001
// Unit Design:    UD-TOK-EDGE-01
// Intent:         Tokenizer edge cases: malformed UTF-8, extreme inputs,
//                 out-of-range IDs. Covers BPE, WordPiece, and Unigram.
// Preconditions:  None (self-contained JSON test data)
// Postconditions: All tokenizers handle edge cases without crash or hang
// =============================================================================

#include "trtmc/tokenizer.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

// ─── Minimal BPE tokenizer JSON (GPT-2 style byte-level) ───
static const char* kBpeJson = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3, " ": 4,
      "he": 5, "ll": 6, "hel": 7, "lo": 8,
      "hello": 9,
      "w": 10, "r": 11, "d": 12,
      "or": 13, "ld": 14,
      "world": 15,
      ".": 16, "!": 17, "?": 18,
      "\u00e9": 19
    },
    "merges": [
      ["h", "e"],
      ["l", "l"],
      ["l", "o"],
      ["he", "l"],
      ["hel", "lo"],
      ["o", "r"],
      ["l", "d"],
      ["or", "ld"],
      ["w", "orld"]
    ]
  },
  "added_tokens": [
    {"id": 100, "content": "<eos>", "special": true}
  ]
})";

// ─── Minimal WordPiece tokenizer JSON ───
static const char* kWordPieceJson = R"({
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
      "hello": 4, "world": 5, ".": 6, "!": 7
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

// ─── Minimal Unigram tokenizer JSON ───
static const char* kUnigramJson = R"({
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["h", -3.0],
      ["e", -3.0],
      ["l", -3.0],
      ["o", -3.0]
    ]
  },
  "pre_tokenizer": {
    "type": "Metaspace",
    "replacement": "\u2581",
    "add_prefix_space": true
  }
})";

static const char* kUnigramLowercaseSequenceJson = R"({
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581the", -1.0]
    ]
  },
  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      {"type": "Precompiled", "precompiled_charsmap": "unused"},
      {"type": "Lowercase"}
    ]
  },
  "pre_tokenizer": {
    "type": "Metaspace",
    "replacement": "\u2581",
    "add_prefix_space": true
  }
})";

// ─── Helper: build a string with specific bytes ───
static std::string bytes(std::initializer_list<unsigned char> bs) {
    std::string s;
    for (auto b : bs)
        s.push_back(static_cast<char>(b));
    return s;
}

// ═══════════════════════════════════════════════════════════
// Malformed UTF-8 tests
// ═══════════════════════════════════════════════════════════

static void test_bpe_malformed_utf8() {
    std::string json(kBpeJson);
    auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

    // Truncated 2-byte sequence: 0xC3 without continuation
    {
        std::string input = bytes({0xC3});
        auto ids = tok->encode(input);
        // Must not crash. Result may vary but must be non-empty or empty.
        (void)ids;
        check(true, "bpe_truncated_2byte_no_crash");
    }

    // Truncated 3-byte sequence: 0xE2 0x96 without third byte
    {
        std::string input = bytes({0xE2, 0x96});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_truncated_3byte_no_crash");
    }

    // Truncated 4-byte sequence: 0xF0 0x9F without 3rd and 4th
    {
        std::string input = bytes({0xF0, 0x9F});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_truncated_4byte_no_crash");
    }

    // Invalid continuation byte (0x80-0xBF as lead byte)
    {
        std::string input = bytes({0x80});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_invalid_continuation_lead_no_crash");
    }

    // 0xFF and 0xFE are never valid in UTF-8
    {
        std::string input = bytes({0xFF, 0xFE});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_0xff_0xfe_no_crash");
    }

    // Overlong 2-byte encoding of ASCII (0xC0 0x80 = U+0000)
    {
        std::string input = bytes({0xC0, 0x80});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_overlong_2byte_no_crash");
    }

    // Mixed valid and invalid: "hello" + 0xFF + "world"
    {
        std::string input = "hello";
        input.push_back(static_cast<char>(0xFF));
        input += "world";
        auto ids = tok->encode(input);
        check(!ids.empty(), "bpe_mixed_valid_invalid_produces_tokens");
    }

    // All-invalid bytes
    {
        std::string input = bytes({0x80, 0x81, 0xFE, 0xFF, 0xC0, 0xC1});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_all_invalid_bytes_no_crash");
    }

    // Decode round-trip with malformed input should not crash
    {
        std::string input = bytes({0xC3}); // truncated
        auto ids = tok->encode(input);
        auto decoded = tok->decode(ids);
        (void)decoded;
        check(true, "bpe_malformed_roundtrip_no_crash");
    }
}

static void test_wordpiece_malformed_utf8() {
    std::string json(kWordPieceJson);
    auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

    // Truncated 2-byte sequence
    {
        std::string input = bytes({0xC3});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "wp_truncated_2byte_no_crash");
    }

    // Invalid continuation byte as lead
    {
        std::string input = bytes({0x80, 0x81, 0x82});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "wp_invalid_continuation_no_crash");
    }

    // 0xFF byte
    {
        std::string input = bytes({0xFF});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "wp_0xff_no_crash");
    }

    // Mixed valid + malformed
    {
        std::string input = "hello";
        input.push_back(static_cast<char>(0xFF));
        auto ids = tok->encode(input);
        // Should produce at least the "hello" token or UNK
        check(!ids.empty(), "wp_mixed_valid_invalid_produces_tokens");
    }

    // Decode round-trip
    {
        std::string input = bytes({0xC3});
        auto ids = tok->encode(input);
        auto decoded = tok->decode(ids);
        (void)decoded;
        check(true, "wp_malformed_roundtrip_no_crash");
    }
}

static void test_unigram_malformed_utf8() {
    std::string json(kUnigramJson);
    auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

    // Truncated multi-byte
    {
        std::string input = bytes({0xC3});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "uni_truncated_2byte_no_crash");
    }

    // Invalid byte
    {
        std::string input = bytes({0xFF});
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "uni_0xff_no_crash");
    }

    // Mixed valid + malformed
    {
        std::string input = "hello";
        input.push_back(static_cast<char>(0xFF));
        auto ids = tok->encode(input);
        check(!ids.empty(), "uni_mixed_valid_invalid_produces_tokens");
    }
}

// ═══════════════════════════════════════════════════════════
// Out-of-range token ID tests
// ═══════════════════════════════════════════════════════════

static void test_bpe_out_of_range_ids() {
    std::string json(kBpeJson);
    auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

    // Negative ID
    check(tok->token_for_id(-1).empty(), "bpe_token_for_id_negative");
    check(tok->token_for_id(-1000).empty(), "bpe_token_for_id_very_negative");

    // Very large ID
    check(tok->token_for_id(999999).empty(), "bpe_token_for_id_huge");

    // INT32_MIN / INT32_MAX
    check(tok->token_for_id(INT32_MIN).empty(), "bpe_token_for_id_int32_min");
    check(tok->token_for_id(INT32_MAX).empty(), "bpe_token_for_id_int32_max");

    // Decode with out-of-range IDs should not crash
    {
        auto text = tok->decode({-1, 999999, INT32_MIN});
        (void)text;
        check(true, "bpe_decode_oob_ids_no_crash");
    }
}

static void test_wordpiece_out_of_range_ids() {
    std::string json(kWordPieceJson);
    auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

    check(tok->token_for_id(INT32_MIN).empty(), "wp_token_for_id_int32_min");
    check(tok->token_for_id(INT32_MAX).empty(), "wp_token_for_id_int32_max");

    // Decode with OOB IDs
    {
        auto text = tok->decode({-1, 999999, INT32_MIN});
        (void)text;
        check(true, "wp_decode_oob_ids_no_crash");
    }
}

static void test_unigram_out_of_range_ids() {
    std::string json(kUnigramJson);
    auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

    check(tok->token_for_id(INT32_MIN).empty(), "uni_token_for_id_int32_min");
    check(tok->token_for_id(INT32_MAX).empty(), "uni_token_for_id_int32_max");

    // Decode with OOB IDs
    {
        auto text = tok->decode({-1, 999999, INT32_MIN});
        (void)text;
        check(true, "uni_decode_oob_ids_no_crash");
    }
}

// ═══════════════════════════════════════════════════════════
// Extreme input tests
// ═══════════════════════════════════════════════════════════

static void test_bpe_extreme_inputs() {
    std::string json(kBpeJson);
    auto tok = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);

    // Very long string (10K characters)
    {
        std::string long_input(10000, 'h');
        auto ids = tok->encode(long_input);
        check(!ids.empty(), "bpe_10k_chars_produces_tokens");
        auto decoded = tok->decode(ids);
        (void)decoded;
        check(true, "bpe_10k_chars_roundtrip_no_crash");
    }

    // Single null byte (not NUL-terminated issue — std::string handles it)
    {
        std::string input(1, '\0');
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_null_byte_no_crash");
    }

    // String of only spaces
    {
        std::string input(100, ' ');
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_only_spaces_no_crash");
    }

    // String of only newlines
    {
        std::string input(100, '\n');
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "bpe_only_newlines_no_crash");
    }

    // Large decode with many IDs
    {
        std::vector<int32_t> ids(1000, 0); // 1000 copies of token 0
        auto decoded = tok->decode(ids);
        check(!decoded.empty(), "bpe_large_decode_produces_text");
    }
}

static void test_wordpiece_extreme_inputs() {
    std::string json(kWordPieceJson);
    auto tok = trtmc::CreateWordPieceTokenizer(json.data(), json.size(), false);

    // Very long word (exceeds max_input_chars_per_word=100) → UNK
    {
        std::string long_word(200, 'a');
        auto ids = tok->encode(long_word);
        check(!ids.empty(), "wp_200_char_word_produces_tokens");
        // Should be [UNK] since word exceeds max_input_chars_per_word
        check(ids[0] == 1, "wp_200_char_word_is_unk");
    }

    // String of only spaces
    {
        std::string input(100, ' ');
        auto ids = tok->encode(input);
        check(ids.empty(), "wp_only_spaces_empty");
    }

    // Null byte
    {
        std::string input(1, '\0');
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "wp_null_byte_no_crash");
    }
}

static void test_unigram_extreme_inputs() {
    std::string json(kUnigramJson);
    auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

    // Very long string
    {
        std::string long_input(10000, 'h');
        auto ids = tok->encode(long_input);
        check(!ids.empty(), "uni_10k_chars_produces_tokens");
    }

    // Null byte
    {
        std::string input(1, '\0');
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "uni_null_byte_no_crash");
    }

    // String of only spaces
    {
        std::string input(100, ' ');
        auto ids = tok->encode(input);
        (void)ids;
        check(true, "uni_only_spaces_no_crash");
    }
}

// ═══════════════════════════════════════════════════════════
// Valid multi-byte UTF-8 (regression: ensure these still work)
// ═══════════════════════════════════════════════════════════

static void test_valid_multibyte_utf8() {
    std::string bpe_json(kBpeJson);
    auto bpe = trtmc::CreateBpeTokenizer(bpe_json.data(), bpe_json.size(), false);

    // 2-byte: é (U+00E9) = 0xC3 0xA9
    {
        std::string input = "\xc3\xa9";
        auto ids = bpe->encode(input);
        (void)ids;
        check(true, "bpe_valid_2byte_utf8_no_crash");
    }

    // 3-byte: € (U+20AC) = 0xE2 0x82 0xAC
    {
        std::string input = "\xe2\x82\xac";
        auto ids = bpe->encode(input);
        (void)ids;
        check(true, "bpe_valid_3byte_utf8_no_crash");
    }

    // 4-byte: 😀 (U+1F600) = 0xF0 0x9F 0x98 0x80
    {
        std::string input = "\xf0\x9f\x98\x80";
        auto ids = bpe->encode(input);
        (void)ids;
        check(true, "bpe_valid_4byte_utf8_no_crash");
    }
}

static void test_unigram_sequence_lowercase_normalizer() {
    std::string json(kUnigramLowercaseSequenceJson);
    auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

    auto ids = tok->encode("The");
    check(ids.size() == 1 && ids[0] == 1, "uni_sequence_lowercase_normalizer");
}

int main() {
    // Malformed UTF-8
    test_bpe_malformed_utf8();
    test_wordpiece_malformed_utf8();
    test_unigram_malformed_utf8();

    // Out-of-range token IDs
    test_bpe_out_of_range_ids();
    test_wordpiece_out_of_range_ids();
    test_unigram_out_of_range_ids();

    // Extreme inputs
    test_bpe_extreme_inputs();
    test_wordpiece_extreme_inputs();
    test_unigram_extreme_inputs();

    // Valid multi-byte UTF-8 (regression)
    test_valid_multibyte_utf8();

    // Normalizer sequences
    test_unigram_sequence_lowercase_normalizer();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All tokenizer edge case tests passed.\n";
    return 0;
}
