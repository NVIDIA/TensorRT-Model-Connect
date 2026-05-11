// Unit tests for SentencePiece Unigram tokenizer.
//
// Tests: JSON parsing, Metaspace pre-tokenizer, Viterbi encoding,
// decoding, token lookup, special tokens, edge cases.
//
// Trace: ARCH-TOK-UNI, UD-TOK-UNI-01
// Intent: Verify Unigram tokenizer correctness against known expected values.
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
    failures++;
}

// ─── Minimal Unigram tokenizer JSON ───
// Vocab is a list of [token, score] pairs.
// Metaspace: spaces become ▁ (U+2581)
static const char* kUnigramJson = R"({
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581", -1.0],
      ["\u2581hello", -5.0],
      ["\u2581world", -5.0],
      ["\u2581the", -4.0],
      ["h", -3.0],
      ["e", -3.0],
      ["l", -3.0],
      ["o", -3.0],
      ["\u2581", -2.0],
      ["!", -2.5],
      [",", -2.5],
      [".", -2.5],
      ["\u2581a", -3.5],
      ["\u2581test", -5.5],
      ["ing", -4.0],
      ["\u2581cat", -5.0],
      ["\u2581dog", -5.0],
      ["s", -3.0]
    ]
  },
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Metaspace",
    "replacement": "\u2581",
    "add_prefix_space": true
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ]
  },
  "added_tokens": [
    {"id": 19, "content": "<s>", "special": true},
    {"id": 20, "content": "</s>", "special": true},
    {"id": 21, "content": "<pad>", "special": true}
  ]
})";

// ─── No-special-tokens tokenizer ───
static const char* kUnigramNoSpecialJson = R"({
  "model": {
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

// ─── Lowercase normalizer tokenizer ───
static const char* kUnigramLowercaseJson = R"({
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "vocab": [
      ["<unk>", 0.0],
      ["\u2581the", -4.0],
      ["\u2581cat", -4.0]
    ]
  },
  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      {"type": "Lowercase"}
    ]
  },
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "WhitespaceSplit"},
      {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true}
    ]
  }
})";

int main()
{
    std::cerr << "Unigram Tokenizer Unit Tests\n\n";

    // ════════════════════════════════════════════════════════════
    // 1. Factory / JSON parsing
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "=== Factory & JSON Parsing ===\n";

        std::string json(kUnigramJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);
        check(tok != nullptr, "create_from_valid_json");

        // Invalid JSON
        bool threw = false;
        try { trtmc::CreateUnigramTokenizer("bad", 3, false); }
        catch (...) { threw = true; }
        check(threw, "reject_invalid_json");

        // BPE type rejected
        const char* bpe = R"({"model":{"type":"BPE","vocab":{},"merges":[]}})";
        threw = false;
        try { trtmc::CreateUnigramTokenizer(bpe, std::strlen(bpe), false); }
        catch (...) { threw = true; }
        check(threw, "reject_bpe_type");

        // WordPiece type rejected
        const char* wp = R"({"model":{"type":"WordPiece","vocab":{},"continuing_subword_prefix":"##"}})";
        threw = false;
        try { trtmc::CreateUnigramTokenizer(wp, std::strlen(wp), false); }
        catch (...) { threw = true; }
        check(threw, "reject_wordpiece_type");
    }

    // ════════════════════════════════════════════════════════════
    // 2. Token/ID lookup
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Token/ID Lookup ===\n";

        std::string json(kUnigramJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

        check(tok->id_for_token("<unk>") == 0, "id_for_unk");
        check(tok->id_for_token("\xe2\x96\x81hello") == 2, "id_for_hello");
        check(tok->id_for_token("nonexistent") == -1, "id_for_missing");

        check(tok->token_for_id(0) == "<unk>", "token_for_0");
        check(tok->token_for_id(2) == "\xe2\x96\x81hello", "token_for_2");
        check(tok->token_for_id(-1) == "", "token_for_negative");
        check(tok->token_for_id(99999) == "", "token_for_oob");
    }

    // ════════════════════════════════════════════════════════════
    // 3. Encoding (no special tokens)
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (no special) ===\n";

        std::string json(kUnigramJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

        // "hello" → metaspace "▁hello" → token id 2
        check_ids(tok->encode("hello"), {2}, "encode_hello");

        // "hello world" → "▁hello▁world" → [2, 3]
        check_ids(tok->encode("hello world"), {2, 3}, "encode_hello_world");

        // Empty
        check_ids(tok->encode(""), {}, "encode_empty");
    }

    // ════════════════════════════════════════════════════════════
    // 4. Normalization
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Normalization ===\n";

        std::string json(kUnigramLowercaseJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);
        check_ids(tok->encode("The Cat"), {1, 2}, "encode_lowercase_normalizer");
    }

    // ════════════════════════════════════════════════════════════
    // 5. Encoding (with special tokens)
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Encoding (with special) ===\n";

        std::string json(kUnigramJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), true);

        // With special: <s> + tokens + </s>
        check_ids(tok->encode("hello"), {19, 2, 20}, "encode_hello_special");
        check_ids(tok->encode(""), {19, 20}, "encode_empty_special");
    }

    // ════════════════════════════════════════════════════════════
    // 6. Decoding
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Decoding ===\n";

        std::string json(kUnigramJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

        // Decode single token
        check(tok->decode({2}) == "hello", "decode_hello");

        // Decode multiple tokens
        check(tok->decode({2, 3}) == "hello world", "decode_hello_world");

        // Decode empty
        check(tok->decode({}) == "", "decode_empty");

        // Decode with special tokens — they should be filtered
        check(tok->decode({19, 2, 3, 20}) == "hello world", "decode_filter_special");
    }

    // ════════════════════════════════════════════════════════════
    // 7. Round-trip
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Round-trip ===\n";

        std::string json(kUnigramJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);

        auto rt = [&](const std::string& input) {
            auto decoded = tok->decode(tok->encode(input));
            check(decoded == input,
                  "roundtrip '" + input + "' -> '" + decoded + "'");
        };

        rt("hello");
        rt("hello world");
        rt("the cat");
    }

    // ════════════════════════════════════════════════════════════
    // 7. Auto-detect (no explicit type)
    // ════════════════════════════════════════════════════════════
    {
        std::cerr << "\n=== Auto-detect ===\n";

        std::string json(kUnigramNoSpecialJson);
        auto tok = trtmc::CreateUnigramTokenizer(json.data(), json.size(), false);
        check(tok != nullptr, "autodetect_unigram");
        check_ids(tok->encode("hello"), {1}, "autodetect_encode");
    }

    // ════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════
    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "\nAll Unigram tokenizer tests passed!\n";
    return 0;
}
