/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-TOK-CPP-03
// Architecture:   ARCH-TOK-001
// Unit Design:    UD-TOK-01
// Intent:         IPA tokenizer: encode known words, heteronyms as graphemes, OOV fallback,
// punctuation Preconditions:  Synthetic mini-data mimicking NeMo IPA format Postconditions: Known
// words tokenized via IPA, OOV/heteronyms fall back to graphemes, punctuation handled
// =============================================================================

// =============================================================================
// test_ipa_tokenizer.cpp — Unit tests for IPA tokenizer (CreateIpaTokenizer)
// =============================================================================
//
// Purpose:
//   Validates the native C++ IPA tokenizer. Tests use synthetic mini-data
//   mimicking a character-level IPA format: each pronunciation
//   is a string of IPA characters (each character = a token), graphemes are
//   uppercase letters (no prefix by default).
//
// Dependencies:
//   - trtmc/tokenizer.h (ITokenizer, CreateIpaTokenizer)
//
// Environment:
//   CPU-only. No GPU, CUDA, TRT, or filesystem access required.
// =============================================================================

#include "trtmc/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// Synthetic test data (mimics NeMo character-level IPA format)
// ---------------------------------------------------------------------------

// Vocabulary (line index = token ID):
//   0: !   (punctuation)
//   1: ,   (punctuation)
//   2: .   (punctuation)
//   3: A   (grapheme)
//   4: B   (grapheme)
//   5: C   (grapheme)
//   6: D   (grapheme)
//   7: E   (grapheme)
//   8: H   (grapheme)
//   9: L   (grapheme)
//  10: O   (grapheme)
//  11: R   (grapheme)
//  12: S   (grapheme)
//  13: T   (grapheme)
//  14: X   (grapheme)
//  15: a   (IPA phoneme char)
//  16: d   (IPA phoneme char)
//  17: e   (IPA phoneme char)
//  18: h   (IPA phoneme char)
//  19: k   (IPA phoneme char)
//  20: l   (IPA phoneme char)
//  21: o   (IPA phoneme char)
//  22: t   (IPA phoneme char)
//  23: z   (IPA phoneme char)
//  24:     (space)
//  25: <pad>
//  26: <oov>
static const char* make_test_vocab() {
    return "!\n"
           ",\n"
           ".\n"
           "A\n"
           "B\n"
           "C\n"
           "D\n"
           "E\n"
           "H\n"
           "L\n"
           "O\n"
           "R\n"
           "S\n"
           "T\n"
           "X\n"
           "a\n"
           "d\n"
           "e\n"
           "h\n"
           "k\n"
           "l\n"
           "o\n"
           "t\n"
           "z\n"
           " \n"
           "<pad>\n"
           "<oov>\n";
}

// Phoneme dict (TSV): word<TAB>ipa_pronunciation_string
// Each pronunciation is a sequence of characters (not space-separated).
// "hello" → "helo" (simplified IPA: h, e, l, o)
// "the" → "de"
// "cat" → "kat"
// "dog" has two pronunciations (ambiguous)
static const char* make_test_dict() {
    return "hello\thelo\n"
           "the\tde\n"
           "cat\tkat\n"
           "dog\tkat\n"
           "dog\tde\n"; // second pronunciation makes "dog" ambiguous
}

// Heteronyms: "read" is a heteronym
static const char* make_test_heteronyms() {
    return "read\n"
           "live\n";
}

// Config JSON: no grapheme prefix (NeMo default), EOS = 2379 (beyond vocab)
static const char* make_test_config() {
    return "{\"grapheme_prefix\": \"\", \"eos_id\": 99, \"ignore_ambiguous_words\": 0}";
}

static std::unique_ptr<trtmc::ITokenizer> make_test_tokenizer() {
    const auto* vocab = make_test_vocab();
    const auto* dict = make_test_dict();
    const auto* het = make_test_heteronyms();
    const auto* cfg = make_test_config();

    return trtmc::CreateIpaTokenizer(dict, std::string(dict).size(), het, std::string(het).size(),
                                     vocab, std::string(vocab).size(), cfg,
                                     std::string(cfg).size());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_encode_known_word() {
    auto tok = make_test_tokenizer();
    // "hello" → pronunciation "helo" → chars: h(18), e(17), l(20), o(21) + EOS(99)
    const auto ids = tok->encode("hello");
    check(ids.size() == 5, "encode hello size == 5");
    check(ids[0] == 18, "encode hello [0] == 18 (h)");
    check(ids[1] == 17, "encode hello [1] == 17 (e)");
    check(ids[2] == 20, "encode hello [2] == 20 (l)");
    check(ids[3] == 21, "encode hello [3] == 21 (o)");
    check(ids[4] == 99, "encode hello [4] == 99 (EOS)");
}

static void test_encode_heteronym_as_graphemes() {
    auto tok = make_test_tokenizer();
    // "read" is a heteronym → uppercase graphemes: R(11), E(7), A(3), D(6) + EOS
    const auto ids = tok->encode("read");
    check(ids.size() == 5, "encode read size == 5");
    check(ids[0] == 11, "encode read [0] == 11 (R)");
    check(ids[1] == 7, "encode read [1] == 7 (E)");
    check(ids[2] == 3, "encode read [2] == 3 (A)");
    check(ids[3] == 6, "encode read [3] == 6 (D)");
    check(ids[4] == 99, "encode read [4] == 99 (EOS)");
}

static void test_encode_oov_as_graphemes() {
    auto tok = make_test_tokenizer();
    // "box" is OOV → uppercase graphemes: B(4), O(10), X(14) + EOS
    const auto ids = tok->encode("box");
    check(ids.size() == 4, "encode box size == 4");
    check(ids[0] == 4, "encode box [0] == 4 (B)");
    check(ids[1] == 10, "encode box [1] == 10 (O)");
    check(ids[2] == 14, "encode box [2] == 14 (X)");
    check(ids[3] == 99, "encode box [3] == 99 (EOS)");
}

static void test_encode_with_punctuation() {
    auto tok = make_test_tokenizer();
    // "hello." → h(18) e(17) l(20) o(21) .(2) + EOS(99)
    const auto ids = tok->encode("hello.");
    check(ids.size() == 6, "encode hello. size == 6");
    check(ids[4] == 2, "encode hello. [4] == 2 (.)");
    check(ids[5] == 99, "encode hello. [5] == 99 (EOS)");
}

static void test_encode_multiple_words() {
    auto tok = make_test_tokenizer();
    // "the cat" → d(16) e(17) <space>(24) k(19) a(15) t(22) + EOS
    const auto ids = tok->encode("the cat");
    check(ids.size() == 7, "encode the cat size == 7");
    check(ids[0] == 16, "encode the cat [0] == 16 (d)");
    check(ids[1] == 17, "encode the cat [1] == 17 (e)");
    check(ids[2] == 24, "encode the cat [2] == 24 (space)");
    check(ids[3] == 19, "encode the cat [3] == 19 (k)");
    check(ids[4] == 15, "encode the cat [4] == 15 (a)");
    check(ids[5] == 22, "encode the cat [5] == 22 (t)");
    check(ids[6] == 99, "encode the cat [6] == 99 (EOS)");
}

static void test_encode_case_insensitive() {
    auto tok = make_test_tokenizer();
    // "Hello" should match "hello" in dict (lowercased lookup)
    const auto ids = tok->encode("Hello");
    check(ids.size() == 5, "encode Hello case-insensitive size == 5");
    check(ids[0] == 18, "encode Hello [0] == 18 (h)");
}

static void test_encode_empty() {
    auto tok = make_test_tokenizer();
    const auto ids = tok->encode("");
    check(ids.size() == 1, "encode empty size == 1");
    check(ids[0] == 99, "encode empty [0] == 99 (EOS)");
}

static void test_encode_consecutive_spaces_dedup() {
    auto tok = make_test_tokenizer();
    const auto ids = tok->encode("the  cat");
    int space_count = 0;
    for (const auto id : ids) {
        if (id == 24)
            ++space_count;
    }
    check(space_count == 1, "encode double space dedup: only 1 space token");
}

static void test_encode_ambiguous_first_pronunciation() {
    // With ignore_ambiguous=false (our test config), use first pronunciation
    auto tok = make_test_tokenizer();
    // "dog" has 2 pronunciations: "kat" and "de". First = "kat"
    const auto ids = tok->encode("dog");
    check(ids.size() == 4, "encode dog (ambiguous, use first) size == 4");
    check(ids[0] == 19, "encode dog [0] == 19 (k)");
    check(ids[1] == 15, "encode dog [1] == 15 (a)");
    check(ids[2] == 22, "encode dog [2] == 22 (t)");
    check(ids[3] == 99, "encode dog [3] == 99 (EOS)");
}

static void test_decode_basic() {
    auto tok = make_test_tokenizer();
    const std::string text = tok->decode({18, 17, 20, 21});
    check(text == "helo", "decode helo");
}

static void test_decode_stops_at_eos() {
    auto tok = make_test_tokenizer();
    const std::string text = tok->decode({18, 17, 99, 20, 21});
    check(text == "he", "decode stops at EOS");
}

static void test_id_for_token() {
    auto tok = make_test_tokenizer();
    check(tok->id_for_token("h") == 18, "id_for_token h == 18");
    check(tok->id_for_token("<pad>") == 25, "id_for_token <pad> == 25");
    check(tok->id_for_token(".") == 2, "id_for_token . == 2");
    check(tok->id_for_token("nonexistent") == -1, "id_for_token nonexistent == -1");
}

static void test_token_for_id() {
    auto tok = make_test_tokenizer();
    check(tok->token_for_id(0) == "!", "token_for_id 0 == !");
    check(tok->token_for_id(25) == "<pad>", "token_for_id 25 == <pad>");
    check(tok->token_for_id(18) == "h", "token_for_id 18 == h");
    check(tok->token_for_id(999) == "", "token_for_id 999 == empty");
    check(tok->token_for_id(-1) == "", "token_for_id -1 == empty");
}

static void test_empty_dict_throws() {
    bool threw = false;
    try {
        const auto* vocab = make_test_vocab();
        trtmc::CreateIpaTokenizer(nullptr, 0, nullptr, 0, vocab, std::string(vocab).size(), nullptr,
                                  0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty dict throws invalid_argument");
}

static void test_empty_vocab_throws() {
    bool threw = false;
    try {
        const auto* dict = make_test_dict();
        trtmc::CreateIpaTokenizer(dict, std::string(dict).size(), nullptr, 0, nullptr, 0, nullptr,
                                  0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty vocab throws invalid_argument");
}

static void test_trailing_space_stripped() {
    auto tok = make_test_tokenizer();
    const auto ids = tok->encode("hello ");
    check(ids.back() == 99, "encode trailing space: last is EOS");
    if (ids.size() >= 2) {
        check(ids[ids.size() - 2] != 24, "encode trailing space: no space before EOS");
    }
}

int main() {
    test_encode_known_word();
    test_encode_heteronym_as_graphemes();
    test_encode_oov_as_graphemes();
    test_encode_with_punctuation();
    test_encode_multiple_words();
    test_encode_case_insensitive();
    test_encode_empty();
    test_encode_consecutive_spaces_dedup();
    test_encode_ambiguous_first_pronunciation();
    test_decode_basic();
    test_decode_stops_at_eos();
    test_id_for_token();
    test_token_for_id();
    test_empty_dict_throws();
    test_empty_vocab_throws();
    test_trailing_space_stripped();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All ipa_tokenizer tests passed.\n";
    return 0;
}
