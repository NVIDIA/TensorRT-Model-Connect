/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-TOK-CPP-01
// Architecture:   ARCH-TOK-001
// Unit Design:    UD-TOK-01
// Intent:         VocabTokenizer encode/decode round-trip, case insensitivity
// Preconditions:  Vocabulary list available
// Postconditions: encode(decode(ids)) == ids
// =============================================================================

// =============================================================================
// test_vocab_tokenizer.cpp — Unit tests for VocabTokenizer
// =============================================================================
//
// Purpose:
//   Validates the VocabTokenizer created via CreateVocabTokenizer(). Tests
//   cover encoding, decoding, id/token lookup, unknown-token fallback,
//   round-trip consistency, and edge cases (empty input, punctuation).
//
// Dependencies:
//   - trtmc/tokenizer.h (ITokenizer, CreateVocabTokenizer)
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

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// Standard test vocabulary:
//   0: <unk>
//   1: <bos>
//   2: <eos>
//   3: <pad>
//   4: hello
//   5: world
//   6: test
//   7: .
//   8: ,
//   9: !
static std::vector<std::string> make_test_vocab()
{
    return {"<unk>", "<bos>", "<eos>", "<pad>", "hello", "world", "test", ".", ",", "!"};
}

static void test_id_for_token()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    check(tok->id_for_token("hello") == 4, "id_for_token hello == 4");
    check(tok->id_for_token("world") == 5, "id_for_token world == 5");
    check(tok->id_for_token("test") == 6, "id_for_token test == 6");
    check(tok->id_for_token("<unk>") == 0, "id_for_token <unk> == 0");
    check(tok->id_for_token("<bos>") == 1, "id_for_token <bos> == 1");
    check(tok->id_for_token("<eos>") == 2, "id_for_token <eos> == 2");
}

static void test_token_for_id()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    check(tok->token_for_id(4) == "hello", "token_for_id 4 == hello");
    check(tok->token_for_id(5) == "world", "token_for_id 5 == world");
    check(tok->token_for_id(0) == "<unk>", "token_for_id 0 == <unk>");
}

static void test_token_for_id_out_of_range()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    // Out-of-range ids should return the unk token
    check(tok->token_for_id(999) == "<unk>", "token_for_id 999 == <unk>");
    check(tok->token_for_id(-1) == "<unk>", "token_for_id -1 == <unk>");
}

static void test_encode_basic()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    const auto ids = tok->encode("hello world");
    check(ids.size() == 2, "encode hello world size == 2");
    check(ids[0] == 4, "encode hello world [0] == 4");
    check(ids[1] == 5, "encode hello world [1] == 5");
}

static void test_encode_with_punctuation()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    const auto ids = tok->encode("hello world.");
    check(ids.size() == 3, "encode hello world. size == 3");
    check(ids[0] == 4, "encode hello world. [0] == 4");
    check(ids[1] == 5, "encode hello world. [1] == 5");
    check(ids[2] == 7, "encode hello world. [2] == 7 (.)");
}

static void test_encode_unknown_token()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    // "foobar" is not in vocab, should map to unk_id (0)
    const auto ids = tok->encode("foobar");
    check(ids.size() == 1, "encode foobar size == 1");
    check(ids[0] == 0, "encode foobar [0] == 0 (unk)");
}

static void test_encode_empty()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    const auto ids = tok->encode("");
    check(ids.empty(), "encode empty == empty");
}

static void test_decode_basic()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    const std::string text = tok->decode({4, 5});
    check(text == "hello world", "decode {4,5} == 'hello world'");
}

static void test_decode_skips_special()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    // <bos>=1, <eos>=2, <pad>=3 should be skipped in decode
    const std::string text = tok->decode({1, 4, 5, 2, 3});
    check(text == "hello world", "decode skips bos/eos/pad");
}

static void test_decode_punctuation_no_space()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    // Punctuation should not have a leading space
    const std::string text = tok->decode({4, 7});
    check(text == "hello.", "decode punctuation no space");
}

static void test_roundtrip()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    const std::string original = "hello world";
    const auto ids = tok->encode(original);
    const std::string decoded = tok->decode(ids);
    check(decoded == original, "roundtrip hello world");
}

static void test_case_insensitive()
{
    auto tok = trtmc::CreateVocabTokenizer(make_test_vocab());
    // VocabTokenizer normalizes to lowercase, so "Hello" should match "hello"
    const auto ids = tok->encode("Hello World");
    check(ids.size() == 2, "case insensitive size == 2");
    check(ids[0] == 4, "case insensitive Hello -> 4");
    check(ids[1] == 5, "case insensitive World -> 5");
}

static void test_empty_vocab_throws()
{
    bool threw = false;
    try
    {
        trtmc::CreateVocabTokenizer({});
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    check(threw, "empty vocab throws invalid_argument");
}

int main()
{
    test_id_for_token();
    test_token_for_id();
    test_token_for_id_out_of_range();
    test_encode_basic();
    test_encode_with_punctuation();
    test_encode_unknown_token();
    test_encode_empty();
    test_decode_basic();
    test_decode_skips_special();
    test_decode_punctuation_no_space();
    test_roundtrip();
    test_case_insensitive();
    test_empty_vocab_throws();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All vocab_tokenizer tests passed.\n";
    return 0;
}
