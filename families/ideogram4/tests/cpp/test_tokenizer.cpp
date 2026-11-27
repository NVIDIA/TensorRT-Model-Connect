/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// CPU checks of the Ideogram4 byte-level BPE on a small tokenizer.json that uses the
// sequential split expressions, added-token flags, ignore_merges and decoder.

#include "families/ideogram4/runtime/tokenizer.h"
#include "tokenizer_fixture.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::ideogram4::test::byte_units;
using trtmc::ideogram4::test::Fixture;
using trtmc::ideogram4::test::tokenizer_json;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAILED: " << label << '\n';
        ++failures;
    }
}

std::vector<std::int32_t> ids(Fixture& fixture, const std::vector<std::string>& tokens) {
    std::vector<std::int32_t> result;
    for (const auto& token : tokens)
        result.push_back(fixture.vocab.at(token).get<std::int32_t>());
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        std::ifstream file(argv[1]);
        const std::string serialized((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
        const trtmc::ideogram4::Tokenizer real(serialized);
        nlohmann::json request;
        std::cin >> request;
        nlohmann::json result = nlohmann::json::array();
        for (const auto& text : request) {
            const auto ids = real.encode(text.get<std::string>());
            result.push_back({{"ids", ids}, {"text", real.decode(ids)}});
        }
        std::cout << result.dump() << '\n';
        return 0;
    }
    Fixture fixture;
    const trtmc::ideogram4::Tokenizer tokenizer(tokenizer_json(fixture));
    const auto space = byte_units()[' '];

    expect(tokenizer.encode("caf\x65\xcc\x81") == tokenizer.encode("caf\xc3\xa9"),
           "NFC joins a decomposed acute accent before byte-level BPE");
    expect(tokenizer.encode("hello") == ids(fixture, {"hello"}), "merges build one word");
    expect(tokenizer.encode("World") == ids(fixture, {"World"}),
           "ignore_merges keeps a vocab word");
    expect(tokenizer.encode("helloWorld") == ids(fixture, {"hello", "World"}),
           "the split expression separates a lower-case run from a capitalized word");
    expect(tokenizer.encode("123") == ids(fixture, {"1", "2", "3"}), "digits are split singly");
    expect(tokenizer.encode(" world") == ids(fixture, {space + "w", "o", "r", "l", "d"}),
           "a leading space joins the word and merges by rank");
    expect(tokenizer.encode("<s>hello</s>") ==
               std::vector<std::int32_t>{1, fixture.vocab.at("hello").get<std::int32_t>(), 2},
           "special tokens inside the text are matched before splitting");
    expect(tokenizer.encode("a\\begin{b") ==
               std::vector<std::int32_t>{fixture.vocab.at("a").get<std::int32_t>(), 3,
                                         fixture.vocab.at("b").get<std::int32_t>()},
           "non-special added tokens are matched");
    expect(tokenizer.decode({1, fixture.vocab.at("hello").get<std::int32_t>(), 2}) == "hello",
           "decode skips special tokens");
    expect(tokenizer.decode({3}) == "\\begin{", "decode keeps non-special added tokens");
    const std::string multilingual = "Gr\xC3\xBC"
                                     "ezi mitenand,\n\tZ\xC3\xBCrich 2025! "
                                     "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E camelCaseHTTPServer";
    expect(tokenizer.decode(tokenizer.encode(multilingual)) == multilingual,
           "UTF-8 text survives an encode/decode round trip");
    expect(tokenizer.vocab_size() == fixture.next_id, "vocabulary size counts every ID");
    try {
        (void)tokenizer.decode({fixture.next_id});
        expect(false, "an unknown token ID is rejected");
    } catch (const std::invalid_argument&) {
    }
    if (failures == 0)
        std::cout << "test_ideogram4_tokenizer: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
