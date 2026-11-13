/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESM's tokenizer is not a fast one, so there is no tokenizer.json to convert
 * and the rule has to be reproduced here. The reference registers every
 * vocabulary entry as a no-split token and runs its trie over the text before
 * anything else; only the leftovers reach its text.split().
 *
 * The visible consequences, and what this checks:
 *   - a residue string is split per character, not on whitespace;
 *   - "<mask>" is one token, not six;
 *   - whitespace disappears rather than becoming a token;
 *   - a character outside the alphabet becomes <unk>.
 */

#include "families/esm/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

// The real ESM-2 vocabulary, in the published order.
const char* kVocab = "<cls>\n<pad>\n<eos>\n<unk>\nL\nA\nG\nV\nS\nE\nR\nT\nI\nD\nP\nK\nQ\nN\nF\n"
                     "Y\nM\nH\nW\nC\nX\nB\nU\nZ\nO\n.\n-\n<null_1>\n<mask>\n";

void check(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected,
           const char* name) {
    if (actual == expected)
        return;
    std::cerr << "FAIL: " << name << " got";
    for (const auto id : actual)
        std::cerr << ' ' << id;
    std::cerr << "\n           expected";
    for (const auto id : expected)
        std::cerr << ' ' << id;
    std::cerr << '\n';
    ++failures;
}

void expect(bool condition, const char* name) {
    if (condition)
        return;
    std::cerr << "FAIL: " << name << '\n';
    ++failures;
}

} // namespace

int main() {
    auto tokenizer = trtmc::CreateEsmTokenizer(kVocab, std::char_traits<char>::length(kVocab));
    if (!tokenizer) {
        std::cerr << "FAIL: tokenizer did not load\n";
        return 1;
    }

    expect(tokenizer->size() == 33, "the vocabulary has 33 entries");
    expect(tokenizer->id_for_token("<cls>") == 0, "<cls> is 0");
    expect(tokenizer->id_for_token("<pad>") == 1, "<pad> is 1");
    expect(tokenizer->id_for_token("<eos>") == 2, "<eos> is 2");
    expect(tokenizer->id_for_token("<mask>") == 32, "<mask> is 32");
    expect(tokenizer->token_for_id(20) == "M", "id 20 is M");

    // One id per residue. A whitespace split would return a single <unk>.
    check(tokenizer->encode("MKTVRQERLK"), {20, 15, 11, 7, 10, 16, 9, 10, 4, 15},
          "a residue run is split per character");

    // The longest match wins, so the six characters of "<mask>" are one id.
    check(tokenizer->encode("MK<mask>TV"), {20, 15, 32, 11, 7}, "<mask> is matched whole");

    // Whitespace is dropped, not tokenized.
    check(tokenizer->encode("M K\tT\nV"), {20, 15, 11, 7}, "whitespace disappears");

    // 'J' is not in the alphabet; 'Z' is.
    check(tokenizer->encode("MJZ"), {20, 3, 27}, "an unknown residue becomes <unk>");

    // A run of unknown characters collapses to one <unk>, which is what the
    // reference's split() plus vocabulary miss produces for one whitespace
    // free run.
    check(tokenizer->encode("MJJJK"), {20, 3, 15}, "an unknown run is one <unk>");

    check(tokenizer->encode(""), {}, "an empty string encodes to nothing");
    check(tokenizer->encode("   "), {}, "whitespace alone encodes to nothing");

    if (failures == 0)
        std::cout << "tokenizer: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
