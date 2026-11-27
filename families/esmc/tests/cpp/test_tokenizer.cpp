/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ESMC's tokenizer.json is a BPE model with no merges and no pre-tokenizer,
 * plus six added tokens. The three rules that survive into ids, and what this
 * checks:
 *   - an added token matches whole and longest first, so "<mask>" is one id;
 *   - every other character is its own token, so a residue string splits per
 *     character and the chain break "|" has an id;
 *   - a character the alphabet does not carry is one <unk> of its own: four
 *     for "mktv", one for a space, and one per multi-byte character.
 *
 * The expectations are the reference tokenizer's output with the framing
 * tokens removed (the pipeline adds <cls> and <eos>). Measured with the
 * tokenizer.json shipped in biohub/ESMC-300M and biohub/ESMC-600M, which
 * tokenize identically.
 */

#include "families/esmc/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

// The real ESMC vocabulary, in the published order. Id 31 is the chain break
// "|", where ESM-2's table has <null_1>.
const char* kVocab = "<cls>\n<pad>\n<eos>\n<unk>\nL\nA\nG\nV\nS\nE\nR\nT\nI\nD\nP\nK\nQ\nN\nF\n"
                     "Y\nM\nH\nW\nC\nX\nB\nU\nZ\nO\n.\n-\n|\n<mask>\n";

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

std::string padded_vocab() {
    // The bundle writes vocab.txt at the engine's width: the 33 real tokens,
    // then one unique placeholder per unused column.
    std::string text = kVocab;
    for (std::int32_t id = 33; id < 64; ++id)
        text += "<esmc_unused_" + std::to_string(id) + ">\n";
    return text;
}

} // namespace

int main() {
    const auto length = std::char_traits<char>::length(kVocab);
    auto tokenizer = trtmc::CreateEsmcTokenizer(kVocab, length);
    if (!tokenizer) {
        std::cerr << "FAIL: tokenizer did not load\n";
        return 1;
    }

    expect(tokenizer->size() == 33, "the vocabulary has 33 entries");
    expect(tokenizer->id_for_token("<cls>") == 0, "<cls> is 0");
    expect(tokenizer->id_for_token("<pad>") == 1, "<pad> is 1");
    expect(tokenizer->id_for_token("<eos>") == 2, "<eos> is 2");
    expect(tokenizer->id_for_token("<unk>") == 3, "<unk> is 3");
    expect(tokenizer->id_for_token("|") == 31, "the chain break is 31");
    expect(tokenizer->id_for_token("<mask>") == 32, "<mask> is 32");
    expect(tokenizer->token_for_id(20) == "M", "id 20 is M");

    // One id per residue, not one per run.
    check(tokenizer->encode("MKTV"), {20, 15, 11, 7}, "a residue run is split per character");
    check(tokenizer->encode("MKTVRQERLK"), {20, 15, 11, 7, 10, 16, 9, 10, 4, 15},
          "a longer residue run");

    // The longest match wins, so the six characters of "<mask>" are one id.
    check(tokenizer->encode("MK<mask>TV"), {20, 15, 32, 11, 7}, "<mask> is matched whole");
    check(tokenizer->encode("<mask>"), {32}, "<mask> alone is one id");

    // The chain break has a vocabulary entry of its own.
    check(tokenizer->encode("MK|TV"), {20, 15, 31, 11, 7}, "the chain break is a token");
    check(tokenizer->encode("|"), {31}, "the chain break alone");

    // Spaces are not dropped: each one is a character outside the alphabet,
    // so it becomes its own <unk>. Table-driven against the reference.
    check(tokenizer->encode("MK TV"), {20, 15, 3, 11, 7}, "a space is one <unk>");
    check(tokenizer->encode("M K\tT\nV"), {20, 3, 15, 3, 11, 3, 7},
          "every whitespace character is one <unk>");
    check(tokenizer->encode("  "), {3, 3}, "two spaces are two <unk>s");

    // An unknown run is one <unk> per character; ESM-2 collapses it to one.
    check(tokenizer->encode("mktv"), {3, 3, 3, 3}, "every lowercase residue is its own <unk>");
    check(tokenizer->encode("MJJJK"), {20, 3, 3, 3, 15}, "three unknown residues are three <unk>s");
    check(tokenizer->encode("MK@TV"), {20, 15, 3, 11, 7}, "'@' is one <unk>");

    // A partial special token is not a special token: '<' and each lowercase
    // letter miss the alphabet separately.
    check(tokenizer->encode("M<mask"), {20, 3, 3, 3, 3, 3},
          "an unterminated <mask is per character");

    // The alphabet carries punctuation that proteins do use, and the
    // remaining residues not in a first scan.
    check(tokenizer->encode("MKBUZOX.-"), {20, 15, 25, 26, 27, 28, 24, 29, 30},
          "residues and the two punctuation tokens");

    // A character is a Unicode scalar value, not a byte: an accented letter and
    // an emoji outside the basic plane cost one <unk> each, not two or four.
    check(tokenizer->encode("MéT"), {20, 3, 11}, "a two-byte character is one <unk>");
    check(tokenizer->encode("M\xF0\x9F\x99\x82T"), {20, 3, 11}, "an emoji is one <unk>");

    check(tokenizer->encode(""), {}, "an empty string encodes to nothing");

    // The bundle's file is 64 lines wide; the reader must accept it and keep
    // the ids of the real tokens unchanged.
    const auto builder = padded_vocab();
    auto padded = trtmc::CreateEsmcTokenizer(builder.data(), builder.size());
    if (!padded) {
        std::cerr << "FAIL: padded vocabulary did not load\n";
        return 1;
    }
    expect(padded->size() == 64, "the padded vocabulary has 64 entries");
    check(padded->encode("MK<mask>TV"), {20, 15, 32, 11, 7}, "padding does not move the ids");
    expect(padded->id_for_token("<esmc_unused_63>") == 63, "the last placeholder is id 63");

    if (failures == 0)
        std::cout << "tokenizer: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
