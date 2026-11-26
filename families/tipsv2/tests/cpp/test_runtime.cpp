/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The TIPSv2 runtime rules that need no engine and no checkpoint:
//
//  * Pillow's bilinear resize, against values recorded from Pillow itself;
//  * the SentencePiece BPE tokenizer, on a hand-built miniature model whose expected ids are
//    derived in the comments rather than recorded from this code;
//  * what the asset reader refuses.
//
// Real-checkpoint tokenizer parity is proved separately, by `tipsv2_tokenizer_check` against the
// reference SentencePiece processor in `tests/test_e2e.py`.

#include "families/tipsv2/runtime/image_preprocess.h"
#include "families/tipsv2/runtime/tokenizer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

template <typename T>
void check_equal(const T& actual, const T& expected, const std::string& what) {
    if (!(actual == expected)) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

std::string describe(const std::vector<std::int32_t>& ids) {
    std::string out = "[";
    for (std::size_t index = 0; index < ids.size(); ++index) {
        out += (index == 0 ? "" : ", ") + std::to_string(ids[index]);
    }
    return out + "]";
}

void check_ids(const trtmc::tipsv2::Tokenizer& tokenizer, const std::string& text,
               const std::vector<std::int32_t>& expected, const std::string& what) {
    const auto actual = tokenizer.encode(text);
    if (actual != expected) {
        std::cerr << "FAIL: " << what << "\n  expected " << describe(expected) << "\n  actual   "
                  << describe(actual) << "\n";
        ++g_failures;
    }
}

// A refusal counts only when it is **this** refusal. `catch (const std::exception&)` accepts
// anything at all - a `std::bad_alloc` from a huge allocation, or another of the family's own
// checks firing first - so a case could pass with the guard it is named after deleted. The
// expected text is part of the message the guard throws.
template <typename Callable>
void check_throws(Callable&& callable, const std::string& what, const std::string& expected) {
    try {
        callable();
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (message.find(expected) != std::string::npos) {
            return;
        }
        std::cerr << "FAIL: " << what << " threw \"" << message << "\", which does not contain \""
                  << expected << "\"\n";
        ++g_failures;
        return;
    }
    std::cerr << "FAIL: " << what << " did not throw\n";
    ++g_failures;
}

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

// ─── the miniature tokenizer model ───
//
// Ids 0-3 are the specials in the released checkpoint's order (pad, eos, unk, bos), 4-259 the 256
// byte pieces, and the rest a handful of normal pieces whose scores are merge ranks: a higher
// (less negative) score merges first, exactly as SentencePiece orders them.

namespace piece_id {
constexpr std::int32_t kPad = 0;
constexpr std::int32_t kEos = 1;
constexpr std::int32_t kUnk = 2;
constexpr std::int32_t kBos = 3;
constexpr std::int32_t kByteBase = 4;
constexpr std::int32_t kSpace = 260;   // "▁"
constexpr std::int32_t kA = 261;       // "a"
constexpr std::int32_t kB = 262;       // "b"
constexpr std::int32_t kSpaceA = 263;  // "▁a"
constexpr std::int32_t kAb = 264;      // "ab"
constexpr std::int32_t kSpaceAb = 265; // "▁ab"
constexpr std::int32_t kSep = 266;     // "[sep]", user defined
constexpr std::int32_t kBa = 267;      // "ba", unused
} // namespace piece_id

constexpr std::int32_t byte_id(unsigned value) {
    return piece_id::kByteBase + static_cast<std::int32_t>(value);
}

void append_u32(std::string& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<char>((value >> shift) & 0xFFU));
    }
}

void append_f32(std::string& out, float value) {
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    append_u32(out, raw);
}

void append_piece(std::string& out, trtmc::tipsv2::PieceType type, float score,
                  const std::string& text) {
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(type)));
    append_f32(out, score);
    append_u32(out, static_cast<std::uint32_t>(text.size()));
    out += text;
}

// A valid Darts double-array with exactly one key, "q" (0x71), replaced by "k".
//
//   units[0]   = 1 << 10, so offset(root) = 1 and the child index for byte b is 1 ^ b;
//   units[112] = (2 << 10) | 0x100 | 0x71, the node for "q": label 0x71, leaf bit set, offset 2;
//   units[114] = 0, the leaf, holding the replacement's offset into the pool (0 -> "k").
//
// Every other unit is zero, whose label is 0, so no other byte can start or continue a key.
std::string miniature_charsmap() {
    std::vector<std::uint32_t> units(115, 0);
    units[0] = 1U << 10U;
    units[112] = (2U << 10U) | 0x100U | 0x71U;
    units[114] = 0U;
    std::string out;
    append_u32(out, static_cast<std::uint32_t>(units.size() * 4));
    for (const std::uint32_t unit : units) {
        append_u32(out, unit);
    }
    out += std::string("k\0", 2); // the replacement pool
    return out;
}

struct Lowercase {
    std::uint32_t code_point;
    std::string lowered;
};

std::string miniature_asset(bool with_user_defined = true) {
    std::string pieces;
    std::uint32_t count = 0;
    const auto add = [&](trtmc::tipsv2::PieceType type, float score, const std::string& text) {
        append_piece(pieces, type, score, text);
        ++count;
    };
    add(trtmc::tipsv2::PieceType::kControl, 0.0F, "<pad>");
    add(trtmc::tipsv2::PieceType::kControl, 0.0F, "</s>");
    add(trtmc::tipsv2::PieceType::kUnknown, 0.0F, "<unk>");
    add(trtmc::tipsv2::PieceType::kControl, 0.0F, "<s>");
    for (unsigned value = 0; value < 256; ++value) {
        char spelled[8] = {};
        std::snprintf(spelled, sizeof(spelled), "<0x%02X>", value);
        add(trtmc::tipsv2::PieceType::kByte, 0.0F, spelled);
    }
    add(trtmc::tipsv2::PieceType::kNormal, -1.0F, "\xE2\x96\x81");
    add(trtmc::tipsv2::PieceType::kNormal, -2.0F, "a");
    add(trtmc::tipsv2::PieceType::kNormal, -3.0F, "b");
    add(trtmc::tipsv2::PieceType::kNormal, -4.0F,
        "\xE2\x96\x81"
        "a");
    add(trtmc::tipsv2::PieceType::kNormal, -5.0F, "ab");
    add(trtmc::tipsv2::PieceType::kNormal, -6.0F,
        "\xE2\x96\x81"
        "ab");
    add(with_user_defined ? trtmc::tipsv2::PieceType::kUserDefined
                          : trtmc::tipsv2::PieceType::kNormal,
        -7.0F, "[sep]");
    add(trtmc::tipsv2::PieceType::kUnused, -8.0F, "ba");

    const std::string charsmap = miniature_charsmap();
    // Only what the cases below need: ASCII upper case, the two Greek letters the sigma cases
    // use, the capital sigma's context-free mapping (which the rule must override), and the one
    // 1:N mapping Python's str.lower() has (U+0130 -> "i" + U+0307).
    std::vector<Lowercase> lowercase;
    for (std::uint32_t value = 'A'; value <= 'Z'; ++value) {
        lowercase.push_back({value, std::string(1, static_cast<char>(value - 'A' + 'a'))});
    }
    lowercase.push_back({0x0130U, "i\xCC\x87"});
    lowercase.push_back({0x039FU, "\xCE\xBF"}); // Ο -> ο
    lowercase.push_back({0x03A3U, "\xCF\x83"}); // Σ -> σ, context-free
    std::sort(lowercase.begin(), lowercase.end(), [](const Lowercase& lhs, const Lowercase& rhs) {
        return lhs.code_point < rhs.code_point;
    });

    // Cased: the two ASCII alphabets and the Greek letters. Case-ignorable: the combining marks.
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> cased{
        {0x41U, 0x5AU}, {0x61U, 0x7AU}, {0x0391U, 0x03A9U}, {0x03B1U, 0x03C9U}};
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> case_ignorable{{0x0300U, 0x036FU}};

    std::string out = "TRTMCTIP";
    append_u32(out, trtmc::tipsv2::TokenizerAsset::kCurrentVersion);
    append_u32(out, 0x0FU); // dummy prefix | remove extra spaces | escape spaces | byte fallback
    append_u32(out, static_cast<std::uint32_t>(piece_id::kUnk));
    append_u32(out, static_cast<std::uint32_t>(piece_id::kBos));
    append_u32(out, static_cast<std::uint32_t>(piece_id::kEos));
    append_u32(out, static_cast<std::uint32_t>(piece_id::kPad));
    append_u32(out, count);
    append_u32(out, static_cast<std::uint32_t>(charsmap.size()));
    append_u32(out, static_cast<std::uint32_t>(lowercase.size()));
    append_u32(out, static_cast<std::uint32_t>(cased.size()));
    append_u32(out, static_cast<std::uint32_t>(case_ignorable.size()));
    out += pieces;
    out += charsmap;
    for (const auto& entry : lowercase) {
        append_u32(out, entry.code_point);
        append_u32(out, static_cast<std::uint32_t>(entry.lowered.size()));
        out += entry.lowered;
    }
    for (const auto& table : {cased, case_ignorable}) {
        for (const auto& range : table) {
            append_u32(out, range.first);
            append_u32(out, range.second);
        }
    }
    return out;
}

void test_tokenizer() {
    const trtmc::tipsv2::Tokenizer tokenizer(
        trtmc::tipsv2::parse_tokenizer_asset(miniature_asset()));

    // "ab" normalises to "▁ab". The pairs are ("▁","a") -> "▁a" at -4 and ("a","b") -> "ab" at
    // -5; -4 wins, then ("▁a","b") -> "▁ab" at -6 merges the rest. One token.
    check_ids(tokenizer, "ab", {piece_id::kSpaceAb}, "dummy prefix and merge order");
    // The same text upper case: the lowercase table runs before the normalizer.
    check_ids(tokenizer, "AB", {piece_id::kSpaceAb}, "lowercase before normalisation");
    // Leading, repeated and trailing spaces collapse to one separator each, and the trailing one
    // is trimmed: "▁a▁b". "▁a" merges; "▁b" is not a piece, so "▁" and "b" stay apart.
    check_ids(tokenizer, "  A  b  ", {piece_id::kSpaceA, piece_id::kSpace, piece_id::kB},
              "whitespace collapse and trailing trim");
    // Nothing but whitespace normalises to nothing at all - not to a lone dummy prefix.
    check_ids(tokenizer, "", {}, "empty prompt");
    check_ids(tokenizer, "   ", {}, "whitespace-only prompt");

    // The charsmap replaces "q" with "k" before anything is merged. Without the trie this would
    // byte-fall-back on 0x71 instead of 0x6B.
    check_ids(tokenizer, "qa", {piece_id::kSpace, byte_id(0x6B), piece_id::kA},
              "precompiled charsmap replacement");
    // U+0130 lowercases to two code points, "i" + U+0307; neither is a piece, so both fall back
    // to their UTF-8 bytes: 0x69, then 0xCC 0x87.
    check_ids(tokenizer, "\xC4\xB0",
              {piece_id::kSpace, byte_id(0x69), byte_id(0xCC), byte_id(0x87)},
              "one-to-many lowercase mapping");
    // A multi-byte character that is not a piece falls back byte by byte, not character by
    // character.
    check_ids(tokenizer, "\xC3\xA9", {piece_id::kSpace, byte_id(0xC3), byte_id(0xA9)},
              "byte fallback for a multi-byte character");
    // One stray byte becomes one U+FFFD, which then falls back to its own three bytes.
    check_ids(tokenizer, "\xFF", {piece_id::kSpace, byte_id(0xEF), byte_id(0xBF), byte_id(0xBD)},
              "malformed UTF-8 becomes U+FFFD");
    // "ba" is an unused piece: the merge happens, and the emitter puts the two halves back.
    check_ids(tokenizer, "ba", {piece_id::kSpace, piece_id::kB, piece_id::kA},
              "unused piece is resegmented");
    // A user-defined symbol is one frozen symbol that no merge may cross: "▁a" still merges, and
    // "[sep]" and "b" stay as they are.
    check_ids(tokenizer, "a[sep]b", {piece_id::kSpaceA, piece_id::kSep, piece_id::kB},
              "user-defined symbol is frozen");
    // A control piece's text is not reachable from a prompt: it is spelled out, byte by byte.
    const auto pad_text = tokenizer.encode("<pad>");
    check(pad_text.size() > 1 && pad_text[0] == piece_id::kSpace,
          "control piece text is not a single token");
    check(std::find(pad_text.begin(), pad_text.end(), piece_id::kPad) == pad_text.end(),
          "a prompt cannot produce the padding id");

    // The byte pieces are spelled "<0xNN>", six bytes, which is longer than any text piece here.
    // ─── the capital sigma, the one character whose lower case depends on its neighbours ───
    //
    // None of ο, σ or ς is a piece here, so each falls back to its own UTF-8 bytes and the ids
    // say which form came out: ο is CE BF, σ is CF 83, ς is CF 82. Without the rule every sigma
    // would be σ and the first two cases below would be the same ids.
    constexpr std::int32_t kOmicron1 = byte_id(0xCE), kOmicron2 = byte_id(0xBF);
    constexpr std::int32_t kSigma1 = byte_id(0xCF), kSigmaSmall2 = byte_id(0x83);
    constexpr std::int32_t kSigmaFinal2 = byte_id(0x82);
    // A cased character before it and nothing after: the final form.
    check_ids(tokenizer, "\xCE\x9F\xCE\xA3",
              {piece_id::kSpace, kOmicron1, kOmicron2, kSigma1, kSigmaFinal2},
              "a sigma at the end of a word is the final form");
    // Nothing before it: not final.
    check_ids(tokenizer, "\xCE\xA3\xCE\x9F",
              {piece_id::kSpace, kSigma1, kSigmaSmall2, kOmicron1, kOmicron2},
              "a sigma at the start of a word is not final");
    // A cased character on both sides: not final.
    check_ids(tokenizer, "\xCE\x9F\xCE\xA3\xCE\x9F",
              {piece_id::kSpace, kOmicron1, kOmicron2, kSigma1, kSigmaSmall2, kOmicron1, kOmicron2},
              "a sigma inside a word is not final");
    // A combining mark follows, which is case-ignorable, so the scan walks past it to the end.
    check_ids(tokenizer, "\xCE\x9F\xCE\xA3\xCC\x81",
              {piece_id::kSpace, kOmicron1, kOmicron2, kSigma1, kSigmaFinal2, byte_id(0xCC),
               byte_id(0x81)},
              "a case-ignorable character does not end the sigma's context");
    check_equal(tokenizer.lowercase("\xCE\x9F\xCE\xA3"), std::string("\xCE\xBF\xCF\x82"),
                "lowercase of a word-final sigma");
    check_equal(tokenizer.lowercase("\xCE\xA3\xCE\x9F"), std::string("\xCF\x83\xCE\xBF"),
                "lowercase of a word-initial sigma");

    check_equal<std::size_t>(tokenizer.longest_piece_bytes(), 6, "longest piece is a byte piece");
    check_equal(tokenizer.lowercase("ABc"), std::string("abc"), "lowercase of ASCII");
    check_equal(tokenizer.normalize("  a  "),
                std::string("\xE2\x96\x81"
                            "a"),
                "normalize");

    // Without the user-defined type "[sep]" is an ordinary piece: nothing freezes it, and with
    // no merge path to build it the text falls back to bytes. The vocabulary is the same size in
    // both fixtures, so only the **ids** can show that the type was read at all.
    const trtmc::tipsv2::Tokenizer plain(
        trtmc::tipsv2::parse_tokenizer_asset(miniature_asset(/*with_user_defined=*/false)));
    check_equal<std::size_t>(plain.asset().pieces.size(), tokenizer.asset().pieces.size(),
                             "both miniature assets have the same vocabulary");
    const auto frozen_ids = tokenizer.encode("a[sep]b");
    const auto plain_ids = plain.encode("a[sep]b");
    if (frozen_ids == plain_ids) {
        std::cerr << "FAIL: the user-defined piece encodes like an ordinary one, so the piece "
                     "type is not being read\n";
        ++g_failures;
    }
    // The frozen one keeps "[sep]" whole; the ordinary one cannot build it and spells it out.
    check_equal<std::size_t>(frozen_ids.size(), 3, "a[sep]b is three symbols when frozen");
    if (plain_ids.size() <= frozen_ids.size()) {
        std::cerr << "FAIL: without the user-defined type the bracketed text should fall back to "
                     "bytes, giving more ids than the frozen piece\n";
        ++g_failures;
    }
}

// The same asset with the two cased ranges swapped, so they are no longer in order.
std::string miniature_asset_with_bad_ranges() {
    std::string asset = miniature_asset();
    // The ranges are the last 5 * 8 bytes: four cased, one case-ignorable.
    const std::size_t ranges = asset.size() - 5 * 8;
    std::string head = asset.substr(0, ranges);
    std::string first = asset.substr(ranges, 8);
    std::string second = asset.substr(ranges + 8, 8);
    return head + second + first + asset.substr(ranges + 16);
}

std::string with_byte(const std::string& asset, std::size_t offset, char value) {
    std::string copy = asset;
    copy[offset] = value;
    return copy;
}

void test_asset_refusals() {
    const std::string asset = miniature_asset();
    using trtmc::tipsv2::parse_tokenizer_asset;
    using trtmc::tipsv2::Tokenizer;

    check_throws([&] { parse_tokenizer_asset(with_byte(asset, 0, 'X')); }, "wrong magic",
                 "wrong magic");
    check_throws([&] { parse_tokenizer_asset(asset.substr(0, asset.size() - 1)); },
                 "truncated asset", "is truncated");
    check_throws([&] { parse_tokenizer_asset(asset + std::string("\0", 1)); }, "trailing bytes",
                 "trailing bytes");
    // Case tables that are not sorted and disjoint, which `in_ranges` binary-searches.
    check_throws([&] { Tokenizer{parse_tokenizer_asset(miniature_asset_with_bad_ranges())}; },
                 "unsorted case ranges", "not sorted and disjoint");
    // The flags word sits at offset 12 (8 magic + 4 version); bit 5 is not defined.
    check_throws([&] { parse_tokenizer_asset(with_byte(asset, 12, static_cast<char>(0x2FU))); },
                 "unknown flag bit", "unknown flags");
    // Byte fallback is required, not optional: without it the reference answers a run of unknown
    // characters with one unknown id where this encoder emits one per symbol, so a bundle built
    // for that mode is refused rather than tokenized by a rule this family never qualified on.
    check_throws(
        [&] { Tokenizer{parse_tokenizer_asset(with_byte(asset, 12, static_cast<char>(0x07U)))}; },
        "a tokenizer without byte fallback", "needs a tokenizer with byte fallback");
    // The version word sits at offset 8.
    check_throws([&] { parse_tokenizer_asset(with_byte(asset, 8, static_cast<char>(2))); },
                 "unsupported version", "version is not supported");

    const std::string charsmap = miniature_charsmap();
    // A trie length that is not a multiple of four, and one that runs past the blob.
    trtmc::tipsv2::PrecompiledCharsMap map;
    check_throws([&] { map.load(std::string_view(charsmap).substr(0, 3)); }, "charsmap too short",
                 "normalization map is truncated");
    std::string broken = charsmap;
    broken[0] = static_cast<char>(5);
    check_throws([&] { map.load(broken); }, "charsmap trie length is not a multiple of four",
                 "invalid trie size");
    std::string oversized = charsmap;
    oversized[1] = static_cast<char>(0xFFU);
    check_throws([&] { map.load(oversized); }, "charsmap trie runs past the blob",
                 "invalid trie size");
}

void test_image_preprocess(const char* path) {
    const auto document = nlohmann::json::parse(read_file(path));
    const auto& cases = document.at("cases");
    check(cases.is_array() && !cases.empty(), "pillow data has cases");

    // Pillow resamples one axis first and the other second, and which one it picks is visible in
    // the output because the intermediate is 8-bit. Both branches of that choice have to be in
    // the data, or half of the rule goes untested.
    bool saw_horizontal_first = false;
    bool saw_vertical_first = false;
    for (const auto& entry : cases) {
        (entry.at("horizontal_first").get<bool>() ? saw_horizontal_first : saw_vertical_first) =
            true;
    }
    check(saw_horizontal_first, "pillow data covers the horizontal-first order");
    check(saw_vertical_first, "pillow data covers the vertical-first order");

    for (const auto& entry : cases) {
        const auto note = entry.at("note").get<std::string>();
        const auto input_h = entry.at("input_height").get<std::int32_t>();
        const auto input_w = entry.at("input_width").get<std::int32_t>();
        const auto output_h = entry.at("output_height").get<std::int32_t>();
        const auto output_w = entry.at("output_width").get<std::int32_t>();
        const auto pixels = entry.at("pixels").get<std::vector<int>>();
        const auto expected = entry.at("expected").get<std::vector<int>>();

        std::vector<float> source(pixels.size());
        for (std::size_t index = 0; index < pixels.size(); ++index) {
            source[index] = static_cast<float>(pixels[index]) / 255.0F;
        }
        trtmc::tipsv2::PreprocessConfig config;
        config.input_image_h = output_h;
        config.input_image_w = output_w;
        const auto actual = preprocess_tipsv2_image(source.data(), input_h, input_w, config);
        // Return, do not merely record: `check_equal` counts the failure and carries on, and the
        // loop below would then index samples the result does not have.
        if (actual.size() != static_cast<std::size_t>(3 * output_h * output_w)) {
            std::cerr << "FAIL: " << note << ": output size " << actual.size() << ", expected "
                      << 3 * output_h * output_w << "\n";
            ++g_failures;
            return;
        }

        const auto plane = static_cast<std::size_t>(output_h) * output_w;
        for (std::int32_t y = 0; y < output_h; ++y) {
            for (std::int32_t x = 0; x < output_w; ++x) {
                for (std::int32_t c = 0; c < 3; ++c) {
                    const auto want = static_cast<float>(
                        expected[static_cast<std::size_t>((y * output_w + x) * 3 + c)]);
                    const auto got = actual[static_cast<std::size_t>(c) * plane +
                                            static_cast<std::size_t>(y) * output_w + x];
                    // The rescale is the only float step; the resize itself is exact integer
                    // arithmetic, so the comparison is against the recorded 8-bit value.
                    // `std::abs(NaN - want) > tol` is false, so a NaN would pass the tolerance
                    // and an output of the right size full of NaNs would look correct.
                    if (!std::isfinite(got) || std::abs(got * 255.0F - want) > 1e-3F) {
                        std::cerr << "FAIL: " << note << " at (" << y << "," << x << "," << c
                                  << "): expected " << want << ", got " << got * 255.0F << "\n";
                        ++g_failures;
                        return;
                    }
                }
            }
        }
    }

    // A non-finite sample is refused at the boundary instead of reaching std::lround.
    std::vector<float> bad(3 * 4, 0.5F);
    bad[5] = std::numeric_limits<float>::quiet_NaN();
    trtmc::tipsv2::PreprocessConfig config;
    config.input_image_h = 2;
    config.input_image_w = 2;
    check_throws([&] { preprocess_tipsv2_image(bad.data(), 2, 2, config); },
                 "non-finite source sample", "non-finite sample");
    check_throws([&] { preprocess_tipsv2_image(nullptr, 2, 2, config); }, "null source",
                 "must be non-empty");

    // An image too large to index safely is refused before any offset is computed - and before
    // the pointer is read, which is why a two-sample buffer can stand in for one. The fixture is
    // deliberately finite here: with the NaN one above, the non-finite check would throw first
    // and this would pass without the size guard existing at all. 120000 x 120000 x 3 is past
    // the limit; 40000 x 20000 x 3 would not have been.
    // Twelve samples: a 2x2 RGB image, which is what the "still resamples" case below reads.
    const std::vector<float> finite(2 * 2 * 3, 0.25F);
    check_throws([&] { preprocess_tipsv2_image(finite.data(), 120000, 120000, config); },
                 "an implausibly large source image", "larger than this family resamples");
    trtmc::tipsv2::PreprocessConfig huge_target = config;
    huge_target.input_image_h = 120000;
    huge_target.input_image_w = 120000;
    check_throws([&] { preprocess_tipsv2_image(finite.data(), 2, 2, huge_target); },
                 "an implausibly large target size", "larger than this family resamples");
    // And a size just inside the limit still resamples rather than throwing, so the guard is a
    // bound and not a blanket refusal.
    trtmc::tipsv2::PreprocessConfig small = config;
    small.input_image_h = 3;
    small.input_image_w = 3;
    const auto resampled = preprocess_tipsv2_image(finite.data(), 2, 2, small);
    check_equal(resampled.size(), static_cast<std::size_t>(27), "a small image still resamples");
    // The finite fixture again, for the reason the comment above gives and this line had
    // ignored: `check_throws` asks only that something threw. With the NaN buffer, deleting the
    // rescale check would leave the non-finite sample to throw instead and this would still
    // pass - a test that cannot fail for the thing it is named after.
    trtmc::tipsv2::PreprocessConfig zero_rescale = config;
    zero_rescale.rescale_factor = 0.0F;
    check_throws([&] { preprocess_tipsv2_image(finite.data(), 2, 2, zero_rescale); },
                 "non-positive rescale factor", "rescale factor must be positive");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_tipsv2_runtime <pillow_bilinear.json>\n";
        return 2;
    }
    try {
        test_tokenizer();
        test_asset_refusals();
        test_image_preprocess(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << "\n";
        ++g_failures;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "tipsv2 runtime checks passed\n";
    return 0;
}
