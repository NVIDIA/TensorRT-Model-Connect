/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/prompt_cleaner.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check_equal(const std::string& actual, const std::string& expected, const char* label) {
    if (actual != expected) {
        std::cerr << "FAIL: " << label << "\nactual:   " << actual << "\nexpected: " << expected
                  << '\n';
        ++failures;
    }
}

void check_rejects_with(const std::string& input, const std::string& expected_message,
                        const char* label) {
    try {
        static_cast<void>(trtmc::wan2_2::clean_t5_prompt(input));
        std::cerr << "FAIL: " << label << " did not throw\n";
        ++failures;
    } catch (const std::invalid_argument& error) {
        check_equal(error.what(), expected_message, label);
    }
}

void test_preserves_official_positive_prompt() {
    constexpr auto prompt =
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a "
        "spotlighted stage";
    check_equal(trtmc::wan2_2::clean_t5_prompt(prompt), prompt,
                "preserves official positive prompt");
}

void test_matches_official_negative_prompt_width_repair() {
    constexpr auto input =
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
    constexpr auto expected = "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,"
                              "整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,"
                              "画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,"
                              "静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走";
    check_equal(trtmc::wan2_2::clean_t5_prompt(input), expected,
                "matches official negative-prompt width repair");
}

void test_matches_double_html_unescape_and_unicode_whitespace() {
    check_equal(trtmc::wan2_2::clean_t5_prompt("  A&amp;quot;B&amp;quot;\t\xE2\x80\x83 C  "),
                "A\"B\" C", "matches double HTML unescape and Unicode whitespace collapse");
    check_equal(trtmc::wan2_2::clean_t5_prompt("&#xFF21;&#66;&#x43;"), "ABC",
                "repairs numeric full-width HTML entity");
}

void test_supported_unicode_whitespace_collapse() {
    const std::string whitespace = "\x09\x0A\x0C\x0D\x20"
                                   "\xC2\xA0\xE1\x9A\x80"
                                   "\xE2\x80\x80\xE2\x80\x81\xE2\x80\x82\xE2\x80\x83"
                                   "\xE2\x80\x84\xE2\x80\x85\xE2\x80\x86\xE2\x80\x87"
                                   "\xE2\x80\x88\xE2\x80\x89\xE2\x80\x8A"
                                   "\xE2\x80\xA8\xE2\x80\xA9\xE2\x80\xAF\xE2\x81\x9F\xE3\x80\x80";
    check_equal(trtmc::wan2_2::clean_t5_prompt("A" + whitespace + "B"), "A B",
                "collapses the qualified Unicode whitespace subset");

    const std::string non_whitespace = "A\xE2\x80\x8B"
                                       "B";
    check_equal(trtmc::wan2_2::clean_t5_prompt(non_whitespace), non_whitespace,
                "preserves zero-width space");
}

void test_supported_numeric_entity_normalization() {
    check_equal(trtmc::wan2_2::clean_t5_prompt("&#0;"), "\xEF\xBF\xBD",
                "numeric null maps to replacement character");
}

void test_rejects_malformed_utf8_with_stable_errors() {
    check_rejects_with(std::string({static_cast<char>(0xF8)}),
                       "Wan2.2 prompt contains invalid UTF-8", "rejects invalid UTF-8 lead");
    check_rejects_with(std::string({static_cast<char>(0xE2), static_cast<char>(0x82)}),
                       "Wan2.2 prompt contains truncated UTF-8", "rejects truncated UTF-8");
    check_rejects_with(std::string({static_cast<char>(0xE2), 'A', static_cast<char>(0xA1)}),
                       "Wan2.2 prompt contains invalid UTF-8 continuation",
                       "rejects invalid UTF-8 continuation");
    check_rejects_with(std::string({static_cast<char>(0xC0), static_cast<char>(0x80)}),
                       "Wan2.2 prompt contains invalid UTF-8 code point",
                       "rejects overlong UTF-8 code point");
    check_rejects_with(
        std::string({static_cast<char>(0xED), static_cast<char>(0xA0), static_cast<char>(0x80)}),
        "Wan2.2 prompt contains invalid UTF-8 code point", "rejects UTF-8 surrogate code point");
    check_rejects_with(std::string({static_cast<char>(0xF4), static_cast<char>(0x90),
                                    static_cast<char>(0x80), static_cast<char>(0x80)}),
                       "Wan2.2 prompt contains invalid UTF-8 code point",
                       "rejects code point above Unicode maximum");
}

} // namespace

int main() {
    test_preserves_official_positive_prompt();
    test_matches_official_negative_prompt_width_repair();
    test_matches_double_html_unescape_and_unicode_whitespace();
    test_supported_unicode_whitespace_collapse();
    test_supported_numeric_entity_normalization();
    test_rejects_malformed_utf8_with_stable_errors();
    if (failures != 0) {
        std::cerr << failures << " Wan2.2 prompt-cleaner test(s) failed\n";
        return 1;
    }
    return 0;
}
