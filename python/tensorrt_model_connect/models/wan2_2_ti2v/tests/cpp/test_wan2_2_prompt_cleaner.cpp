/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "prompt_cleaner.h"

#include <iostream>
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

} // namespace

int main() {
    test_preserves_official_positive_prompt();
    test_matches_official_negative_prompt_width_repair();
    test_matches_double_html_unescape_and_unicode_whitespace();
    if (failures != 0) {
        std::cerr << failures << " Wan2.2 prompt-cleaner test(s) failed\n";
        return 1;
    }
    return 0;
}
