/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Spot checks of the Unicode properties the tokenizers use; the expected
// values are the Unicode Character Database's. Code points are written as
// numbers so that combining marks stay visible.

#include "families/stable_diffusion_3/runtime/unicode.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

namespace unicode = trtmc::stable_diffusion_3::unicode;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::u32string text(std::initializer_list<char32_t> codepoints) {
    return {codepoints};
}

void test_nfc() {
    check(unicode::nfc(text({'c', 'a', 'f', 'e', 0x0301})) == text({'c', 'a', 'f', 0x00E9}),
          "e + acute composes");
    check(unicode::nfc(text({'A', 0x030A})) == text({0x00C5}), "A + ring composes to U+00C5");
    check(unicode::nfc(text({0x212B})) == text({0x00C5}), "the angstrom sign normalizes to U+00C5");
    check(unicode::nfc(text({'a', 0x0323, 0x0302})) == text({0x1EAD}),
          "marks reorder by class, then compose");
    check(unicode::nfc(text({'a', 0x0302, 0x0323})) == text({0x1EAD}),
          "either mark order composes alike");
    check(unicode::nfc(text({0x1112, 0x1161, 0x11AB})) == text({0xD55C}),
          "conjoining jamo compose to U+D55C");
    check(unicode::nfc(text({0x0344})) == text({0x0308, 0x0301}),
          "an excluded composite stays decomposed");
    check(unicode::nfc(text({'a', 'b'})) == text({'a', 'b'}), "ASCII is unchanged");
    check(unicode::nfc_utf8("e\xcc\x81") == "\xc3\xa9", "NFC on UTF-8");
}

void test_lowercase_and_categories() {
    check(unicode::lowercase(0x01A0) == 0x01A1, "Vietnamese O-horn lowercases");
    check(unicode::lowercase(0x01AF) == 0x01B0, "Vietnamese U-horn lowercases");
    check(unicode::lowercase(0x03A3) == 0x03C3, "capital sigma lowercases to sigma");
    check(unicode::lowercase(0x01C5) == 0x01C6, "titlecase dz lowercases");
    check(unicode::lowercase(0xFF21) == 0xFF41, "full-width A lowercases");
    check(unicode::lowercase(0x10A0) == 0x2D00, "Georgian lowercases");
    check(unicode::lowercase('a') == 'a', "lowercase is unchanged");
    check(unicode::is_letter(0x0905) && !unicode::is_letter(0x093F),
          "a Devanagari letter is \\p{L}, its vowel sign is not");
    check(unicode::is_mark(0x093F) && unicode::is_mark(0x0301),
          "vowel signs and accents are marks");
    check(unicode::is_number(0x0966) && unicode::is_number(0x00BD) && unicode::is_number(0x2460),
          "Devanagari digits, fractions and circled numbers are \\p{N}");
    check(!unicode::is_letter(0x0660) && unicode::is_number(0x0660),
          "Arabic-Indic digits are numbers, not letters");
}

void test_grapheme_controls() {
    check(unicode::is_grapheme_control('\t') && unicode::is_grapheme_control('\n') &&
              unicode::is_grapheme_control('\r') && unicode::is_grapheme_control(0x0085),
          "tab, LF, CR and NEL end a grapheme cluster");
    check(unicode::is_grapheme_control(0x200B) && unicode::is_grapheme_control(0x200F) &&
              unicode::is_grapheme_control(0x2028) && unicode::is_grapheme_control(0xFEFF),
          "zero-width space, RLM, line separator and BOM are controls");
    check(!unicode::is_grapheme_control(0x200C) && !unicode::is_grapheme_control(0x200D),
          "ZWNJ and ZWJ extend a cluster");
    check(!unicode::is_grapheme_control(0x0600) && !unicode::is_grapheme_control(0xE0061),
          "prepended concatenation marks and tag characters are not controls");
    check(!unicode::is_grapheme_control(' ') && !unicode::is_grapheme_control('a') &&
              !unicode::is_grapheme_control(0x0301),
          "space, letters and marks are not controls");
    check(unicode::is_grapheme_prepend(0x0600) && unicode::is_grapheme_prepend(0x0D4E) &&
              unicode::is_grapheme_prepend(0x110BD),
          "prepended concatenation marks and the Malayalam dot reph are Prepend");
    check(!unicode::is_grapheme_prepend('a') && !unicode::is_grapheme_prepend(0x061C),
          "letters and the Arabic letter mark are not Prepend");
    check(unicode::is_grapheme_extend(0x0301) && unicode::is_grapheme_extend(0x200D) &&
              unicode::is_grapheme_extend(0x0E33) && unicode::is_grapheme_extend(0xFF9F) &&
              unicode::is_grapheme_extend(0xE0061) && unicode::is_grapheme_extend(0x1F3FD) &&
              unicode::is_grapheme_extend(0x093F),
          "marks, ZWJ, Thai sara am, halfwidth sound marks, tags, skin tones and spacing marks "
          "extend");
    check(!unicode::is_grapheme_extend(0x102C) && !unicode::is_grapheme_extend(0x1038) &&
              !unicode::is_grapheme_extend('a') && !unicode::is_grapheme_extend(0x200B),
          "the spacing marks UAX #29 excepts, letters and zero-width space do not extend");
    check(unicode::is_white_space(0x85) && unicode::is_white_space(0x3000) &&
              unicode::is_white_space('\v') && !unicode::is_white_space(0x200B) &&
              !unicode::is_white_space(0x1C),
          "White_Space includes NEL and the ideographic space, not ZWSP or U+001C");
}

} // namespace

int main() {
    test_nfc();
    test_lowercase_and_categories();
    test_grapheme_controls();
    if (failures != 0) {
        std::fprintf(stderr, "%d stable_diffusion_3 unicode check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("stable_diffusion_3 unicode checks passed\n");
    return EXIT_SUCCESS;
}
