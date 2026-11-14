/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side rules of the Granite Speech runtime: reflect padding, the audio placeholder count,
// first-maximum argmax, byte-level detokenisation and its UTF-8 repair, runtime.json checks.
#include "families/granite_speech/runtime/pipeline.h"
#include "families/granite_speech/runtime/text.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

template <class F>
void check_throws(F&& call, const char* what) {
    try {
        call();
    } catch (const std::exception&) {
        return;
    }
    std::printf("FAIL: %s did not throw\n", what);
    ++failures;
}

using namespace trtmc::granite_speech;

void reflect_padding_matches_numpy() {
    const std::vector<float> x{0, 1, 2, 3, 4};
    // numpy.pad(x, 2, mode="reflect")
    check(reflect_pad(x.data(), x.size(), 2) == std::vector<float>{2, 1, 0, 1, 2, 3, 4, 3, 2},
          "reflect padding of 2");
    check_throws([&] { reflect_pad(x.data(), 2, 2); }, "reflect padding wider than the clip");
}

RuntimeConfig config() {
    RuntimeConfig c;
    c.hop_length = 160;
    c.window_size = 15;
    c.num_queries = 3;
    return c;
}

void placeholder_count_matches_the_processor() {
    // GraniteSpeechFeatureExtractor._get_num_audio_features: mel = n // 160 + 1, frames = mel // 2,
    // windows = ceil(frames / 15), tokens = 3 * windows. The checkpoint sample has 399040 samples
    // and 252 placeholders.
    const auto c = config();
    check(c.audio_tokens(399040) == 252, "checkpoint sample placeholder count");
    check(c.audio_tokens(257) == 3, "shortest clip placeholder count");
    check(c.audio_tokens(4959) == 3, "15 frames stay in one window");
    check(c.audio_tokens(4960) == 6, "16 frames start a second window");
}

void argmax_takes_the_first_maximum() {
    const std::vector<float> values{0.5F, 2.0F, -1.0F, 2.0F};
    check(first_argmax(values.data(), values.size()) == 1, "first of two equal maxima");
}

void utf8_repair_matches_the_reference() {
    check(utf8_lossy("abc") == "abc", "ASCII unchanged");
    check(utf8_lossy("p\xC3\xAA"
                     "cheur") == "p\xC3\xAA"
                                 "cheur",
          "two-byte character unchanged");
    const std::string r = "\xEF\xBF\xBD";
    // bytes(...).decode("utf-8", errors="replace") gives these replacements.
    check(utf8_lossy("a\xE2\x82") == "a" + r, "truncated three-byte sequence");
    check(utf8_lossy("\xE2\x41") == r + "A", "invalid continuation keeps the next byte");
    check(utf8_lossy("\xC0\xAF") == r + r, "overlong lead bytes");
    check(utf8_lossy("\xED\xA0\x80") == r + r + r, "surrogate code point");
    check(utf8_lossy("\xF4\x90\x80\x80") == r + r + r + r, "above U+10FFFF");
    check(utf8_lossy("\xF0\x9F\x98") == r, "truncated four-byte sequence");
}

void vocabulary_skips_special_ids() {
    const Vocabulary vocabulary({"he", "llo", "", " w\xC3", "\xA9"}, {2});
    check(vocabulary.decode({0, 1, 2}) == "hello", "special id skipped");
    check(vocabulary.decode({3, 4}) == " w\xC3\xA9", "character split across tokens");
    check(vocabulary.decode({3}) == " w\xEF\xBF\xBD", "dangling lead byte repaired");
    check_throws([&] { vocabulary.decode({5}); }, "id outside the vocabulary");
    const auto parsed = Vocabulary::from_json(R"({"tokens":["6869",""],"skip":[1]})");
    check(parsed.decode({0, 1}) == "hi", "hex table from JSON");
}

std::string runtime_json(const std::string& precision, long long max_samples) {
    return R"({"precision":")" + precision +
           R"(","sample_rate":16000,"n_fft":512,"hop_length":160,"window_size":15,"num_queries":3,)"
           R"("min_samples":257,"max_samples":)" +
           std::to_string(max_samples) +
           R"(,"capacity":64,"hidden_size":8,"num_layers":2,"kv_width":4,"vocab_size":10,)"
           R"("audio_token_id":9,"eos_token_id":8,"prompt_prefix_ids":[1,2],"prompt_suffix_ids":[3])"
           R"(})";
}

void runtime_json_is_checked() {
    const auto parsed = RuntimeConfig::from_json(runtime_json("bf16", 20000));
    check(parsed.work == trtmc::DType::kBFloat16 && parsed.capacity == 64, "valid runtime.json");
    check_throws([] { RuntimeConfig::from_json(runtime_json("fp16", 20000)); },
                 "unknown precision");
    // 200000 samples need 126 placeholders, which with three prompt ids exceed 64 tokens.
    check_throws([] { RuntimeConfig::from_json(runtime_json("fp32", 200000)); },
                 "sample range over capacity");
}

} // namespace

int main() {
    reflect_padding_matches_numpy();
    placeholder_count_matches_the_processor();
    argmax_takes_the_first_maximum();
    utf8_repair_matches_the_reference();
    vocabulary_skips_special_ids();
    runtime_json_is_checked();
    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("granite_speech runtime: all checks passed\n");
    return 0;
}
