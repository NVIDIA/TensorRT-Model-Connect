/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side rules of the Granite Speech 5 runtime: the reference's frame counts and tail sample,
// reflect padding, channel mixing, greedy CTC collapse, the tokenizer's decoding (special tokens,
// byte fallback, UTF-8 repair, leading space) and the bundle's JSON checks.
#include "families/granite_speech5/runtime/ctc.h"

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

using namespace trtmc::granite_speech5;

const char* kRuntime =
    R"({"precision": "bf16", "sample_rate": 16000, "n_fft": 512, "hop_length": 160,
  "frame_stacking": 2, "subsampling_layers": 2, "min_samples": 1120, "max_samples": 480000,
  "vocab_size": 16384})";

std::string with(const std::string& key, const std::string& value) {
    std::string json = kRuntime;
    const auto at = json.find("\"" + key + "\": ");
    const auto start = at + key.size() + 4;
    const auto end = json.find_first_of(",}", start);
    return json.substr(0, start) + value + json.substr(end);
}

void frame_counts_match_the_reference() {
    const auto g = Geometry::from_json(kRuntime);
    check(g.bf16 && g.sample_rate == 16000 && g.n_fft == 512 && g.hop_length == 160,
          "geometry fields");
    // The LibriSpeech fixture: 373040 samples -> 2331 hop frames -> 2332 kept -> 1166 stacked ->
    // 291.
    check(g.mel_frames(373040) == 2332 && g.output_frames(373040) == 291,
          "LibriSpeech frame count");
    check(!g.needs_tail_sample(373040), "373040 samples need no tail sample");
    // An odd number of whole hops: the reference appends one zero sample to complete the pair.
    check(g.needs_tail_sample(7 * 160) && g.waveform_length(7 * 160) == 7 * 160 + 1 + 512,
          "tail sample");
    check(!g.needs_tail_sample(7 * 160 + 1) && !g.needs_tail_sample(8 * 160), "no tail sample");
    // The shortest clip with one CTC frame: 7 hops -> 8 kept frames -> 4 stacked -> 2 -> 1.
    check(g.output_frames(1120) == 1 && g.output_frames(1119) == 0, "shortest clip");
    check(g.output_frames(16000) == 12 && g.output_frames(480000) == 375,
          "one second and the longest clip");
}

void runtime_json_is_checked() {
    check_throws([] { Geometry::from_json(with("n_fft", "512.0")); }, "a float n_fft");
    check_throws([] { Geometry::from_json(with("n_fft", "511")); }, "an odd n_fft");
    check_throws([] { Geometry::from_json(with("hop_length", "1")); }, "a one-sample hop");
    check_throws([] { Geometry::from_json(with("frame_stacking", "3")); }, "frame stacking 3");
    check_throws([] { Geometry::from_json(with("precision", "\"fp16\"")); },
                 "an unknown precision");
    check_throws([] { Geometry::from_json(with("min_samples", "1119")); },
                 "a shortest clip without a frame");
    check_throws([] { Geometry::from_json(with("max_samples", "1000")); }, "an empty sample range");
    check_throws([] { Geometry::from_json(with("vocab_size", "4294967296")); },
                 "a vocabulary beyond int32");
    check_throws([] { Geometry::from_json(with("subsampling_layers", "-1")); },
                 "negative subsampling");
    check_throws([] { Geometry::from_json(with("sample_rate", "null")); }, "a null sample rate");
    std::string repeated = kRuntime;
    repeated.insert(1, "\"n_fft\": 400, ");
    check_throws([&] { Geometry::from_json(repeated); }, "a repeated key");
    std::string extra = kRuntime;
    extra.insert(1, "\"window\": \"hann\", ");
    check_throws([&] { Geometry::from_json(extra); }, "an unknown key");
    check_throws([] { Geometry::from_json("[]"); }, "a JSON array");
}

void reflect_padding_matches_numpy() {
    Geometry g = Geometry::from_json(kRuntime);
    g.n_fft = 8;
    g.hop_length = 2;
    // numpy.pad([0, 1, 2, 3, 4], 4, mode="reflect"); 5 samples are 2 hops (even): no tail sample.
    check(pad_waveform({0, 1, 2, 3, 4}, g) ==
              std::vector<float>{4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0},
          "reflect padding");
    // 6 samples are 3 whole hops: the zero tail sample is appended before the reflection.
    check(pad_waveform({0, 1, 2, 3, 4, 5}, g) ==
              std::vector<float>{4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2},
          "reflect padding after the tail sample");
    check_throws([&] { pad_waveform({0, 1, 2}, g); }, "reflect padding wider than the clip");
}

void channels_are_averaged() {
    const std::vector<float> stereo{1.0F, 3.0F, -2.0F, -2.0F, 0.5F, 0.25F};
    check(mix_mono(stereo.data(), stereo.size(), 2) == std::vector<float>{2.0F, -2.0F, 0.375F},
          "stereo mean");
    check(mix_mono(stereo.data(), stereo.size(), 1) == stereo, "mono copy");
    // Finite samples whose float sum overflows still average to a finite sample.
    const std::vector<float> loud{3.0e38F, 3.0e38F, -3.0e38F, -3.0e38F};
    check(mix_mono(loud.data(), loud.size(), 2) == std::vector<float>{3.0e38F, -3.0e38F},
          "a loud stereo mean stays finite");
    check_throws([&] { mix_mono(stereo.data(), 5, 2); }, "a partial audio frame");
    check_throws([&] { mix_mono(stereo.data(), stereo.size(), 0); }, "zero channels");
    check_throws([&] { mix_mono(nullptr, 0, 1); }, "no samples");
}

void ctc_collapse_keeps_doubled_tokens() {
    const std::vector<std::int32_t> frames{0, 2, 2, 0, 2, 3, 3, 0, 0, 7};
    check(collapse(frames.data(), static_cast<std::int64_t>(frames.size()), 0) ==
              std::vector<std::int32_t>{2, 2, 3, 7},
          "collapse then drop blanks");
    check(collapse(frames.data(), 0, 0).empty(), "no frames");
    const std::vector<std::int32_t> blanks{0, 0, 0};
    check(collapse(blanks.data(), 3, 0).empty(), "only blanks");
}

void utf8_validation_is_strict() {
    check(valid_utf8("plain") && valid_utf8("\xE3\x81\x82") && valid_utf8("\xF0\x9F\x98\x80"),
          "valid UTF-8");
    check(!valid_utf8("\xE3\x81"), "a truncated sequence");
    check(!valid_utf8("\xC0\xAF"), "an overlong two-byte form");
    check(!valid_utf8("\xE0\x80\xAF"), "an overlong three-byte form");
    check(!valid_utf8("\xED\xA0\x80"), "a surrogate");
    check(!valid_utf8("\xF4\x90\x80\x80"), "a code point above U+10FFFF");
    check(!valid_utf8("\xFF"), "an invalid lead byte");
    check(!valid_utf8("\x80"), "a lone continuation byte");
}

const char* kVocabulary = R"({"blank_id": 0, "entries": [
  [0, ""], [0, ""], [1, " hello"], [1, " world"], [2, 227], [2, 129], [2, 130], [1, ","], [2, 255],
  [1, "  "]]})";

void decoding_matches_the_tokenizer() {
    const auto v = Vocabulary::from_json(kVocabulary, 10);
    check(v.blank_id() == 0 && v.size() == 10, "vocabulary size and blank");
    check(v.decode({2, 3}) == "hello world", "one leading space stripped");
    check(v.decode({7, 2}) == ", hello", "no leading space to strip");
    check(v.decode({9, 2}) == "  hello", "only one leading space stripped");
    check(v.decode({4, 5, 6}) == "\xE3\x81\x82", "a byte run that is UTF-8");
    check(v.decode({4, 1, 5, 6}) == "\xE3\x81\x82", "a special token does not end a byte run");
    check(v.decode({4, 5, 2}) == "\xEF\xBF\xBD\xEF\xBF\xBD hello",
          "one U+FFFD per byte of a broken run");
    check(v.decode({8}) == "\xEF\xBF\xBD", "an invalid byte");
    check(v.decode({}).empty() && v.decode({1, 0}).empty(), "nothing or only special tokens");
    check_throws([&] { v.decode({10}); }, "an id outside the vocabulary");
}

void vocabulary_json_is_checked() {
    check_throws([] { Vocabulary::from_json(kVocabulary, 11); }, "too few entries");
    check_throws(
        [] { Vocabulary::from_json(R"({"blank_id": 2, "entries": [[0, ""], [1, "a"]]})", 2); },
        "a blank outside the vocabulary");
    check_throws(
        [] { Vocabulary::from_json(R"({"blank_id": 0, "entries": [[0, ""], [2, 256]]})", 2); },
        "a byte above 255");
    check_throws(
        [] { Vocabulary::from_json(R"({"blank_id": 0, "entries": [[0, ""], [3, "a"]]})", 2); },
        "an unknown kind");
    check_throws(
        [] { Vocabulary::from_json(R"({"blank_id": 0, "entries": [[0, "x"], [1, "a"]]})", 2); },
        "a skipped entry with text");
    check_throws(
        [] {
            Vocabulary::from_json(
                R"({"blank_id": 0, "blank_id": 1, "entries": [[0, ""], [1, "a"]]})", 2);
        },
        "a repeated key");
    check_throws(
        [] { Vocabulary::from_json(R"({"blank_id": 0.0, "entries": [[0, ""], [1, "a"]]})", 2); },
        "a float blank id");
}

} // namespace

int main() {
    frame_counts_match_the_reference();
    runtime_json_is_checked();
    reflect_padding_matches_numpy();
    channels_are_averaged();
    ctc_collapse_keeps_doubled_tokens();
    utf8_validation_is_strict();
    decoding_matches_the_tokenizer();
    vocabulary_json_is_checked();
    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("granite_speech5 runtime checks passed\n");
    return 0;
}
