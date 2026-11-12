/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side contracts of moss_transcribe_diarize that need no GPU: runtime.json
// validation, audio chunking and time-marker expansion, the log-mel front end,
// token decoding and the transcript parser.

#include "families/moss_transcribe_diarize/runtime/log_mel.h"
#include "families/moss_transcribe_diarize/runtime/prompt_plan.h"
#include "families/moss_transcribe_diarize/runtime/runtime_config.h"
#include "families/moss_transcribe_diarize/runtime/text_output.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtd = trtmc::moss_transcribe_diarize;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

template <typename Callable>
void expect_throw(Callable&& callable, const std::string& what) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    check(false, what + " did not throw");
}

std::vector<char> bytes_of(const std::string& text) {
    return std::vector<char>(text.begin(), text.end());
}

std::string runtime_json(const std::string& precision, int digits) {
    std::string digit_ids;
    for (int i = 0; i < digits; ++i)
        digit_ids += (i ? "," : "") + std::to_string(15 + i);
    return R"({"precision":")" + precision + R"(","sample_rate":16000,"chunk_samples":480000,)" +
           R"("n_fft":400,"hop_length":160,"num_mel_bins":80,"mel_frames":3000,)" +
           R"("tokens_per_chunk":375,"samples_per_token":1280,"hidden_size":1024,)" +
           R"("num_layers":28,"num_kv_heads":8,"head_dim":128,"vocab_size":151936,)" +
           R"("max_sequence_length":4096,"prefill_chunk_tokens":256,"audio_token_id":151671,)" +
           R"("time_markers":{"enabled":true,"audio_tokens_per_second":12.5,)" +
           R"("every_seconds":5,"digit_token_ids":[)" + digit_ids + R"(]},)" +
           R"("prompt":"p","prompt_prefix_ids":[151644,872],"prompt_suffix_ids":[151670,198],)" +
           R"("eos_token_ids":[151645],"default_max_new_tokens":5120})";
}

void test_runtime_config() {
    const auto config = mtd::parse_runtime_config(bytes_of(runtime_json("bf16", 10)));
    check(config.tokens_per_chunk == 375 && config.samples_per_token == 1280, "geometry");
    check(config.markers.digit_token_ids[9] == 24, "digit ids");
    check(config.eos_token_ids == std::vector<std::int32_t>{151645}, "eos ids");
    expect_throw([] { mtd::parse_runtime_config(bytes_of(runtime_json("fp16", 10))); },
                 "fp16 runtime.json");
    expect_throw([] { mtd::parse_runtime_config(bytes_of(runtime_json("bf16", 9))); },
                 "nine digit ids");
}

void test_chunks() {
    auto chunks = mtd::plan_chunks(1, 480000, 1280);
    check(chunks.size() == 1 && chunks[0].audio_tokens == 1, "one sample -> one token");
    chunks = mtd::plan_chunks(1281, 480000, 1280);
    check(chunks.size() == 1 && chunks[0].audio_tokens == 2, "1281 samples -> two tokens");
    chunks = mtd::plan_chunks(480000, 480000, 1280);
    check(chunks.size() == 1 && chunks[0].audio_tokens == 375, "full chunk -> 375 tokens");
    chunks = mtd::plan_chunks(480001, 480000, 1280);
    check(chunks.size() == 2 && chunks[1].first_sample == 480000 && chunks[1].samples == 1 &&
              chunks[1].audio_tokens == 1,
          "second chunk of one sample");
    expect_throw([] { mtd::plan_chunks(0, 480000, 1280); }, "empty audio");
}

void test_audio_span() {
    mtd::TimeMarkerConfig markers;
    markers.enabled = true;
    markers.audio_tokens_per_second = 12.5;
    markers.every_seconds = 5;
    for (int i = 0; i < 10; ++i)
        markers.digit_token_ids[i] = 100 + i;
    const std::int32_t pad = 7;

    // 70 tokens = 5.6 s: int(12.5 * 5) = 62 placeholders, marker "5", 8 more.
    auto span = mtd::audio_span(70, markers, pad);
    std::vector<std::int32_t> expected(62, pad);
    expected.push_back(105);
    expected.insert(expected.end(), 8, pad);
    check(span == expected, "5.6 s span");

    // 375 tokens = 30 s: markers 5..30 at multiples of 62, then 3 placeholders.
    span = mtd::audio_span(375, markers, pad);
    expected.clear();
    for (int second = 5; second <= 30; second += 5) {
        expected.insert(expected.end(), 62, pad);
        for (const char digit : std::to_string(second))
            expected.push_back(100 + (digit - '0'));
    }
    expected.insert(expected.end(), 3, pad);
    check(span == expected, "30 s span");
    check(std::count(span.begin(), span.end(), pad) == 375, "every audio token is placed");

    // 62 tokens = 4.96 s: no marker yet.
    check(mtd::audio_span(62, markers, pad) == std::vector<std::int32_t>(62, pad), "under 5 s");
    markers.enabled = false;
    check(mtd::audio_span(375, markers, pad) == std::vector<std::int32_t>(375, pad), "disabled");
    check(mtd::audio_span(0, markers, pad).empty(), "no audio tokens");
}

void test_prompt() {
    auto config = mtd::parse_runtime_config(bytes_of(runtime_json("fp32", 10)));
    const auto plan = mtd::build_prompt(config, 70);
    check(plan.input_ids.size() == 2 + 71 + 2, "prompt length");
    check(plan.input_ids.front() == 151644 && plan.input_ids.back() == 198, "prompt framing");
    std::int32_t expected_row = 0;
    bool rows_ok = true;
    for (std::size_t i = 0; i < plan.input_ids.size(); ++i) {
        if (plan.input_ids[i] == config.audio_token_id)
            rows_ok &= plan.audio_index[i] == expected_row++;
        else
            rows_ok &= plan.audio_index[i] == -1;
    }
    check(rows_ok && expected_row == 70, "audio rows fill placeholders in order");
    config.prompt_prefix_ids.push_back(config.audio_token_id);
    expect_throw([&] { mtd::build_prompt(config, 70); }, "placeholder inside the fixed prompt");
}

void test_utf8_and_strip() {
    check(mtd::utf8_lossy("abc\xE4\xBD\xA0") == "abc\xE4\xBD\xA0", "valid UTF-8");
    check(mtd::utf8_lossy("a\xE4\xBD") == "a\xEF\xBF\xBD", "truncated sequence");
    check(mtd::utf8_lossy("\xFF"
                          "a") == "\xEF\xBF\xBD"
                                  "a",
          "invalid lead byte");
    check(mtd::utf8_lossy("\xED\xA0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD",
          "surrogate bytes");
    check(mtd::utf8_lossy("\xE4\xBD"
                          "a") == "\xEF\xBF\xBD"
                                  "a",
          "interrupted sequence");
    check(mtd::python_strip("\xE3\x80\x80 hi \n\t") == "hi", "strip ideographic space");
    check(mtd::python_strip(" \x1c ") == "", "strip control separators");
    check(mtd::python_strip("\xC2\xA0x\xE2\x80\x8B") == "x\xE2\x80\x8B",
          "zero width space is not whitespace");
}

std::vector<char> decode_table(const std::vector<std::string>& pieces,
                               const std::vector<std::uint8_t>& flags) {
    std::vector<char> table = {'M', 'T', 'D', 'T', 'O', 'K', 'V', '1'};
    auto put = [&table](std::uint32_t value) {
        const auto* raw = reinterpret_cast<const char*>(&value);
        table.insert(table.end(), raw, raw + sizeof(value));
    };
    put(static_cast<std::uint32_t>(pieces.size()));
    std::uint32_t offset = 0;
    put(offset);
    for (const auto& piece : pieces)
        put(offset += static_cast<std::uint32_t>(piece.size()));
    table.insert(table.end(), flags.begin(), flags.end());
    for (const auto& piece : pieces)
        table.insert(table.end(), piece.begin(), piece.end());
    return table;
}

void test_token_decoder() {
    const mtd::TokenDecoder decoder(
        decode_table({" [0.5]", "\xE4\xBD", "\xA0", "<|im_end|>", "", " ok "}, {0, 0, 0, 1, 0, 0}));
    check(decoder.decode({0, 1, 2, 3}) == "[0.5]\xE4\xBD\xA0", "bytes join across tokens");
    check(decoder.decode({1, 3}) == "\xEF\xBF\xBD", "dangling bytes become U+FFFD");
    check(decoder.decode({5, 99, -1, 4}) == "ok", "unknown ids are dropped, text stripped");
    expect_throw([] { mtd::TokenDecoder(std::vector<char>{'x'}); }, "bad table");
}

void test_transcript_parser() {
    auto segments = mtd::parse_transcript("[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]");
    check(segments.size() == 2 && segments[0].start_seconds == 0.48 &&
              segments[0].end_seconds == 1.66 && segments[0].speaker == "S01" &&
              segments[0].text == "Welcome" && segments[1].speaker == "S02" &&
              segments[1].text == "Ready" && segments[1].end_seconds == 13.81,
          "compact transcript");
    segments = mtd::parse_transcript(
        "[0.48][S01]\xE4\xBD\xA0\xE5\xA5\xBD[1.66][12.26][S02]\xE5\x8F\xAF[13.81]");
    check(segments.size() == 2 && segments[0].text == "\xE4\xBD\xA0\xE5\xA5\xBD", "multibyte text");
    segments = mtd::parse_transcript("[0][S01]x[2024]y, z[001]w[4]");
    check(segments.size() == 1 && segments[0].text == "x[2024]y, z[001]w" &&
              segments[0].end_seconds == 4.0,
          "numeric brackets inside text");
    segments = mtd::parse_transcript("noise [bad][0.1][S01]hello[0.9]");
    check(segments.size() == 1 && segments[0].start_seconds == 0.1 && segments[0].text == "hello",
          "noise before the first segment");
    segments = mtd::parse_transcript("[0][S01]a[1]\n [2][S02]b[3]");
    check(segments.size() == 2 && segments[1].start_seconds == 2.0 && segments[1].text == "b",
          "whitespace between segments");
    segments = mtd::parse_transcript("[5][S01]late[3]tail");
    check(segments.empty(), "end before start keeps reading text, never closes");
    segments = mtd::parse_transcript("[1][S01]  [2]");
    check(segments.empty(), "empty text is skipped");
    segments = mtd::parse_transcript("[1]\xE3\x80\x80[S03]x[2]");
    check(segments.size() == 1 && segments[0].speaker == "S03", "unicode space before speaker");
}

void test_timestamps_ignore_the_locale() {
    // A comma-decimal LC_NUMERIC changes what strtod reads. The check runs only
    // where such a locale really changes strtod; it never trusts the name alone.
    const char* previous = std::setlocale(LC_NUMERIC, nullptr);
    const std::string saved = previous != nullptr ? previous : "C";
    bool comma_locale = false;
    for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"}) {
        if (std::setlocale(LC_NUMERIC, name) != nullptr && std::strtod("6.15", nullptr) != 6.15) {
            comma_locale = true;
            break;
        }
    }
    if (comma_locale) {
        const auto segments = mtd::parse_transcript("[0.48][S01]x[6.15]");
        check(segments.size() == 1 && segments[0].start_seconds == 0.48 &&
                  segments[0].end_seconds == 6.15,
              "timestamps under a comma-decimal locale");
    } else {
        std::cout << "note: no comma-decimal locale installed; locale check skipped\n";
    }
    std::setlocale(LC_NUMERIC, saved.c_str());
}

void test_dft() {
    for (const std::size_t length : {7U, 400U}) {
        std::vector<std::complex<double>> signal(length);
        for (std::size_t n = 0; n < length; ++n)
            signal[n] = {std::sin(0.37 * n) + 0.1 * n, std::cos(1.3 * n)};
        auto fast = signal;
        mtd::Dft(length).forward(fast);
        double worst = 0.0;
        for (std::size_t k = 0; k < length; ++k) {
            std::complex<double> exact{0.0, 0.0};
            for (std::size_t n = 0; n < length; ++n) {
                const double angle = -2.0 * 3.14159265358979323846 *
                                     static_cast<double>(k * n % length) /
                                     static_cast<double>(length);
                exact += signal[n] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            worst = std::max(worst, std::abs(fast[k] - exact) / (1.0 + std::abs(exact)));
        }
        check(worst < 1e-9, "Bluestein DFT of length " + std::to_string(length));
    }
}

// Straightforward reference of the Whisper log-mel recipe on a small geometry.
std::vector<double> naive_log_mel(const std::vector<float>& samples, int n_fft, int hop, int chunk,
                                  const mtd::MelFilterBank& bank) {
    std::vector<double> x(static_cast<std::size_t>(chunk), 0.0);
    std::copy(samples.begin(), samples.end(), x.begin());
    const int pad = n_fft / 2;
    std::vector<double> y;
    for (int i = pad; i > 0; --i)
        y.push_back(x[static_cast<std::size_t>(i)]);
    y.insert(y.end(), x.begin(), x.end());
    for (int i = 1; i <= pad; ++i)
        y.push_back(x[static_cast<std::size_t>(chunk - 1 - i)]);
    const int frames = chunk / hop;
    std::vector<double> logs(static_cast<std::size_t>(bank.mel_bins * frames));
    double peak = -1e300;
    for (int t = 0; t < frames; ++t) {
        for (int m = 0; m < bank.mel_bins; ++m) {
            double mel = 0.0;
            for (int k = 0; k < bank.freq_bins; ++k) {
                std::complex<double> bin{0.0, 0.0};
                for (int n = 0; n < n_fft; ++n) {
                    const double w = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * n / n_fft);
                    const double angle = -2.0 * 3.14159265358979323846 * k * n / n_fft;
                    bin += y[static_cast<std::size_t>(t * hop + n)] * w *
                           std::complex<double>(std::cos(angle), std::sin(angle));
                }
                mel +=
                    bank.weights[static_cast<std::size_t>(k * bank.mel_bins + m)] * std::norm(bin);
            }
            const double value = std::log10(std::max(mel, 1e-10));
            logs[static_cast<std::size_t>(m * frames + t)] = value;
            peak = std::max(peak, value);
        }
    }
    for (auto& value : logs)
        value = (std::max(value, peak - 8.0) + 4.0) / 4.0;
    return logs;
}

void test_log_mel() {
    mtd::MelFilterBank bank;
    bank.freq_bins = 5;
    bank.mel_bins = 3;
    bank.weights = {0.5F, 0.0F, 0.0F, 1.0F, 0.2F, 0.0F, 0.0F, 0.9F,
                    0.0F, 0.0F, 0.3F, 0.7F, 0.0F, 0.0F, 1.0F};
    const mtd::LogMelExtractor extractor(bank, 8, 2, 32, 16);
    for (const int count : {32, 11}) {
        std::vector<float> samples(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
            samples[static_cast<std::size_t>(i)] = static_cast<float>(std::sin(0.7 * i) * 0.3);
        const auto actual = extractor.chunk(samples.data(), count);
        const auto expected = naive_log_mel(samples, 8, 2, 32, bank);
        double worst = 0.0;
        for (std::size_t i = 0; i < expected.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(actual[i]) - expected[i]));
        check(actual.size() == expected.size() && worst < 1e-6,
              "log-mel of " + std::to_string(count) + " samples");
    }
    expect_throw(
        [&] {
            std::vector<float> long_audio(33, 0.0F);
            extractor.chunk(long_audio.data(), 33);
        },
        "chunk longer than the window");
    expect_throw([&] { mtd::LogMelExtractor(bank, 10, 2, 32, 16); }, "bank/n_fft mismatch");
}

} // namespace

int main() {
    test_runtime_config();
    test_chunks();
    test_audio_span();
    test_prompt();
    test_utf8_and_strip();
    test_token_decoder();
    test_transcript_parser();
    test_timestamps_ignore_the_locale();
    test_dft();
    test_log_mel();
    if (failures != 0) {
        std::cerr << failures << " moss_transcribe_diarize host check(s) failed\n";
        return 1;
    }
    std::cout << "moss_transcribe_diarize host checks passed\n";
    return 0;
}
