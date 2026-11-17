/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only checks of the Voxtral front end (FFT, log-mel framing, downmix) and
// text side (vocabulary decoding, UTF-8 replacement, transcription prompt).

#include "families/voxtral/runtime/frontend.h"
#include "families/voxtral/runtime/text.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::voxtral::Fft;
using trtmc::voxtral::FrontendConfig;

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

template <class Function>
bool throws(Function function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

constexpr double kPi = 3.14159265358979323846;

std::vector<std::complex<double>> direct_dft(const std::vector<std::complex<double>>& input) {
    const auto n = input.size();
    std::vector<std::complex<double>> output(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> sum = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            sum += input[j] * std::polar(1.0, -2.0 * kPi * static_cast<double>(j * k % n) / n);
        output[k] = sum;
    }
    return output;
}

void test_fft() {
    std::uint32_t state = 12345;
    auto next = [&] {
        state = state * 1664525U + 1013904223U;
        return static_cast<double>(state >> 8) / static_cast<double>(1U << 24) - 0.5;
    };
    for (std::int32_t length : {1, 2, 6, 15, 30, 64, 400}) {
        std::vector<std::complex<double>> input(static_cast<std::size_t>(length));
        for (auto& value : input)
            value = {next(), next()};
        std::vector<std::complex<double>> output(input.size());
        Fft fft(length);
        fft.forward(input.data(), output.data());
        const auto expected = direct_dft(input);
        double error = 0.0, scale = 0.0;
        for (std::size_t k = 0; k < input.size(); ++k) {
            error = std::max(error, std::abs(output[k] - expected[k]));
            scale = std::max(scale, std::abs(expected[k]));
        }
        check(error <= 1e-12 * std::max(1.0, scale),
              "fft matches the direct DFT, n=" + std::to_string(length));
    }
    check(throws([] { Fft fft(7); }), "fft rejects a length with a prime factor above 5");
    check(throws([] { Fft fft(0); }), "fft rejects an empty length");
}

void test_downmix_and_chunks() {
    const float stereo[] = {1.0F, 3.0F, -2.0F, 0.0F};
    const auto mono = trtmc::voxtral::downmix(stereo, 4, 2);
    check(mono.size() == 2 && mono[0] == 2.0F && mono[1] == -1.0F, "downmix averages channels");
    const auto same = trtmc::voxtral::downmix(stereo, 4, 1);
    check(same.size() == 4 && same[3] == 0.0F, "downmix keeps mono audio");
    check(throws([&] { trtmc::voxtral::downmix(stereo, 3, 2); }), "downmix rejects partial frames");
    check(throws([&] { trtmc::voxtral::downmix(stereo, 4, 0); }), "downmix rejects zero channels");
    check(throws([&] { trtmc::voxtral::downmix(nullptr, 0, 1); }), "downmix rejects empty audio");
    check(trtmc::voxtral::chunk_count(1, 480000) == 1, "one sample is one chunk");
    check(trtmc::voxtral::chunk_count(480000, 480000) == 1, "exactly 30 s is one chunk");
    check(trtmc::voxtral::chunk_count(480001, 480000) == 2, "one sample more is two chunks");
}

// Independent restatement of the feature definition on a tiny geometry:
// zero padding to whole chunks, reflect padding of n_fft / 2, periodic Hann,
// power spectrum, filter bank, log10, clip-wide floor, scaling, last frame dropped.
std::vector<float> reference_features(const std::vector<float>& mono,
                                      const std::vector<float>& filters,
                                      const FrontendConfig& config) {
    const auto chunks = static_cast<std::int64_t>(
        std::max<std::size_t>(1, (mono.size() + config.chunk_samples - 1) / config.chunk_samples));
    std::vector<double> padded(static_cast<std::size_t>(chunks * config.chunk_samples), 0.0);
    std::copy(mono.begin(), mono.end(), padded.begin());
    const std::int64_t pad = config.n_fft / 2;
    std::vector<double> framed;
    for (std::int64_t index = pad; index > 0; --index)
        framed.push_back(padded[static_cast<std::size_t>(index)]);
    framed.insert(framed.end(), padded.begin(), padded.end());
    for (std::int64_t index = 1; index <= pad; ++index)
        framed.push_back(padded[padded.size() - 1 - static_cast<std::size_t>(index)]);
    const std::int32_t bins = config.n_fft / 2 + 1;
    const std::int64_t frames = chunks * config.mel_frames_per_chunk;
    std::vector<double> values(static_cast<std::size_t>(frames * config.num_mel_bins));
    double maximum = -1e300;
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        std::vector<std::complex<double>> input(static_cast<std::size_t>(config.n_fft));
        for (std::int32_t n = 0; n < config.n_fft; ++n) {
            const double window = 0.5 - 0.5 * std::cos(2.0 * kPi * n / config.n_fft);
            input[static_cast<std::size_t>(n)] =
                framed[static_cast<std::size_t>(frame * config.hop_length + n)] * window;
        }
        const auto spectrum = direct_dft(input);
        for (std::int32_t mel = 0; mel < config.num_mel_bins; ++mel) {
            double energy = 0.0;
            for (std::int32_t bin = 0; bin < bins; ++bin)
                energy += filters[static_cast<std::size_t>(bin * config.num_mel_bins + mel)] *
                          std::norm(spectrum[static_cast<std::size_t>(bin)]);
            const double value = std::log10(std::max(energy, 1e-10));
            values[static_cast<std::size_t>(frame * config.num_mel_bins + mel)] = value;
            maximum = std::max(maximum, value);
        }
    }
    std::vector<float> features(values.size());
    const std::int64_t per_chunk = config.mel_frames_per_chunk;
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        const std::int64_t chunk = frame / per_chunk;
        for (std::int32_t mel = 0; mel < config.num_mel_bins; ++mel) {
            const double value = std::max(
                values[static_cast<std::size_t>(frame * config.num_mel_bins + mel)], maximum - 8.0);
            features[static_cast<std::size_t>((chunk * config.num_mel_bins + mel) * per_chunk +
                                              frame % per_chunk)] =
                static_cast<float>((value + 4.0) / 4.0);
        }
    }
    return features;
}

void test_log_mel() {
    FrontendConfig config;
    config.sample_rate = 16;
    config.chunk_samples = 24;
    config.n_fft = 8;
    config.hop_length = 2;
    config.num_mel_bins = 3;
    config.mel_frames_per_chunk = 12;
    const std::vector<float> filters = {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,  0.5F, 0.5F,
                                        0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.25F, 0.75F};
    std::vector<float> mono(30);
    for (std::size_t index = 0; index < mono.size(); ++index)
        mono[index] = static_cast<float>(std::sin(0.7 * static_cast<double>(index)) +
                                         0.01 * static_cast<double>(index % 5));
    const auto features = trtmc::voxtral::log_mel_features(mono, filters, config);
    const auto expected = reference_features(mono, filters, config);
    check(features.size() == 2U * 3U * 12U, "two chunks of mels x frames");
    double error = 0.0;
    for (std::size_t index = 0; index < std::min(features.size(), expected.size()); ++index)
        error = std::max(error, static_cast<double>(std::abs(features[index] - expected[index])));
    check(features.size() == expected.size() && error < 1e-6, "log-mel matches its definition");
    // A silent tail far below the clip maximum is floored at maximum - 8.
    const float top = *std::max_element(features.begin(), features.end());
    const float bottom = *std::min_element(features.begin(), features.end());
    check(std::abs((top - bottom) - 2.0F) < 1e-6F,
          "silence is floored eight decades below the peak");
    config.chunk_samples = 25;
    check(throws([&] { trtmc::voxtral::log_mel_features(mono, filters, config); }),
          "inconsistent geometry is rejected");
}

std::vector<char> table(const std::vector<std::string>& pieces) {
    std::vector<char> bytes;
    auto put = [&](std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8)
            bytes.push_back(static_cast<char>((value >> shift) & 0xFFU));
    };
    put(static_cast<std::uint32_t>(pieces.size()));
    std::uint32_t offset = 0;
    put(offset);
    for (const auto& piece : pieces) {
        offset += static_cast<std::uint32_t>(piece.size());
        put(offset);
    }
    for (const auto& piece : pieces)
        bytes.insert(bytes.end(), piece.begin(), piece.end());
    return bytes;
}

void test_text() {
    using trtmc::voxtral::decode_utf8_replace;
    const std::string fffd = "\xEF\xBF\xBD";
    check(decode_utf8_replace("a\xE2\x82\xAC"
                              "b") == "a\xE2\x82\xAC"
                                      "b",
          "valid UTF-8 is kept");
    check(decode_utf8_replace("\xED\xA0\x80") == fffd + fffd + fffd, "surrogates are replaced");
    check(decode_utf8_replace("\xF0\x80\x80") == fffd + fffd + fffd, "overlong lead is replaced");
    check(decode_utf8_replace("\xF4\x90\x80\x80") == fffd + fffd + fffd + fffd,
          "code points above U+10FFFF are replaced");
    check(decode_utf8_replace("\xE2\x82"
                              "A") == fffd + "A",
          "a broken sequence is one U+FFFD");
    check(decode_utf8_replace("\xC0\xAF") == fffd + fffd, "invalid leads are replaced");
    check(decode_utf8_replace("\xF0\x9F\x98") == fffd, "a truncated tail is one U+FFFD");
    check(decode_utf8_replace("\xF0\x9F\x98\x80") == "\xF0\x9F\x98\x80", "four-byte UTF-8 is kept");

    const trtmc::voxtral::Vocabulary vocabulary(
        table({"", "", "", "lang", ":", "en", " Hello", "\xE2\x82", "\xAC", "\xFF"}), 3);
    check(vocabulary.size() == 10, "vocabulary size");
    check(vocabulary.decode({6}) == " Hello", "regular tokens decode to their bytes");
    check(vocabulary.decode({3, 4, 5, 6}) == " Hello", "a leading language tag is removed");
    check(vocabulary.decode({6, 3, 4, 5}) == " Hellolang:en", "only a leading tag is removed");
    check(vocabulary.decode({7, 8}) == "\xE2\x82\xAC", "a code point split over tokens is joined");
    check(vocabulary.decode({7, 1, 8}) == fffd + fffd, "special tokens split decoding runs");
    check(vocabulary.decode({0, 9, 2}) == fffd, "special tokens are ignored");
    check(throws([&] { vocabulary.decode({10}); }), "out-of-range ids are rejected");
    check(throws([] { trtmc::voxtral::Vocabulary broken({'\x01', 0, 0}, 0); }),
          "a truncated vocabulary is rejected");

    trtmc::voxtral::PromptContract contract;
    contract.prefix = {1, 3, 25};
    contract.suffix = {4, 34};
    contract.language_suffix = {{"en", {4, 9909, 1058, 1262, 34}}};
    contract.audio_token_id = 24;
    contract.audio_tokens_per_chunk = 375;
    const auto automatic = trtmc::voxtral::transcription_prompt(contract, 2, std::nullopt);
    check(automatic.size() == 3U + 750U + 2U && automatic[2] == 25 && automatic[3] == 24 &&
              automatic[752] == 24 && automatic[753] == 4 && automatic.back() == 34,
          "automatic-language prompt layout");
    const auto english = trtmc::voxtral::transcription_prompt(contract, 1, "en");
    check(english.size() == 3U + 375U + 5U && english[379] == 9909, "language prompt layout");
    check(throws([&] { trtmc::voxtral::transcription_prompt(contract, 1, "xx"); }),
          "unknown languages are rejected");
    check(throws([&] { trtmc::voxtral::transcription_prompt(contract, 0, std::nullopt); }),
          "an empty clip is rejected");
}

} // namespace

int main() {
    test_fft();
    test_downmix_and_chunks();
    test_log_mel();
    test_text();
    if (g_failures != 0) {
        std::cerr << g_failures << " voxtral front-end check(s) failed\n";
        return 1;
    }
    std::cout << "voxtral front-end checks passed\n";
    return 0;
}
