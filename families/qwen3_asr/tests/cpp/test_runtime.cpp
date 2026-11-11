/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU tests of the Qwen3-ASR runtime helpers. Expected values were produced by the official
// Hugging Face Transformers 5.22.2 code (Qwen3ASRFeatureExtractor, Qwen3ASRProcessor,
// _detect_and_fix_repetitions, _parse_single_output) and Python's UTF-8 decoder.

#include "families/qwen3_asr/runtime/frontend.h"
#include "families/qwen3_asr/runtime/runtime_config.h"
#include "families/qwen3_asr/runtime/text.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace q = trtmc::qwen3_asr;

namespace {

int failures = 0;

void check(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

// The golden signals: an integer LCG noise plus a square wave, exact in float32 on both sides.
std::vector<float> golden_signal(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    std::uint32_t state = seed;
    for (std::size_t n = 0; n < count; ++n) {
        state = 1664525U * state + 1013904223U;
        const auto noise =
            static_cast<float>(static_cast<std::int32_t>((state >> 16) & 0xFFFFU) - 32768) /
            32768.0F;
        const float square = (n / 20) % 2 ? 0.25F : -0.25F;
        values[n] = square + 0.05F * noise;
    }
    return values;
}

// transformers.audio_utils.mel_filter_bank(201, 128, 0, 8000, 16000, "slaney", "slaney"); the
// production runtime reads the filterbank the builder computed from the bundle instead.
q::MelFilterbank slaney_filters() {
    const int n_freq = 201;
    const int n_mel = 128;
    const auto to_mel = [](double hz) {
        return hz >= 1000.0 ? 15.0 + std::log(hz / 1000.0) * (27.0 / std::log(6.4))
                            : 3.0 * hz / 200.0;
    };
    const auto to_hz = [](double mel) {
        return mel >= 15.0 ? 1000.0 * std::exp((std::log(6.4) / 27.0) * (mel - 15.0))
                           : 200.0 * mel / 3.0;
    };
    const double low = to_mel(0.0);
    const double high = to_mel(8000.0);
    std::vector<double> centers(static_cast<std::size_t>(n_mel + 2));
    for (int i = 0; i < n_mel + 2; ++i)
        centers[static_cast<std::size_t>(i)] = to_hz(low + (high - low) * i / (n_mel + 1));
    q::MelFilterbank filters{std::vector<float>(static_cast<std::size_t>(n_freq * n_mel)), n_freq,
                             n_mel};
    for (int f = 0; f < n_freq; ++f) {
        const double hz = 8000.0 * f / (n_freq - 1);
        for (int m = 0; m < n_mel; ++m) {
            const auto c = [&](int i) { return centers[static_cast<std::size_t>(i)]; };
            const double down = (hz - c(m)) / (c(m + 1) - c(m));
            const double up = (c(m + 2) - hz) / (c(m + 2) - c(m + 1));
            const double weight = std::max(0.0, std::min(down, up)) * 2.0 / (c(m + 2) - c(m));
            filters.weights[static_cast<std::size_t>(f * n_mel + m)] = static_cast<float>(weight);
        }
    }
    return filters;
}

q::FrontendConfig frontend() {
    q::FrontendConfig config;
    config.sample_rate = 16000;
    config.n_fft = 400;
    config.hop_length = 160;
    config.min_samples = 8000;
    config.chunk_frames = 100;
    config.tokens_per_chunk = 13;
    config.window_chunks = 8;
    return config;
}

struct Pick {
    int mel;
    int frame;
    float value;
};

void check_log_mel(std::size_t count, std::uint32_t seed, int frames, double sum, float maximum,
                   float minimum, const std::vector<Pick>& picks) {
    const auto mel = q::log_mel(golden_signal(count, seed), frontend(), slaney_filters());
    const std::string label = "log-mel of the " + std::to_string(count) + "-sample signal: ";
    check(mel.n_mel == 128 && mel.frames == frames, label + "frame count");
    if (mel.frames != frames)
        return;
    double total = 0.0;
    float high = -1e30F, low = 1e30F;
    for (const float value : mel.values) {
        total += value;
        high = std::max(high, value);
        low = std::min(low, value);
    }
    check(std::abs(total - sum) <= 1e-5 * std::abs(sum), label + "sum " + std::to_string(total));
    check(std::abs(high - maximum) <= 2e-4F && std::abs(low - minimum) <= 2e-4F, label + "range");
    for (const auto& pick : picks) {
        const float value = mel.values[static_cast<std::size_t>(pick.mel) * mel.frames +
                                       static_cast<std::size_t>(pick.frame)];
        check(std::abs(value - pick.value) <= 2e-4F,
              label + "bin " + std::to_string(pick.mel) + " frame " + std::to_string(pick.frame) +
                  " = " + std::to_string(value) + ", expected " + std::to_string(pick.value));
    }
}

void test_geometry() {
    const std::vector<std::pair<int, int>> expected = {{1, 1},  {2, 1},   {3, 1},  {4, 1},
                                                       {5, 1},  {7, 1},   {8, 1},  {9, 2},
                                                       {50, 7}, {77, 10}, {99, 13}};
    for (const auto& [frames, tokens] : expected)
        check(q::post_cnn_length(frames) == tokens, "post-CNN length of " + std::to_string(frames));
    check(q::post_cnn_length(0) == 0, "an empty chunk keeps no token");
    // Token counts from the clip length alone: 23.315 s, 0.3125 s (padded to min_length), 1.0 s.
    check(q::audio_token_count(373040, frontend()) == 303 &&
              q::audio_token_count(5000, frontend()) == 7 &&
              q::audio_token_count(16000, frontend()) == 13 &&
              q::audio_token_count(12345, frontend()) == 10,
          "audio tokens from the sample count");
    check(q::post_cnn_length(100) == 13, "a full chunk keeps 13 tokens");

    // 2331 mel frames (the LibriSpeech fixture): 23 full chunks and 31 frames, 303 tokens.
    q::LogMel mel{std::vector<float>(128 * 2331, 1.0F), 128, 2331};
    const auto batch = q::encoder_batch(mel, frontend());
    check(batch.windows == 3, "24 chunks fill 3 windows of 8");
    check(batch.audio_tokens == 303, "the processor's placeholder count for 2331 frames");
    check(batch.chunk_tokens.size() == 24 && batch.chunk_tokens[22] == 13 &&
              batch.chunk_tokens[23] == 4,
          "only the last chunk is partial");
    check(batch.features.size() == 24U * 128U * 100U, "dense window grid");
    const auto last = [&](int m, int t) { return batch.features[(23U * 128U + m) * 100U + t]; };
    check(last(5, 30) == 1.0F && last(5, 31) == 0.0F && last(127, 99) == 0.0F,
          "frames past the clip are zero, as the processor pads them");

    // 77 frames: one partial chunk, padded to a window of 8 empty chunks.
    q::LogMel short_mel{std::vector<float>(128 * 77, 2.0F), 128, 77};
    const auto small = q::encoder_batch(short_mel, frontend());
    check(small.windows == 1 && small.audio_tokens == 10 && small.chunk_tokens[0] == 10 &&
              small.chunk_tokens[1] == 0 && small.chunk_tokens[7] == 0,
          "short clip occupies one window");
}

void test_power_spectrum() {
    std::vector<float> frame(400);
    std::uint32_t state = 3;
    for (auto& value : frame) {
        state = 1664525U * state + 1013904223U;
        value = static_cast<float>(static_cast<std::int32_t>(state >> 8) % 2001 - 1000) / 1000.0F;
    }
    const auto power = q::detail::power_spectrum(frame);
    double worst = 0.0;
    for (std::size_t k = 0; k <= 200; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t j = 0; j < 400; ++j) {
            const double angle =
                -2.0 * 3.14159265358979323846 * static_cast<double>((j * k) % 400) / 400.0;
            sum += static_cast<double>(frame[j]) *
                   std::complex<double>(std::cos(angle), std::sin(angle));
        }
        worst = std::max(worst, std::abs(power[k] - std::norm(sum)) / (std::norm(sum) + 1e-12));
    }
    check(worst < 1e-9,
          "Bluestein power spectrum equals the direct DFT, worst " + std::to_string(worst));
}

void test_text() {
    const std::vector<std::pair<std::u32string, std::u32string>> repetitions = {
        {std::u32string(25, U'a') + U"bc", U"abc"},
        {std::u32string(20, U'a') + U"bc", std::u32string(20, U'a') + U"bc"},
        {U"xy" +
             [] {
                 std::u32string s;
                 for (int i = 0; i < 25; ++i)
                     s += U"ab";
                 return s;
             }() +
             U"c",
         U"xyabc"},
        {U"ok " + std::u32string(30, U'哈') + U" end", U"ok 哈 end"},
        {std::u32string(22, U'\U0001F600') + U"x", U"\U0001F600x"},
        {[] {
             std::u32string s;
             for (int i = 0; i < 30; ++i)
                 s += U"word ";
             return s;
         }(),
         U"word "},
        {U"no repetition here", U"no repetition here"},
    };
    for (std::size_t i = 0; i < repetitions.size(); ++i)
        check(q::fix_repetitions(repetitions[i].first, 20) == repetitions[i].second,
              "repetition case " + std::to_string(i));

    const std::vector<std::pair<std::string, std::string>> transcripts = {
        {"language English<asr_text>Hello world.", "Hello world."},
        {"  language Chinese<asr_text> \xE4\xBD\xA0\xE5\xA5\xBD\xE3\x80\x80",
         "\xE4\xBD\xA0\xE5\xA5\xBD"},
        {"plain transcript without a tag", "plain transcript without a tag"},
        {"language None<asr_text>", ""},
        {"", ""},
        {"   ", ""},
        {"system\nassistant\nlanguage English<asr_text>after the assistant tag",
         "after the assistant tag"},
        {"language English<asr_text>" +
             [] {
                 std::string s;
                 for (int i = 0; i < 30; ++i)
                     s += "ha";
                 return s;
             }() +
             "!",
         "ha!"},
    };
    for (std::size_t i = 0; i < transcripts.size(); ++i)
        check(q::transcription_from_output(transcripts[i].first, "<asr_text>", 20) ==
                  transcripts[i].second,
              "transcript case " + std::to_string(i));

    const std::vector<std::pair<std::string, std::u32string>> utf8 = {
        {"\xE4\xBD", U"�"},
        {"\xFF\x41", U"�"
                     U"A"},
        {"\xED\xA0\x80", U"���"},
        {"\xF0\x9F\x98", U"�"},
        {"ok\xC3\xA9\xE2\x82\xAC", U"oké€"},
        {"\xF4\x90\x80\x80", U"����"},
        {"\xC0\xAF", U"��"},
    };
    for (std::size_t i = 0; i < utf8.size(); ++i)
        check(q::decode_utf8_replace(utf8[i].first) == utf8[i].second,
              "UTF-8 case " + std::to_string(i));
    check(q::encode_utf8(U"é€\U0001F600") == "\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80",
          "UTF-8 encoding of 2-, 3- and 4-byte characters");
}

std::vector<char> vocabulary_bytes() {
    // Ids: 0 "Hel", 1 "lo" , 2 special "<|im_end|>", 3 unknown, 4 first half of U+20AC, 5 rest.
    const std::vector<std::pair<std::uint8_t, std::string>> entries = {
        {0, "Hel"}, {0, "lo"}, {1, "<|im_end|>"}, {2, ""}, {0, "\xE2\x82"}, {0, "\xAC"}};
    std::vector<char> data = {'Q', '3', 'A', 'V'};
    const auto put = [&](const void* value, std::size_t size) {
        const auto* bytes = static_cast<const char*>(value);
        data.insert(data.end(), bytes, bytes + size);
    };
    const std::uint32_t version = 1;
    const auto count = static_cast<std::uint32_t>(entries.size());
    put(&version, 4);
    put(&count, 4);
    for (const auto& [flag, text] : entries) {
        const auto length = static_cast<std::uint16_t>(text.size());
        put(&flag, 1);
        put(&length, 2);
        put(text.data(), text.size());
    }
    return data;
}

void test_vocabulary() {
    const auto vocabulary = q::Vocabulary::parse(vocabulary_bytes());
    check(vocabulary.size() == 6, "vocabulary size");
    check(vocabulary.decode({0, 1, 2, 3}) == "Hello", "special and unknown ids are skipped");
    check(vocabulary.decode({4, 5}) == "\xE2\x82\xAC", "a character split over two tokens joins");
    check(vocabulary.decode({4}) == "\xEF\xBF\xBD", "a dangling partial character becomes U+FFFD");
    bool rejected = false;
    try {
        (void)vocabulary.decode({6});
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    check(rejected, "an id outside the vocabulary is rejected");
    auto broken = vocabulary_bytes();
    broken.pop_back();
    rejected = false;
    try {
        (void)q::Vocabulary::parse(broken);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "a truncated table is rejected");
}

void test_runtime_config() {
    const std::string json = R"({
        "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "num_mel_bins": 128,
        "min_samples": 8000, "chunk_frames": 100, "tokens_per_chunk": 13, "window_chunks": 8,
        "encoder_windows_per_launch": 8, "prefill_chunk_tokens": 1024, "audio_embed_size": 1024, "vocab_size": 151936, "num_layers": 28,
        "num_kv_heads": 8, "head_dim": 128, "max_cache_length": 4096, "cache_dtype": "bf16",
        "key_row_bucket": 64, "audio_token_id": 151676,
        "prompt_prefix_ids": [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669],
        "prompt_suffix_ids": [151670, 151645, 198, 151644, 77091, 198],
        "stop_token_ids": [151643, 151645],
        "languages": [{"code": "en", "name": "English", "ids": [11528, 6364, 151704]},
                      {"code": "yue", "name": "Cantonese", "ids": [11528, 72366, 151704]}],
        "asr_text_marker": "<asr_text>", "repetition_threshold": 20,
        "default_max_output_tokens": 512})";
    const auto config = q::parse_runtime_config(std::vector<char>(json.begin(), json.end()));
    check(config.cache_dtype == trtmc::DType::kBFloat16 && config.max_cache_length == 4096,
          "runtime.json decoder fields");
    check(&q::find_language(config, "en") == &config.languages[0] &&
              &q::find_language(config, "ENGLISH") == &config.languages[0] &&
              &q::find_language(config, "Yue") == &config.languages[1],
          "languages resolve by code or name, case-insensitively");
    bool rejected = false;
    try {
        (void)q::find_language(config, "xx");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "an unknown language is rejected");

    const auto ids = q::prompt_ids(config, 3, &config.languages[0]);
    const std::vector<std::int32_t> expected = {
        151644, 8948,   198,    151645, 198,    151644, 872, 198,   151669, 151676, 151676,
        151676, 151670, 151645, 198,    151644, 77091,  198, 11528, 6364,   151704};
    check(ids == expected, "prompt layout with a forced language");
    check(q::prompt_ids(config, 1, nullptr).size() == 16, "prompt without a forced language");
    check(q::key_rows(config, 1) == 64 && q::key_rows(config, 64) == 64 &&
              q::key_rows(config, 65) == 128 && q::key_rows(config, 4090) == 4096,
          "key rows round up to the bucket and stop at the capacity");
}

} // namespace

int main() {
    test_geometry();
    test_power_spectrum();
    check_log_mel(12345, 7, 77, 4423.647558152676, 1.401354193687439F, -0.598645806312561F,
                  {{0, 0, 0.85620892F},     {0, 1, 0.316575766F},    {0, 38, 0.100931168F},
                   {0, 76, 0.0350292921F},  {1, 0, 0.953773379F},    {1, 1, 0.414140165F},
                   {1, 38, 0.198495626F},   {1, 76, 0.132593751F},   {17, 0, 1.09582257F},
                   {17, 1, 1.2052511F},     {17, 38, 1.20376503F},   {17, 76, 1.20885301F},
                   {64, 0, 0.353634775F},   {64, 1, 0.242504001F},   {64, 38, 0.493529022F},
                   {64, 76, 0.389502108F},  {100, 0, 0.36492461F},   {100, 1, 0.475355685F},
                   {100, 38, 0.426216543F}, {100, 76, 0.411494672F}, {127, 0, 0.262794197F},
                   {127, 1, 0.435814679F},  {127, 38, 0.367391765F}, {127, 76, 0.362913668F}});
    // 5000 samples are zero-padded to 8000 first; the padding is audio, so frames 32-49 sit at the
    // floor (maximum - 8 + 4) / 4.
    check_log_mel(5000, 11, 50, 617.7397121191025, 1.40006422996521F, -0.59993577003479F,
                  {{0, 0, 0.888298273F},    {0, 1, 0.431471467F},    {0, 25, 0.33904916F},
                   {0, 49, -0.59993577F},   {1, 0, 0.985862732F},    {1, 1, 0.529035926F},
                   {1, 25, 0.436613619F},   {1, 49, -0.59993577F},   {17, 0, 1.09498668F},
                   {17, 1, 1.20245004F},    {17, 25, 1.20494556F},   {17, 49, -0.59993577F},
                   {64, 0, 0.509602785F},   {64, 1, 0.372945547F},   {64, 25, 0.414817035F},
                   {64, 49, -0.59993577F},  {100, 0, 0.321453094F},  {100, 1, 0.3528862F},
                   {100, 25, 0.405678988F}, {100, 49, -0.59993577F}, {127, 0, 0.3744452F},
                   {127, 1, 0.351815164F},  {127, 25, 0.393980861F}, {127, 49, -0.59993577F}});
    test_text();
    test_vocabulary();
    test_runtime_config();
    if (failures == 0)
        std::cout << "qwen3_asr runtime helpers: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
