/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Host-side VibeVoice-ASR pieces: audio front end, encoder chunk plan, argmax,
// prompt layout, transcript parsing and runtime.json validation.
#include "families/vibevoice_asr/runtime/audio_frontend.h"
#include "families/vibevoice_asr/runtime/engine_io.h"
#include "families/vibevoice_asr/runtime/host_plan.h"
#include "families/vibevoice_asr/runtime/prompt.h"
#include "families/vibevoice_asr/runtime/runtime_config.h"
#include "families/vibevoice_asr/runtime/transcript.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace trtmc::vibevoice_asr;
using trtmc::DType;

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

template <class Error, class Function>
void check_throws_as(Function&& function, const std::string& what, const char* message = nullptr) {
    try {
        function();
    } catch (const Error& error) {
        if (message != nullptr && std::string(error.what()).find(message) == std::string::npos)
            check(false, what + " threw another error: " + error.what());
        return;
    } catch (const std::exception& error) {
        check(false, what + " threw another exception type: " + error.what());
        return;
    }
    check(false, what + " should throw");
}

template <class Function>
void check_no_throw(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::exception& error) {
        check(false, what + " threw: " + error.what());
    }
}

// The largest |values[i] - reference(i)| over [begin, end); infinite when either side
// is not finite, which std::max would otherwise skip, or when [begin, end) does not lie
// inside `values`, so a short output fails instead of being read past its end.
template <class Reference>
double max_error(const std::vector<float>& values, std::size_t begin, std::size_t end,
                 Reference reference) {
    if (begin > end || end > values.size())
        return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        const double expected = reference(i);
        if (!std::isfinite(values[i]) || !std::isfinite(expected))
            return std::numeric_limits<double>::infinity();
        worst = std::max(worst, std::fabs(values[i] - expected));
    }
    return worst;
}

std::string utf8(std::uint32_t code_point) {
    std::string text;
    if (code_point < 0x80) {
        text += static_cast<char>(code_point);
    } else if (code_point < 0x800) {
        text += static_cast<char>(0xC0 | (code_point >> 6));
        text += static_cast<char>(0x80 | (code_point & 0x3F));
    } else {
        text += static_cast<char>(0xE0 | (code_point >> 12));
        text += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
        text += static_cast<char>(0x80 | (code_point & 0x3F));
    }
    return text;
}

// A byte-level BPE with one token per byte and the special tokens the prompt
// uses; enough to check the prompt layout. `merges` adds merged tokens (ids
// from 262) so a test can see where the pre-tokenizer splits.
std::string
byte_tokenizer_json(const nlohmann::json& normalizer = nullptr,
                    const std::vector<std::pair<std::string, std::string>>& merges = {}) {
    nlohmann::json vocab = nlohmann::json::object();
    std::vector<int> printable;
    for (int b = '!'; b <= '~'; ++b)
        printable.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b)
        printable.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b)
        printable.push_back(b);
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const bool direct = std::find(printable.begin(), printable.end(), b) != printable.end();
        vocab[utf8(direct ? static_cast<std::uint32_t>(b) : 256U + extra++)] = b;
    }
    const char* specials[] = {"<|endoftext|>",        "<|im_start|>",       "<|im_end|>",
                              "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"};
    nlohmann::json added = nlohmann::json::array();
    int id = 256;
    for (const char* special : specials) {
        vocab[special] = id;
        added.push_back({{"id", id++},
                         {"content", special},
                         {"special", true},
                         {"single_word", false},
                         {"lstrip", false},
                         {"rstrip", false},
                         {"normalized", false}});
    }
    nlohmann::json merge_rules = nlohmann::json::array();
    for (const auto& [left, right] : merges) {
        vocab[left + right] = id++;
        merge_rules.push_back({left, right});
    }
    const std::string regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
                              "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    nlohmann::json tokenizer = {
        {"added_tokens", added},
        {"normalizer", normalizer},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Split"},
             {"pattern", {{"Regex", regex}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}}}}},
        {"post_processor", nullptr},
        {"decoder",
         {{"type", "ByteLevel"},
          {"add_prefix_space", false},
          {"trim_offsets", false},
          {"use_regex", false}}},
        {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merge_rules}}}};
    return tokenizer.dump();
}

std::vector<std::int32_t> bytes_of(const std::string& text) {
    std::vector<std::int32_t> ids;
    for (const unsigned char byte : text)
        ids.push_back(byte);
    return ids;
}

nlohmann::json runtime_json() {
    return {{"tensor_parallel_size", 1},
            {"sample_rate", 24000},
            {"hop_length", 3200},
            {"encoder_chunk_frames", 450},
            {"encoder_context_frames", 58},
            {"normalize_target_dbfs", -25.0},
            {"normalize_eps", 1e-6},
            {"hidden_size", 3584},
            {"num_layers", 28},
            {"num_kv_heads", 4},
            {"head_dim", 128},
            {"vocab_size", 262},
            {"max_sequence_length", 4096},
            {"max_prefill_rows", 256},
            {"system_prompt", "You are a helpful assistant."},
            {"transcription_keys", "Start time, End time, Speaker ID, Content"},
            {"token_ids",
             {{"eos", 256},
              {"im_start", 257},
              {"im_end", 258},
              {"speech_start", 259},
              {"speech_end", 260},
              {"speech_pad", 261}}},
            {"tokenizer_checks", {{{"text", "user"}, {"ids", {117, 115, 101, 114}}}}}};
}

void test_downmix() {
    const std::vector<float> stereo{1.0F, 3.0F, -2.0F, 2.0F};
    const auto mono = downmix(stereo.data(), 2, 2);
    check(mono == std::vector<float>({2.0F, 0.0F}), "downmix averages the channels");
    const std::vector<float> bad{0.0F, std::numeric_limits<float>::quiet_NaN()};
    check_throws_as<std::invalid_argument>([&] { downmix(bad.data(), 2, 1); },
                                           "downmix of a NaN sample", "only finite samples");
}

void test_resample() {
    const std::vector<float> same{0.1F, 0.2F};
    check(resample(same, 24000, 24000) == same, "a 24 kHz input passes through");
    const std::pair<int, std::size_t> cases[] = {{16000, 16001}, {44100, 44101}, {48000, 999}};
    for (const auto& [rate, count] : cases) {
        const std::vector<float> input(count, 0.0F);
        const auto expected = (count * 24000 + rate - 1) / rate;
        check(resample(input, rate, 24000).size() == expected,
              "resampled length is ceil(n * 24000 / rate) for " + std::to_string(rate));
    }
    // A 440 Hz tone keeps its shape away from the edges.
    const double pi = 3.14159265358979323846;
    std::vector<float> tone(16000);
    for (std::size_t i = 0; i < tone.size(); ++i)
        tone[i] = static_cast<float>(0.5 * std::sin(2.0 * pi * 440.0 * i / 16000.0));
    const auto out = resample(tone, 16000, 24000);
    check(out.size() == 24000, "one second at 16 kHz is 24000 samples at 24 kHz");
    const double worst = max_error(out, 2400, out.size() - 2400, [&](std::size_t i) {
        return 0.5 * std::sin(2.0 * pi * 440.0 * i / 24000.0);
    });
    check(worst < 1e-3, "16 kHz to 24 kHz tone error " + std::to_string(worst));

    // Common rates keep the exact phase table: the default equals an unbounded table.
    check(out == resample(tone, 16000, 24000, std::numeric_limits<std::size_t>::max()),
          "16 kHz keeps the exact phase table");
    // 192001 Hz is coprime with 24 kHz: 24000 output phases, whose exact table would
    // hold about 12 million coefficients (94 MiB). The default takes the bounded,
    // interpolated path and keeps the tone and the ceiling length.
    const int odd_rate = 192001;
    std::vector<float> odd(static_cast<std::size_t>(odd_rate / 5));
    for (std::size_t i = 0; i < odd.size(); ++i)
        odd[i] = static_cast<float>(0.5 * std::sin(2.0 * pi * 440.0 * i / odd_rate));
    const auto odd_out = resample(odd, odd_rate, 24000);
    check(odd_out == resample(odd, odd_rate, 24000, 0), "192001 Hz takes the bounded path");
    ResampleStats odd_work;
    resample(odd, odd_rate, 24000, kMaxResampleTable, &odd_work);
    check(odd_work.interpolated && odd_work.table_coefficients <= kMaxResampleTable,
          "the default budget interpolates the kernel at 192001 Hz");
    ResampleStats common_work;
    resample(tone, 16000, 24000, kMaxResampleTable, &common_work);
    check(!common_work.interpolated && common_work.table_coefficients <= kMaxResampleTable,
          "16 kHz uses the exact phase table within the budget");
    check(odd_out.size() == (odd.size() * 24000 + odd_rate - 1) / odd_rate,
          "resampled length is ceil(n * 24000 / 192001)");
    const double odd_worst = max_error(odd_out, 600, odd_out.size() - 600, [&](std::size_t i) {
        return 0.5 * std::sin(2.0 * pi * 440.0 * i / 24000.0);
    });
    check(odd_worst < 1e-3, "192001 Hz to 24 kHz tone error " + std::to_string(odd_worst));
    // A larger coprime rate: an exact phase table would hold 128 million coefficients.
    const std::vector<float> extreme(4000, 0.25F);
    check(resample(extreme, 2000003, 24000).size() == (4000U * 24000U + 2000003U - 1U) / 2000003U,
          "resampled length is ceil(n * 24000 / 2000003)");
    // The interpolated kernel agrees with the exact phase table (44101 Hz: 24000 phases).
    std::vector<float> signal(4410);
    for (std::size_t i = 0; i < signal.size(); ++i)
        signal[i] = static_cast<float>(0.4 * std::sin(0.37 * i) + 0.3 * std::cos(0.011 * i * i));
    const auto exact = resample(signal, 44101, 24000, std::numeric_limits<std::size_t>::max());
    const auto interpolated = resample(signal, 44101, 24000, 0);
    const double difference =
        exact.size() == interpolated.size()
            ? max_error(interpolated, 0, interpolated.size(),
                        [&](std::size_t i) { return static_cast<double>(exact[i]); })
            : std::numeric_limits<double>::infinity();
    check(difference < 1e-6, "interpolated and exact taps differ by " + std::to_string(difference));
    // The coefficient budget picks the path: 44101 Hz's exact table is over the default
    // budget, so the default call gives the interpolated output, not the exact one.
    const auto by_default = resample(signal, 44101, 24000);
    check(by_default == interpolated && by_default != exact,
          "44101 Hz takes the interpolated path by default");
    check(resample(tone, 16000, 24000, 0).size() == out.size(), "both paths give the same length");
    check(!(max_error({std::numeric_limits<float>::quiet_NaN()}, 0, 1,
                      [](std::size_t) { return 0.0; }) < 1.0),
          "a NaN sample counts as an error");

    // A resampled sample beyond the float range is rejected, not passed on as infinity:
    // between two FLT_MAX samples at 12 kHz the 24 kHz output overshoots, on both paths.
    const std::vector<float> loud(2, std::numeric_limits<float>::max());
    check_throws_as<std::invalid_argument>([&] { resample(loud, 12000, 24000); },
                                           "FLT_MAX samples at 12 kHz");
    check_throws_as<std::invalid_argument>([&] { resample(loud, 12000, 24000, 0); },
                                           "FLT_MAX samples at 12 kHz, interpolated");

    // Only the taps on the input are visited: at 2,147,483,647 Hz an output's kernel
    // spans 5,726,624 taps, one of which falls on a one-sample clip.
    const auto one = input_taps(-2863311, 5726624, 1);
    check(one.begin == 2863311 && one.end == 2863312, "one tap of a one-sample clip");
    const auto inside = input_taps(10, 5, 100);
    check(inside.begin == 0 && inside.end == 5, "taps inside the input");
    const auto ahead = input_taps(-10, 5, 100);
    check(ahead.begin == ahead.end, "taps before the input");
    const auto behind = input_taps(200, 5, 100);
    check(behind.begin == behind.end, "taps after the input");
    // The loops apply only taps that fall on the input: a one-sample clip at 2147483647 Hz
    // costs one tap, not the 5,726,624 of the kernel, and two samples at 8 kHz at most two
    // taps per output sample on the exact-table path.
    ResampleStats high_work;
    const auto high =
        resample(std::vector<float>{0.5F}, 2147483647, 24000, kMaxResampleTable, &high_work);
    check(high.size() == 1 && std::isfinite(high[0]), "a one-sample clip at 2147483647 Hz");
    check(high_work.taps == 1,
          "one tap for a one-sample clip at 2147483647 Hz, not " + std::to_string(high_work.taps));
    ResampleStats short_work;
    const auto short_out =
        resample(std::vector<float>{0.5F, -0.5F}, 8000, 24000, kMaxResampleTable, &short_work);
    check(!short_work.interpolated &&
              short_work.taps <= 2 * static_cast<std::int64_t>(short_out.size()),
          "at most two taps per output for a two-sample clip");

    // resampled_length gives resample()'s length without resampling, so a request
    // is sized before its audio work: 100000 samples at 1 Hz would be 2.4e9.
    for (const auto& [rate, count] : cases)
        check(resampled_length(count, rate, 24000) ==
                  resample(std::vector<float>(count, 0.0F), rate, 24000).size(),
              "resampled_length matches resample at " + std::to_string(rate));
    check(resampled_length(odd.size(), odd_rate, 24000) == odd_out.size(),
          "resampled_length matches resample at 192001 Hz");
    check(resampled_length(24000, 24000, 24000) == 24000, "equal rates keep the length");
    check(resampled_length(100000, 1, 24000) == 2400000000ULL, "100000 samples at 1 Hz");
    check_throws_as<std::invalid_argument>(
        [] { resampled_length(std::numeric_limits<std::size_t>::max(), 1, 24000); },
        "a resampled length beyond int64");
    check_throws_as<std::invalid_argument>([] { resampled_length(10, 0, 24000); },
                                           "a zero sample rate");
}

void test_normalize() {
    std::vector<float> quiet(1000);
    for (std::size_t i = 0; i < quiet.size(); ++i)
        quiet[i] = 0.01F * static_cast<float>(static_cast<int>(i % 7) - 3);
    normalize_loudness(quiet, -25.0, 1e-6);
    double energy = 0.0;
    for (float value : quiet)
        energy += static_cast<double>(value) * value;
    const double rms = std::sqrt(energy / quiet.size());
    check(std::fabs(rms - std::pow(10.0, -25.0 / 20.0)) < 1e-5, "RMS normalised to -25 dBFS");
    check(quiet[0] < 0.0F && quiet[6] > 0.0F, "normalization keeps the sign of each sample");
    // RMS 0.01 scales the impulse to 5.6, above the clipping limit.
    std::vector<float> impulse(10000, 0.0F);
    impulse[50] = 1.0F;
    normalize_loudness(impulse, -25.0, 1e-6);
    const bool finite = std::all_of(impulse.begin(), impulse.end(),
                                    [](float value) { return std::isfinite(value); });
    const float peak = *std::max_element(impulse.begin(), impulse.end());
    check(finite && peak <= 1.0F && peak > 0.999F, "the clipping guard divides by the peak");
    // Silence stays finite zeros for the bundle's eps and the smallest eps runtime.json accepts.
    for (const double eps : {1e-6, static_cast<double>(std::numeric_limits<float>::min())}) {
        std::vector<float> silence(64, 0.0F);
        normalize_loudness(silence, -25.0, eps);
        check(
            std::all_of(silence.begin(), silence.end(), [](float value) { return value == 0.0F; }),
            "silence stays zero with eps " + std::to_string(eps));
    }
}

void test_chunks_and_argmax() {
    const auto chunks = plan_encoder_chunks(1000, 450, 60);
    check(chunks.size() == 3, "three encoder chunks");
    if (chunks.size() == 3) {
        check(chunks[0].first_frame == 0 && chunks[0].frames == 450 &&
                  chunks[0].context_frames == 0,
              "the first chunk has no left context");
        check(chunks[1].first_frame == 450 && chunks[1].frames == 450 &&
                  chunks[1].context_frames == 60,
              "the second chunk carries the receptive field");
        check(chunks[2].first_frame == 900 && chunks[2].frames == 100 &&
                  chunks[2].context_frames == 60,
              "the last chunk ends at the last frame");
    }
    const auto single = plan_encoder_chunks(3, 450, 60);
    check(single.size() == 1 && single[0].frames == 3, "a short clip is one chunk");
    check_throws_as<std::invalid_argument>([] { plan_encoder_chunks(0, 450, 60); },
                                           "an empty chunk plan", "positive sizes");
    const float tied[] = {0.5F, 2.0F, 1.0F, 2.0F};
    check(argmax(tied, 4) == 1, "argmax keeps the lowest index of a tie");
    // A prompt of max_sequence_length - 2 tokens leaves room for two output tokens:
    // generation goes on after the first and stops after the second.
    const std::int64_t unlimited = std::numeric_limits<std::int64_t>::max();
    check(!generation_complete(4094, 1, unlimited, 4096), "one output token leaves room");
    check(generation_complete(4094, 2, unlimited, 4096),
          "prompt plus output stays within max_sequence_length");
    check(generation_complete(10, 3, 3, 4096), "max_new_tokens stops generation");
    check(!generation_complete(10, 2, 3, 4096), "generation continues below both limits");
    // An explicit max_output_tokens must fit with the prompt instead of being cut
    // short by the sequence limit without notice.
    check_no_throw([] { check_output_budget(4000, 96, 4096); },
                   "prompt plus max_output_tokens filling max_sequence_length");
    check_throws_as<std::invalid_argument>([] { check_output_budget(4000, 97, 4096); },
                                           "prompt plus max_output_tokens one past the limit");
    check_throws_as<std::invalid_argument>(
        [] { check_output_budget(4000, std::numeric_limits<std::int64_t>::max(), 4096); },
        "the largest max_output_tokens");
}

void test_transcript() {
    const auto full = parse_transcript(
        "assistant\n[{\"Start\":0.0,\"End\":2.5,\"Speaker\":0,\"Content\":\"Hello there.\"},"
        "{\"Start\":2.5,\"End\":4,\"Speaker\":1,\"Content\":\"Hi.\"}]\n");
    check(full.structured && full.text == "Hello there. Hi.", "contents are joined");
    check(full.segments.size() == 2 && full.segments[1].start_seconds == 2.5 &&
              full.segments[1].end_seconds == 4.0 && full.segments[1].text == "Hi.",
          "segments keep their timestamps");
    const auto truncated = parse_transcript(
        "[{\"Start\":0,\"End\":1,\"Speaker\":0,\"Content\":\"one\"},{\"Start\":1,\"End\":2,\"Con");
    check(truncated.structured && truncated.segments.size() == 1 && truncated.text == "one",
          "a truncated list keeps its complete segments");
    const auto aliased = parse_transcript(
        "[{\"Start time\":1.25,\"End time\":2,\"Speaker ID\":0,\"Content\":\"x\"}]");
    check(aliased.segments.size() == 1 && aliased.segments[0].start_seconds == 1.25,
          "official key aliases are accepted");
    const auto untimed = parse_transcript("[{\"Speaker\":0,\"Content\":\"no times\"}]");
    check(untimed.structured && untimed.text == "no times" && untimed.segments.empty(),
          "untimed segments give text without invented timestamps");
    const auto bad_time = parse_transcript("[{\"Start\":\"a\",\"End\":1,\"Content\":\"x\"}]");
    check(!bad_time.structured && bad_time.segments.empty(), "non-numeric timestamps are rejected");
    const auto plain = parse_transcript("  assistant\n just words \n");
    check(!plain.structured && plain.text == "just words", "plain output is stripped");
    const auto word = parse_transcript("assistantship is demanding");
    check(!word.structured && word.text == "assistantship is demanding",
          "a word that starts like the role header is kept");
    check(parse_transcript("assistant").text.empty(), "a bare role header");
    // A truncated list keeps only its complete top-level objects: a segment inside an
    // unclosed nested list is not promoted, and a cut-off second object leaves the first.
    const auto nested = parse_transcript("[{\"Start\":0,\"End\":1,\"Content\":\"one\"},"
                                         "[{\"Start\":1,\"End\":2,\"Content\":\"nested\"}");
    check(nested.structured && nested.segments.size() == 1 && nested.text == "one",
          "a segment inside an unclosed nested list");
    const auto cut =
        parse_transcript("[{\"Start\":0,\"End\":1,\"Content\":\"one\"},{\"Start\":1,\"En");
    check(cut.structured && cut.segments.size() == 1 && cut.text == "one",
          "a cut-off second segment");
    // A number that overflows a double is a parse error, so the output is plain text.
    const auto infinite = parse_transcript("[{\"Start\":1e999,\"End\":1,\"Content\":\"a\"}]");
    check(!infinite.structured && infinite.segments.empty(), "an infinite timestamp");
    const auto no_comma = parse_transcript("[{\"Start\":0,\"End\":1,\"Content\":\"one\"}"
                                           "{\"Start\":1,\"End\":2,\"Content\":\"two\"}");
    check(no_comma.structured && no_comma.segments.size() == 1 && no_comma.text == "one",
          "recovery stops at a missing comma");
    // Backticks inside a fenced segment are content: the closing fence ends the document.
    const auto ticks =
        parse_transcript("```json\n[{\"Start\":0,\"End\":1,\"Content\":\"say ``` now\"}]\n```");
    check(ticks.structured && ticks.segments.size() == 1 && ticks.text == "say ``` now",
          "backticks inside a fenced segment");
    const auto open_fence =
        parse_transcript("```json\n[{\"Start\":0,\"End\":1,\"Content\":\"a\"},{\"Start\":1");
    check(open_fence.structured && open_fence.segments.size() == 1 && open_fence.text == "a",
          "a fenced list cut before its end");
    // An untimed segment keeps its text; the timed ones keep their times.
    const auto mixed =
        parse_transcript("[{\"Start\":0,\"End\":1,\"Content\":\"one\"},{\"Content\":\"two\"}]");
    check(mixed.structured && mixed.text == "one two" && mixed.segments.size() == 1 &&
              mixed.segments[0].text == "one" && mixed.segments[0].end_seconds == 1.0,
          "a timed segment next to an untimed one");
    // Any whitespace ends the role header, as a line break does.
    for (const char* separator : {" ", "\t", "\r\n"}) {
        const auto spaced = parse_transcript(std::string("assistant") + separator +
                                             "[{\"Start\":0,\"End\":1,\"Content\":\"a\"}]");
        check(spaced.structured && spaced.segments.size() == 1 && spaced.text == "a",
              std::string("a role header followed by ") + nlohmann::json(separator).dump());
    }
    const auto fenced =
        parse_transcript("assistant\n```json\n[{\"Start\":0,\"End\":1,\"Content\":\"a\"}]\n```\n");
    check(fenced.structured && fenced.segments.size() == 1 && fenced.text == "a",
          "a fenced list is parsed");
    const auto inner_fence =
        parse_transcript("[{\"Start\":0,\"End\":1,\"Content\":\"say ```json now\"}]");
    check(inner_fence.structured && inner_fence.segments.size() == 1 &&
              inner_fence.text == "say ```json now",
          "a fence marker inside a segment is content");
    const auto single = parse_transcript("{\"Start\":0.5,\"End\":1,\"Content\":\"b\"}");
    check(single.structured && single.segments.size() == 1 &&
              single.segments[0].start_seconds == 0.5,
          "a single segment object is parsed");
}

void test_runtime_config() {
    const auto config = parse_runtime_config(runtime_json().dump());
    check(config.token_ids.speech_pad == 261 && config.tokenizer_checks.size() == 1,
          "runtime.json parses");
    auto parallel = runtime_json();
    parallel["tensor_parallel_size"] = 2;
    check_throws_as<std::runtime_error>([&] { parse_runtime_config(parallel.dump()); },
                                        "tensor parallel runtime.json", "runs on one GPU");
    auto outside = runtime_json();
    outside["token_ids"]["eos"] = 262;
    check_throws_as<std::runtime_error>([&] { parse_runtime_config(outside.dump()); },
                                        "an out-of-vocabulary token", "out of vocabulary");
    auto unchecked = runtime_json();
    unchecked["tokenizer_checks"] = nlohmann::json::array();
    check_throws_as<std::runtime_error>([&] { parse_runtime_config(unchecked.dump()); },
                                        "missing tokenizer checks", "is inconsistent");
    // Probe ids are integers in [0, vocab_size): no truncation of 117.9 to 117.
    for (const auto& ids :
         {nlohmann::json{117.9}, nlohmann::json{-1}, nlohmann::json{262}, nlohmann::json{"117"}}) {
        auto probe = runtime_json();
        probe["tokenizer_checks"][0]["ids"] = ids;
        check_throws_as<std::runtime_error>([&] { parse_runtime_config(probe.dump()); },
                                            "a tokenizer check id of " + ids.dump(),
                                            "is not a vocabulary id");
    }
    // eps must keep the float gain of silent input finite: 1e-300 is 0 as a float,
    // 1e-40 is subnormal and overflows the gain.
    for (const char* eps : {"1e-300", "1e-40"}) {
        auto tiny = runtime_json();
        tiny["normalize_eps"] = std::strtod(eps, nullptr);
        check_throws_as<std::runtime_error>([&] { parse_runtime_config(tiny.dump()); },
                                            std::string("normalize_eps ") + eps, "is inconsistent");
    }
    auto smallest = runtime_json();
    smallest["normalize_eps"] = static_cast<double>(std::numeric_limits<float>::min());
    check(parse_runtime_config(smallest.dump()).normalize_eps > 0.0,
          "the smallest normal float eps is accepted");
    // A silence gain that rounds to zero as a float would silence every input.
    auto zero_gain = runtime_json();
    zero_gain["normalize_target_dbfs"] = -750.0;
    zero_gain["normalize_eps"] = 3e38;
    check_throws_as<std::runtime_error>([&] { parse_runtime_config(zero_gain.dump()); },
                                        "a silence gain of -750 dBFS over eps 3e38");
    // Loud input must not be silenced either: the float gain stays above zero for the
    // largest finite RMS (-750 dBFS rounds it to zero; eps 4e36 overflows rms + eps).
    for (const auto& [target, eps] : {std::pair{-750.0, 1e-3}, std::pair{-25.0, 4e36}}) {
        auto loud = runtime_json();
        loud["normalize_target_dbfs"] = target;
        loud["normalize_eps"] = eps;
        check_throws_as<std::runtime_error>([&] { parse_runtime_config(loud.dump()); },
                                            "a gain that rounds to zero for loud input at " +
                                                std::to_string(target) + " dBFS, eps " +
                                                std::to_string(eps),
                                            "is inconsistent");
    }
    // Token ids are decoder rows in [0, vocab_size): row 0 is a valid id.
    auto row_zero = runtime_json();
    row_zero["token_ids"]["eos"] = 0;
    std::int32_t row_zero_eos = -1;
    check_no_throw([&] { row_zero_eos = parse_runtime_config(row_zero.dump()).token_ids.eos; },
                   "a token id of 0");
    check(row_zero_eos == 0, "a token id of 0 is kept");
    // An eps above FLT_MAX has no float value, and a target whose float amplitude
    // underflows to zero would silence every input.
    for (const auto& [field, value] :
         {std::pair{"normalize_eps", 1e39}, std::pair{"normalize_target_dbfs", -1000.0}}) {
        auto bad = runtime_json();
        bad[field] = value;
        check_throws_as<std::runtime_error>([&] { parse_runtime_config(bad.dump()); },
                                            std::string(field) + " " + std::to_string(value));
    }
    // runtime.json defects are bundle defects, reported as std::runtime_error like
    // the sibling families' and the core loader's.
    auto fractional = runtime_json();
    fractional["tokenizer_checks"][0]["ids"] = nlohmann::json::array({117.9});
    auto parallel_bad = runtime_json();
    parallel_bad["tensor_parallel_size"] = 2;
    auto missing = runtime_json();
    missing.erase("hop_length");
    auto mistyped = runtime_json();
    mistyped["normalize_eps"] = "small";
    for (const auto& [bad, what] :
         {std::pair{fractional, "a fractional probe id"},
          std::pair{parallel_bad, "tensor parallel"}, std::pair{missing, "a missing field"},
          std::pair{mistyped, "a string eps"}})
        check_throws_as<std::runtime_error>([&] { parse_runtime_config(bad.dump()); },
                                            std::string("runtime.json with ") + what);
}

void test_tokenizer() {
    constexpr std::int32_t kVocab = 264; // bytes, specials and two merged tokens
    // Unicode NFC, the normalizer Qwen2.5's tokenizer.json declares: e + U+0301 is U+00E9.
    const auto nfc_json = byte_tokenizer_json({{"type", "NFC"}});
    const auto nfc = CreateBpeTokenizer(nfc_json.data(), nfc_json.size(), false, kVocab);
    check(nfc->encode("Cafe\xCC\x81") == bytes_of("Caf\xC3\xA9"), "NFC composes e + U+0301");
    check(nfc->encode("Caf\xC3\xA9") == bytes_of("Caf\xC3\xA9"), "NFC keeps a composed character");
    // \p{N} digits are their own pre-tokens: the merges a+U+0661 (bytes D9 A1) and
    // U+0661+U+0662 must not apply across those boundaries.
    const std::string arabic_one = "\xD9\xA1"; // U+0661 ARABIC-INDIC DIGIT ONE
    const std::string arabic_two = "\xD9\xA2"; // U+0662 ARABIC-INDIC DIGIT TWO
    const auto split_json =
        byte_tokenizer_json(nullptr, {{"a", utf8(0xD9)}, {utf8(0xA1), utf8(0xD9)}});
    const auto split = CreateBpeTokenizer(split_json.data(), split_json.size(), false, kVocab);
    check(split->encode("a" + arabic_one) == bytes_of("a" + arabic_one),
          "a letter and an Arabic-Indic digit are separate pre-tokens");
    check(split->encode(arabic_one + arabic_two) == bytes_of(arabic_one + arabic_two),
          "each Arabic-Indic digit is its own pre-token");
    // Text that is not UTF-8 is rejected, with or without NFC (which would otherwise
    // replace the bytes depending on the rest of the text).
    for (const char* text : {"a\xF0\x9F", "\xE4\xB8",
                             "a\xFF"
                             "b",
                             "a\xFF"
                             "b e\xCC\x81"}) {
        check_throws_as<std::invalid_argument>([&] { split->encode(text); },
                                               "encoding invalid UTF-8", "valid UTF-8");
        check_throws_as<std::invalid_argument>([&] { nfc->encode(text); },
                                               "encoding invalid UTF-8 with NFC", "valid UTF-8");
    }
    // ByteLevel decoding joins all bytes, then replaces what is not UTF-8 (U+FFFD).
    const std::string replacement = "\xEF\xBF\xBD";
    check(split->decode({0xC3, 0xA9}) == "\xC3\xA9", "a character split over two tokens");
    check(split->decode({0x61, 0xC3}) == "a" + replacement, "a cut-off last character");
    check(split->decode({0xE4, 0xB8}) == replacement, "one replacement per maximal subpart");
    check(split->decode({0x61, 0xFF, 0x62}) == "a" + replacement + "b", "an invalid byte");
    check(split->decode({0xC0, 0x80}) == replacement + replacement, "an overlong encoding");
    // An added token without an integer id or a string content is rejected with the
    // parser's own error before any read.
    for (const char* field : {"id", "content"}) {
        auto broken = nlohmann::json::parse(byte_tokenizer_json());
        broken["added_tokens"][0].erase(field);
        const auto text = broken.dump();
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
            std::string("an added token without ") + field,
            "needs a string content and an integer id");
    }
    // Added-token ids are bounded by the decoder vocabulary before the vocabulary grows.
    for (const std::int64_t id :
         {std::int64_t{kVocab}, std::int64_t{2147483647}, std::int64_t{-1}}) {
        auto stray = nlohmann::json::parse(byte_tokenizer_json());
        stray["added_tokens"][0]["id"] = id;
        const auto text = stray.dump();
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
            "an added token id of " + std::to_string(id), "outside the decoder vocabulary");
    }
    const auto json = byte_tokenizer_json();
    check_throws_as<std::invalid_argument>(
        [&] { CreateBpeTokenizer(json.data(), json.size(), false, 0); },
        "a tokenizer without a decoder vocabulary", "decoder vocabulary size");
    // Base-vocabulary ids are decoder rows too: the merged tokens 262 and 263 lie
    // outside a decoder vocabulary of 262, and an id must be an integer.
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(split_json.data(), split_json.size(), false, 262); },
        "a merged token id at the decoder vocabulary");
    auto fractional = nlohmann::json::parse(byte_tokenizer_json());
    fractional["model"]["vocab"]["a"] = 97.5;
    const auto fractional_text = fractional.dump();
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(fractional_text.data(), fractional_text.size(), false, kVocab); },
        "a fractional vocabulary id");
    // A normalizer or decoder the tokenizer does not implement rejects the bundle
    // instead of being skipped.
    for (
        const char* normalizer :
        {R"({"type": "NFKC"})",
         R"({"type": "Replace", "pattern": {"String": "\u2019"}, "content": "'"})",
         R"({"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "StripAccents"}]})",
         R"({"type": "Prepend", "prepend": "\u2581"})",
         R"({"type": "Sequence", "normalizers": [{"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "\u2581"}]}]})"}) {
        const auto text = byte_tokenizer_json(nlohmann::json::parse(normalizer));
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
            std::string("normalizer ") + normalizer);
    }
    const auto sequence_nfc = byte_tokenizer_json(
        nlohmann::json::parse(R"({"type": "Sequence", "normalizers": [{"type": "NFC"}]})"));
    check_no_throw(
        [&] { CreateBpeTokenizer(sequence_nfc.data(), sequence_nfc.size(), false, kVocab); },
        "a Sequence of NFC");
    for (
        const char* decoder :
        {R"({"type": "WordPiece"})",
         R"({"type": "Sequence", "decoders": [{"type": "Strip", "content": " ", "start": 2, "stop": 0}]})",
         R"({"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"Regex": "x"}, "content": "y"}]})"}) {
        auto with_decoder = nlohmann::json::parse(byte_tokenizer_json());
        with_decoder["decoder"] = nlohmann::json::parse(decoder);
        const auto text = with_decoder.dump();
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
            std::string("decoder ") + decoder);
    }
    // A token with a character outside the byte-level alphabet decodes as written, as the
    // reference's ByteLevel decoder does: here a non-special added token, "e-acute cat".
    const std::string written = "\xC3\xA9\xE7\x8C\xAB";
    auto with_added = nlohmann::json::parse(byte_tokenizer_json());
    with_added["added_tokens"].push_back({{"id", 262},
                                          {"content", written},
                                          {"special", false},
                                          {"single_word", false},
                                          {"lstrip", false},
                                          {"rstrip", false},
                                          {"normalized", false}});
    const auto added_text = with_added.dump();
    const auto added = CreateBpeTokenizer(added_text.data(), added_text.size(), false, kVocab);
    check(added->decode({262}) == written, "an added token outside the byte alphabet");
    check(added->decode({0x61, 262, 0xC3, 0xA9}) == "a" + written + "\xC3\xA9",
          "an added token between byte tokens");
    // Only Qwen2.5's layout is accepted: another pre-tokenizer, normalizer, BPE option or
    // added-token option would be tokenized differently from the reference without notice.
    auto own_regex = nlohmann::json::parse(byte_tokenizer_json());
    own_regex["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = true;
    auto other_split = nlohmann::json::parse(byte_tokenizer_json());
    other_split["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\s+";
    auto direct = nlohmann::json::parse(byte_tokenizer_json());
    direct["pre_tokenizer"] = nlohmann::json::parse(
        R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})");
    auto metaspace = nlohmann::json::parse(byte_tokenizer_json());
    metaspace["pre_tokenizer"] =
        nlohmann::json::parse(R"({"type": "Metaspace", "replacement": "\u2581"})");
    auto lowercase = nlohmann::json::parse(byte_tokenizer_json());
    lowercase["normalizer"] = nlohmann::json::parse(R"({"type": "Lowercase"})");
    auto byte_fallback = nlohmann::json::parse(byte_tokenizer_json());
    byte_fallback["model"]["byte_fallback"] = true;
    auto ignore_merges = nlohmann::json::parse(byte_tokenizer_json());
    ignore_merges["model"]["ignore_merges"] = true;
    auto lstrip = nlohmann::json::parse(byte_tokenizer_json());
    lstrip["added_tokens"][0]["lstrip"] = true;
    auto normalized = nlohmann::json::parse(byte_tokenizer_json());
    normalized["added_tokens"][0]["normalized"] = true;
    // Settings with which the reference would change the ids: truncation, padding, a
    // post-processor that adds tokens, an unknown token for bytes missing from the vocabulary.
    auto truncation = nlohmann::json::parse(byte_tokenizer_json());
    truncation["truncation"] = nlohmann::json::parse(
        R"({"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0})");
    auto padding = nlohmann::json::parse(byte_tokenizer_json());
    padding["padding"] = nlohmann::json::parse(
        R"({"strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": null, "pad_id": 256, "pad_type_id": 0, "pad_token": "<|endoftext|>"})");
    auto template_post = nlohmann::json::parse(byte_tokenizer_json());
    template_post["post_processor"] = nlohmann::json::parse(
        R"({"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}], "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}], "special_tokens": {"<|endoftext|>": {"id": "<|endoftext|>", "ids": [256], "tokens": ["<|endoftext|>"]}}})");
    auto unknown = nlohmann::json::parse(byte_tokenizer_json());
    unknown["model"]["unk_token"] = "<|endoftext|>";
    for (const auto& [layout, what] :
         {std::pair{own_regex, "ByteLevel with its own regex"},
          std::pair{other_split, "another Split pattern"},
          std::pair{direct, "a direct ByteLevel pre-tokenizer"},
          std::pair{metaspace, "a Metaspace pre-tokenizer"},
          std::pair{lowercase, "a Lowercase normalizer"},
          std::pair{byte_fallback, "BPE byte_fallback"},
          std::pair{ignore_merges, "BPE ignore_merges"}, std::pair{lstrip, "an lstrip added token"},
          std::pair{normalized, "a normalized added token"}, std::pair{truncation, "truncation"},
          std::pair{padding, "padding"},
          std::pair{template_post, "a post-processor that adds tokens"},
          std::pair{unknown, "an unknown token"}}) {
        const auto text = layout.dump();
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
            std::string("a tokenizer with ") + what, "is not supported");
    }
    // Qwen2.5's own ByteLevel post-processor adds no token, and a null option is its default.
    auto byte_level_post = nlohmann::json::parse(byte_tokenizer_json());
    byte_level_post["post_processor"] = nlohmann::json::parse(
        R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})");
    auto null_option = nlohmann::json::parse(byte_tokenizer_json());
    null_option["model"]["byte_fallback"] = nullptr;
    for (const auto& [layout, what] : {std::pair{byte_level_post, "a ByteLevel post-processor"},
                                       std::pair{null_option, "a null byte_fallback"}}) {
        const auto text = layout.dump();
        check_no_throw([&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
                       std::string("a tokenizer with ") + what);
    }
    // The merge table must read the same in the batch merge and in the reference's
    // one-merge-at-a-time BPE: an operand that only a later-ranked merge makes, an entry
    // that is not one pair, a pair or result outside the vocabulary, or two entry formats
    // in one table are rejected.
    const auto later = byte_tokenizer_json(nullptr, {{"ab", "a"}, {"a", "b"}});
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(later.data(), later.size(), false, kVocab); },
        "a merge whose operand only a later merge makes", "merge");
    const auto merged = nlohmann::json::parse(byte_tokenizer_json(nullptr, {{"a", "b"}}));
    for (const char* merges : {R"([["a"]])", R"([["a", "b", "c"]])", R"(["a b c"])", R"(["ab"])",
                               R"([["a", "c"]])", R"([["a", "b"], "a b"])"}) {
        auto bad = merged;
        bad["model"]["merges"] = nlohmann::json::parse(merges);
        const auto text = bad.dump();
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
            std::string("the merges ") + merges, "merge");
    }
    for (const char* merges : {R"(["a b"])", R"(["#version: 0.2", "a b"])"}) {
        auto good = merged;
        good["model"]["merges"] = nlohmann::json::parse(merges);
        const auto text = good.dump();
        std::vector<std::int32_t> ids;
        check_no_throw(
            [&] {
                ids = CreateBpeTokenizer(text.data(), text.size(), false, kVocab)->encode("ab");
            },
            std::string("the merges ") + merges);
        check(ids == std::vector<std::int32_t>{262}, std::string("ab merges with ") + merges);
    }
    // Added-token ids are the ones the reference assigns (it re-adds tokens by content):
    // a token listed at row 97 would replace "a" in decoding, and an empty token is dropped
    // by the reference, so both reject the bundle before any mapping changes.
    auto row_taken = nlohmann::json::parse(byte_tokenizer_json());
    row_taken["added_tokens"].push_back({{"id", 97},
                                         {"content", "UNUSED_SENTINEL"},
                                         {"special", false},
                                         {"single_word", false},
                                         {"lstrip", false},
                                         {"rstrip", false},
                                         {"normalized", false}});
    auto empty_token = nlohmann::json::parse(byte_tokenizer_json());
    empty_token["added_tokens"][0]["content"] = "";
    for (const auto& [layout, what, message] :
         {std::tuple{row_taken, "an added token at another token's row",
                     "the reference assigns it 262"},
          std::tuple{empty_token, "an empty added token", "nonempty content"}}) {
        const auto text = layout.dump();
        check_throws_as<std::runtime_error>(
            [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); }, what, message);
    }
    // One token per vocabulary id: the reference merges by operand ids.
    auto shared_row = nlohmann::json::parse(byte_tokenizer_json());
    shared_row["model"]["vocab"]["UNUSED_SENTINEL"] = 97;
    const auto shared_text = shared_row.dump();
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(shared_text.data(), shared_text.size(), false, kVocab); },
        "two vocabulary tokens on one id", "used by more than one token");
    // New added-token ids count up from the number of vocabulary entries, whatever ids the
    // vocabulary uses: with <|box_start|> moved to row 263, the 262 entries put the next
    // new token at 262, as the reference assigns it.
    const auto sparse_with = [](const std::vector<std::pair<int, std::string>>& new_tokens) {
        auto sparse = nlohmann::json::parse(byte_tokenizer_json());
        sparse["model"]["vocab"]["<|box_start|>"] = 263;
        for (auto& token : sparse["added_tokens"]) {
            if (token["content"] == "<|box_start|>")
                token["id"] = 263;
        }
        for (const auto& [id, content] : new_tokens)
            sparse["added_tokens"].push_back({{"id", id},
                                              {"content", content},
                                              {"special", true},
                                              {"single_word", false},
                                              {"lstrip", false},
                                              {"rstrip", false},
                                              {"normalized", false}});
        return sparse.dump();
    };
    for (const auto& [listed, accepted] : {std::pair{262, true}, std::pair{264, false}}) {
        const auto text = sparse_with({{listed, "<|new|>"}});
        const auto load = [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab + 2); };
        if (accepted)
            check_no_throw(load, "a new added token at 262 after a sparse vocabulary");
        else
            check_throws_as<std::runtime_error>(load, "a new added token at 264", "assigns it 262");
    }
    // The reference gives a second new token the next id, 263, the vocabulary row of
    // <|box_start|>, which would then decode as the new token.
    const auto collision = sparse_with({{262, "<|new|>"}, {263, "<|new2|>"}});
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(collision.data(), collision.size(), false, kVocab + 2); },
        "a new added token on a vocabulary row", "the vocabulary row of <|box_start|>");
    // The validated layout selects the byte-level path, not the vocabulary: a non-special
    // "\u2581" added token matches only where it is written.
    auto marker = nlohmann::json::parse(byte_tokenizer_json());
    marker["added_tokens"].push_back({{"id", 262},
                                      {"content", "\xE2\x96\x81"},
                                      {"special", false},
                                      {"single_word", false},
                                      {"lstrip", false},
                                      {"rstrip", false},
                                      {"normalized", false}});
    const auto marker_text = marker.dump();
    const auto with_marker =
        CreateBpeTokenizer(marker_text.data(), marker_text.size(), false, kVocab);
    check(with_marker->encode("a b") == bytes_of("a b"),
          "a \"\u2581\" added token keeps the byte-level path");
    check(with_marker->encode("a"
                              "\xE2\x96\x81"
                              "b") == std::vector<std::int32_t>({97, 262, 98}),
          "the \"\u2581\" added token where it is written");
    // A field of the wrong JSON type is a bundle defect too, not a JSON library error.
    auto mistyped = nlohmann::json::parse(byte_tokenizer_json());
    mistyped["decoder"]["type"] = 7;
    const auto mistyped_text = mistyped.dump();
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(mistyped_text.data(), mistyped_text.size(), false, kVocab); },
        "a decoder type that is not a string", "malformed");
    // Added-token defects are bundle defects as well.
    auto listless = nlohmann::json::parse(byte_tokenizer_json());
    listless["added_tokens"] = nlohmann::json::object();
    const auto listless_text = listless.dump();
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(listless_text.data(), listless_text.size(), false, kVocab); },
        "added_tokens that is not a list");
}

// The tensors of a bundle built for runtime_json(), as its modules report them at load.
EngineTensors encoder_tensors(const RuntimeConfig& config) {
    const std::int64_t longest =
        (static_cast<std::int64_t>(config.encoder_chunk_frames) + config.encoder_context_frames) *
        config.hop_length;
    EngineTensors tensors;
    tensors["audio"] = {true, true, DType::kFloat32, {}, {1, 1, 1, 1}, {1, 1, 1, longest}};
    tensors["audio_features"] = {false, false, DType::kBFloat16, {508, config.hidden_size}, {}, {}};
    return tensors;
}

EngineTensors decoder_tensors(const RuntimeConfig& config, std::int64_t rows) {
    EngineTensors tensors;
    tensors["input_ids"] = {true, true, DType::kInt32, {}, {1}, {rows}};
    tensors["audio_mask"] = {true, true, DType::kInt32, {}, {1}, {rows}};
    tensors["audio_embeds"] = {
        true, true, DType::kBFloat16, {}, {1, config.hidden_size}, {rows, config.hidden_size}};
    tensors["cache_write_index"] = {true, false, DType::kInt32, {1}, {}, {}};
    tensors["logits"] = {false, false, DType::kFloat32, {1, config.vocab_size}, {}, {}};
    return tensors;
}

void test_engine_io() {
    const auto config = parse_runtime_config(runtime_json().dump());
    check_no_throw([&] { check_encoder_io(encoder_tensors(config), config); },
                   "the encoder the builder makes");
    check_no_throw([&] { check_decoder_io(decoder_tensors(config, 256), config, 256); },
                   "the prefill profile the builder makes");
    check_no_throw([&] { check_decoder_io(decoder_tensors(config, 1), config, 1); },
                   "the decode profile the builder makes");
    // Each mismatch fails at load, not at the first request that reaches it.
    using Change = void (*)(EngineTensors&);
    const std::pair<const char*, Change> encoder_changes[] = {
        {"an audio profile shorter than the longest chunk",
         [](EngineTensors& t) { t["audio"].profile_max[3] = 1000; }},
        {"an audio profile that starts above one sample",
         [](EngineTensors& t) { t["audio"].profile_min[3] = 3200; }},
        {"FP16 audio", [](EngineTensors& t) { t["audio"].dtype = DType::kFloat16; }},
        {"audio features of another width",
         [](EngineTensors& t) { t["audio_features"].shape[1] = 3583; }},
        {"no audio features", [](EngineTensors& t) { t.erase("audio_features"); }},
        {"audio as an output", [](EngineTensors& t) { t["audio"].input = false; }},
    };
    for (const auto& [what, change] : encoder_changes) {
        auto tensors = encoder_tensors(config);
        change(tensors);
        check_throws_as<std::runtime_error>([&] { check_encoder_io(tensors, config); },
                                            std::string("an encoder with ") + what);
    }
    const std::pair<const char*, Change> decoder_changes[] = {
        {"a prefill profile below max_prefill_rows",
         [](EngineTensors& t) { t["input_ids"].profile_max[0] = 255; }},
        {"audio embeddings of another width",
         [](EngineTensors& t) { t["audio_embeds"].profile_max[1] = 3583; }},
        {"logits of another vocabulary", [](EngineTensors& t) { t["logits"].shape[1] = 261; }},
        {"a wider cache_write_index", [](EngineTensors& t) { t["cache_write_index"].shape = {2}; }},
        {"an audio mask profile from two rows",
         [](EngineTensors& t) { t["audio_mask"].profile_min[0] = 2; }},
    };
    for (const auto& [what, change] : decoder_changes) {
        auto tensors = decoder_tensors(config, 256);
        change(tensors);
        check_throws_as<std::runtime_error>([&] { check_decoder_io(tensors, config, 256); },
                                            std::string("a decoder with ") + what);
    }
}

void test_prompt() {
    const auto json = byte_tokenizer_json();
    const auto tokenizer = CreateBpeTokenizer(json.data(), json.size(), false, 262);
    check(tokenizer != nullptr, "the byte tokenizer loads");
    if (!tokenizer)
        return;
    const auto config = parse_runtime_config(runtime_json().dump());
    check(!config.tokenizer_checks.empty() &&
              tokenizer->encode("user") == config.tokenizer_checks[0].ids,
          "byte ids match");
    check(user_suffix(config, 181440, "") ==
              "This is a 7.56 seconds audio, please transcribe it with these keys: Start time, "
              "End time, Speaker ID, Content",
          "the plain suffix");
    check(user_suffix(config, 24000, "  Names: Ada \n") ==
              "This is a 1.00 seconds audio, with extra info: Names: Ada\n\nPlease transcribe it "
              "with these keys: Start time, End time, Speaker ID, Content",
          "the context suffix");
    check(user_suffix(config, 24000, " \n ").find("extra info") == std::string::npos,
          "blank context is absent");
    // Python str.strip() also removes NBSP and the ideographic space.
    check(user_suffix(config, 24000, "\xC2\xA0Names: Ada\xE3\x80\x80").find("info: Names: Ada\n") !=
              std::string::npos,
          "Unicode whitespace is stripped like str.strip()");
    check(user_suffix(config, 24000, "\xC2\xA0\xE3\x80\x80").find("extra info") ==
              std::string::npos,
          "Unicode-blank context is absent");
    // f"{x:.2f}" rounds the binary value: 2.675 -> 2.67, 0.005 -> 0.01.
    check(user_suffix(config, 64200, "").find("a 2.67 seconds") != std::string::npos, "2.675 s");
    check(user_suffix(config, 120, "").find("a 0.01 seconds") != std::string::npos, "0.005 s");
    if (std::setlocale(LC_NUMERIC, "de_DE.UTF-8") != nullptr) {
        check(user_suffix(config, 181440, "").find("7.56") != std::string::npos,
              "the duration ignores a comma-decimal locale");
        std::setlocale(LC_NUMERIC, "C");
    }
    const auto prompt = build_prompt(*tokenizer, config, 6401, "");
    check(prompt.speech_rows == 3, "ceil(6401 / 3200) speech rows");
    // Indexed checks only when the speech span and its end marker lie inside the prompt.
    const bool span_inside = prompt.speech_offset >= 1 && prompt.speech_rows >= 0 &&
                             static_cast<std::size_t>(prompt.speech_offset) +
                                     static_cast<std::size_t>(prompt.speech_rows) <
                                 prompt.ids.size();
    check(span_inside, "the speech span lies inside the prompt");
    if (span_inside) {
        check(prompt.ids.front() == 257, "the prompt opens the system turn");
        check(prompt.ids[static_cast<std::size_t>(prompt.speech_offset) - 1] == 259,
              "speech start precedes the placeholders");
        for (int row = 0; row < prompt.speech_rows; ++row)
            check(prompt.ids[static_cast<std::size_t>(prompt.speech_offset + row)] == 261,
                  "speech placeholder rows");
        check(prompt.ids[static_cast<std::size_t>(prompt.speech_offset + prompt.speech_rows)] ==
                  260,
              "speech end follows the placeholders");
        check(prompt.ids.back() == '\n', "no generation prompt after the user turn");
    }
    check(std::count(prompt.ids.begin(), prompt.ids.end(), 258) == 2, "two turns end");
    check_throws_as<std::invalid_argument>([&] { build_prompt(*tokenizer, config, 0, ""); },
                                           "an empty clip", "at least one audio sample");
    // A request is sized before its audio is resampled: 100000 samples at 1 Hz need
    // 750,000 speech rows, refused without building them.
    check_throws_as<std::invalid_argument>(
        [&] { build_prompt(*tokenizer, config, resampled_length(100000, 1, 24000), ""); },
        "a 1 Hz clip of 100000 samples");
}

} // namespace

int main() {
    test_downmix();
    test_resample();
    test_normalize();
    test_chunks_and_argmax();
    test_transcript();
    test_runtime_config();
    test_prompt();
    test_tokenizer();
    test_engine_io();
    if (failures != 0) {
        std::cerr << failures << " VibeVoice-ASR host check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "VibeVoice-ASR host checks passed\n";
    return EXIT_SUCCESS;
}
