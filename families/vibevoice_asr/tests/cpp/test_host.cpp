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

template <class Function>
void check_throws(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    check(false, what + " should throw");
}

template <class Error, class Function>
void check_throws_as(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const Error&) {
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
    check_throws([&] { downmix(bad.data(), 2, 1); }, "downmix of a NaN sample");
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
    double worst = 0.0;
    for (std::size_t i = 2400; i + 2400 < out.size(); ++i)
        worst = std::max(worst, std::fabs(out[i] - 0.5 * std::sin(2.0 * pi * 440.0 * i / 24000.0)));
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
    check(odd_out.size() == (odd.size() * 24000 + odd_rate - 1) / odd_rate,
          "resampled length is ceil(n * 24000 / 192001)");
    double odd_worst = 0.0;
    for (std::size_t i = 600; i + 600 < odd_out.size(); ++i)
        odd_worst = std::max(
            odd_worst, std::fabs(odd_out[i] - 0.5 * std::sin(2.0 * pi * 440.0 * i / 24000.0)));
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
    double difference = 0.0;
    for (std::size_t i = 0; i < exact.size(); ++i)
        difference =
            std::max(difference, static_cast<double>(std::fabs(exact[i] - interpolated[i])));
    check(exact.size() == interpolated.size() && difference < 1e-6,
          "interpolated and exact taps differ by " + std::to_string(difference));
    check(resample(tone, 16000, 24000, 0).size() == out.size(), "both paths give the same length");

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
    const float peak = *std::max_element(impulse.begin(), impulse.end());
    check(peak <= 1.0F && peak > 0.999F, "the clipping guard divides by the peak");
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
    check(chunks[0].first_frame == 0 && chunks[0].frames == 450 && chunks[0].context_frames == 0,
          "the first chunk has no left context");
    check(chunks[1].first_frame == 450 && chunks[1].frames == 450 && chunks[1].context_frames == 60,
          "the second chunk carries the receptive field");
    check(chunks[2].first_frame == 900 && chunks[2].frames == 100 && chunks[2].context_frames == 60,
          "the last chunk ends at the last frame");
    const auto single = plan_encoder_chunks(3, 450, 60);
    check(single.size() == 1 && single[0].frames == 3, "a short clip is one chunk");
    check_throws([] { plan_encoder_chunks(0, 450, 60); }, "an empty chunk plan");
    const float tied[] = {0.5F, 2.0F, 1.0F, 2.0F};
    check(argmax(tied, 4) == 1, "argmax keeps the lowest index of a tie");
    // A prompt of max_sequence_length - 2 tokens leaves room for two output tokens.
    const std::int64_t unlimited = std::numeric_limits<std::int64_t>::max();
    std::int64_t generated = 0;
    do
        ++generated;
    while (!generation_complete(4094, generated, unlimited, 4096));
    check(generated == 2, "prompt plus output stays within max_sequence_length");
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
    check_throws([&] { parse_runtime_config(parallel.dump()); }, "tensor parallel runtime.json");
    auto outside = runtime_json();
    outside["token_ids"]["eos"] = 262;
    check_throws([&] { parse_runtime_config(outside.dump()); }, "an out-of-vocabulary token");
    auto unchecked = runtime_json();
    unchecked["tokenizer_checks"] = nlohmann::json::array();
    check_throws([&] { parse_runtime_config(unchecked.dump()); }, "missing tokenizer checks");
    // Probe ids are integers in [0, vocab_size): no truncation of 117.9 to 117.
    for (const auto& ids :
         {nlohmann::json{117.9}, nlohmann::json{-1}, nlohmann::json{262}, nlohmann::json{"117"}}) {
        auto probe = runtime_json();
        probe["tokenizer_checks"][0]["ids"] = ids;
        check_throws([&] { parse_runtime_config(probe.dump()); },
                     "a tokenizer check id of " + ids.dump());
    }
    // eps must keep the float gain of silent input finite: 1e-300 is 0 as a float,
    // 1e-40 is subnormal and overflows the gain.
    for (const char* eps : {"1e-300", "1e-40"}) {
        auto tiny = runtime_json();
        tiny["normalize_eps"] = std::strtod(eps, nullptr);
        check_throws([&] { parse_runtime_config(tiny.dump()); },
                     std::string("normalize_eps ") + eps);
    }
    auto smallest = runtime_json();
    smallest["normalize_eps"] = static_cast<double>(std::numeric_limits<float>::min());
    check(parse_runtime_config(smallest.dump()).normalize_eps > 0.0,
          "the smallest normal float eps is accepted");
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
        check_throws([&] { split->encode(text); }, "encoding invalid UTF-8");
        check_throws([&] { nfc->encode(text); }, "encoding invalid UTF-8 with NFC");
    }
    // ByteLevel decoding joins all bytes, then replaces what is not UTF-8 (U+FFFD).
    const std::string replacement = "\xEF\xBF\xBD";
    check(split->decode({0xC3, 0xA9}) == "\xC3\xA9", "a character split over two tokens");
    check(split->decode({0x61, 0xC3}) == "a" + replacement, "a cut-off last character");
    check(split->decode({0xE4, 0xB8}) == replacement, "one replacement per maximal subpart");
    check(split->decode({0x61, 0xFF, 0x62}) == "a" + replacement + "b", "an invalid byte");
    check(split->decode({0xC0, 0x80}) == replacement + replacement, "an overlong encoding");
    // An added token without an integer id or a string content is rejected, not read.
    for (const char* field : {"id", "content"}) {
        auto broken = nlohmann::json::parse(byte_tokenizer_json());
        broken["added_tokens"][0].erase(field);
        const auto text = broken.dump();
        check_throws([&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
                     std::string("an added token without ") + field);
    }
    // Added-token ids are bounded by the decoder vocabulary before the vocabulary grows.
    for (const std::int64_t id :
         {std::int64_t{kVocab}, std::int64_t{2147483647}, std::int64_t{-1}}) {
        auto stray = nlohmann::json::parse(byte_tokenizer_json());
        stray["added_tokens"][0]["id"] = id;
        const auto text = stray.dump();
        check_throws([&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
                     "an added token id of " + std::to_string(id));
    }
    const auto json = byte_tokenizer_json();
    check_throws([&] { CreateBpeTokenizer(json.data(), json.size(), false, 0); },
                 "a tokenizer without a decoder vocabulary");
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
    check(tokenizer->encode("user") == config.tokenizer_checks[0].ids, "byte ids match");
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
    check(prompt.ids.front() == 257, "the prompt opens the system turn");
    check(prompt.ids[static_cast<std::size_t>(prompt.speech_offset) - 1] == 259,
          "speech start precedes the placeholders");
    for (int row = 0; row < prompt.speech_rows; ++row)
        check(prompt.ids[static_cast<std::size_t>(prompt.speech_offset + row)] == 261,
              "speech placeholder rows");
    check(prompt.ids[static_cast<std::size_t>(prompt.speech_offset + prompt.speech_rows)] == 260,
          "speech end follows the placeholders");
    check(std::count(prompt.ids.begin(), prompt.ids.end(), 258) == 2, "two turns end");
    check(prompt.ids.back() == '\n', "no generation prompt after the user turn");
    check_throws([&] { build_prompt(*tokenizer, config, 0, ""); }, "an empty clip");
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
