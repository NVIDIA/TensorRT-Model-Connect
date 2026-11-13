/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Host-side VibeVoice-ASR pieces: audio front end, encoder chunk plan, argmax,
// prompt layout, transcript parsing and runtime.json validation.
#include "families/vibevoice_asr/runtime/audio_frontend.h"
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

// A byte-level BPE with one token per byte (no merges) and the special
// tokens the prompt uses; enough to check the prompt layout.
std::string byte_tokenizer_json() {
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
    const std::string regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
                              "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    nlohmann::json tokenizer = {
        {"added_tokens", added},
        {"normalizer", nullptr},
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
        {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", nlohmann::json::array()}}}};
    return tokenizer.dump();
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
}

void test_normalize() {
    std::vector<float> quiet(1000);
    for (std::size_t i = 0; i < quiet.size(); ++i)
        quiet[i] = 0.01F * static_cast<float>((i % 7) - 3);
    normalize_loudness(quiet, -25.0, 1e-6);
    double energy = 0.0;
    for (float value : quiet)
        energy += static_cast<double>(value) * value;
    const double rms = std::sqrt(energy / quiet.size());
    check(std::fabs(rms - std::pow(10.0, -25.0 / 20.0)) < 1e-5, "RMS normalised to -25 dBFS");
    // RMS 0.01 scales the impulse to 5.6, above the clipping limit.
    std::vector<float> impulse(10000, 0.0F);
    impulse[50] = 1.0F;
    normalize_loudness(impulse, -25.0, 1e-6);
    const float peak = *std::max_element(impulse.begin(), impulse.end());
    check(peak <= 1.0F && peak > 0.999F, "the clipping guard divides by the peak");
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
}

void test_prompt() {
    const auto json = byte_tokenizer_json();
    const auto tokenizer = CreateBpeTokenizer(json.data(), json.size(), false);
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
    if (failures != 0) {
        std::cerr << failures << " VibeVoice-ASR host check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "VibeVoice-ASR host checks passed\n";
    return EXIT_SUCCESS;
}
