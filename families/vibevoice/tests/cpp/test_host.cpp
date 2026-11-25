/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side checks of the vibevoice runtime against tests/data/host_fixtures.json:
// the DPM-Solver against diffusers (float32 and bfloat16 samples), the noise
// stream against its Python mirror, script parsing and prompt rendering against
// the reference chat template, WAV reading and voice-prompt preparation. Also the
// tokenizer on synthetic tokenizer.json files, runtime.json parsing and the engine
// I/O contract. With VIBEVOICE_TOKENIZER_JSON pointing at the Qwen2.5 tokenizer.json,
// the rendered prompts and the tokenizer probes are also tokenized and compared
// with the reference token ids.

#include "families/vibevoice/runtime/engine_io.h"
#include "families/vibevoice/runtime/noise.h"
#include "families/vibevoice/runtime/runtime_config.h"
#include "families/vibevoice/runtime/scheduler.h"
#include "families/vibevoice/runtime/script.h"
#include "families/vibevoice/runtime/tokenizer.h"
#include "families/vibevoice/runtime/voice.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace trtmc::vibevoice;
using trtmc::DType;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

template <class Error, class Function>
void check_throws_as(Function&& function, const std::string& what, const char* message = nullptr) {
    try {
        function();
    } catch (const Error& error) {
        if (message != nullptr && std::string(error.what()).find(message) == std::string::npos)
            expect(false, what + " threw another error: " + error.what());
        return;
    } catch (const std::exception& error) {
        expect(false, what + " threw another exception type: " + error.what());
        return;
    }
    expect(false, what + " should throw");
}

template <class Function>
void check_no_throw(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::exception& error) {
        expect(false, what + " threw: " + error.what());
    }
}

// The rows of the language model's embedding table that the Qwen2.5 tokenizer indexes.
constexpr std::int32_t kQwenVocab = 151936;

nlohmann::json load_fixtures() {
    std::ifstream file(std::string(VIBEVOICE_TEST_DATA) + "/host_fixtures.json");
    if (!file)
        throw std::runtime_error("host_fixtures.json is missing");
    return nlohmann::json::parse(file);
}

PromptFrame frame() {
    PromptFrame value;
    value.system_prompt = " Transform the text provided by various speakers into speech output, "
                          "utilizing the distinct voice of each respective speaker.\n";
    value.voice_input_header = " Voice input:\n";
    value.text_input_header = " Text input:\n";
    value.speech_output_header = " Speech output:\n";
    value.audio_bos_token = "<|vision_start|>";
    value.audio_eos_token = "<|vision_end|>";
    value.audio_token = "<|vision_pad|>";
    return value;
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

// A byte-level BPE in Qwen2.5's layout with one token per byte and six special
// tokens (ids 256-261). `merges` adds merged tokens (ids from 262) so a test can
// see where the pre-tokenizer splits.
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

void test_noise(const nlohmann::json& fixtures) {
    NoiseStream noise(fixtures.at("noise").at("seed").get<std::uint64_t>());
    for (const auto& expected : fixtures.at("noise").at("values"))
        expect(noise.normal() == expected.get<float>(), "noise stream matches its Python mirror");
}

void test_timesteps(const nlohmann::json& fixtures) {
    // numpy's float64 points rounded half to even: the 1000-step fixtures, and schedules
    // whose float64 point lies one ulp off an exact half, where numpy rounds the ulp
    // (numpy.linspace: 11 * (25 / 22) is 12.500000000000002, 7 * (29 / 14) is
    // 14.500000000000002, 21 * (34 / 28) is 25.499999999999996).
    std::vector<std::tuple<std::int32_t, std::int32_t, std::vector<std::int64_t>>> cases;
    for (const auto& [steps, expected] : fixtures.at("timesteps").items())
        cases.emplace_back(1000, std::stoi(steps), expected.get<std::vector<std::int64_t>>());
    cases.emplace_back(1000, 2, std::vector<std::int64_t>{999, 500});
    cases.emplace_back(26, 22,
                       std::vector<std::int64_t>{25, 24, 23, 22, 20, 19, 18, 17, 16, 15, 14,
                                                 13, 11, 10, 9,  8,  7,  6,  5,  3,  2,  1});
    cases.emplace_back(
        30, 14, std::vector<std::int64_t>{29, 27, 25, 23, 21, 19, 17, 15, 12, 10, 8, 6, 4, 2});
    cases.emplace_back(35, 28, std::vector<std::int64_t>{34, 33, 32, 30, 29, 28, 27, 25, 24, 23,
                                                         22, 21, 19, 18, 17, 16, 15, 13, 12, 11,
                                                         10, 8,  7,  6,  5,  4,  2,  1});
    // The rounding mode the host selects changes none of them, and stays as it was set.
    const int saved = std::fegetround();
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        std::vector<std::vector<std::int64_t>> actual;
        std::fesetround(mode);
        for (const auto& item : cases)
            actual.push_back(linspace_timesteps(std::get<0>(item), std::get<1>(item)));
        const bool kept = std::fegetround() == mode;
        std::fesetround(saved);
        const auto label = " in rounding mode " + std::to_string(mode);
        expect(kept, "linspace_timesteps keeps the caller's rounding mode" + label);
        for (std::size_t index = 0; index < cases.size(); ++index)
            expect(actual[index] == std::get<2>(cases[index]),
                   "timesteps for " + std::to_string(std::get<1>(cases[index])) + " of " +
                       std::to_string(std::get<0>(cases[index])) + " steps" + label);
    }
    // Up to 2^20 training steps the float64 points cannot cross a half-integer they do not
    // lie on; longer schedules are refused.
    check_no_throw([] { linspace_timesteps(1 << 20, 10); }, "a schedule of 2^20 training steps");
    check_throws_as<std::invalid_argument>([] { linspace_timesteps((1 << 20) + 1, 10); },
                                           "a schedule of 2^20 + 1 training steps",
                                           "at most 1048576 training steps");
}

// The largest relative difference of `actual` from `expected`; infinite when the
// lengths differ or either side is not finite, which std::max would otherwise skip.
double worst_difference(const std::vector<float>& actual, const std::vector<float>& expected) {
    if (actual.size() != expected.size())
        return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double scale = std::max(1.0, std::fabs(static_cast<double>(expected[index])));
        const double difference =
            std::fabs(static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
        if (!std::isfinite(difference))
            return std::numeric_limits<double>::infinity();
        worst = std::max(worst, difference / scale);
    }
    return worst;
}

void test_schedulers(const nlohmann::json& fixtures) {
    const auto sigmas = fixtures.at("train_sigmas").get<std::vector<float>>();
    for (const auto& item : fixtures.at("schedulers")) {
        const bool bf16 = item.at("precision") == "bf16";
        DpmSolver solver(sigmas, item.at("steps").get<std::int32_t>(), bf16);
        const auto label = "scheduler " + item.at("precision").get<std::string>() + " x" +
                           std::to_string(item.at("steps").get<int>());
        expect(solver.timesteps() == item.at("timesteps").get<std::vector<std::int64_t>>(),
               label + " timesteps");
        expect(static_cast<std::size_t>(solver.steps()) == item.at("outputs").size(),
               label + " has one output per step");
        auto sample = item.at("initial").get<std::vector<float>>();
        solver.reset();
        double worst = 0.0;
        for (std::int32_t step = 0; step < solver.steps(); ++step) {
            const auto velocity = item.at("velocities").at(step).get<std::vector<float>>();
            solver.step(step, velocity, sample);
            worst = std::max(
                worst,
                worst_difference(sample, item.at("outputs").at(step).get<std::vector<float>>()));
        }
        // float32 math in a different evaluation order than torch can differ by a few ulp;
        // a bfloat16 rounding decision flipped by such an ulp moves one value by one bf16 step.
        expect(worst <= (bf16 ? 8e-3 : 1e-5),
               label + " matches diffusers (worst " + std::to_string(worst) + ")");
    }
    expect(worst_difference({1.0F, std::nanf("")}, {1.0F, 1.0F}) ==
                   std::numeric_limits<double>::infinity() &&
               worst_difference({1.0F}, {1.0F, 2.0F}) == std::numeric_limits<double>::infinity(),
           "a NaN or a short output is the worst difference");
}

void test_schedule_limits(const nlohmann::json& fixtures) {
    const auto sigmas = fixtures.at("train_sigmas").get<std::vector<float>>();
    // linspace(0, 999, 1001) holds 499.5 and 500.499; both round to timestep 500.
    check_throws_as<std::invalid_argument>([&] { check_schedule(sigmas, 1000); },
                                           "1000 steps of the 1000-step schedule",
                                           "repeats a noise level");
    check_throws_as<std::invalid_argument>(
        [&] { static_cast<void>(DpmSolver(sigmas, 1000, false)); },
        "a solver over a repeated timestep", "repeats a noise level");
    check_no_throw([&] { check_schedule(sigmas, 999); }, "999 steps of the 1000-step schedule");
    // Equal noise levels at distinct timesteps divide by zero in the same way.
    auto flat = sigmas;
    flat[500] = flat[499];
    check_throws_as<std::invalid_argument>([&] { check_schedule(flat, 999); },
                                           "two timesteps with one noise level",
                                           "repeats a noise level");
    check_throws_as<std::invalid_argument>([&] { check_schedule(sigmas, 0); }, "zero steps",
                                           "must lie in");
    // A sigma that is not positive and finite is refused before its logarithm is taken (a
    // host that traps division by zero would stop there); timestep 999 starts every schedule.
    for (const float bad : {0.0F, std::numeric_limits<float>::infinity()}) {
        auto broken = sigmas;
        broken[999] = bad;
        std::feclearexcept(FE_ALL_EXCEPT);
        check_throws_as<std::invalid_argument>([&] { check_schedule(broken, 10); },
                                               "a schedule with a sigma of " + std::to_string(bad),
                                               "positive and finite");
        expect(std::fetestexcept(FE_DIVBYZERO) == 0,
               "a sigma of " + std::to_string(bad) + " is refused without a logarithm of 0");
    }
}

void test_scheduler_state(const nlohmann::json& fixtures) {
    const auto sigmas = fixtures.at("train_sigmas").get<std::vector<float>>();
    // The midpoint update reads the previous step's prediction: a wider latent without
    // reset() is refused before the solver's history changes.
    DpmSolver solver(sigmas, 3, false);
    DpmSolver fresh(sigmas, 3, false);
    solver.reset();
    fresh.reset();
    std::vector<float> narrow = {0.5F};
    std::vector<float> same = {0.5F};
    solver.step(0, {0.25F}, narrow);
    fresh.step(0, {0.25F}, same);
    std::vector<float> wide = {0.5F, -0.5F};
    const std::vector<float> wide_velocity = {0.25F, 0.125F};
    check_throws_as<std::invalid_argument>([&] { solver.step(1, wide_velocity, wide); },
                                           "a step of a wider latent without reset()",
                                           "call reset() first");
    for (std::int32_t step = 1; step < 3; ++step) {
        solver.step(step, {0.25F}, narrow);
        fresh.step(step, {0.25F}, same);
    }
    expect(narrow == same, "a refused step leaves the solver's history as it was");
    solver.reset();
    check_no_throw([&] { solver.step(0, wide_velocity, wide); }, "a wider latent after reset()");
    // The final step goes to sigma 0 without a logarithm of 0 (the diffusers fixtures check
    // its value: the data prediction).
    for (const bool bf16 : {false, true}) {
        DpmSolver last(sigmas, 2, bf16);
        last.reset();
        std::vector<float> sample = {0.75F, -0.5F};
        last.step(0, {0.5F, 0.25F}, sample);
        std::feclearexcept(FE_ALL_EXCEPT);
        last.step(1, {0.5F, 0.25F}, sample);
        expect(std::fetestexcept(FE_DIVBYZERO) == 0 && std::isfinite(sample[0]) &&
                   std::isfinite(sample[1]),
               std::string("the final step to sigma 0 takes no logarithm of 0") +
                   (bf16 ? " (bf16)" : ""));
    }
}

std::vector<char> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void test_scripts(const nlohmann::json& fixtures) {
    std::unique_ptr<ITokenizer> tokenizer;
    if (const char* path = std::getenv("VIBEVOICE_TOKENIZER_JSON")) {
        const auto json = read_file(path);
        tokenizer = CreateBpeTokenizer(json.data(), json.size(), false, kQwenVocab);
    }
    for (const auto& [name, item] : fixtures.at("scripts").items()) {
        const auto turns = parse_script(item.at("text").get<std::string>());
        const auto expected = item.at("turns");
        bool same = turns.size() == expected.size();
        for (std::size_t index = 0; same && index < turns.size(); ++index)
            same = turns[index].speaker == expected[index][0].get<int>() &&
                   turns[index].text == expected[index][1].get<std::string>();
        expect(same, "script " + name + " parses as the reference turns");
        if (!item.contains("prompt"))
            continue;
        const auto prompt = render_prompt(frame(), turns, {});
        expect(prompt == item.at("prompt").get<std::string>(),
               "script " + name + " renders the reference prompt");
        if (tokenizer)
            expect(tokenizer->encode(prompt) == item.at("ids").get<std::vector<std::int32_t>>(),
                   "script " + name + " tokenizes like the reference");
    }
    if (tokenizer) {
        for (const auto& [name, item] : fixtures.at("tokenizer_probes").items())
            expect(tokenizer->encode(item.at("text").get<std::string>()) ==
                       item.at("ids").get<std::vector<std::int32_t>>(),
                   "probe " + name + " tokenizes like the reference");
    }
    bool rejected = false;
    try {
        parse_script("Speaker 1: one\nnot a speaker line");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "a script mixing speaker and plain lines is refused");
    rejected = false;
    try {
        parse_script(" \n\t");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "empty text is refused");
    // Text that cannot fit is refused while it is parsed, before any copy: a script counts
    // the text its turns keep (not the "Speaker <id>:" prefixes or blank lines), plain text
    // all of it.
    check_throws_as<std::length_error>([] { parse_script(std::string(1000, 'a'), 999); },
                                       "plain text over its byte budget", "more than 999 bytes");
    check_no_throw([] { parse_script(std::string(1000, 'a'), 1000); },
                   "plain text at its byte budget");
    const std::string budgeted = "Speaker 1: abc\n\n\n\nSpeaker 2: defg\n";
    check_no_throw([&] { parse_script(budgeted, 7); }, "a script whose turns keep 7 bytes");
    check_throws_as<std::length_error>([&] { parse_script(budgeted, 6); },
                                       "a script over its byte budget", "more than 6 bytes");
    // Every rendered turn takes at least one position, so short turns are counted too.
    std::string short_turns;
    for (int turn = 0; turn < 1000; ++turn)
        short_turns += "Speaker 1: a\n";
    check_throws_as<std::length_error>(
        [&] { parse_script(short_turns, std::numeric_limits<std::size_t>::max(), 999); },
        "a thousand one-letter turns over a budget of 999", "more than 999 turns");
    check_no_throw(
        [&] { parse_script(short_turns, std::numeric_limits<std::size_t>::max(), 1000); },
        "a thousand one-letter turns within a budget of 1000");
    // Plain text is one turn, under the turn limit as well.
    check_throws_as<std::length_error>(
        [] { parse_script("Hello", std::numeric_limits<std::size_t>::max(), 0); },
        "plain text where no turn may be", "more than 0 turns");
    check_no_throw([] { parse_script("Hello", std::numeric_limits<std::size_t>::max(), 1); },
                   "plain text where one turn may be");
    // Only the colon makes a speaker line: digits past the int32 range without one are text,
    // with one an id that is too large.
    std::vector<Turn> elected;
    check_no_throw([&] { elected = parse_script("Speaker 2147483648 was elected."); },
                   "a number past int32 in plain text");
    expect(elected.size() == 1 && elected[0].speaker == 0 &&
               elected[0].text == "Speaker 2147483648 was elected.",
           "a number past int32 without a colon stays text");
    check_no_throw([] { parse_script("Speaker " + std::string(40, '9') + " apples"); },
                   "forty digits without a colon");
    check_throws_as<std::invalid_argument>([] { parse_script("Speaker 2147483648: hi"); },
                                           "a speaker id past int32", "too large");
    check_no_throw([] { parse_script("Speaker 2147483647: hi"); }, "the largest int32 speaker id");
    const auto voiced = render_prompt(frame(), {{0, "Hi."}}, {2});
    expect(
        voiced.find(
            " Voice input:\n Speaker 0:<|vision_start|><|vision_pad|><|vision_pad|><|vision_end|>\n"
            " Text input:\n") != std::string::npos,
        "voice placeholders follow the reference template");
}

// A RIFF/WAVE file with one fmt chunk and one data chunk; `declared` overrides the
// data chunk's size field and `riff` the container's.
void write_wav(const std::string& path, std::uint16_t format, std::uint16_t bits,
               const std::vector<unsigned char>& payload, std::int64_t declared = -1,
               std::int64_t riff = -1) {
    std::ofstream file(path, std::ios::binary);
    auto put32 = [&](std::uint32_t value) { file.write(reinterpret_cast<const char*>(&value), 4); };
    auto put16 = [&](std::uint16_t value) { file.write(reinterpret_cast<const char*>(&value), 2); };
    const auto size = declared >= 0 ? static_cast<std::uint32_t>(declared)
                                    : static_cast<std::uint32_t>(payload.size());
    file.write("RIFF", 4);
    put32(riff >= 0 ? static_cast<std::uint32_t>(riff)
                    : 36 + static_cast<std::uint32_t>(payload.size()));
    file.write("WAVEfmt ", 8);
    put32(16);
    put16(format);
    put16(1);
    put32(24000);
    put32(24000U * bits / 8U);
    put16(static_cast<std::uint16_t>(bits / 8U));
    put16(bits);
    file.write("data", 4);
    put32(size);
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
}

template <class T>
std::vector<unsigned char> bytes_of_values(const std::vector<T>& values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

void test_wav() {
    const std::string path = "vibevoice_test_voice.wav";
    const std::vector<std::int16_t> pcm = {0, 16384, -16384, 32767, -32768, 1};
    write_wav(path, 1, 16, bytes_of_values(pcm));
    const auto audio = read_wav(path, 6);
    expect(audio.sample_rate == 24000 && audio.channels == 1 && audio.samples.size() == pcm.size(),
           "PCM16 WAV header is read");
    expect(audio.samples[1] == 0.5F && audio.samples[4] == -1.0F, "PCM16 samples scale by 1/32768");
    // The sample count is checked against the limit before the samples are read.
    check_throws_as<std::invalid_argument>([&] { read_wav(path, 5); },
                                           "a WAV with more samples than the limit",
                                           "holds more than 5 samples");
    write_wav(path, 1, 16, bytes_of_values(pcm), 4096);
    check_throws_as<std::invalid_argument>([&] { read_wav(path, 1 << 20); },
                                           "a data chunk longer than the file", "truncated");
    // The declared RIFF size bounds the chunks and must fit the file; an odd chunk keeps
    // its pad byte inside it.
    write_wav(path, 1, 16, bytes_of_values(pcm), -1, 36 + 12 + 100);
    check_throws_as<std::invalid_argument>([&] { read_wav(path, 16); },
                                           "a RIFF size beyond the file", "RIFF size");
    write_wav(path, 1, 24, {0x00, 0x00, 0x40});
    check_throws_as<std::invalid_argument>([&] { read_wav(path, 16); },
                                           "an odd chunk without its pad byte", "truncated");
    write_wav(path, 1, 24, {0x00, 0x00, 0x40, 0x00}, 3);
    std::vector<float> padded;
    check_no_throw([&] { padded = read_wav(path, 16).samples; }, "an odd chunk with its pad byte");
    expect(padded.size() == 1 && padded[0] == 0.5F, "a padded 24-bit sample is read");
    for (const float bad : {std::nanf(""), std::numeric_limits<float>::infinity()}) {
        write_wav(path, 3, 32, bytes_of_values(std::vector<float>{0.25F, bad, -0.25F}));
        check_throws_as<std::invalid_argument>(
            [&] { read_wav(path, 16); }, "a float WAV with a nonfinite sample", "NaN or infinite");
    }
    std::remove(path.c_str());

    // Every synthesized chunk is checked before it joins the result.
    const float finite[] = {0.0F, -1.0F, 1e30F};
    const float broken[] = {0.0F, std::nanf(""), 1.0F};
    expect(all_finite(finite, 3) && !all_finite(broken, 3) && all_finite(broken, 1),
           "a NaN sample is found");
    const auto prepared = prepare_voice(audio.samples, -25.0F, 1e-6F, 4);
    expect(prepared.size() == 8 && prepared[6] == 0.0F && prepared[7] == 0.0F,
           "voice is zero-padded to whole frames");
    double peak = 0.0;
    for (const float sample : prepared)
        peak = std::max(peak, static_cast<double>(std::fabs(sample)));
    expect(peak <= 1.0, "voice peak is limited to 1");
    check_throws_as<std::invalid_argument>(
        [&] { prepare_voice({0.5F, std::nanf("")}, -25.0F, 1e-6F, 4); },
        "a voice prompt with a NaN sample", "NaN or infinite");
    check_throws_as<std::invalid_argument>(
        [&] { prepare_voice({0.5F, -std::numeric_limits<float>::infinity()}, -25.0F, 1e-6F, 4); },
        "a voice prompt with an infinite sample", "NaN or infinite");
}

void test_tokenizer() {
    constexpr std::int32_t kVocab = 264; // bytes, specials and two merged tokens
    // Unicode NFC, the normalizer Qwen2.5's tokenizer.json declares: e + U+0301 is U+00E9.
    const auto nfc_json = byte_tokenizer_json({{"type", "NFC"}});
    const auto nfc = CreateBpeTokenizer(nfc_json.data(), nfc_json.size(), false, kVocab);
    expect(nfc->encode("Cafe\xCC\x81") == bytes_of("Caf\xC3\xA9"), "NFC composes e + U+0301");
    expect(nfc->encode("Caf\xC3\xA9") == bytes_of("Caf\xC3\xA9"), "NFC keeps a composed character");
    // \p{N} digits are their own pre-tokens: the merges a+U+0661 (bytes D9 A1) and
    // U+0661+U+0662 must not apply across those boundaries.
    const std::string arabic_one = "\xD9\xA1"; // U+0661 ARABIC-INDIC DIGIT ONE
    const std::string arabic_two = "\xD9\xA2"; // U+0662 ARABIC-INDIC DIGIT TWO
    const auto split_json =
        byte_tokenizer_json(nullptr, {{"a", utf8(0xD9)}, {utf8(0xA1), utf8(0xD9)}});
    const auto split = CreateBpeTokenizer(split_json.data(), split_json.size(), false, kVocab);
    expect(split->encode("a" + arabic_one) == bytes_of("a" + arabic_one),
           "a letter and an Arabic-Indic digit are separate pre-tokens");
    expect(split->encode(arabic_one + arabic_two) == bytes_of(arabic_one + arabic_two),
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
    expect(split->decode({0xC3, 0xA9}) == "\xC3\xA9", "a character split over two tokens");
    expect(split->decode({0x61, 0xC3}) == "a" + replacement, "a cut-off last character");
    expect(split->decode({0xE4, 0xB8}) == replacement, "one replacement per maximal subpart");
    expect(split->decode({0x61, 0xFF, 0x62}) == "a" + replacement + "b", "an invalid byte");
    expect(split->decode({0xC0, 0x80}) == replacement + replacement, "an overlong encoding");
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
    // Every byte needs its byte-level token: without "z" (byte 122), encode() would drop
    // that byte while the token budget counts it.
    auto no_z = nlohmann::json::parse(byte_tokenizer_json());
    no_z["model"]["vocab"].erase("z");
    const auto no_z_text = no_z.dump();
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(no_z_text.data(), no_z_text.size(), false, kVocab); },
        "a vocabulary without the token of byte 122", "lacks the byte-level token of byte 122");
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
    expect(added->decode({262}) == written, "an added token outside the byte alphabet");
    expect(added->decode({0x61, 262, 0xC3, 0xA9}) == "a" + written + "\xC3\xA9",
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
    // The reference merges one pair at a time, the lowest rank first, so a merge that
    // uses another merge's result applies even when that merge has a later rank; an
    // entry that is not one pair, a pair or result outside the vocabulary, or two entry
    // formats in one table are rejected.
    const auto later = byte_tokenizer_json(nullptr, {{"ab", "a"}, {"a", "b"}});
    const auto reordered = CreateBpeTokenizer(later.data(), later.size(), false, kVocab);
    expect(reordered->encode("abab") == std::vector<std::int32_t>({262, 'b'}),
           "abab merges as the reference does: aba, b");
    expect(reordered->encode("ababab") == std::vector<std::int32_t>({262, 'b', 263}),
           "ababab merges as the reference does: aba, b, ab");
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
        expect(ids == std::vector<std::int32_t>{262}, std::string("ab merges with ") + merges);
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
    expect(with_marker->encode("a b") == bytes_of("a b"),
           "a \"\u2581\" added token keeps the byte-level path");
    expect(with_marker->encode("a"
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

void test_tokenizer_contractions_and_budget() {
    constexpr std::int32_t kVocab = 264; // bytes, specials and two merged tokens
    // Qwen2.5's Split pattern matches contractions case-insensitively, (?i:'s|'t|...),
    // and lets an apostrophe lead a letter run, [^\r\n\p{L}\p{N}]?\p{L}+.
    const auto json = byte_tokenizer_json(nullptr, {{"'", "S"}, {"'", "n"}});
    const auto tokenizer = CreateBpeTokenizer(json.data(), json.size(), false, kVocab);
    expect(tokenizer->encode("IT'S") == std::vector<std::int32_t>({'I', 'T', 262}),
           "an upper-case contraction is one pre-token");
    expect(tokenizer->encode("rock'n") == std::vector<std::int32_t>({'r', 'o', 'c', 'k', 263}),
           "an apostrophe leads the letters after it");
    // The token budget ends the work as soon as the ids cannot fit.
    expect(tokenizer->encode_within("abc", 3) == bytes_of("abc"), "a text inside its token budget");
    check_throws_as<std::length_error>([&] { tokenizer->encode_within("abcd", 3); },
                                       "a text over its token budget", "more than 3 tokens");
    // No token covers more than 20 bytes here and NFC shrinks UTF-8 at most threefold, so a
    // text is refused from its size alone, before it is copied, normalized or split.
    expect(tokenizer->max_text_bytes(3) == 239 && tokenizer->max_text_bytes(4) == 299,
           "the longest text that can fit n tokens is (n + 1) * 3 * 20 - 1 bytes");
    check_throws_as<std::length_error>([&] { tokenizer->encode_within(std::string(241, 'a'), 3); },
                                       "a text too large for its budget",
                                       "tokenizer input of 241 bytes cannot fit 3 tokens");
    // A piece is refused from its length before it is expanded and merged.
    check_throws_as<std::length_error>(
        [&] { tokenizer->encode_within(std::string(2000, 'a'), 60); },
        "a 2000-letter word over a budget of 60", "needs at least 100 tokens");
    // Each digit is a piece of its own and pieces are cut one at a time: six million of
    // them stop at the budget instead of all being held first.
    const std::string digits(6000000, '7');
    check_throws_as<std::length_error>([&] { tokenizer->encode_within(digits, 100000); },
                                       "six million one-digit pieces over a budget of 100000",
                                       "needs at least 100001 tokens");
}

void test_tokenizer_repeated_keys() {
    constexpr std::int32_t kVocab = 262;
    // nlohmann would keep the second of two equal keys; the tokenizer refuses the file,
    // also for a key inside the vocabulary.
    auto text = byte_tokenizer_json();
    const auto at = text.find("\"vocab\":{");
    text.insert(at + 9, "\"a\":97,");
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(text.data(), text.size(), false, kVocab); },
        "a vocabulary that repeats a token", "repeats the key a");
    // The special flag is a boolean, as the reference reads it.
    auto worded = nlohmann::json::parse(byte_tokenizer_json());
    worded["added_tokens"][0]["special"] = "false";
    const auto worded_text = worded.dump();
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(worded_text.data(), worded_text.size(), false, kVocab); },
        "an added token whose special flag is a string", "special is not supported");
    const auto plain = byte_tokenizer_json();
    const auto twice = "{\"normalizer\":null," + plain.substr(1);
    check_throws_as<std::runtime_error>(
        [&] { CreateBpeTokenizer(twice.data(), twice.size(), false, kVocab); },
        "a tokenizer that repeats a setting", "repeats the key normalizer");
}

void test_tokenizer_required_options() {
    constexpr std::int32_t kVocab = 262;
    // The reference refuses a tokenizer.json whose required boolean options are absent or null:
    // every added-token option, Split's invert, every ByteLevel component's add_prefix_space and
    // trim_offsets; use_regex may be absent (true) but not null; a BPE model option may be either.
    const auto load = [&](const std::function<void(nlohmann::json&)>& edit) {
        auto document = nlohmann::json::parse(byte_tokenizer_json());
        document["post_processor"] = nlohmann::json::parse(
            R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})");
        edit(document);
        const auto text = document.dump();
        return CreateBpeTokenizer(text.data(), text.size(), false, kVocab);
    };
    const auto refused = [&](const std::function<void(nlohmann::json&)>& edit,
                             const std::string& what, const std::string& message) {
        check_throws_as<std::runtime_error>([&] { load(edit); }, what, message.c_str());
    };
    for (const char* key : {"special", "lstrip", "rstrip", "single_word", "normalized"}) {
        refused([&](nlohmann::json& d) { d["added_tokens"][0][key] = nullptr; },
                std::string("an added token with a null ") + key,
                std::string(key) + " is not supported: null");
        refused([&](nlohmann::json& d) { d["added_tokens"][0].erase(key); },
                std::string("an added token without ") + key, std::string(key) + " is missing");
    }
    refused([](nlohmann::json& d) { d["pre_tokenizer"]["pretokenizers"][0]["invert"] = nullptr; },
            "a Split with a null invert", "invert is not supported: null");
    refused([](nlohmann::json& d) { d["pre_tokenizer"]["pretokenizers"][0].erase("invert"); },
            "a Split without invert", "invert is missing");
    const std::vector<std::pair<std::string, std::function<nlohmann::json&(nlohmann::json&)>>>
        components = {
            {"the pre-tokenizer's ByteLevel",
             [](nlohmann::json& d) -> nlohmann::json& {
                 return d["pre_tokenizer"]["pretokenizers"][1];
             }},
            {"the decoder", [](nlohmann::json& d) -> nlohmann::json& { return d["decoder"]; }},
            {"the post-processor",
             [](nlohmann::json& d) -> nlohmann::json& { return d["post_processor"]; }}};
    for (const auto& [name, component] : components) {
        for (const char* key : {"add_prefix_space", "trim_offsets"}) {
            refused([&](nlohmann::json& d) { component(d)[key] = nullptr; },
                    name + " with a null " + key, std::string(key) + " is not supported: null");
            refused([&](nlohmann::json& d) { component(d).erase(key); }, name + " without " + key,
                    std::string(key) + " is missing");
        }
        refused([&](nlohmann::json& d) { component(d)["use_regex"] = nullptr; },
                name + " with a null use_regex", "use_regex is not supported: null");
    }
    // An absent use_regex is true: the pre-tokenizer's own regex, another layout.
    refused([](nlohmann::json& d) { d["pre_tokenizer"]["pretokenizers"][1].erase("use_regex"); },
            "a ByteLevel pre-tokenizer without use_regex", "pre_tokenizer is not supported");
    check_no_throw(
        [&] {
            load([](nlohmann::json& d) {
                d["decoder"].erase("use_regex");
                d["post_processor"].erase("use_regex");
                d["model"]["byte_fallback"] = nullptr;
                d["model"].erase("ignore_merges");
            });
        },
        "a decoder and post-processor without use_regex, BPE options null or absent");
}

nlohmann::json runtime_json() {
    return {{"precision", "bf16"},
            {"hidden_size", 8},
            {"num_layers", 2},
            {"num_kv_heads", 1},
            {"head_dim", 4},
            {"vocab_size", 300},
            {"capacity", 64},
            {"prefill_chunk", 16},
            {"rope_inv_freq", {1.0, 0.01}},
            {"control_token_ids", {256, 257, 258, 259}},
            {"eos_token_id", 256},
            {"audio_bos_token_id", 257},
            {"audio_eos_token_id", 258},
            {"audio_token_id", 259},
            {"system_prompt", frame().system_prompt},
            {"voice_input_header", frame().voice_input_header},
            {"text_input_header", frame().text_input_header},
            {"speech_output_header", frame().speech_output_header},
            {"audio_bos_token", frame().audio_bos_token},
            {"audio_eos_token", frame().audio_eos_token},
            {"audio_token", frame().audio_token},
            {"sample_rate", 24000},
            {"hop_length", 4},
            {"latent_size", 2},
            {"target_dbfs", -25.0},
            {"normalize_eps", 1e-6},
            {"max_voice_samples", 16},
            {"stream_state_size", 10},
            {"train_sigmas", {0.1, 0.2, 0.3, 0.4}},
            {"default_guidance_scale", 1.3},
            {"default_diffusion_steps", 2}};
}

void test_runtime_config() {
    RuntimeConfig config;
    check_no_throw([&] { config = parse_runtime_config(runtime_json().dump()); },
                   "a consistent runtime.json");
    expect(config.vocab_size == 300 && config.hop_length == 4 &&
               config.control_token_ids == std::vector<std::int32_t>({256, 257, 258, 259}),
           "runtime.json values are read");
    const auto rejects = [](const std::string& what,
                            const std::function<void(nlohmann::json&)>& edit, const char* message) {
        auto json = runtime_json();
        edit(json);
        const auto text = json.dump();
        check_throws_as<std::runtime_error>([&] { parse_runtime_config(text); }, what, message);
    };
    // Integers are checked before any conversion: no truncation, no wraparound.
    rejects(
        "a fractional hidden_size", [](auto& j) { j["hidden_size"] = 8.5; },
        "hidden_size must be a JSON integer");
    rejects(
        "a hidden_size above int32", [](auto& j) { j["hidden_size"] = 4294967304ULL; },
        "hidden_size lies outside");
    rejects(
        "an unsigned capacity above int64",
        [](auto& j) { j["capacity"] = 18446744073709551615ULL; }, "capacity lies outside");
    rejects("a negative capacity", [](auto& j) { j["capacity"] = -64; }, "capacity lies outside");
    rejects(
        "a string token id", [](auto& j) { j["eos_token_id"] = "256"; },
        "eos_token_id must be a JSON integer");
    rejects(
        "a token id outside the vocabulary", [](auto& j) { j["audio_token_id"] = 300; },
        "audio_token_id lies outside");
    rejects(
        "control ids that are not the four token ids",
        [](auto& j) { j["control_token_ids"] = {256, 257, 258, 260}; }, "control tokens");
    rejects("a missing vocabulary size", [](auto& j) { j.erase("vocab_size"); }, "malformed");
    // Floats are finite float32 values, and derived values stay usable.
    rejects(
        "a guidance scale above float32", [](auto& j) { j["default_guidance_scale"] = 1e39; },
        "finite float32");
    rejects(
        "a loudness target whose gain overflows", [](auto& j) { j["target_dbfs"] = 4000.0; },
        "loudness gain");
    rejects(
        "an eps below the smallest normal float", [](auto& j) { j["normalize_eps"] = 1e-45; },
        "loudness gain");
    rejects(
        "a rotary frequency above 1", [](auto& j) { j["rope_inv_freq"] = {1.5, 0.01}; },
        "rope_inv_freq");
    rejects(
        "repeated training sigmas", [](auto& j) { j["train_sigmas"] = {0.1, 0.2, 0.2, 0.4}; },
        "strictly increasing");
    rejects(
        "a zero training sigma", [](auto& j) { j["train_sigmas"] = {0.0, 0.2, 0.3, 0.4}; },
        "strictly increasing");
    rejects(
        "default steps beyond the schedule", [](auto& j) { j["default_diffusion_steps"] = 5; },
        "default_diffusion_steps lies outside");
    // linspace(0, 3, 5) rounds 1.5 and 2.25 to timestep 2.
    rejects(
        "default steps that repeat a timestep", [](auto& j) { j["default_diffusion_steps"] = 4; },
        "repeats a noise level");
    rejects(
        "voice samples that are not whole frames", [](auto& j) { j["max_voice_samples"] = 18; },
        "whole frames");
    rejects(
        "a KV cache too large to size",
        [](auto& j) {
            j["num_layers"] = 2147483647;
            j["capacity"] = 2147483647;
            j["num_kv_heads"] = 2147483647;
        },
        "overflows");
    // nlohmann keeps the last of two equal keys; the runtime refuses the file.
    const auto text = runtime_json().dump();
    const auto duplicated = "{\"hidden_size\": 16, " + text.substr(1);
    check_throws_as<std::runtime_error>([&] { parse_runtime_config(duplicated); }, "a repeated key",
                                        "repeats the key hidden_size");
    expect(usable_gain(-25.0F, 1e-6), "the released loudness settings give a usable gain");
}

// The tensors of engines built for runtime_json(), as their modules report them.
EngineTensors lm_tensors(const RuntimeConfig& c, std::int64_t rows) {
    const auto type = DType::kBFloat16;
    const std::int64_t kv = c.num_kv_heads;
    const std::int64_t dim = c.head_dim;
    EngineTensors tensors;
    tensors["inputs_embeds"] = {
        true, true, type, {}, {1, 1, c.hidden_size}, {1, rows, c.hidden_size}};
    tensors["rope_cos"] = {true, true, DType::kFloat32, {}, {1, dim}, {rows, dim}};
    tensors["rope_sin"] = tensors["rope_cos"];
    for (std::int32_t layer = 0; layer < c.num_layers; ++layer) {
        const auto suffix = std::to_string(layer);
        tensors["past_key_" + suffix] = {
            true, true, type, {}, {1, 0, kv, dim}, {1, c.capacity - 1, kv, dim}};
        tensors["past_value_" + suffix] = tensors["past_key_" + suffix];
        tensors["present_key_" + suffix] = {false, false, type, {1, rows, kv, dim}, {}, {}};
        tensors["present_value_" + suffix] = tensors["present_key_" + suffix];
    }
    tensors["hidden"] = {false, false, type, {1, c.hidden_size}, {}, {}};
    tensors["log_normalizer"] = {false, false, DType::kFloat32, {1}, {}, {}};
    tensors["control_logits"] = {false, false, DType::kFloat32, {1, 4}, {}, {}};
    return tensors;
}

EngineTensors embed_tensors(const RuntimeConfig& c) {
    EngineTensors tensors;
    tensors["token_ids"] = {true, true, DType::kInt32, {}, {1}, {512}};
    tensors["embeds"] = {false, false, DType::kBFloat16, {512, c.hidden_size}, {}, {}};
    return tensors;
}

EngineTensors head_tensors(const RuntimeConfig& c) {
    EngineTensors tensors;
    tensors["noisy_latent"] = {
        true, true, DType::kFloat32, {}, {1, c.latent_size}, {2, c.latent_size}};
    tensors["timestep"] = {true, true, DType::kFloat32, {}, {1}, {2}};
    tensors["condition"] = {
        true, true, DType::kBFloat16, {}, {1, c.hidden_size}, {2, c.hidden_size}};
    tensors["velocity"] = {false, false, DType::kFloat32, {2, c.latent_size}, {}, {}};
    return tensors;
}

EngineTensors feedback_tensors(const RuntimeConfig& c) {
    EngineTensors tensors;
    tensors["latent"] = {true, false, DType::kFloat32, {1, c.latent_size}, {}, {}};
    tensors["state_in"] = {true, false, DType::kBFloat16, {c.stream_state_size}, {}, {}};
    tensors["audio"] = {false, false, DType::kFloat32, {1, c.hop_length}, {}, {}};
    tensors["next_embed"] = {false, false, DType::kBFloat16, {1, c.hidden_size}, {}, {}};
    tensors["state_out"] = {false, false, DType::kBFloat16, {c.stream_state_size}, {}, {}};
    return tensors;
}

EngineTensors voice_tensors(const RuntimeConfig& c) {
    const std::int64_t frames = c.max_voice_samples / c.hop_length;
    EngineTensors tensors;
    tensors["audio"] = {
        true, true, DType::kFloat32, {}, {1, 1, 1, c.hop_length}, {1, 1, 1, c.max_voice_samples}};
    tensors["noise"] = {
        true, true, DType::kFloat32, {}, {1, 1, c.latent_size}, {1, frames, c.latent_size}};
    tensors["noise_scale"] = {true, false, DType::kFloat32, {1}, {}, {}};
    tensors["embeds"] = {false, false, DType::kBFloat16, {1, frames, c.hidden_size}, {}, {}};
    return tensors;
}

void test_engine_io() {
    const auto c = parse_runtime_config(runtime_json().dump());
    check_no_throw([&] { check_lm_io(lm_tensors(c, c.prefill_chunk), c, c.prefill_chunk); },
                   "the prefill profile of a matching language model");
    check_no_throw([&] { check_lm_io(lm_tensors(c, 1), c, 1); }, "its decode profile");
    check_no_throw([&] { check_embed_io(embed_tensors(c), c); }, "a matching embedding engine");
    check_no_throw([&] { check_head_io(head_tensors(c), c); }, "a matching diffusion head");
    check_no_throw([&] { check_feedback_io(feedback_tensors(c), c); },
                   "a matching feedback engine");
    check_no_throw([&] { check_voice_io(voice_tensors(c), c); }, "a matching voice encoder");
    const auto names = lm_tensor_names(c);
    expect(std::find(names.begin(), names.end(), "past_key_2") != names.end(),
           "the language-model names include the first layer past runtime.json's");

    // Engines that do not match the buffers the runtime sizes from runtime.json.
    auto wide = lm_tensors(c, c.prefill_chunk);
    wide["hidden"].shape = {1, 2 * c.hidden_size};
    check_throws_as<std::runtime_error>([&] { check_lm_io(wide, c, c.prefill_chunk); },
                                        "a wider hidden output", "hidden has shape");
    // An output resolved with the past rows would be written past the cache's end.
    auto whole_cache = lm_tensors(c, c.prefill_chunk);
    whole_cache["present_key_0"].shape = {1, c.capacity - 1 + c.prefill_chunk, c.num_kv_heads,
                                          c.head_dim};
    check_throws_as<std::runtime_error>([&] { check_lm_io(whole_cache, c, c.prefill_chunk); },
                                        "a present output that holds the past rows",
                                        "present_key_0 has shape");
    auto short_cache = lm_tensors(c, c.prefill_chunk);
    short_cache["past_key_1"].profile_max[1] = c.capacity - 2;
    check_throws_as<std::runtime_error>([&] { check_lm_io(short_cache, c, c.prefill_chunk); },
                                        "a cache profile shorter than capacity",
                                        "past_key_1 profile spans");
    auto deeper = lm_tensors(c, 1);
    deeper["past_key_2"] = deeper["past_key_1"];
    check_throws_as<std::runtime_error>([&] { check_lm_io(deeper, c, 1); },
                                        "an engine with another layer", "more layers");
    auto float_input = lm_tensors(c, 1);
    float_input["inputs_embeds"].dtype = DType::kFloat32;
    check_throws_as<std::runtime_error>([&] { check_lm_io(float_input, c, 1); },
                                        "a float32 input in a bfloat16 bundle",
                                        "another data type");
    check_throws_as<std::runtime_error>([&] { check_lm_io(lm_tensors(c, 1), c, c.prefill_chunk); },
                                        "a decode profile used for prefill",
                                        "inputs_embeds profile spans");
    auto no_output = embed_tensors(c);
    no_output.erase("embeds");
    check_throws_as<std::runtime_error>([&] { check_embed_io(no_output, c); },
                                        "an embedding engine without embeds", "is not an output");
    auto single = head_tensors(c);
    single["noisy_latent"].profile_max[0] = 1;
    check_throws_as<std::runtime_error>([&] { check_head_io(single, c); },
                                        "a diffusion head for one row",
                                        "noisy_latent profile spans");
    auto state = feedback_tensors(c);
    state["state_in"].shape = {c.stream_state_size + 1};
    check_throws_as<std::runtime_error>([&] { check_feedback_io(state, c); },
                                        "another stream-state size", "state_in has shape");
    auto shorter = voice_tensors(c);
    shorter["audio"].profile_max[3] = c.max_voice_samples - c.hop_length;
    check_throws_as<std::runtime_error>([&] { check_voice_io(shorter, c); },
                                        "a voice encoder for shorter prompts",
                                        "audio profile spans");
}

} // namespace

int main() {
    try {
        const auto fixtures = load_fixtures();
        test_noise(fixtures);
        test_timesteps(fixtures);
        test_schedulers(fixtures);
        test_schedule_limits(fixtures);
        test_scheduler_state(fixtures);
        test_scripts(fixtures);
        test_wav();
        test_tokenizer();
        test_tokenizer_contractions_and_budget();
        test_tokenizer_repeated_keys();
        test_tokenizer_required_options();
        test_runtime_config();
        test_engine_io();
    } catch (const std::exception& error) {
        std::cerr << "vibevoice host test error: " << error.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " vibevoice host check(s) failed\n";
        return 1;
    }
    std::cout << "vibevoice host checks passed\n";
    return 0;
}
