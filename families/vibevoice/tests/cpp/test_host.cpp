/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side checks of the vibevoice runtime against tests/data/host_fixtures.json:
// the DPM-Solver against diffusers (float32 and bfloat16 samples), the noise
// stream against its Python mirror, script parsing and prompt rendering against
// the reference chat template, WAV reading and voice-prompt preparation. With
// VIBEVOICE_TOKENIZER_JSON pointing at the Qwen2.5 tokenizer.json, the rendered
// prompts are also tokenized and compared with the reference token ids.

#include "families/vibevoice/runtime/noise.h"
#include "families/vibevoice/runtime/scheduler.h"
#include "families/vibevoice/runtime/script.h"
#include "families/vibevoice/runtime/tokenizer.h"
#include "families/vibevoice/runtime/voice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trtmc::vibevoice;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

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

void test_noise(const nlohmann::json& fixtures) {
    NoiseStream noise(fixtures.at("noise").at("seed").get<std::uint64_t>());
    for (const auto& expected : fixtures.at("noise").at("values"))
        expect(noise.normal() == expected.get<float>(), "noise stream matches its Python mirror");
}

void test_timesteps(const nlohmann::json& fixtures) {
    for (const auto& [steps, expected] : fixtures.at("timesteps").items()) {
        const auto actual = linspace_timesteps(1000, std::stoi(steps));
        expect(actual == expected.get<std::vector<std::int64_t>>(),
               "timesteps for " + steps + " steps");
    }
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
        auto sample = item.at("initial").get<std::vector<float>>();
        solver.reset();
        double worst = 0.0;
        for (std::int32_t step = 0; step < solver.steps(); ++step) {
            const auto velocity = item.at("velocities").at(step).get<std::vector<float>>();
            solver.step(step, velocity, sample);
            const auto expected = item.at("outputs").at(step).get<std::vector<float>>();
            for (std::size_t index = 0; index < sample.size(); ++index) {
                const double scale = std::max(1.0, std::fabs(static_cast<double>(expected[index])));
                worst = std::max(
                    worst, std::fabs(static_cast<double>(sample[index] - expected[index])) / scale);
            }
        }
        // float32 math in a different evaluation order than torch can differ by a few ulp;
        // a bfloat16 rounding decision flipped by such an ulp moves one value by one bf16 step.
        expect(worst <= (bf16 ? 8e-3 : 1e-5),
               label + " matches diffusers (worst " + std::to_string(worst) + ")");
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
        tokenizer = CreateBpeTokenizer(json.data(), json.size());
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
    const auto voiced = render_prompt(frame(), {{0, "Hi."}}, {2});
    expect(
        voiced.find(
            " Voice input:\n Speaker 0:<|vision_start|><|vision_pad|><|vision_pad|><|vision_end|>\n"
            " Text input:\n") != std::string::npos,
        "voice placeholders follow the reference template");
}

void test_wav() {
    const std::string path = "vibevoice_test_voice.wav";
    const std::vector<std::int16_t> pcm = {0, 16384, -16384, 32767, -32768, 1};
    {
        std::ofstream file(path, std::ios::binary);
        auto put32 = [&](std::uint32_t value) {
            file.write(reinterpret_cast<const char*>(&value), 4);
        };
        auto put16 = [&](std::uint16_t value) {
            file.write(reinterpret_cast<const char*>(&value), 2);
        };
        file.write("RIFF", 4);
        put32(36 + static_cast<std::uint32_t>(pcm.size() * 2));
        file.write("WAVEfmt ", 8);
        put32(16);
        put16(1);
        put16(1);
        put32(24000);
        put32(48000);
        put16(2);
        put16(16);
        file.write("data", 4);
        put32(static_cast<std::uint32_t>(pcm.size() * 2));
        file.write(reinterpret_cast<const char*>(pcm.data()),
                   static_cast<std::streamsize>(pcm.size() * 2));
    }
    const auto audio = read_wav(path);
    std::remove(path.c_str());
    expect(audio.sample_rate == 24000 && audio.channels == 1 && audio.samples.size() == pcm.size(),
           "PCM16 WAV header is read");
    expect(audio.samples[1] == 0.5F && audio.samples[4] == -1.0F, "PCM16 samples scale by 1/32768");
    const auto prepared = prepare_voice(audio.samples, -25.0F, 1e-6F, 4);
    expect(prepared.size() == 8 && prepared[6] == 0.0F && prepared[7] == 0.0F,
           "voice is zero-padded to whole frames");
    double peak = 0.0;
    for (const float sample : prepared)
        peak = std::max(peak, static_cast<double>(std::fabs(sample)));
    expect(peak <= 1.0, "voice peak is limited to 1");
}

} // namespace

int main() {
    try {
        const auto fixtures = load_fixtures();
        test_noise(fixtures);
        test_timesteps(fixtures);
        test_schedulers(fixtures);
        test_scripts(fixtures);
        test_wav();
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
