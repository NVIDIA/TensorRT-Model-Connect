/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

struct Options {
    std::string bundle;
    std::string runtime_root;
    std::int32_t chunk_frames{16};
    std::int32_t max_new_tokens{750};
};

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " MODEL.bundle --runtime-root DIR [--chunk-frames N]"
                 " [--max-new-tokens N]\n";
}

std::string take_value(int& index, int argc, char** argv, const std::string& option) {
    if (++index >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

std::int32_t positive_integer(const std::string& value, const std::string& option) {
    std::int32_t parsed = 0;
    const char* end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed <= 0)
        throw std::invalid_argument(option + " requires a positive integer");
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool saw_runtime_root = false;
    bool saw_chunk_frames = false;
    bool saw_max_new_tokens = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--runtime-root") {
            if (saw_runtime_root)
                throw std::invalid_argument("--runtime-root may be specified only once");
            saw_runtime_root = true;
            options.runtime_root = take_value(index, argc, argv, argument);
        } else if (argument == "--chunk-frames") {
            if (saw_chunk_frames)
                throw std::invalid_argument("--chunk-frames may be specified only once");
            saw_chunk_frames = true;
            options.chunk_frames =
                positive_integer(take_value(index, argc, argv, argument), argument);
        } else if (argument == "--max-new-tokens") {
            if (saw_max_new_tokens)
                throw std::invalid_argument("--max-new-tokens may be specified only once");
            saw_max_new_tokens = true;
            options.max_new_tokens =
                positive_integer(take_value(index, argc, argv, argument), argument);
        } else if (argument.empty() || argument.front() == '-') {
            throw std::invalid_argument("unknown option: " + argument);
        } else if (options.bundle.empty()) {
            options.bundle = argument;
        } else {
            throw std::invalid_argument("only one bundle may be specified");
        }
    }
    if (options.bundle.empty())
        throw std::invalid_argument("a bundle is required");
    if (options.runtime_root.empty())
        throw std::invalid_argument("--runtime-root is required");
    return options;
}

bool blank(const std::string& line) {
    return line.find_first_not_of(" \t\r\n") == std::string::npos;
}

void write_floats(const float* values, std::size_t count) {
    if (count == 0)
        return;
    if (values == nullptr || std::fwrite(values, sizeof(float), count, stdout) != count)
        throw std::runtime_error("failed to write FP32 PCM to stdout");
    if (std::fflush(stdout) != 0)
        throw std::runtime_error("failed to flush FP32 PCM to stdout");
}

int run(const Options& options) {
    std::cerr << "[audio-streaming] loading bundle\n";
    auto task = trtmc::load_task(options.bundle, options.runtime_root);
    auto* streaming = dynamic_cast<trtmc::IStreamingAudioGeneration*>(task.get());
    if (streaming == nullptr)
        throw std::runtime_error("bundle does not implement streaming audio generation");

    trtmc::AudioGenerationConfig config;
    config.max_new_tokens = options.max_new_tokens;
    std::cerr << "[audio-streaming] ready; reading one prompt per line\n";

    std::string prompt;
    std::int32_t utterance = 0;
    while (std::getline(std::cin, prompt)) {
        if (blank(prompt))
            continue;
        ++utterance;
        std::int64_t callback_samples = 0;
        std::int32_t sample_rate = 0;
        std::cerr << "[audio-streaming] utterance " << utterance << " start\n";
        const auto reported_samples = streaming->generate_audio_streaming(
            prompt, config,
            [&](const float* samples, std::int32_t count, std::int32_t rate) {
                if (samples == nullptr || count <= 0 || rate <= 0)
                    throw std::runtime_error("streaming callback returned invalid audio metadata");
                if (sample_rate != 0 && sample_rate != rate)
                    throw std::runtime_error("streaming callback changed sample rate");
                sample_rate = rate;
                write_floats(samples, static_cast<std::size_t>(count));
                callback_samples += count;
            },
            options.chunk_frames);
        if (reported_samples <= 0 || callback_samples <= 0)
            throw std::runtime_error("streaming task produced no audio samples");

        const float utterance_end = 0.0F;
        write_floats(&utterance_end, 1);
        std::cerr << "[audio-streaming] utterance " << utterance
                  << " done; samples=" << callback_samples << "; sample_rate=" << sample_rate
                  << "; reported_samples=" << reported_samples << '\n';
    }
    if (std::cin.bad())
        throw std::runtime_error("failed to read prompts from stdin");
    std::cerr << "[audio-streaming] EOF; utterances=" << utterance << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        usage(argv[0]);
        return 0;
    }
    try {
        return run(parse_options(argc, argv));
    } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << '\n';
        usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
