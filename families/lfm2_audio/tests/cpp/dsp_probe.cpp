/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/audio_dsp.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
std::vector<float> read(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("cannot read input");
    const auto size = f.tellg();
    if (size < 0 || size % sizeof(float))
        throw std::runtime_error("invalid input size");
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(values.data()), size);
    return values;
}
int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::runtime_error(
                "usage: dsp_probe mel|resample|istft input output rate filters");
        auto input = read(argv[2]);
        std::vector<float> output;
        const std::string mode = argv[1];
        if (mode == "mel") {
            auto result = trtmc::lfm2_audio::extract_mel(input.data(), input.size(),
                                                         std::stoi(argv[4]), read(argv[5]));
            output = std::move(result.values);
            std::cout << result.frames << '\n';
        } else if (mode == "resample")
            output = trtmc::lfm2_audio::resample(input.data(), input.size(), std::stoi(argv[4]));
        else if (mode == "istft_stream") {
            const int chunk = std::stoi(argv[4]);
            if (chunk <= 0 || input.size() % 1282)
                throw std::invalid_argument("invalid stream geometry");
            trtmc::lfm2_audio::StreamingIstft synthesis;
            const auto frames = input.size() / 1282;
            for (std::size_t i = 0; i < frames; i += static_cast<std::size_t>(chunk)) {
                auto values = synthesis.append(input.data() + i * 1282,
                                               std::min<std::size_t>(chunk, frames - i));
                output.insert(output.end(), values.begin(), values.end());
                if (output.size() != (std::min<std::size_t>(frames, i + chunk) * 320 > 480
                                          ? std::min<std::size_t>(frames, i + chunk) * 320 - 480
                                          : 0))
                    throw std::runtime_error("incorrect stable ISTFT prefix");
            }
            auto tail = synthesis.finish();
            output.insert(output.end(), tail.begin(), tail.end());
            if (!synthesis.finish().empty())
                throw std::runtime_error("ISTFT finish must be idempotent");
            bool rejected = false;
            try {
                synthesis.append(input.data(), frames);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("finished ISTFT accepted input");
            synthesis.reset();
            auto again = synthesis.append(input.data(), frames);
            tail = synthesis.finish();
            again.insert(again.end(), tail.begin(), tail.end());
            if (again != output)
                throw std::runtime_error("ISTFT reset changed the waveform");
        } else if (mode == "istft")
            output = trtmc::lfm2_audio::inverse_spectrogram(input);
        else
            throw std::runtime_error("unknown probe mode");
        std::ofstream f(argv[3], std::ios::binary);
        f.write(reinterpret_cast<const char*>(output.data()),
                static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!f)
            throw std::runtime_error("cannot write output");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
