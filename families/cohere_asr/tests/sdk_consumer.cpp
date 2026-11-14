/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <trtmc/audio.hpp>
#include <vector>
std::vector<float> read_audio(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(float))
        throw std::runtime_error("invalid PCM file");
    auto size = file.tellg();
    std::vector<float> values(static_cast<std::size_t>(size) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), size);
    if (!file)
        throw std::runtime_error("cannot read PCM");
    return values;
}
int main(int argc, char** argv) {
    if (argc < 4)
        return 2;
    try {
        std::vector<trtmc::SpeechTranscriptionResult> results;
        {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            auto task = model.task<trtmc::SpeechTranscription>();
            for (int i = 3; i < argc; ++i) {
                auto audio = read_audio(argv[i]);
                results.push_back(
                    task.run({{{audio.data(), audio.size()}, 16000, 1}, std::string("ar")}));
            }
            auto audio = read_audio(argv[3]);
            results.push_back(
                task.run({{{audio.data(), audio.size()}, 16000, 1}, std::string("ar")}));
        }
        for (const auto& result : results) {
            std::cout << "{\"text_hex\":\"";
            for (unsigned char c : result.text())
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<unsigned int>(c);
            std::cout << std::dec << "\",\"token_ids\":[";
            for (std::size_t i = 0; i < result.token_ids().size(); ++i)
                std::cout << (i ? "," : "") << result.token_ids()[i];
            std::cout << "]}\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
