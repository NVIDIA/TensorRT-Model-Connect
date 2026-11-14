/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Direct C++ SDK consumer for the hubert family. Loads one bundle and runs
 * each float32 PCM file, in order, through the public SpeechTranscription
 * Task, printing one JSON object per clip.
 *
 * Usage:
 *   test_hubert_sdk_cpp BUNDLE RUNTIME_ROOT RATE CHANNELS PCM_F32 [PCM_F32 ...]
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <trtmc/audio.hpp>
#include <trtmc/trtmc.hpp>
#include <vector>

namespace {

std::vector<float> read_pcm(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    if (!input || bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::invalid_argument("input must be nonempty float32 PCM: " + path);
    std::vector<float> pcm(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(pcm.data()), bytes);
    if (!input)
        throw std::runtime_error("incomplete PCM input: " + path);
    return pcm;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT RATE CHANNELS PCM_F32 [PCM_F32 ...]\n";
        return 2;
    }
    try {
        const auto rate = std::stoul(argv[3]);
        const auto channels = std::stoul(argv[4]);
        if (rate == 0 || rate > UINT32_MAX || channels == 0 || channels > UINT32_MAX)
            throw std::invalid_argument("invalid audio rate or channel count");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::SpeechTranscription>();
        for (int index = 5; index < argc; ++index) {
            const auto pcm = read_pcm(argv[index]);
            const trtmc::SpeechTranscriptionRequest request{{{pcm.data(), pcm.size()},
                                                             static_cast<std::uint32_t>(rate),
                                                             static_cast<std::uint32_t>(channels)},
                                                            {}};
            const auto result = task.run(request);
            const auto ids = result.token_ids();
            std::cout << nlohmann::json({{"input", argv[index]},
                                         {"text", std::string(result.text())},
                                         {"token_ids", std::vector<std::int32_t>(
                                                           ids.data(), ids.data() + ids.size())}})
                             .dump()
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
