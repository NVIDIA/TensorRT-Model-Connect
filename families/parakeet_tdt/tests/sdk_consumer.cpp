/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <trtmc/audio.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: consumer BUNDLE RUNTIME_ROOT PCM_F32 RATE CHANNELS MAX_TOKENS\n";
        return 2;
    }
    try {
        const auto rate = std::stoul(argv[4]);
        const auto channels = std::stoul(argv[5]);
        const auto limit = std::stoll(argv[6]);
        if (rate > UINT32_MAX || !rate || channels > UINT32_MAX || !channels || limit <= 0)
            throw std::invalid_argument("invalid audio metadata or token limit");
        std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
        const auto bytes = input.tellg();
        if (!input || bytes <= 0 || bytes % sizeof(float) != 0)
            throw std::invalid_argument("input must be nonempty float32 PCM");
        std::vector<float> pcm(static_cast<size_t>(bytes) / sizeof(float));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(pcm.data()), bytes);
        if (!input)
            throw std::runtime_error("incomplete PCM input");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::SpeechTranscription>();
        trtmc::SpeechTranscriptionRequest request{{{pcm.data(), pcm.size()},
                                                   static_cast<uint32_t>(rate),
                                                   static_cast<uint32_t>(channels)},
                                                  {}};
        auto result = task.run(request, {{"max_new_tokens", static_cast<int64_t>(limit)}});
        std::cout << nlohmann::json({{"text", std::string(result.text())}}).dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
