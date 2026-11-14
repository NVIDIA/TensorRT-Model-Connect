/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Direct C++ SDK call of moss_transcribe_diarize's speech_transcription Task.
// Prints the transcript, its ids and its segments as one JSON line.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <trtmc/audio.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: consumer BUNDLE RUNTIME_ROOT PCM_F32_16KHZ_MONO MAX_NEW_TOKENS\n";
        return 2;
    }
    try {
        const auto limit = std::stoll(argv[4]);
        std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
        const auto bytes = static_cast<std::streamoff>(input.tellg());
        if (!input || bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
            throw std::invalid_argument("input must be nonempty float32 PCM");
        std::vector<float> pcm(static_cast<std::size_t>(bytes) / sizeof(float));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(pcm.data()), bytes);
        if (!input)
            throw std::runtime_error("incomplete PCM input");

        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::SpeechTranscription>();
        const trtmc::SpeechTranscriptionRequest request{{{pcm.data(), pcm.size()}, 16000U, 1U}, {}};
        const auto result = task.run(request, {{"max_new_tokens", static_cast<int64_t>(limit)}});

        nlohmann::json segments = nlohmann::json::array();
        for (const auto& segment : result.segments())
            segments.push_back({{"start_seconds", segment.start_seconds},
                                {"end_seconds", segment.end_seconds},
                                {"text", std::string(segment.text)}});
        const auto ids = result.token_ids();
        std::cout << nlohmann::json({{"text", std::string(result.text())},
                                     {"token_ids", std::vector<int32_t>(ids.begin(), ids.end())},
                                     {"segments", segments}})
                         .dump()
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
