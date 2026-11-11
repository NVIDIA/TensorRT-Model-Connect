/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++ SDK consumer for the family-owned Qwen3-ASR end-to-end test: transcribes one or more
// float32 PCM files (comma-separated) in order on one loaded model through
// trtmc::SpeechTranscription and prints one JSON line of transcript and token ids per file. Later
// requests must not see the KV-cache rows an earlier, longer request left behind.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <trtmc/audio.hpp>
#include <vector>

namespace {

std::string json_string(std::string_view value) {
    std::string out = "\"";
    for (const char raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += raw;
        } else if (c < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        } else {
            out += raw;
        }
    }
    return out + "\"";
}

std::vector<float> read_pcm(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (!input || bytes <= 0 || bytes % sizeof(float) != 0)
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
    if (argc != 7 && argc != 8) {
        std::cerr << "Usage: consumer BUNDLE RUNTIME_ROOT PCM_F32[,PCM_F32...] RATE CHANNELS "
                     "MAX_OUTPUT_TOKENS [SOURCE_LANGUAGE]\n";
        return 2;
    }
    try {
        const auto rate = std::stoul(argv[4]);
        const auto channels = std::stoul(argv[5]);
        const auto limit = std::stoll(argv[6]);
        if (rate == 0 || rate > UINT32_MAX || channels == 0 || channels > UINT32_MAX || limit <= 0)
            throw std::invalid_argument("invalid audio metadata or token limit");
        std::vector<std::vector<float>> inputs;
        std::istringstream paths(argv[3]);
        for (std::string path; std::getline(paths, path, ',');)
            inputs.push_back(read_pcm(path));

        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::SpeechTranscription>();
        for (const auto& pcm : inputs) {
            trtmc::SpeechTranscriptionRequest request{
                {{pcm.data(), pcm.size()},
                 static_cast<std::uint32_t>(rate),
                 static_cast<std::uint32_t>(channels)},
                argc == 8 ? std::optional<std::string>{argv[7]} : std::nullopt};
            const auto result =
                task.run(request, {{"max_output_tokens", static_cast<std::int64_t>(limit)}});
            std::string ids;
            for (const auto id : result.token_ids())
                ids += (ids.empty() ? "" : ",") + std::to_string(id);
            std::cout << "{\"task\":\"speech_transcription\",\"text\":"
                      << json_string(result.text()) << ",\"token_ids\":[" << ids << "]}\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
