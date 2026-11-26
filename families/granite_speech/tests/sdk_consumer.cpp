/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++ SDK call of speech_transcription: prints {"text", "token_ids"} as one JSON line.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/audio.hpp>
#include <vector>

namespace {

std::string escape(std::string_view text) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: consumer BUNDLE RUNTIME_ROOT PCM_F32 RATE MAX_NEW_TOKENS\n";
        return 2;
    }
    try {
        const auto rate = std::stoul(argv[4]);
        const auto limit = std::stoll(argv[5]);
        if (rate == 0 || rate > UINT32_MAX || limit <= 0)
            throw std::invalid_argument("invalid sample rate or token limit");
        std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
        const auto bytes = input.tellg();
        if (!input || bytes <= 0 || bytes % sizeof(float) != 0)
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
        trtmc::SpeechTranscriptionRequest request{
            {{pcm.data(), pcm.size()}, static_cast<std::uint32_t>(rate), 1}, {}};
        const auto result =
            task.run(request, {{"max_new_tokens", static_cast<std::int64_t>(limit)}});
        std::cout << "{\"text\":\"" << escape(result.text()) << "\",\"token_ids\":[";
        const auto ids = result.token_ids();
        for (std::size_t index = 0; index < ids.size(); ++index)
            std::cout << (index ? "," : "") << ids[index];
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
