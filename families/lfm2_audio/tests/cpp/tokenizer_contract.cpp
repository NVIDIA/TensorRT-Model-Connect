/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/byte_level_decoder.h"
#include "families/lfm2_audio/runtime/pretokenizer.h"
#include "families/lfm2_audio/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace trtmc::lfm2_audio {
// Substitute only bundle storage. The factory itself and all tokenizer
// operations are compiled from the unchanged production source.
struct BundleReader {
    std::vector<char> tokenizer;
};
const std::vector<char>& require_section(const BundleReader& reader, std::string_view name) {
    if (name != "tokenizer.json")
        throw std::logic_error("unexpected bundle section");
    return reader.tokenizer;
}
#include "tokenizer_factory.inc"
} // namespace trtmc::lfm2_audio

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    std::ifstream input(argv[1], std::ios::binary);
    trtmc::lfm2_audio::BundleReader reader{
        std::vector<char>(std::istreambuf_iterator<char>(input), {})};
    const bool reject = std::string(argv[2]) == "reject";
    try {
        const auto tokenizer = trtmc::lfm2_audio::create_tokenizer(reader);
        const auto ids = tokenizer->encode("hello");
        if (reject) {
            std::cerr << "unsupported tokenizer accepted; ids:";
            for (auto id : ids)
                std::cerr << ' ' << id;
            std::cerr << '\n';
            return 1;
        }
        if (ids != std::vector<std::int32_t>{2, 3, 4, 4, 5} || tokenizer->encode("hello") != ids ||
            tokenizer->decode(ids) != "hello")
            throw std::runtime_error("pinned tokenizer added special tokens or changed text");
    } catch (const std::runtime_error& error) {
        if (reject && std::string(error.what()).find("pre_tokenizer") != std::string::npos)
            return 0;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
