/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test.
 *
 *   BUNDLE RUNTIME_ROOT MAX_NEW_TOKENS PROMPT IMAGE.rgb HEIGHT WIDTH [IMAGE.rgb HEIGHT WIDTH]...
 *
 * Each IMAGE.rgb holds packed HWC RGB8 pixels. Prints one JSON line with the
 * generated text and token ids.
 */
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/language.hpp>
#include <vector>

namespace {
void json_string(std::string_view value) {
    std::cout << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\')
            std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32)
            std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(byte) << std::dec << std::setfill(' ');
        else
            std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}

std::vector<std::uint8_t> read_pixels(const std::string& path, std::uint32_t height,
                                      std::uint32_t width) {
    std::ifstream input(path, std::ios::binary);
    std::vector<std::uint8_t> pixels((std::istreambuf_iterator<char>(input)),
                                     std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof())
        throw std::runtime_error("failed to read " + path);
    if (pixels.size() != static_cast<std::size_t>(height) * width * 3)
        throw std::runtime_error(path + " does not hold HEIGHT x WIDTH RGB8 pixels");
    return pixels;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 8 || (argc - 5) % 3 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT MAX_NEW_TOKENS PROMPT IMAGE.rgb HEIGHT WIDTH...\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::ImagesTextToText>();
        std::vector<std::vector<std::uint8_t>> storage;
        std::vector<trtmc::ImagesTextPart> parts;
        for (int i = 5; i < argc; i += 3) {
            const auto height = static_cast<std::uint32_t>(std::stoul(argv[i + 1]));
            const auto width = static_cast<std::uint32_t>(std::stoul(argv[i + 2]));
            storage.push_back(read_pixels(argv[i], height, width));
        }
        for (int i = 5, image = 0; i < argc; i += 3, ++image) {
            const auto& pixels = storage[static_cast<std::size_t>(image)];
            parts.emplace_back(
                trtmc::ImageInput{trtmc::Span<const std::uint8_t>{pixels.data(), pixels.size()},
                                  static_cast<std::uint32_t>(std::stoul(argv[i + 1])),
                                  static_cast<std::uint32_t>(std::stoul(argv[i + 2]))});
        }
        parts.emplace_back(trtmc::TextPart{argv[4]});
        trtmc::Config config;
        config.add("max_new_tokens", static_cast<std::int64_t>(std::stoll(argv[3])));
        const auto result =
            task.run(trtmc::ImagesTextToTextRequest::from_parts(std::move(parts)), config);
        std::cout << "{\"task\":\"images_text_to_text\",\"text\":";
        json_string(result.text());
        std::cout << ",\"token_ids\":[";
        const auto ids = result.token_ids();
        for (std::size_t i = 0; i < ids.size(); ++i)
            std::cout << (i ? "," : "") << ids[i];
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the result");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
