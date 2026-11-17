/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test: one document
 * image as 8-bit RGB pixels and one question in, the answer text out. */
#include <cstdint>
#include <cstdlib>
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

std::uint32_t dimension(const char* text) {
    const auto value = std::stoul(text);
    if (value == 0 || value > 1U << 16)
        throw std::invalid_argument("image dimensions must lie in 1..65536");
    return static_cast<std::uint32_t>(value);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT RGB_U8_FILE WIDTH HEIGHT QUESTION\n";
        return 2;
    }
    try {
        const auto width = dimension(argv[4]);
        const auto height = dimension(argv[5]);
        std::ifstream file(argv[3], std::ios::binary);
        const std::vector<std::uint8_t> pixels{std::istreambuf_iterator<char>(file),
                                               std::istreambuf_iterator<char>()};
        if (pixels.size() != static_cast<std::size_t>(width) * height * 3)
            throw std::runtime_error("RGB file size does not match WIDTH x HEIGHT x 3");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<trtmc::ImagesTextToText>();
        if (!task.config_fields().empty())
            throw std::runtime_error("layoutlm must expose no runtime Config fields");
        std::vector<trtmc::ImagesTextPart> parts;
        parts.emplace_back(trtmc::ImageInput{
            trtmc::Span<const std::uint8_t>{pixels.data(), pixels.size()}, height, width});
        parts.emplace_back(trtmc::TextPart{argv[6]});
        const auto result = task.run(trtmc::ImagesTextToTextRequest::from_parts(std::move(parts)));
        std::cout << "{\"task\":\"images_text_to_text\",\"format\":\"uint8\",\"text\":";
        json_string(result.text());
        std::cout << ",\"token_ids\":[";
        const auto ids = result.token_ids();
        for (std::size_t index = 0; index < ids.size(); ++index)
            std::cout << (index ? "," : "") << ids[index];
        std::cout << "]}\n";
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
