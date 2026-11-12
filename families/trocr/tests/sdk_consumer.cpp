/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned TrOCR end-to-end test.
// Loads the model once and runs images_text_to_text on every image given, in
// order, each with one image and an empty text part (and max_new_tokens when
// it is not 0). The model is released before the results are read; each
// result is printed as one JSON line: the generated token ids and the text as
// hexadecimal UTF-8 bytes. Images are interleaved RGB float32 in [0, 1].
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <trtmc/language.hpp>
#include <vector>

namespace {

std::uint64_t number(const char* argument, bool allow_zero) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || (!value && !allow_zero) ||
        value > static_cast<unsigned long long>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("arguments must be int32 values");
    return value;
}

struct Image {
    std::vector<float> pixels;
    std::uint32_t height, width;
};

Image read_image(const char* path, const char* height_text, const char* width_text) {
    static_assert(sizeof(float) == 4, "input uses float32");
    const auto height = static_cast<std::uint32_t>(number(height_text, false));
    const auto width = static_cast<std::uint32_t>(number(width_text, false));
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max());
    if (height > limit / width / 3 / sizeof(float))
        throw std::invalid_argument("image byte count overflows input storage");
    Image image{std::vector<float>(static_cast<std::size_t>(height) * width * 3), height, width};
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(image.pixels.data()),
              static_cast<std::streamsize>(image.pixels.size() * sizeof(float)));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("input must contain exactly HEIGHT x WIDTH x 3 float32 values");
    return image;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 7 || (argc - 4) % 3 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT MAX_NEW_TOKENS RGB_F32 HEIGHT WIDTH"
                     " [RGB_F32 HEIGHT WIDTH ...]\n";
        return 2;
    }
    try {
        const auto max_new_tokens = static_cast<std::int64_t>(number(argv[3], true));

        // One loaded model serves every request in order; the model and the
        // inputs go out of scope before the results are read.
        auto results = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::ImagesTextToText>();
            const auto fields = task.config_fields();
            if (fields.size() != 1 || fields[0].name != "max_new_tokens")
                throw std::runtime_error("TrOCR must expose exactly max_new_tokens");
            trtmc::Config config;
            if (max_new_tokens)
                config = trtmc::Config{{"max_new_tokens", max_new_tokens}};
            std::vector<trtmc::ImagesTextToText::Result> out;
            for (int index = 4; index + 2 < argc; index += 3) {
                const auto image = read_image(argv[index], argv[index + 1], argv[index + 2]);
                out.push_back(
                    task.run(trtmc::ImagesTextToTextRequest::from_parts(
                                 {trtmc::ImageInput({image.pixels.data(), image.pixels.size()},
                                                    image.height, image.width),
                                  trtmc::TextPart{""}}),
                             config));
            }
            return out;
        }();

        for (const auto& result : results) {
            const auto token_ids = result.token_ids();
            std::cout << "{\"task\":\"images_text_to_text\",\"token_ids\":[";
            for (std::size_t i = 0; i < token_ids.size(); ++i)
                std::cout << (i ? "," : "") << token_ids[i];
            std::cout << "],\"text_hex\":\"";
            for (const unsigned char byte : result.text()) {
                char hex[3];
                std::snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned>(byte));
                std::cout << hex;
            }
            std::cout << "\"}\n";
        }
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the text results");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
