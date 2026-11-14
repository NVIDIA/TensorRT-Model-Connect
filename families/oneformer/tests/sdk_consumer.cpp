/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned OneFormer end-to-end test.
// Reads interleaved RGB uint8 pixels, runs image_to_semantic_segmentation,
// releases the model, then prints the label map's per-class counts and its
// FNV-1a 64 digest (over little-endian int32 labels) as one JSON line.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <trtmc/perception.hpp>
#include <vector>

namespace {

std::uint32_t dimension(const char* argument) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || !value ||
        value > static_cast<unsigned long long>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("image dimensions must be positive int32 values");
    return static_cast<std::uint32_t>(value);
}

std::string quoted(std::string_view value) {
    std::string result = "\"";
    for (const char c : value) {
        if (c == '"' || c == '\\')
            result += '\\';
        result += c;
    }
    return result + '"';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT RGB_U8 HEIGHT WIDTH\n";
        return 2;
    }
    try {
        const auto height = dimension(argv[4]), width = dimension(argv[5]);
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max());
        if (height > limit / width / 3)
            throw std::invalid_argument("image byte count overflows input storage");
        std::vector<std::uint8_t> input(static_cast<std::size_t>(height) * width * 3);
        std::ifstream file(argv[3], std::ios::binary);
        file.read(reinterpret_cast<char*>(input.data()),
                  static_cast<std::streamsize>(input.size()));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("input must contain exactly HEIGHT x WIDTH x 3 uint8 values");

        // The model and its input go out of scope before the result is read.
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::ImageToSemanticSegmentation>();
            if (!task.config_fields().empty())
                throw std::runtime_error("OneFormer must expose no runtime Config fields");
            return task.run({trtmc::ImageInput({input.data(), input.size()}, height, width)});
        }();
        input.clear();
        input.shrink_to_fit();

        const auto& view = result.view();
        if (view.height != height || view.width != width ||
            view.pixel_count != static_cast<std::uint64_t>(height) * width || !view.labels ||
            !view.class_ids.size || view.class_names.size != view.class_ids.size)
            throw std::runtime_error("semantic result does not describe the input image");
        std::vector<std::uint64_t> counts(static_cast<std::size_t>(view.class_ids.size));
        std::uint64_t digest = 1469598103934665603ULL;
        for (std::uint64_t i = 0; i < view.pixel_count; ++i) {
            const auto label = view.labels[i];
            if (label < 0 || static_cast<std::uint64_t>(label) >= view.class_ids.size ||
                view.class_ids.data[label] != label)
                throw std::runtime_error("label " + std::to_string(label) +
                                         " is not a result class id");
            ++counts[static_cast<std::size_t>(label)];
            const auto bits = static_cast<std::uint32_t>(label);
            for (int byte = 0; byte < 4; ++byte) {
                digest ^= (bits >> (8 * byte)) & 0xFFU;
                digest *= 1099511628211ULL;
            }
        }

        std::cout << "{\"task\":\"image_to_semantic_segmentation\",\"height\":" << view.height
                  << ",\"width\":" << view.width << ",\"class_names\":[";
        for (std::uint64_t i = 0; i < view.class_names.size; ++i)
            std::cout << (i ? "," : "")
                      << quoted(trtmc::detail::string_view(view.class_names.data[i]));
        std::cout << "],\"counts\":[";
        for (std::size_t i = 0; i < counts.size(); ++i)
            std::cout << (i ? "," : "") << counts[i];
        char hex[17];
        std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(digest));
        std::cout << "],\"fnv1a64\":\"" << hex
                  << "\",\"has_scores\":" << (view.score_count ? "true" : "false") << "}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the segmentation summary");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
