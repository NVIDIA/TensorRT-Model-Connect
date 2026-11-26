/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test: semantic segmentation of one
 * float32 RGB image. Writes the int32 label map to OUT_LABELS and one JSON summary line to stdout.
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/perception.hpp>
#include <vector>

namespace {
std::uint32_t dimension(const char* argument) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || !value || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("image dimensions must be positive int32 values");
    return static_cast<std::uint32_t>(value);
}
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
} // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH OUT_LABELS\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "input uses float32");
        const auto height = dimension(argv[4]), width = dimension(argv[5]);
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max());
        if (height > limit / width / 3 / sizeof(float))
            throw std::invalid_argument("image byte count overflows input storage");
        const auto count = static_cast<std::size_t>(height) * width * 3;
        std::vector<float> input(count);
        std::ifstream file(argv[3], std::ios::binary);
        file.read(reinterpret_cast<char*>(input.data()),
                  static_cast<std::streamsize>(count * sizeof(float)));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error(
                "input must contain exactly HEIGHT x WIDTH x 3 RGB float32 values");
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::ImageToSemanticSegmentation>();
            if (!task.config_fields().empty())
                throw std::runtime_error("Mask2Former must expose no runtime Config fields");
            return task.run({trtmc::ImageInput({input.data(), input.size()}, height, width)});
        }();
        input.clear();
        input.shrink_to_fit();
        const auto view = result.view();
        if (view.height != height || view.width != width ||
            view.pixel_count != static_cast<std::uint64_t>(height) * width || !view.class_ids.size)
            throw std::runtime_error("semantic segmentation must label every pixel of the input");
        std::ofstream labels(argv[6], std::ios::binary);
        labels.write(reinterpret_cast<const char*>(view.labels),
                     static_cast<std::streamsize>(view.pixel_count * sizeof(std::int32_t)));
        labels.close();
        if (!labels)
            throw std::runtime_error("unable to write the label map");
        std::cout << "{\"task\":\"image_to_semantic_segmentation\",\"height\":" << view.height
                  << ",\"width\":" << view.width << ",\"pixel_count\":" << view.pixel_count
                  << ",\"class_ids\":[";
        for (std::uint64_t i = 0; i < view.class_ids.size; ++i)
            std::cout << (i ? "," : "") << view.class_ids.data[i];
        std::cout << "],\"class_names\":[";
        for (std::uint64_t i = 0; i < view.class_names.size; ++i) {
            if (i)
                std::cout << ',';
            json_string(trtmc::detail::string_view(view.class_names.data[i]));
        }
        std::cout << "],\"vocabulary_id\":";
        json_string(trtmc::detail::string_view(view.vocabulary_id));
        std::cout << ",\"has_ignore_label\":" << view.has_ignore_label
                  << ",\"has_background_label\":" << view.has_background_label
                  << ",\"score_count\":" << view.score_count << "}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the segmentation summary");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
