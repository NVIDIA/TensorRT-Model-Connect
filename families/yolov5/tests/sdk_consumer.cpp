/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned YOLOv5 end-to-end test.
// Reads interleaved RGB float32 in [0, 1], runs image_to_boxes, releases the
// model, then prints the owned result in the CLI's detect JSON layout.
#include <cmath>
#include <cstdint>
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "input uses float32");
        const auto height = dimension(argv[4]), width = dimension(argv[5]);
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max());
        if (height > limit / width / 3 / sizeof(float))
            throw std::invalid_argument("image byte count overflows input storage");
        std::vector<float> input(static_cast<std::size_t>(height) * width * 3);
        std::ifstream file(argv[3], std::ios::binary);
        file.read(reinterpret_cast<char*>(input.data()),
                  static_cast<std::streamsize>(input.size() * sizeof(float)));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error(
                "input must contain exactly HEIGHT x WIDTH x 3 float32 values");

        // The model and its input go out of scope before the result is read.
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::ImageToBoxes>();
            if (!task.config_fields().empty())
                throw std::runtime_error("YOLOv5 must expose no runtime Config fields");
            return task.run({trtmc::ImageInput({input.data(), input.size()}, height, width)});
        }();
        input.clear();
        input.shrink_to_fit();

        const auto view = result.view();
        if (view.image_height != height || view.image_width != width || (view.count && !view.boxes))
            throw std::runtime_error("detection result does not describe the input image");
        for (std::uint64_t i = 0; i < view.count; ++i) {
            const auto& item = view.boxes[i];
            if (!std::isfinite(item.box.x_min) || !std::isfinite(item.box.y_min) ||
                !std::isfinite(item.box.x_max) || !std::isfinite(item.box.y_max) ||
                item.box.x_min > item.box.x_max || item.box.y_min > item.box.y_max ||
                !std::isfinite(item.score) || item.class_id < 0 ||
                (i && item.score > view.boxes[i - 1].score))
                throw std::runtime_error("detection " + std::to_string(i) +
                                         " is invalid or out of score order");
        }

        std::cout.precision(std::numeric_limits<float>::max_digits10);
        std::cout << "{\"task\":\"image_to_boxes\",\"image_height\":" << view.image_height
                  << ",\"image_width\":" << view.image_width << ",\"boxes\":[";
        for (std::uint64_t i = 0; i < view.count; ++i) {
            const auto& box = view.boxes[i].box;
            std::cout << (i ? "," : "") << box.x_min << ',' << box.y_min << ',' << box.x_max << ','
                      << box.y_max;
        }
        std::cout << "],\"scores\":[";
        for (std::uint64_t i = 0; i < view.count; ++i)
            std::cout << (i ? "," : "") << view.boxes[i].score;
        std::cout << "],\"classes\":[";
        for (std::uint64_t i = 0; i < view.count; ++i)
            std::cout << (i ? "," : "") << view.boxes[i].class_id;
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write detection output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
