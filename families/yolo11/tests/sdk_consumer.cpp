/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public SDK consumer for the family-owned end-to-end test. */
#include <cmath>
#include <fstream>
#include <iomanip>
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
    if (consumed != text.size() || !value || value > std::numeric_limits<std::int32_t>::max())
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
            const auto task = model.task<trtmc::ImageToBoxes>();
            if (!task.config_fields().empty())
                throw std::runtime_error("YOLO11 must expose no runtime Config fields");
            return task.run({trtmc::ImageInput({input.data(), input.size()}, height, width)});
        }();
        input.clear();
        input.shrink_to_fit();

        const auto view = result.view();
        if (view.image_height != height || view.image_width != width)
            throw std::runtime_error("detection output dimensions mismatch input image");

        for (std::size_t i = 0; i < view.count; ++i) {
            const auto& item = view.boxes[i];
            if (!std::isfinite(item.box.x_min) || !std::isfinite(item.box.y_min) ||
                !std::isfinite(item.box.x_max) || !std::isfinite(item.box.y_max) ||
                item.box.x_min > item.box.x_max || item.box.y_min > item.box.y_max ||
                !std::isfinite(item.score) || item.class_id < 0) {
                throw std::runtime_error(
                    "detection contains invalid box coordinates, score, or class");
            }
        }

        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"image_to_boxes\",\"input_shape\":[" << height << ',' << width
                  << ",3],\"image_height\":" << view.image_height
                  << ",\"image_width\":" << view.image_width << ",\"count\":" << view.count
                  << ",\"boxes\":[";
        for (std::size_t i = 0; i < view.count; ++i) {
            if (i)
                std::cout << ',';
            std::cout << "{\"box\":[" << view.boxes[i].box.x_min << ',' << view.boxes[i].box.y_min
                      << ',' << view.boxes[i].box.x_max << ',' << view.boxes[i].box.y_max
                      << "],\"score\":" << view.boxes[i].score
                      << ",\"class_id\":" << view.boxes[i].class_id << '}';
        }
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
