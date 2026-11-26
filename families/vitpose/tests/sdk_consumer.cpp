/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the family-owned end-to-end test: the body keypoints of one person
 * in a float32 RGB image, optionally inside a person box. Prints one JSON line to stdout. */
#include <cmath>
#include <cstdio>
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
double coordinate(const char* argument) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value))
        throw std::invalid_argument("person box values must be finite numbers");
    return value;
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
// The C consumer's "%.9g": nine significant digits round-trip every float32.
void number(float value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
    std::cout << text;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 10) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH [X_MIN Y_MIN X_MAX Y_MAX]\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "input uses float32");
        const auto height = dimension(argv[4]), width = dimension(argv[5]);
        trtmc::Config config;
        if (argc == 10) {
            config.add("person_box", std::vector<double>{coordinate(argv[6]), coordinate(argv[7]),
                                                         coordinate(argv[8]), coordinate(argv[9])});
        }
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
            const auto task = model.task<trtmc::ImageTextToPoints>();
            const auto fields = task.config_fields();
            if (fields.size() != 1 || fields[0].name != "person_box" ||
                fields[0].kind != trtmc::ConfigKind::F64List || fields[0].default_value)
                throw std::runtime_error(
                    "ViTPose must expose exactly the person_box F64 list field");
            return task.run(
                {trtmc::ImageInput({input.data(), input.size()}, height, width), "person"}, config);
        }();
        input.clear();
        input.shrink_to_fit();
        const auto& view = result.view();
        for (std::uint64_t i = 0; i < view.count; ++i) {
            if (!view.points[i].has_confidence)
                throw std::runtime_error("every keypoint must carry its heatmap score");
        }
        std::cout << "{\"task\":\"image_text_to_points\",\"points\":[";
        for (std::uint64_t i = 0; i < view.count; ++i) {
            std::cout << (i ? "," : "");
            number(view.points[i].point.x);
            std::cout << ',';
            number(view.points[i].point.y);
        }
        std::cout << "],\"labels\":[";
        for (std::uint64_t i = 0; i < view.count; ++i) {
            if (i)
                std::cout << ',';
            json_string(trtmc::detail::string_view(view.points[i].label));
        }
        std::cout << "],\"scores\":[";
        for (std::uint64_t i = 0; i < view.count; ++i) {
            std::cout << (i ? "," : "");
            number(view.points[i].confidence);
        }
        std::cout << "],\"raw_response\":";
        json_string(trtmc::detail::string_view(view.raw_response));
        std::cout << ",\"parse_complete\":" << (view.parse_complete ? "true" : "false") << "}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the keypoints");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
