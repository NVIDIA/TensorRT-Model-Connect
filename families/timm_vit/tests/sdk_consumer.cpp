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
#include <string_view>
#include <trtmc/features.hpp>
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
            const auto task = model.task<trtmc::ImageToClassScores>();
            if (!task.config_fields().empty())
                throw std::runtime_error("ViT must expose no runtime Config fields");
            return task.run({trtmc::ImageInput({input.data(), input.size()}, height, width)});
        }();
        input.clear();
        input.shrink_to_fit();
        const auto scores = result.scores();
        if (scores.empty() || result.kind() != TRTMC_SCORE_LOGIT)
            throw std::runtime_error("ViT must provide complete, unnormalized logits");
        std::size_t top = 0;
        for (std::size_t i = 0; i < scores.size(); ++i) {
            if (!std::isfinite(scores[i]))
                throw std::runtime_error("classification contains nonfinite logits");
            if (scores[i] > scores[top])
                top = i;
        }
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"image_to_class_scores\",\"input_shape\":[" << height << ','
                  << width << ",3],\"kind\":\"logit\",\"score_kind\":" << result.kind()
                  << ",\"vocabulary_id\":";
        json_string(result.vocabulary_id());
        std::cout << ",\"labels\":[";
        const auto labels = result.labels();
        for (std::size_t i = 0; i < labels.size(); ++i) {
            if (i)
                std::cout << ',';
            json_string(labels[i]);
        }
        std::cout << "],\"top_class\":" << top << ",\"top_score\":" << scores[top]
                  << ",\"score_count\":" << scores.size() << ",\"scores\":[";
        for (std::size_t i = 0; i < scores.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << scores[i];
        }
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write classification output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
