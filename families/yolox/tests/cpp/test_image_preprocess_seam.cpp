/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/yolox/runtime/image_preprocess_seam.h"
#include "families/yolox/runtime/pipeline.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main(int argc, char** argv) {
    try {
        trtmc::YoloxPreprocessConfig config;
        trtmc::YoloxLetterbox letterbox;
        if (argc == 5 || argc == 6) {
            // Family test seam: raw interleaved RGB float32 in, BGR CHW out.
            const int height = std::stoi(argv[2]);
            const int width = std::stoi(argv[3]);
            if (argc == 6)
                config.input_image_h = config.input_image_w = std::stoi(argv[5]);
            require(height > 0 && width > 0, "invalid image dimensions");
            std::vector<float> pixels(static_cast<std::size_t>(height) * width * 3U);
            std::ifstream input(argv[1], std::ios::binary);
            input.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
            require(static_cast<bool>(input), "could not read input pixels");
            const auto values =
                trtmc::preprocess_yolox_image(pixels.data(), height, width, config, letterbox);
            std::ofstream output(argv[4], std::ios::binary);
            output.write(reinterpret_cast<const char*>(values.data()),
                         values.size() * sizeof(float));
            require(static_cast<bool>(output), "could not write preprocessed pixels");
            return 0;
        }
        require(argc == 1, "expected no arguments or input height width output [size]");
        config.input_image_h = config.input_image_w = 2;
        const float red_blue[] = {1, 0, 0, 0, 0, 1};
        const auto values = trtmc::preprocess_yolox_image(red_blue, 1, 2, config, letterbox);
        const std::vector<float> expected = {0, 255, 114, 114, 0, 0, 114, 114, 255, 0, 114, 114};
        require(values == expected, "BGR bytes, top-left placement or bottom padding is wrong");
        require(letterbox.scale == 1 && letterbox.pad_x == 0 && letterbox.pad_y == 0,
                "YOLOX must not center its letterbox");
        config.input_image_h = config.input_image_w = 3;
        const float corners[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
        const auto sampled = trtmc::preprocess_yolox_image(corners, 2, 2, config, letterbox);
        const std::vector<float> expected_samples = {0, 0,   0,   0,   64, 128, 0,   128, 255,
                                                     0, 0,   0,   128, 64, 0,   255, 128, 0,
                                                     0, 128, 255, 0,   64, 128, 0,   0,   0};
        require(sampled == expected_samples,
                "bilinear channel selection, coordinates or byte rounding is wrong");
        config.input_image_h = config.input_image_w = 4;
        const std::vector<float> constant(2U * 3U * 3U, 1.0F);
        const auto resized =
            trtmc::preprocess_yolox_image(constant.data(), 2, 3, config, letterbox);
        require(std::fabs(letterbox.scale - 4.0F / 3.0F) < 1e-6F, "resize ratio is wrong");
        for (std::size_t channel = 0; channel < 3; ++channel) {
            for (std::size_t index = 0; index < 16; ++index) {
                require(resized[channel * 16 + index] == (index < 8 ? 255.0F : 114.0F),
                        "resized dimensions must truncate, with padding at the bottom");
            }
        }
        bool rejected = false;
        try {
            trtmc::preprocess_yolox_image(nullptr, 1, 2, config, letterbox);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "empty input was accepted");
        const std::vector<trtmc::DetectionBox> boxes = {{-100, 0, 10, 10, 0.9F, 0},
                                                        {0, 0, 10, 10, 0.8F, 0},
                                                        {0, 0, 10, 10, 0.7F, 1},
                                                        {0, 0, 10, 10, 0.6F, 0}};
        const auto kept = trtmc::suppress_yolox_boxes(boxes, 0.45F, 8400);
        require(kept.size() == 3, "NMS must preserve unbounded boxes and distinct classes");
        require(kept[0].x_min == -100 && kept[1].score == 0.8F && kept[2].class_id == 1,
                "NMS output ordering or coordinates are wrong");
        require(trtmc::suppress_yolox_boxes(boxes, 0.45F, 0).empty(), "zero cap is not empty");
        std::cout << "YOLOX preprocessing and NMS checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
