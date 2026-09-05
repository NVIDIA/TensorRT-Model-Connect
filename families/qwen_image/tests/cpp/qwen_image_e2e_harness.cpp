/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_floats(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot open float32 input: " + path);
    const std::streamoff bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error("invalid float32 input size: " + path);
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!input)
        throw std::runtime_error("cannot read float32 input: " + path);
    return values;
}

void write_ppm(const trtmc::ImageResult& image, const std::string& path) {
    if (image.height <= 0 || image.width <= 0 || image.channels != 3 || image.num_frames != 1)
        throw std::runtime_error("native result must be one non-empty RGB image");
    const auto expected =
        static_cast<std::size_t>(image.height) * static_cast<std::size_t>(image.width) * 3U;
    if (image.pixels.size() != expected)
        throw std::runtime_error("native result pixel count does not match its dimensions");

    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("cannot open output image: " + path);
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    for (float value : image.pixels) {
        value = std::clamp(value, 0.0F, 1.0F);
        const auto byte = static_cast<unsigned char>(value * 255.0F + 0.5F);
        output.write(reinterpret_cast<const char*>(&byte), 1);
    }
    if (!output)
        throw std::runtime_error("cannot write output image: " + path);
}

trtmc::ImageGenerationConfig parse_config(char** argv) {
    trtmc::ImageGenerationConfig config;
    config.initial_latents = read_floats(argv[6]);
    config.height = std::stoi(argv[7]);
    config.width = std::stoi(argv[8]);
    config.num_steps = std::stoi(argv[9]);
    config.seed = std::stoi(argv[10]);
    config.guidance_scale = std::stof(argv[11]);
    config.cfg_scale = std::stof(argv[12]);
    config.negative_prompt = argv[13];
    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 14 && argc != 17) {
            throw std::invalid_argument(
                "usage: qwen_image_e2e_harness BUNDLE RUNTIME_ROOT OUTPUT TASK PROMPT "
                "INITIAL_LATENTS HEIGHT WIDTH STEPS SEED GUIDANCE CFG NEGATIVE_PROMPT "
                "[EDIT_PIXELS EDIT_HEIGHT EDIT_WIDTH]");
        }
        const std::string task_name = argv[4];
        auto task = trtmc::load_task(argv[1], argv[2]);
        if (task_name != task->task())
            throw std::runtime_error("loaded bundle task does not match requested task");

        const auto config = parse_config(argv);
        trtmc::ImageResult result;
        if (task_name == trtmc::IImageGeneration::kTask) {
            if (argc != 14)
                throw std::invalid_argument("image generation does not accept edit pixels");
            auto* interface = dynamic_cast<trtmc::IImageGeneration*>(task.get());
            if (interface == nullptr)
                throw std::runtime_error("bundle does not implement IImageGeneration");
            result = interface->generate_image(argv[5], config);
        } else if (task_name == trtmc::IImageEditing::kTask) {
            if (argc != 17)
                throw std::invalid_argument("image editing requires edit pixels and dimensions");
            auto pixels = read_floats(argv[14]);
            const auto height = std::stoi(argv[15]);
            const auto width = std::stoi(argv[16]);
            if (height <= 0 || width <= 0)
                throw std::invalid_argument("edit input dimensions must be positive");
            const auto expected =
                static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3U;
            if (pixels.size() != expected)
                throw std::runtime_error("edit input pixel count does not match its dimensions");
            auto* interface = dynamic_cast<trtmc::IImageEditing*>(task.get());
            if (interface == nullptr)
                throw std::runtime_error("bundle does not implement IImageEditing");
            result = interface->generate_image(argv[5], pixels.data(), height, width, config);
        } else {
            throw std::invalid_argument("unsupported Qwen Image task: " + task_name);
        }
        write_ppm(result, argv[3]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
