/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string bundle;
    std::string image;
    std::string state;
    std::string runtime_root;
    double control_hz{50.0};
};

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " MODEL.bundle --image FRAME.png --state STATE.f32 "
                 "--runtime-root DIR [--control-hz 50]\n";
}

std::string take_value(int& index, int argc, char** argv, const std::string& option) {
    if (++index >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--image")
            options.image = take_value(index, argc, argv, argument);
        else if (argument == "--state")
            options.state = take_value(index, argc, argv, argument);
        else if (argument == "--runtime-root")
            options.runtime_root = take_value(index, argc, argv, argument);
        else if (argument == "--control-hz")
            options.control_hz = std::stod(take_value(index, argc, argv, argument));
        else if (!argument.empty() && argument[0] == '-')
            throw std::invalid_argument("unknown option: " + argument);
        else if (options.bundle.empty())
            options.bundle = argument;
        else
            throw std::invalid_argument("only one bundle may be specified");
    }
    if (options.bundle.empty() || options.image.empty() || options.state.empty() ||
        options.runtime_root.empty() || !std::isfinite(options.control_hz) ||
        options.control_hz <= 0.0)
        throw std::invalid_argument(
            "bundle, image, state, runtime root, and a positive control rate are required");
    return options;
}

std::vector<float> read_state(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("failed to open recorded state: " + path);
    const auto bytes = input.tellg();
    if (bytes != static_cast<std::streamoff>(14 * sizeof(float)))
        throw std::runtime_error("recorded state must contain exactly 14 float32 values");
    std::vector<float> state(14);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(state.data()), bytes);
    if (!input)
        throw std::runtime_error("failed to read recorded state");
    return state;
}

struct Image {
    std::vector<float> pixels;
    int width{0};
    int height{0};
};

Image read_image(const std::string& path) {
    Image image;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &image.width, &image.height, &channels, 3);
    if (data == nullptr)
        throw std::runtime_error("failed to decode recorded image: " + path);
    const auto count = static_cast<std::size_t>(image.width) * image.height * 3U;
    image.pixels.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        image.pixels[index] = static_cast<float>(data[index]) / 255.0F;
    stbi_image_free(data);
    return image;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto image = read_image(options.image);
        if (image.pixels.empty() || image.height != 480 || image.width != 640)
            throw std::runtime_error("recorded image must be 640x480 RGB");
        const auto state = read_state(options.state);

        auto task = trtmc::load_task(options.bundle, options.runtime_root);
        auto* control = dynamic_cast<trtmc::IRobotControl*>(task.get());
        if (control == nullptr)
            throw std::runtime_error("bundle does not implement the robot_control Task API");
        const trtmc::RobotObservation observation{{image.pixels.data(), image.pixels.size()},
                                                  image.height,
                                                  image.width,
                                                  3,
                                                  {state.data(), state.size()}};
        const auto chunk = control->predict_action_chunk(observation);
        if (chunk.num_actions != 100 || chunk.action_dim != 14 || chunk.actions.size() != 1400)
            throw std::runtime_error("qualified ACT bundle must return a 100x14 action chunk");

        using Clock = std::chrono::steady_clock;
        const auto period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / options.control_hz));
        const auto start = Clock::now() + std::chrono::milliseconds(10);
        for (int32_t step = 0; step < chunk.num_actions; ++step) {
            std::this_thread::sleep_until(start + step * period);
            const float* action = chunk.actions.data() + static_cast<std::size_t>(step) * 14;
            std::cout << step;
            for (int32_t joint = 0; joint < 14; ++joint)
                std::cout << (joint == 0 ? ',' : ' ') << action[joint];
            std::cout << '\n';
        }
        std::cerr << "Emitted 100 recorded-replay actions at " << options.control_hz
                  << " Hz; inference_ms=" << chunk.inference_ms
                  << "; within_training_bounds=" << std::boolalpha << chunk.within_training_bounds
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
