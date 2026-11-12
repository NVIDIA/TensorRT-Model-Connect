/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned Stable Video Diffusion E2E.
// Reads interleaved RGB uint8 pixels, passes them as float32 in [0, 1], runs
// initial_image_text_to_video with the given seed and step count, releases the
// model, then writes the THWC float32 frames and prints a summary.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <trtmc/video.hpp>
#include <vector>

namespace {

// The published pipeline's default frame rate, which the run does not override.
constexpr double kDefaultFps = 7.0;

std::int64_t integer(const char* argument, std::int64_t minimum, std::int64_t maximum) {
    const std::string text(argument);
    std::size_t consumed = 0;
    const auto value = std::stoll(text, &consumed);
    if (consumed != text.size() || value < minimum || value > maximum)
        throw std::invalid_argument("argument out of range: " + text);
    return value;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT RGB_U8 HEIGHT WIDTH SEED STEPS FRAMES_F32\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "frames are float32");
        const auto height = static_cast<std::uint32_t>(integer(argv[4], 1, 16384));
        const auto width = static_cast<std::uint32_t>(integer(argv[5], 1, 16384));
        const auto seed = integer(argv[6], 0, 4294967295LL);
        const auto steps = integer(argv[7], 1, 1000);
        const std::size_t count = static_cast<std::size_t>(height) * width * 3;
        std::vector<unsigned char> bytes(count);
        std::ifstream file(argv[3], std::ios::binary);
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("input must contain exactly HEIGHT x WIDTH x 3 uint8 values");
        std::vector<float> input(count);
        for (std::size_t index = 0; index < count; ++index)
            input[index] = static_cast<float>(bytes[index]) / 255.0F;

        // The model and its input go out of scope before the result is read.
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::InitialImageTextToVideo>();
            if (task.config_fields().size() != 7)
                throw std::runtime_error("expected seven Config fields");
            trtmc::Config config;
            config.add("seed", seed);
            config.add("num_steps", steps);
            return task.run({trtmc::ImageInput({input.data(), input.size()}, height, width), ""},
                            config);
        }();
        input.clear();
        input.shrink_to_fit();

        const auto frames = result.frames();
        const auto timestamps = result.timestamps_seconds();
        if (frames.empty() || timestamps.size() != frames.size() ||
            result.conditioned_prefix_frames() != 0)
            throw std::runtime_error(
                "video result has no frames, a conditioned prefix or no clock");
        std::ofstream output(argv[8], std::ios::binary);
        for (std::size_t index = 0; index < frames.size(); ++index) {
            const auto& frame = frames[index];
            if (frame.height != height || frame.width != width || frame.channels != 3 ||
                frame.pixel_count != count || !frame.pixels ||
                std::fabs(timestamps[index] - static_cast<double>(index) / kDefaultFps) > 1e-12)
                throw std::runtime_error("frame " + std::to_string(index) +
                                         " does not match the request");
            output.write(reinterpret_cast<const char*>(frame.pixels),
                         static_cast<std::streamsize>(frame.pixel_count * sizeof(float)));
        }
        output.close();
        if (!output)
            throw std::runtime_error("failed to write the frames");
        std::cout << "{\"task\":\"initial_image_text_to_video\",\"frames\":" << frames.size()
                  << ",\"height\":" << height << ",\"width\":" << width
                  << ",\"inference_ms\":" << result.inference_ms() << "}\n";
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
