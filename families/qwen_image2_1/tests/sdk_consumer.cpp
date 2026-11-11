/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned Qwen-Image 2.1 end-to-end test.
// Replays caller-supplied initial latents through text_to_image, optionally with
// classifier-free guidance, releases the model, then writes the owned float32
// HWC pixels and prints their shape.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <trtmc/image.hpp>
#include <vector>

namespace {

std::vector<float> read_floats(const char* path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    if (!file || bytes.empty() || bytes.size() % sizeof(float) != 0)
        throw std::runtime_error("LATENTS_F32 must be a non-empty float32 file");
    std::vector<float> values(bytes.size() / sizeof(float));
    std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(values.data()));
    return values;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 7 && argc != 9) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT PROMPT LATENTS_F32 NUM_STEPS PIXELS_F32"
                     " [CFG_SCALE NEGATIVE_PROMPT]\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "latents and pixels use float32");
        const std::string steps_text(argv[5]);
        std::size_t consumed = 0;
        const auto steps = std::stoll(steps_text, &consumed);
        if (consumed != steps_text.size() || steps < 1)
            throw std::invalid_argument("NUM_STEPS must be a positive integer");
        auto latents = read_floats(argv[4]);

        // The model and its inputs go out of scope before the result is read.
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::TextToImage>();
            const auto fields = task.config_fields();
            const bool declares_steps =
                std::any_of(fields.begin(), fields.end(), [](const trtmc::ConfigField& field) {
                    return field.name == "num_steps" && field.kind == trtmc::ConfigKind::I64;
                });
            if (!declares_steps)
                throw std::runtime_error(
                    "Qwen-Image 2.1 must declare the integer Config field num_steps");
            trtmc::Config config;
            config.add("num_steps", static_cast<std::int64_t>(steps));
            if (argc == 9) {
                config.add("cfg_scale", std::stod(argv[7]));
                config.add("negative_prompt", std::string(argv[8]));
            }
            return task.run({argv[3], {latents.data(), latents.size()}}, config);
        }();
        latents.clear();
        latents.shrink_to_fit();

        const auto pixels = result.pixels();
        const auto expected =
            static_cast<std::uint64_t>(result.height()) * result.width() * result.channels();
        if (pixels.empty() || pixels.size() != expected)
            throw std::runtime_error("image result does not describe one HWC image");
        std::ofstream output(argv[6], std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(pixels.data()),
                     static_cast<std::streamsize>(pixels.size() * sizeof(float)));
        output.close();
        if (!output)
            throw std::runtime_error("failed to write PIXELS_F32");

        std::cout << "{\"task\":\"text_to_image\",\"height\":" << result.height()
                  << ",\"width\":" << result.width() << ",\"channels\":" << result.channels()
                  << "}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write the result shape");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
