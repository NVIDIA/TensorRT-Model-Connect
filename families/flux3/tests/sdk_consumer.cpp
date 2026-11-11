/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Public C++17 SDK consumer for the family-owned flux3 end-to-end test.
// Runs text_to_image, then writes the image as raw float32 HWC RGB values in
// [0, 1] and prints its shape as JSON.
//
// Usage: test_flux3_sdk_cpp BUNDLE RUNTIME_ROOT PROMPT OUTPUT_F32
//            [--initial-latents-raw FILE] [--seed N] [--num-steps N]
// FILE holds raw float32 latents laid out [128, height / 16, width / 16].
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/image.hpp>
#include <vector>

namespace {

struct Arguments {
    std::string bundle;
    std::string runtime_root;
    std::string prompt;
    std::string output;
    std::string latents_path;
    trtmc::Config config;
};

std::int64_t integer(const std::string& text, const std::string& option) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoll(text, &consumed);
        if (consumed == text.size())
            return value;
    } catch (const std::logic_error&) {
        // Reported below with the option name.
    }
    throw std::invalid_argument(option + " must be an integer, got '" + text + "'");
}

Arguments parse_arguments(int argc, char** argv) {
    if (argc < 5 || (argc - 5) % 2 != 0)
        throw std::invalid_argument("Usage: " + std::string(argv[0]) +
                                    " BUNDLE RUNTIME_ROOT PROMPT OUTPUT_F32"
                                    " [--initial-latents-raw FILE] [--seed N] [--num-steps N]");
    Arguments arguments{argv[1], argv[2], argv[3], argv[4], {}, {}};
    for (int index = 5; index < argc; index += 2) {
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--initial-latents-raw")
            arguments.latents_path = value;
        else if (option == "--seed")
            arguments.config.add("seed", integer(value, option));
        else if (option == "--num-steps")
            arguments.config.add("num_steps", integer(value, option));
        else
            throw std::invalid_argument("unknown option " + option);
    }
    return arguments;
}

std::vector<float> read_float32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("unable to open " + path);
    const std::streamoff bytes = file.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error(path + " must hold a nonzero whole number of float32 values");
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("failed to read " + path);
    return values;
}

void check_fields(const trtmc::TextToImage& task) {
    const auto fields = task.config_fields();
    if (fields.size() != 2)
        throw std::runtime_error("flux3 must expose exactly the seed and num_steps Config fields");
    for (const auto& field : fields) {
        const bool known = field.name == "seed" || field.name == "num_steps";
        if (!known || field.kind != trtmc::ConfigKind::I64 || !field.default_value)
            throw std::runtime_error("unexpected flux3 Config field '" + field.name + "'");
    }
}

void check_image(const trtmc::ImageGenerationResult& image) {
    const auto pixels = image.pixels();
    const auto expected = std::uint64_t{image.height()} * image.width() * 3;
    if (image.is_worker() || image.channels() != 3 || image.height() == 0 || image.width() == 0 ||
        pixels.size() != expected)
        throw std::runtime_error("text_to_image returned an invalid image shape");
    for (const float value : pixels) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::runtime_error("text_to_image returned a pixel outside [0, 1]");
    }
}

void write_float32(const std::string& path, trtmc::Span<const float> values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    file.close();
    if (!file)
        throw std::runtime_error("failed to write " + path);
}

void run(const Arguments& arguments) {
    const auto latents = arguments.latents_path.empty() ? std::vector<float>{}
                                                        : read_float32(arguments.latents_path);
    // The model goes out of scope before the result is read.
    const auto image = [&] {
        trtmc::LoadOptions options;
        options.runtime_root = arguments.runtime_root;
        const auto model = trtmc::Model::load(arguments.bundle, options);
        const auto task = model.task<trtmc::TextToImage>();
        check_fields(task);
        return task.run({arguments.prompt, {latents.data(), latents.size()}}, arguments.config);
    }();
    check_image(image);
    write_float32(arguments.output, image.pixels());
    std::cout << "{\"task\":\"text_to_image\",\"height\":" << image.height()
              << ",\"width\":" << image.width() << ",\"channels\":" << image.channels() << "}\n";
    std::cout.flush();
    if (!std::cout)
        throw std::runtime_error("failed to write the result summary");
}

} // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    try {
        arguments = parse_arguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    try {
        run(arguments);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
