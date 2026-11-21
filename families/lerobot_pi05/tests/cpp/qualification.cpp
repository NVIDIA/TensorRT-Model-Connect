/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runs one recorded Pi 0.5 observation through a bundle with the public C++ SDK and writes
// the action chunk, so a Python harness can compare it with the LeRobot reference.
//
//   trtmc_lerobot_pi05_qualification --bundle B --runtime-root R
//       --camera NAME=FILE.f32 ... --state FILE.f32 --text "..." [--noise FILE.f32]
//       [--seed N] [--repeat N] [--check-rejections] --output FILE.f32
//
// --check-rejections also sends malformed requests built from the given observation
// and reports each one's status code and message.
//
// A camera file is a PNG (sent as uint8) or contiguous float32 HWC pixels in [0, 1];
// state and noise are float32.

#include "trtmc/action.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../../third_party/stb/stb_image.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes % sizeof(float) != 0)
        throw std::runtime_error(path + " is not a float32 file");
    std::vector<float> values(bytes / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    return values;
}

struct Camera {
    std::string name;
    std::vector<float> pixels;       // float32 input
    std::vector<std::uint8_t> bytes; // uint8 input, decoded from a PNG
    std::uint32_t height{0};
    std::uint32_t width{0};
};

Camera read_camera(const std::string& name, const std::string& path) {
    Camera camera{name, {}, {}, 0, 0};
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".png") == 0) {
        int width = 0, height = 0, channels = 0;
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> decoded(
            stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
        if (!decoded)
            throw std::runtime_error("cannot decode " + path);
        camera.width = static_cast<std::uint32_t>(width);
        camera.height = static_cast<std::uint32_t>(height);
        camera.bytes.assign(decoded.get(),
                            decoded.get() + static_cast<std::size_t>(width) * height * 3);
        return camera;
    }
    camera.pixels = read_floats(path);
    std::uint32_t side = 1;
    while (static_cast<std::size_t>(side) * side * 3 < camera.pixels.size())
        ++side;
    if (static_cast<std::size_t>(side) * side * 3 != camera.pixels.size())
        throw std::runtime_error("camera " + name + " is not a square RGB image");
    camera.height = camera.width = side;
    return camera;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string bundle, runtime_root, state_path, text, noise_path, output;
        std::vector<Camera> cameras;
        std::int64_t seed = -1;
        int repeat = 1;
        bool check_rejections = false;
        for (int index = 1; index < argc; ++index) {
            const std::string flag = argv[index];
            const auto value = [&]() -> std::string {
                if (index + 1 >= argc)
                    throw std::runtime_error("missing value for " + flag);
                return argv[++index];
            };
            if (flag == "--bundle")
                bundle = value();
            else if (flag == "--runtime-root")
                runtime_root = value();
            else if (flag == "--state")
                state_path = value();
            else if (flag == "--text")
                text = value();
            else if (flag == "--noise")
                noise_path = value();
            else if (flag == "--output")
                output = value();
            else if (flag == "--seed")
                seed = std::stoll(value());
            else if (flag == "--repeat")
                repeat = std::stoi(value());
            else if (flag == "--check-rejections")
                check_rejections = true;
            else if (flag == "--camera") {
                const auto spec = value();
                const auto split = spec.find('=');
                if (split == std::string::npos)
                    throw std::runtime_error("--camera needs NAME=FILE");
                cameras.push_back(read_camera(spec.substr(0, split), spec.substr(split + 1)));
            } else
                throw std::runtime_error("unknown flag " + flag);
        }
        if (bundle.empty() || runtime_root.empty() || output.empty() || state_path.empty())
            throw std::runtime_error("--bundle, --runtime-root, --state and --output are required");

        trtmc::LoadOptions load;
        load.runtime_root = runtime_root;
        const auto load_started = std::chrono::steady_clock::now();
        auto model = trtmc::Model::load(bundle, load);
        const double load_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - load_started)
                                   .count();
        const auto task = model.task<trtmc::ImagesStateTextToActionChunk>();

        const auto state = read_floats(state_path);
        std::vector<trtmc::NamedImage> images;
        for (const auto& camera : cameras) {
            images.push_back(
                {camera.name, camera.bytes.empty()
                                  ? trtmc::ImageInput({camera.pixels.data(), camera.pixels.size()},
                                                      camera.height, camera.width)
                                  : trtmc::ImageInput({camera.bytes.data(), camera.bytes.size()},
                                                      camera.height, camera.width)});
        }
        trtmc::Config config;
        if (!noise_path.empty()) {
            const auto noise = read_floats(noise_path);
            config.add("noise", std::vector<double>(noise.begin(), noise.end()));
        } else if (seed >= 0) {
            config.add("seed", seed);
        }

        const trtmc::ImagesStateTextToActionChunkRequest request{
            {{images.data(), images.size()}, {state.data(), state.size()}, text}};
        std::vector<double> latencies;
        trtmc::ImagesStateTextActionChunkResult last = task.run(request, config);
        for (int run = 0; run < repeat; ++run) {
            const auto started = std::chrono::steady_clock::now();
            last = task.run(request, config);
            latencies.push_back(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count());
        }

        std::ostringstream rejections;
        if (check_rejections) {
            const auto attempt = [&](const char* label, const std::vector<trtmc::NamedImage>& cams,
                                     const std::vector<float>& values,
                                     const std::string& instruction) {
                std::string code = "ok", message;
                try {
                    (void)task.run(
                        {{{cams.data(), cams.size()}, {values.data(), values.size()}, instruction}},
                        config);
                } catch (const trtmc::Error& error) {
                    code = std::to_string(static_cast<int>(error.code()));
                    message = error.what();
                }
                rejections << (rejections.tellp() > 0 ? "," : "") << "{\"case\":\"" << label
                           << "\",\"code\":\"" << code << "\",\"message\":\"";
                for (const char c : message)
                    rejections << (c == '"' || c == '\\' ? "\\" : "") << c;
                rejections << "\"}";
            };
            auto unknown = images;
            unknown.back().name = "observation.images.not_a_camera";
            const std::vector<trtmc::NamedImage> missing(images.begin(), images.end() - 1);
            const std::vector<float> short_state(state.begin(), state.end() - 1);
            attempt("unknown_camera", unknown, state, text);
            attempt("missing_camera", missing, state, text);
            attempt("short_state", images, short_state, text);
            attempt("empty_text", images, state, "");
            std::vector<float> small(3 * 8 * 8, 0.5F);
            auto resized = images;
            resized.back().image = trtmc::ImageInput({small.data(), small.size()}, 8, 8);
            attempt("wrong_image_size", resized, state, text);
        }
        const auto matrix = last.actions();
        std::ofstream out(output, std::ios::binary);
        out.write(reinterpret_cast<const char*>(matrix.values.data()),
                  static_cast<std::streamsize>(matrix.values.size() * sizeof(float)));
        std::cout << "{\"load_ms\":" << load_ms << ",\"rejections\":[" << rejections.str()
                  << "],\"rows\":" << matrix.rows << ",\"columns\":" << matrix.columns
                  << ",\"inference_ms\":" << last.inference_ms() << ",\"wall_ms\":[";
        for (std::size_t i = 0; i < latencies.size(); ++i)
            std::cout << (i ? "," : "") << latencies[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
