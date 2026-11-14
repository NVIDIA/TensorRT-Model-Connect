/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * C++ SDK consumer for the stable_diffusion_3 family. Loads one bundle and
 * runs text_to_image through the public C++ wrapper (trtmc/trtmc.hpp +
 * trtmc/image.hpp): the declared config fields, repeated generations with
 * their wall time, and the rejection of malformed requests.
 *
 * Usage:
 *   test_stable_diffusion_3_sdk_cpp <bundle_path> <runtime_root> [runs] [steps]
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <trtmc/image.hpp>
#include <trtmc/trtmc.hpp>
#include <vector>

namespace {

void require(bool condition, const std::string& what) {
    if (!condition)
        throw std::runtime_error(what);
}

template <class Call>
void require_rejected(Call&& call, const std::string& what) {
    try {
        call();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("accepted " + what);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <bundle_path> <runtime_root> [runs] [steps]\n";
        return 1;
    }
    const int runs = argc > 3 ? std::stoi(argv[3]) : 1;
    const std::int64_t steps = argc > 4 ? std::stoll(argv[4]) : 4;
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        const auto load_start = std::chrono::steady_clock::now();
        auto model = trtmc::Model::load(argv[1], options);
        const std::chrono::duration<double> load_time =
            std::chrono::steady_clock::now() - load_start;
        std::cout << "load: " << load_time.count() << " s\n";

        const auto task = model.task<trtmc::TextToImage>();
        std::set<std::string> names;
        for (const auto& field : task.config_fields())
            names.insert(field.name);
        require(names ==
                    std::set<std::string>{"guidance_scale", "negative_prompt", "num_steps", "seed"},
                "text_to_image declares an unexpected config field set");

        for (int run = 0; run < runs; ++run) {
            trtmc::Config config;
            config.add("num_steps", steps);
            config.add("guidance_scale", 4.5);
            config.add("seed", static_cast<std::int64_t>(run));
            const auto start = std::chrono::steady_clock::now();
            const auto image =
                task.run({"A capybara holding a sign that reads Hello World"}, config);
            const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
            require(image.channels() == 3 && image.height() > 0 && image.width() > 0,
                    "text_to_image returned an invalid image shape");
            const auto pixels = image.pixels();
            require(pixels.size() == static_cast<std::size_t>(image.height()) * image.width() * 3U,
                    "text_to_image pixel count does not match its shape");
            double sum = 0.0;
            for (const float value : pixels) {
                require(std::isfinite(value) && value >= 0.0F && value <= 1.0F,
                        "text_to_image pixel outside [0, 1]");
                sum += value;
            }
            std::cout << "run " << run << ": " << image.height() << "x" << image.width()
                      << " steps=" << steps << " mean=" << sum / static_cast<double>(pixels.size())
                      << " seconds=" << seconds.count() << "\n";
        }

        trtmc::Config zero_steps;
        zero_steps.add("num_steps", std::int64_t{0});
        require_rejected([&] { (void)task.run({"a cat"}, zero_steps); }, "num_steps = 0");
        trtmc::Config unknown;
        unknown.add("cfg_scale", 2.0);
        require_rejected([&] { (void)task.run({"a cat"}, unknown); }, "an undeclared config");
        const std::vector<float> short_latents(7, 0.0F);
        require_rejected(
            [&] { (void)task.run({"a cat", {short_latents.data(), short_latents.size()}}); },
            "initial latents of the wrong size");
    } catch (const std::exception& error) {
        std::cerr << "stable_diffusion_3 C++ SDK consumer error: " << error.what() << "\n";
        return 1;
    }
    std::cout << "stable_diffusion_3 C++ SDK consumer: all checks passed.\n";
    return 0;
}
