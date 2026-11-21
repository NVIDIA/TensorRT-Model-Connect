/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * C++ SDK consumer for the chroma family: text_to_image through the public
 * convenience wrapper, Config discovery, and latent replay determinism.
 *
 * Usage: test_chroma_sdk_cpp <bundle_path> <runtime_root>
 */

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <trtmc/image.hpp>
#include <trtmc/trtmc.hpp>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <bundle_path> <runtime_root>\n";
        return 2;
    }
    try {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        auto task = model.task<trtmc::TextToImage>();

        std::set<std::string> names;
        for (const auto& field : task.config_fields())
            names.insert(std::string(field.name));
        for (const char* name : {"negative_prompt", "num_inference_steps", "guidance_scale", "seed",
                                 "height", "width"})
            if (names.count(name) == 0)
                throw std::runtime_error(std::string("missing Config field ") + name);

        const trtmc::Config config{{"num_inference_steps", std::int64_t{2}},
                                   {"guidance_scale", 3.0},
                                   {"negative_prompt", std::string("blurry")}};
        const auto first = task.run({"A lighthouse at dusk", {}},
                                    trtmc::Config{{"num_inference_steps", std::int64_t{2}},
                                                  {"guidance_scale", 3.0},
                                                  {"seed", std::int64_t{11}}});
        if (first.channels() != 3 ||
            first.pixels().size() != std::size_t{first.height()} * first.width() * 3)
            throw std::runtime_error("text_to_image returned an invalid image");

        // Replay: the same packed latents give the same image.
        const std::size_t latent_count = std::size_t{first.height() / 8} * (first.width() / 8) * 16;
        std::vector<float> latents(latent_count);
        for (std::size_t index = 0; index < latents.size(); ++index)
            latents[index] = std::sin(static_cast<float>(index) * 0.37F);
        const trtmc::Span<const float> view{latents.data(), latents.size()};
        const auto replay_a = task.run({"A lighthouse at dusk", view}, config);
        const auto replay_b = task.run({"A lighthouse at dusk", view}, config);
        const auto a = replay_a.pixels();
        const auto b = replay_b.pixels();
        for (std::size_t index = 0; index < a.size(); ++index) {
            if (!std::isfinite(a[index]) || a[index] != b[index])
                throw std::runtime_error("latent replay is not deterministic");
        }
        std::cout << "text_to_image: " << first.width() << "x" << first.height()
                  << ", replay deterministic\n";

        bool rejected = false;
        try {
            (void)task.run({"x", view}, trtmc::Config{{"seed", std::int64_t{1}}});
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("seed together with initial latents was accepted");
    } catch (const std::exception& error) {
        std::cerr << "chroma C++ SDK consumer error: " << error.what() << "\n";
        return 1;
    }
    std::cout << "chroma C++ SDK consumer: passed\n";
    return 0;
}
