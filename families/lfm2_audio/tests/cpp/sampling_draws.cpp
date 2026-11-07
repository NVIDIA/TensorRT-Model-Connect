/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/sampling.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: sampling_draws SEED COUNT");
        const auto seed = std::stoull(argv[1]);
        const auto count = std::stoull(argv[2]);
        if (seed > std::numeric_limits<std::uint32_t>::max() || count == 0)
            throw std::invalid_argument("invalid draw fixture parameters");
        std::mt19937 recorded(static_cast<std::uint32_t>(seed));
        std::mt19937 sampled(static_cast<std::uint32_t>(seed));
        std::cout << "{\"seed\":" << seed << ",\"count\":" << count
                  << ",\"algorithm\":\"mt19937-uniform-real-double\",\"uniforms\":["
                  << std::setprecision(17);
        for (std::size_t i = 0; i < count; ++i) {
            // The same expression used by the family sampler; a separate
            // stream checks that actual sample() consumes precisely one draw.
            const double uniform = std::uniform_real_distribution<double>{0, 1}(recorded);
            const int token = trtmc::lfm2_audio::sample({0, 0, 0}, 1.0F, 0, sampled);
            if (token != static_cast<int>(uniform * 3))
                throw std::runtime_error("native draw consumption changed");
            if (i)
                std::cout << ',';
            std::cout << uniform;
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
