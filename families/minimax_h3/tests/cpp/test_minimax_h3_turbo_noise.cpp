/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/turbo_noise.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_values(const std::vector<float>& values) {
    for (const float value : values) {
        std::uint32_t word;
        std::memcpy(&word, &value, sizeof(word));
        require(std::isfinite(value), "Noise must be finite");
        require((word & 0xffffU) == 0, "Noise must be BF16-representable");
    }
}

} // namespace

int main(int argc, char** argv) {
    using trtmc::minimax_h3::make_minimax_h3_turbo_noise;
    try {
        // Optional raw FP32 export is for an external PyTorch oracle; it never
        // changes generation and refuses to replace an existing reference file.
        if (argc == 5 && std::string(argv[1]) == "--dump") {
            const std::size_t count = std::stoull(argv[2]);
            const std::uint64_t seed = std::stoull(argv[3]);
            const std::filesystem::path destination(argv[4]);
            require(!std::filesystem::exists(destination), "Refusing to overwrite noise output");
            const auto start = std::chrono::steady_clock::now();
            const auto values = make_minimax_h3_turbo_noise(count, seed);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            check_values(values);
            std::ofstream output(destination, std::ios::binary);
            output.write(reinterpret_cast<const char*>(values.data()),
                         static_cast<std::streamsize>(values.size() * sizeof(float)));
            output.close();
            require(static_cast<bool>(output), "Noise export failed");
            std::cout << "count=" << count << " seed=" << seed << " host_seconds=" << seconds << '\n';
            return 0;
        }
        require(argc == 1, "Usage: test_minimax_h3_turbo_noise [--dump COUNT SEED OUTPUT.f32]");
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "Skipped: CUDA device properties are unavailable\n";
            return 77;
        }
        require(make_minimax_h3_turbo_noise(0, 42).empty(), "Empty noise request failed");
        bool rejected = false;
        try {
            make_minimax_h3_turbo_noise(std::numeric_limits<std::size_t>::max(), 42);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "Unsupported indexing must be rejected before allocating");
        for (const std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{255},
                                        std::size_t{256}, std::size_t{257}, std::size_t{1025}}) {
            const auto first = make_minimax_h3_turbo_noise(count, 42);
            const auto second = make_minimax_h3_turbo_noise(count, 42);
            require(first.size() == count, "Wrong number of noise values");
            require(first == second, "Noise must be deterministic for a fresh seed");
            check_values(first);
        }
        const auto first = make_minimax_h3_turbo_noise(262147, 42);
        const auto other_seed = make_minimax_h3_turbo_noise(first.size(), 43);
        require(first != other_seed, "Video and audio seeds must be independent");
        require(first != make_minimax_h3_turbo_noise(first.size(), (std::uint64_t{1} << 32) + 42),
                "Philox must use both halves of the seed");
        check_values(first);
        double sum = 0.0;
        double squares = 0.0;
        for (const float value : first) {
            sum += value;
            squares += static_cast<double>(value) * value;
        }
        const double mean = sum / first.size();
        const double variance = squares / first.size() - mean * mean;
        require(std::abs(mean) < 0.02 && variance > 0.97 && variance < 1.03,
                "Noise must have standard normal statistics");
        std::cout << "Turbo host Philox tests passed; mean=" << mean
                  << " variance=" << variance << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
