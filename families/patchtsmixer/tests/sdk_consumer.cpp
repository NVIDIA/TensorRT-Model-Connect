/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the official PatchTSMixer checkpoint. */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <trtmc/numeric.hpp>
#include <vector>

namespace {
void json_string(trtmc_string_view value) {
    std::cout << '"';
    for (std::uint64_t i = 0; i < value.size; ++i) {
        const auto byte = static_cast<unsigned char>(value.data[i]);
        if (byte == '"' || byte == '\\')
            std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32)
            std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(byte) << std::dec << std::setfill(' ');
        else
            std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}
void json_strings(trtmc_strings_view values) {
    std::cout << '[';
    for (std::uint64_t i = 0; i < values.size; ++i) {
        if (i)
            std::cout << ',';
        json_string(values.data[i]);
    }
    std::cout << ']';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT VALUES_F32\n";
        return 2;
    }
    try {
        static_assert(sizeof(float) == 4, "input uses float32");
        std::ifstream file(argv[3], std::ios::binary | std::ios::ate);
        const auto input_bytes = file.tellg();
        if (input_bytes <= 0 || static_cast<std::size_t>(input_bytes) % (7 * sizeof(float)) != 0)
            throw std::runtime_error("input must contain complete seven-channel float32 timesteps");
        std::vector<float> input(static_cast<std::size_t>(input_bytes) / sizeof(float));
        const auto input_rows = input.size() / 7;
        file.seekg(0);
        file.read(reinterpret_cast<char*>(input.data()),
                  static_cast<std::streamsize>(input.size() * sizeof(float)));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("unable to read the complete float32 history");
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::SeriesToPointForecast>();
            return task.run({{{{input.data(), input.size()}, input_rows, 7}, {}}});
        }();
        input.clear();
        input.shrink_to_fit();
        const auto view = result.view();
        if (view.values.rows != 96 || view.values.columns != 7 || view.values.count != 96 * 7 ||
            view.axes.horizon_steps.size != 96)
            throw std::runtime_error(
                "forecast must retain the exact [96,7] horizon/channel layout");
        for (std::uint64_t i = 0; i < view.values.count; ++i)
            if (!std::isfinite(view.values.data[i]))
                throw std::runtime_error("forecast contains a nonfinite value");
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"series_to_point_forecast\",\"input_shape\":[" << input_rows
                  << ",7],\"shape\":[96,7],\"axes\":[\"horizon\",\"channel\"],\"horizon_steps\":[";
        for (std::uint64_t i = 0; i < view.axes.horizon_steps.size; ++i) {
            if (i)
                std::cout << ',';
            std::cout << view.axes.horizon_steps.data[i];
        }
        std::cout << "],\"channel_names\":";
        json_strings(view.axes.channel_names);
        std::cout << ",\"channel_units\":";
        json_strings(view.axes.channel_units);
        std::cout << ",\"values\":[";
        for (std::uint64_t i = 0; i < view.values.count; ++i) {
            if (i)
                std::cout << ',';
            std::cout << view.values.data[i];
        }
        std::cout << "]}\n";
        std::cout.flush();
        if (!std::cout)
            throw std::runtime_error("failed to write forecast output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
