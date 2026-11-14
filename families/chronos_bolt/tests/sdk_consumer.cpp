/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C++ SDK consumer for the official Chronos-Bolt checkpoint. */
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
        if (input_bytes <= 0 || static_cast<std::size_t>(input_bytes) % sizeof(float) != 0)
            throw std::runtime_error("input must contain complete float32 timesteps");
        std::vector<float> input(static_cast<std::size_t>(input_bytes) / sizeof(float));
        const auto input_rows = input.size();
        file.seekg(0);
        file.read(reinterpret_cast<char*>(input.data()),
                  static_cast<std::streamsize>(input.size() * sizeof(float)));
        if (!file || file.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("unable to read the complete float32 history");
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::SeriesToQuantileForecast>();
            return task.run({{{{input.data(), input.size()}, input_rows, 1}, {}}});
        }();
        input.clear();
        input.shrink_to_fit();
        const auto view = result.view();
        if (view.channels != 1 || view.quantile_levels.size == 0 || view.horizon == 0 ||
            view.value_count != view.quantile_levels.size * view.horizon * view.channels ||
            view.axes.horizon_steps.size != view.horizon)
            throw std::runtime_error("forecast must retain the exact [quantile,horizon,1] layout");
        for (std::uint64_t i = 0; i < view.value_count; ++i)
            if (!std::isfinite(view.values[i]))
                throw std::runtime_error("forecast contains a nonfinite value");
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "{\"task\":\"series_to_quantile_forecast\",\"input_shape\":[" << input_rows
                  << ",1],\"shape\":[" << view.quantile_levels.size << "," << view.horizon
                  << ",1],\"axes\":[\"quantile\",\"horizon\",\"channel\"],"
                     "\"quantile_levels\":[";
        for (std::uint64_t i = 0; i < view.quantile_levels.size; ++i) {
            if (i)
                std::cout << ',';
            std::cout << view.quantile_levels.data[i];
        }
        std::cout << "],\"horizon_steps\":[";
        for (std::uint64_t i = 0; i < view.axes.horizon_steps.size; ++i) {
            if (i)
                std::cout << ',';
            std::cout << view.axes.horizon_steps.data[i];
        }
        std::cout << "],\"channel_names\":";
        json_strings(view.axes.channel_names);
        std::cout << ",\"channel_units\":";
        json_strings(view.axes.channel_units);
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << ",\"values\":[";
        for (std::uint64_t i = 0; i < view.value_count; ++i) {
            if (i)
                std::cout << ',';
            std::cout << view.values[i];
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
