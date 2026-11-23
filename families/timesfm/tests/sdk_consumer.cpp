/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <trtmc/numeric.hpp>
#include <vector>

namespace {
std::vector<float> read_f32(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = stream.tellg();
    if (bytes <= 0 || static_cast<std::size_t>(bytes) % sizeof(float))
        throw std::runtime_error("input must contain complete float32 timesteps");
    std::vector<float> result(static_cast<std::size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    if (!stream || stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("could not read complete input");
    return result;
}

template <class Function>
void rejects(trtmc_status expected, Function operation) {
    try {
        operation();
    } catch (const trtmc::Error& error) {
        if (error.code() == expected)
            return;
        throw;
    }
    throw std::runtime_error("SDK accepted an invalid request/configuration");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6)
        return 2;
    try {
        static_assert(sizeof(float) == 4, "input uses float32");
        auto values = read_f32(argv[3]);
        const auto mask = read_f32(argv[4]);
        if (mask.size() != values.size())
            throw std::runtime_error("mask must match the history");
        std::vector<std::uint8_t> observed;
        for (const auto value : mask) {
            if (value != 0 && value != 1)
                throw std::runtime_error("mask must be binary");
            observed.push_back(value == 1);
        }
        const auto frequency = std::stoll(argv[5]);
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::SeriesToPointForecast>();
            const auto fields = task.config_fields();
            if (fields.size() != 1 || fields[0].name != "frequency")
                throw std::runtime_error("frequency must be discoverable");
            const trtmc::SeriesToPointForecastRequest request{trtmc::SeriesHistory::from_flat(
                {values.data(), values.size()}, {observed.data(), observed.size()})};
            rejects(TRTMC_INVALID_CONFIG, [&] { task.run(request, {{"frequency", true}}); });
            rejects(TRTMC_INVALID_CONFIG, [&] { task.run(request, {{"frequency", -1}}); });
            rejects(TRTMC_INVALID_CONFIG, [&] { task.run(request, {{"unknown", 0}}); });
            auto first = task.run(request, {{"frequency", frequency}});
            const auto retained = first.view();
            const auto repeated = task.run(request, {{"frequency", frequency}});
            const auto other = repeated.view();
            if (retained.values.count != other.values.count ||
                !std::equal(retained.values.data, retained.values.data + retained.values.count,
                            other.values.data))
                throw std::runtime_error("repeated requests changed a retained result");
            if (frequency == 0) {
                const auto default_result = task.run(request);
                const auto default_view = default_result.view();
                if (default_view.values.count != retained.values.count ||
                    !std::equal(retained.values.data, retained.values.data + retained.values.count,
                                default_view.values.data))
                    throw std::runtime_error("omitted frequency must default to zero");
            }
            return first;
        }();
        values.clear();
        values.shrink_to_fit();
        observed.clear();
        observed.shrink_to_fit();
        const auto view = result.view();
        if (!view.values.rows || view.values.columns != 1 ||
            view.values.count != view.values.rows ||
            view.axes.horizon_steps.size != view.values.rows || view.axes.channel_names.size ||
            view.axes.channel_units.size)
            throw std::runtime_error("point output must retain [horizon,1] and axes");
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                  << "{\"task\":\"series_to_point_forecast\",\"kind\":\"point\","
                     "\"shape\":["
                  << view.values.rows
                  << ",1],\"axes\":[\"horizon\",\"channel\"],"
                     "\"channel_names\":[],\"channel_units\":[],\"horizon_steps\":[";
        for (std::uint64_t i = 0; i < view.axes.horizon_steps.size; ++i) {
            if (view.axes.horizon_steps.data[i] != static_cast<std::int64_t>(i) + 1)
                throw std::runtime_error("horizon offsets must be complete");
            std::cout << (i ? "," : "") << view.axes.horizon_steps.data[i];
        }
        std::cout << "],\"values\":[";
        for (std::uint64_t i = 0; i < view.values.count; ++i) {
            if (!std::isfinite(view.values.data[i]))
                throw std::runtime_error("nonfinite point forecast");
            std::cout << (i ? "," : "") << view.values.data[i];
        }
        std::cout << "]}\n";
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
