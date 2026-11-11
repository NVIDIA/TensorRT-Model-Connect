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
#include <string>
#include <trtmc/numeric.hpp>
#include <vector>

namespace {
std::vector<float> read_f32(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (bytes <= 0 || static_cast<std::size_t>(bytes) % sizeof(float))
        throw std::runtime_error("input must contain complete float32 values");
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!input)
        throw std::runtime_error("could not read input");
    return values;
}
void array(const float* data, std::uint64_t count) {
    std::cout << '[';
    for (std::uint64_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i]))
            throw std::runtime_error("output must be finite");
        std::cout << (i ? "," : "") << data[i];
    }
    std::cout << ']';
}
std::vector<float> snapshot(const trtmc_point_forecast_view_v1& view) {
    return {view.values.data, view.values.data + view.values.count};
}
std::vector<float> snapshot(const trtmc_regression_values_view_v1& view) {
    return {view.values.data, view.values.data + view.values.size};
}
std::vector<float> snapshot(const trtmc_regression_distribution_view_v1& view) {
    std::vector<float> data;
    for (std::uint64_t i = 0; i < view.parameter_count; ++i)
        data.insert(data.end(), view.parameters[i].values,
                    view.parameters[i].values + view.parameters[i].target_count);
    return data;
}
void emit(const trtmc_point_forecast_view_v1& view) {
    if (!view.values.rows || !view.values.columns ||
        view.values.count != view.values.rows * view.values.columns ||
        view.axes.horizon_steps.size != view.values.rows || view.axes.channel_names.size ||
        view.axes.channel_units.size)
        throw std::runtime_error("invalid forecast metadata");
    std::cout << "\"kind\":\"point\",\"shape\":[" << view.values.rows << ',' << view.values.columns
              << "],\"axes\":[\"horizon\",\"channel\"],\"channel_names\":[],\"channel_units\":[],"
                 "\"horizon_steps\":[";
    for (std::uint64_t i = 0; i < view.axes.horizon_steps.size; ++i) {
        if (view.axes.horizon_steps.data[i] != static_cast<std::int64_t>(i) + 1)
            throw std::runtime_error("invalid horizon offsets");
        std::cout << (i ? "," : "") << view.axes.horizon_steps.data[i];
    }
    std::cout << "],\"values\":";
    array(view.values.data, view.values.count);
}
void emit(const trtmc_regression_values_view_v1& view) {
    if (!view.values.size || view.target_names.size || view.target_units.size)
        throw std::runtime_error("invalid regression metadata");
    std::cout << "\"kind\":\"regression_values\",\"target_count\":" << view.values.size
              << ",\"target_names\":[],\"target_units\":[],\"axes\":[\"target\"],\"values\":";
    array(view.values.data, view.values.size);
}
void emit(const trtmc_regression_distribution_view_v1& view) {
    const char* distribution = view.distribution == TRTMC_DISTRIBUTION_NORMAL      ? "normal"
                               : view.distribution == TRTMC_DISTRIBUTION_STUDENT_T ? "student_t"
                               : view.distribution == TRTMC_DISTRIBUTION_NEGATIVE_BINOMIAL
                                   ? "negative_binomial"
                                   : nullptr;
    if (!distribution || !view.target_count || !view.parameter_count || view.target_names.size ||
        view.target_units.size)
        throw std::runtime_error("invalid distribution metadata");
    std::cout << "\"kind\":\"regression_distribution\",\"distribution\":\"" << distribution
              << "\",\"target_count\":" << view.target_count
              << ",\"target_names\":[],\"target_units\":[],\"parameters\":{";
    for (std::uint64_t i = 0; i < view.parameter_count; ++i) {
        const auto& item = view.parameters[i];
        if (item.target_count != view.target_count)
            throw std::runtime_error("parameter target count mismatch");
        std::cout << (i ? "," : "") << '"' << trtmc::detail::string_view(item.name) << "\":";
        array(item.values, item.target_count);
    }
    std::cout << '}';
}
template <class Function>
void rejects(trtmc_status expected, Function function) {
    try {
        function();
    } catch (const trtmc::Error& error) {
        if (error.code() == expected)
            return;
        throw;
    }
    throw std::runtime_error("invalid call did not fail");
}
template <class Task, class Request>
void exercise(char** argv) {
    auto values = read_f32(argv[3]);
    const auto mask = read_f32(argv[4]);
    const auto channels = std::stoull(argv[6]);
    if (!channels || values.size() % channels || mask.size() != values.size())
        throw std::runtime_error("invalid multichannel input");
    std::vector<std::uint8_t> observed;
    for (const auto flag : mask) {
        if (flag != 0 && flag != 1)
            throw std::runtime_error("mask must be binary");
        observed.push_back(flag == 1);
    }
    std::vector<float> expected;
    auto result = [&] {
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        const auto task = model.task<Task>();
        const auto fields = task.config_fields();
        if (fields.size() != 1 || fields[0].name != "frequency")
            throw std::runtime_error("frequency schema must be discoverable");
        if (Task::kTask == trtmc::SeriesToPointForecast::kTask)
            rejects(TRTMC_UNSUPPORTED, [&] { model.task<trtmc::SeriesToRegressionValues>(); });
        else
            rejects(TRTMC_UNSUPPORTED, [&] { model.task<trtmc::SeriesToPointForecast>(); });
        const Request request{{{{values.data(), values.size()}, values.size() / channels, channels},
                               {observed.data(), observed.size()}}};
        rejects(TRTMC_INVALID_ARGUMENT, [&] { task.run(Request{}); });
        rejects(TRTMC_INVALID_CONFIG, [&] { task.run(request, {{"frequency", true}}); });
        rejects(TRTMC_INVALID_CONFIG, [&] { task.run(request, {{"frequency", 1}}); });
        rejects(TRTMC_INVALID_CONFIG, [&] { task.run(request, {{"unknown", 0}}); });
        auto first = task.run(request);
        expected = snapshot(first.view());
        auto repeated = task.run(request, {{"frequency", 0}});
        if (snapshot(first.view()) != expected || snapshot(repeated.view()) != expected)
            throw std::runtime_error("repeated calls changed an owned result or zero default");
        return first;
    }();
    values.clear();
    values.shrink_to_fit();
    observed.clear();
    observed.shrink_to_fit();
    if (snapshot(result.view()) != expected)
        throw std::runtime_error("result must outlive model and input buffers");
    std::cout << "{\"task\":\"" << Task::kTask << "\",";
    emit(result.view());
    std::cout << "}\n";
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 7)
        return 2;
    try {
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        const std::string task = argv[5];
        if (task == trtmc::SeriesToPointForecast::kTask)
            exercise<trtmc::SeriesToPointForecast, trtmc::SeriesToPointForecastRequest>(argv);
        else if (task == trtmc::SeriesToRegressionValues::kTask)
            exercise<trtmc::SeriesToRegressionValues, trtmc::SeriesToRegressionValuesRequest>(argv);
        else if (task == trtmc::SeriesToRegressionDistribution::kTask)
            exercise<trtmc::SeriesToRegressionDistribution,
                     trtmc::SeriesToRegressionDistributionRequest>(argv);
        else
            throw std::runtime_error("unsupported test Task");
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
