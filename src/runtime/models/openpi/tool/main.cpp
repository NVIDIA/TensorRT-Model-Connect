/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/api.h"
#include "runtime/models/openpi/tool/action_request_json.h"
#include "runtime/models/openpi/tool/qualification_diagnostics.h"
#include "trtmc/pipeline.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct RunnerArgs {
    std::string bundle_path;
    std::string request_json;
    std::string output_json;
    std::string qualification_diagnostics;
    int benchmark{0};
    int warmup{1};
};

void print_usage(std::ostream& output) {
    output << "Usage: trtmc-openpi <bundle.trtfb> --request-json PATH [--output-json PATH]\n"
              "                    [--benchmark N] [--warmup N]\n"
              "                    [--qualification-diagnostics DIR]\n";
}

int parse_nonnegative_integer(std::string_view text, std::string_view option,
                              bool strictly_positive) {
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < 0 ||
        (strictly_positive && value == 0)) {
        throw std::invalid_argument(std::string(option) + (strictly_positive
                                                               ? " expects an integer > 0"
                                                               : " expects an integer >= 0"));
    }
    return value;
}

std::string require_option_value(int argc, char** argv, int& index, std::string_view option) {
    if (index + 1 >= argc)
        throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[++index];
}

void parse_option(RunnerArgs& result, int argc, char** argv, int& index, std::string_view option) {
    const auto value = require_option_value(argc, argv, index, option);
    if (option == "--request-json") {
        result.request_json = value;
    } else if (option == "--output-json") {
        result.output_json = value;
    } else if (option == "--qualification-diagnostics") {
        result.qualification_diagnostics = value;
    } else if (option == "--benchmark") {
        result.benchmark = parse_nonnegative_integer(value, option, true);
    } else {
        result.warmup = parse_nonnegative_integer(value, option, false);
    }
}

bool is_value_option(std::string_view argument) {
    return argument == "--request-json" || argument == "--output-json" ||
           argument == "--qualification-diagnostics" || argument == "--benchmark" ||
           argument == "--warmup";
}

void parse_argument(RunnerArgs& result, int argc, char** argv, int& index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
        print_usage(std::cout);
        std::exit(EXIT_SUCCESS);
    }
    if (is_value_option(argument)) {
        parse_option(result, argc, argv, index, argument);
        return;
    }
    if (!argument.empty() && argument.front() == '-')
        throw std::invalid_argument("unknown option: " + std::string(argument));
    if (result.bundle_path.empty()) {
        result.bundle_path = argument;
        return;
    }
    throw std::invalid_argument("unexpected positional argument: " + std::string(argument));
}

void validate_args(const RunnerArgs& result) {
    if (result.bundle_path.empty() || result.request_json.empty())
        throw std::invalid_argument("OpenPI runner requires bundle + --request-json");
    if (!result.qualification_diagnostics.empty() && result.benchmark > 0) {
        throw std::invalid_argument(
            "--qualification-diagnostics cannot be combined with --benchmark");
    }
}

RunnerArgs parse_args(int argc, char** argv) {
    if (argc < 2)
        throw std::invalid_argument("missing OpenPI bundle path");

    RunnerArgs result;
    for (int index = 1; index < argc; ++index)
        parse_argument(result, argc, argv, index);
    validate_args(result);
    return result;
}

trtmc::openpi::ActionRequest load_action_request(const RunnerArgs& args) {
    const auto document = trtmc::openpi::tool::read_action_request_json(args.request_json);
    const std::filesystem::path request_dir =
        std::filesystem::path(args.request_json).parent_path();

    trtmc::openpi::ActionRequest request;
    request.prompt = document.prompt;
    request.state = document.state;
    request.initial_noise = document.initial_noise;
    request.seed = document.seed;
    request.denoise_steps = document.denoise_steps;
    request.cameras.reserve(document.cameras.size());
    for (const auto& camera_file : document.cameras) {
        std::filesystem::path image_path(camera_file.path);
        if (image_path.is_relative() && !request_dir.empty())
            image_path = request_dir / image_path;
        auto image = trtmc::io::read_image(image_path.string());
        if (image.empty())
            throw std::runtime_error("failed to load OpenPI camera image: " + image_path.string());

        trtmc::openpi::RobotImage camera;
        camera.name = camera_file.name;
        camera.pixels = std::move(image.pixels);
        camera.height = image.height;
        camera.width = image.width;
        camera.channels = 3;
        camera.valid = camera_file.valid;
        if (!camera.valid)
            std::fill(camera.pixels.begin(), camera.pixels.end(), 0.0F);
        request.cameras.push_back(std::move(camera));
    }
    return request;
}

double nearest_rank_percentile(std::vector<double> values, std::size_t percentile) {
    if (values.empty() || percentile == 0U || percentile > 100U)
        throw std::invalid_argument("invalid benchmark percentile request");
    std::sort(values.begin(), values.end());
    const std::size_t rank = (percentile * values.size() + 99U) / 100U;
    return values[rank - 1U];
}

void write_result(const RunnerArgs& args, const trtmc::openpi::ActionResult& result) {
    const std::string output = trtmc::openpi::tool::serialize_action_result_json(result);
    if (args.output_json.empty()) {
        std::cout << output << '\n';
        return;
    }

    const std::filesystem::path output_path(args.output_json);
    const auto parent = output_path.parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream stream(output_path, std::ios::out | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("failed to open action result JSON: " + args.output_json);
    stream << output << '\n';
    if (!stream)
        throw std::runtime_error("failed to write action result JSON: " + args.output_json);
    std::cerr << "Actions saved: " << args.output_json << '\n';
}

int run(const RunnerArgs& args) {
    const auto request = load_action_request(args);
    auto pipeline = trtmc::load(args.bundle_path);
    if (!pipeline)
        throw std::runtime_error("failed to load OpenPI bundle");

    auto* action_pipeline = dynamic_cast<trtmc::openpi::IOpenPIActionPipeline*>(pipeline.get());
    if (action_pipeline == nullptr)
        throw std::runtime_error("bundle does not implement OpenPI action inference");

    trtmc::openpi::ActionResult result;
    if (!args.qualification_diagnostics.empty()) {
        auto* diagnostic_pipeline =
            dynamic_cast<trtmc::openpi::IOpenPIDiagnosticPipeline*>(pipeline.get());
        if (diagnostic_pipeline == nullptr)
            throw std::runtime_error("bundle does not implement OpenPI qualification capture");
        auto diagnostics = diagnostic_pipeline->predict_actions_with_diagnostics(request);
        result = diagnostics.result;
        const auto manifest = trtmc::openpi::tool::write_qualification_diagnostics(
            diagnostics, args.qualification_diagnostics, pipeline->model_id());
        std::cerr << "Qualification diagnostics saved: " << manifest.string() << '\n';
    } else if (args.benchmark > 0) {
        for (int iteration = 0; iteration < args.warmup; ++iteration)
            result = action_pipeline->predict_actions(request);

        std::vector<double> wall_times_ms;
        wall_times_ms.reserve(static_cast<std::size_t>(args.benchmark));
        for (int iteration = 0; iteration < args.benchmark; ++iteration) {
            const auto begin = std::chrono::steady_clock::now();
            result = action_pipeline->predict_actions(request);
            const auto end = std::chrono::steady_clock::now();
            wall_times_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
        const double mean = std::accumulate(wall_times_ms.begin(), wall_times_ms.end(), 0.0) /
                            static_cast<double>(wall_times_ms.size());
        std::cerr << std::fixed << std::setprecision(6)
                  << "[trtmc.openpi.benchmark] action_ms=" << mean
                  << " p50_ms=" << nearest_rank_percentile(wall_times_ms, 50U)
                  << " p95_ms=" << nearest_rank_percentile(wall_times_ms, 95U)
                  << " iterations=" << args.benchmark << " warmup=" << args.warmup << '\n';
    } else {
        result = action_pipeline->predict_actions(request);
    }

    write_result(args, result);
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
}
