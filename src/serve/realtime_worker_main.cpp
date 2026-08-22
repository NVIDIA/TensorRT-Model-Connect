/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "serve/realtime_worker.h"
#include "trtmc/pipeline.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaxPathBytes = 4096U;
constexpr std::size_t kMaxSearchDirectories = 64U;

struct Options {
    std::string bundle;
    std::vector<std::string> backend_dirs;
    std::vector<std::string> model_plugin_dirs;
};

void print_usage() {
    std::cerr << "Usage: trtmc_realtime_worker --bundle PATH "
                 "[--backend-dir DIR] [--model-plugin-dir DIR]\n";
}

bool take_path(int argc, char** argv, int& index, std::string& output) {
    if (index + 1 >= argc)
        return false;
    output = argv[++index];
    return !output.empty() && output.size() <= kMaxPathBytes;
}

bool append_path(int argc, char** argv, int& index, std::vector<std::string>& output) {
    std::string value;
    if (!take_path(argc, argv, index, value) || output.size() >= kMaxSearchDirectories)
        return false;
    output.push_back(std::move(value));
    return true;
}

bool parse_option(int argc, char** argv, int& index, Options& options) {
    const std::string option = argv[index];
    if (option == "--bundle")
        return options.bundle.empty() && take_path(argc, argv, index, options.bundle);
    if (option == "--backend-dir")
        return append_path(argc, argv, index, options.backend_dirs);
    if (option == "--model-plugin-dir")
        return append_path(argc, argv, index, options.model_plugin_dirs);
    return false;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        if (!parse_option(argc, argv, index, options))
            return false;
    }
    return !options.bundle.empty();
}

std::unique_ptr<trtmc::IPipeline> load_pipeline(const Options& options) {
    trtmc::LoadOptions load_options;
    load_options.backend_search_paths = options.backend_dirs;
    load_options.model_plugin_search_paths = options.model_plugin_dirs;
    return trtmc::load(options.bundle, load_options);
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "Error: invalid realtime worker arguments\n";
        print_usage();
        return 2;
    }

    return trtmc::serve::run_realtime_worker(
        [&options]() {
            try {
                return load_pipeline(options);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: bundle or runtime configuration was rejected\n";
                throw;
            } catch (...) {
                std::cerr << "Error: speech pipeline initialization failed\n";
                throw;
            }
        },
        std::cin, std::cout);
}
