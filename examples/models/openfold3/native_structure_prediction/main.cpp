/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/pipeline.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string bundle;
    std::string request;
    std::string output;
    std::string backend_dir;
    std::string model_plugin_dir;
};

std::string takeValue(int& index, int argc, char** argv, const std::string& option) {
    if (++index >= argc)
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--request")
            options.request = takeValue(index, argc, argv, argument);
        else if (argument == "--output")
            options.output = takeValue(index, argc, argv, argument);
        else if (argument == "--backend-dir")
            options.backend_dir = takeValue(index, argc, argv, argument);
        else if (argument == "--model-plugin-dir")
            options.model_plugin_dir = takeValue(index, argc, argv, argument);
        else if (!argument.empty() && argument.front() == '-')
            throw std::invalid_argument("unknown option: " + argument);
        else if (options.bundle.empty())
            options.bundle = argument;
        else
            throw std::invalid_argument("only one bundle may be specified");
    }
    if (options.bundle.empty() || options.request.empty() || options.output.empty())
        throw std::invalid_argument("bundle, --request, and --output are required");
    return options;
}

std::string readFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open input: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void writeFile(const std::filesystem::path& path, const std::string& contents) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output)
        throw std::runtime_error("failed to write output: " + path.string());
}

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " MODEL.bundle --request query.json --output prediction.cif "
                 "[--backend-dir DIR] [--model-plugin-dir DIR]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        trtmc::LoadOptions load_options;
        if (!options.backend_dir.empty())
            load_options.backend_search_paths.push_back(options.backend_dir);
        if (!options.model_plugin_dir.empty())
            load_options.model_plugin_search_paths.push_back(options.model_plugin_dir);
        auto pipeline = trtmc::load(options.bundle, load_options);
        const auto request =
            pipeline->prepare_structure_input(readFile(options.request), options.request);
        const auto result = pipeline->predict_structure(request);
        writeFile(options.output, result.structure);
        writeFile(options.output + ".metadata.json", result.metadata_json);
        std::cout << "Wrote " << result.confidence.plddt.size()
                  << " atom confidence values; average pLDDT=" << result.confidence.complex_plddt
                  << "; pTM=" << result.confidence.ptm << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
