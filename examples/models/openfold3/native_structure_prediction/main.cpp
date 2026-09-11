/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/openfold3/structure_prediction.h"
#include "trtmc/runtime/family_loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uintmax_t kMaximumRequestBytes = 1U << 20;

struct Options {
    std::string bundle;
    std::string request;
    std::string output;
    std::string metadata;
    std::string runtime_root;
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
        else if (argument == "--metadata")
            options.metadata = takeValue(index, argc, argv, argument);
        else if (argument == "--runtime-root")
            options.runtime_root = takeValue(index, argc, argv, argument);
        else if (!argument.empty() && argument.front() == '-')
            throw std::invalid_argument("unknown option: " + argument);
        else if (options.bundle.empty())
            options.bundle = argument;
        else
            throw std::invalid_argument("only one bundle may be specified");
    }
    if (options.bundle.empty() || options.request.empty() || options.output.empty() ||
        options.metadata.empty() || options.runtime_root.empty())
        throw std::invalid_argument(
            "bundle, --request, --output, --metadata, and --runtime-root are required");
    if (std::filesystem::absolute(options.output).lexically_normal() ==
        std::filesystem::absolute(options.metadata).lexically_normal())
        throw std::invalid_argument("structure and metadata outputs must be different files");
    return options;
}

std::string readRequest(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || !std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status))
        throw std::invalid_argument("structure request must be a regular non-symlink file");
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > kMaximumRequestBytes)
        throw std::invalid_argument("structure request exceeds the 1 MiB limit");
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open input: " + path.string());
    std::string result(static_cast<std::size_t>(size), '\0');
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!input && !result.empty())
        throw std::runtime_error("failed to read input: " + path.string());
    return result;
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
                 "--metadata prediction.json --runtime-root DIR\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parseOptions(argc, argv);
        auto task = trtmc::load_task(options.bundle, options.runtime_root);
        auto* prediction = dynamic_cast<trtmc::openfold3::IStructurePrediction*>(task.get());
        if (prediction == nullptr)
            throw std::runtime_error("bundle does not implement structure prediction");
        const auto result = prediction->predict_structure(readRequest(options.request));
        writeFile(options.output, result.structure);
        writeFile(options.metadata, result.metadata_json);
        std::cout << "Wrote structure and confidence metadata\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
