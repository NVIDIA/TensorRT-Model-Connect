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
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uintmax_t kMaximumRequestBytes = 1U << 20;

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
        throw std::runtime_error("cannot open structure request: " + path.string());
    std::string result(static_cast<std::size_t>(size), '\0');
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!input && !result.empty())
        throw std::runtime_error("cannot read structure request: " + path.string());
    return result;
}

void writeOutput(const std::filesystem::path& path, const std::string& value) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output)
        throw std::runtime_error("cannot write output: " + path.string());
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: openfold3_qualification BUNDLE RUNTIME_ROOT REQUEST CIF METADATA");
        const auto structure_path = std::filesystem::absolute(argv[4]).lexically_normal();
        const auto metadata_path = std::filesystem::absolute(argv[5]).lexically_normal();
        if (structure_path == metadata_path)
            throw std::invalid_argument("structure and metadata outputs must be different files");
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* prediction = dynamic_cast<trtmc::openfold3::IStructurePrediction*>(task.get());
        if (prediction == nullptr)
            throw std::runtime_error("bundle does not implement IStructurePrediction");
        const auto result = prediction->predict_structure(readRequest(argv[3]));
        writeOutput(structure_path, result.structure);
        writeOutput(metadata_path, result.metadata_json);
        std::cout << result.metadata_json;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
