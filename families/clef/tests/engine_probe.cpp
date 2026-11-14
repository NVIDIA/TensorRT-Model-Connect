/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

std::vector<char> read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read " + path.string());
    return {(std::istreambuf_iterator<char>(input)), {}};
}

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: engine_probe PLAN DATA_DIRECTORY");
        const auto plan = read(argv[1]);
        const std::filesystem::path root(argv[2]);
        std::ifstream metadata(root / "inputs.json");
        const auto spec = nlohmann::json::parse(metadata);
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        auto engine = backend->create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok())
            throw std::runtime_error("cannot load TensorRT engine");
        std::unordered_map<std::string, std::vector<char>> buffers;
        trtmc::TensorMap inputs;
        for (const auto& entry : spec.items()) {
            buffers[entry.key()] = read(root / (entry.key() + ".bin"));
            const auto dtype = entry.value().at("dtype").get<std::string>();
            inputs[entry.key()] = {buffers.at(entry.key()).data(),
                                   entry.value().at("shape").get<std::vector<int64_t>>(),
                                   dtype == "bf16"    ? trtmc::DType::kBFloat16
                                   : dtype == "int32" ? trtmc::DType::kInt32
                                                      : trtmc::DType::kFloat32};
            if (inputs.at(entry.key()).nbytes() != buffers.at(entry.key()).size())
                throw std::runtime_error("invalid tensor bytes: " + entry.key());
        }
        const auto outputs = engine->forward(inputs);
        nlohmann::json info = nlohmann::json::object();
        for (const auto& [name, value] : outputs) {
            std::ofstream out(root / (name + ".out.bin"), std::ios::binary);
            out.write(static_cast<const char*>(value.data), value.nbytes());
            info[name] = {{"shape", value.shape}, {"bytes", value.nbytes()}};
        }
        std::cout << info.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
