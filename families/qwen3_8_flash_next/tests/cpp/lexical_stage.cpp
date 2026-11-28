/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/runtime/trt_backend.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<char> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error(std::string("Missing input: ") + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    try {
        const auto plan = read_file(argv[1]);
        auto input = read_file(argv[2]);
        if (input.empty() || input.size() % 2560 != 0)
            throw std::runtime_error("Invalid packed rows");
        const auto n = static_cast<int64_t>(input.size() / 2560);
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        if (!backend)
            throw std::runtime_error("Backend creation failed");
        auto module = backend->create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error("Could not load lexical stage");
        auto output = module->forward({{"words", {input.data(), {n, 640}, trtmc::DType::kInt32}}});
        const auto& values = output.at("values");
        if (values.shape != std::vector<int64_t>{n, 2560} ||
            values.dtype != trtmc::DType::kFloat16 || values.nbytes() != input.size() * 2)
            throw std::runtime_error("Incorrect lexical output contract");
        std::ofstream out(argv[3], std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.write(static_cast<const char*>(values.data), values.nbytes());
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
