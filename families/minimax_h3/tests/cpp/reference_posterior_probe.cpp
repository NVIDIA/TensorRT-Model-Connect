/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Actual native backend/CPU RNG probe, not an ITrtModule or generator double.
#include "families/minimax_h3/runtime/torch_cuda_normal.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<char> read_file(const std::string& name) {
    const auto size = std::filesystem::file_size(name);
    if (size == 0 || size > 16 * 1024 * 1024)
        throw std::invalid_argument("posterior probe input must be nonempty and at most 16 MiB");
    std::vector<char> data(static_cast<std::size_t>(size));
    std::ifstream file(name, std::ios::binary);
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file)
        throw std::runtime_error("cannot read posterior probe input");
    return data;
}

void write_values(const std::string& name, const float* values, std::size_t count) {
    if (std::filesystem::exists(name))
        throw std::invalid_argument("posterior probe will not replace an existing result");
    std::ofstream file(name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (!file)
        throw std::runtime_error("cannot write posterior probe output");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: probe RUNTIME_ROOT PLAN MOMENTS OUTPUT_PREFIX PIXELS");
        std::size_t parsed = 0;
        const int64_t pixels = std::stoll(argv[5], &parsed);
        if (parsed != std::string(argv[5]).size() || pixels < 1 || pixels > 65536)
            throw std::invalid_argument("posterior probe latent pixels out of bounds");
        const std::string root = argv[1];
        const auto close_library = [](void* value) { (void)dlclose(value); };
        std::unique_ptr<void, decltype(close_library)> library(
            dlopen((root + "/libtrtmc_backend_trt.so").c_str(), RTLD_NOW | RTLD_LOCAL),
            close_library);
        if (!library)
            throw std::runtime_error(dlerror());
        using Create = trtmc::IBackend* (*)();
        using Destroy = void (*)(trtmc::IBackend*);
        const auto create = reinterpret_cast<Create>(dlsym(library.get(), "trtmc_create_backend"));
        const auto destroy =
            reinterpret_cast<Destroy>(dlsym(library.get(), "trtmc_destroy_backend"));
        if (!create || !destroy)
            throw std::runtime_error("native TensorRT backend exports are missing");
        std::unique_ptr<trtmc::IBackend, Destroy> backend(create(), destroy);
        if (!backend || std::string(backend->name()) != "trt")
            throw std::runtime_error("native TensorRT backend failed to initialize");
        auto plan = read_file(argv[2]);
        auto module = backend->create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error("native posterior plan failed to deserialize");
        std::vector<char>().swap(plan); // Backend owns the engine, not the serialized buffer.
        const auto bytes = read_file(argv[3]);
        if (bytes.size() != static_cast<std::size_t>(48 * pixels) * sizeof(float))
            throw std::invalid_argument("posterior moments payload shape mismatch");
        std::vector<float> moments(static_cast<std::size_t>(48 * pixels));
        std::copy(bytes.begin(), bytes.end(), reinterpret_cast<char*>(moments.data()));
        auto noise =
            trtmc::minimax_h3::torch_cuda_normal(static_cast<std::size_t>(24 * pixels), 42);
        write_values(std::string(argv[4]) + ".noise.bin", noise.data(), noise.size());
        module->reset_execution_context();
        for (int request = 0; request < 2; ++request) {
            trtmc::TensorMap inputs;
            inputs.emplace("moments",
                           trtmc::Tensor{moments.data(), {48, pixels}, trtmc::DType::kFloat32});
            inputs.emplace("noise",
                           trtmc::Tensor{noise.data(), {24, pixels}, trtmc::DType::kFloat32});
            const auto outputs = module->forward(inputs);
            module->sync();
            const auto output = outputs.at("sampled_latents");
            if (!output.data || output.dtype != trtmc::DType::kFloat32 ||
                output.shape != std::vector<int64_t>({24, pixels}))
                throw std::runtime_error("native posterior returned an incompatible tensor");
            const auto* values = static_cast<const float*>(output.data);
            if (!std::all_of(values, values + output.numel(),
                             [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("native posterior returned a nonfinite sample");
            write_values(std::string(argv[4]) + (request == 0 ? ".cold.bin" : ".warm.bin"), values,
                         output.numel());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
