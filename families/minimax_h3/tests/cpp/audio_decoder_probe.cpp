/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/audio_decoder.h"
#include "families/minimax_h3/runtime/plan_loader.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void write(const std::string& path, const std::vector<float>& data) {
    if (std::filesystem::exists(path))
        throw std::runtime_error("audio probe will not overwrite its outputs");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!file)
        throw std::runtime_error("audio probe could not write waveform");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "usage: probe RUNTIME_ROOT COMPONENT_BUNDLE ROWS OUTPUT_PREFIX staged|resident");
        const std::string root = argv[1], bundle = argv[2], input = argv[3], output = argv[4],
                          mode = argv[5];
        if (mode != "staged" && mode != "resident")
            throw std::invalid_argument("invalid audio residency");
        const auto close = [](void* handle) { (void)dlclose(handle); };
        std::unique_ptr<void, decltype(close)> library(
            dlopen((root + "/libtrtmc_backend_trt.so").c_str(), RTLD_NOW | RTLD_LOCAL), close);
        if (!library)
            throw std::runtime_error(dlerror());
        using Create = trtmc::IBackend* (*)();
        using Destroy = void (*)(trtmc::IBackend*);
        const auto create = reinterpret_cast<Create>(dlsym(library.get(), "trtmc_create_backend"));
        const auto destroy =
            reinterpret_cast<Destroy>(dlsym(library.get(), "trtmc_destroy_backend"));
        if (!create || !destroy)
            throw std::runtime_error("native audio backend exports missing");
        std::unique_ptr<trtmc::IBackend, Destroy> backend(create(), destroy);
        if (!backend || std::string(backend->name()) != "trt")
            throw std::runtime_error("native audio backend failed");
        const auto source =
            std::make_shared<trtmc::MiniMaxH3PlanSource>(trtmc::BundleReader(bundle));
        const auto file_loader =
            trtmc::make_minimax_h3_plan_loader(*backend, source, {"audio_vae.plan"});
        int loads = 0;
        trtmc::minimax_h3::ReferenceModuleLoader loader = [&](const std::string& name,
                                                              cudaStream_t stream) {
            auto module = file_loader(name, stream);
            ++loads;
            return module;
        };
        std::vector<float> rows(2 * 207 * 32);
        std::ifstream file(input, std::ios::binary);
        file.read(reinterpret_cast<char*>(rows.data()),
                  static_cast<std::streamsize>(rows.size() * sizeof(float)));
        if (!file || file.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("native audio probe needs full 207-frame FP32 rows");
        using namespace trtmc::minimax_h3;
        AudioDecoder owner(loader, 207, nullptr,
                           mode == "staged" ? AudioDecoderResidency::Staged
                                            : AudioDecoderResidency::Resident);
        const auto cold = owner.run(rows);
        write(output + ".cold.bin", cold.interleaved_samples);
        const auto warm = owner.run(rows);
        write(output + ".warm.bin", warm.interleaved_samples);
        const auto& a = cold.interleaved_samples;
        const auto& b = warm.interleaved_samples;
        if (a.size() != 2 * 165600 || b.size() != a.size() ||
            std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0 ||
            loads != (mode == "staged" ? 2 : 1) || cold.sample_rate != 32000 || cold.channels != 2)
            throw std::runtime_error("native audio lifecycle/shape/reuse contract failed");
        std::cout << "{\"loads\":" << loads
                  << ",\"sample_rate\":32000,\"channels\":2,\"samples_per_channel\":165600}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
