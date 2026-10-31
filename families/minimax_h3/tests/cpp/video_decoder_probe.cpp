/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/plan_loader.h"
#include "families/minimax_h3/runtime/video_decoder.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/resource.h>

namespace {
void write(const std::string& path, const std::vector<float>& values) {
    if (std::filesystem::exists(path))
        throw std::runtime_error("video probe will not overwrite outputs");
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out)
        throw std::runtime_error("video probe could not write output");
}
void compare_file(const std::string& path, const std::vector<float>& values) {
    std::ifstream in(path, std::ios::binary);
    std::array<float, 16384> chunk;
    for (std::size_t offset = 0; offset < values.size(); offset += chunk.size()) {
        const auto count = std::min(chunk.size(), values.size() - offset);
        in.read(reinterpret_cast<char*>(chunk.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
        if (!in || std::memcmp(chunk.data(), values.data() + offset, count * sizeof(float)))
            throw std::runtime_error("video probe cold/warm differs byte-for-byte");
    }
    if (in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("video probe cold file size mismatch");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7)
            throw std::invalid_argument("usage: probe RUNTIME_ROOT COMPONENT_BUNDLE ROWS "
                                        "CONFIG_JSON OUTPUT_PREFIX staged|resident");
        const std::string root = argv[1], bundle = argv[2], rows_path = argv[3],
                          config_path = argv[4], prefix = argv[5], mode = argv[6];
        if (mode != "staged" && mode != "resident")
            throw std::invalid_argument("invalid video residency");
        std::ifstream config_file(config_path);
        const auto json = nlohmann::json::parse(config_file);
        trtmc::minimax_h3::VideoDecoderConfig config;
        config.latent_frames = json.at("latent_frames").get<int32_t>();
        config.latent_height = json.at("latent_height").get<int32_t>();
        config.latent_width = json.at("latent_width").get<int32_t>();
        config.tile_batch = json.at("tile_batch").get<int32_t>();
        const auto mean = json.at("latents_mean").get<std::vector<float>>();
        const auto deviations = json.at("latents_std").get<std::vector<float>>();
        if (mean.size() != 24 || deviations.size() != 24 || config.latent_frames != 37 ||
            config.latent_height != 84 || config.latent_width != 48 ||
            (config.tile_batch != 1 && config.tile_batch != 28))
            throw std::runtime_error("video probe requires the declared full portrait profile");
        std::copy(mean.begin(), mean.end(), config.normalization.mean.begin());
        std::copy(deviations.begin(), deviations.end(), config.normalization.std.begin());
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
            throw std::runtime_error("video backend exports missing");
        std::unique_ptr<trtmc::IBackend, Destroy> backend(create(), destroy);
        if (!backend || std::string(backend->name()) != "trt")
            throw std::runtime_error("video backend initialization failed");
        const auto source =
            std::make_shared<trtmc::MiniMaxH3PlanSource>(trtmc::BundleReader(bundle));
        const auto file_loader = trtmc::make_minimax_h3_plan_loader(*backend, source, {"vae.plan"});
        uint64_t loads = 0;
        trtmc::minimax_h3::ReferenceModuleLoader loader = [&](const std::string& name,
                                                              cudaStream_t stream) {
            auto module = file_loader(name, stream);
            ++loads;
            return module;
        };
        std::vector<float> rows(24 * 37 * 84 * 48);
        std::ifstream input(rows_path, std::ios::binary);
        input.read(reinterpret_cast<char*>(rows.data()),
                   static_cast<std::streamsize>(rows.size() * sizeof(float)));
        if (!input || input.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("video probe input size mismatch");
        using namespace trtmc::minimax_h3;
        VideoDecoder owner(loader, config, nullptr,
                           mode == "staged" ? VideoDecoderResidency::Staged
                                            : VideoDecoderResidency::Resident);
        using Clock = std::chrono::steady_clock;
        uint64_t cold_forwards = 0, warm_forwards = 0;
        double cold_seconds = 0, warm_seconds = 0;
        for (bool warm : {false, true}) {
            const auto start = Clock::now();
            const auto result = owner.run(rows);
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            if (result.frames != 124 || result.height != 1344 || result.width != 768 ||
                result.rgb.size() != 124ULL * 1344 * 768 * 3 ||
                result.forwards != (config.tile_batch == 1 ? 196U : 7U))
                throw std::runtime_error("video probe shape/actual forward count mismatch");
            write(prefix + (warm ? ".warm.f32.bin" : ".cold.f32.bin"), result.rgb);
            if (warm) {
                compare_file(prefix + ".cold.f32.bin", result.rgb);
                warm_forwards = result.forwards;
                warm_seconds = seconds;
            } else {
                cold_forwards = result.forwards;
                cold_seconds = seconds;
            }
            // Destroy the previous owned output before the next request. Do
            // not inflate a full-frame memory probe with two retained videos.
        }
        if (loads != (mode == "staged" ? 2U : 1U))
            throw std::runtime_error("video probe residency mismatch");
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("video probe getrusage failed");
        std::cout << nlohmann::json{{"loads", loads},
                                    {"cold_forwards", cold_forwards},
                                    {"warm_forwards", warm_forwards},
                                    {"cold_seconds", cold_seconds},
                                    {"warm_seconds", warm_seconds},
                                    {"native_process_max_rss_kib", usage.ru_maxrss},
                                    {"frames", 124},
                                    {"height", 1344},
                                    {"width", 768},
                                    {"tile_batch", config.tile_batch}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
