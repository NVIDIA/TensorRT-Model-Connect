/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/native_library.h"

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc::hstu {
namespace {

struct LibraryFile {
    std::filesystem::path directory;
    ~LibraryFile() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

} // namespace

void validate_native_attention_manifest(const std::vector<char>& data, bool history_cache_enabled,
                                        int major, int minor) {
    const auto manifest = nlohmann::json::parse(data.begin(), data.end());
    // Original schema1 paged bundles have no mode field. Preserve their exact
    // creator/page contract while making the new uncached mode explicit.
    const auto mode = manifest.value("attention_mode", std::string("paged"));
    if (mode != (history_cache_enabled ? "paged" : "dense"))
        throw std::invalid_argument(
            "hstu native attention mode does not match history-cache configuration");
    const bool dense = mode == "dense";
    const auto digest = manifest.at("digest").get<std::string>();
    if (digest.size() != 64 ||
        !std::all_of(digest.begin(), digest.end(),
                     [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }) ||
        manifest.at("namespace") != "trtmc_hstu_" + digest ||
        manifest.at("creator") != (dense ? "HstuDenseAttention" : "HstuPagedAttention") ||
        manifest.at("creator_version") != "1" || manifest.at("dtype") != "bf16" ||
        manifest.at("heads") != 4 || manifest.at("head_dim") != 64 ||
        manifest.at("page_size") != (dense ? 0 : 128) || manifest.at("max_capacity") != 1024 ||
        manifest.at("scaling_seqlen") != 1024 || manifest.at("causal") != true ||
        manifest.at("target_group_size") != 1 || manifest.at("context") != false)
        throw std::invalid_argument("hstu unsupported native attention specialization");
    const auto& target = manifest.at("compute_capability");
    if (!target.is_array() || target.size() != 2 || !target[0].is_number_integer() ||
        !target[1].is_number_integer() || target[0] < 8 || target[0] > 99 || target[1] < 0 ||
        target[1] > 9)
        throw std::invalid_argument("hstu invalid native attention target");
    const auto target_major = target[0].get<int>();
    const auto target_minor = target[1].get<int>();
    if (manifest.at("schema_version") == 1) {
        // Preserve loading of previously exported Blackwell artifacts.
        if (manifest.at("adapter_abi") != 1 || manifest.at("gpu_arch") != "sm_103a" ||
            target_major != 10 || target_minor != 3)
            throw std::invalid_argument("hstu unsupported legacy attention specialization");
    } else if (manifest.at("schema_version") == 2) {
        const auto arch = "sm_" + std::to_string(target_major) + std::to_string(target_minor);
        if (!manifest.contains("attention_mode") || manifest.at("adapter_abi") != 2 ||
            manifest.at("provider") != "original_cuda_m64" || manifest.at("tile_m") != 64 ||
            manifest.at("tile_n") != 128 || manifest.at("warps") != 4 || target_major < 8 ||
            target_major > 99 || target_minor < 0 || target_minor > 9 ||
            manifest.at("gpu_arch") != arch)
            throw std::invalid_argument("hstu unsupported native attention provider contract");
    } else {
        throw std::invalid_argument("hstu unsupported native attention schema");
    }
    if (major != target_major || minor != target_minor)
        throw std::invalid_argument("hstu native attention bundle target differs from the serving "
                                    "GPU; rebuild for this GPU");
}

ModulePluginLibrary native_attention_library(const std::vector<char>& manifest,
                                             const std::vector<char>& library,
                                             bool history_cache_enabled) {
    if (library.empty())
        throw std::invalid_argument("hstu native attention library is empty");
    int device = -1;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("hstu could not inspect the serving GPU");
    validate_native_attention_manifest(manifest, history_cache_enabled, properties.major,
                                       properties.minor);
#ifdef _WIN32
    throw std::runtime_error("hstu original native attention currently requires Linux");
#else
    auto file = std::make_shared<LibraryFile>();
    const auto pattern = (std::filesystem::temp_directory_path() / "trtmc-hstu-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const auto* created = mkdtemp(writable.data());
    if (!created)
        throw std::runtime_error("hstu could not create a private native library directory");
    file->directory = created;
    const auto path = file->directory / "attention.so";
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.write(library.data(), static_cast<std::streamsize>(library.size()));
    stream.close();
    return {path.string(), std::move(file)};
#endif
}

} // namespace trtmc::hstu
