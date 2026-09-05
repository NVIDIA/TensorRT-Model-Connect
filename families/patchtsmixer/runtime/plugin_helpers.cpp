/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtsmixer/runtime/plugin_helpers.h"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace trtmc::patchtsmixer {
namespace {

std::int32_t require_env_int(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0')
        throw std::runtime_error(std::string("PatchTSMixer TP runtime requires ") + name);

    std::int32_t value = 0;
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end)
        throw std::runtime_error(std::string("PatchTSMixer TP runtime has invalid ") + name);
    return value;
}

} // namespace

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, const char* name) {
    const auto& data = require_section(bundle, name);
    return std::string(data.begin(), data.end());
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error("PatchTSMixer TensorRT engine failed to load");
    return module;
}

std::int32_t require_rank(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size == 1)
        return 0;

    const auto world_size = require_env_int("OMPI_COMM_WORLD_SIZE");
    const auto rank = require_env_int("OMPI_COMM_WORLD_RANK");
    const auto local_rank = require_env_int("OMPI_COMM_WORLD_LOCAL_RANK");
    if (world_size != tensor_parallel_size) {
        throw std::runtime_error(
            "PatchTSMixer OMPI_COMM_WORLD_SIZE must equal tensor_parallel_size");
    }
    if (rank < 0 || rank >= world_size)
        throw std::runtime_error("PatchTSMixer OMPI_COMM_WORLD_RANK is outside the world");
    if (local_rank < 0) {
        throw std::runtime_error("PatchTSMixer OMPI_COMM_WORLD_LOCAL_RANK must be non-negative");
    }

    int device_count = 0;
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        throw std::runtime_error(std::string("PatchTSMixer cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(count_status));
    }
    if (local_rank >= device_count) {
        throw std::runtime_error(
            "PatchTSMixer OMPI_COMM_WORLD_LOCAL_RANK has no visible CUDA device");
    }
    const auto device_status = cudaSetDevice(local_rank);
    if (device_status != cudaSuccess) {
        throw std::runtime_error(std::string("PatchTSMixer cudaSetDevice failed: ") +
                                 cudaGetErrorString(device_status));
    }
    return rank;
}

} // namespace trtmc::patchtsmixer
