/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/runtime_kv/runtime_kv_plugin_api.h"

#include "plugins/runtime_kv/cudnn_attention.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#if TRTMC_RUNTIME_KV_CUDNN_SDPA
#include <cudnn.h>
#include <nvrtc.h>
#endif

#include <array>
#include <dlfcn.h>
#include <sstream>
#include <string>

#ifndef TRTMC_CUDNN_FRONTEND_REVISION
#define TRTMC_CUDNN_FRONTEND_REVISION "unavailable"
#endif

namespace {

std::string cuda_version_text(int version) {
    if (version <= 0)
        return "unavailable";
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

std::string cudnn_version_text(std::size_t version) {
    if (version == 0)
        return "unavailable";
    return std::to_string(version / 10000) + "." + std::to_string((version / 100) % 100) + "." +
           std::to_string(version % 100);
}

std::string current_driver_release() {
    using NvmlReturn = int;
    using NvmlInitFn = NvmlReturn (*)();
    using NvmlShutdownFn = NvmlReturn (*)();
    using NvmlDriverVersionFn = NvmlReturn (*)(char*, unsigned int);
    constexpr NvmlReturn kNvmlSuccess = 0;

    void* handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
        return "unavailable";
    auto init = reinterpret_cast<NvmlInitFn>(dlsym(handle, "nvmlInit_v2"));
    auto shutdown = reinterpret_cast<NvmlShutdownFn>(dlsym(handle, "nvmlShutdown"));
    auto version =
        reinterpret_cast<NvmlDriverVersionFn>(dlsym(handle, "nvmlSystemGetDriverVersion"));
    if (init == nullptr || shutdown == nullptr || version == nullptr) {
        dlclose(handle);
        return "unavailable";
    }

    std::array<char, 128> buffer{};
    std::string result = "unavailable";
    if (init() == kNvmlSuccess) {
        if (version(buffer.data(), static_cast<unsigned int>(buffer.size())) == kNvmlSuccess &&
            buffer.front() != '\0') {
            result = buffer.data();
        }
        (void)shutdown();
    }
    dlclose(handle);
    return result;
}

std::string current_runtime_stack_json() {
    int cuda_runtime = 0;
    const cudaError_t cuda_version_status = cudaRuntimeGetVersion(&cuda_runtime);

    int cuda_device = -1;
    cudaDeviceProp properties{};
    const cudaError_t device_status = cudaGetDevice(&cuda_device);
    const cudaError_t properties_status = device_status == cudaSuccess
                                              ? cudaGetDeviceProperties(&properties, cuda_device)
                                              : device_status;
    const std::string sm =
        properties_status == cudaSuccess
            ? "sm" + std::to_string(properties.major) + std::to_string(properties.minor)
            : "unavailable";

    std::string cudnn = "unavailable";
    std::string nvrtc = "unavailable";
#if TRTMC_RUNTIME_KV_CUDNN_SDPA
    cudnn = cudnn_version_text(cudnnGetVersion());
    int nvrtc_major = 0;
    int nvrtc_minor = 0;
    if (nvrtcVersion(&nvrtc_major, &nvrtc_minor) == NVRTC_SUCCESS) {
        nvrtc = std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor);
    }
#endif

    std::ostringstream json;
    json << "{\"sm\":\"" << sm << "\","
         << "\"tensorrt\":\"" << getInferLibMajorVersion() << "." << getInferLibMinorVersion()
         << "." << getInferLibPatchVersion() << "." << getInferLibBuildVersion() << "\","
         << "\"cuda_runtime\":\""
         << (cuda_version_status == cudaSuccess ? cuda_version_text(cuda_runtime)
                                                : std::string("unavailable"))
         << "\","
         << "\"cudnn_backend\":\"" << cudnn << "\","
         << "\"cudnn_frontend_revision\":\"" << TRTMC_CUDNN_FRONTEND_REVISION << "\","
         << "\"nvrtc\":\"" << nvrtc << "\","
         << "\"driver\":\"" << current_driver_release() << "\"}";
    return json.str();
}

} // namespace

extern "C" int32_t trtmc_runtime_kv_plugin_abi_version() noexcept {
    return trtmc::runtime_kv::kRuntimeKvPluginDsoAbi;
}

extern "C" std::uint64_t trtmc_runtime_kv_plugin_capabilities() noexcept {
    return trtmc::runtime_kv::native_cudnn_attention_available()
               ? trtmc::runtime_kv::kRuntimeKvCapabilityCudnnSdpa
               : 0;
}

extern "C" const char* trtmc_runtime_kv_plugin_runtime_stack_json_v1() noexcept {
    thread_local std::string value;
    try {
        value = current_runtime_stack_json();
    } catch (...) {
        value =
            R"({"sm":"unavailable","tensorrt":"unavailable","cuda_runtime":"unavailable","cudnn_backend":"unavailable","cudnn_frontend_revision":"unavailable","nvrtc":"unavailable","driver":"unavailable"})";
    }
    return value.c_str();
}

extern "C" void trtmc_runtime_kv_plugin_force_link() noexcept {}
