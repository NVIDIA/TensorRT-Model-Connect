/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sana_wm/native_ops.h"

#include <atomic>
#include <dlfcn.h>

namespace trtmc {
namespace {

std::atomic<bool> g_native_ops_required{false};

template <typename Fn>
Fn resolve_native_op(const char* symbol, std::string& error) {
    dlerror();
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, symbol));
    if (fn == nullptr) {
        const char* message = dlerror();
        if (g_native_ops_required.load(std::memory_order_acquire))
            error = message != nullptr ? message : std::string("missing native op: ") + symbol;
        else
            error.clear();
    }
    return fn;
}

} // namespace

void require_sana_wm_native_ops() {
    g_native_ops_required.store(true, std::memory_order_release);
}

bool torch_cuda_bfloat16_randn(int32_t channels, int32_t frames, int32_t height, int32_t width,
                               uint64_t seed, float* output, std::string& error) {
    using Fn = bool (*)(int32_t, int32_t, int32_t, int32_t, uint64_t, float*, std::string&);
    auto fn = resolve_native_op<Fn>("trtmc_sana_wm_bfloat16_randn", error);
    return fn != nullptr && fn(channels, frames, height, width, seed, output, error);
}

bool torch_cuda_bfloat16_sana_ucpe_raymats(const float* camera_conditions,
                                           std::size_t camera_condition_count, int32_t frames,
                                           int32_t height, int32_t width, float* raymats,
                                           float* raymats_inv, std::string& error) {
    using Fn = bool (*)(const float*, std::size_t, int32_t, int32_t, int32_t, float*, float*,
                        std::string&);
    auto fn = resolve_native_op<Fn>("trtmc_sana_wm_bfloat16_ucpe_raymats", error);
    return fn != nullptr && fn(camera_conditions, camera_condition_count, frames, height, width,
                               raymats, raymats_inv, error);
}

bool torch_float32_sana_chunk_plucker(const float* poses, std::size_t pose_count,
                                      const float* intrinsics, std::size_t intrinsics_count,
                                      int32_t num_frames, int32_t chunk_count, int32_t height,
                                      int32_t width, int32_t time_stride, float* output,
                                      std::string& error) {
    using Fn = bool (*)(const float*, std::size_t, const float*, std::size_t, int32_t, int32_t,
                        int32_t, int32_t, int32_t, float*, std::string&);
    auto fn = resolve_native_op<Fn>("trtmc_sana_wm_float32_chunk_plucker", error);
    return fn != nullptr && fn(poses, pose_count, intrinsics, intrinsics_count, num_frames,
                               chunk_count, height, width, time_stride, output, error);
}

bool torch_cuda_bfloat16_ltx_flow_step(const float* model_output, std::size_t model_output_count,
                                       const float* sample, std::size_t sample_count,
                                       int32_t channels, int32_t frames, int32_t height,
                                       int32_t width, float timestep, float cfg_scale,
                                       const std::vector<float>& sigmas, float* output,
                                       std::string& error) {
    using Fn =
        bool (*)(const float*, std::size_t, const float*, std::size_t, int32_t, int32_t, int32_t,
                 int32_t, float, float, const std::vector<float>&, float*, std::string&);
    auto fn = resolve_native_op<Fn>("trtmc_sana_wm_bfloat16_ltx_flow_step", error);
    return fn != nullptr && fn(model_output, model_output_count, sample, sample_count, channels,
                               frames, height, width, timestep, cfg_scale, sigmas, output, error);
}

bool torch_cuda_bfloat16_refiner_mix(const float* clean, const float* noise, std::size_t count,
                                     float sigma, float* output, std::string& error) {
    using Fn = bool (*)(const float*, const float*, std::size_t, float, float*, std::string&);
    auto fn = resolve_native_op<Fn>("trtmc_sana_wm_bfloat16_refiner_mix", error);
    return fn != nullptr && fn(clean, noise, count, sigma, output, error);
}

bool torch_cuda_bfloat16_refiner_euler_step(const float* sample, const float* denoised,
                                            std::size_t count, float sigma, float sigma_next,
                                            float* output, std::string& error) {
    using Fn =
        bool (*)(const float*, const float*, std::size_t, float, float, float*, std::string&);
    auto fn = resolve_native_op<Fn>("trtmc_sana_wm_bfloat16_refiner_euler_step", error);
    return fn != nullptr && fn(sample, denoised, count, sigma, sigma_next, output, error);
}

} // namespace trtmc
