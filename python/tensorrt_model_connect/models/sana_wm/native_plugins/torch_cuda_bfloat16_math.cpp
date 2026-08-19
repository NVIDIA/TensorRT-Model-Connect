/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "torch_cuda_bfloat16_math.h"

#include <algorithm>
#include <cstring>
#include <exception>

#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/randn.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

namespace trtmc {

bool torch_cuda_bfloat16_randn(int32_t channels, int32_t frames, int32_t height, int32_t width,
                               uint64_t seed, float* output, std::string& error) {
    error.clear();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (output == nullptr || channels <= 0 || frames <= 0 || height <= 0 || width <= 0) {
        error = "invalid BF16 randn inputs";
        return false;
    }

    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        return false;

    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess) {
            error = "cudaGetDevice failed for BF16 randn";
            return false;
        }
        c10::cuda::CUDAGuard device_guard(device_index);
        auto generator = at::cuda::detail::createCUDAGenerator(device_index);
        generator.set_current_seed(seed);
        const auto options =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);
        auto noise = at::randn({1, channels, frames, height, width}, generator, options);
        auto host = noise.to(at::kFloat).to(at::kCPU).contiguous();
        const auto count = static_cast<std::size_t>(channels) * static_cast<std::size_t>(frames) *
                           static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        std::memcpy(output, host.data_ptr<float>(), count * sizeof(float));
        return true;
    } catch (const c10::Error& exc) {
        error = exc.what();
        return false;
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
#else
    (void)channels;
    (void)frames;
    (void)height;
    (void)width;
    (void)seed;
    (void)output;
    return false;
#endif
}

bool torch_cuda_bfloat16_sana_ucpe_raymats(const float* camera_conditions,
                                           std::size_t camera_condition_count, int32_t frames,
                                           int32_t height, int32_t width, float* raymats,
                                           float* raymats_inv, std::string& error) {
    error.clear();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    constexpr int32_t kCameraConditionWidth = 20;
    const auto expected_camera_count = static_cast<std::size_t>(frames) * kCameraConditionWidth;
    if (camera_conditions == nullptr || raymats == nullptr || raymats_inv == nullptr ||
        frames <= 0 || height <= 0 || width <= 0 ||
        camera_condition_count != expected_camera_count) {
        error = "invalid SANA-WM UCPE inputs";
        return false;
    }

    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        return false;

    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess) {
            error = "cudaGetDevice failed for SANA-WM UCPE";
            return false;
        }
        c10::cuda::CUDAGuard device_guard(device_index);
        const auto cpu_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const auto cuda_bf16 =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);

        auto camera_cpu = at::from_blob(const_cast<float*>(camera_conditions),
                                        {1, frames, kCameraConditionWidth}, cpu_f32);
        auto camera = camera_cpu.to(cuda_bf16);
        auto c2w = camera.slice(-1, 0, 16).reshape({1, frames, 4, 4});
        auto fx = camera.select(-1, 16).reshape({frames});
        auto fy = camera.select(-1, 17).reshape({frames});
        auto cx = camera.select(-1, 18).reshape({frames});
        auto cy = camera.select(-1, 19).reshape({frames});
        auto xi = at::zeros({frames}, cuda_bf16);

        const auto fov_from_fx = [&xi](const at::Tensor& focal, int32_t extent) {
            auto a = at::div(at::mul(focal, 2.0), static_cast<double>(extent));
            auto phi = at::atan(at::reciprocal(a));
            auto denom = at::sqrt(at::add(at::mul(a, a), 1.0));
            auto ratio = at::clamp(at::div(xi, denom), -1.0, 1.0);
            auto theta = at::add(at::asin(ratio), phi);
            return at::rad2deg(at::mul(theta, 2.0));
        };
        const auto fx_from_fov = [&xi](const at::Tensor& fov, int32_t extent) {
            auto theta = at::deg2rad(at::mul(fov, 0.5));
            auto denom = at::clamp_min(at::sin(theta), 0.0078125);
            auto numerator = at::mul(at::add(at::cos(theta), xi), extent * 0.5);
            return at::div(numerator, denom);
        };

        auto x_fov = fov_from_fx(fx, width);
        auto y_fov = fov_from_fx(fy, height);
        auto projected_fx = fx_from_fov(x_fov, width).reshape({frames, 1, 1});
        auto projected_fy = fx_from_fov(y_fov, height).reshape({frames, 1, 1});
        auto projected_cx = at::div(cx, 1.0).reshape({frames, 1, 1});
        auto projected_cy = at::div(cy, 1.0).reshape({frames, 1, 1});
        auto projected_xi = xi.reshape({frames, 1, 1});

        auto x_line = at::linspace(0, width - 1, width, cuda_bf16);
        auto y_line = at::linspace(0, height - 1, height, cuda_bf16);
        auto xs = x_line.reshape({1, width}).expand({height, width});
        auto ys = y_line.reshape({height, 1}).expand({height, width});
        auto grid =
            at::stack({xs, ys, at::ones_like(xs)}, -1).unsqueeze(0).repeat({frames, 1, 1, 1});
        auto u = grid.select(-1, 0);
        auto v = grid.select(-1, 1);
        auto x = at::div(at::sub(u, projected_cx), projected_fx);
        auto y = at::div(at::sub(v, projected_cy), projected_fy);
        auto r2 = at::add(at::mul(x, x), at::mul(y, y));
        auto one_minus_xi_squared = at::rsub(at::mul(projected_xi, projected_xi), 1.0);
        auto alpha =
            at::add(projected_xi, at::sqrt(at::add(at::mul(one_minus_xi_squared, r2), 1.0)));
        auto gamma = at::div(alpha, at::add(r2, 1.0));
        std::vector<at::Tensor> d_cam_components = {at::mul(gamma, x), at::mul(gamma, y),
                                                    at::sub(gamma, projected_xi)};
        auto d_cam = at::stack(d_cam_components, -1).reshape({1, frames, height, width, 3});

        auto r_cam = c2w.slice(-2, 0, 3).slice(-1, 0, 3);
        auto t_cam = c2w.slice(-2, 0, 3).select(-1, 3);
        auto d_world = at::einsum("btij,bthwj->bthwi", std::vector<at::Tensor>{r_cam, d_cam});
        auto cam_y =
            r_cam.select(-1, 1).reshape({1, frames, 1, 1, 3}).expand({1, frames, height, width, 3});
        const auto normalize = [](const at::Tensor& value) {
            auto denom = value.norm(2.0, {-1}, true).clamp_min(1.0e-6);
            return at::div(value, denom);
        };
        auto z_ray = normalize(d_world);
        auto x_ray = normalize(at::cross(cam_y, z_ray, -1));
        auto y_ray = normalize(at::cross(z_ray, x_ray, -1));
        auto r_l2w = at::stack(std::vector<at::Tensor>{x_ray, y_ray, z_ray}, -1);
        auto r_w2l = r_l2w.transpose(-2, -1);
        auto t_world = t_cam.reshape({1, frames, 1, 1, 3}).expand({1, frames, height, width, 3});
        auto t_w2l =
            at::neg(at::einsum("bthwij,bthwj->bthwi", std::vector<at::Tensor>{r_w2l, t_world}));

        auto matrices = at::zeros({1, frames, height, width, 4, 4}, cuda_bf16);
        matrices.slice(-2, 0, 3).slice(-1, 0, 3).copy_(r_w2l);
        matrices.slice(-2, 0, 3).select(-1, 3).copy_(t_w2l);
        matrices.select(-2, 3).select(-1, 3).fill_(1.0);

        auto inverse = at::linalg_inv(matrices.to(at::kFloat)).to(at::kBFloat16);

        const auto matrix_count = static_cast<std::size_t>(frames) *
                                  static_cast<std::size_t>(height) *
                                  static_cast<std::size_t>(width) * 16U;
        auto matrices_host = matrices.to(at::kFloat).to(at::kCPU).contiguous();
        auto inverse_host = inverse.to(at::kFloat).to(at::kCPU).contiguous();
        std::memcpy(raymats, matrices_host.data_ptr<float>(), matrix_count * sizeof(float));
        std::memcpy(raymats_inv, inverse_host.data_ptr<float>(), matrix_count * sizeof(float));
        return true;
    } catch (const c10::Error& exc) {
        error = exc.what();
        return false;
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
#else
    (void)camera_conditions;
    (void)camera_condition_count;
    (void)frames;
    (void)height;
    (void)width;
    (void)raymats;
    (void)raymats_inv;
    return false;
#endif
}

bool torch_float32_sana_chunk_plucker(const float* poses, std::size_t pose_count,
                                      const float* intrinsics, std::size_t intrinsics_count,
                                      int32_t num_frames, int32_t chunk_count, int32_t height,
                                      int32_t width, int32_t time_stride, float* output,
                                      std::string& error) {
    error.clear();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (poses == nullptr || intrinsics == nullptr || output == nullptr || num_frames <= 0 ||
        chunk_count <= 0 || height <= 0 || width <= 0 || time_stride <= 0 ||
        pose_count != static_cast<std::size_t>(num_frames) * 16U ||
        intrinsics_count != static_cast<std::size_t>(num_frames) * 4U) {
        error = "invalid SANA-WM chunk Plucker inputs";
        return false;
    }

    try {
        const auto cpu_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        auto pose_tensor = at::from_blob(const_cast<float*>(poses), {num_frames, 4, 4}, cpu_f32);
        auto intrinsics_tensor =
            at::from_blob(const_cast<float*>(intrinsics), {num_frames, 4}, cpu_f32);
        auto x_grid = at::arange(width, cpu_f32).reshape({1, width}).expand({height, width});
        auto y_grid = at::arange(height, cpu_f32).reshape({height, 1}).expand({height, width});

        std::vector<at::Tensor> chunks;
        chunks.reserve(static_cast<std::size_t>(chunk_count));
        for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int32_t time_index = chunk * time_stride;
            const int32_t start = std::max(0, time_index - (time_stride - 1));
            const int32_t end = std::min(num_frames, start + time_stride);
            auto chunk_poses = pose_tensor.slice(0, start, end);
            auto chunk_intrinsics = intrinsics_tensor.slice(0, start, end);
            const auto actual_frames = static_cast<int32_t>(chunk_poses.size(0));
            if (actual_frames < time_stride) {
                const int32_t pad = time_stride - actual_frames;
                chunk_poses = at::cat(
                    std::vector<at::Tensor>{
                        chunk_poses,
                        chunk_poses.slice(0, actual_frames - 1, actual_frames).repeat({pad, 1, 1})},
                    0);
                chunk_intrinsics =
                    at::cat(std::vector<at::Tensor>{chunk_intrinsics,
                                                    chunk_intrinsics
                                                        .slice(0, actual_frames - 1, actual_frames)
                                                        .repeat({pad, 1})},
                            0);
            }

            auto fx = chunk_intrinsics.select(1, 0).reshape({time_stride, 1, 1});
            auto fy = chunk_intrinsics.select(1, 1).reshape({time_stride, 1, 1});
            auto cx = chunk_intrinsics.select(1, 2).reshape({time_stride, 1, 1});
            auto cy = chunk_intrinsics.select(1, 3).reshape({time_stride, 1, 1});
            auto expanded_x = x_grid.unsqueeze(0).expand({time_stride, height, width});
            auto expanded_y = y_grid.unsqueeze(0).expand({time_stride, height, width});
            auto x_cam = at::div(at::sub(expanded_x, cx), fx);
            auto y_cam = at::div(at::sub(expanded_y, cy), fy);
            auto z_cam = at::ones_like(x_cam);
            auto dirs_cam = at::stack(std::vector<at::Tensor>{x_cam, y_cam, z_cam}, -1);

            auto rotation = chunk_poses.slice(1, 0, 3).slice(2, 0, 3);
            auto translation = chunk_poses.slice(1, 0, 3).select(2, 3);
            auto dirs_world =
                at::einsum("tij,thwj->thwi", std::vector<at::Tensor>{rotation, dirs_cam});
            dirs_world = at::div(dirs_world, dirs_world.norm(2.0, {-1}, true));
            auto origins = translation.reshape({time_stride, 1, 1, 3}).expand_as(dirs_world);
            auto moments = at::cross(origins, dirs_world, -1);
            auto plucker = at::cat(std::vector<at::Tensor>{dirs_world, moments}, -1);
            chunks.push_back(
                plucker.permute({0, 3, 1, 2}).reshape({time_stride * 6, height, width}));
        }

        auto packed = at::stack(chunks, 0).permute({1, 0, 2, 3}).contiguous();
        const auto output_count =
            static_cast<std::size_t>(time_stride) * 6U * static_cast<std::size_t>(chunk_count) *
            static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        std::memcpy(output, packed.data_ptr<float>(), output_count * sizeof(float));
        return true;
    } catch (const c10::Error& exc) {
        error = exc.what();
        return false;
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
#else
    (void)poses;
    (void)pose_count;
    (void)intrinsics;
    (void)intrinsics_count;
    (void)num_frames;
    (void)chunk_count;
    (void)height;
    (void)width;
    (void)time_stride;
    (void)output;
    return false;
#endif
}

bool torch_cuda_bfloat16_ltx_flow_step(const float* model_output, std::size_t model_output_count,
                                       const float* sample, std::size_t sample_count,
                                       int32_t channels, int32_t frames, int32_t height,
                                       int32_t width, float timestep, float cfg_scale,
                                       const std::vector<float>& sigmas, float* output,
                                       std::string& error) {
    error.clear();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (model_output == nullptr || sample == nullptr || output == nullptr || channels <= 0 ||
        frames <= 0 || height <= 0 || width <= 0 || sigmas.size() < 2U) {
        error = "invalid LTX flow-step inputs";
        return false;
    }
    const auto token_count = static_cast<std::size_t>(frames) * static_cast<std::size_t>(height) *
                             static_cast<std::size_t>(width);
    const auto expected_count = static_cast<std::size_t>(channels) * token_count;
    const bool do_cfg = cfg_scale > 1.0F;
    const auto expected_model_count = expected_count * (do_cfg ? 2U : 1U);
    if (sample_count != expected_count || model_output_count != expected_model_count) {
        error = "LTX flow-step tensor size mismatch";
        return false;
    }

    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        return false;

    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess) {
            error = "cudaGetDevice failed for LTX flow step";
            return false;
        }
        c10::cuda::CUDAGuard device_guard(device_index);
        const auto cpu_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const auto cuda_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        const auto cuda_bf16 =
            at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, device_index);

        auto sample_cpu = at::from_blob(const_cast<float*>(sample),
                                        {1, channels, frames, height, width}, cpu_f32);
        auto sample_bf16 = sample_cpu.to(cuda_f32).to(at::kBFloat16);
        auto raw_cpu = at::from_blob(const_cast<float*>(model_output),
                                     {do_cfg ? 2 : 1, channels, frames, height, width}, cpu_f32);
        auto raw = raw_cpu.to(cuda_f32).to(at::kBFloat16);

        auto noise_pred = raw.select(0, 0);
        if (do_cfg) {
            auto delta = at::sub(raw.select(0, 1), noise_pred);
            auto scaled = at::mul(delta, static_cast<double>(cfg_scale));
            noise_pred = at::add(noise_pred, scaled);
        }
        auto scheduler_model_output = at::neg(noise_pred)
                                          .reshape({1, channels, static_cast<int64_t>(token_count)})
                                          .transpose(1, 2);
        auto scheduler_sample =
            sample_bf16.reshape({1, channels, static_cast<int64_t>(token_count)}).transpose(1, 2);

        auto condition_mask = at::zeros({1, channels, frames, height, width}, cuda_bf16);
        condition_mask.select(2, 0).fill_(1.0);
        auto inverse_condition_mask = at::sub(at::ones_like(condition_mask), condition_mask);
        auto timestep_tensor = at::scalar_tensor(timestep, cuda_f32);
        auto expanded_timestep = timestep_tensor.expand({1, channels, frames, height, width});
        auto capped_timestep =
            at::minimum(expanded_timestep, at::mul(inverse_condition_mask, 1000.0));
        auto per_token_timesteps =
            capped_timestep.reshape({1, channels, static_cast<int64_t>(token_count)}).select(1, 0);
        auto per_token_sigmas = at::div(per_token_timesteps, 1000.0);

        auto sigmas_cpu = at::from_blob(const_cast<float*>(sigmas.data()),
                                        {static_cast<int64_t>(sigmas.size())}, cpu_f32);
        auto sigmas_cuda = sigmas_cpu.to(cuda_f32);
        auto sigma_column = sigmas_cuda.reshape({static_cast<int64_t>(sigmas.size()), 1, 1});
        auto lower_mask = at::lt(sigma_column, at::sub(per_token_sigmas.unsqueeze(0), 1.0e-6));
        auto lower_sigmas = at::amax(at::mul(lower_mask, sigma_column), {0});
        auto dt = at::sub(per_token_sigmas.unsqueeze(-1), lower_sigmas.unsqueeze(-1));
        auto previous_sample =
            at::add(scheduler_sample.to(at::kFloat), at::mul(dt, scheduler_model_output));
        previous_sample =
            previous_sample.transpose(1, 2).reshape({1, channels, frames, height, width});

        auto update_mask =
            at::lt(at::sub(at::div(timestep_tensor, 1000.0), 1.0e-6), inverse_condition_mask);
        auto next = at::where(update_mask, previous_sample, sample_bf16).to(at::kBFloat16);
        auto host = next.to(at::kFloat).to(at::kCPU).contiguous();
        std::memcpy(output, host.data_ptr<float>(), sample_count * sizeof(float));
        return true;
    } catch (const c10::Error& exc) {
        error = exc.what();
        return false;
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
#else
    (void)model_output;
    (void)model_output_count;
    (void)sample;
    (void)sample_count;
    (void)channels;
    (void)frames;
    (void)height;
    (void)width;
    (void)timestep;
    (void)cfg_scale;
    (void)sigmas;
    (void)output;
    return false;
#endif
}

bool torch_cuda_bfloat16_refiner_mix(const float* clean, const float* noise, std::size_t count,
                                     float sigma, float* output, std::string& error) {
    error.clear();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (clean == nullptr || noise == nullptr || output == nullptr || count == 0U) {
        error = "invalid refiner blend inputs";
        return false;
    }

    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        return false;

    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess) {
            error = "cudaGetDevice failed for refiner blend";
            return false;
        }
        c10::cuda::CUDAGuard device_guard(device_index);
        const auto cpu_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const auto cuda_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        auto clean_cpu =
            at::from_blob(const_cast<float*>(clean), {static_cast<int64_t>(count)}, cpu_f32);
        auto noise_cpu =
            at::from_blob(const_cast<float*>(noise), {static_cast<int64_t>(count)}, cpu_f32);
        auto clean_bf16 = clean_cpu.to(cuda_f32).to(at::kBFloat16);
        auto noise_bf16 = noise_cpu.to(cuda_f32).to(at::kBFloat16);
        auto mixed = at::add(at::mul(clean_bf16, 1.0 - static_cast<double>(sigma)),
                             at::mul(noise_bf16, static_cast<double>(sigma)));
        auto host = mixed.to(at::kFloat).to(at::kCPU).contiguous();
        std::memcpy(output, host.data_ptr<float>(), count * sizeof(float));
        return true;
    } catch (const c10::Error& exc) {
        error = exc.what();
        return false;
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
#else
    (void)clean;
    (void)noise;
    (void)count;
    (void)sigma;
    (void)output;
    return false;
#endif
}

bool torch_cuda_bfloat16_refiner_euler_step(const float* sample, const float* denoised,
                                            std::size_t count, float sigma, float sigma_next,
                                            float* output, std::string& error) {
    error.clear();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (sample == nullptr || denoised == nullptr || output == nullptr || count == 0U ||
        sigma <= 0.0F) {
        error = "invalid refiner Euler inputs";
        return false;
    }

    int32_t device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        return false;

    try {
        int32_t device_index = 0;
        if (cudaGetDevice(&device_index) != cudaSuccess) {
            error = "cudaGetDevice failed for refiner Euler step";
            return false;
        }
        c10::cuda::CUDAGuard device_guard(device_index);
        const auto cpu_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const auto cuda_f32 = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device_index);
        auto sample_cpu =
            at::from_blob(const_cast<float*>(sample), {static_cast<int64_t>(count)}, cpu_f32);
        auto denoised_cpu =
            at::from_blob(const_cast<float*>(denoised), {static_cast<int64_t>(count)}, cpu_f32);
        auto sample_f32 = sample_cpu.to(cuda_f32).to(at::kBFloat16).to(at::kFloat);
        auto denoised_f32 = denoised_cpu.to(cuda_f32);
        auto sigma_tensor = at::scalar_tensor(sigma, cuda_f32);
        auto sigma_next_tensor = at::scalar_tensor(sigma_next, cuda_f32);
        auto velocity = at::div(at::sub(sample_f32, denoised_f32), sigma_tensor);
        auto next = at::add(sample_f32, at::mul(velocity, at::sub(sigma_next_tensor, sigma_tensor)))
                        .to(at::kBFloat16);
        auto host = next.to(at::kFloat).to(at::kCPU).contiguous();
        std::memcpy(output, host.data_ptr<float>(), count * sizeof(float));
        return true;
    } catch (const c10::Error& exc) {
        error = exc.what();
        return false;
    } catch (const std::exception& exc) {
        error = exc.what();
        return false;
    }
#else
    (void)sample;
    (void)denoised;
    (void)count;
    (void)sigma;
    (void)sigma_next;
    (void)output;
    return false;
#endif
}

} // namespace trtmc

extern "C" bool trtmc_sana_wm_bfloat16_randn(int32_t channels, int32_t frames, int32_t height,
                                             int32_t width, uint64_t seed, float* output,
                                             std::string& error) {
    return trtmc::torch_cuda_bfloat16_randn(channels, frames, height, width, seed, output, error);
}

extern "C" bool trtmc_sana_wm_bfloat16_ucpe_raymats(const float* camera_conditions,
                                                    std::size_t camera_condition_count,
                                                    int32_t frames, int32_t height, int32_t width,
                                                    float* raymats, float* raymats_inv,
                                                    std::string& error) {
    return trtmc::torch_cuda_bfloat16_sana_ucpe_raymats(camera_conditions, camera_condition_count,
                                                        frames, height, width, raymats, raymats_inv,
                                                        error);
}

extern "C" bool trtmc_sana_wm_float32_chunk_plucker(
    const float* poses, std::size_t pose_count, const float* intrinsics,
    std::size_t intrinsics_count, int32_t num_frames, int32_t chunk_count, int32_t height,
    int32_t width, int32_t time_stride, float* output, std::string& error) {
    return trtmc::torch_float32_sana_chunk_plucker(poses, pose_count, intrinsics, intrinsics_count,
                                                   num_frames, chunk_count, height, width,
                                                   time_stride, output, error);
}

extern "C" bool trtmc_sana_wm_bfloat16_ltx_flow_step(
    const float* model_output, std::size_t model_output_count, const float* sample,
    std::size_t sample_count, int32_t channels, int32_t frames, int32_t height, int32_t width,
    float timestep, float cfg_scale, const std::vector<float>& sigmas, float* output,
    std::string& error) {
    return trtmc::torch_cuda_bfloat16_ltx_flow_step(model_output, model_output_count, sample,
                                                    sample_count, channels, frames, height, width,
                                                    timestep, cfg_scale, sigmas, output, error);
}

extern "C" bool trtmc_sana_wm_bfloat16_refiner_mix(const float* clean, const float* noise,
                                                   std::size_t count, float sigma, float* output,
                                                   std::string& error) {
    return trtmc::torch_cuda_bfloat16_refiner_mix(clean, noise, count, sigma, output, error);
}

extern "C" bool trtmc_sana_wm_bfloat16_refiner_euler_step(const float* sample,
                                                          const float* denoised, std::size_t count,
                                                          float sigma, float sigma_next,
                                                          float* output, std::string& error) {
    return trtmc::torch_cuda_bfloat16_refiner_euler_step(sample, denoised, count, sigma, sigma_next,
                                                         output, error);
}
