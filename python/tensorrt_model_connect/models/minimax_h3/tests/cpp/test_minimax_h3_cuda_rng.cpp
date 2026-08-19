/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"
#include "torch_cuda_normal.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint32_t bits(float value) {
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

uint64_t update_fnv1a(uint64_t hash, const std::vector<float>& values) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    for (std::size_t index = 0; index < values.size() * sizeof(float); ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

void test_scheduler_matches_cpu() {
    // Exceeds the production launch cap so the grid-stride path is covered.
    constexpr std::size_t count = 1100003;
    constexpr float timestep = 0.421875F;
    constexpr float sigma = 0.578125F;
    constexpr float sigma_next = 0.53125F;
    std::vector<float> sample(count);
    std::vector<float> velocity(count);
    for (std::size_t index = 0; index < count; ++index) {
        sample[index] = static_cast<float>(static_cast<int32_t>(index % 211U) - 105) / 19.0F;
        velocity[index] =
            static_cast<float>(static_cast<int32_t>((index * 37U) % 307U) - 153) / 23.0F;
    }
    std::vector<float> expected = sample;
    trtmc::minimax_h3_scheduler_step(expected.data(), velocity.data(), count, timestep, sigma,
                                     sigma_next);

    cudaStream_t stream = nullptr;
    float* device_sample = nullptr;
    float* device_velocity = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_sample), count * sizeof(float)),
                   "sample cudaMalloc");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_velocity), count * sizeof(float)),
                   "velocity cudaMalloc");
        check_cuda(cudaMemcpyAsync(device_sample, sample.data(), count * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "sample cudaMemcpyAsync");
        check_cuda(cudaMemcpyAsync(device_velocity, velocity.data(), count * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "velocity cudaMemcpyAsync");
        trtmc::minimax_h3::scheduler_step_cuda_async(device_sample, device_velocity, count,
                                                     timestep, sigma, sigma_next, stream);
        check_cuda(cudaMemcpyAsync(sample.data(), device_sample, count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "result cudaMemcpyAsync");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    } catch (...) {
        cudaFree(device_velocity);
        cudaFree(device_sample);
        cudaStreamDestroy(stream);
        throw;
    }
    check_cuda(cudaFree(device_velocity), "velocity cudaFree");
    check_cuda(cudaFree(device_sample), "sample cudaFree");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    for (std::size_t index = 0; index < count; ++index) {
        const float tolerance = 2.0e-6F + 2.0e-6F * std::abs(expected[index]);
        if (!std::isfinite(sample[index]) ||
            std::abs(sample[index] - expected[index]) > tolerance) {
            throw std::runtime_error("CUDA scheduler differs from CPU at index " +
                                     std::to_string(index));
        }
    }
}

void test_scheduler_rejects_invalid_inputs() {
    float* const dummy = reinterpret_cast<float*>(static_cast<uintptr_t>(1));
    const auto rejects_invalid = [](auto&& operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    if (!rejects_invalid([&] {
            trtmc::minimax_h3::scheduler_step_cuda_async(nullptr, dummy, 1, 0.25F, 0.75F, 0.5F,
                                                         nullptr);
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::scheduler_step_cuda_async(dummy, nullptr, 1, 0.25F, 0.75F, 0.5F,
                                                         nullptr);
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::scheduler_step_cuda_async(dummy, dummy, 1, 0.25F, 0.0F, 0.5F,
                                                         nullptr);
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::scheduler_step_cuda_async(dummy, dummy, 1, 0.25F, -0.75F, 0.5F,
                                                         nullptr);
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::scheduler_step_cuda_async(
                dummy, dummy, 1, 0.25F, std::numeric_limits<float>::quiet_NaN(), 0.5F, nullptr);
        })) {
        throw std::runtime_error("CUDA scheduler accepted invalid inputs");
    }

    trtmc::minimax_h3::scheduler_step_cuda_async(dummy, dummy, 0, 0.25F, 0.75F, 0.5F, nullptr);

    if constexpr (std::numeric_limits<std::size_t>::max() >
                  static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
        try {
            trtmc::minimax_h3::scheduler_step_cuda_async(
                dummy, dummy, static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) + 1U,
                0.25F, 0.75F, 0.5F, nullptr);
        } catch (const std::overflow_error&) {
            return;
        }
        throw std::runtime_error("CUDA scheduler accepted an overflowing tensor size");
    }
}

void test_cuda_vae_tile_extraction() {
    constexpr int32_t video_rows_count = 37296;
    constexpr int32_t patch_dim = 96;
    constexpr int32_t tile_count = 28;
    constexpr int32_t channels = 24;
    constexpr int32_t frames = 7;
    constexpr int32_t tile_size = 16;
    const std::size_t video_count = static_cast<std::size_t>(video_rows_count) * patch_dim;
    const std::size_t tile_values =
        static_cast<std::size_t>(tile_count) * channels * frames * tile_size * tile_size;
    std::vector<float> video_rows(video_count);
    for (int32_t row = 0; row < video_rows_count; ++row) {
        for (int32_t column = 0; column < patch_dim; ++column)
            video_rows[static_cast<std::size_t>(row) * patch_dim + column] =
                static_cast<float>(row * 128 + column);
    }
    std::vector<float> tiles(tile_values);
    trtmc::minimax_h3::VaeLatentNormalization normalization{};
    for (int32_t channel = 0; channel < channels; ++channel) {
        normalization.mean[channel] = static_cast<float>(channel) * 0.25F;
        normalization.std[channel] = 0.5F;
    }

    cudaStream_t stream = nullptr;
    float* device_video = nullptr;
    float* device_tiles = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_video), video_count * sizeof(float)),
                   "VAE video cudaMalloc");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_tiles), tile_values * sizeof(float)),
                   "VAE tiles cudaMalloc");
        check_cuda(cudaMemcpyAsync(device_video, video_rows.data(), video_count * sizeof(float),
                                   cudaMemcpyHostToDevice, stream),
                   "VAE video cudaMemcpyAsync");
        trtmc::minimax_h3::extract_vae_tiles_cuda_async(device_video, device_tiles, 6,
                                                        normalization, stream);
        check_cuda(cudaMemcpyAsync(tiles.data(), device_tiles, tile_values * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "VAE tiles cudaMemcpyAsync");
        check_cuda(cudaStreamSynchronize(stream), "VAE extraction cudaStreamSynchronize");
    } catch (...) {
        cudaFree(device_tiles);
        cudaFree(device_video);
        cudaStreamDestroy(stream);
        throw;
    }
    check_cuda(cudaFree(device_tiles), "VAE tiles cudaFree");
    check_cuda(cudaFree(device_video), "VAE video cudaFree");
    check_cuda(cudaStreamDestroy(stream), "VAE cudaStreamDestroy");

    const auto check_sample = [&](int32_t tile, int32_t channel, int32_t frame, int32_t y,
                                  int32_t x, int32_t latent_y_start, int32_t latent_x_start) {
        const int32_t latent_frame = 6 * 5 + frame;
        const int32_t latent_y = latent_y_start + y;
        const int32_t latent_x = latent_x_start + x;
        const int32_t row = ((latent_frame * 24 + latent_y / 2) * 42 + latent_x / 2);
        const int32_t column = channel * 4 + (latent_y % 2) * 2 + latent_x % 2;
        const float source = video_rows[static_cast<std::size_t>(row) * patch_dim + column];
        const float expected = source * normalization.std[channel] + normalization.mean[channel];
        const std::size_t target =
            ((((static_cast<std::size_t>(tile) * channels + channel) * frames + frame) * tile_size +
              y) *
                 tile_size +
             x);
        if (tiles[target] != expected)
            throw std::runtime_error("CUDA VAE tile extraction index mismatch");
    };
    check_sample(0, 0, 0, 0, 0, 0, 0);
    check_sample(8, 7, 3, 5, 9, 10, 11);
    check_sample(27, 23, 6, 15, 15, 32, 68);
}

void test_cuda_vae_helpers_reject_invalid_inputs() {
    trtmc::minimax_h3::VaeLatentNormalization latent_normalization{};
    trtmc::minimax_h3::VaePixelNormalization pixel_normalization{};
    float* const dummy = reinterpret_cast<float*>(static_cast<uintptr_t>(1));
    const auto rejects_invalid = [](auto&& operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    if (!rejects_invalid([&] {
            trtmc::minimax_h3::extract_vae_tiles_cuda_async(nullptr, dummy, 0, latent_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::extract_vae_tiles_cuda_async(dummy, dummy, 7, latent_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::assemble_vae_clip_cuda_async(
                nullptr, dummy, dummy, 0, pixel_normalization, reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::assemble_vae_clip_cuda_async(
                dummy, dummy, dummy, -1, pixel_normalization, reinterpret_cast<cudaStream_t>(1));
        })) {
        throw std::runtime_error("CUDA VAE helper accepted invalid inputs");
    }
}

} // namespace

int main() {
    constexpr std::size_t video_count = 24ULL * 37ULL * 48ULL * 84ULL;
    constexpr std::size_t audio_count = 414ULL * 32ULL;
    const auto video = trtmc::minimax_h3::torch_cuda_normal(video_count, 0);
    const auto offset = trtmc::minimax_h3::torch_cuda_normal_consumed_offset(video_count);
    const auto audio = trtmc::minimax_h3::torch_cuda_normal(audio_count, 0, offset);
    constexpr uint32_t expected_video[] = {0xbf901b85U, 0xbf93808aU, 0xbe804bd6U, 0xbede255eU,
                                           0x3f594515U, 0x3f312784U, 0xbea1cc6dU, 0xc0075fc2U};
    constexpr uint32_t expected_audio[] = {0x3f6c3b90U, 0xbf07dfc2U, 0xbfb125e1U, 0xbe1fae64U,
                                           0x3e1c49afU, 0xbf9724beU, 0xbf320601U, 0xc02f1fdaU};
    int failures = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        if (bits(video[index]) != expected_video[index]) {
            std::cerr << "FAIL: H3 video torch.randn mismatch at " << index << " actual=0x"
                      << std::hex << bits(video[index]) << " expected=0x" << expected_video[index]
                      << std::dec << '\n';
            ++failures;
        }
        if (bits(audio[index]) != expected_audio[index]) {
            std::cerr << "FAIL: H3 sequential audio torch.randn mismatch at " << index
                      << " actual=0x" << std::hex << bits(audio[index]) << " expected=0x"
                      << expected_audio[index] << std::dec << '\n';
            ++failures;
        }
    }
    uint64_t hash = update_fnv1a(14695981039346656037ULL, video);
    hash = update_fnv1a(hash, audio);
    if (hash != 0xb68438da31d3c096ULL) {
        std::cerr << "FAIL: H3 full video+audio torch.randn hash mismatch actual=0x" << std::hex
                  << hash << std::dec << '\n';
        ++failures;
    }
    try {
        test_scheduler_matches_cpu();
        test_scheduler_rejects_invalid_inputs();
        test_cuda_vae_tile_extraction();
        test_cuda_vae_helpers_reject_invalid_inputs();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
