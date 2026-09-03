/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"
#include "runtime/models/minimax_h3/torch_cuda_normal.h"

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

class TestTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

uint32_t bits(float value) {
    uint32_t output = 0;
    std::memcpy(&output, &value, sizeof(output));
    return output;
}

uint32_t ulp_distance(float actual, uint32_t expected_bits) {
    const uint32_t actual_bits = bits(actual);
    if ((actual_bits >> 31) != (expected_bits >> 31))
        return std::numeric_limits<uint32_t>::max();
    return actual_bits >= expected_bits ? actual_bits - expected_bits : expected_bits - actual_bits;
}

bool has_standard_normal_semantics(const std::vector<float>& values) {
    long double sum = 0.0;
    long double squared_sum = 0.0;
    for (const float value : values) {
        if (!std::isfinite(value))
            return false;
        sum += value;
        squared_sum += static_cast<long double>(value) * value;
    }
    const long double mean = sum / values.size();
    const long double variance = squared_sum / values.size() - mean * mean;
    return std::abs(mean) < 0.05L && variance > 0.9L && variance < 1.1L;
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

void test_pipeline_destruction_flushes_runtime_cache() {
    bool flushed = false;
    trtmc::MiniMaxH3ModuleLoader unused_loader =
        [](const std::string&, cudaStream_t,
           const std::vector<trtmc::ModuleExternalBinding>&) -> std::unique_ptr<trtmc::ITrtModule> {
        throw std::runtime_error("test loader must not be invoked");
    };
    {
        trtmc::MiniMaxH3Pipeline pipeline(std::move(unused_loader),
                                          std::make_unique<TestTokenizer>(), "test-minimax-h3",
                                          false, 0.025F, {}, {}, [&] { flushed = true; });
    }
    if (!flushed)
        throw std::runtime_error("MiniMax-H3 pipeline destruction did not flush runtime cache");
}

trtmc::MiniMaxH3ModuleLoader make_unused_pipeline_loader() {
    return
        [](const std::string&, cudaStream_t,
           const std::vector<trtmc::ModuleExternalBinding>&) -> std::unique_ptr<trtmc::ITrtModule> {
            throw std::runtime_error("test loader must not be invoked");
        };
}

void test_pipeline_rejects_initial_latents_before_plan_loading() {
    trtmc::MiniMaxH3Pipeline pipeline(make_unused_pipeline_loader(),
                                      std::make_unique<TestTokenizer>(), "test-minimax-h3");
    trtmc::VideoGenerationRequest request;
    request.mode = trtmc::VideoGenerationMode::kTextToVideoAudio;
    request.prompt = "must-not-tokenize";
    request.config.initial_latents = {0.0F};
    bool rejected = false;
    try {
        (void)pipeline.generate_video(request);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("does not accept initial_latents") !=
                   std::string::npos;
    }
    if (!rejected)
        throw std::runtime_error("MiniMax-H3 did not reject caller-supplied initial latents");
}

void test_pipeline_explicit_cache_finalization_is_once_and_terminal() {
    int finalizations = 0;
    {
        trtmc::MiniMaxH3Pipeline pipeline(make_unused_pipeline_loader(),
                                          std::make_unique<TestTokenizer>(), "test-minimax-h3",
                                          false, 0.025F, {}, {}, [&] { ++finalizations; });
        pipeline.finalize_runtime_cache();
        pipeline.finalize_runtime_cache();
        if (finalizations != 1)
            throw std::runtime_error("explicit runtime-cache finalization was not idempotent");

        bool generation_rejected = false;
        try {
            (void)pipeline.generate_video("must-not-run");
        } catch (const std::runtime_error& error) {
            generation_rejected =
                std::string(error.what()).find("after runtime-cache finalization") !=
                std::string::npos;
        }
        if (!generation_rejected) {
            throw std::runtime_error(
                "MiniMax-H3 allowed generation after runtime-cache finalization");
        }
    }
    if (finalizations != 1)
        throw std::runtime_error("pipeline destructor repeated successful cache finalization");
}

void test_pipeline_cache_finalization_failure_propagates_and_retries() {
    int attempts = 0;
    {
        trtmc::MiniMaxH3Pipeline pipeline(
            make_unused_pipeline_loader(), std::make_unique<TestTokenizer>(), "test-minimax-h3",
            false, 0.025F, {}, {}, [&] {
                ++attempts;
                if (attempts == 1)
                    throw std::runtime_error("synthetic runtime-cache persistence failure");
            });
        bool propagated = false;
        try {
            pipeline.finalize_runtime_cache();
        } catch (const std::runtime_error& error) {
            propagated =
                std::string(error.what()).find("synthetic runtime-cache") != std::string::npos;
        }
        if (!propagated)
            throw std::runtime_error("explicit cache-persistence failure was not propagated");
    }
    if (attempts != 2) {
        throw std::runtime_error(
            "pipeline destructor did not retry a failed runtime-cache finalization");
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
    constexpr int32_t video_rows_count = 102816;
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
        trtmc::minimax_h3::extract_vae_tiles_cuda_async(device_video, device_tiles, 19, 20, 768,
                                                        1344, 4, 7, normalization, stream);
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
        const int32_t latent_frame = 19 * 5 + frame;
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

void test_cuda_vae_extreme_canvas_extraction_is_bounded() {
    constexpr int32_t output_height = 512;
    constexpr int32_t output_width = 2016;
    constexpr int32_t tile_rows = 3;
    constexpr int32_t tile_columns = 11;
    constexpr int32_t tile_count = tile_rows * tile_columns;
    constexpr int32_t latent_frames = 37;
    constexpr int32_t latent_height = output_height / 16;
    constexpr int32_t latent_width = output_width / 16;
    constexpr int32_t patch_dim = 96;
    constexpr int32_t channels = 24;
    constexpr int32_t tile_frames = 7;
    constexpr int32_t tile_size = 16;
    const std::size_t video_rows_count =
        static_cast<std::size_t>(latent_frames) * (latent_height / 2) * (latent_width / 2);
    std::vector<float> video_rows(video_rows_count * patch_dim);
    for (std::size_t row = 0; row < video_rows_count; ++row) {
        for (int32_t column = 0; column < patch_dim; ++column)
            video_rows[row * patch_dim + column] = static_cast<float>(row * 128 + column);
    }

    const std::size_t tile_values =
        static_cast<std::size_t>(tile_count) * channels * tile_frames * tile_size * tile_size;
    constexpr float guard = -12345.0F;
    std::vector<float> guarded_tiles(tile_values + 2, guard);
    trtmc::minimax_h3::VaeLatentNormalization normalization{};
    for (float& value : normalization.std)
        value = 1.0F;

    cudaStream_t stream = nullptr;
    float* device_video = nullptr;
    float* device_guarded_tiles = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "extreme cudaStreamCreate");
    try {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&device_video), video_rows.size() * sizeof(float)),
            "extreme video cudaMalloc");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_guarded_tiles),
                              guarded_tiles.size() * sizeof(float)),
                   "extreme tiles cudaMalloc");
        check_cuda(cudaMemcpyAsync(device_video, video_rows.data(),
                                   video_rows.size() * sizeof(float), cudaMemcpyHostToDevice,
                                   stream),
                   "extreme video cudaMemcpyAsync");
        check_cuda(cudaMemcpyAsync(device_guarded_tiles, guarded_tiles.data(),
                                   guarded_tiles.size() * sizeof(float), cudaMemcpyHostToDevice,
                                   stream),
                   "extreme guard cudaMemcpyAsync");
        trtmc::minimax_h3::extract_vae_tiles_cuda_async(device_video, device_guarded_tiles + 1, 0,
                                                        7, output_height, output_width, tile_rows,
                                                        tile_columns, normalization, stream);
        check_cuda(cudaMemcpyAsync(guarded_tiles.data(), device_guarded_tiles,
                                   guarded_tiles.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream),
                   "extreme tiles cudaMemcpyAsync");
        check_cuda(cudaStreamSynchronize(stream), "extreme extraction cudaStreamSynchronize");
    } catch (...) {
        cudaFree(device_guarded_tiles);
        cudaFree(device_video);
        cudaStreamDestroy(stream);
        throw;
    }
    check_cuda(cudaFree(device_guarded_tiles), "extreme tiles cudaFree");
    check_cuda(cudaFree(device_video), "extreme video cudaFree");
    check_cuda(cudaStreamDestroy(stream), "extreme cudaStreamDestroy");

    if (guarded_tiles.front() != guard || guarded_tiles.back() != guard)
        throw std::runtime_error("CUDA VAE extreme canvas extraction wrote outside its tile batch");
    constexpr int32_t tile = tile_count - 1;
    constexpr int32_t channel = channels - 1;
    constexpr int32_t frame = tile_frames - 1;
    constexpr int32_t y = tile_size - 1;
    constexpr int32_t x = tile_size - 1;
    constexpr int32_t latent_y = 16 + y;
    constexpr int32_t latent_x = 110 + x;
    const int32_t row =
        ((frame * (latent_height / 2) + latent_y / 2) * (latent_width / 2) + latent_x / 2);
    const int32_t column = channel * 4 + (latent_y % 2) * 2 + latent_x % 2;
    const float expected = video_rows[static_cast<std::size_t>(row) * patch_dim + column];
    const std::size_t target =
        1 + ((((static_cast<std::size_t>(tile) * channels + channel) * tile_frames + frame) *
                  tile_size +
              y) *
                 tile_size +
             x);
    if (guarded_tiles[target] != expected)
        throw std::runtime_error("CUDA VAE extreme canvas last tile index mismatch");
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
            trtmc::minimax_h3::extract_vae_tiles_cuda_async(nullptr, dummy, 0, 7, 768, 1344, 4, 7,
                                                            latent_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::extract_vae_tiles_cuda_async(dummy, dummy, 20, 20, 768, 1344, 4, 7,
                                                            latent_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::assemble_vae_clip_cuda_async(nullptr, dummy, dummy, 0, 7, 768, 1344,
                                                            4, 7, pixel_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::assemble_vae_clip_cuda_async(dummy, dummy, dummy, -1, 7, 768, 1344,
                                                            4, 7, pixel_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::assemble_vae_clip_cuda_async(dummy, dummy, dummy, 0, 21, 768, 1344,
                                                            4, 7, pixel_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
        }) ||
        !rejects_invalid([&] {
            trtmc::minimax_h3::extract_vae_tiles_cuda_async(dummy, dummy, 0, 7, 768, 1344, 4, 6,
                                                            latent_normalization,
                                                            reinterpret_cast<cudaStream_t>(1));
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
        if (ulp_distance(video[index], expected_video[index]) > 1) {
            std::cerr << "FAIL: H3 video torch.randn mismatch at " << index << " actual=0x"
                      << std::hex << bits(video[index]) << " expected=0x" << expected_video[index]
                      << std::dec << '\n';
            ++failures;
        }
        if (ulp_distance(audio[index], expected_audio[index]) > 1) {
            std::cerr << "FAIL: H3 sequential audio torch.randn mismatch at " << index
                      << " actual=0x" << std::hex << bits(audio[index]) << " expected=0x"
                      << expected_audio[index] << std::dec << '\n';
            ++failures;
        }
    }
    uint64_t hash = update_fnv1a(14695981039346656037ULL, video);
    hash = update_fnv1a(hash, audio);
    // MSVC's scalar libm sin/cos is allowed to differ by one ULP from the
    // AArch64/PyTorch reproduction anchor above. Keep a platform-exact full
    // tensor hash as the strong regression check instead of weakening the
    // whole tensor comparison to a broad tolerance.
#ifdef _WIN32
#ifdef _DLL
    // /MD uses the shared MSVC/UCRT scalar sin/cos implementation.
    constexpr uint64_t expected_hash = 0xdfade50efdb42d51ULL;
#else
    // The redistributable Windows H3 build uses /MT. Its scalar libm path is
    // deterministic but has a distinct full-tensor bit anchor.
    constexpr uint64_t expected_hash = 0x05f090b1886fbca6ULL;
#endif
#else
    constexpr uint64_t expected_hash = 0xb68438da31d3c096ULL;
#endif
    if (hash != expected_hash) {
        std::cerr << "FAIL: H3 full video+audio torch.randn hash mismatch actual=0x" << std::hex
                  << hash << std::dec << '\n';
        ++failures;
    }
    if (!has_standard_normal_semantics(video) || !has_standard_normal_semantics(audio)) {
        std::cerr << "FAIL: H3 torch.randn tensor lost standard-normal semantics\n";
        ++failures;
    }
    try {
        test_scheduler_matches_cpu();
        test_pipeline_destruction_flushes_runtime_cache();
        test_pipeline_rejects_initial_latents_before_plan_loading();
        test_pipeline_explicit_cache_finalization_is_once_and_terminal();
        test_pipeline_cache_finalization_failure_propagates_and_retries();
        test_scheduler_rejects_invalid_inputs();
        test_cuda_vae_tile_extraction();
        test_cuda_vae_extreme_canvas_extraction_is_bounded();
        test_cuda_vae_helpers_reject_invalid_inputs();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
