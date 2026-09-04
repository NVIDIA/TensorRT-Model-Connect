/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/torch_cuda_normal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::minimax_h3 {
namespace {

constexpr std::size_t kStateSize = 624;
constexpr std::size_t kStatePeriod = 397;
constexpr uint32_t kMatrixA = 0x9908b0dfU;
constexpr uint32_t kUpperMask = 0x80000000U;
constexpr uint32_t kLowerMask = 0x7fffffffU;
constexpr float kUniformScale = 1.0F / 16777216.0F;
constexpr double kPi = 3.14159265358979323846;
constexpr uint32_t kSchedulerBlockSize = 256;
constexpr uint32_t kSchedulerMaxBlocks = 4096;
constexpr int32_t kVaeTileRows = 4;
constexpr int32_t kVaeTileColumns = 7;
constexpr int32_t kVaeTileCount = kVaeTileRows * kVaeTileColumns;
constexpr int32_t kVaeLatentChannels = 24;
constexpr int32_t kVaeTileInputFrames = 7;
constexpr int32_t kVaeTileLatentSize = 16;
constexpr int32_t kVaePatchHeight = 2;
constexpr int32_t kVaePatchWidth = 2;
constexpr int32_t kVaePatchDim = 96;
constexpr int32_t kVaeLatentHeight = 48;
constexpr int32_t kVaeLatentWidth = 84;
constexpr int32_t kVaeTileFrames = 28;
constexpr int32_t kVaeTileSize = 256;
constexpr int32_t kVaeOutputChannels = 3;
constexpr int32_t kVaeOutputHeight = 768;
constexpr int32_t kVaeOutputWidth = 1344;
constexpr int32_t kVaeChunkFrames = 17;
constexpr int32_t kVaeTemporalOverlapFrames = 5;
constexpr int32_t kVaeTemporalPrePadding = 3;
constexpr int32_t kVaeTrailingOverlapStart = 23;

// This is PyTorch's at::mt19937 engine, not std::mt19937. The state transition
// and low-24-bit uniform mapping are part of torch.Generator CPU determinism.
class TorchMt19937 {
  public:
    explicit TorchMt19937(uint64_t seed) { seed_state(seed); }

    uint32_t next() {
        if (--left_ == 0)
            next_state();
        uint32_t value = state_[next_++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680U;
        value ^= (value << 15) & 0xefc60000U;
        value ^= value >> 18;
        return value;
    }

  private:
    static uint32_t twist(uint32_t first, uint32_t second) {
        const uint32_t mixed = (first & kUpperMask) | (second & kLowerMask);
        return (mixed >> 1) ^ ((second & 1U) != 0U ? kMatrixA : 0U);
    }

    void seed_state(uint64_t seed) {
        state_[0] = static_cast<uint32_t>(seed & 0xffffffffULL);
        for (std::size_t index = 1; index < kStateSize; ++index) {
            state_[index] = 1812433253U * (state_[index - 1] ^ (state_[index - 1] >> 30)) +
                            static_cast<uint32_t>(index);
        }
        left_ = 1;
        next_ = 0;
    }

    void next_state() {
        uint32_t* current = state_.data();
        left_ = static_cast<int32_t>(kStateSize);
        next_ = 0;
        for (std::size_t count = kStateSize - kStatePeriod; count > 0; --count, ++current)
            *current = current[kStatePeriod] ^ twist(current[0], current[1]);
        for (std::size_t count = kStatePeriod - 1; count > 0; --count, ++current)
            *current = current[kStatePeriod - kStateSize] ^ twist(current[0], current[1]);
        *current = current[kStatePeriod - kStateSize] ^ twist(current[0], state_[0]);
    }

    std::array<uint32_t, kStateSize> state_{};
    int32_t left_{1};
    std::size_t next_{0};
};

float uniform(TorchMt19937& generator) {
    return static_cast<float>(generator.next() & 0x00ffffffU) * kUniformScale;
}

void normal_fill_16(float* values) {
    for (std::size_t index = 0; index < 8; ++index) {
        const float first = 1.0F - values[index];
        const float second = values[index + 8];
        const float radius = std::sqrt(-2.0F * std::log(first));
        const float theta = static_cast<float>((2.0F * kPi) * static_cast<double>(second));
        values[index] = std::fma(radius * std::cos(theta), 1.0F, 0.0F);
        values[index + 8] = std::fma(radius * std::sin(theta), 1.0F, 0.0F);
    }
}

__global__ void scheduler_step_kernel(float* sample, const float* velocity, int64_t count,
                                      float sigma_from_timestep, float ratio) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        const float current = sample[index];
        const float denoised = current + sigma_from_timestep * velocity[index];
        sample[index] = ratio * current + (1.0F - ratio) * denoised;
    }
}

__device__ __forceinline__ int32_t latent_y_start(int32_t tile_y) {
    switch (tile_y) {
    case 0:
        return 0;
    case 1:
        return 10;
    case 2:
        return 21;
    default:
        return 32;
    }
}

__device__ __forceinline__ int32_t latent_x_start(int32_t tile_x) {
    switch (tile_x) {
    case 0:
        return 0;
    case 1:
        return 11;
    case 2:
        return 22;
    case 3:
        return 33;
    case 4:
        return 44;
    case 5:
        return 56;
    default:
        return 68;
    }
}

__device__ __forceinline__ int32_t output_y_start(int32_t tile_y) {
    switch (tile_y) {
    case 0:
        return 0;
    case 1:
        return 160;
    case 2:
        return 336;
    default:
        return 512;
    }
}

__device__ __forceinline__ int32_t output_x_start(int32_t tile_x) {
    switch (tile_x) {
    case 0:
        return 0;
    case 1:
        return 176;
    case 2:
        return 352;
    case 3:
        return 528;
    case 4:
        return 704;
    case 5:
        return 896;
    default:
        return 1088;
    }
}

__device__ __forceinline__ int32_t height_overlap(int32_t boundary) {
    return boundary == 0 ? 96 : 80;
}

__device__ __forceinline__ int32_t width_overlap(int32_t boundary) {
    return boundary < 4 ? 80 : 64;
}

__global__ void extract_vae_tiles_kernel(const float* video_rows, float* latent_tiles,
                                         int32_t clip_index, VaeLatentNormalization normalization,
                                         int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += stride) {
        int64_t remaining = linear;
        const int32_t x = static_cast<int32_t>(remaining % kVaeTileLatentSize);
        remaining /= kVaeTileLatentSize;
        const int32_t y = static_cast<int32_t>(remaining % kVaeTileLatentSize);
        remaining /= kVaeTileLatentSize;
        const int32_t frame = static_cast<int32_t>(remaining % kVaeTileInputFrames);
        remaining /= kVaeTileInputFrames;
        const int32_t channel = static_cast<int32_t>(remaining % kVaeLatentChannels);
        const int32_t tile = static_cast<int32_t>(remaining / kVaeLatentChannels);

        const int32_t latent_frame = clip_index * kVaeTemporalOverlapFrames + frame;
        const int32_t latent_y = latent_y_start(tile / kVaeTileColumns) + y;
        const int32_t latent_x = latent_x_start(tile % kVaeTileColumns) + x;
        const int32_t patch_row =
            ((latent_frame * (kVaeLatentHeight / kVaePatchHeight) + latent_y / kVaePatchHeight) *
                 (kVaeLatentWidth / kVaePatchWidth) +
             latent_x / kVaePatchWidth);
        const int32_t patch_column = channel * kVaePatchHeight * kVaePatchWidth +
                                     (latent_y % kVaePatchHeight) * kVaePatchWidth +
                                     latent_x % kVaePatchWidth;
        const float value =
            video_rows[static_cast<int64_t>(patch_row) * kVaePatchDim + patch_column];
        latent_tiles[linear] = value * normalization.std[channel] + normalization.mean[channel];
    }
}

__device__ __forceinline__ int32_t output_tile_y(int32_t y) {
    if (y < 160)
        return 0;
    if (y < 336)
        return 1;
    if (y < 512)
        return 2;
    return 3;
}

__device__ __forceinline__ int32_t output_tile_x(int32_t x) {
    if (x < 176)
        return 0;
    if (x < 352)
        return 1;
    if (x < 528)
        return 2;
    if (x < 704)
        return 3;
    if (x < 896)
        return 4;
    if (x < 1088)
        return 5;
    return 6;
}

__device__ __forceinline__ float decoded_tile_value(const float* decoded_tiles, int32_t tile,
                                                    int32_t channel, int32_t frame, int32_t y,
                                                    int32_t x) {
    const int64_t index =
        ((((static_cast<int64_t>(tile) * kVaeOutputChannels + channel) * kVaeTileFrames + frame) *
              kVaeTileSize +
          y) *
             kVaeTileSize +
         x);
    return decoded_tiles[index];
}

__device__ __forceinline__ float spatially_stitched_value(const float* decoded_tiles,
                                                          int32_t channel, int32_t frame,
                                                          int32_t output_y, int32_t output_x) {
    const int32_t tile_y = output_tile_y(output_y);
    const int32_t tile_x = output_tile_x(output_x);
    const int32_t tile = tile_y * kVaeTileColumns + tile_x;
    const int32_t y = output_y - output_y_start(tile_y);
    const int32_t x = output_x - output_x_start(tile_x);
    float value = decoded_tile_value(decoded_tiles, tile, channel, frame, y, x);

    // Match stitch_one_spatial_tile exactly: vertical ownership/blend first,
    // followed by the horizontal blend from the left tile.
    if (tile_y > 0 && y < height_overlap(tile_y - 1)) {
        const int32_t overlap = height_overlap(tile_y - 1);
        const float weight_b = static_cast<float>(y) / overlap;
        const float upper = decoded_tile_value(decoded_tiles, tile - kVaeTileColumns, channel,
                                               frame, kVaeTileSize - overlap + y, x);
        value = upper * (1.0F - weight_b) + value * weight_b;
    }
    if (tile_x > 0 && x < width_overlap(tile_x - 1)) {
        const int32_t overlap = width_overlap(tile_x - 1);
        const float weight_b = static_cast<float>(x) / overlap;
        const float left = decoded_tile_value(decoded_tiles, tile - 1, channel, frame, y,
                                              kVaeTileSize - overlap + x);
        value = left * (1.0F - weight_b) + value * weight_b;
    }
    return value;
}

__device__ __forceinline__ float normalize_pixel(float value, int32_t channel,
                                                 VaePixelNormalization normalization) {
    value = value * normalization.std[channel] + normalization.mean[channel];
    if (value < 0.0F)
        value = 0.0F;
    else if (value > 1.0F)
        value = 1.0F;
    return value;
}

__global__ void assemble_vae_chunk_kernel(const float* decoded_tiles, const float* overlap,
                                          float* frame_major_rgb, int32_t clip_index,
                                          VaePixelNormalization normalization, int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += stride) {
        int64_t remaining = linear;
        const int32_t output_x = static_cast<int32_t>(remaining % kVaeOutputWidth);
        remaining /= kVaeOutputWidth;
        const int32_t output_y = static_cast<int32_t>(remaining % kVaeOutputHeight);
        remaining /= kVaeOutputHeight;
        const int32_t frame = static_cast<int32_t>(remaining % kVaeChunkFrames);
        const int32_t channel = static_cast<int32_t>(remaining / kVaeChunkFrames);

        float value = spatially_stitched_value(decoded_tiles, channel,
                                               kVaeTemporalPrePadding + frame, output_y, output_x);
        if (clip_index > 0 && frame < kVaeTemporalOverlapFrames) {
            const int64_t overlap_index =
                (((static_cast<int64_t>(channel) * kVaeTemporalOverlapFrames + frame) *
                      kVaeOutputHeight +
                  output_y) *
                     kVaeOutputWidth +
                 output_x);
            const float weight_b = static_cast<float>(frame) / kVaeTemporalOverlapFrames;
            value = overlap[overlap_index] * (1.0F - weight_b) + value * weight_b;
        }
        value = normalize_pixel(value, channel, normalization);
        const int32_t output_frame = clip_index * kVaeChunkFrames + frame;
        const int64_t output_index =
            (((static_cast<int64_t>(output_frame) * kVaeOutputHeight + output_y) * kVaeOutputWidth +
              output_x) *
                 kVaeOutputChannels +
             channel);
        frame_major_rgb[output_index] = value;
    }
}

__global__ void update_vae_overlap_kernel(const float* decoded_tiles, float* overlap,
                                          float* frame_major_rgb, int32_t clip_index,
                                          VaePixelNormalization normalization, int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += stride) {
        int64_t remaining = linear;
        const int32_t output_x = static_cast<int32_t>(remaining % kVaeOutputWidth);
        remaining /= kVaeOutputWidth;
        const int32_t output_y = static_cast<int32_t>(remaining % kVaeOutputHeight);
        remaining /= kVaeOutputHeight;
        const int32_t frame = static_cast<int32_t>(remaining % kVaeTemporalOverlapFrames);
        const int32_t channel = static_cast<int32_t>(remaining / kVaeTemporalOverlapFrames);

        const float value = spatially_stitched_value(
            decoded_tiles, channel, kVaeTrailingOverlapStart + frame, output_y, output_x);
        overlap[linear] = value;
        if (clip_index == 6) {
            const int32_t output_frame = kVaeChunkFrames * 7 + frame;
            const int64_t output_index =
                (((static_cast<int64_t>(output_frame) * kVaeOutputHeight + output_y) *
                      kVaeOutputWidth +
                  output_x) *
                     kVaeOutputChannels +
                 channel);
            frame_major_rgb[output_index] = normalize_pixel(value, channel, normalization);
        }
    }
}

uint32_t launch_grid(std::size_t count) {
    const std::size_t requested_blocks =
        count / kSchedulerBlockSize + (count % kSchedulerBlockSize != 0 ? 1U : 0U);
    return static_cast<uint32_t>(std::min<std::size_t>(requested_blocks, kSchedulerMaxBlocks));
}

void check_kernel_launch(const char* label) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(label) + " failed: " + cudaGetErrorString(status));
}

} // namespace

uint64_t torch_cuda_normal_consumed_offset(std::size_t count) {
    if (count < 16)
        return static_cast<uint64_t>(2 * ((count + 1) / 2));
    return static_cast<uint64_t>(count + (count % 16 == 0 ? 0 : 16));
}

std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed, uint64_t offset) {
    if (count == 0)
        return {};
    if (count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("MiniMax-H3 CPU RNG tensor is too large");
    if (count < 16)
        throw std::invalid_argument("MiniMax-H3 CPU RNG requires at least 16 samples");
    TorchMt19937 generator(seed);
    for (uint64_t index = 0; index < offset; ++index)
        (void)generator.next();
    std::vector<float> output(count);
    for (float& value : output)
        value = uniform(generator);
    for (std::size_t index = 0; index + 15 < count; index += 16)
        normal_fill_16(output.data() + index);
    if (count % 16 != 0) {
        float tail[16];
        for (float& value : tail)
            value = uniform(generator);
        normal_fill_16(tail);
        std::memcpy(output.data() + count - 16, tail, sizeof(tail));
    }
    return output;
}

void scheduler_step_cuda_async(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next, cudaStream_t stream) {
    if (sample == nullptr || velocity == nullptr || !(sigma > 0.0F))
        throw std::invalid_argument("MiniMax-H3 CUDA scheduler received invalid inputs");
    if (count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("MiniMax-H3 CUDA scheduler tensor is too large");
    if (count == 0)
        return;

    const uint32_t grid = launch_grid(count);
    const float sigma_from_timestep = 1.0F - timestep;
    const float ratio = sigma_next / sigma;
    scheduler_step_kernel<<<grid, kSchedulerBlockSize, 0, stream>>>(
        sample, velocity, static_cast<int64_t>(count), sigma_from_timestep, ratio);
    check_kernel_launch("MiniMax-H3 CUDA scheduler launch");
}

void extract_vae_tiles_cuda_async(const float* video_rows, float* latent_tiles, int32_t clip_index,
                                  VaeLatentNormalization normalization, cudaStream_t stream) {
    if (video_rows == nullptr || latent_tiles == nullptr || stream == nullptr || clip_index < 0 ||
        clip_index >= 7)
        throw std::invalid_argument("MiniMax-H3 CUDA VAE extraction received invalid inputs");
    constexpr std::size_t count = static_cast<std::size_t>(kVaeTileCount) * kVaeLatentChannels *
                                  kVaeTileInputFrames * kVaeTileLatentSize * kVaeTileLatentSize;
    extract_vae_tiles_kernel<<<launch_grid(count), kSchedulerBlockSize, 0, stream>>>(
        video_rows, latent_tiles, clip_index, normalization, static_cast<int64_t>(count));
    check_kernel_launch("MiniMax-H3 CUDA VAE extraction launch");
}

void assemble_vae_clip_cuda_async(const float* decoded_tiles, float* overlap,
                                  float* frame_major_rgb, int32_t clip_index,
                                  VaePixelNormalization normalization, cudaStream_t stream) {
    if (decoded_tiles == nullptr || overlap == nullptr || frame_major_rgb == nullptr ||
        stream == nullptr || clip_index < 0 || clip_index >= 7)
        throw std::invalid_argument("MiniMax-H3 CUDA VAE assembly received invalid inputs");
    constexpr std::size_t chunk_count = static_cast<std::size_t>(kVaeOutputChannels) *
                                        kVaeChunkFrames * kVaeOutputHeight * kVaeOutputWidth;
    assemble_vae_chunk_kernel<<<launch_grid(chunk_count), kSchedulerBlockSize, 0, stream>>>(
        decoded_tiles, overlap, frame_major_rgb, clip_index, normalization,
        static_cast<int64_t>(chunk_count));
    check_kernel_launch("MiniMax-H3 CUDA VAE chunk assembly launch");

    constexpr std::size_t overlap_count = static_cast<std::size_t>(kVaeOutputChannels) *
                                          kVaeTemporalOverlapFrames * kVaeOutputHeight *
                                          kVaeOutputWidth;
    update_vae_overlap_kernel<<<launch_grid(overlap_count), kSchedulerBlockSize, 0, stream>>>(
        decoded_tiles, overlap, frame_major_rgb, clip_index, normalization,
        static_cast<int64_t>(overlap_count));
    check_kernel_launch("MiniMax-H3 CUDA VAE overlap assembly launch");
}

} // namespace trtmc::minimax_h3
