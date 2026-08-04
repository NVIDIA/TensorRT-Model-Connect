/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include "runtime/models/minimax_h3/torch_cuda_normal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t kTextRows = 537;
constexpr int32_t kTextDim = 5120;
constexpr int32_t kAudioLatents = 207;
constexpr int32_t kAudioRows = 414;
constexpr int32_t kAudioChannels = 32;
constexpr int32_t kLatentFrames = 37;
constexpr int32_t kLatentHeight = 48;
constexpr int32_t kLatentWidth = 84;
constexpr int32_t kLatentChannels = 24;
constexpr int32_t kPatchHeight = 2;
constexpr int32_t kPatchWidth = 2;
constexpr int32_t kPatchDim = 96;
constexpr int32_t kVideoRows = 37296;
constexpr int32_t kSequenceRows = 38247;
constexpr int32_t kLayers = 50;
constexpr int32_t kHidden = 5376;
constexpr int32_t kTimestepSlots = 4;
constexpr int32_t kModalityCount = 3;
constexpr int32_t kAdalnRows = kTimestepSlots * kModalityCount;
constexpr int32_t kSteps = 50;
constexpr int32_t kOutputFrames = 124;
constexpr int32_t kOutputHeight = 768;
constexpr int32_t kOutputWidth = 1344;
constexpr int32_t kTileBatch = 28;
constexpr int32_t kTileFrames = 28;
constexpr int32_t kTileSize = 256;
constexpr int32_t kTileLatentSize = 16;
constexpr int32_t kTileInputFrames = 7;
constexpr int32_t kTileCount = 28;
static_assert(kTileBatch == kTileCount);

constexpr std::array<float, kLatentChannels> kLatentMean = {
    0.8580903411F,  -0.9606591463F, 1.0661640167F,  -0.5090325475F, -0.2727581859F, -1.3675414324F,
    -0.2553254962F, -0.2690755427F, -0.5376840830F, -0.0464097299F, 0.6657370329F,  0.1969012767F,
    -0.5460608006F, -0.4035342038F, -0.2368302494F, 0.2592845261F,  -0.3013394475F, 0.2113419920F,
    -1.1206848621F, 0.3581933379F,  -0.0422514379F, 0.2604829967F,  0.2286409289F,  0.7056031823F};
constexpr std::array<float, kLatentChannels> kLatentStd = {
    1.2223774195F, 1.2767263651F, 1.6831774712F, 1.7549455166F, 1.5636216402F, 2.1941435337F,
    0.9653137922F, 1.0569885969F, 0.8419489264F, 0.7729952931F, 1.8955937624F, 0.9468418360F,
    0.7996809483F, 0.4498890042F, 0.7197399735F, 0.6936293244F, 2.9610950947F, 2.7694199085F,
    3.0496184826F, 2.1088054180F, 3.2762262821F, 3.1627357006F, 2.2816812992F, 2.6127843857F};
constexpr std::array<float, 3> kPixelMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kPixelStd = {0.229F, 0.224F, 0.225F};

constexpr std::size_t kVideoLatentCount =
    static_cast<std::size_t>(kLatentChannels) * kLatentFrames * kLatentHeight * kLatentWidth;
constexpr std::size_t kAudioCount = static_cast<std::size_t>(kAudioRows) * kAudioChannels;

struct RawTensor {
    std::vector<std::byte> bytes;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
};

struct StepModulation {
    std::array<RawTensor, kLayers> blocks;
    RawTensor final;
};

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

const Tensor& require_output(const TensorMap& outputs, const std::string& name) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.data == nullptr)
        throw std::runtime_error("MiniMax-H3 engine did not return " + name);
    return it->second;
}

RawTensor copy_raw(const Tensor& tensor, DType expected_dtype, std::size_t expected_numel,
                   const char* label) {
    if (tensor.dtype != expected_dtype || tensor.numel() != expected_numel)
        throw std::runtime_error(std::string("MiniMax-H3 invalid ") + label + " output");
    RawTensor result;
    result.shape = tensor.shape;
    result.dtype = tensor.dtype;
    result.bytes.resize(tensor.nbytes());
    std::memcpy(result.bytes.data(), tensor.data, tensor.nbytes());
    return result;
}

std::vector<float> copy_float(const Tensor& tensor, std::size_t expected_numel, const char* label) {
    if (tensor.dtype != DType::kFloat32 || tensor.numel() != expected_numel)
        throw std::runtime_error(std::string("MiniMax-H3 invalid ") + label + " output");
    const auto* begin = static_cast<const float*>(tensor.data);
    return std::vector<float>(begin, begin + expected_numel);
}

std::array<float, 256> timestep_features(float timestep) {
    std::array<float, 256> output{};
    for (int32_t index = 0; index < 128; ++index) {
        const double frequency = std::exp(-std::log(10000.0) * index / 128.0);
        const double phase = static_cast<double>(timestep) * frequency;
        output[index] = static_cast<float>(std::cos(phase));
        output[128 + index] = static_cast<float>(std::sin(phase));
    }
    return output;
}

std::vector<float> make_adaln_features(float video_timestep, float audio_timestep) {
    std::vector<float> result(kTimestepSlots * 256, 0.0F);
    const auto video = timestep_features(video_timestep);
    const auto audio = timestep_features(audio_timestep);
    std::copy(video.begin(), video.end(), result.begin());
    std::copy(audio.begin(), audio.end(), result.begin() + 256);
    return result;
}

std::vector<float> patchify_video(const std::vector<float>& latent) {
    if (latent.size() != kVideoLatentCount)
        throw std::invalid_argument("MiniMax-H3 video latent count is invalid");
    std::vector<float> rows(static_cast<std::size_t>(kVideoRows) * kPatchDim);
    std::size_t target = 0;
    for (int32_t frame = 0; frame < kLatentFrames; ++frame) {
        for (int32_t y = 0; y < kLatentHeight; y += kPatchHeight) {
            for (int32_t x = 0; x < kLatentWidth; x += kPatchWidth) {
                for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                    for (int32_t py = 0; py < kPatchHeight; ++py) {
                        for (int32_t px = 0; px < kPatchWidth; ++px) {
                            const auto source =
                                ((((static_cast<std::size_t>(channel) * kLatentFrames + frame) *
                                       kLatentHeight +
                                   y + py) *
                                  kLatentWidth) +
                                 x + px);
                            rows[target++] = latent[source];
                        }
                    }
                }
            }
        }
    }
    return rows;
}

std::vector<float> unpatchify_video(const std::vector<float>& rows) {
    if (rows.size() != static_cast<std::size_t>(kVideoRows) * kPatchDim)
        throw std::invalid_argument("MiniMax-H3 video rows are invalid");
    std::vector<float> latent(kVideoLatentCount);
    std::size_t source = 0;
    for (int32_t frame = 0; frame < kLatentFrames; ++frame) {
        for (int32_t y = 0; y < kLatentHeight; y += kPatchHeight) {
            for (int32_t x = 0; x < kLatentWidth; x += kPatchWidth) {
                for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                    for (int32_t py = 0; py < kPatchHeight; ++py) {
                        for (int32_t px = 0; px < kPatchWidth; ++px) {
                            const auto target =
                                ((((static_cast<std::size_t>(channel) * kLatentFrames + frame) *
                                       kLatentHeight +
                                   y + py) *
                                  kLatentWidth) +
                                 x + px);
                            latent[target] = rows[source++];
                        }
                    }
                }
            }
        }
    }
    return latent;
}

std::vector<float> make_position_ids() {
    std::vector<float> positions(static_cast<std::size_t>(kSequenceRows) * 3, 0.0F);
    for (int32_t index = 0; index < kTextRows; ++index)
        positions[static_cast<std::size_t>(index) * 3] = static_cast<float>(index);

    const double sqrt_area = std::sqrt(static_cast<double>(kLatentHeight * kLatentWidth));
    const double height_ratio = kLatentHeight / sqrt_area;
    const double width_ratio = kLatentWidth / sqrt_area;
    std::array<double, kLatentHeight / kPatchHeight> height_grid{};
    std::array<double, kLatentWidth / kPatchWidth> width_grid{};
    const double height_left = (1.0 - height_ratio) / 2.0;
    const double width_left = (1.0 - width_ratio) / 2.0;
    for (std::size_t i = 0; i < height_grid.size(); ++i)
        height_grid[i] =
            (height_left + static_cast<double>(i) * height_ratio / height_grid.size()) * 32.0;
    for (std::size_t i = 0; i < width_grid.size(); ++i)
        width_grid[i] =
            (width_left + static_cast<double>(i) * width_ratio / width_grid.size()) * 32.0;

    for (int32_t channel = 0; channel < 2; ++channel) {
        for (int32_t index = 0; index < kAudioLatents; ++index) {
            const int32_t row = kTextRows + channel * kAudioLatents + index;
            positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(kTextRows + index);
            positions[static_cast<std::size_t>(row) * 3 + 2] =
                static_cast<float>(channel == 0 ? width_grid.front() : width_grid.back());
        }
    }

    double time = kTextRows;
    int32_t row = kTextRows + kAudioRows;
    for (int32_t frame = 0; frame < kLatentFrames; ++frame) {
        for (double y : height_grid) {
            for (double x : width_grid) {
                positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(time);
                positions[static_cast<std::size_t>(row) * 3 + 1] = static_cast<float>(y);
                positions[static_cast<std::size_t>(row) * 3 + 2] = static_cast<float>(x);
                ++row;
            }
        }
        const int32_t multiple = frame % 5 == 0 ? 1 : 4;
        time += (5.0 / 3.0) * multiple;
    }
    if (row != kSequenceRows)
        throw std::logic_error("MiniMax-H3 position row construction failed");
    return positions;
}

struct DenoiserMetadata {
    std::vector<float> positions;
    std::vector<int32_t> adaln_indices;
    std::vector<int32_t> timestep_indices;
};

DenoiserMetadata make_denoiser_metadata() {
    DenoiserMetadata result;
    result.positions = make_position_ids();
    result.adaln_indices.resize(kSequenceRows);
    result.timestep_indices.resize(kSequenceRows);
    for (int32_t row = 0; row < kSequenceRows; ++row) {
        int32_t tag = 0;
        int32_t timestep = 0;
        if (row < kTextRows) {
            tag = 1;
        } else if (row < kTextRows + kAudioRows) {
            tag = 2;
            timestep = 1;
        }
        result.timestep_indices[row] = timestep;
        result.adaln_indices[row] = timestep * kModalityCount + tag;
    }
    return result;
}

std::vector<StepModulation> precompute_modulations(ITrtModule& module,
                                                   const MiniMaxH3Schedule& video_schedule,
                                                   const MiniMaxH3Schedule& audio_schedule) {
    std::vector<StepModulation> result(video_schedule.timesteps.size());
    for (std::size_t step = 0; step < result.size(); ++step) {
        auto features =
            make_adaln_features(video_schedule.timesteps[step], audio_schedule.timesteps[step]);
        TensorMap inputs;
        inputs.emplace("timestep_features",
                       Tensor{features.data(), {kTimestepSlots, 256}, DType::kFloat32});
        const auto outputs = module.forward(inputs);
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            const std::string name = "block_modulation_" + std::to_string(layer);
            result[step].blocks[layer] =
                copy_raw(require_output(outputs, name), DType::kBFloat16,
                         static_cast<std::size_t>(kAdalnRows) * 6 * kHidden, name.c_str());
        }
        result[step].final =
            copy_raw(require_output(outputs, "final_modulation"), DType::kBFloat16,
                     static_cast<std::size_t>(kTimestepSlots) * 2 * kHidden, "final_modulation");
    }
    return result;
}

void append_modulation_inputs(TensorMap& inputs, StepModulation& modulation) {
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::string name = "block_modulation_" + std::to_string(layer);
        auto& value = modulation.blocks[layer];
        inputs.emplace(name, Tensor{value.bytes.data(), value.shape, value.dtype});
    }
    inputs.emplace("final_modulation", Tensor{modulation.final.bytes.data(), modulation.final.shape,
                                              modulation.final.dtype});
}

void denormalize_latents(std::vector<float>& latent) {
    const std::size_t per_channel =
        static_cast<std::size_t>(kLatentFrames) * kLatentHeight * kLatentWidth;
    for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
        float* values = latent.data() + static_cast<std::size_t>(channel) * per_channel;
        for (std::size_t index = 0; index < per_channel; ++index)
            values[index] = values[index] * kLatentStd[channel] + kLatentMean[channel];
    }
}

std::vector<float> extract_tiles(const std::vector<float>& latent, int32_t clip) {
    constexpr std::array<int32_t, 4> y_starts = {0, 10, 21, 32};
    constexpr std::array<int32_t, 7> x_starts = {0, 11, 22, 33, 44, 56, 68};
    const std::size_t one_tile = static_cast<std::size_t>(kLatentChannels) * kTileInputFrames *
                                 kTileLatentSize * kTileLatentSize;
    std::vector<float> result(static_cast<std::size_t>(kTileBatch) * one_tile);
    for (int32_t tile = 0; tile < kTileBatch; ++tile) {
        const int32_t tile_y = tile / 7;
        const int32_t tile_x = tile % 7;
        for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
            for (int32_t frame = 0; frame < kTileInputFrames; ++frame) {
                for (int32_t y = 0; y < kTileLatentSize; ++y) {
                    const auto source =
                        ((((static_cast<std::size_t>(channel) * kLatentFrames + clip * 5 + frame) *
                               kLatentHeight +
                           y_starts[tile_y] + y) *
                          kLatentWidth) +
                         x_starts[tile_x]);
                    const auto target =
                        ((((static_cast<std::size_t>(tile) * kLatentChannels + channel) *
                               kTileInputFrames +
                           frame) *
                              kTileLatentSize +
                          y) *
                         kTileLatentSize);
                    std::copy_n(latent.begin() + static_cast<std::ptrdiff_t>(source),
                                kTileLatentSize,
                                result.begin() + static_cast<std::ptrdiff_t>(target));
                }
            }
        }
    }
    return result;
}

std::vector<float> stitch_spatial_tiles(const std::vector<float>& tiles) {
    constexpr std::array<int32_t, 3> height_overlaps = {96, 80, 80};
    constexpr std::array<int32_t, 6> width_overlaps = {80, 80, 80, 80, 64, 64};
    constexpr std::array<int32_t, 4> output_y = {0, 160, 336, 512};
    constexpr std::array<int32_t, 7> output_x = {0, 176, 352, 528, 704, 896, 1088};
    const std::size_t one_tile = static_cast<std::size_t>(3) * kTileFrames * kTileSize * kTileSize;
    if (tiles.size() != static_cast<std::size_t>(kTileCount) * one_tile)
        throw std::runtime_error("MiniMax-H3 decoded VAE tile count is invalid");
    std::vector<float> clip(static_cast<std::size_t>(3) * kTileFrames * kOutputHeight *
                            kOutputWidth);
    const auto tile_value = [&](int32_t tile, int32_t channel, int32_t frame, int32_t y,
                                int32_t x) {
        return tiles[((((static_cast<std::size_t>(tile) * 3 + channel) * kTileFrames + frame) *
                           kTileSize +
                       y) *
                      kTileSize) +
                     x];
    };
    for (int32_t tile_y = 0; tile_y < 4; ++tile_y) {
        const int32_t kept_height = tile_y < 3 ? kTileSize - height_overlaps[tile_y] : kTileSize;
        for (int32_t tile_x = 0; tile_x < 7; ++tile_x) {
            const int32_t kept_width = tile_x < 6 ? kTileSize - width_overlaps[tile_x] : kTileSize;
            const int32_t tile = tile_y * 7 + tile_x;
            for (int32_t channel = 0; channel < 3; ++channel) {
                for (int32_t frame = 0; frame < kTileFrames; ++frame) {
                    for (int32_t y = 0; y < kept_height; ++y) {
                        for (int32_t x = 0; x < kept_width; ++x) {
                            float value = tile_value(tile, channel, frame, y, x);
                            if (tile_y > 0 && y < height_overlaps[tile_y - 1]) {
                                const int32_t overlap = height_overlaps[tile_y - 1];
                                const float weight_b = static_cast<float>(y) / overlap;
                                const float upper = tile_value(tile - 7, channel, frame,
                                                               kTileSize - overlap + y, x);
                                value = upper * (1.0F - weight_b) + value * weight_b;
                            }
                            if (tile_x > 0 && x < width_overlaps[tile_x - 1]) {
                                const int32_t overlap = width_overlaps[tile_x - 1];
                                const float weight_b = static_cast<float>(x) / overlap;
                                const float left = tile_value(tile - 1, channel, frame, y,
                                                              kTileSize - overlap + x);
                                value = left * (1.0F - weight_b) + value * weight_b;
                            }
                            const auto target =
                                ((((static_cast<std::size_t>(channel) * kTileFrames + frame) *
                                       kOutputHeight +
                                   output_y[tile_y] + y) *
                                  kOutputWidth) +
                                 output_x[tile_x] + x);
                            clip[target] = value;
                        }
                    }
                }
            }
        }
    }
    return clip;
}

void append_temporal_chunk(std::vector<float>& video, const std::vector<float>& clip,
                           const std::vector<float>& previous_overlap) {
    constexpr int32_t chunk_frames = 17;
    constexpr int32_t pre_padding = 3;
    constexpr int32_t overlap_frames = 5;
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    const std::size_t old_frames = video.size() / (3 * plane);
    std::vector<float> expanded(static_cast<std::size_t>(3) * (old_frames + chunk_frames) * plane);
    for (int32_t channel = 0; channel < 3; ++channel) {
        if (old_frames > 0) {
            std::copy_n(video.begin() + static_cast<std::ptrdiff_t>(channel * old_frames * plane),
                        old_frames * plane,
                        expanded.begin() + static_cast<std::ptrdiff_t>(
                                               channel * (old_frames + chunk_frames) * plane));
        }
        for (int32_t frame = 0; frame < chunk_frames; ++frame) {
            const auto source =
                (static_cast<std::size_t>(channel) * kTileFrames + pre_padding + frame) * plane;
            const auto target = (static_cast<std::size_t>(channel) * (old_frames + chunk_frames) +
                                 old_frames + frame) *
                                plane;
            if (!previous_overlap.empty() && frame < overlap_frames) {
                const float weight_b = static_cast<float>(frame) / overlap_frames;
                const auto prior =
                    (static_cast<std::size_t>(channel) * overlap_frames + frame) * plane;
                for (std::size_t pixel = 0; pixel < plane; ++pixel)
                    expanded[target + pixel] = previous_overlap[prior + pixel] * (1.0F - weight_b) +
                                               clip[source + pixel] * weight_b;
            } else {
                std::copy_n(clip.begin() + static_cast<std::ptrdiff_t>(source), plane,
                            expanded.begin() + static_cast<std::ptrdiff_t>(target));
            }
        }
    }
    video.swap(expanded);
}

std::vector<float> trailing_overlap(const std::vector<float>& clip) {
    constexpr int32_t overlap_frames = 5;
    constexpr int32_t start = 23;
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    std::vector<float> result(static_cast<std::size_t>(3) * overlap_frames * plane);
    for (int32_t channel = 0; channel < 3; ++channel) {
        const auto source = (static_cast<std::size_t>(channel) * kTileFrames + start) * plane;
        const auto target = static_cast<std::size_t>(channel) * overlap_frames * plane;
        std::copy_n(clip.begin() + static_cast<std::ptrdiff_t>(source), overlap_frames * plane,
                    result.begin() + static_cast<std::ptrdiff_t>(target));
    }
    return result;
}

void append_final_overlap(std::vector<float>& video, const std::vector<float>& overlap) {
    constexpr int32_t overlap_frames = 5;
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    const std::size_t old_frames = video.size() / (3 * plane);
    std::vector<float> expanded(static_cast<std::size_t>(3) * (old_frames + overlap_frames) *
                                plane);
    for (int32_t channel = 0; channel < 3; ++channel) {
        std::copy_n(video.begin() + static_cast<std::ptrdiff_t>(channel * old_frames * plane),
                    old_frames * plane,
                    expanded.begin() + static_cast<std::ptrdiff_t>(
                                           channel * (old_frames + overlap_frames) * plane));
        std::copy_n(overlap.begin() + static_cast<std::ptrdiff_t>(channel * overlap_frames * plane),
                    overlap_frames * plane,
                    expanded.begin() +
                        static_cast<std::ptrdiff_t>(
                            (channel * (old_frames + overlap_frames) + old_frames) * plane));
    }
    video.swap(expanded);
}

void postprocess_video(std::vector<float>& video) {
    const std::size_t per_channel =
        static_cast<std::size_t>(kOutputFrames) * kOutputHeight * kOutputWidth;
    for (int32_t channel = 0; channel < 3; ++channel) {
        float* values = video.data() + static_cast<std::size_t>(channel) * per_channel;
        for (std::size_t index = 0; index < per_channel; ++index)
            values[index] =
                std::clamp(values[index] * kPixelStd[channel] + kPixelMean[channel], 0.0F, 1.0F);
    }
}

std::vector<float> to_frame_major_rgb(const std::vector<float>& video) {
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    const std::size_t per_channel = static_cast<std::size_t>(kOutputFrames) * plane;
    std::vector<float> pixels(static_cast<std::size_t>(kOutputFrames) * plane * 3);
    for (int32_t frame = 0; frame < kOutputFrames; ++frame) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const auto target = (static_cast<std::size_t>(frame) * plane + pixel) * 3;
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto source = static_cast<std::size_t>(channel) * per_channel +
                                    static_cast<std::size_t>(frame) * plane + pixel;
                pixels[target + channel] = video[source];
            }
        }
    }
    return pixels;
}

} // namespace

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift) {
    if (grid_points < 2 || shift <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 schedule arguments are invalid");
    MiniMaxH3Schedule result;
    result.sigmas.reserve(grid_points);
    for (int32_t index = 0; index < grid_points; ++index) {
        const float base = static_cast<float>(1.0 - static_cast<double>(index) / (grid_points - 1));
        const float sigma = shift * base / (1.0F + (shift - 1.0F) * base);
        if (result.sigmas.empty() || sigma != result.sigmas.back())
            result.sigmas.push_back(sigma);
    }
    if (result.sigmas.size() < 2 || result.sigmas.back() != 0.0F)
        throw std::runtime_error("MiniMax-H3 sigma grid collapsed unexpectedly");
    result.timesteps.reserve(result.sigmas.size() - 1);
    for (std::size_t index = 0; index + 1 < result.sigmas.size(); ++index)
        result.timesteps.push_back(1.0F - result.sigmas[index]);
    return result;
}

void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next) {
    if (sample == nullptr || velocity == nullptr || !(sigma > 0.0F))
        throw std::invalid_argument("MiniMax-H3 scheduler received invalid inputs");
    const float sigma_from_timestep = 1.0F - timestep;
    const float ratio = sigma_next / sigma;
    for (std::size_t index = 0; index < count; ++index) {
        const float denoised = sample[index] + sigma_from_timestep * velocity[index];
        sample[index] = ratio * sample[index] + (1.0F - ratio) * denoised;
    }
}

MiniMaxH3Pipeline::MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader,
                                     std::unique_ptr<ITokenizer> tokenizer, std::string model_id)
    : loader_(std::move(loader)), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)) {
    if (!loader_ || !tokenizer_)
        throw std::invalid_argument("MiniMax-H3 pipeline requires a loader and tokenizer");
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to create its CUDA stream");
}

MiniMaxH3Pipeline::~MiniMaxH3Pipeline() {
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

ImageResult MiniMaxH3Pipeline::generate_image(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    if ((cfg.height > 0 && cfg.height != kOutputHeight) ||
        (cfg.width > 0 && cfg.width != kOutputWidth) ||
        (cfg.num_steps > 0 && cfg.num_steps != kSteps))
        throw std::invalid_argument(
            "MiniMax-H3 native profile is fixed at 124 frames, 768x1344, 50 grid points");
    const int64_t seed = cfg.seed >= 0 ? cfg.seed : 0;
    const auto total_begin = Clock::now();

    const auto ids = tokenizer_->encode(prompt);
    if (ids.size() != kTextRows)
        throw std::invalid_argument(
            "MiniMax-H3 GB300 profile requires exactly 537 prompt tokens; got " +
            std::to_string(ids.size()));
    std::vector<float> text_embeddings;
    const auto text_begin = Clock::now();
    {
        auto module = loader_("text_encoder_plan", stream_);
        TensorMap inputs;
        inputs.emplace("input_ids",
                       Tensor{const_cast<int32_t*>(ids.data()), {kTextRows}, DType::kInt32});
        const auto outputs = module->forward(inputs);
        text_embeddings =
            copy_float(require_output(outputs, "encoder_hidden_states"),
                       static_cast<std::size_t>(kTextRows) * kTextDim, "text encoder");
        module->sync();
    }
    const auto text_end = Clock::now();

    const auto video_schedule = make_minimax_h3_schedule(kSteps, 12.0F);
    const auto audio_schedule = make_minimax_h3_schedule(kSteps, 3.0F);
    std::vector<StepModulation> modulations;
    const auto adaln_begin = Clock::now();
    {
        auto module = loader_("adaln_precompute_plan", stream_);
        modulations = precompute_modulations(*module, video_schedule, audio_schedule);
        module->sync();
    }
    const auto adaln_end = Clock::now();

    auto video_tensor =
        minimax_h3::torch_cuda_normal(kVideoLatentCount, static_cast<uint64_t>(seed));
    const auto audio_offset = minimax_h3::torch_cuda_normal_consumed_offset(kVideoLatentCount);
    auto audio_rows =
        minimax_h3::torch_cuda_normal(kAudioCount, static_cast<uint64_t>(seed), audio_offset);
    auto video_rows = patchify_video(video_tensor);
    video_tensor.clear();
    video_tensor.shrink_to_fit();
    auto metadata = make_denoiser_metadata();

    const auto denoiser_begin = Clock::now();
    {
        auto module = loader_("denoiser_plan", stream_);
        for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
            TensorMap inputs;
            inputs.emplace("video_hidden_states",
                           Tensor{video_rows.data(), {kVideoRows, kPatchDim}, DType::kFloat32});
            inputs.emplace(
                "audio_hidden_states",
                Tensor{audio_rows.data(), {kAudioRows, kAudioChannels}, DType::kFloat32});
            inputs.emplace("encoder_hidden_states",
                           Tensor{text_embeddings.data(), {kTextRows, kTextDim}, DType::kFloat32});
            inputs.emplace("position_ids",
                           Tensor{metadata.positions.data(), {kSequenceRows, 3}, DType::kFloat32});
            inputs.emplace("adaln_indices",
                           Tensor{metadata.adaln_indices.data(), {kSequenceRows}, DType::kInt32});
            inputs.emplace(
                "timestep_indices",
                Tensor{metadata.timestep_indices.data(), {kSequenceRows}, DType::kInt32});
            append_modulation_inputs(inputs, modulations[step]);
            const auto outputs = module->forward(inputs);
            auto video_all =
                copy_float(require_output(outputs, "video_velocity"),
                           static_cast<std::size_t>(kSequenceRows) * kPatchDim, "video velocity");
            auto audio_all = copy_float(require_output(outputs, "audio_velocity"),
                                        static_cast<std::size_t>(kSequenceRows) * kAudioChannels,
                                        "audio velocity");
            const auto video_begin =
                video_all.begin() + static_cast<std::ptrdiff_t>(kTextRows + kAudioRows) * kPatchDim;
            std::vector<float> video_velocity(video_begin, video_begin + video_rows.size());
            const auto audio_begin =
                audio_all.begin() + static_cast<std::ptrdiff_t>(kTextRows) * kAudioChannels;
            std::vector<float> audio_velocity(audio_begin, audio_begin + audio_rows.size());
            minimax_h3_scheduler_step(video_rows.data(), video_velocity.data(), video_rows.size(),
                                      video_schedule.timesteps[step], video_schedule.sigmas[step],
                                      video_schedule.sigmas[step + 1]);
            minimax_h3_scheduler_step(audio_rows.data(), audio_velocity.data(), audio_rows.size(),
                                      audio_schedule.timesteps[step], audio_schedule.sigmas[step],
                                      audio_schedule.sigmas[step + 1]);
            std::cerr << "[minimax-h3] denoiser " << (step + 1) << '/'
                      << video_schedule.timesteps.size() << '\n';
        }
        module->sync();
    }
    const auto denoiser_end = Clock::now();
    modulations.clear();
    modulations.shrink_to_fit();
    text_embeddings.clear();
    text_embeddings.shrink_to_fit();
    audio_rows.clear();
    audio_rows.shrink_to_fit();

    auto latent = unpatchify_video(video_rows);
    video_rows.clear();
    video_rows.shrink_to_fit();
    denormalize_latents(latent);
    std::vector<float> video;
    std::vector<float> overlap;
    const auto vae_begin = Clock::now();
    {
        auto module = loader_("vae_tile_decoder_plan", stream_);
        constexpr std::size_t output_count =
            static_cast<std::size_t>(kTileBatch) * 3 * kTileFrames * kTileSize * kTileSize;
        for (int32_t clip_index = 0; clip_index < 7; ++clip_index) {
            auto latent_tiles = extract_tiles(latent, clip_index);
            TensorMap inputs;
            inputs.emplace("latent_tiles", Tensor{latent_tiles.data(),
                                                  {kTileBatch, kLatentChannels, kTileInputFrames,
                                                   kTileLatentSize, kTileLatentSize},
                                                  DType::kFloat32});
            const auto outputs = module->forward(inputs);
            auto decoded_tiles = copy_float(require_output(outputs, "decoded_tiles"), output_count,
                                            "VAE decoded tiles");
            auto clip = stitch_spatial_tiles(decoded_tiles);
            append_temporal_chunk(video, clip, overlap);
            overlap = trailing_overlap(clip);
            std::cerr << "[minimax-h3] VAE clip " << (clip_index + 1) << "/7\n";
        }
        module->sync();
    }
    const auto vae_end = Clock::now();
    latent.clear();
    latent.shrink_to_fit();
    append_final_overlap(video, overlap);
    const std::size_t expected_pixels =
        static_cast<std::size_t>(3) * kOutputFrames * kOutputHeight * kOutputWidth;
    if (video.size() != expected_pixels)
        throw std::runtime_error("MiniMax-H3 VAE produced the wrong video geometry");
    postprocess_video(video);
    auto pixels = to_frame_major_rgb(video);
    video.clear();
    video.shrink_to_fit();

    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[minimax-h3.perf] text_encoder_ms=" << milliseconds(text_begin, text_end)
              << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
              << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " total_ms=" << milliseconds(total_begin, total_end) << '\n';
    ImageResult result;
    result.height = kOutputHeight;
    result.width = kOutputWidth;
    result.channels = 3;
    result.num_frames = kOutputFrames;
    result.pixels = std::move(pixels);
    return result;
}

} // namespace trtmc
