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
#include <initializer_list>
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

constexpr std::array<int32_t, 3> kTileHeightOverlaps = {96, 80, 80};
constexpr std::array<int32_t, 6> kTileWidthOverlaps = {80, 80, 80, 80, 64, 64};
constexpr std::array<int32_t, 4> kTileOutputY = {0, 160, 336, 512};
constexpr std::array<int32_t, 7> kTileOutputX = {0, 176, 352, 528, 704, 896, 1088};

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

minimax_h3::VaeLatentNormalization vae_latent_normalization() {
    minimax_h3::VaeLatentNormalization result{};
    std::copy(kLatentMean.begin(), kLatentMean.end(), result.mean);
    std::copy(kLatentStd.begin(), kLatentStd.end(), result.std);
    return result;
}

minimax_h3::VaePixelNormalization vae_pixel_normalization() {
    minimax_h3::VaePixelNormalization result{};
    std::copy(kPixelMean.begin(), kPixelMean.end(), result.mean);
    std::copy(kPixelStd.begin(), kPixelStd.end(), result.std);
    return result;
}

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

void fill_audio_position_ids(std::vector<float>& positions,
                             const std::array<double, kLatentWidth / kPatchWidth>& width_grid) {
    for (int32_t channel = 0; channel < 2; ++channel) {
        for (int32_t index = 0; index < kAudioLatents; ++index) {
            const int32_t row = kTextRows + channel * kAudioLatents + index;
            positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(kTextRows + index);
            positions[static_cast<std::size_t>(row) * 3 + 2] =
                static_cast<float>(channel == 0 ? width_grid.front() : width_grid.back());
        }
    }
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

    fill_audio_position_ids(positions, width_grid);

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

void append_block_modulation_inputs(TensorMap& inputs, StepModulation& modulation,
                                    int32_t first_layer, int32_t end_layer) {
    for (int32_t layer = first_layer; layer < end_layer; ++layer) {
        const std::string name = "block_modulation_" + std::to_string(layer);
        auto& value = modulation.blocks[layer];
        inputs.emplace(name, Tensor{value.bytes.data(), value.shape, value.dtype});
    }
}

void append_final_modulation_input(TensorMap& inputs, StepModulation& modulation) {
    inputs.emplace("final_modulation", Tensor{modulation.final.bytes.data(), modulation.final.shape,
                                              modulation.final.dtype});
}

void bind_external_checked(ITrtModule& module, const char* name, void* pointer, bool is_input,
                           DType dtype, std::initializer_list<int64_t> shape) {
    const bool direction_matches = is_input ? module.has_input(name) : module.has_output(name);
    const std::vector<int64_t> expected_shape(shape);
    if (pointer == nullptr || !direction_matches || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != expected_shape)
        throw std::runtime_error(std::string("MiniMax-H3 split plan ABI mismatch for ") + name);
    module.bind_external(name, pointer);
    if (module.device_ptr(name) != pointer)
        throw std::runtime_error(std::string("MiniMax-H3 external binding failed for ") + name);
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

void stitch_one_spatial_tile(const float* tiles, std::vector<float>& clip, int32_t tile_y,
                             int32_t tile_x, int32_t kept_height, int32_t kept_width) {
    const int32_t tile = tile_y * 7 + tile_x;
    const auto tile_value = [&](int32_t source_tile, int32_t channel, int32_t frame, int32_t y,
                                int32_t x) {
        return tiles[((((static_cast<std::size_t>(source_tile) * 3 + channel) * kTileFrames +
                        frame) *
                           kTileSize +
                       y) *
                      kTileSize) +
                     x];
    };
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (int32_t frame = 0; frame < kTileFrames; ++frame) {
            for (int32_t y = 0; y < kept_height; ++y) {
                for (int32_t x = 0; x < kept_width; ++x) {
                    float value = tile_value(tile, channel, frame, y, x);
                    if (tile_y > 0 && y < kTileHeightOverlaps[tile_y - 1]) {
                        const int32_t overlap = kTileHeightOverlaps[tile_y - 1];
                        const float weight_b = static_cast<float>(y) / overlap;
                        const float upper =
                            tile_value(tile - 7, channel, frame, kTileSize - overlap + y, x);
                        value = upper * (1.0F - weight_b) + value * weight_b;
                    }
                    if (tile_x > 0 && x < kTileWidthOverlaps[tile_x - 1]) {
                        const int32_t overlap = kTileWidthOverlaps[tile_x - 1];
                        const float weight_b = static_cast<float>(x) / overlap;
                        const float left =
                            tile_value(tile - 1, channel, frame, y, kTileSize - overlap + x);
                        value = left * (1.0F - weight_b) + value * weight_b;
                    }
                    const auto target =
                        ((((static_cast<std::size_t>(channel) * kTileFrames + frame) *
                               kOutputHeight +
                           kTileOutputY[tile_y] + y) *
                          kOutputWidth) +
                         kTileOutputX[tile_x] + x);
                    clip[target] = value;
                }
            }
        }
    }
}

void stitch_spatial_tiles(const Tensor& tiles, std::vector<float>& clip) {
    const std::size_t one_tile = static_cast<std::size_t>(3) * kTileFrames * kTileSize * kTileSize;
    if (tiles.dtype != DType::kFloat32 || tiles.data == nullptr ||
        tiles.numel() != static_cast<std::size_t>(kTileCount) * one_tile)
        throw std::runtime_error("MiniMax-H3 decoded VAE tile count is invalid");
    const auto* values = static_cast<const float*>(tiles.data);
    clip.resize(static_cast<std::size_t>(3) * kTileFrames * kOutputHeight * kOutputWidth);
    for (int32_t tile_y = 0; tile_y < 4; ++tile_y) {
        const int32_t kept_height =
            tile_y < 3 ? kTileSize - kTileHeightOverlaps[tile_y] : kTileSize;
        for (int32_t tile_x = 0; tile_x < 7; ++tile_x) {
            const int32_t kept_width =
                tile_x < 6 ? kTileSize - kTileWidthOverlaps[tile_x] : kTileSize;
            stitch_one_spatial_tile(values, clip, tile_y, tile_x, kept_height, kept_width);
        }
    }
}

void write_temporal_chunk(std::vector<float>& video, std::size_t old_frames,
                          const std::vector<float>& clip,
                          const std::vector<float>& previous_overlap) {
    constexpr int32_t chunk_frames = 17;
    constexpr int32_t pre_padding = 3;
    constexpr int32_t overlap_frames = 5;
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    if (video.size() != static_cast<std::size_t>(3) * kOutputFrames * plane ||
        old_frames + chunk_frames > kOutputFrames)
        throw std::invalid_argument("MiniMax-H3 temporal output buffer is invalid");
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (int32_t frame = 0; frame < chunk_frames; ++frame) {
            const auto source =
                (static_cast<std::size_t>(channel) * kTileFrames + pre_padding + frame) * plane;
            const auto target = (static_cast<std::size_t>(channel) * kOutputFrames + old_frames +
                                 static_cast<std::size_t>(frame)) *
                                plane;
            if (!previous_overlap.empty() && frame < overlap_frames) {
                const float weight_b = static_cast<float>(frame) / overlap_frames;
                const auto prior =
                    (static_cast<std::size_t>(channel) * overlap_frames + frame) * plane;
                for (std::size_t pixel = 0; pixel < plane; ++pixel)
                    video[target + pixel] = previous_overlap[prior + pixel] * (1.0F - weight_b) +
                                            clip[source + pixel] * weight_b;
            } else {
                std::copy_n(clip.begin() + static_cast<std::ptrdiff_t>(source), plane,
                            video.begin() + static_cast<std::ptrdiff_t>(target));
            }
        }
    }
}

void update_trailing_overlap(const std::vector<float>& clip, std::vector<float>& result) {
    constexpr int32_t overlap_frames = 5;
    constexpr int32_t start = 23;
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    result.resize(static_cast<std::size_t>(3) * overlap_frames * plane);
    for (int32_t channel = 0; channel < 3; ++channel) {
        const auto source = (static_cast<std::size_t>(channel) * kTileFrames + start) * plane;
        const auto target = static_cast<std::size_t>(channel) * overlap_frames * plane;
        std::copy_n(clip.begin() + static_cast<std::ptrdiff_t>(source), overlap_frames * plane,
                    result.begin() + static_cast<std::ptrdiff_t>(target));
    }
}

void write_final_overlap(std::vector<float>& video, std::size_t old_frames,
                         const std::vector<float>& overlap) {
    constexpr int32_t overlap_frames = 5;
    const std::size_t plane = static_cast<std::size_t>(kOutputHeight) * kOutputWidth;
    if (video.size() != static_cast<std::size_t>(3) * kOutputFrames * plane ||
        old_frames + overlap_frames != kOutputFrames ||
        overlap.size() != static_cast<std::size_t>(3) * overlap_frames * plane)
        throw std::invalid_argument("MiniMax-H3 final temporal overlap is invalid");
    for (int32_t channel = 0; channel < 3; ++channel) {
        std::copy_n(overlap.begin() + static_cast<std::ptrdiff_t>(channel * overlap_frames * plane),
                    overlap_frames * plane,
                    video.begin() + static_cast<std::ptrdiff_t>(
                                        (channel * kOutputFrames + old_frames) * plane));
    }
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

void validate_generate_config(const GenerateConfig& cfg) {
    if ((cfg.height > 0 && cfg.height != kOutputHeight) ||
        (cfg.width > 0 && cfg.width != kOutputWidth) ||
        (cfg.num_steps > 0 && cfg.num_steps != kSteps))
        throw std::invalid_argument(
            "MiniMax-H3 native profile is fixed at 124 frames, 768x1344, 50 grid points");
}

} // namespace

struct MiniMaxH3Pipeline::ResidentState {
    std::string prompt;
    std::vector<float> text_embeddings;
    std::vector<StepModulation> modulations;
    std::unique_ptr<DeviceTensor> head_hidden;
    std::unique_ptr<DeviceTensor> head_residual;
    std::unique_ptr<DeviceTensor> previous_head_residual;
    std::unique_ptr<DeviceTensor> tail_residual;
    std::unique_ptr<DeviceTensor> video_rows;
    std::unique_ptr<DeviceTensor> audio_rows;
    std::unique_ptr<DeviceTensor> video_velocity;
    std::unique_ptr<DeviceTensor> audio_velocity;
    std::unique_ptr<DeviceTensor> vae_latent_tiles;
    std::unique_ptr<DeviceTensor> vae_decoded_tiles;
    std::unique_ptr<DeviceTensor> vae_overlap;
    std::unique_ptr<DeviceTensor> frame_major_rgb;
    std::unique_ptr<ITrtModule> denoiser;
    std::unique_ptr<ITrtModule> denoiser_head;
    std::unique_ptr<ITrtModule> denoiser_tail;
    std::unique_ptr<ITrtModule> denoiser_finish;
    std::unique_ptr<ITrtModule> vae;
};

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
                                     std::unique_ptr<ITokenizer> tokenizer, std::string model_id,
                                     bool first_block_cache, float cache_threshold)
    : loader_(std::move(loader)), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)),
      resident_(std::make_unique<ResidentState>()), first_block_cache_(first_block_cache),
      cache_threshold_(cache_threshold) {
    if (!loader_ || !tokenizer_)
        throw std::invalid_argument("MiniMax-H3 pipeline requires a loader and tokenizer");
    if (!std::isfinite(cache_threshold_) || cache_threshold_ <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 cache threshold must be finite and positive");
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to create its CUDA stream");
}

MiniMaxH3Pipeline::~MiniMaxH3Pipeline() {
    resident_.reset();
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

ImageResult MiniMaxH3Pipeline::generate_image(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    validate_generate_config(cfg);
    const int64_t seed = cfg.seed >= 0 ? cfg.seed : 0;
    const auto total_begin = Clock::now();

    const bool text_cache_hit = resident_->prompt == prompt && !resident_->text_embeddings.empty();
    const auto text_begin = Clock::now();
    if (!text_cache_hit) {
        // The text encoder is the largest plan. Drop resident execution modules
        // before loading it so prompt changes retain the previous peak-memory
        // behavior on smaller devices.
        resident_->denoiser.reset();
        resident_->denoiser_head.reset();
        resident_->denoiser_tail.reset();
        resident_->denoiser_finish.reset();
        resident_->head_hidden.reset();
        resident_->head_residual.reset();
        resident_->previous_head_residual.reset();
        resident_->tail_residual.reset();
        resident_->video_rows.reset();
        resident_->audio_rows.reset();
        resident_->video_velocity.reset();
        resident_->audio_velocity.reset();
        resident_->vae.reset();
        resident_->vae_latent_tiles.reset();
        resident_->vae_decoded_tiles.reset();
        resident_->vae_overlap.reset();
        resident_->frame_major_rgb.reset();
        resident_->prompt.clear();
        resident_->text_embeddings.clear();
        const auto ids = tokenizer_->encode(prompt);
        if (ids.size() != kTextRows)
            throw std::invalid_argument(
                "MiniMax-H3 GB300 profile requires exactly 537 prompt tokens; got " +
                std::to_string(ids.size()));
        auto module = loader_("text_encoder_plan", stream_);
        module->set_timing_label("text_encoder_plan");
        TensorMap inputs;
        inputs.emplace("input_ids",
                       Tensor{const_cast<int32_t*>(ids.data()), {kTextRows}, DType::kInt32});
        const auto outputs = module->forward(inputs);
        resident_->text_embeddings =
            copy_float(require_output(outputs, "encoder_hidden_states"),
                       static_cast<std::size_t>(kTextRows) * kTextDim, "text encoder");
        module->sync();
        resident_->prompt = prompt;
    }
    const auto text_end = Clock::now();

    const auto video_schedule = make_minimax_h3_schedule(kSteps, 12.0F);
    const auto audio_schedule = make_minimax_h3_schedule(kSteps, 3.0F);
    const bool adaln_cache_hit = !resident_->modulations.empty();
    const auto adaln_begin = Clock::now();
    if (!adaln_cache_hit) {
        auto module = loader_("adaln_precompute_plan", stream_);
        module->set_timing_label("adaln_precompute_plan");
        resident_->modulations = precompute_modulations(*module, video_schedule, audio_schedule);
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
    const bool denoiser_resident_hit =
        first_block_cache_
            ? resident_->denoiser_head != nullptr && resident_->denoiser_tail != nullptr &&
                  resident_->denoiser_finish != nullptr && resident_->head_hidden != nullptr &&
                  resident_->head_hidden->ok() && resident_->head_residual != nullptr &&
                  resident_->head_residual->ok() && resident_->previous_head_residual != nullptr &&
                  resident_->previous_head_residual->ok() && resident_->tail_residual != nullptr &&
                  resident_->tail_residual->ok() && resident_->video_rows != nullptr &&
                  resident_->video_rows->ok() && resident_->audio_rows != nullptr &&
                  resident_->audio_rows->ok() && resident_->video_velocity != nullptr &&
                  resident_->video_velocity->ok() && resident_->audio_velocity != nullptr &&
                  resident_->audio_velocity->ok()
            : resident_->denoiser != nullptr;
    if (first_block_cache_ && !denoiser_resident_hit) {
        auto head = loader_("denoiser_head_plan", stream_);
        auto tail = loader_("denoiser_tail_plan", stream_);
        auto finish = loader_("denoiser_finish_plan", stream_);
        head->set_timing_label("denoiser_head_plan");
        tail->set_timing_label("denoiser_tail_plan");
        finish->set_timing_label("denoiser_finish_plan");

        DeviceTensor head_hidden({kSequenceRows, kHidden}, DType::kBFloat16, stream_);
        DeviceTensor head_residual({kSequenceRows, kHidden}, DType::kBFloat16, stream_);
        DeviceTensor previous_head_residual({kSequenceRows, kHidden}, DType::kBFloat16, stream_);
        DeviceTensor tail_residual({kSequenceRows, kHidden}, DType::kBFloat16, stream_);
        DeviceTensor video_rows_device({kVideoRows, kPatchDim}, DType::kFloat32, stream_);
        DeviceTensor audio_rows_device({kAudioRows, kAudioChannels}, DType::kFloat32, stream_);
        DeviceTensor video_velocity({kVideoRows, kPatchDim}, DType::kFloat32, stream_);
        DeviceTensor audio_velocity({kAudioRows, kAudioChannels}, DType::kFloat32, stream_);
        if (!head_hidden.ok() || !head_residual.ok() || !previous_head_residual.ok() ||
            !tail_residual.ok() || !video_rows_device.ok() || !audio_rows_device.ok() ||
            !video_velocity.ok() || !audio_velocity.ok())
            throw std::runtime_error("MiniMax-H3 failed to allocate FirstBlockCache buffers");

        auto resident_head_hidden = std::make_unique<DeviceTensor>(std::move(head_hidden));
        auto resident_head_residual = std::make_unique<DeviceTensor>(std::move(head_residual));
        auto resident_previous_head_residual =
            std::make_unique<DeviceTensor>(std::move(previous_head_residual));
        auto resident_tail_residual = std::make_unique<DeviceTensor>(std::move(tail_residual));
        auto resident_video_rows = std::make_unique<DeviceTensor>(std::move(video_rows_device));
        auto resident_audio_rows = std::make_unique<DeviceTensor>(std::move(audio_rows_device));
        auto resident_video_velocity = std::make_unique<DeviceTensor>(std::move(video_velocity));
        auto resident_audio_velocity = std::make_unique<DeviceTensor>(std::move(audio_velocity));

        bind_external_checked(*head, "head_hidden", resident_head_hidden->data(), false,
                              DType::kBFloat16, {kSequenceRows, kHidden});
        bind_external_checked(*head, "head_residual", resident_head_residual->data(), false,
                              DType::kBFloat16, {kSequenceRows, kHidden});
        bind_external_checked(*head, "previous_head_residual",
                              resident_previous_head_residual->data(), true, DType::kBFloat16,
                              {kSequenceRows, kHidden});
        bind_external_checked(*head, "video_hidden_states", resident_video_rows->data(), true,
                              DType::kFloat32, {kVideoRows, kPatchDim});
        bind_external_checked(*head, "audio_hidden_states", resident_audio_rows->data(), true,
                              DType::kFloat32, {kAudioRows, kAudioChannels});
        bind_external_checked(*tail, "head_hidden", resident_head_hidden->data(), true,
                              DType::kBFloat16, {kSequenceRows, kHidden});
        bind_external_checked(*tail, "tail_residual", resident_tail_residual->data(), false,
                              DType::kBFloat16, {kSequenceRows, kHidden});
        bind_external_checked(*finish, "head_hidden", resident_head_hidden->data(), true,
                              DType::kBFloat16, {kSequenceRows, kHidden});
        bind_external_checked(*finish, "tail_residual", resident_tail_residual->data(), true,
                              DType::kBFloat16, {kSequenceRows, kHidden});
        bind_external_checked(*finish, "video_velocity", resident_video_velocity->data(), false,
                              DType::kFloat32, {kVideoRows, kPatchDim});
        bind_external_checked(*finish, "audio_velocity", resident_audio_velocity->data(), false,
                              DType::kFloat32, {kAudioRows, kAudioChannels});

        resident_->denoiser_head = std::move(head);
        resident_->denoiser_tail = std::move(tail);
        resident_->denoiser_finish = std::move(finish);
        resident_->head_hidden = std::move(resident_head_hidden);
        resident_->head_residual = std::move(resident_head_residual);
        resident_->previous_head_residual = std::move(resident_previous_head_residual);
        resident_->tail_residual = std::move(resident_tail_residual);
        resident_->video_rows = std::move(resident_video_rows);
        resident_->audio_rows = std::move(resident_audio_rows);
        resident_->video_velocity = std::move(resident_video_velocity);
        resident_->audio_velocity = std::move(resident_audio_velocity);
    } else if (!first_block_cache_ && !resident_->denoiser) {
        resident_->denoiser = loader_("denoiser_plan", stream_);
        resident_->denoiser->set_timing_label("denoiser_plan");
    }
    int32_t full_denoiser_steps = 0;
    int32_t skipped_denoiser_steps = 0;
    if (first_block_cache_) {
        auto& head = *resident_->denoiser_head;
        auto& tail = *resident_->denoiser_tail;
        auto& finish = *resident_->denoiser_finish;
        head.reset_execution_context();
        tail.reset_execution_context();
        finish.reset_execution_context();
        if (cudaMemsetAsync(resident_->previous_head_residual->data(), 0,
                            resident_->previous_head_residual->nbytes(), stream_) != cudaSuccess)
            throw std::runtime_error("MiniMax-H3 failed to reset FirstBlockCache state");
        if (!resident_->video_rows->copy_from_host(video_rows.data()) ||
            !resident_->audio_rows->copy_from_host(audio_rows.data()))
            throw std::runtime_error("MiniMax-H3 failed to upload FirstBlockCache latents");

        for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
            auto& modulation = resident_->modulations[step];
            TensorMap head_inputs;
            head_inputs.emplace(
                "encoder_hidden_states",
                Tensor{resident_->text_embeddings.data(), {kTextRows, kTextDim}, DType::kFloat32});
            head_inputs.emplace(
                "position_ids",
                Tensor{metadata.positions.data(), {kSequenceRows, 3}, DType::kFloat32});
            head_inputs.emplace(
                "adaln_indices",
                Tensor{metadata.adaln_indices.data(), {kSequenceRows}, DType::kInt32});
            append_block_modulation_inputs(head_inputs, modulation, 0, 1);
            const auto head_outputs = head.forward(head_inputs);
            const float metric =
                copy_float(require_output(head_outputs, "cache_metric"), 1, "cache metric")[0];
            const bool compute_tail =
                step == 0 || !std::isfinite(metric) || metric > cache_threshold_;

            if (compute_tail) {
                TensorMap tail_inputs;
                tail_inputs.emplace(
                    "position_ids",
                    Tensor{metadata.positions.data(), {kSequenceRows, 3}, DType::kFloat32});
                tail_inputs.emplace(
                    "adaln_indices",
                    Tensor{metadata.adaln_indices.data(), {kSequenceRows}, DType::kInt32});
                append_block_modulation_inputs(tail_inputs, modulation, 1, kLayers);
                tail.forward_async(tail_inputs);
                if (!resident_->previous_head_residual->copy_from(*resident_->head_residual))
                    throw std::runtime_error("MiniMax-H3 failed to update FirstBlockCache state");
                ++full_denoiser_steps;
            } else {
                ++skipped_denoiser_steps;
            }

            TensorMap finish_inputs;
            finish_inputs.emplace(
                "timestep_indices",
                Tensor{metadata.timestep_indices.data(), {kSequenceRows}, DType::kInt32});
            append_final_modulation_input(finish_inputs, modulation);
            finish.forward_async(finish_inputs);
            minimax_h3::scheduler_step_cuda_async(
                static_cast<float*>(resident_->video_rows->data()),
                static_cast<const float*>(resident_->video_velocity->data()), video_rows.size(),
                video_schedule.timesteps[step], video_schedule.sigmas[step],
                video_schedule.sigmas[step + 1], stream_);
            minimax_h3::scheduler_step_cuda_async(
                static_cast<float*>(resident_->audio_rows->data()),
                static_cast<const float*>(resident_->audio_velocity->data()), audio_rows.size(),
                audio_schedule.timesteps[step], audio_schedule.sigmas[step],
                audio_schedule.sigmas[step + 1], stream_);
            std::cerr << "[minimax-h3] denoiser " << (step + 1) << '/'
                      << video_schedule.timesteps.size() << " cache_metric=" << metric
                      << " compute_tail=" << (compute_tail ? 1 : 0) << '\n';
        }
        finish.sync();
    } else {
        auto& module = *resident_->denoiser;
        module.reset_execution_context();
        for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
            TensorMap inputs;
            inputs.emplace("video_hidden_states",
                           Tensor{video_rows.data(), {kVideoRows, kPatchDim}, DType::kFloat32});
            inputs.emplace(
                "audio_hidden_states",
                Tensor{audio_rows.data(), {kAudioRows, kAudioChannels}, DType::kFloat32});
            inputs.emplace(
                "encoder_hidden_states",
                Tensor{resident_->text_embeddings.data(), {kTextRows, kTextDim}, DType::kFloat32});
            inputs.emplace("position_ids",
                           Tensor{metadata.positions.data(), {kSequenceRows, 3}, DType::kFloat32});
            inputs.emplace("adaln_indices",
                           Tensor{metadata.adaln_indices.data(), {kSequenceRows}, DType::kInt32});
            inputs.emplace(
                "timestep_indices",
                Tensor{metadata.timestep_indices.data(), {kSequenceRows}, DType::kInt32});
            append_modulation_inputs(inputs, resident_->modulations[step]);
            const auto outputs = module.forward(inputs);
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
            ++full_denoiser_steps;
            std::cerr << "[minimax-h3] denoiser " << (step + 1) << '/'
                      << video_schedule.timesteps.size() << '\n';
        }
        module.sync();
    }
    const auto denoiser_end = Clock::now();
    audio_rows.clear();
    audio_rows.shrink_to_fit();

    std::vector<float> latent;
    if (!first_block_cache_) {
        latent = unpatchify_video(video_rows);
        denormalize_latents(latent);
    }
    video_rows.clear();
    video_rows.shrink_to_fit();
    const std::size_t expected_pixels =
        static_cast<std::size_t>(3) * kOutputFrames * kOutputHeight * kOutputWidth;
    std::vector<float> pixels;
    const auto vae_begin = Clock::now();
    const bool vae_resident_hit =
        first_block_cache_
            ? resident_->vae != nullptr && resident_->vae_latent_tiles != nullptr &&
                  resident_->vae_latent_tiles->ok() && resident_->vae_decoded_tiles != nullptr &&
                  resident_->vae_decoded_tiles->ok() && resident_->vae_overlap != nullptr &&
                  resident_->vae_overlap->ok() && resident_->frame_major_rgb != nullptr &&
                  resident_->frame_major_rgb->ok()
            : resident_->vae != nullptr;
    if (first_block_cache_) {
        if (!vae_resident_hit) {
            auto module = loader_("vae_tile_decoder_plan", stream_);
            module->set_timing_label("vae_tile_decoder_plan");
            DeviceTensor latent_tiles(
                {kTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize},
                DType::kFloat32, stream_);
            DeviceTensor decoded_tiles({kTileBatch, 3, kTileFrames, kTileSize, kTileSize},
                                       DType::kFloat32, stream_);
            DeviceTensor overlap({3, 5, kOutputHeight, kOutputWidth}, DType::kFloat32, stream_);
            DeviceTensor frame_major_rgb({kOutputFrames, kOutputHeight, kOutputWidth, 3},
                                         DType::kFloat32, stream_);
            if (!latent_tiles.ok() || !decoded_tiles.ok() || !overlap.ok() || !frame_major_rgb.ok())
                throw std::runtime_error("MiniMax-H3 failed to allocate CUDA VAE buffers");

            auto resident_latent_tiles = std::make_unique<DeviceTensor>(std::move(latent_tiles));
            auto resident_decoded_tiles = std::make_unique<DeviceTensor>(std::move(decoded_tiles));
            auto resident_overlap = std::make_unique<DeviceTensor>(std::move(overlap));
            auto resident_frame_major_rgb =
                std::make_unique<DeviceTensor>(std::move(frame_major_rgb));
            bind_external_checked(
                *module, "latent_tiles", resident_latent_tiles->data(), true, DType::kFloat32,
                {kTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize});
            bind_external_checked(*module, "decoded_tiles", resident_decoded_tiles->data(), false,
                                  DType::kFloat32,
                                  {kTileBatch, 3, kTileFrames, kTileSize, kTileSize});

            resident_->vae = std::move(module);
            resident_->vae_latent_tiles = std::move(resident_latent_tiles);
            resident_->vae_decoded_tiles = std::move(resident_decoded_tiles);
            resident_->vae_overlap = std::move(resident_overlap);
            resident_->frame_major_rgb = std::move(resident_frame_major_rgb);
        }

        auto& module = *resident_->vae;
        module.reset_execution_context();
        const auto latent_normalization = vae_latent_normalization();
        const auto pixel_normalization = vae_pixel_normalization();
        TensorMap no_inputs;
        for (int32_t clip_index = 0; clip_index < 7; ++clip_index) {
            minimax_h3::extract_vae_tiles_cuda_async(
                static_cast<const float*>(resident_->video_rows->data()),
                static_cast<float*>(resident_->vae_latent_tiles->data()), clip_index,
                latent_normalization, stream_);
            module.forward_async(no_inputs);
            minimax_h3::assemble_vae_clip_cuda_async(
                static_cast<const float*>(resident_->vae_decoded_tiles->data()),
                static_cast<float*>(resident_->vae_overlap->data()),
                static_cast<float*>(resident_->frame_major_rgb->data()), clip_index,
                pixel_normalization, stream_);
            std::cerr << "[minimax-h3] VAE clip " << (clip_index + 1) << "/7\n";
        }
        module.sync();
        pixels.resize(expected_pixels);
        if (!resident_->frame_major_rgb->copy_to_host(pixels.data()))
            throw std::runtime_error("MiniMax-H3 failed to download CUDA VAE output");
    } else {
        if (!resident_->vae) {
            resident_->vae = loader_("vae_tile_decoder_plan", stream_);
            resident_->vae->set_timing_label("vae_tile_decoder_plan");
        }
        std::vector<float> video(expected_pixels);
        std::size_t decoded_frames = 0;
        std::vector<float> overlap;
        std::vector<float> clip;
        auto& module = *resident_->vae;
        module.reset_execution_context();
        constexpr std::size_t output_count =
            static_cast<std::size_t>(kTileBatch) * 3 * kTileFrames * kTileSize * kTileSize;
        for (int32_t clip_index = 0; clip_index < 7; ++clip_index) {
            auto latent_tiles = extract_tiles(latent, clip_index);
            TensorMap inputs;
            inputs.emplace("latent_tiles", Tensor{latent_tiles.data(),
                                                  {kTileBatch, kLatentChannels, kTileInputFrames,
                                                   kTileLatentSize, kTileLatentSize},
                                                  DType::kFloat32});
            const auto outputs = module.forward(inputs);
            const Tensor decoded_tiles = require_output(outputs, "decoded_tiles");
            if (decoded_tiles.numel() != output_count)
                throw std::runtime_error("MiniMax-H3 invalid VAE decoded tiles output");
            stitch_spatial_tiles(decoded_tiles, clip);
            write_temporal_chunk(video, decoded_frames, clip, overlap);
            decoded_frames += 17;
            update_trailing_overlap(clip, overlap);
            std::cerr << "[minimax-h3] VAE clip " << (clip_index + 1) << "/7\n";
        }
        module.sync();
        write_final_overlap(video, decoded_frames, overlap);
        decoded_frames += 5;
        if (video.size() != expected_pixels || decoded_frames != kOutputFrames)
            throw std::runtime_error("MiniMax-H3 VAE produced the wrong video geometry");
        postprocess_video(video);
        pixels = to_frame_major_rgb(video);
    }
    const auto vae_end = Clock::now();
    latent.clear();
    latent.shrink_to_fit();

    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[minimax-h3.perf] text_encoder_ms=" << milliseconds(text_begin, text_end)
              << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
              << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " total_ms=" << milliseconds(total_begin, total_end)
              << " text_cache_hit=" << (text_cache_hit ? 1 : 0)
              << " adaln_cache_hit=" << (adaln_cache_hit ? 1 : 0)
              << " denoiser_resident_hit=" << (denoiser_resident_hit ? 1 : 0)
              << " vae_resident_hit=" << (vae_resident_hit ? 1 : 0)
              << " first_block_cache=" << (first_block_cache_ ? 1 : 0)
              << " cache_threshold=" << cache_threshold_
              << " full_denoiser_steps=" << full_denoiser_steps
              << " skipped_denoiser_steps=" << skipped_denoiser_steps << '\n';
    ImageResult result;
    result.height = kOutputHeight;
    result.width = kOutputWidth;
    result.channels = 3;
    result.num_frames = kOutputFrames;
    result.pixels = std::move(pixels);
    return result;
}

} // namespace trtmc
