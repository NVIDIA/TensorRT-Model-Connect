/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include "runtime/models/minimax_h3/conditioner_preprocess.h"
#include "runtime/models/minimax_h3/media_preprocess.h"
#include "runtime/models/minimax_h3/ref2va_conditioner.h"
#include "runtime/models/minimax_h3/torch_cuda_normal.h"
#include "runtime/models/minimax_h3/vae_encoder_tiles.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
constexpr int32_t kOutputAudioChannels = 2;
constexpr int32_t kAudioSampleRate = 32000;
constexpr int32_t kAudioHopLength = 800;
constexpr int32_t kOutputAudioSamples = kAudioLatents * kAudioHopLength;
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
static_assert(kAudioRows == kOutputAudioChannels * kAudioLatents);

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

float bfloat16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t float_to_bfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

std::vector<float> copy_bfloat16_as_float(const Tensor& tensor, std::size_t expected_numel,
                                          const char* label) {
    if (tensor.dtype != DType::kBFloat16 || tensor.numel() != expected_numel)
        throw std::runtime_error(std::string("MiniMax-H3 invalid ") + label + " output");
    const auto* values = static_cast<const uint16_t*>(tensor.data);
    std::vector<float> result(expected_numel);
    std::transform(values, values + expected_numel, result.begin(), bfloat16_to_float);
    return result;
}

std::vector<uint16_t> floats_to_bfloat16(const std::vector<float>& values) {
    std::vector<uint16_t> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), float_to_bfloat16);
    return result;
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

void validate_patchify_shape(const std::vector<float>& latent, int32_t frames, int32_t height,
                             int32_t width) {
    const std::size_t expected =
        static_cast<std::size_t>(kLatentChannels) * frames * height * width;
    if (frames <= 0 || height <= 0 || width <= 0 || height % kPatchHeight != 0 ||
        width % kPatchWidth != 0 || latent.size() != expected)
        throw std::invalid_argument("MiniMax-H3 video latent count is invalid");
}

std::vector<float> patchify_video_shape(const std::vector<float>& latent, int32_t frames,
                                        int32_t height, int32_t width) {
    validate_patchify_shape(latent, frames, height, width);
    const int32_t row_count = frames * (height / kPatchHeight) * (width / kPatchWidth);
    std::vector<float> rows(static_cast<std::size_t>(row_count) * kPatchDim);
    std::size_t target = 0;
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t y = 0; y < height; y += kPatchHeight) {
            for (int32_t x = 0; x < width; x += kPatchWidth) {
                for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                    for (int32_t py = 0; py < kPatchHeight; ++py) {
                        for (int32_t px = 0; px < kPatchWidth; ++px) {
                            const auto source =
                                ((((static_cast<std::size_t>(channel) * frames + frame) * height +
                                   y + py) *
                                  width) +
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

std::vector<float> patchify_video(const std::vector<float>& latent) {
    return patchify_video_shape(latent, kLatentFrames, kLatentHeight, kLatentWidth);
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

std::vector<float> unpack_audio_latents(const std::vector<float>& rows) {
    if (rows.size() != kAudioCount)
        throw std::invalid_argument("MiniMax-H3 audio rows are invalid");

    // The denoiser stores channel-major rows as [2 * T, C]. The mono audio
    // VAE consumes the stereo channels as two independent batch items in
    // [2, C, T], matching the Hugging Face pipeline boundary exactly.
    std::vector<float> latents(kAudioCount);
    for (int32_t channel = 0; channel < kOutputAudioChannels; ++channel) {
        for (int32_t frame = 0; frame < kAudioLatents; ++frame) {
            for (int32_t latent_channel = 0; latent_channel < kAudioChannels; ++latent_channel) {
                const auto source =
                    (static_cast<std::size_t>(channel) * kAudioLatents + frame) * kAudioChannels +
                    latent_channel;
                const auto target =
                    (static_cast<std::size_t>(channel) * kAudioChannels + latent_channel) *
                        kAudioLatents +
                    frame;
                latents[target] = rows[source];
            }
        }
    }
    return latents;
}

std::vector<double> spatial_position_grid(int32_t dimension, int32_t patch, double sqrt_area) {
    const int32_t count = dimension / patch;
    const double ratio = dimension / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    std::vector<double> grid(static_cast<std::size_t>(count));
    for (int32_t index = 0; index < count; ++index)
        grid[static_cast<std::size_t>(index)] =
            (left + static_cast<double>(index) * ratio / count) * 32.0;
    return grid;
}

std::vector<double> temporal_position_grid(int32_t num_latent_frames, double origin) {
    std::vector<double> grid(static_cast<std::size_t>(num_latent_frames));
    double position = origin;
    for (int32_t frame = 0; frame < num_latent_frames; ++frame) {
        grid[static_cast<std::size_t>(frame)] = position;
        const int32_t multiple = frame % 5 == 0 ? 1 : 4;
        position += (5.0 / 3.0) * multiple;
    }
    return grid;
}

double temporal_position_span(int32_t num_latent_frames) {
    double span = 0.0;
    for (int32_t frame = 0; frame < num_latent_frames; ++frame)
        span += (5.0 / 3.0) * (frame % 5 == 0 ? 1 : 4);
    return span;
}

struct DenoiserMetadata {
    std::vector<float> positions;
    std::vector<int32_t> adaln_indices;
    std::vector<int32_t> timestep_indices;
};

DenoiserMetadata make_denoiser_metadata() {
    auto layout = make_minimax_h3_fl2va_layout(std::vector<int32_t>(kTextRows, 1), kLatentFrames,
                                               kLatentHeight, kLatentWidth, kAudioLatents);
    if (layout.sequence_rows != kSequenceRows)
        throw std::logic_error("MiniMax-H3 fixed profile layout row count changed");
    DenoiserMetadata result;
    result.adaln_indices.resize(kSequenceRows);
    result.timestep_indices.assign(kSequenceRows, 0);
    for (int32_t row : layout.audio_indices)
        result.timestep_indices[static_cast<std::size_t>(row)] = 1;
    for (int32_t row = 0; row < kSequenceRows; ++row)
        result.adaln_indices[static_cast<std::size_t>(row)] =
            result.timestep_indices[static_cast<std::size_t>(row)] * kModalityCount +
            layout.token_tags[static_cast<std::size_t>(row)];
    result.positions = std::move(layout.position_ids);
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

constexpr double kMinReferenceDurationSeconds = 2.0;
constexpr double kMaxReferenceDurationSeconds = 15.0;

std::size_t checked_media_elements(std::initializer_list<int32_t> dimensions,
                                   const std::string& label) {
    std::size_t result = 1;
    for (const int32_t dimension : dimensions) {
        if (dimension <= 0 ||
            result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension))
            throw std::invalid_argument(label + " has invalid dimensions");
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

void validate_media_values(const std::vector<float>& values, std::size_t expected, float minimum,
                           float maximum, const std::string& label) {
    if (values.size() != expected)
        throw std::invalid_argument(label + " buffer size does not match its dimensions");
    if (!std::all_of(values.begin(), values.end(), [minimum, maximum](float value) {
            return std::isfinite(value) && value >= minimum && value <= maximum;
        }))
        throw std::invalid_argument(label + " contains invalid samples");
}

void validate_media_image(const MediaImageInput& image, const std::string& label) {
    const auto expected = checked_media_elements({image.height, image.width, 3}, label);
    validate_media_values(image.pixels, expected, 0.0F, 1.0F, label);
}

double validate_media_audio(const MultiChannelAudioResult& audio, const std::string& label) {
    if (audio.num_channels < 1 || audio.num_channels > 2 || audio.num_samples <= 0 ||
        audio.sample_rate <= 0)
        throw std::invalid_argument(label + " must be mono or stereo audio with a positive rate");
    const auto expected = checked_media_elements({audio.num_channels, audio.num_samples}, label);
    validate_media_values(audio.samples, expected, -1.0F, 1.0F, label);
    const double duration = static_cast<double>(audio.num_samples) / audio.sample_rate;
    if (duration < kMinReferenceDurationSeconds || duration > kMaxReferenceDurationSeconds)
        throw std::invalid_argument(label + " duration must be between 2 and 15 seconds");
    return duration;
}

double validate_media_video(const MediaVideoInput& video, const std::string& label) {
    if (!std::isfinite(video.fps) || video.fps <= 0.0F)
        throw std::invalid_argument(label + " must declare a positive finite frame rate");
    const auto expected =
        checked_media_elements({video.num_frames, video.height, video.width, 3}, label);
    validate_media_values(video.pixels, expected, 0.0F, 1.0F, label);
    const double duration = static_cast<double>(video.num_frames) / video.fps;
    if (duration < kMinReferenceDurationSeconds || duration > kMaxReferenceDurationSeconds)
        throw std::invalid_argument(label + " duration must be between 2 and 15 seconds");
    if (video.soundtrack.has_value())
        (void)validate_media_audio(*video.soundtrack, label + " soundtrack");
    return duration;
}

struct ReferenceSummary {
    int32_t images{0};
    int32_t videos{0};
    int32_t audios{0};
    double video_duration{0.0};
    double audio_duration{0.0};
    double soundtrack_duration{0.0};
};

void validate_reference_payload(const AudioVideoReference& reference, const std::string& label) {
    const bool has_image = !reference.image.pixels.empty();
    const bool has_video = !reference.video.pixels.empty();
    const bool has_audio = !reference.audio.samples.empty();
    if (static_cast<int>(has_image) + static_cast<int>(has_video) + static_cast<int>(has_audio) !=
        1)
        throw std::invalid_argument(label + " must contain exactly its declared media payload");
    const bool matches_kind =
        reference.kind == AudioVideoReferenceKind::kImage
            ? has_image
            : (reference.kind == AudioVideoReferenceKind::kVideo ? has_video : has_audio);
    if (!matches_kind)
        throw std::invalid_argument(label + " media payload does not match its declared modality");
}

void validate_reference(const AudioVideoReference& reference, std::size_t index,
                        ReferenceSummary& summary) {
    const std::string label = "MiniMax-H3 reference " + std::to_string(index);
    validate_reference_payload(reference, label);
    switch (reference.kind) {
    case AudioVideoReferenceKind::kImage:
        validate_media_image(reference.image, label + " image");
        ++summary.images;
        return;
    case AudioVideoReferenceKind::kVideo:
        summary.video_duration += validate_media_video(reference.video, label + " video");
        if (reference.video.soundtrack.has_value())
            summary.soundtrack_duration +=
                validate_media_audio(*reference.video.soundtrack, label + " video soundtrack");
        ++summary.videos;
        return;
    case AudioVideoReferenceKind::kAudio:
        summary.audio_duration += validate_media_audio(reference.audio, label + " audio");
        ++summary.audios;
        return;
    }
    throw std::invalid_argument(label + " has an unknown modality");
}

void validate_reference_summary(const ReferenceSummary& summary, std::size_t total) {
    if (summary.images > 9 || summary.videos > 3 || summary.audios > 3 || total > 12)
        throw std::invalid_argument("MiniMax-H3 reference count exceeds the model-card limits");
    if (summary.images + summary.videos == 0)
        throw std::invalid_argument(
            "MiniMax-H3 audio references require at least one image or video reference");
    if (summary.video_duration > kMaxReferenceDurationSeconds ||
        summary.audio_duration > kMaxReferenceDurationSeconds ||
        summary.soundtrack_duration > kMaxReferenceDurationSeconds)
        throw std::invalid_argument(
            "MiniMax-H3 reference duration exceeds the 15-second per-modality limit");
}

struct Fl2vaLayoutGeometry {
    int32_t text_rows{0};
    int32_t rows_per_frame{0};
    int32_t condition_rows{0};
    int32_t audio_rows{0};
    int32_t video_rows{0};
    int32_t condition_start{0};
    int32_t audio_start{0};
    int32_t video_start{0};
    int32_t sequence_rows{0};
    std::vector<double> height_grid;
    std::vector<double> width_grid;
};

void validate_fl2va_layout_inputs(const std::vector<int32_t>& text_token_tags,
                                  int32_t num_latent_frames, int32_t latent_height,
                                  int32_t latent_width, int32_t num_audio_latents,
                                  const std::vector<MiniMaxH3KeyframeAnchor>& keyframe_anchors) {
    const std::array<int32_t, 4> dimensions = {num_latent_frames, num_audio_latents, latent_height,
                                               latent_width};
    if (text_token_tags.empty() ||
        !std::all_of(dimensions.begin(), dimensions.end(), [](int32_t value) { return value > 0; }))
        throw std::invalid_argument("MiniMax-H3 packed layout geometry is invalid");
    if (latent_height % kPatchHeight != 0 || latent_width % kPatchWidth != 0 ||
        keyframe_anchors.size() > 2)
        throw std::invalid_argument("MiniMax-H3 packed layout geometry is invalid");
    if (!std::all_of(text_token_tags.begin(), text_token_tags.end(),
                     [](int32_t tag) { return tag == 0 || tag == 1; }))
        throw std::invalid_argument("MiniMax-H3 text token tags must be text or video");
}

Fl2vaLayoutGeometry make_fl2va_layout_geometry(std::size_t text_rows, std::size_t keyframes,
                                               int32_t num_latent_frames, int32_t latent_height,
                                               int32_t latent_width, int32_t num_audio_latents) {
    const int64_t rows_per_frame =
        static_cast<int64_t>(latent_height / kPatchHeight) * (latent_width / kPatchWidth);
    const int64_t condition_rows = static_cast<int64_t>(keyframes) * rows_per_frame;
    const int64_t audio_rows = static_cast<int64_t>(kOutputAudioChannels) * num_audio_latents;
    const int64_t video_rows = static_cast<int64_t>(num_latent_frames) * rows_per_frame;
    const int64_t sequence_rows =
        static_cast<int64_t>(text_rows) + condition_rows + audio_rows + video_rows;
    if (sequence_rows > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("MiniMax-H3 packed layout is too large");

    Fl2vaLayoutGeometry geometry;
    geometry.text_rows = static_cast<int32_t>(text_rows);
    geometry.rows_per_frame = static_cast<int32_t>(rows_per_frame);
    geometry.condition_rows = static_cast<int32_t>(condition_rows);
    geometry.audio_rows = static_cast<int32_t>(audio_rows);
    geometry.video_rows = static_cast<int32_t>(video_rows);
    geometry.condition_start = geometry.text_rows;
    geometry.audio_start = geometry.condition_start + geometry.condition_rows;
    geometry.video_start = geometry.audio_start + geometry.audio_rows;
    geometry.sequence_rows = static_cast<int32_t>(sequence_rows);
    const double sqrt_area = std::sqrt(static_cast<double>(latent_height) * latent_width);
    geometry.height_grid = spatial_position_grid(latent_height, kPatchHeight, sqrt_area);
    geometry.width_grid = spatial_position_grid(latent_width, kPatchWidth, sqrt_area);
    return geometry;
}

MiniMaxH3PackedLayout initialize_fl2va_layout(const std::vector<int32_t>& text_token_tags,
                                              const Fl2vaLayoutGeometry& geometry) {
    MiniMaxH3PackedLayout result;
    result.sequence_rows = geometry.sequence_rows;
    result.position_ids.assign(static_cast<std::size_t>(geometry.sequence_rows) * 3, 0.0F);
    result.token_tags.resize(static_cast<std::size_t>(geometry.sequence_rows));
    result.num_condition_video_rows = geometry.condition_rows;
    result.text_indices.resize(static_cast<std::size_t>(geometry.text_rows));
    std::iota(result.text_indices.begin(), result.text_indices.end(), 0);
    std::copy(text_token_tags.begin(), text_token_tags.end(), result.token_tags.begin());
    for (int32_t row = 0; row < geometry.text_rows; ++row)
        result.position_ids[static_cast<std::size_t>(row) * 3] = static_cast<float>(row);
    return result;
}

void fill_fl2va_condition_layout(MiniMaxH3PackedLayout& result, const Fl2vaLayoutGeometry& geometry,
                                 int32_t num_latent_frames,
                                 const std::vector<MiniMaxH3KeyframeAnchor>& anchors) {
    for (std::size_t condition = 0; condition < anchors.size(); ++condition) {
        const double anchor_time =
            anchors[condition] == MiniMaxH3KeyframeAnchor::kFirst
                ? static_cast<double>(geometry.text_rows)
                : geometry.text_rows + temporal_position_span(num_latent_frames) - (5.0 / 3.0);
        int32_t row =
            geometry.condition_start + static_cast<int32_t>(condition) * geometry.rows_per_frame;
        for (double y : geometry.height_grid) {
            for (double x : geometry.width_grid) {
                const auto offset = static_cast<std::size_t>(row) * 3;
                result.position_ids[offset] = static_cast<float>(anchor_time);
                result.position_ids[offset + 1] = static_cast<float>(y);
                result.position_ids[offset + 2] = static_cast<float>(x);
                result.token_tags[static_cast<std::size_t>(row++)] = 0;
            }
        }
    }
}

void fill_fl2va_audio_layout(MiniMaxH3PackedLayout& result, const Fl2vaLayoutGeometry& geometry,
                             int32_t num_audio_latents) {
    result.audio_indices.resize(static_cast<std::size_t>(geometry.audio_rows));
    std::iota(result.audio_indices.begin(), result.audio_indices.end(), geometry.audio_start);
    for (int32_t channel = 0; channel < kOutputAudioChannels; ++channel) {
        for (int32_t frame = 0; frame < num_audio_latents; ++frame) {
            const int32_t row = geometry.audio_start + channel * num_audio_latents + frame;
            const auto offset = static_cast<std::size_t>(row) * 3;
            result.position_ids[offset] = static_cast<float>(geometry.text_rows + frame);
            result.position_ids[offset + 2] = static_cast<float>(
                channel == 0 ? geometry.width_grid.front() : geometry.width_grid.back());
            result.token_tags[static_cast<std::size_t>(row)] = 2;
        }
    }
}

void fill_fl2va_video_layout(MiniMaxH3PackedLayout& result, const Fl2vaLayoutGeometry& geometry,
                             int32_t num_latent_frames) {
    result.video_indices.reserve(
        static_cast<std::size_t>(geometry.condition_rows + geometry.video_rows));
    for (int32_t row = geometry.condition_start; row < geometry.audio_start; ++row)
        result.video_indices.push_back(row);
    const auto time_grid =
        temporal_position_grid(num_latent_frames, static_cast<double>(geometry.text_rows));
    int32_t row = geometry.video_start;
    for (double time : time_grid) {
        for (double y : geometry.height_grid) {
            for (double x : geometry.width_grid) {
                const auto offset = static_cast<std::size_t>(row) * 3;
                result.position_ids[offset] = static_cast<float>(time);
                result.position_ids[offset + 1] = static_cast<float>(y);
                result.position_ids[offset + 2] = static_cast<float>(x);
                result.token_tags[static_cast<std::size_t>(row)] = 0;
                result.video_indices.push_back(row++);
            }
        }
    }
    if (row != result.sequence_rows)
        throw std::logic_error("MiniMax-H3 packed layout construction failed");
}

void validate_ref2va_audio_reference_geometry(const MiniMaxH3PreparedReferenceLayout& reference) {
    const std::array<int32_t, 3> geometry = {reference.num_latent_frames, reference.latent_height,
                                             reference.latent_width};
    if (std::any_of(geometry.begin(), geometry.end(), [](int32_t value) { return value != 0; }))
        throw std::invalid_argument("MiniMax-H3 audio reference has video geometry");
}

int64_t ref2va_video_rows(const MiniMaxH3PreparedReferenceLayout& reference) {
    if (reference.kind == AudioVideoReferenceKind::kAudio) {
        validate_ref2va_audio_reference_geometry(reference);
        return 0;
    }
    const std::array<int32_t, 3> dimensions = {reference.num_latent_frames, reference.latent_height,
                                               reference.latent_width};
    if (!std::all_of(dimensions.begin(), dimensions.end(), [](int32_t value) { return value > 0; }))
        throw std::invalid_argument("MiniMax-H3 reference video latent geometry is invalid");
    if (reference.latent_height % kPatchHeight != 0 || reference.latent_width % kPatchWidth != 0)
        throw std::invalid_argument("MiniMax-H3 reference video latent geometry is invalid");
    if (reference.kind == AudioVideoReferenceKind::kImage) {
        if (reference.num_latent_frames != 1)
            throw std::invalid_argument("MiniMax-H3 image reference must have one latent frame");
        if (reference.num_audio_latents != 0)
            throw std::invalid_argument("MiniMax-H3 image reference cannot carry audio rows");
    }
    return static_cast<int64_t>(reference.num_latent_frames) *
           (reference.latent_height / kPatchHeight) * (reference.latent_width / kPatchWidth);
}

int64_t ref2va_audio_rows(const MiniMaxH3PreparedReferenceLayout& reference) {
    if (reference.num_audio_latents < 0 ||
        (reference.kind == AudioVideoReferenceKind::kAudio && reference.num_audio_latents == 0))
        throw std::invalid_argument("MiniMax-H3 reference audio latent geometry is invalid");
    return static_cast<int64_t>(reference.num_audio_latents) * kOutputAudioChannels;
}

struct Ref2vaLayoutGeometry {
    int32_t condition_video_rows{0};
    int32_t condition_audio_rows{0};
    int32_t target_audio_rows{0};
    int32_t target_video_rows{0};
    int32_t sequence_rows{0};
    std::vector<double> target_width_grid;
};

Ref2vaLayoutGeometry
make_ref2va_layout_geometry(std::size_t text_rows,
                            const std::vector<MiniMaxH3PreparedReferenceLayout>& references,
                            int32_t num_latent_frames, int32_t latent_height, int32_t latent_width,
                            int32_t num_audio_latents) {
    int64_t condition_video_rows = 0;
    int64_t condition_audio_rows = 0;
    for (const auto& reference : references) {
        condition_video_rows += ref2va_video_rows(reference);
        condition_audio_rows += ref2va_audio_rows(reference);
    }
    const int64_t target_rows_per_frame =
        static_cast<int64_t>(latent_height / kPatchHeight) * (latent_width / kPatchWidth);
    const int64_t target_audio_rows =
        static_cast<int64_t>(num_audio_latents) * kOutputAudioChannels;
    const int64_t target_video_rows =
        static_cast<int64_t>(num_latent_frames) * target_rows_per_frame;
    const int64_t sequence_rows = static_cast<int64_t>(text_rows) + condition_video_rows +
                                  condition_audio_rows + target_audio_rows + target_video_rows;
    if (sequence_rows > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("MiniMax-H3 Ref2VA packed layout is too large");

    Ref2vaLayoutGeometry geometry;
    geometry.condition_video_rows = static_cast<int32_t>(condition_video_rows);
    geometry.condition_audio_rows = static_cast<int32_t>(condition_audio_rows);
    geometry.target_audio_rows = static_cast<int32_t>(target_audio_rows);
    geometry.target_video_rows = static_cast<int32_t>(target_video_rows);
    geometry.sequence_rows = static_cast<int32_t>(sequence_rows);
    const double sqrt_area = std::sqrt(static_cast<double>(latent_height) * latent_width);
    geometry.target_width_grid = spatial_position_grid(latent_width, kPatchWidth, sqrt_area);
    return geometry;
}

MiniMaxH3PackedLayout initialize_ref2va_layout(const std::vector<int32_t>& text_token_tags,
                                               const Ref2vaLayoutGeometry& geometry) {
    MiniMaxH3PackedLayout result;
    result.sequence_rows = geometry.sequence_rows;
    result.position_ids.assign(static_cast<std::size_t>(geometry.sequence_rows) * 3, 0.0F);
    result.token_tags.resize(static_cast<std::size_t>(geometry.sequence_rows));
    result.text_indices.resize(text_token_tags.size());
    std::iota(result.text_indices.begin(), result.text_indices.end(), 0);
    std::copy(text_token_tags.begin(), text_token_tags.end(), result.token_tags.begin());
    for (int32_t row = 0; row < static_cast<int32_t>(text_token_tags.size()); ++row)
        result.position_ids[static_cast<std::size_t>(row) * 3] = static_cast<float>(row);
    return result;
}

void append_ref2va_audio(MiniMaxH3PackedLayout& result, int32_t start, int32_t latent_count,
                         double origin, const std::vector<double>& width_grid) {
    for (int32_t channel = 0; channel < kOutputAudioChannels; ++channel) {
        for (int32_t frame = 0; frame < latent_count; ++frame) {
            const int32_t row = start + channel * latent_count + frame;
            const auto offset = static_cast<std::size_t>(row) * 3;
            result.position_ids[offset] = static_cast<float>(origin + frame);
            result.position_ids[offset + 2] =
                static_cast<float>(channel == 0 ? width_grid.front() : width_grid.back());
            result.audio_indices.push_back(row);
        }
    }
}

std::vector<double> append_ref2va_video(MiniMaxH3PackedLayout& result, int32_t start,
                                        const MiniMaxH3PreparedReferenceLayout& reference,
                                        double origin) {
    const double sqrt_area =
        std::sqrt(static_cast<double>(reference.latent_height) * reference.latent_width);
    const auto height_grid =
        spatial_position_grid(reference.latent_height, kPatchHeight, sqrt_area);
    const auto width_grid = spatial_position_grid(reference.latent_width, kPatchWidth, sqrt_area);
    const auto time_grid = temporal_position_grid(reference.num_latent_frames, origin);
    int32_t row = start;
    for (double time : time_grid) {
        for (double y : height_grid) {
            for (double x : width_grid) {
                const auto offset = static_cast<std::size_t>(row) * 3;
                result.position_ids[offset] = static_cast<float>(time);
                result.position_ids[offset + 1] = static_cast<float>(y);
                result.position_ids[offset + 2] = static_cast<float>(x);
                result.video_indices.push_back(row++);
            }
        }
    }
    return width_grid;
}

void append_ref2va_reference(MiniMaxH3PackedLayout& result,
                             const MiniMaxH3PreparedReferenceLayout& reference,
                             const std::vector<double>& target_width_grid, int32_t& cursor,
                             double& rotary_time) {
    if (reference.kind == AudioVideoReferenceKind::kImage) {
        (void)append_ref2va_video(result, cursor, reference, rotary_time);
        cursor += static_cast<int32_t>(ref2va_video_rows(reference));
        rotary_time += 1.0;
        return;
    }
    if (reference.kind == AudioVideoReferenceKind::kAudio) {
        append_ref2va_audio(result, cursor, reference.num_audio_latents, rotary_time,
                            target_width_grid);
        cursor += static_cast<int32_t>(ref2va_audio_rows(reference));
        rotary_time += reference.num_audio_latents;
        return;
    }

    const int32_t audio_start = cursor;
    cursor += static_cast<int32_t>(ref2va_audio_rows(reference));
    const int32_t video_start = cursor;
    const auto reference_width_grid =
        append_ref2va_video(result, video_start, reference, rotary_time);
    cursor += static_cast<int32_t>(ref2va_video_rows(reference));
    append_ref2va_audio(result, audio_start, reference.num_audio_latents, rotary_time,
                        reference_width_grid);
    rotary_time += std::max(static_cast<double>(reference.num_audio_latents),
                            temporal_position_span(reference.num_latent_frames));
}

void tag_ref2va_media_rows(MiniMaxH3PackedLayout& result) {
    for (int32_t row : result.audio_indices)
        result.token_tags[static_cast<std::size_t>(row)] = 2;
    for (int32_t row : result.video_indices)
        result.token_tags[static_cast<std::size_t>(row)] = 0;
}

void validate_generate_config(const GenerateConfig& cfg) {
    if ((cfg.height > 0 && cfg.height != kOutputHeight) ||
        (cfg.width > 0 && cfg.width != kOutputWidth) ||
        (cfg.num_steps > 0 && cfg.num_steps != kSteps))
        throw std::invalid_argument(
            "MiniMax-H3 native profile is fixed at 124 frames, 768x1344, 50 grid points");
    if (cfg.guidance_scale >= 0.0F || !cfg.negative_prompt.empty())
        throw std::invalid_argument("MiniMax-H3 uses guidance-distilled weights and does not "
                                    "accept guidance or a negative prompt");
    if (!cfg.initial_latents.empty())
        throw std::invalid_argument(
            "MiniMax-H3 native initial-latent inputs are not implemented for this profile");
}

std::vector<MediaImageInput> prepare_fl2va_keyframes(const AudioVideoRequest& request) {
    std::vector<MediaImageInput> result;
    if (!request.first_image.pixels.empty())
        result.push_back(request.first_image);
    if (!request.last_image.pixels.empty())
        result.push_back(request.last_image);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = minimax_h3_prepare_keyframe_image(result[index], kOutputHeight,
                                                          kOutputWidth, index == 0);
    }
    return result;
}

struct CompactVisionFeatures {
    std::vector<float> main;
    std::array<std::vector<float>, 3> deepstack;
};

CompactVisionFeatures encode_fl2va_vision(const std::vector<MediaImageInput>& keyframes,
                                          const MiniMaxH3ModuleLoader& loader,
                                          cudaStream_t stream) {
    CompactVisionFeatures result;
    if (keyframes.empty())
        return result;
    auto module = loader("vision_conditioner_plan", stream);
    module->set_timing_label("vision_conditioner_plan");
    constexpr std::size_t feature_count =
        static_cast<std::size_t>(kMiniMaxH3ConditionerMergedRows) * kTextDim;
    for (const auto& keyframe : keyframes) {
        auto pixels = minimax_h3_preprocess_conditioner_keyframe(keyframe);
        TensorMap inputs;
        inputs.emplace("pixel_values",
                       Tensor{pixels.data(),
                              {kMiniMaxH3ConditionerPatchRows, kMiniMaxH3ConditionerPatchVector},
                              DType::kFloat32});
        const auto outputs = module->forward(inputs);
        auto main = copy_bfloat16_as_float(require_output(outputs, "image_features"), feature_count,
                                           "vision conditioner");
        result.main.insert(result.main.end(), main.begin(), main.end());
        for (std::size_t level = 0; level < result.deepstack.size(); ++level) {
            const std::string name = "deepstack_features_" + std::to_string(level);
            auto features =
                copy_bfloat16_as_float(require_output(outputs, name), feature_count, name.c_str());
            result.deepstack[level].insert(result.deepstack[level].end(), features.begin(),
                                           features.end());
        }
    }
    module->sync();
    return result;
}

struct Fl2vaTextConditioning {
    MiniMaxH3ConditionerPresentation presentation;
    std::vector<float> embeddings;
};

Fl2vaTextConditioning encode_fl2va_presentation(const AudioVideoRequest& request,
                                                const std::vector<MediaImageInput>& keyframes,
                                                ITokenizer& tokenizer,
                                                const MiniMaxH3ModuleLoader& loader,
                                                cudaStream_t stream) {
    Fl2vaTextConditioning result;
    result.presentation = minimax_h3_make_conditioner_presentation(
        request.prompt, !request.first_image.pixels.empty(), !request.last_image.pixels.empty(),
        [&tokenizer](const std::string& text) { return tokenizer.encode(text); });
    const auto compact = encode_fl2va_vision(keyframes, loader, stream);
    const auto aligned = minimax_h3_scatter_vision_features(result.presentation, compact.main,
                                                            compact.deepstack, kTextDim);
    auto main_bf16 = floats_to_bfloat16(aligned.vision_embeddings);
    std::array<std::vector<uint16_t>, 3> deepstack_bf16;
    for (std::size_t level = 0; level < deepstack_bf16.size(); ++level)
        deepstack_bf16[level] = floats_to_bfloat16(aligned.deepstack_embeddings[level]);

    auto module = loader("language_conditioner_plan", stream);
    module->set_timing_label("language_conditioner_plan");
    const int32_t rows = result.presentation.sequence_rows;
    TensorMap inputs;
    inputs.emplace("input_ids",
                   Tensor{result.presentation.input_ids.data(), {rows}, DType::kInt32});
    inputs.emplace("mrope_position_ids",
                   Tensor{result.presentation.mrope_position_ids.data(), {3, rows}, DType::kInt32});
    inputs.emplace("vision_embeddings",
                   Tensor{main_bf16.data(), {rows, kTextDim}, DType::kBFloat16});
    inputs.emplace("vision_selector",
                   Tensor{result.presentation.vision_selector.data(), {rows, 1}, DType::kInt32});
    for (std::size_t level = 0; level < deepstack_bf16.size(); ++level) {
        inputs.emplace("deepstack_embeddings_" + std::to_string(level),
                       Tensor{deepstack_bf16[level].data(), {rows, kTextDim}, DType::kBFloat16});
    }
    const auto outputs = module->forward(inputs);
    result.embeddings =
        copy_float(require_output(outputs, "encoder_hidden_states"),
                   static_cast<std::size_t>(rows) * kTextDim, "language conditioner");
    module->sync();
    return result;
}

std::vector<float> normalize_keyframe_for_vae(const MediaImageInput& keyframe) {
    const std::size_t plane = static_cast<std::size_t>(keyframe.height) * keyframe.width;
    std::vector<float> result(plane * 3);
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            result[static_cast<std::size_t>(channel) * plane + pixel] =
                (keyframe.pixels[pixel * 3 + channel] - kPixelMean[channel]) / kPixelStd[channel];
        }
    }
    return result;
}

std::vector<float> encode_visual_moments(const std::vector<float>& normalized,
                                         int32_t source_frames, int32_t height, int32_t width,
                                         ITrtModule& module) {
    const auto spatial = make_minimax_h3_vae_spatial_tile_plan(height, width);
    const auto temporal = make_minimax_h3_vae_temporal_chunk_plan(source_frames);
    std::vector<std::vector<float>> stitched_chunks;
    stitched_chunks.reserve(temporal.chunks.size());
    for (const auto& chunk : temporal.chunks) {
        std::vector<std::vector<float>> raw_tiles;
        raw_tiles.reserve(spatial.rows.size() * spatial.columns.size());
        for (const auto& row : spatial.rows) {
            for (const auto& column : spatial.columns) {
                auto tile = minimax_h3_extract_vae_encoder_tile(normalized, source_frames, height,
                                                                width, chunk, row, column);
                TensorMap inputs;
                inputs.emplace("normalized_rgb",
                               Tensor{tile.data(),
                                      {1, 3, chunk.engine_num_frames, kMiniMaxH3VaeEncoderTileSize,
                                       kMiniMaxH3VaeEncoderTileSize},
                                      DType::kFloat32});
                const auto outputs = module.forward(inputs);
                const std::size_t expected = static_cast<std::size_t>(kMiniMaxH3VaeMomentChannels) *
                                             chunk.raw_moment_frames * 16 * 16;
                raw_tiles.push_back(copy_float(require_output(outputs, "posterior_moments"),
                                               expected, "VAE encoder tile"));
            }
        }
        stitched_chunks.push_back(
            minimax_h3_stitch_vae_encoder_tiles(raw_tiles, spatial, chunk.raw_moment_frames));
    }
    module.sync();
    return minimax_h3_assemble_vae_temporal_moments(
        stitched_chunks, height / kMiniMaxH3VaeSpatialCompression,
        width / kMiniMaxH3VaeSpatialCompression, temporal);
}

std::vector<float> sample_keyframe_posterior(const std::vector<float>& moments) {
    constexpr std::size_t latent_count =
        static_cast<std::size_t>(kLatentChannels) * kLatentHeight * kLatentWidth;
    if (moments.size() != 2 * latent_count)
        throw std::runtime_error("MiniMax-H3 visual VAE returned invalid posterior moments");
    const auto noise = minimax_h3::torch_cuda_normal(latent_count, 42);
    std::vector<float> result(latent_count);
    const std::size_t plane = static_cast<std::size_t>(kLatentHeight) * kLatentWidth;
    for (std::size_t index = 0; index < latent_count; ++index) {
        const float log_variance = std::clamp(moments[latent_count + index], -30.0F, 20.0F);
        const float sampled = moments[index] + std::exp(0.5F * log_variance) * noise[index];
        const float rounded = __half2float(__float2half_rn(sampled));
        const int32_t channel = static_cast<int32_t>(index / plane);
        result[index] = (rounded - kLatentMean[channel]) / kLatentStd[channel];
    }
    return result;
}

struct Fl2vaVideoConditioning {
    std::vector<float> rows;
    uint64_t request_rng_offset{0};
};

Fl2vaVideoConditioning encode_fl2va_keyframe_latents(const std::vector<MediaImageInput>& keyframes,
                                                     int64_t request_seed,
                                                     const MiniMaxH3ModuleLoader& loader,
                                                     cudaStream_t stream) {
    Fl2vaVideoConditioning result;
    if (keyframes.empty())
        return result;
    auto module = loader("vae_encoder_tile_t1_plan", stream);
    module->set_timing_label("vae_encoder_tile_t1_plan");
    constexpr std::size_t latent_count =
        static_cast<std::size_t>(kLatentChannels) * kLatentHeight * kLatentWidth;
    for (const auto& keyframe : keyframes) {
        auto pixels = normalize_keyframe_for_vae(keyframe);
        auto moments = encode_visual_moments(pixels, 1, kOutputHeight, kOutputWidth, *module);
        auto latent = sample_keyframe_posterior(moments);
        auto latent_rows = patchify_video_shape(latent, 1, kLatentHeight, kLatentWidth);
        auto noise = minimax_h3::torch_cuda_normal(
            latent_count, static_cast<uint64_t>(request_seed), result.request_rng_offset);
        result.request_rng_offset += minimax_h3::torch_cuda_normal_consumed_offset(latent_count);
        auto noise_rows = patchify_video_shape(noise, 1, kLatentHeight, kLatentWidth);
        constexpr float timestep = 0.999F;
        for (std::size_t index = 0; index < latent_rows.size(); ++index)
            latent_rows[index] =
                timestep * latent_rows[index] + (1.0F - timestep) * noise_rows[index];
        result.rows.insert(result.rows.end(), latent_rows.begin(), latent_rows.end());
    }
    return result;
}

std::vector<MiniMaxH3PreparedReference>
prepare_ref2va_references(const AudioVideoRequest& request) {
    std::vector<MiniMaxH3PreparedReference> result;
    result.reserve(request.references.size());
    constexpr double target_duration = static_cast<double>(kOutputFrames) / 24.0;
    for (std::size_t index = 0; index < request.references.size(); ++index) {
        const auto& source = request.references[index];
        MiniMaxH3PreparedReference prepared;
        prepared.reference_index = index;
        prepared.kind = source.kind;
        if (source.kind == AudioVideoReferenceKind::kImage) {
            prepared.image = minimax_h3_prepare_reference_image(source.image);
            prepared.qwen_grid_h = prepared.image.height / kMiniMaxH3Ref2VAPatchSize;
            prepared.qwen_grid_w = prepared.image.width / kMiniMaxH3Ref2VAPatchSize;
        } else if (source.kind == AudioVideoReferenceKind::kVideo) {
            prepared.video = minimax_h3_prepare_reference_video(source.video, kOutputFrames);
            prepared.qwen_grid_h = prepared.video.height / kMiniMaxH3Ref2VAPatchSize;
            prepared.qwen_grid_w = prepared.video.width / kMiniMaxH3Ref2VAPatchSize;
            if (source.video.soundtrack.has_value())
                prepared.audio =
                    minimax_h3_prepare_reference_audio(*source.video.soundtrack, target_duration);
        } else {
            prepared.audio = minimax_h3_prepare_reference_audio(source.audio, target_duration);
        }
        prepared.qwen_patch_rows = prepared.qwen_grid_h * prepared.qwen_grid_w;
        result.push_back(std::move(prepared));
    }
    return result;
}

CompactVisionFeatures
encode_ref2va_vision(const MiniMaxH3Ref2VAConditionerPresentation& presentation,
                     const MiniMaxH3ModuleLoader& loader, cudaStream_t stream) {
    CompactVisionFeatures result;
    auto module = loader("vision_conditioner_plan", stream);
    module->set_timing_label("vision_conditioner_plan");
    module->reset_execution_context();
    for (const auto& vision : presentation.vision_inputs) {
        const int32_t patch_rows = vision.grid_h * vision.grid_w;
        const int32_t merged_rows =
            patch_rows / (kMiniMaxH3Ref2VASpatialMergeSize * kMiniMaxH3Ref2VASpatialMergeSize);
        TensorMap inputs;
        inputs.emplace("pixel_values", Tensor{const_cast<float*>(vision.pixel_values.data()),
                                              {patch_rows, kMiniMaxH3Ref2VAPatchVectorSize},
                                              DType::kFloat32});
        inputs.emplace("position_indices",
                       Tensor{const_cast<int32_t*>(vision.position_indices.data()),
                              {patch_rows, 4},
                              DType::kInt32});
        inputs.emplace("position_weights",
                       Tensor{const_cast<float*>(vision.position_weights.data()),
                              {patch_rows, 4},
                              DType::kFloat32});
        inputs.emplace("vision_position_ids",
                       Tensor{const_cast<int32_t*>(vision.vision_position_ids.data()),
                              {patch_rows, 2},
                              DType::kInt32});
        const auto outputs = module->forward(inputs);
        const std::size_t feature_count = static_cast<std::size_t>(merged_rows) * kTextDim;
        auto main = copy_bfloat16_as_float(require_output(outputs, "image_features"), feature_count,
                                           "Ref2VA vision conditioner");
        result.main.insert(result.main.end(), main.begin(), main.end());
        for (std::size_t level = 0; level < result.deepstack.size(); ++level) {
            const std::string name = "deepstack_features_" + std::to_string(level);
            auto features =
                copy_bfloat16_as_float(require_output(outputs, name), feature_count, name.c_str());
            result.deepstack[level].insert(result.deepstack[level].end(), features.begin(),
                                           features.end());
        }
    }
    module->sync();
    return result;
}

MiniMaxH3ConditionerVisionFeatures
scatter_ref2va_vision_features(const MiniMaxH3Ref2VAConditionerPresentation& presentation,
                               const CompactVisionFeatures& compact) {
    const std::size_t compact_rows = presentation.vision_scatter_indices.size();
    const std::size_t compact_values = compact_rows * kTextDim;
    if (compact.main.size() != compact_values ||
        !std::all_of(
            compact.deepstack.begin(), compact.deepstack.end(),
            [compact_values](const auto& values) { return values.size() == compact_values; }))
        throw std::runtime_error("MiniMax-H3 Ref2VA compact vision features are incomplete");
    MiniMaxH3ConditionerVisionFeatures result;
    result.sequence_rows = presentation.sequence_rows;
    result.feature_dim = kTextDim;
    const std::size_t sequence_values =
        static_cast<std::size_t>(presentation.sequence_rows) * kTextDim;
    result.vision_embeddings.assign(sequence_values, 0.0F);
    for (auto& values : result.deepstack_embeddings)
        values.assign(sequence_values, 0.0F);
    result.vision_selector = presentation.vision_selector;
    for (std::size_t compact_row = 0; compact_row < compact_rows; ++compact_row) {
        const int32_t sequence_row = presentation.vision_scatter_indices[compact_row];
        if (sequence_row < 0 || sequence_row >= presentation.sequence_rows)
            throw std::runtime_error("MiniMax-H3 Ref2VA vision scatter row is out of range");
        const auto source = static_cast<std::ptrdiff_t>(compact_row * kTextDim);
        const auto target = static_cast<std::ptrdiff_t>(sequence_row) * kTextDim;
        std::copy_n(compact.main.begin() + source, kTextDim,
                    result.vision_embeddings.begin() + target);
        for (std::size_t level = 0; level < result.deepstack_embeddings.size(); ++level)
            std::copy_n(compact.deepstack[level].begin() + source, kTextDim,
                        result.deepstack_embeddings[level].begin() + target);
    }
    return result;
}

struct Ref2vaTextConditioning {
    MiniMaxH3Ref2VAConditionerPresentation presentation;
    std::vector<float> embeddings;
};

Ref2vaTextConditioning
encode_ref2va_presentation(const AudioVideoRequest& request,
                           const std::vector<MiniMaxH3PreparedReference>& prepared_references,
                           ITokenizer& tokenizer, const MiniMaxH3ModuleLoader& loader,
                           cudaStream_t stream) {
    Ref2vaTextConditioning result;
    result.presentation = minimax_h3_build_ref2va_conditioner_presentation(
        request.prompt, request.references, prepared_references,
        [&tokenizer](const std::string& text) { return tokenizer.encode(text); });
    const auto compact = encode_ref2va_vision(result.presentation, loader, stream);
    const auto aligned = scatter_ref2va_vision_features(result.presentation, compact);
    auto main_bf16 = floats_to_bfloat16(aligned.vision_embeddings);
    std::array<std::vector<uint16_t>, 3> deepstack_bf16;
    for (std::size_t level = 0; level < deepstack_bf16.size(); ++level)
        deepstack_bf16[level] = floats_to_bfloat16(aligned.deepstack_embeddings[level]);
    auto module = loader("language_conditioner_plan", stream);
    module->set_timing_label("language_conditioner_plan");
    const int32_t rows = result.presentation.sequence_rows;
    TensorMap inputs;
    inputs.emplace("input_ids",
                   Tensor{result.presentation.input_ids.data(), {rows}, DType::kInt32});
    inputs.emplace("mrope_position_ids",
                   Tensor{result.presentation.mrope_position_ids.data(), {3, rows}, DType::kInt32});
    inputs.emplace("vision_embeddings",
                   Tensor{main_bf16.data(), {rows, kTextDim}, DType::kBFloat16});
    inputs.emplace("vision_selector",
                   Tensor{result.presentation.vision_selector.data(), {rows, 1}, DType::kInt32});
    for (std::size_t level = 0; level < deepstack_bf16.size(); ++level) {
        inputs.emplace("deepstack_embeddings_" + std::to_string(level),
                       Tensor{deepstack_bf16[level].data(), {rows, kTextDim}, DType::kBFloat16});
    }
    const auto outputs = module->forward(inputs);
    result.embeddings =
        copy_float(require_output(outputs, "encoder_hidden_states"),
                   static_cast<std::size_t>(rows) * kTextDim, "Ref2VA language conditioner");
    module->sync();
    return result;
}

std::vector<float> normalize_ref2va_visual(const MiniMaxH3PreparedReference& reference,
                                           int32_t frames) {
    const int32_t height = reference.kind == AudioVideoReferenceKind::kImage
                               ? reference.image.height
                               : reference.video.height;
    const int32_t width = reference.kind == AudioVideoReferenceKind::kImage ? reference.image.width
                                                                            : reference.video.width;
    const auto& pixels = reference.kind == AudioVideoReferenceKind::kImage ? reference.image.pixels
                                                                           : reference.video.pixels;
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    if (pixels.size() < static_cast<std::size_t>(frames) * plane * 3)
        throw std::runtime_error("MiniMax-H3 Ref2VA visual buffer is too short");
    std::vector<float> result(static_cast<std::size_t>(3) * frames * plane);
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (int32_t frame = 0; frame < frames; ++frame) {
            for (std::size_t pixel = 0; pixel < plane; ++pixel) {
                const auto source = (static_cast<std::size_t>(frame) * plane + pixel) * 3 + channel;
                const auto target =
                    (static_cast<std::size_t>(channel) * frames + frame) * plane + pixel;
                result[target] = (pixels[source] - kPixelMean[channel]) / kPixelStd[channel];
            }
        }
    }
    return result;
}

std::vector<float> encode_ref2va_visual_moments(const MiniMaxH3PreparedReference& reference,
                                                ITrtModule& module) {
    const bool image = reference.kind == AudioVideoReferenceKind::kImage;
    const int32_t source_frames =
        image ? 1 : minimax_h3_trim_reference_num_frames(reference.video.num_frames);
    const int32_t height = image ? reference.image.height : reference.video.height;
    const int32_t width = image ? reference.image.width : reference.video.width;
    const auto normalized = normalize_ref2va_visual(reference, source_frames);
    return encode_visual_moments(normalized, source_frames, height, width, module);
}

std::vector<float> sample_ref2va_posterior(const std::vector<float>& moments, int32_t frames,
                                           int32_t height, int32_t width) {
    const std::size_t plane = static_cast<std::size_t>(height) * width;
    const std::size_t latent_count = static_cast<std::size_t>(kLatentChannels) * frames * plane;
    if (moments.size() != 2 * latent_count)
        throw std::runtime_error("MiniMax-H3 Ref2VA VAE returned invalid posterior moments");
    const auto noise = minimax_h3::torch_cuda_normal(latent_count, 42);
    std::vector<float> result(latent_count);
    const std::size_t channel_stride = static_cast<std::size_t>(frames) * plane;
    for (std::size_t index = 0; index < latent_count; ++index) {
        const float log_variance = std::clamp(moments[latent_count + index], -30.0F, 20.0F);
        const float sampled = moments[index] + std::exp(0.5F * log_variance) * noise[index];
        const float rounded = __half2float(__float2half_rn(sampled));
        const int32_t channel = static_cast<int32_t>(index / channel_stride);
        result[index] = (rounded - kLatentMean[channel]) / kLatentStd[channel];
    }
    return result;
}

struct Ref2vaConditioning {
    std::vector<float> video_rows;
    std::vector<float> audio_rows;
    std::vector<MiniMaxH3PreparedReferenceLayout> layouts;
    uint64_t request_rng_offset{0};
};

void append_ref2va_visual_condition(Ref2vaConditioning& result,
                                    const MiniMaxH3PreparedReference& reference,
                                    std::size_t reference_index, int64_t request_seed,
                                    ITrtModule& module) {
    const bool image = reference.kind == AudioVideoReferenceKind::kImage;
    const int32_t source_frames =
        image ? 1 : minimax_h3_trim_reference_num_frames(reference.video.num_frames);
    const int32_t latent_height =
        (image ? reference.image.height : reference.video.height) / kMiniMaxH3VaeSpatialCompression;
    const int32_t latent_width =
        (image ? reference.image.width : reference.video.width) / kMiniMaxH3VaeSpatialCompression;
    auto moments = encode_ref2va_visual_moments(reference, module);
    auto latent = sample_ref2va_posterior(
        moments, make_minimax_h3_vae_temporal_chunk_plan(source_frames).output_moment_frames,
        latent_height, latent_width);
    const int32_t latent_frames = static_cast<int32_t>(
        latent.size() / (static_cast<std::size_t>(kLatentChannels) * latent_height * latent_width));
    auto rows = patchify_video_shape(latent, latent_frames, latent_height, latent_width);
    auto noise = minimax_h3::torch_cuda_normal(latent.size(), static_cast<uint64_t>(request_seed),
                                               result.request_rng_offset);
    result.request_rng_offset += minimax_h3::torch_cuda_normal_consumed_offset(latent.size());
    auto noise_rows = patchify_video_shape(noise, latent_frames, latent_height, latent_width);
    constexpr float timestep = 0.999F;
    constexpr float noise_scale = 1.0F - timestep;
    for (std::size_t index = 0; index < rows.size(); ++index)
        rows[index] = timestep * rows[index] + noise_scale * noise_rows[index];
    result.video_rows.insert(result.video_rows.end(), rows.begin(), rows.end());
    result.layouts[reference_index] = {reference.kind, latent_frames, latent_height, latent_width,
                                       0};
}

void append_ref2va_audio_condition(Ref2vaConditioning& result,
                                   const MiniMaxH3PreparedReference& reference,
                                   std::size_t reference_index, ITrtModule& module) {
    if (!reference.audio.has_value())
        return;
    const auto aligned = minimax_h3_align_reference_audio_for_vae(*reference.audio);
    TensorMap inputs;
    inputs.emplace("audio_samples", Tensor{const_cast<float*>(aligned.samples.data()),
                                           {2, 1, aligned.num_samples},
                                           DType::kFloat32});
    const auto outputs = module.forward(inputs);
    const int32_t latent_frames = aligned.num_samples / kAudioHopLength;
    auto rows = copy_float(require_output(outputs, "audio_condition_rows"),
                           static_cast<std::size_t>(2) * latent_frames * kAudioChannels,
                           "Ref2VA audio VAE encoder");
    result.audio_rows.insert(result.audio_rows.end(), rows.begin(), rows.end());
    result.layouts[reference_index].num_audio_latents = latent_frames;
}

void encode_ref2va_audio_conditions(Ref2vaConditioning& result,
                                    const std::vector<MiniMaxH3PreparedReference>& references,
                                    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream) {
    if (!std::any_of(references.begin(), references.end(),
                     [](const auto& reference) { return reference.audio.has_value(); }))
        return;
    auto module = loader("audio_vae_encoder_plan", stream);
    module->set_timing_label("audio_vae_encoder_plan");
    for (std::size_t index = 0; index < references.size(); ++index)
        append_ref2va_audio_condition(result, references[index], index, *module);
    module->sync();
}

Ref2vaConditioning
encode_ref2va_conditions(const std::vector<MiniMaxH3PreparedReference>& references,
                         int64_t request_seed, const MiniMaxH3ModuleLoader& loader,
                         cudaStream_t stream) {
    Ref2vaConditioning result;
    result.layouts.resize(references.size());
    for (std::size_t index = 0; index < references.size(); ++index)
        result.layouts[index].kind = references[index].kind;
    auto remaining_images = static_cast<int32_t>(
        std::count_if(references.begin(), references.end(), [](const auto& reference) {
            return reference.kind == AudioVideoReferenceKind::kImage;
        }));
    auto remaining_videos = static_cast<int32_t>(
        std::count_if(references.begin(), references.end(), [](const auto& reference) {
            return reference.kind == AudioVideoReferenceKind::kVideo;
        }));
    std::unique_ptr<ITrtModule> image_module;
    std::unique_ptr<ITrtModule> video_module;
    for (std::size_t index = 0; index < references.size(); ++index) {
        const auto& reference = references[index];
        if (reference.kind == AudioVideoReferenceKind::kAudio)
            continue;
        auto& module =
            reference.kind == AudioVideoReferenceKind::kImage ? image_module : video_module;
        auto& remaining =
            reference.kind == AudioVideoReferenceKind::kImage ? remaining_images : remaining_videos;
        const char* plan = reference.kind == AudioVideoReferenceKind::kImage
                               ? "vae_encoder_tile_t1_plan"
                               : "vae_encoder_tile_t17_plan";
        if (!module) {
            module = loader(plan, stream);
            module->set_timing_label(plan);
        }
        append_ref2va_visual_condition(result, reference, index, request_seed, *module);
        if (--remaining == 0)
            module.reset();
    }
    image_module.reset();
    video_module.reset();
    encode_ref2va_audio_conditions(result, references, loader, stream);
    return result;
}

void validate_ref2va_runtime_rows(int32_t text_rows, const MiniMaxH3PackedLayout& layout) {
    constexpr int32_t max_text_rows = 262144;
    constexpr int32_t max_condition_video_rows = 258120;
    constexpr int32_t max_condition_audio_rows = 2408;
    if (text_rows < 1 || text_rows > max_text_rows || layout.num_condition_video_rows < 4096 ||
        layout.num_condition_video_rows > max_condition_video_rows ||
        layout.num_condition_audio_rows < 0 ||
        layout.num_condition_audio_rows > max_condition_audio_rows)
        throw std::runtime_error("MiniMax-H3 Ref2VA request exceeds the bundled dynamic profile");
}

std::vector<float> distinct_conditioned_timesteps(float video_timestep, float audio_timestep,
                                                  bool has_video_conditions,
                                                  bool has_audio_conditions) {
    std::vector<float> values = {video_timestep, audio_timestep};
    if (has_video_conditions)
        values.push_back(0.999F);
    if (has_audio_conditions)
        values.push_back(1.0F);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

int32_t timestep_slot(const std::vector<float>& values, float value) {
    const auto found = std::find(values.begin(), values.end(), value);
    if (found == values.end())
        throw std::logic_error("MiniMax-H3 conditioned timestep slot is missing");
    return static_cast<int32_t>(std::distance(values.begin(), found));
}

std::vector<float> make_conditioned_adaln_features(const std::vector<float>& timesteps) {
    if (timesteps.empty() || timesteps.size() > kTimestepSlots)
        throw std::invalid_argument("MiniMax-H3 conditioned timestep count is invalid");
    std::vector<float> result(kTimestepSlots * 256, 0.0F);
    for (std::size_t slot = 0; slot < timesteps.size(); ++slot) {
        const auto features = timestep_features(timesteps[slot]);
        std::copy(features.begin(), features.end(), result.begin() + slot * 256);
    }
    return result;
}

struct ConditionedDenoiserStep {
    StepModulation modulation;
    std::vector<int32_t> timestep_indices;
};

std::vector<ConditionedDenoiserStep>
precompute_conditioned_steps(ITrtModule& module, const MiniMaxH3PackedLayout& layout,
                             const MiniMaxH3Schedule& video_schedule,
                             const MiniMaxH3Schedule& audio_schedule) {
    if (video_schedule.timesteps.size() != audio_schedule.timesteps.size())
        throw std::invalid_argument("MiniMax-H3 conditioned schedules have different lengths");
    std::vector<ConditionedDenoiserStep> result(video_schedule.timesteps.size());
    const bool has_video_conditions = layout.num_condition_video_rows > 0;
    const bool has_audio_conditions = layout.num_condition_audio_rows > 0;
    for (std::size_t step = 0; step < result.size(); ++step) {
        const auto values = distinct_conditioned_timesteps(
            video_schedule.timesteps[step], audio_schedule.timesteps[step], has_video_conditions,
            has_audio_conditions);
        auto features = make_conditioned_adaln_features(values);
        TensorMap inputs;
        inputs.emplace("timestep_features",
                       Tensor{features.data(), {kTimestepSlots, 256}, DType::kFloat32});
        const auto outputs = module.forward(inputs);
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            const std::string name = "block_modulation_" + std::to_string(layer);
            result[step].modulation.blocks[layer] =
                copy_raw(require_output(outputs, name), DType::kBFloat16,
                         static_cast<std::size_t>(kAdalnRows) * 6 * kHidden, name.c_str());
        }
        result[step].modulation.final =
            copy_raw(require_output(outputs, "final_modulation"), DType::kBFloat16,
                     static_cast<std::size_t>(kTimestepSlots) * 2 * kHidden, "final_modulation");
        const int32_t video_slot = timestep_slot(values, video_schedule.timesteps[step]);
        const int32_t audio_slot = timestep_slot(values, audio_schedule.timesteps[step]);
        const int32_t condition_video_slot =
            has_video_conditions ? timestep_slot(values, 0.999F) : video_slot;
        const int32_t condition_audio_slot =
            has_audio_conditions ? timestep_slot(values, 1.0F) : audio_slot;
        result[step].timestep_indices = make_minimax_h3_conditioned_timestep_indices(
            layout, video_slot, audio_slot, condition_video_slot, condition_audio_slot);
    }
    module.sync();
    return result;
}

struct DenoiserStats {
    int32_t full_steps{0};
    int32_t skipped_steps{0};
};

DenoiserStats run_fl2va_denoiser(ITrtModule& module, const std::vector<float>& text_embeddings,
                                 const MiniMaxH3PackedLayout& layout,
                                 const std::vector<float>& condition_rows,
                                 const MiniMaxH3Schedule& video_schedule,
                                 const MiniMaxH3Schedule& audio_schedule,
                                 std::vector<ConditionedDenoiserStep>& steps,
                                 std::vector<float>& video_rows, std::vector<float>& audio_rows) {
    DenoiserStats stats;
    module.reset_execution_context();
    std::vector<float> all_video(condition_rows.size() + video_rows.size());
    std::copy(condition_rows.begin(), condition_rows.end(), all_video.begin());
    for (std::size_t step = 0; step < steps.size(); ++step) {
        std::copy(video_rows.begin(), video_rows.end(),
                  all_video.begin() + static_cast<std::ptrdiff_t>(condition_rows.size()));
        TensorMap inputs;
        inputs.emplace("video_hidden_states",
                       Tensor{all_video.data(),
                              {static_cast<int64_t>(all_video.size() / kPatchDim), kPatchDim},
                              DType::kFloat32});
        inputs.emplace("audio_hidden_states",
                       Tensor{audio_rows.data(), {kAudioRows, kAudioChannels}, DType::kFloat32});
        inputs.emplace("encoder_hidden_states",
                       Tensor{const_cast<float*>(text_embeddings.data()),
                              {static_cast<int64_t>(text_embeddings.size() / kTextDim), kTextDim},
                              DType::kFloat32});
        inputs.emplace("position_ids", Tensor{const_cast<float*>(layout.position_ids.data()),
                                              {layout.sequence_rows, 3},
                                              DType::kFloat32});
        inputs.emplace("token_tags", Tensor{const_cast<int32_t*>(layout.token_tags.data()),
                                            {layout.sequence_rows},
                                            DType::kInt32});
        inputs.emplace(
            "timestep_indices",
            Tensor{steps[step].timestep_indices.data(), {layout.sequence_rows}, DType::kInt32});
        append_modulation_inputs(inputs, steps[step].modulation);
        const auto outputs = module.forward(inputs);
        auto video_velocity = copy_float(require_output(outputs, "video_velocity"),
                                         video_rows.size(), "conditioned video velocity");
        auto audio_velocity = copy_float(require_output(outputs, "audio_velocity"),
                                         audio_rows.size(), "conditioned audio velocity");
        minimax_h3_scheduler_step(video_rows.data(), video_velocity.data(), video_rows.size(),
                                  video_schedule.timesteps[step], video_schedule.sigmas[step],
                                  video_schedule.sigmas[step + 1]);
        minimax_h3_scheduler_step(audio_rows.data(), audio_velocity.data(), audio_rows.size(),
                                  audio_schedule.timesteps[step], audio_schedule.sigmas[step],
                                  audio_schedule.sigmas[step + 1]);
        ++stats.full_steps;
        std::cerr << "[minimax-h3] conditioned denoiser " << (step + 1) << '/' << steps.size()
                  << '\n';
    }
    module.sync();
    return stats;
}

DenoiserStats run_ref2va_denoiser(ITrtModule& module, const std::vector<float>& text_embeddings,
                                  const MiniMaxH3PackedLayout& layout,
                                  const std::vector<float>& condition_video_rows,
                                  const std::vector<float>& condition_audio_rows,
                                  const MiniMaxH3Schedule& video_schedule,
                                  const MiniMaxH3Schedule& audio_schedule,
                                  std::vector<ConditionedDenoiserStep>& steps,
                                  std::vector<float>& video_rows, std::vector<float>& audio_rows) {
    DenoiserStats stats;
    module.reset_execution_context();
    std::vector<float> all_video(condition_video_rows.size() + video_rows.size());
    std::vector<float> all_audio(condition_audio_rows.size() + audio_rows.size());
    std::copy(condition_video_rows.begin(), condition_video_rows.end(), all_video.begin());
    std::copy(condition_audio_rows.begin(), condition_audio_rows.end(), all_audio.begin());
    for (std::size_t step = 0; step < steps.size(); ++step) {
        std::copy(video_rows.begin(), video_rows.end(),
                  all_video.begin() + static_cast<std::ptrdiff_t>(condition_video_rows.size()));
        std::copy(audio_rows.begin(), audio_rows.end(),
                  all_audio.begin() + static_cast<std::ptrdiff_t>(condition_audio_rows.size()));
        TensorMap inputs;
        inputs.emplace("video_hidden_states",
                       Tensor{all_video.data(),
                              {static_cast<int64_t>(all_video.size() / kPatchDim), kPatchDim},
                              DType::kFloat32});
        inputs.emplace(
            "audio_hidden_states",
            Tensor{all_audio.data(),
                   {static_cast<int64_t>(all_audio.size() / kAudioChannels), kAudioChannels},
                   DType::kFloat32});
        inputs.emplace("encoder_hidden_states",
                       Tensor{const_cast<float*>(text_embeddings.data()),
                              {static_cast<int64_t>(text_embeddings.size() / kTextDim), kTextDim},
                              DType::kFloat32});
        inputs.emplace("video_indices", Tensor{const_cast<int32_t*>(layout.video_indices.data()),
                                               {static_cast<int64_t>(layout.video_indices.size())},
                                               DType::kInt32});
        inputs.emplace("audio_indices", Tensor{const_cast<int32_t*>(layout.audio_indices.data()),
                                               {static_cast<int64_t>(layout.audio_indices.size())},
                                               DType::kInt32});
        inputs.emplace("position_ids", Tensor{const_cast<float*>(layout.position_ids.data()),
                                              {layout.sequence_rows, 3},
                                              DType::kFloat32});
        inputs.emplace("token_tags", Tensor{const_cast<int32_t*>(layout.token_tags.data()),
                                            {layout.sequence_rows},
                                            DType::kInt32});
        inputs.emplace(
            "timestep_indices",
            Tensor{steps[step].timestep_indices.data(), {layout.sequence_rows}, DType::kInt32});
        append_modulation_inputs(inputs, steps[step].modulation);
        const auto outputs = module.forward(inputs);
        auto video_velocity = copy_float(require_output(outputs, "video_velocity"),
                                         video_rows.size(), "Ref2VA video velocity");
        auto audio_velocity = copy_float(require_output(outputs, "audio_velocity"),
                                         audio_rows.size(), "Ref2VA audio velocity");
        minimax_h3_scheduler_step(video_rows.data(), video_velocity.data(), video_rows.size(),
                                  video_schedule.timesteps[step], video_schedule.sigmas[step],
                                  video_schedule.sigmas[step + 1]);
        minimax_h3_scheduler_step(audio_rows.data(), audio_velocity.data(), audio_rows.size(),
                                  audio_schedule.timesteps[step], audio_schedule.sigmas[step],
                                  audio_schedule.sigmas[step + 1]);
        ++stats.full_steps;
        std::cerr << "[minimax-h3] Ref2VA denoiser " << (step + 1) << '/' << steps.size() << '\n';
    }
    module.sync();
    return stats;
}

bool device_tensors_ready(std::initializer_list<const DeviceTensor*> tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const DeviceTensor* tensor) {
        return tensor != nullptr && tensor->ok();
    });
}

void assign_timestep_rows(std::vector<int32_t>& result, const std::vector<int32_t>& rows,
                          std::size_t count, int32_t slot, const char* label) {
    if (count > rows.size())
        throw std::invalid_argument(std::string("MiniMax-H3 ") + label +
                                    " timestep count is invalid");
    for (std::size_t index = 0; index < count; ++index) {
        const int32_t row = rows[index];
        if (row < 0 || static_cast<std::size_t>(row) >= result.size())
            throw std::invalid_argument(std::string("MiniMax-H3 ") + label +
                                        " timestep row is out of bounds");
        result[static_cast<std::size_t>(row)] = slot;
    }
}

} // namespace

std::vector<float> minimax_h3_unpack_audio_latents(const std::vector<float>& rows) {
    return unpack_audio_latents(rows);
}

std::vector<int32_t> make_minimax_h3_conditioned_timestep_indices(
    const MiniMaxH3PackedLayout& layout, int32_t video_slot, int32_t audio_slot,
    int32_t condition_video_slot, int32_t condition_audio_slot) {
    if (layout.sequence_rows <= 0 || layout.num_condition_video_rows < 0 ||
        layout.num_condition_audio_rows < 0)
        throw std::invalid_argument("MiniMax-H3 conditioned timestep layout is invalid");
    std::vector<int32_t> result(static_cast<std::size_t>(layout.sequence_rows), video_slot);
    assign_timestep_rows(result, layout.audio_indices, layout.audio_indices.size(), audio_slot,
                         "audio");
    assign_timestep_rows(result, layout.video_indices,
                         static_cast<std::size_t>(layout.num_condition_video_rows),
                         condition_video_slot, "video-condition");
    assign_timestep_rows(result, layout.audio_indices,
                         static_cast<std::size_t>(layout.num_condition_audio_rows),
                         condition_audio_slot, "audio-condition");
    return result;
}

void validate_minimax_h3_request(const AudioVideoRequest& request) {
    if (request.prompt.empty())
        throw std::invalid_argument("MiniMax-H3 requires a non-empty prompt");

    const bool has_first_image = !request.first_image.pixels.empty();
    const bool has_last_image = !request.last_image.pixels.empty();
    if (!request.references.empty() && (has_first_image || has_last_image))
        throw std::invalid_argument(
            "MiniMax-H3 keyframes and omni-references are mutually exclusive");
    if (has_first_image)
        validate_media_image(request.first_image, "MiniMax-H3 first image");
    if (has_last_image)
        validate_media_image(request.last_image, "MiniMax-H3 last image");

    if (!request.references.empty()) {
        ReferenceSummary summary;
        for (std::size_t index = 0; index < request.references.size(); ++index)
            validate_reference(request.references[index], index, summary);
        validate_reference_summary(summary, request.references.size());
    }
    validate_generate_config(request.config);
}

MiniMaxH3PackedLayout
make_minimax_h3_fl2va_layout(const std::vector<int32_t>& text_token_tags, int32_t num_latent_frames,
                             int32_t latent_height, int32_t latent_width, int32_t num_audio_latents,
                             const std::vector<MiniMaxH3KeyframeAnchor>& keyframe_anchors) {
    validate_fl2va_layout_inputs(text_token_tags, num_latent_frames, latent_height, latent_width,
                                 num_audio_latents, keyframe_anchors);
    const auto geometry = make_fl2va_layout_geometry(
        text_token_tags.size(), keyframe_anchors.size(), num_latent_frames, latent_height,
        latent_width, num_audio_latents);
    auto result = initialize_fl2va_layout(text_token_tags, geometry);
    fill_fl2va_condition_layout(result, geometry, num_latent_frames, keyframe_anchors);
    fill_fl2va_audio_layout(result, geometry, num_audio_latents);
    fill_fl2va_video_layout(result, geometry, num_latent_frames);
    return result;
}

MiniMaxH3PackedLayout
make_minimax_h3_ref2va_layout(const std::vector<int32_t>& text_token_tags,
                              const std::vector<MiniMaxH3PreparedReferenceLayout>& references,
                              int32_t num_latent_frames, int32_t latent_height,
                              int32_t latent_width, int32_t num_audio_latents) {
    validate_fl2va_layout_inputs(text_token_tags, num_latent_frames, latent_height, latent_width,
                                 num_audio_latents, {});
    if (references.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA layout requires at least one reference");
    const auto geometry =
        make_ref2va_layout_geometry(text_token_tags.size(), references, num_latent_frames,
                                    latent_height, latent_width, num_audio_latents);
    auto result = initialize_ref2va_layout(text_token_tags, geometry);
    int32_t cursor = static_cast<int32_t>(text_token_tags.size());
    double rotary_time = static_cast<double>(text_token_tags.size());
    for (const auto& reference : references)
        append_ref2va_reference(result, reference, geometry.target_width_grid, cursor, rotary_time);
    result.num_condition_video_rows = geometry.condition_video_rows;
    result.num_condition_audio_rows = geometry.condition_audio_rows;

    append_ref2va_audio(result, cursor, num_audio_latents, rotary_time, geometry.target_width_grid);
    cursor += geometry.target_audio_rows;
    const MiniMaxH3PreparedReferenceLayout target{
        AudioVideoReferenceKind::kVideo, num_latent_frames, latent_height, latent_width, 0};
    (void)append_ref2va_video(result, cursor, target, rotary_time);
    cursor += geometry.target_video_rows;
    if (cursor != result.sequence_rows)
        throw std::logic_error("MiniMax-H3 Ref2VA packed layout construction failed");
    tag_ref2va_media_rows(result);
    return result;
}

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
    std::unique_ptr<ITrtModule> audio_vae;

    void clear_execution_modules();

    void load_text_embeddings(const std::string& requested_prompt, ITokenizer& tokenizer,
                              const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    void load_modulations(const MiniMaxH3Schedule& video_schedule,
                          const MiniMaxH3Schedule& audio_schedule,
                          const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    bool prepare_denoiser(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                          bool first_block_cache);
    DenoiserStats run_denoiser(bool first_block_cache, DenoiserMetadata& metadata,
                               const MiniMaxH3Schedule& video_schedule,
                               const MiniMaxH3Schedule& audio_schedule,
                               std::vector<float>& video_rows_host,
                               std::vector<float>& audio_rows_host, float cache_threshold,
                               cudaStream_t stream);
    bool prepare_vae(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                     bool first_block_cache);
    std::vector<float> decode_vae(bool first_block_cache, const std::vector<float>& latent,
                                  std::size_t expected_pixels, cudaStream_t stream);
    bool prepare_audio_vae(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    MultiChannelAudioResult decode_audio(const std::vector<float>& audio_rows);

    bool denoiser_is_resident(bool first_block_cache) const;
    void load_first_block_cache_denoiser(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    DenoiserStats run_first_block_cache_denoiser(DenoiserMetadata& metadata,
                                                 const MiniMaxH3Schedule& video_schedule,
                                                 const MiniMaxH3Schedule& audio_schedule,
                                                 std::vector<float>& video_rows_host,
                                                 std::vector<float>& audio_rows_host,
                                                 float cache_threshold, cudaStream_t stream);
    DenoiserStats run_monolithic_denoiser(DenoiserMetadata& metadata,
                                          const MiniMaxH3Schedule& video_schedule,
                                          const MiniMaxH3Schedule& audio_schedule,
                                          std::vector<float>& video_rows_host,
                                          std::vector<float>& audio_rows_host);
    bool vae_is_resident(bool first_block_cache) const;
    void load_first_block_cache_vae(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    std::vector<float> decode_first_block_cache_vae(std::size_t expected_pixels,
                                                    cudaStream_t stream);
    std::vector<float> decode_monolithic_vae(const std::vector<float>& latent,
                                             std::size_t expected_pixels);
};

void MiniMaxH3Pipeline::ResidentState::clear_execution_modules() {
    denoiser.reset();
    denoiser_head.reset();
    denoiser_tail.reset();
    denoiser_finish.reset();
    head_hidden.reset();
    head_residual.reset();
    previous_head_residual.reset();
    tail_residual.reset();
    video_rows.reset();
    audio_rows.reset();
    video_velocity.reset();
    audio_velocity.reset();
    vae.reset();
    audio_vae.reset();
    vae_latent_tiles.reset();
    vae_decoded_tiles.reset();
    vae_overlap.reset();
    frame_major_rgb.reset();
}

void MiniMaxH3Pipeline::ResidentState::load_text_embeddings(const std::string& requested_prompt,
                                                            ITokenizer& tokenizer,
                                                            const MiniMaxH3ModuleLoader& loader,
                                                            cudaStream_t stream) {
    // The text encoder is the largest plan. Drop resident execution modules
    // before loading it so prompt changes retain the previous peak-memory
    // behavior on smaller devices.
    clear_execution_modules();
    prompt.clear();
    text_embeddings.clear();
    const auto ids = tokenizer.encode(requested_prompt);
    if (ids.size() != kTextRows)
        throw std::invalid_argument(
            "MiniMax-H3 GB300 profile requires exactly 537 prompt tokens; got " +
            std::to_string(ids.size()));
    auto module = loader("text_encoder_plan", stream);
    module->set_timing_label("text_encoder_plan");
    TensorMap inputs;
    inputs.emplace("input_ids",
                   Tensor{const_cast<int32_t*>(ids.data()), {kTextRows}, DType::kInt32});
    const auto outputs = module->forward(inputs);
    text_embeddings = copy_float(require_output(outputs, "encoder_hidden_states"),
                                 static_cast<std::size_t>(kTextRows) * kTextDim, "text encoder");
    module->sync();
    prompt = requested_prompt;
}

void MiniMaxH3Pipeline::ResidentState::load_modulations(const MiniMaxH3Schedule& video_schedule,
                                                        const MiniMaxH3Schedule& audio_schedule,
                                                        const MiniMaxH3ModuleLoader& loader,
                                                        cudaStream_t stream) {
    auto module = loader("adaln_precompute_plan", stream);
    module->set_timing_label("adaln_precompute_plan");
    modulations = precompute_modulations(*module, video_schedule, audio_schedule);
    module->sync();
}

bool MiniMaxH3Pipeline::ResidentState::denoiser_is_resident(bool first_block_cache) const {
    if (!first_block_cache)
        return denoiser != nullptr;
    return denoiser_head != nullptr && denoiser_tail != nullptr && denoiser_finish != nullptr &&
           device_tensors_ready({head_hidden.get(), head_residual.get(),
                                 previous_head_residual.get(), tail_residual.get(),
                                 video_rows.get(), audio_rows.get(), video_velocity.get(),
                                 audio_velocity.get()});
}

void MiniMaxH3Pipeline::ResidentState::load_first_block_cache_denoiser(
    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream) {
    auto head = loader("denoiser_head_plan", stream);
    auto tail = loader("denoiser_tail_plan", stream);
    auto finish = loader("denoiser_finish_plan", stream);
    head->set_timing_label("denoiser_head_plan");
    tail->set_timing_label("denoiser_tail_plan");
    finish->set_timing_label("denoiser_finish_plan");

    DeviceTensor new_head_hidden({kSequenceRows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_head_residual({kSequenceRows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_previous_head_residual({kSequenceRows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_tail_residual({kSequenceRows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_video_rows({kVideoRows, kPatchDim}, DType::kFloat32, stream);
    DeviceTensor new_audio_rows({kAudioRows, kAudioChannels}, DType::kFloat32, stream);
    DeviceTensor new_video_velocity({kVideoRows, kPatchDim}, DType::kFloat32, stream);
    DeviceTensor new_audio_velocity({kAudioRows, kAudioChannels}, DType::kFloat32, stream);
    if (!device_tensors_ready({&new_head_hidden, &new_head_residual, &new_previous_head_residual,
                               &new_tail_residual, &new_video_rows, &new_audio_rows,
                               &new_video_velocity, &new_audio_velocity}))
        throw std::runtime_error("MiniMax-H3 failed to allocate FirstBlockCache buffers");

    auto resident_head_hidden = std::make_unique<DeviceTensor>(std::move(new_head_hidden));
    auto resident_head_residual = std::make_unique<DeviceTensor>(std::move(new_head_residual));
    auto resident_previous_head_residual =
        std::make_unique<DeviceTensor>(std::move(new_previous_head_residual));
    auto resident_tail_residual = std::make_unique<DeviceTensor>(std::move(new_tail_residual));
    auto resident_video_rows = std::make_unique<DeviceTensor>(std::move(new_video_rows));
    auto resident_audio_rows = std::make_unique<DeviceTensor>(std::move(new_audio_rows));
    auto resident_video_velocity = std::make_unique<DeviceTensor>(std::move(new_video_velocity));
    auto resident_audio_velocity = std::make_unique<DeviceTensor>(std::move(new_audio_velocity));

    bind_external_checked(*head, "head_hidden", resident_head_hidden->data(), false,
                          DType::kBFloat16, {kSequenceRows, kHidden});
    bind_external_checked(*head, "head_residual", resident_head_residual->data(), false,
                          DType::kBFloat16, {kSequenceRows, kHidden});
    bind_external_checked(*head, "previous_head_residual", resident_previous_head_residual->data(),
                          true, DType::kBFloat16, {kSequenceRows, kHidden});
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

    denoiser_head = std::move(head);
    denoiser_tail = std::move(tail);
    denoiser_finish = std::move(finish);
    head_hidden = std::move(resident_head_hidden);
    head_residual = std::move(resident_head_residual);
    previous_head_residual = std::move(resident_previous_head_residual);
    tail_residual = std::move(resident_tail_residual);
    video_rows = std::move(resident_video_rows);
    audio_rows = std::move(resident_audio_rows);
    video_velocity = std::move(resident_video_velocity);
    audio_velocity = std::move(resident_audio_velocity);
}

bool MiniMaxH3Pipeline::ResidentState::prepare_denoiser(const MiniMaxH3ModuleLoader& loader,
                                                        cudaStream_t stream,
                                                        bool first_block_cache) {
    const bool resident_hit = denoiser_is_resident(first_block_cache);
    if (resident_hit)
        return true;
    if (first_block_cache) {
        load_first_block_cache_denoiser(loader, stream);
    } else {
        denoiser = loader("denoiser_plan", stream);
        denoiser->set_timing_label("denoiser_plan");
    }
    return false;
}

DenoiserStats MiniMaxH3Pipeline::ResidentState::run_first_block_cache_denoiser(
    DenoiserMetadata& metadata, const MiniMaxH3Schedule& video_schedule,
    const MiniMaxH3Schedule& audio_schedule, std::vector<float>& video_rows_host,
    std::vector<float>& audio_rows_host, float cache_threshold, cudaStream_t stream) {
    DenoiserStats stats;
    auto& head = *denoiser_head;
    auto& tail = *denoiser_tail;
    auto& finish = *denoiser_finish;
    head.reset_execution_context();
    tail.reset_execution_context();
    finish.reset_execution_context();
    if (cudaMemsetAsync(previous_head_residual->data(), 0, previous_head_residual->nbytes(),
                        stream) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to reset FirstBlockCache state");
    if (!video_rows->copy_from_host(video_rows_host.data()) ||
        !audio_rows->copy_from_host(audio_rows_host.data()))
        throw std::runtime_error("MiniMax-H3 failed to upload FirstBlockCache latents");

    for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
        auto& modulation = modulations[step];
        TensorMap head_inputs;
        head_inputs.emplace("encoder_hidden_states",
                            Tensor{text_embeddings.data(), {kTextRows, kTextDim}, DType::kFloat32});
        head_inputs.emplace("position_ids",
                            Tensor{metadata.positions.data(), {kSequenceRows, 3}, DType::kFloat32});
        head_inputs.emplace("adaln_indices",
                            Tensor{metadata.adaln_indices.data(), {kSequenceRows}, DType::kInt32});
        append_block_modulation_inputs(head_inputs, modulation, 0, 1);
        const auto head_outputs = head.forward(head_inputs);
        const float metric =
            copy_float(require_output(head_outputs, "cache_metric"), 1, "cache metric")[0];
        const bool compute_tail = step == 0 || !std::isfinite(metric) || metric > cache_threshold;

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
            if (!previous_head_residual->copy_from(*head_residual))
                throw std::runtime_error("MiniMax-H3 failed to update FirstBlockCache state");
            ++stats.full_steps;
        } else {
            ++stats.skipped_steps;
        }

        TensorMap finish_inputs;
        finish_inputs.emplace(
            "timestep_indices",
            Tensor{metadata.timestep_indices.data(), {kSequenceRows}, DType::kInt32});
        append_final_modulation_input(finish_inputs, modulation);
        finish.forward_async(finish_inputs);
        minimax_h3::scheduler_step_cuda_async(
            static_cast<float*>(video_rows->data()),
            static_cast<const float*>(video_velocity->data()), video_rows_host.size(),
            video_schedule.timesteps[step], video_schedule.sigmas[step],
            video_schedule.sigmas[step + 1], stream);
        minimax_h3::scheduler_step_cuda_async(
            static_cast<float*>(audio_rows->data()),
            static_cast<const float*>(audio_velocity->data()), audio_rows_host.size(),
            audio_schedule.timesteps[step], audio_schedule.sigmas[step],
            audio_schedule.sigmas[step + 1], stream);
        std::cerr << "[minimax-h3] denoiser " << (step + 1) << '/'
                  << video_schedule.timesteps.size() << " cache_metric=" << metric
                  << " compute_tail=" << static_cast<int>(compute_tail) << '\n';
    }
    finish.sync();
    if (!audio_rows->copy_to_host(audio_rows_host.data()))
        throw std::runtime_error("MiniMax-H3 failed to download denoised audio latents");
    return stats;
}

DenoiserStats MiniMaxH3Pipeline::ResidentState::run_monolithic_denoiser(
    DenoiserMetadata& metadata, const MiniMaxH3Schedule& video_schedule,
    const MiniMaxH3Schedule& audio_schedule, std::vector<float>& video_rows_host,
    std::vector<float>& audio_rows_host) {
    DenoiserStats stats;
    auto& module = *denoiser;
    module.reset_execution_context();
    for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
        TensorMap inputs;
        inputs.emplace("video_hidden_states",
                       Tensor{video_rows_host.data(), {kVideoRows, kPatchDim}, DType::kFloat32});
        inputs.emplace(
            "audio_hidden_states",
            Tensor{audio_rows_host.data(), {kAudioRows, kAudioChannels}, DType::kFloat32});
        inputs.emplace("encoder_hidden_states",
                       Tensor{text_embeddings.data(), {kTextRows, kTextDim}, DType::kFloat32});
        inputs.emplace("position_ids",
                       Tensor{metadata.positions.data(), {kSequenceRows, 3}, DType::kFloat32});
        inputs.emplace("adaln_indices",
                       Tensor{metadata.adaln_indices.data(), {kSequenceRows}, DType::kInt32});
        inputs.emplace("timestep_indices",
                       Tensor{metadata.timestep_indices.data(), {kSequenceRows}, DType::kInt32});
        append_modulation_inputs(inputs, modulations[step]);
        const auto outputs = module.forward(inputs);
        auto video_all =
            copy_float(require_output(outputs, "video_velocity"),
                       static_cast<std::size_t>(kSequenceRows) * kPatchDim, "video velocity");
        auto audio_all =
            copy_float(require_output(outputs, "audio_velocity"),
                       static_cast<std::size_t>(kSequenceRows) * kAudioChannels, "audio velocity");
        const auto video_begin =
            video_all.begin() + static_cast<std::ptrdiff_t>(kTextRows + kAudioRows) * kPatchDim;
        std::vector<float> video_velocity_host(video_begin, video_begin + video_rows_host.size());
        const auto audio_begin =
            audio_all.begin() + static_cast<std::ptrdiff_t>(kTextRows) * kAudioChannels;
        std::vector<float> audio_velocity_host(audio_begin, audio_begin + audio_rows_host.size());
        minimax_h3_scheduler_step(video_rows_host.data(), video_velocity_host.data(),
                                  video_rows_host.size(), video_schedule.timesteps[step],
                                  video_schedule.sigmas[step], video_schedule.sigmas[step + 1]);
        minimax_h3_scheduler_step(audio_rows_host.data(), audio_velocity_host.data(),
                                  audio_rows_host.size(), audio_schedule.timesteps[step],
                                  audio_schedule.sigmas[step], audio_schedule.sigmas[step + 1]);
        ++stats.full_steps;
        std::cerr << "[minimax-h3] denoiser " << (step + 1) << '/'
                  << video_schedule.timesteps.size() << '\n';
    }
    module.sync();
    return stats;
}

DenoiserStats MiniMaxH3Pipeline::ResidentState::run_denoiser(
    bool first_block_cache, DenoiserMetadata& metadata, const MiniMaxH3Schedule& video_schedule,
    const MiniMaxH3Schedule& audio_schedule, std::vector<float>& video_rows_host,
    std::vector<float>& audio_rows_host, float cache_threshold, cudaStream_t stream) {
    if (first_block_cache) {
        return run_first_block_cache_denoiser(metadata, video_schedule, audio_schedule,
                                              video_rows_host, audio_rows_host, cache_threshold,
                                              stream);
    }
    return run_monolithic_denoiser(metadata, video_schedule, audio_schedule, video_rows_host,
                                   audio_rows_host);
}

bool MiniMaxH3Pipeline::ResidentState::vae_is_resident(bool first_block_cache) const {
    if (!first_block_cache)
        return vae != nullptr;
    return vae != nullptr && device_tensors_ready({vae_latent_tiles.get(), vae_decoded_tiles.get(),
                                                   vae_overlap.get(), frame_major_rgb.get()});
}

void MiniMaxH3Pipeline::ResidentState::load_first_block_cache_vae(
    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream) {
    auto module = loader("vae_tile_decoder_plan", stream);
    module->set_timing_label("vae_tile_decoder_plan");
    DeviceTensor latent_tiles(
        {kTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize},
        DType::kFloat32, stream);
    DeviceTensor decoded_tiles({kTileBatch, 3, kTileFrames, kTileSize, kTileSize}, DType::kFloat32,
                               stream);
    DeviceTensor overlap({3, 5, kOutputHeight, kOutputWidth}, DType::kFloat32, stream);
    DeviceTensor output_pixels({kOutputFrames, kOutputHeight, kOutputWidth, 3}, DType::kFloat32,
                               stream);
    if (!device_tensors_ready({&latent_tiles, &decoded_tiles, &overlap, &output_pixels}))
        throw std::runtime_error("MiniMax-H3 failed to allocate CUDA VAE buffers");

    auto resident_latent_tiles = std::make_unique<DeviceTensor>(std::move(latent_tiles));
    auto resident_decoded_tiles = std::make_unique<DeviceTensor>(std::move(decoded_tiles));
    auto resident_overlap = std::make_unique<DeviceTensor>(std::move(overlap));
    auto resident_frame_major_rgb = std::make_unique<DeviceTensor>(std::move(output_pixels));
    bind_external_checked(
        *module, "latent_tiles", resident_latent_tiles->data(), true, DType::kFloat32,
        {kTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize});
    bind_external_checked(*module, "decoded_tiles", resident_decoded_tiles->data(), false,
                          DType::kFloat32, {kTileBatch, 3, kTileFrames, kTileSize, kTileSize});

    vae = std::move(module);
    vae_latent_tiles = std::move(resident_latent_tiles);
    vae_decoded_tiles = std::move(resident_decoded_tiles);
    vae_overlap = std::move(resident_overlap);
    frame_major_rgb = std::move(resident_frame_major_rgb);
}

bool MiniMaxH3Pipeline::ResidentState::prepare_vae(const MiniMaxH3ModuleLoader& loader,
                                                   cudaStream_t stream, bool first_block_cache) {
    const bool resident_hit = vae_is_resident(first_block_cache);
    if (resident_hit)
        return true;
    if (first_block_cache) {
        load_first_block_cache_vae(loader, stream);
    } else {
        vae = loader("vae_tile_decoder_plan", stream);
        vae->set_timing_label("vae_tile_decoder_plan");
    }
    return false;
}

std::vector<float>
MiniMaxH3Pipeline::ResidentState::decode_first_block_cache_vae(std::size_t expected_pixels,
                                                               cudaStream_t stream) {
    auto& module = *vae;
    module.reset_execution_context();
    const auto latent_normalization = vae_latent_normalization();
    const auto pixel_normalization = vae_pixel_normalization();
    TensorMap no_inputs;
    for (int32_t clip_index = 0; clip_index < 7; ++clip_index) {
        minimax_h3::extract_vae_tiles_cuda_async(static_cast<const float*>(video_rows->data()),
                                                 static_cast<float*>(vae_latent_tiles->data()),
                                                 clip_index, latent_normalization, stream);
        module.forward_async(no_inputs);
        minimax_h3::assemble_vae_clip_cuda_async(
            static_cast<const float*>(vae_decoded_tiles->data()),
            static_cast<float*>(vae_overlap->data()), static_cast<float*>(frame_major_rgb->data()),
            clip_index, pixel_normalization, stream);
        std::cerr << "[minimax-h3] VAE clip " << (clip_index + 1) << "/7\n";
    }
    module.sync();
    std::vector<float> pixels(expected_pixels);
    if (!frame_major_rgb->copy_to_host(pixels.data()))
        throw std::runtime_error("MiniMax-H3 failed to download CUDA VAE output");
    return pixels;
}

std::vector<float>
MiniMaxH3Pipeline::ResidentState::decode_monolithic_vae(const std::vector<float>& latent,
                                                        std::size_t expected_pixels) {
    std::vector<float> video(expected_pixels);
    std::size_t decoded_frames = 0;
    std::vector<float> overlap;
    std::vector<float> clip;
    auto& module = *vae;
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
    return to_frame_major_rgb(video);
}

std::vector<float> MiniMaxH3Pipeline::ResidentState::decode_vae(bool first_block_cache,
                                                                const std::vector<float>& latent,
                                                                std::size_t expected_pixels,
                                                                cudaStream_t stream) {
    if (first_block_cache)
        return decode_first_block_cache_vae(expected_pixels, stream);
    return decode_monolithic_vae(latent, expected_pixels);
}

bool MiniMaxH3Pipeline::ResidentState::prepare_audio_vae(const MiniMaxH3ModuleLoader& loader,
                                                         cudaStream_t stream) {
    if (audio_vae != nullptr)
        return true;
    audio_vae = loader("audio_vae_decoder_plan", stream);
    audio_vae->set_timing_label("audio_vae_decoder_plan");
    return false;
}

MultiChannelAudioResult
MiniMaxH3Pipeline::ResidentState::decode_audio(const std::vector<float>& audio_rows_host) {
    auto audio_latents = unpack_audio_latents(audio_rows_host);
    TensorMap inputs;
    inputs.emplace("audio_latents", Tensor{audio_latents.data(),
                                           {kOutputAudioChannels, kAudioChannels, kAudioLatents},
                                           DType::kFloat32});
    const auto outputs = audio_vae->forward(inputs);
    auto samples = copy_float(require_output(outputs, "waveform"),
                              static_cast<std::size_t>(kOutputAudioChannels) * kOutputAudioSamples,
                              "audio VAE waveform");
    audio_vae->sync();
    if (!std::all_of(samples.begin(), samples.end(), [](float value) {
            return std::isfinite(value) && value >= -1.0F && value <= 1.0F;
        }))
        throw std::runtime_error("MiniMax-H3 audio VAE produced invalid waveform samples");

    MultiChannelAudioResult result;
    result.samples = std::move(samples);
    result.num_samples = kOutputAudioSamples;
    result.sample_rate = kAudioSampleRate;
    result.num_channels = kOutputAudioChannels;
    return result;
}

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
                                     bool first_block_cache, float cache_threshold,
                                     MiniMaxH3Workflow workflow,
                                     MiniMaxH3ProfileModuleLoader profile_loader)
    : loader_(std::move(loader)), profile_loader_(std::move(profile_loader)),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)),
      resident_(std::make_unique<ResidentState>()), first_block_cache_(first_block_cache),
      cache_threshold_(cache_threshold), workflow_(workflow) {
    if (!loader_ || !tokenizer_)
        throw std::invalid_argument("MiniMax-H3 pipeline requires a loader and tokenizer");
    if (!std::isfinite(cache_threshold_) || cache_threshold_ <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 cache threshold must be finite and positive");
    if (workflow_ != MiniMaxH3Workflow::kT2va && !profile_loader_)
        throw std::invalid_argument(
            "MiniMax-H3 conditioned workflows require an optimization-profile loader");
    if (workflow_ != MiniMaxH3Workflow::kT2va && first_block_cache_)
        throw std::invalid_argument(
            "MiniMax-H3 conditioned workflows do not support FirstBlockCache");
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
    if (workflow_ == MiniMaxH3Workflow::kFl2va) {
        AudioVideoRequest request;
        request.prompt = prompt;
        request.config = cfg;
        return generate_fl2va(request).video;
    }
    if (workflow_ == MiniMaxH3Workflow::kRef2va)
        throw std::runtime_error("MiniMax-H3 Ref2VA requires ordered references");
    return generate_joint(prompt, cfg, false).video;
}

AudioVideoResult MiniMaxH3Pipeline::generate_audio_video(const std::string& prompt,
                                                         const GenerateConfig& cfg) {
    if (workflow_ == MiniMaxH3Workflow::kFl2va) {
        AudioVideoRequest request;
        request.prompt = prompt;
        request.config = cfg;
        return generate_fl2va(request);
    }
    if (workflow_ == MiniMaxH3Workflow::kRef2va)
        throw std::runtime_error("MiniMax-H3 Ref2VA requires ordered references");
    return generate_joint(prompt, cfg, true);
}

AudioVideoResult MiniMaxH3Pipeline::generate_audio_video(const AudioVideoRequest& request) {
    validate_minimax_h3_request(request);
    if (workflow_ == MiniMaxH3Workflow::kFl2va) {
        if (!request.references.empty())
            throw std::runtime_error("MiniMax-H3 FL2VA does not accept omni-references");
        return generate_fl2va(request);
    }
    if (workflow_ == MiniMaxH3Workflow::kRef2va)
        return generate_ref2va(request);
    if (!request.first_image.pixels.empty() || !request.last_image.pixels.empty() ||
        !request.references.empty())
        throw std::runtime_error(
            "MiniMax-H3 T2VA bundle does not contain FL2VA conditioning plans");
    return generate_audio_video(request.prompt, request.config);
}

AudioVideoResult MiniMaxH3Pipeline::generate_fl2va(const AudioVideoRequest& request) {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    validate_minimax_h3_request(request);
    if (workflow_ != MiniMaxH3Workflow::kFl2va || !request.references.empty())
        throw std::invalid_argument("MiniMax-H3 FL2VA received an incompatible request");
    const int64_t seed = request.config.seed >= 0 ? request.config.seed : 0;
    const auto total_begin = Clock::now();
    resident_->clear_execution_modules();

    const auto keyframes = prepare_fl2va_keyframes(request);
    const auto text_begin = Clock::now();
    auto text = encode_fl2va_presentation(request, keyframes, *tokenizer_, loader_, stream_);
    const auto text_end = Clock::now();

    const auto condition_begin = Clock::now();
    auto conditions = encode_fl2va_keyframe_latents(keyframes, seed, loader_, stream_);
    const auto condition_end = Clock::now();
    std::vector<MiniMaxH3KeyframeAnchor> anchors;
    if (!request.first_image.pixels.empty())
        anchors.push_back(MiniMaxH3KeyframeAnchor::kFirst);
    if (!request.last_image.pixels.empty())
        anchors.push_back(MiniMaxH3KeyframeAnchor::kLast);
    auto layout = make_minimax_h3_fl2va_layout(text.presentation.h3_token_tags, kLatentFrames,
                                               kLatentHeight, kLatentWidth, kAudioLatents, anchors);
    if (layout.num_condition_video_rows * kPatchDim != static_cast<int32_t>(conditions.rows.size()))
        throw std::runtime_error("MiniMax-H3 FL2VA condition rows disagree with packed layout");

    const auto video_schedule = make_minimax_h3_schedule(kSteps, 12.0F);
    const auto audio_schedule = make_minimax_h3_schedule(kSteps, 3.0F);
    const auto adaln_begin = Clock::now();
    auto adaln = loader_("adaln_precompute_plan", stream_);
    adaln->set_timing_label("adaln_precompute_plan");
    auto denoiser_steps =
        precompute_conditioned_steps(*adaln, layout, video_schedule, audio_schedule);
    adaln.reset();
    const auto adaln_end = Clock::now();

    auto video_tensor = minimax_h3::torch_cuda_normal(
        kVideoLatentCount, static_cast<uint64_t>(seed), conditions.request_rng_offset);
    const uint64_t audio_offset = conditions.request_rng_offset +
                                  minimax_h3::torch_cuda_normal_consumed_offset(kVideoLatentCount);
    auto audio_rows =
        minimax_h3::torch_cuda_normal(kAudioCount, static_cast<uint64_t>(seed), audio_offset);
    auto video_rows = patchify_video(video_tensor);
    video_tensor.clear();
    video_tensor.shrink_to_fit();

    const auto denoiser_begin = Clock::now();
    auto denoiser =
        profile_loader_("fl2va_denoiser_plan", stream_, static_cast<int32_t>(keyframes.size()));
    denoiser->set_timing_label("fl2va_denoiser_plan");
    const auto denoiser_stats =
        run_fl2va_denoiser(*denoiser, text.embeddings, layout, conditions.rows, video_schedule,
                           audio_schedule, denoiser_steps, video_rows, audio_rows);
    denoiser.reset();
    const auto denoiser_end = Clock::now();

    auto latent = unpatchify_video(video_rows);
    denormalize_latents(latent);
    video_rows.clear();
    video_rows.shrink_to_fit();
    constexpr std::size_t expected_pixels =
        static_cast<std::size_t>(3) * kOutputFrames * kOutputHeight * kOutputWidth;
    const auto vae_begin = Clock::now();
    resident_->prepare_vae(loader_, stream_, false);
    auto pixels = resident_->decode_vae(false, latent, expected_pixels, stream_);
    const auto vae_end = Clock::now();
    latent.clear();
    latent.shrink_to_fit();

    const auto audio_begin = Clock::now();
    resident_->prepare_audio_vae(loader_, stream_);
    auto audio = resident_->decode_audio(audio_rows);
    const auto audio_end = Clock::now();
    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[minimax-h3.fl2va.perf] language_ms=" << milliseconds(text_begin, text_end)
              << " condition_ms=" << milliseconds(condition_begin, condition_end)
              << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
              << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " audio_vae_decoder_ms=" << milliseconds(audio_begin, audio_end)
              << " total_ms=" << milliseconds(total_begin, total_end)
              << " keyframes=" << keyframes.size()
              << " text_rows=" << text.presentation.sequence_rows
              << " full_denoiser_steps=" << denoiser_stats.full_steps << '\n';

    AudioVideoResult result;
    result.video.height = kOutputHeight;
    result.video.width = kOutputWidth;
    result.video.channels = 3;
    result.video.num_frames = kOutputFrames;
    result.video.pixels = std::move(pixels);
    result.audio = std::move(audio);
    return result;
}

AudioVideoResult MiniMaxH3Pipeline::generate_ref2va(const AudioVideoRequest& request) {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    validate_minimax_h3_request(request);
    if (workflow_ != MiniMaxH3Workflow::kRef2va || request.references.empty() ||
        !request.first_image.pixels.empty() || !request.last_image.pixels.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA received an incompatible request");
    const int64_t seed = request.config.seed >= 0 ? request.config.seed : 0;
    const auto total_begin = Clock::now();
    resident_->clear_execution_modules();

    const auto prepare_begin = Clock::now();
    const auto prepared_references = prepare_ref2va_references(request);
    const auto prepare_end = Clock::now();
    const auto text_begin = Clock::now();
    auto text =
        encode_ref2va_presentation(request, prepared_references, *tokenizer_, loader_, stream_);
    const auto text_end = Clock::now();

    const auto condition_begin = Clock::now();
    auto conditions = encode_ref2va_conditions(prepared_references, seed, loader_, stream_);
    const auto condition_end = Clock::now();
    auto layout =
        make_minimax_h3_ref2va_layout(text.presentation.h3_token_tags, conditions.layouts,
                                      kLatentFrames, kLatentHeight, kLatentWidth, kAudioLatents);
    validate_ref2va_runtime_rows(text.presentation.sequence_rows, layout);
    if (layout.num_condition_video_rows * kPatchDim !=
            static_cast<int32_t>(conditions.video_rows.size()) ||
        layout.num_condition_audio_rows * kAudioChannels !=
            static_cast<int32_t>(conditions.audio_rows.size()))
        throw std::runtime_error("MiniMax-H3 Ref2VA condition rows disagree with packed layout");

    const auto video_schedule = make_minimax_h3_schedule(kSteps, 12.0F);
    const auto audio_schedule = make_minimax_h3_schedule(kSteps, 3.0F);
    const auto adaln_begin = Clock::now();
    auto adaln = loader_("adaln_precompute_plan", stream_);
    adaln->set_timing_label("adaln_precompute_plan");
    auto denoiser_steps =
        precompute_conditioned_steps(*adaln, layout, video_schedule, audio_schedule);
    adaln.reset();
    const auto adaln_end = Clock::now();

    auto video_tensor = minimax_h3::torch_cuda_normal(
        kVideoLatentCount, static_cast<uint64_t>(seed), conditions.request_rng_offset);
    const uint64_t audio_offset = conditions.request_rng_offset +
                                  minimax_h3::torch_cuda_normal_consumed_offset(kVideoLatentCount);
    auto audio_rows =
        minimax_h3::torch_cuda_normal(kAudioCount, static_cast<uint64_t>(seed), audio_offset);
    auto video_rows = patchify_video(video_tensor);
    video_tensor.clear();
    video_tensor.shrink_to_fit();

    const auto denoiser_begin = Clock::now();
    auto denoiser = profile_loader_("ref2va_denoiser_plan", stream_, 0);
    denoiser->set_timing_label("ref2va_denoiser_plan");
    const auto denoiser_stats = run_ref2va_denoiser(
        *denoiser, text.embeddings, layout, conditions.video_rows, conditions.audio_rows,
        video_schedule, audio_schedule, denoiser_steps, video_rows, audio_rows);
    denoiser.reset();
    const auto denoiser_end = Clock::now();

    auto latent = unpatchify_video(video_rows);
    denormalize_latents(latent);
    video_rows.clear();
    video_rows.shrink_to_fit();
    constexpr std::size_t expected_pixels =
        static_cast<std::size_t>(3) * kOutputFrames * kOutputHeight * kOutputWidth;
    const auto vae_begin = Clock::now();
    resident_->prepare_vae(loader_, stream_, false);
    auto pixels = resident_->decode_vae(false, latent, expected_pixels, stream_);
    const auto vae_end = Clock::now();
    latent.clear();
    latent.shrink_to_fit();

    const auto audio_begin = Clock::now();
    resident_->prepare_audio_vae(loader_, stream_);
    auto audio = resident_->decode_audio(audio_rows);
    const auto audio_end = Clock::now();
    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[minimax-h3.ref2va.perf] prepare_ms=" << milliseconds(prepare_begin, prepare_end)
              << " language_ms=" << milliseconds(text_begin, text_end)
              << " condition_ms=" << milliseconds(condition_begin, condition_end)
              << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
              << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " audio_vae_decoder_ms=" << milliseconds(audio_begin, audio_end)
              << " total_ms=" << milliseconds(total_begin, total_end)
              << " references=" << prepared_references.size()
              << " text_rows=" << text.presentation.sequence_rows
              << " condition_video_rows=" << layout.num_condition_video_rows
              << " condition_audio_rows=" << layout.num_condition_audio_rows
              << " full_denoiser_steps=" << denoiser_stats.full_steps << '\n';

    AudioVideoResult result;
    result.video.height = kOutputHeight;
    result.video.width = kOutputWidth;
    result.video.channels = 3;
    result.video.num_frames = kOutputFrames;
    result.video.pixels = std::move(pixels);
    result.audio = std::move(audio);
    return result;
}

AudioVideoResult MiniMaxH3Pipeline::generate_joint(const std::string& prompt,
                                                   const GenerateConfig& cfg, bool decode_audio) {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    validate_generate_config(cfg);
    const int64_t seed = cfg.seed >= 0 ? cfg.seed : 0;
    const auto total_begin = Clock::now();

    const bool text_cache_hit = resident_->prompt == prompt && !resident_->text_embeddings.empty();
    const auto text_begin = Clock::now();
    if (!text_cache_hit)
        resident_->load_text_embeddings(prompt, *tokenizer_, loader_, stream_);
    const auto text_end = Clock::now();

    const auto video_schedule = make_minimax_h3_schedule(kSteps, 12.0F);
    const auto audio_schedule = make_minimax_h3_schedule(kSteps, 3.0F);
    const bool adaln_cache_hit = !resident_->modulations.empty();
    const auto adaln_begin = Clock::now();
    if (!adaln_cache_hit)
        resident_->load_modulations(video_schedule, audio_schedule, loader_, stream_);
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
        resident_->prepare_denoiser(loader_, stream_, first_block_cache_);
    const DenoiserStats denoiser_stats =
        resident_->run_denoiser(first_block_cache_, metadata, video_schedule, audio_schedule,
                                video_rows, audio_rows, cache_threshold_, stream_);
    const auto denoiser_end = Clock::now();

    std::vector<float> latent;
    if (!first_block_cache_) {
        latent = unpatchify_video(video_rows);
        denormalize_latents(latent);
    }
    video_rows.clear();
    video_rows.shrink_to_fit();
    const std::size_t expected_pixels =
        static_cast<std::size_t>(3) * kOutputFrames * kOutputHeight * kOutputWidth;
    const auto vae_begin = Clock::now();
    const bool vae_resident_hit = resident_->prepare_vae(loader_, stream_, first_block_cache_);
    auto pixels = resident_->decode_vae(first_block_cache_, latent, expected_pixels, stream_);
    const auto vae_end = Clock::now();
    latent.clear();
    latent.shrink_to_fit();

    MultiChannelAudioResult audio;
    bool audio_vae_resident_hit = false;
    const auto audio_vae_begin = Clock::now();
    if (decode_audio) {
        audio_vae_resident_hit = resident_->prepare_audio_vae(loader_, stream_);
        audio = resident_->decode_audio(audio_rows);
    }
    const auto audio_vae_end = Clock::now();
    audio_rows.clear();
    audio_rows.shrink_to_fit();

    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[minimax-h3.perf] text_encoder_ms=" << milliseconds(text_begin, text_end)
              << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
              << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " audio_vae_decoder_ms=" << milliseconds(audio_vae_begin, audio_vae_end)
              << " total_ms=" << milliseconds(total_begin, total_end)
              << " text_cache_hit=" << static_cast<int>(text_cache_hit)
              << " adaln_cache_hit=" << static_cast<int>(adaln_cache_hit)
              << " denoiser_resident_hit=" << static_cast<int>(denoiser_resident_hit)
              << " vae_resident_hit=" << static_cast<int>(vae_resident_hit)
              << " audio_vae_resident_hit=" << static_cast<int>(audio_vae_resident_hit)
              << " first_block_cache=" << static_cast<int>(first_block_cache_)
              << " cache_threshold=" << cache_threshold_
              << " full_denoiser_steps=" << denoiser_stats.full_steps
              << " skipped_denoiser_steps=" << denoiser_stats.skipped_steps << '\n';
    AudioVideoResult result;
    result.video.height = kOutputHeight;
    result.video.width = kOutputWidth;
    result.video.channels = 3;
    result.video.num_frames = kOutputFrames;
    result.video.pixels = std::move(pixels);
    result.audio = std::move(audio);
    return result;
}

} // namespace trtmc
