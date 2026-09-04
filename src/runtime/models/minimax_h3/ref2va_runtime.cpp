/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/ref2va_runtime.h"

#include "runtime/models/minimax_h3/torch_cuda_normal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kTargetFps = 24;
constexpr int32_t kAudioRate = 32000;
constexpr int32_t kAudioHop = 800;
constexpr int32_t kAudioChannels = 2;
constexpr int32_t kVisionPatch = 16;
constexpr int32_t kVisionMerge = 2;
constexpr int32_t kVisionTemporalPatch = 2;
constexpr int32_t kVisionPatchWidth = 1536;
constexpr int32_t kVisionTableSize = 48;
constexpr int32_t kVideoPatch = 2;
constexpr int32_t kVisionStartToken = 151652;
constexpr int32_t kVisionEndToken = 151653;
constexpr int32_t kImagePadToken = 151655;
constexpr int32_t kVideoPadToken = 151656;
constexpr int32_t kVideoTag = 0;
constexpr int32_t kTextTag = 1;
constexpr int32_t kAudioTag = 2;
constexpr float kConditionVideoTimestep = 0.999F;
constexpr float kConditionAudioTimestep = 1.0F;
constexpr int32_t kMinVideoRows = 18870;
constexpr int32_t kOptVideoRows = 44592;
constexpr int32_t kMinAudioRows = 414;
constexpr int32_t kOptAudioRows = 414;
constexpr int32_t kMinTextRows = 1;
constexpr int32_t kOptTextRows = 7433;
constexpr int32_t kMinPackedRows = 19285;
constexpr int32_t kOptPackedRows = 52439;
constexpr int32_t kLatentChannels = 24;
constexpr int32_t kPosteriorChannels = 48;
constexpr int32_t kVaeTile = 256;
constexpr int32_t kLatentTile = 16;
constexpr int32_t kTileOverlap = 64;
constexpr int32_t kTileAlignment = 16;
constexpr uint64_t kPosteriorSeed = 42;

constexpr std::array<float, 3> kPixelMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kPixelStd{0.229F, 0.224F, 0.225F};
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

std::size_t checked_product(std::initializer_list<std::size_t> values, const char* label) {
    std::size_t result = 1;
    for (std::size_t value : values) {
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value)
            throw std::overflow_error(std::string("MiniMax-H3 Ref2VA overflow in ") + label);
        result *= value;
    }
    return result;
}

int32_t checked_i32(int64_t value, const char* label) {
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
        throw std::overflow_error(std::string("MiniMax-H3 Ref2VA overflow in ") + label);
    return static_cast<int32_t>(value);
}

int64_t round_half_to_even(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("MiniMax-H3 Ref2VA cannot round a non-finite value");
    const double lower = std::floor(value);
    const double fraction = value - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && static_cast<int64_t>(lower) % 2 != 0)) {
        rounded += 1.0;
    }
    if (rounded < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("MiniMax-H3 Ref2VA rounded value is out of range");
    return static_cast<int64_t>(rounded);
}

std::string format_timestamp(double seconds) {
    const int64_t tenths = round_half_to_even(seconds * 10.0);
    std::ostringstream stream;
    if (tenths < 0)
        stream << '-';
    const uint64_t magnitude =
        tenths < 0 ? static_cast<uint64_t>(-(tenths + 1)) + 1U : static_cast<uint64_t>(tenths);
    stream << magnitude / 10U << '.' << magnitude % 10U;
    return stream.str();
}

void validate_image_metadata(const VideoImageInput& image, const char* label) {
    if (image.height <= 0 || image.width <= 0 || (image.channels != 3 && image.channels != 4))
        throw std::invalid_argument(std::string("MiniMax-H3 Ref2VA invalid ") + label);
    const auto required = checked_product({static_cast<std::size_t>(image.height),
                                           static_cast<std::size_t>(image.width),
                                           static_cast<std::size_t>(image.channels)},
                                          label);
    if (image.pixels.size() != required)
        throw std::invalid_argument(std::string("MiniMax-H3 Ref2VA malformed ") + label);
    if (image.width > 4LL * image.height || image.height > 4LL * image.width)
        throw std::invalid_argument("MiniMax-H3 Ref2VA visual references must be within 1:4..4:1");
}

double validate_audio_metadata(const AudioResult& audio, const char* label) {
    if (audio.sample_rate <= 0 || (audio.channels != 1 && audio.channels != 2) ||
        audio.samples.empty() ||
        audio.samples.size() % static_cast<std::size_t>(audio.channels) != 0 ||
        (audio.num_samples != 0 &&
         audio.num_samples != static_cast<int32_t>(audio.samples.size()))) {
        throw std::invalid_argument(std::string("MiniMax-H3 Ref2VA invalid ") + label);
    }
    const double frames =
        static_cast<double>(audio.samples.size() / static_cast<std::size_t>(audio.channels));
    return frames / audio.sample_rate;
}

void validate_duration(double duration, const char* label) {
    if (!std::isfinite(duration) || duration < 2.0 || duration > 15.0)
        throw std::invalid_argument(std::string("MiniMax-H3 Ref2VA ") + label +
                                    " must be 2..15 seconds long");
}

int32_t merged_vision_rows(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0 || height % kVisionPatch != 0 || width % kVisionPatch != 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA vision canvas is not patch aligned");
    const int64_t grid_h = height / kVisionPatch;
    const int64_t grid_w = width / kVisionPatch;
    if (grid_h % kVisionMerge || grid_w % kVisionMerge)
        throw std::invalid_argument("MiniMax-H3 Ref2VA vision grid is not merge aligned");
    return checked_i32((grid_h / kVisionMerge) * (grid_w / kVisionMerge), "vision rows");
}

struct Grid3 {
    int32_t temporal{1};
    int32_t height{0};
    int32_t width{0};
};

std::vector<int32_t> make_mrope(const std::vector<int32_t>& token_types,
                                const std::vector<Grid3>& image_grids,
                                const std::vector<Grid3>& video_grids) {
    if (token_types.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA presentation cannot be empty");
    std::vector<Grid3> expanded_videos;
    for (const Grid3& grid : video_grids) {
        if (grid.temporal <= 0)
            throw std::invalid_argument("MiniMax-H3 Ref2VA video grid is invalid");
        for (int32_t index = 0; index < grid.temporal; ++index)
            expanded_videos.push_back({1, grid.height, grid.width});
    }
    std::size_t image_cursor = 0;
    std::size_t video_cursor = 0;
    std::array<std::vector<int32_t>, 3> axes;
    for (auto& axis : axes)
        axis.reserve(token_types.size());
    int32_t position = 0;
    std::size_t cursor = 0;
    while (cursor < token_types.size()) {
        const int32_t modality = token_types[cursor];
        if (modality < 0 || modality > 2)
            throw std::invalid_argument("MiniMax-H3 Ref2VA Qwen token type is invalid");
        std::size_t end = cursor + 1;
        while (end < token_types.size() && token_types[end] == modality)
            ++end;
        const int32_t length = checked_i32(static_cast<int64_t>(end - cursor), "MRoPE run");
        if (modality == 0) {
            for (int32_t offset = 0; offset < length; ++offset) {
                for (auto& axis : axes)
                    axis.push_back(position + offset);
            }
            position += length;
        } else {
            const Grid3* grid = nullptr;
            if (modality == 1) {
                if (image_cursor >= image_grids.size())
                    throw std::invalid_argument("MiniMax-H3 Ref2VA image-grid metadata is short");
                grid = &image_grids[image_cursor++];
            } else {
                if (video_cursor >= expanded_videos.size())
                    throw std::invalid_argument("MiniMax-H3 Ref2VA video-grid metadata is short");
                grid = &expanded_videos[video_cursor++];
            }
            if (grid->height % kVisionMerge || grid->width % kVisionMerge)
                throw std::invalid_argument("MiniMax-H3 Ref2VA MRoPE grid is not merge aligned");
            const int32_t merged_h = grid->height / kVisionMerge;
            const int32_t merged_w = grid->width / kVisionMerge;
            if (grid->temporal * merged_h * merged_w != length)
                throw std::invalid_argument("MiniMax-H3 Ref2VA visual run/grid mismatch");
            for (int32_t temporal = 0; temporal < grid->temporal; ++temporal) {
                for (int32_t row = 0; row < merged_h; ++row) {
                    for (int32_t column = 0; column < merged_w; ++column) {
                        axes[0].push_back(position + temporal);
                        axes[1].push_back(position + row);
                        axes[2].push_back(position + column);
                    }
                }
            }
            position += std::max(grid->height, grid->width) / kVisionMerge;
        }
        cursor = end;
    }
    if (image_cursor != image_grids.size() || video_cursor != expanded_videos.size())
        throw std::invalid_argument("MiniMax-H3 Ref2VA has unused MRoPE grid metadata");
    std::vector<int32_t> result;
    result.reserve(token_types.size() * 3U);
    for (const auto& axis : axes)
        result.insert(result.end(), axis.begin(), axis.end());
    return result;
}

void validate_geometry(const Ref2vaEncodedReferenceGeometry& geometry) {
    if (geometry.kind == VideoReferenceKind::kAudio) {
        if (geometry.latent_frames != 0 || geometry.latent_height != 0 ||
            geometry.latent_width != 0 || geometry.audio_latents <= 0)
            throw std::invalid_argument("MiniMax-H3 Ref2VA audio geometry is invalid");
        return;
    }
    if (geometry.latent_frames <= 0 || geometry.latent_height <= 0 || geometry.latent_width <= 0 ||
        geometry.latent_height % kVideoPatch != 0 || geometry.latent_width % kVideoPatch != 0 ||
        geometry.audio_latents < 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA visual latent geometry is invalid");
    if (geometry.kind == VideoReferenceKind::kImage && geometry.latent_frames != 1)
        throw std::invalid_argument("MiniMax-H3 Ref2VA image must encode to one latent frame");
}

std::vector<double> spatial_axis(int32_t dimension, double sqrt_area) {
    const double ratio = dimension / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    const int32_t count = dimension / kVideoPatch;
    std::vector<double> result(static_cast<std::size_t>(count));
    for (int32_t index = 0; index < count; ++index)
        result[static_cast<std::size_t>(index)] =
            (left + ratio * static_cast<double>(index) / count) * 32.0;
    return result;
}

struct FrameGrid {
    std::vector<std::array<double, 2>> values;
    std::vector<double> widths;
};

FrameGrid make_frame_grid(int32_t height, int32_t width) {
    const double sqrt_area = std::sqrt(static_cast<double>(height) * width);
    const auto heights = spatial_axis(height, sqrt_area);
    const auto widths = spatial_axis(width, sqrt_area);
    FrameGrid result;
    result.widths = widths;
    result.values.reserve(heights.size() * widths.size());
    for (double y : heights) {
        for (double x : widths)
            result.values.push_back({y, x});
    }
    return result;
}

std::vector<double> temporal_grid(int32_t frames, double origin) {
    std::vector<double> result(static_cast<std::size_t>(frames));
    double cursor = origin;
    constexpr std::array<int32_t, 5> increments{1, 4, 4, 4, 4};
    for (int32_t frame = 0; frame < frames; ++frame) {
        result[static_cast<std::size_t>(frame)] = cursor;
        cursor += (5.0 / 3.0) * increments[static_cast<std::size_t>(frame % 5)];
    }
    return result;
}

double temporal_span(int32_t frames) {
    double result = 0.0;
    constexpr std::array<int32_t, 5> increments{1, 4, 4, 4, 4};
    for (int32_t frame = 0; frame < frames; ++frame)
        result += (5.0 / 3.0) * increments[static_cast<std::size_t>(frame % 5)];
    return result;
}

void fill_audio_positions(std::vector<float>& positions, int32_t begin, int32_t audio_latents,
                          double rotary_time, const std::vector<double>& widths) {
    if (audio_latents == 0)
        return;
    if (widths.empty())
        throw std::logic_error("MiniMax-H3 Ref2VA audio position width grid is empty");
    for (int32_t channel = 0; channel < kAudioChannels; ++channel) {
        for (int32_t frame = 0; frame < audio_latents; ++frame) {
            const int32_t row = begin + channel * audio_latents + frame;
            positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(rotary_time + frame);
            positions[static_cast<std::size_t>(row) * 3 + 2] =
                static_cast<float>(channel == 0 ? widths.front() : widths.back());
        }
    }
}

void fill_video_positions(std::vector<float>& positions, int32_t begin, int32_t frames,
                          const FrameGrid& frame_grid, double rotary_time) {
    const auto times = temporal_grid(frames, rotary_time);
    int32_t row = begin;
    for (double time : times) {
        for (const auto& spatial : frame_grid.values) {
            positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(time);
            positions[static_cast<std::size_t>(row) * 3 + 1] = static_cast<float>(spatial[0]);
            positions[static_cast<std::size_t>(row) * 3 + 2] = static_cast<float>(spatial[1]);
            ++row;
        }
    }
}

void require_static_input(ITrtModule& module, const std::string& name, DType dtype,
                          const std::vector<int64_t>& shape) {
    if (!module.has_input(name) || module.input_is_dynamic(name) ||
        module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
        throw std::runtime_error("MiniMax-H3 Ref2VA static input ABI mismatch for " + name);
}

void require_dynamic_input(ITrtModule& module, const std::string& name, DType dtype,
                           const std::vector<int64_t>& minimum, const std::vector<int64_t>& optimum,
                           const std::vector<int64_t>& maximum) {
    if (!module.has_input(name) || !module.input_is_dynamic(name) ||
        module.tensor_dtype(name) != dtype || module.optimization_profile_count() != 1 ||
        module.input_profile_shape(name, 0, ProfileShapeSelector::kMin) != minimum ||
        module.input_profile_shape(name, 0, ProfileShapeSelector::kOpt) != optimum ||
        module.input_profile_shape(name, 0, ProfileShapeSelector::kMax) != maximum)
        throw std::runtime_error("MiniMax-H3 Ref2VA dynamic input ABI mismatch for " + name);
}

void require_output(ITrtModule& module, const std::string& name, DType dtype,
                    const std::vector<int64_t>& maximum) {
    if (!module.has_output(name) || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != maximum)
        throw std::runtime_error("MiniMax-H3 Ref2VA output ABI mismatch for " + name);
}

void require_counts(ITrtModule& module, std::size_t inputs, std::size_t outputs,
                    const char* label) {
    if (!module.ok() || module.optimization_profile_count() != 1 ||
        module.input_info().size() != inputs || module.output_info().size() != outputs)
        throw std::runtime_error(std::string("MiniMax-H3 Ref2VA ") + label +
                                 " plan has an unexpected I/O contract");
}

const Tensor& find_output(const TensorMap& outputs, const std::string& name) {
    const auto iterator = outputs.find(name);
    if (iterator == outputs.end() || iterator->second.data == nullptr)
        throw std::runtime_error("MiniMax-H3 Ref2VA engine did not return " + name);
    return iterator->second;
}

Ref2vaOwnedTensor copy_owned(const Tensor& tensor, DType dtype, std::size_t expected_numel,
                             const std::string& label) {
    if (tensor.dtype != dtype || tensor.numel() != expected_numel || tensor.data == nullptr)
        throw std::runtime_error("MiniMax-H3 Ref2VA invalid " + label + " output");
    Ref2vaOwnedTensor result;
    result.dtype = tensor.dtype;
    result.shape = tensor.shape;
    result.bytes.resize(tensor.nbytes());
    std::memcpy(result.bytes.data(), tensor.data, tensor.nbytes());
    return result;
}

std::vector<float> copy_float_output(const Tensor& tensor, std::size_t count,
                                     const std::string& label) {
    if (tensor.dtype != DType::kFloat32 || tensor.numel() != count || tensor.data == nullptr)
        throw std::runtime_error("MiniMax-H3 Ref2VA invalid " + label + " output");
    const auto* values = static_cast<const float*>(tensor.data);
    return {values, values + count};
}

struct RefTileAxis {
    std::vector<int32_t> starts;
    std::vector<int32_t> overlaps;
};

RefTileAxis make_ref_tile_axis(int32_t length) {
    if (length < kVaeTile || length % kTileAlignment != 0)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA VAE axis must be at least 256 and divisible by 16");
    if (length == kVaeTile)
        return {{0}, {}};
    int32_t count = length / kVaeTile + (length % kVaeTile != 0 ? 1 : 0);
    while (static_cast<int64_t>(kVaeTile) * count -
               static_cast<int64_t>(kTileOverlap) * (count - 1) <
           length)
        ++count;
    RefTileAxis result;
    result.overlaps.assign(static_cast<std::size_t>(count - 1), kTileOverlap);
    const int32_t slack = kVaeTile * count - kTileOverlap * (count - 1) - length;
    if (slack < 0 || slack % kTileAlignment)
        throw std::logic_error("MiniMax-H3 Ref2VA VAE tile slack is invalid");
    for (int32_t index = 0; index < slack / kTileAlignment; ++index)
        result.overlaps[static_cast<std::size_t>(index % (count - 1))] += kTileAlignment;
    result.starts.reserve(static_cast<std::size_t>(count));
    result.starts.push_back(0);
    for (int32_t index = 0; index + 1 < count; ++index)
        result.starts.push_back(result.starts.back() + kVaeTile - result.overlaps[index]);
    if (result.starts.back() + kVaeTile != length)
        throw std::logic_error("MiniMax-H3 Ref2VA VAE tiles do not cover the axis");
    return result;
}

struct RefTileLayout {
    RefTileAxis y;
    RefTileAxis x;
    int32_t rows{0};
    int32_t columns{0};
    int32_t count{0};
};

RefTileLayout make_ref_tile_layout(int32_t height, int32_t width) {
    RefTileLayout result;
    result.y = make_ref_tile_axis(height);
    result.x = make_ref_tile_axis(width);
    result.rows = checked_i32(result.y.starts.size(), "VAE tile rows");
    result.columns = checked_i32(result.x.starts.size(), "VAE tile columns");
    result.count =
        checked_i32(static_cast<int64_t>(result.rows) * result.columns, "VAE tile count");
    return result;
}

std::size_t posterior_tile_index(int32_t tile, int32_t channel, int32_t y, int32_t x) {
    return (((static_cast<std::size_t>(tile) * kPosteriorChannels + channel) * kLatentTile + y) *
                kLatentTile +
            x);
}

std::vector<float> stitch_ref_posterior_tiles(const std::vector<float>& tiles, int32_t height,
                                              int32_t width) {
    const auto layout = make_ref_tile_layout(height, width);
    const std::size_t expected = checked_product(
        {static_cast<std::size_t>(layout.count), static_cast<std::size_t>(kPosteriorChannels),
         static_cast<std::size_t>(kLatentTile), static_cast<std::size_t>(kLatentTile)},
        "posterior tiles");
    if (tiles.size() != expected)
        throw std::invalid_argument("MiniMax-H3 Ref2VA posterior tile buffer is invalid");
    const int32_t latent_height = height / 16;
    const int32_t latent_width = width / 16;
    std::vector<float> result(static_cast<std::size_t>(kPosteriorChannels) * latent_height *
                              latent_width);
    for (int32_t tile_y = 0; tile_y < layout.rows; ++tile_y) {
        const int32_t y_start = layout.y.starts[static_cast<std::size_t>(tile_y)] / 16;
        const int32_t prior_y_overlap =
            tile_y > 0 ? layout.y.overlaps[static_cast<std::size_t>(tile_y - 1)] / 16 : 0;
        const int32_t kept_height =
            kLatentTile - (tile_y + 1 < layout.rows
                               ? layout.y.overlaps[static_cast<std::size_t>(tile_y)] / 16
                               : 0);
        for (int32_t tile_x = 0; tile_x < layout.columns; ++tile_x) {
            const int32_t x_start = layout.x.starts[static_cast<std::size_t>(tile_x)] / 16;
            const int32_t prior_x_overlap =
                tile_x > 0 ? layout.x.overlaps[static_cast<std::size_t>(tile_x - 1)] / 16 : 0;
            const int32_t kept_width =
                kLatentTile - (tile_x + 1 < layout.columns
                                   ? layout.x.overlaps[static_cast<std::size_t>(tile_x)] / 16
                                   : 0);
            const int32_t tile = tile_y * layout.columns + tile_x;
            for (int32_t channel = 0; channel < kPosteriorChannels; ++channel) {
                for (int32_t y = 0; y < kept_height; ++y) {
                    for (int32_t x = 0; x < kept_width; ++x) {
                        float value = tiles[posterior_tile_index(tile, channel, y, x)];
                        if (tile_y > 0 && y < prior_y_overlap) {
                            const float weight = static_cast<float>(y) / prior_y_overlap;
                            const float prior =
                                tiles[posterior_tile_index(tile - layout.columns, channel,
                                                           kLatentTile - prior_y_overlap + y, x)];
                            value = prior * (1.0F - weight) + value * weight;
                        }
                        if (tile_x > 0 && x < prior_x_overlap) {
                            const float weight = static_cast<float>(x) / prior_x_overlap;
                            const float prior = tiles[posterior_tile_index(
                                tile - 1, channel, y, kLatentTile - prior_x_overlap + x)];
                            value = prior * (1.0F - weight) + value * weight;
                        }
                        result[(static_cast<std::size_t>(channel) * latent_height + y_start + y) *
                                   latent_width +
                               x_start + x] = value;
                    }
                }
            }
        }
    }
    return result;
}

std::vector<float> sample_ref_posterior(const std::vector<float>& posterior, int32_t latent_frames,
                                        int32_t latent_height, int32_t latent_width) {
    if (latent_frames <= 0 || latent_height <= 0 || latent_width <= 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA posterior geometry is invalid");
    const std::size_t plane = checked_product({static_cast<std::size_t>(latent_frames),
                                               static_cast<std::size_t>(latent_height),
                                               static_cast<std::size_t>(latent_width)},
                                              "posterior plane");
    if (posterior.size() != static_cast<std::size_t>(kPosteriorChannels) * plane)
        throw std::invalid_argument("MiniMax-H3 Ref2VA posterior buffer is invalid");
    const auto epsilon =
        torch_cuda_normal(static_cast<std::size_t>(kLatentChannels) * plane, kPosteriorSeed);
    std::vector<float> result(static_cast<std::size_t>(kLatentChannels) * plane);
    for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
        for (std::size_t index = 0; index < plane; ++index) {
            const std::size_t sample = static_cast<std::size_t>(channel) * plane + index;
            const float mean = posterior[sample];
            const float logvar = std::clamp(
                posterior[static_cast<std::size_t>(channel + kLatentChannels) * plane + index],
                -30.0F, 20.0F);
            const float value = mean + std::exp(0.5F * logvar) * epsilon[sample];
            const float fp16 = __half2float(__float2half_rn(value));
            result[sample] = (fp16 - kLatentMean[static_cast<std::size_t>(channel)]) /
                             kLatentStd[static_cast<std::size_t>(channel)];
        }
    }
    return result;
}

std::vector<float> patchify_ref_video(const std::vector<float>& latent, int32_t frames,
                                      int32_t height, int32_t width) {
    const std::size_t plane =
        checked_product({static_cast<std::size_t>(frames), static_cast<std::size_t>(height),
                         static_cast<std::size_t>(width)},
                        "latent plane");
    if (frames <= 0 || height <= 0 || width <= 0 || height % 2 || width % 2 ||
        latent.size() != static_cast<std::size_t>(kLatentChannels) * plane)
        throw std::invalid_argument("MiniMax-H3 Ref2VA latent patch input is invalid");
    const int32_t rows =
        checked_i32(static_cast<int64_t>(frames) * (height / 2) * (width / 2), "patch rows");
    std::vector<float> result(static_cast<std::size_t>(rows) * 96U);
    std::size_t target = 0;
    for (int32_t frame = 0; frame < frames; ++frame) {
        for (int32_t y = 0; y < height; y += 2) {
            for (int32_t x = 0; x < width; x += 2) {
                for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                    for (int32_t patch_y = 0; patch_y < 2; ++patch_y) {
                        for (int32_t patch_x = 0; patch_x < 2; ++patch_x) {
                            const std::size_t source =
                                (((static_cast<std::size_t>(channel) * frames + frame) * height +
                                  y + patch_y) *
                                     width +
                                 x + patch_x);
                            result[target++] = latent[source];
                        }
                    }
                }
            }
        }
    }
    return result;
}

VideoImageInput copy_video_frame(const VideoClipInput& video, int32_t frame) {
    if (frame < 0 || frame >= video.num_frames)
        throw std::invalid_argument("MiniMax-H3 Ref2VA frame index is out of range");
    const std::size_t stride = checked_product({static_cast<std::size_t>(video.height),
                                                static_cast<std::size_t>(video.width),
                                                static_cast<std::size_t>(video.channels)},
                                               "video frame");
    VideoImageInput result;
    result.height = video.height;
    result.width = video.width;
    result.channels = video.channels;
    const auto begin = video.pixels.begin() + static_cast<std::ptrdiff_t>(frame * stride);
    result.pixels.assign(begin, begin + static_cast<std::ptrdiff_t>(stride));
    return result;
}

} // namespace

Ref2vaPreparedRequest prepare_ref2va_request(const VideoGenerationRequest& request,
                                             int32_t output_frames) {
    if (request.mode != VideoGenerationMode::kReferenceToVideoAudio)
        throw std::invalid_argument("MiniMax-H3 Ref2VA received a non-Ref2VA request");
    if (request.first_frame || request.last_frame)
        throw std::invalid_argument("MiniMax-H3 Ref2VA cannot contain FL2VA keyframes");
    if (output_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA output frame count must be positive");
    if (request.references.empty() || request.references.size() > kRef2vaMaxReferences)
        throw std::invalid_argument("MiniMax-H3 Ref2VA requires 1..12 ordered references");

    Ref2vaPreparedRequest result;
    for (const auto& reference : request.references) {
        switch (reference.kind) {
        case VideoReferenceKind::kImage:
            ++result.summary.image_count;
            validate_image_metadata(reference.image, "reference image");
            break;
        case VideoReferenceKind::kVideo: {
            ++result.summary.video_count;
            if (reference.video.num_frames <= 0 || reference.video.fps_numerator <= 0 ||
                reference.video.fps_denominator <= 0 || reference.video.height <= 0 ||
                reference.video.width <= 0 ||
                (reference.video.channels != 3 && reference.video.channels != 4))
                throw std::invalid_argument("MiniMax-H3 Ref2VA invalid reference video");
            const auto frame_values =
                checked_product({static_cast<std::size_t>(reference.video.num_frames),
                                 static_cast<std::size_t>(reference.video.height),
                                 static_cast<std::size_t>(reference.video.width),
                                 static_cast<std::size_t>(reference.video.channels)},
                                "reference video");
            if (reference.video.pixels.size() != frame_values)
                throw std::invalid_argument("MiniMax-H3 Ref2VA malformed reference video");
            if (reference.video.width > 4LL * reference.video.height ||
                reference.video.height > 4LL * reference.video.width)
                throw std::invalid_argument(
                    "MiniMax-H3 Ref2VA visual references must be within 1:4..4:1");
            const double duration = static_cast<double>(reference.video.num_frames) *
                                    reference.video.fps_denominator / reference.video.fps_numerator;
            validate_duration(duration, "reference video");
            result.summary.total_video_seconds += duration;
            if (!reference.video.soundtrack.samples.empty()) {
                (void)validate_audio_metadata(reference.video.soundtrack, "video soundtrack");
                ++result.summary.audio_bearing_count;
            }
            break;
        }
        case VideoReferenceKind::kAudio: {
            ++result.summary.explicit_audio_count;
            ++result.summary.audio_bearing_count;
            const double duration = validate_audio_metadata(reference.audio, "reference audio");
            validate_duration(duration, "reference audio");
            result.summary.total_explicit_audio_seconds += duration;
            break;
        }
        }
    }
    if (result.summary.image_count > kRef2vaMaxImages ||
        result.summary.video_count > kRef2vaMaxVideos ||
        result.summary.explicit_audio_count > kRef2vaMaxExplicitAudios)
        throw std::invalid_argument("MiniMax-H3 Ref2VA reference count exceeds a modality limit");
    if (result.summary.image_count == 0 && result.summary.video_count == 0)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA requires at least one reference image or video");
    if (result.summary.total_video_seconds > 15.0)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA total reference-video duration exceeds 15 seconds");
    if (result.summary.total_explicit_audio_seconds > 15.0)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA total explicit-audio duration exceeds 15 seconds");
    result.references = normalize_minimax_h3_references(request.references, output_frames);
    return result;
}

int32_t snap_ref2va_video_frames_down(int32_t frames) {
    if (frames <= 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA video frame count must be positive");
    const int32_t chunks = std::max(1, (frames - 5) / 17);
    return chunks * 17 + 5;
}

Ref2vaVideoEncodeSchedule make_ref2va_video_encode_schedule(int32_t normalized_frames) {
    Ref2vaVideoEncodeSchedule result;
    result.normalized_frames = normalized_frames;
    result.snapped_frames = snap_ref2va_video_frames_down(normalized_frames);
    if (result.snapped_frames > normalized_frames)
        throw std::invalid_argument("MiniMax-H3 Ref2VA video is too short for the 17*n+5 schedule");
    result.clip_count = (result.snapped_frames + 16) / 17;
    result.repeated_tail_frames = result.clip_count * 17 - result.snapped_frames;
    result.raw_posterior_frames = result.clip_count * 5;
    result.dropped_tail_latents = 3;
    result.output_latent_frames = result.raw_posterior_frames - result.dropped_tail_latents;
    if (result.repeated_tail_frames != 12)
        throw std::logic_error("MiniMax-H3 Ref2VA temporal encoder invariant failed");
    return result;
}

int32_t ref2va_audio_latent_frames(int32_t samples) {
    if (samples <= 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA audio sample count must be positive");
    return samples / kAudioHop + (samples % kAudioHop != 0 ? 1 : 0);
}

Ref2vaQwenVideoSample make_ref2va_qwen_video_sample(int32_t frames, double fps, double sample_fps,
                                                    int32_t temporal_patch) {
    if (frames <= 0 || !std::isfinite(fps) || !std::isfinite(sample_fps) || fps <= 0.0 ||
        sample_fps <= 0.0 || sample_fps > fps || temporal_patch <= 0)
        throw std::invalid_argument("MiniMax-H3 Ref2VA Qwen video sampling is invalid");
    Ref2vaQwenVideoSample result;
    const double stride = fps / sample_fps;
    double cursor = 0.0;
    while (round_half_to_even(cursor) < frames) {
        const int64_t rounded = round_half_to_even(cursor);
        const int32_t index = checked_i32(rounded, "Qwen sampled frame");
        if (result.frame_indices.empty() || index > result.frame_indices.back())
            result.frame_indices.push_back(index);
        cursor += stride;
    }
    if (result.frame_indices.size() < static_cast<std::size_t>(temporal_patch)) {
        const int64_t minimum = round_half_to_even((temporal_patch - 1) * stride) + 1;
        throw std::invalid_argument("MiniMax-H3 Ref2VA Qwen video needs at least " +
                                    std::to_string(minimum) + " normalized frames");
    }
    std::vector<double> timestamps(result.frame_indices.size());
    for (std::size_t index = 0; index < timestamps.size(); ++index)
        timestamps[index] = static_cast<double>(index) / sample_fps;
    while (timestamps.size() % static_cast<std::size_t>(temporal_patch) != 0)
        timestamps.push_back(timestamps.back());
    for (std::size_t index = 0; index < timestamps.size(); index += temporal_patch) {
        result.timestamp_seconds.push_back(
            (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2.0);
    }
    return result;
}

Ref2vaPresentationBlueprint
make_ref2va_presentation_blueprint(const std::string& prompt,
                                   const std::vector<VideoReferenceInput>& references) {
    if (references.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA presentation needs references");
    Ref2vaPresentationBlueprint result;
    int32_t image_count = 0;
    int32_t video_count = 0;
    int32_t audio_count = 0;
    for (std::size_t reference_index = 0; reference_index < references.size(); ++reference_index) {
        const auto& reference = references[reference_index];
        const bool has_audio = reference.kind == VideoReferenceKind::kAudio ||
                               (reference.kind == VideoReferenceKind::kVideo &&
                                !reference.video.soundtrack.samples.empty());
        if (has_audio) {
            result.pieces.push_back({Ref2vaPresentationModality::kText,
                                     "<Audio " + std::to_string(++audio_count) + ">: "});
        }
        if (reference.kind == VideoReferenceKind::kImage) {
            result.pieces.push_back({Ref2vaPresentationModality::kText,
                                     "<Picture " + std::to_string(++image_count) + ">: "});
            result.pieces.push_back({Ref2vaPresentationModality::kImage,
                                     {},
                                     reference.image.height,
                                     reference.image.width});
            result.vision_invocations.push_back(
                {reference_index, 0, 0, reference.image.height, reference.image.width, true});
        } else if (reference.kind == VideoReferenceKind::kVideo) {
            result.pieces.push_back({Ref2vaPresentationModality::kText,
                                     "<Video " + std::to_string(++video_count) + ">: "});
            const auto sample = make_ref2va_qwen_video_sample(reference.video.num_frames);
            for (std::size_t block = 0; block < sample.timestamp_seconds.size(); ++block) {
                result.pieces.push_back(
                    {Ref2vaPresentationModality::kText,
                     "<" + format_timestamp(sample.timestamp_seconds[block]) + " seconds>"});
                result.pieces.push_back({Ref2vaPresentationModality::kVideo,
                                         {},
                                         reference.video.height,
                                         reference.video.width});
                const std::size_t first_index = block * 2;
                const int32_t first = sample.frame_indices[first_index];
                const int32_t second = first_index + 1 < sample.frame_indices.size()
                                           ? sample.frame_indices[first_index + 1]
                                           : sample.frame_indices.back();
                result.vision_invocations.push_back({reference_index, first, second,
                                                     reference.video.height, reference.video.width,
                                                     false});
            }
        }
    }
    result.pieces.push_back({Ref2vaPresentationModality::kText, prompt});
    return result;
}

Ref2vaMaterializedPresentation
materialize_ref2va_presentation(const Ref2vaPresentationBlueprint& blueprint,
                                const ITokenizer& tokenizer) {
    if (blueprint.pieces.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA presentation blueprint is empty");
    Ref2vaMaterializedPresentation result;
    std::vector<Grid3> image_grids;
    std::vector<Grid3> video_grids;
    for (const auto& piece : blueprint.pieces) {
        if (piece.modality == Ref2vaPresentationModality::kText) {
            const auto ids = tokenizer.encode(piece.text);
            result.input_ids.insert(result.input_ids.end(), ids.begin(), ids.end());
            result.qwen_token_types.insert(result.qwen_token_types.end(), ids.size(), 0);
            result.h3_token_tags.insert(result.h3_token_tags.end(), ids.size(), kTextTag);
            continue;
        }
        const int32_t rows = merged_vision_rows(piece.height, piece.width);
        const int32_t grid_h = piece.height / kVisionPatch;
        const int32_t grid_w = piece.width / kVisionPatch;
        const bool image = piece.modality == Ref2vaPresentationModality::kImage;
        result.input_ids.push_back(kVisionStartToken);
        result.qwen_token_types.push_back(0);
        result.h3_token_tags.push_back(kVideoTag);
        const int32_t first_row = checked_i32(result.input_ids.size(), "vision row index");
        result.input_ids.insert(result.input_ids.end(), static_cast<std::size_t>(rows),
                                image ? kImagePadToken : kVideoPadToken);
        result.qwen_token_types.insert(result.qwen_token_types.end(),
                                       static_cast<std::size_t>(rows), image ? 1 : 2);
        result.h3_token_tags.insert(result.h3_token_tags.end(), static_cast<std::size_t>(rows),
                                    kVideoTag);
        for (int32_t row = 0; row < rows; ++row)
            result.vision_row_indices.push_back(first_row + row);
        result.input_ids.push_back(kVisionEndToken);
        result.qwen_token_types.push_back(0);
        result.h3_token_tags.push_back(kVideoTag);
        (image ? image_grids : video_grids).push_back({1, grid_h, grid_w});
    }
    if (result.input_ids.empty() || result.input_ids.size() > kRef2vaMaxTextRows)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA presentation exceeds the 262144-row Qwen context");
    result.mrope_position_ids = make_mrope(result.qwen_token_types, image_grids, video_grids);
    result.vision_rows = checked_i32(result.vision_row_indices.size(), "vision features");
    return result;
}

Ref2vaVisionInputs make_ref2va_vision_inputs(const VideoImageInput& first,
                                             const VideoImageInput& second) {
    validate_image_metadata(first, "first Qwen vision frame");
    validate_image_metadata(second, "second Qwen vision frame");
    if (first.channels != 3 || second.channels != 3 || first.height != second.height ||
        first.width != second.width || first.height % 32 || first.width % 32)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA Qwen frame pair must be equal-size RGB/multiple-of-32");
    const int32_t grid_height = first.height / kVisionPatch;
    const int32_t grid_width = first.width / kVisionPatch;
    Ref2vaVisionInputs result;
    result.patch_rows =
        checked_i32(static_cast<int64_t>(grid_height) * grid_width, "Qwen patch rows");
    result.pixel_values.resize(static_cast<std::size_t>(result.patch_rows) * kVisionPatchWidth);
    result.interp_indices.resize(static_cast<std::size_t>(result.patch_rows) * 4U);
    result.interp_weights.resize(static_cast<std::size_t>(result.patch_rows) * 4U);
    result.vision_position_ids.resize(static_cast<std::size_t>(result.patch_rows) * 2U);
    const std::array<const VideoImageInput*, kVisionTemporalPatch> frames{&first, &second};
    int32_t patch_row = 0;
    for (int32_t merge_y = 0; merge_y < grid_height; merge_y += kVisionMerge) {
        for (int32_t merge_x = 0; merge_x < grid_width; merge_x += kVisionMerge) {
            for (int32_t inner_y = 0; inner_y < kVisionMerge; ++inner_y) {
                for (int32_t inner_x = 0; inner_x < kVisionMerge; ++inner_x) {
                    const int32_t patch_y = merge_y + inner_y;
                    const int32_t patch_x = merge_x + inner_x;
                    result.vision_position_ids[static_cast<std::size_t>(patch_row) * 2U] = patch_y;
                    result.vision_position_ids[static_cast<std::size_t>(patch_row) * 2U + 1U] =
                        patch_x;
                    std::size_t column = static_cast<std::size_t>(patch_row) * kVisionPatchWidth;
                    for (int32_t channel = 0; channel < 3; ++channel) {
                        for (const auto* frame : frames) {
                            for (int32_t y = 0; y < kVisionPatch; ++y) {
                                for (int32_t x = 0; x < kVisionPatch; ++x) {
                                    const int32_t source_y = patch_y * kVisionPatch + y;
                                    const int32_t source_x = patch_x * kVisionPatch + x;
                                    const std::size_t source =
                                        (static_cast<std::size_t>(source_y) * frame->width +
                                         source_x) *
                                            3U +
                                        channel;
                                    result.pixel_values[column++] =
                                        (frame->pixels[source] - 0.5F) / 0.5F;
                                }
                            }
                        }
                    }
                    const double source_y = grid_height == 1
                                                ? 0.0
                                                : static_cast<double>(patch_y) *
                                                      (kVisionTableSize - 1) / (grid_height - 1);
                    const double source_x = grid_width == 1
                                                ? 0.0
                                                : static_cast<double>(patch_x) *
                                                      (kVisionTableSize - 1) / (grid_width - 1);
                    const int32_t y0 = static_cast<int32_t>(std::floor(source_y));
                    const int32_t x0 = static_cast<int32_t>(std::floor(source_x));
                    const int32_t y1 = std::min(y0 + 1, kVisionTableSize - 1);
                    const int32_t x1 = std::min(x0 + 1, kVisionTableSize - 1);
                    const float wy = static_cast<float>(source_y - y0);
                    const float wx = static_cast<float>(source_x - x0);
                    const std::array<int32_t, 4> indices{
                        y0 * kVisionTableSize + x0, y0 * kVisionTableSize + x1,
                        y1 * kVisionTableSize + x0, y1 * kVisionTableSize + x1};
                    const std::array<float, 4> weights{(1.0F - wy) * (1.0F - wx), (1.0F - wy) * wx,
                                                       wy * (1.0F - wx), wy * wx};
                    std::copy(indices.begin(), indices.end(),
                              result.interp_indices.begin() +
                                  static_cast<std::ptrdiff_t>(patch_row * 4));
                    std::copy(weights.begin(), weights.end(),
                              result.interp_weights.begin() +
                                  static_cast<std::ptrdiff_t>(patch_row * 4));
                    ++patch_row;
                }
            }
        }
    }
    if (patch_row != result.patch_rows)
        throw std::logic_error("MiniMax-H3 Ref2VA Qwen patch accounting failed");
    return result;
}

Ref2vaVisionFeatures run_ref2va_vision_encoder(ITrtModule& module,
                                               const Ref2vaVisionInputs& inputs) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kVisionEncoder);
    if (inputs.patch_rows <= 0 || inputs.patch_rows % 4 || inputs.patch_rows > 65536 ||
        inputs.pixel_values.size() !=
            static_cast<std::size_t>(inputs.patch_rows) * kVisionPatchWidth ||
        inputs.interp_indices.size() != static_cast<std::size_t>(inputs.patch_rows) * 4U ||
        inputs.interp_weights.size() != static_cast<std::size_t>(inputs.patch_rows) * 4U ||
        inputs.vision_position_ids.size() != static_cast<std::size_t>(inputs.patch_rows) * 2U)
        throw std::invalid_argument("MiniMax-H3 Ref2VA Qwen vision inputs are inconsistent");
    TensorMap plan_inputs;
    plan_inputs.emplace("pixel_values", Tensor{const_cast<float*>(inputs.pixel_values.data()),
                                               {inputs.patch_rows, kVisionPatchWidth},
                                               DType::kFloat32});
    plan_inputs.emplace("interp_indices", Tensor{const_cast<int32_t*>(inputs.interp_indices.data()),
                                                 {inputs.patch_rows, 4},
                                                 DType::kInt32});
    plan_inputs.emplace("interp_weights", Tensor{const_cast<float*>(inputs.interp_weights.data()),
                                                 {inputs.patch_rows, 4},
                                                 DType::kFloat32});
    plan_inputs.emplace("vision_position_ids",
                        Tensor{const_cast<int32_t*>(inputs.vision_position_ids.data()),
                               {inputs.patch_rows, 2},
                               DType::kInt32});
    const auto outputs = module.forward(plan_inputs);
    Ref2vaVisionFeatures result;
    result.rows = inputs.patch_rows / 4;
    const std::size_t count = static_cast<std::size_t>(result.rows) * 5120U;
    result.vision_embeds =
        copy_float_output(find_output(outputs, "vision_embeds"), count, "vision_embeds");
    result.deepstack_0 =
        copy_float_output(find_output(outputs, "deepstack_0"), count, "deepstack_0");
    result.deepstack_1 =
        copy_float_output(find_output(outputs, "deepstack_1"), count, "deepstack_1");
    result.deepstack_2 =
        copy_float_output(find_output(outputs, "deepstack_2"), count, "deepstack_2");
    return result;
}

Ref2vaVisionFeatures
run_ref2va_reference_vision_encoder(ITrtModule& module,
                                    const std::vector<VideoReferenceInput>& references,
                                    const Ref2vaPresentationBlueprint& blueprint) {
    Ref2vaVisionFeatures result;
    const auto append = [](std::vector<float>& destination, const std::vector<float>& source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    for (const auto& invocation : blueprint.vision_invocations) {
        if (invocation.reference_index >= references.size())
            throw std::invalid_argument(
                "MiniMax-H3 Ref2VA vision invocation references a missing input");
        const auto& reference = references[invocation.reference_index];
        VideoImageInput first;
        VideoImageInput second;
        if (invocation.is_image) {
            if (reference.kind != VideoReferenceKind::kImage)
                throw std::invalid_argument(
                    "MiniMax-H3 Ref2VA image invocation kind is inconsistent");
            first = reference.image;
            second = reference.image;
        } else {
            if (reference.kind != VideoReferenceKind::kVideo)
                throw std::invalid_argument(
                    "MiniMax-H3 Ref2VA video invocation kind is inconsistent");
            first = copy_video_frame(reference.video, invocation.first_frame);
            second = copy_video_frame(reference.video, invocation.second_frame);
        }
        const auto features =
            run_ref2va_vision_encoder(module, make_ref2va_vision_inputs(first, second));
        append(result.vision_embeds, features.vision_embeds);
        append(result.deepstack_0, features.deepstack_0);
        append(result.deepstack_1, features.deepstack_1);
        append(result.deepstack_2, features.deepstack_2);
        result.rows += features.rows;
    }
    return result;
}

std::vector<float> run_ref2va_text_encoder(ITrtModule& module,
                                           const Ref2vaMaterializedPresentation& presentation,
                                           const Ref2vaVisionFeatures& features) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kTextEncoder);
    const int32_t text_rows = checked_i32(presentation.input_ids.size(), "Qwen text rows");
    const int32_t vision_rows =
        checked_i32(presentation.vision_row_indices.size(), "Qwen vision rows");
    const std::size_t feature_values = static_cast<std::size_t>(vision_rows) * 5120U;
    if (text_rows <= 0 || text_rows > kRef2vaMaxTextRows || features.rows != vision_rows ||
        presentation.mrope_position_ids.size() != static_cast<std::size_t>(text_rows) * 3U ||
        presentation.h3_token_tags.size() != static_cast<std::size_t>(text_rows) ||
        features.vision_embeds.size() != feature_values ||
        features.deepstack_0.size() != feature_values ||
        features.deepstack_1.size() != feature_values ||
        features.deepstack_2.size() != feature_values)
        throw std::invalid_argument("MiniMax-H3 Ref2VA Qwen text inputs are inconsistent");
    std::vector<float> vision_mask(static_cast<std::size_t>(text_rows), 0.0F);
    for (int32_t index : presentation.vision_row_indices) {
        if (index < 0 || index >= text_rows)
            throw std::invalid_argument("MiniMax-H3 Ref2VA vision row index is out of range");
        vision_mask[static_cast<std::size_t>(index)] = 1.0F;
    }
    int32_t vision_count = vision_rows;
    int32_t dummy_vision_index = 0;
    std::vector<float> dummy_vision(5120U, 0.0F);
    const int32_t bound_vision_rows = std::max(vision_rows, 1);
    int32_t* vision_indices = vision_rows == 0
                                  ? &dummy_vision_index
                                  : const_cast<int32_t*>(presentation.vision_row_indices.data());
    TensorMap inputs;
    inputs.emplace(
        "input_ids",
        Tensor{const_cast<int32_t*>(presentation.input_ids.data()), {text_rows}, DType::kInt32});
    inputs.emplace("mrope_position_ids",
                   Tensor{const_cast<int32_t*>(presentation.mrope_position_ids.data()),
                          {3, text_rows},
                          DType::kInt32});
    inputs.emplace("vision_mask", Tensor{vision_mask.data(), {text_rows, 1}, DType::kFloat32});
    inputs.emplace("vision_count", Tensor{&vision_count, {1}, DType::kInt32});
    inputs.emplace("vision_row_indices",
                   Tensor{vision_indices, {bound_vision_rows}, DType::kInt32});
    for (auto [name, values] : {std::pair<const char*, const std::vector<float>*>(
                                    "vision_embeds", &features.vision_embeds),
                                {"deepstack_0", &features.deepstack_0},
                                {"deepstack_1", &features.deepstack_1},
                                {"deepstack_2", &features.deepstack_2}}) {
        float* data = vision_rows == 0 ? dummy_vision.data() : const_cast<float*>(values->data());
        inputs.emplace(name, Tensor{data, {bound_vision_rows, 5120}, DType::kFloat32});
    }
    const auto outputs = module.forward(inputs);
    return copy_float_output(find_output(outputs, "encoder_hidden_states"),
                             static_cast<std::size_t>(text_rows) * 5120U, "encoder_hidden_states");
}

Ref2vaEncodedCondition run_ref2va_image_vae_encoder(ITrtModule& module,
                                                    const VideoImageInput& image) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kKeyframeVaeEncoder);
    validate_image_metadata(image, "normalized reference image");
    if (image.channels != 3 || image.height % 32 || image.width % 32)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA normalized image must be RGB/multiple-of-32");
    const auto layout = make_ref_tile_layout(image.height, image.width);
    const std::size_t one_input_tile = 3U * kVaeTile * kVaeTile;
    const std::size_t one_output_tile =
        static_cast<std::size_t>(kPosteriorChannels) * kLatentTile * kLatentTile;
    std::vector<float> posterior_tiles(static_cast<std::size_t>(layout.count) * one_output_tile);
    for (int32_t first_tile = 0; first_tile < layout.count; first_tile += 33) {
        const int32_t batch = std::min(33, layout.count - first_tile);
        std::vector<float> pixels(static_cast<std::size_t>(batch) * one_input_tile);
        for (int32_t local = 0; local < batch; ++local) {
            const int32_t tile = first_tile + local;
            const int32_t tile_y = tile / layout.columns;
            const int32_t tile_x = tile % layout.columns;
            const int32_t y_start = layout.y.starts[static_cast<std::size_t>(tile_y)];
            const int32_t x_start = layout.x.starts[static_cast<std::size_t>(tile_x)];
            for (int32_t channel = 0; channel < 3; ++channel) {
                for (int32_t y = 0; y < kVaeTile; ++y) {
                    for (int32_t x = 0; x < kVaeTile; ++x) {
                        const std::size_t source =
                            (static_cast<std::size_t>(y_start + y) * image.width + x_start + x) *
                                3U +
                            channel;
                        const std::size_t target =
                            ((static_cast<std::size_t>(local) * 3U + channel) * kVaeTile + y) *
                                kVaeTile +
                            x;
                        pixels[target] =
                            (image.pixels[source] - kPixelMean[channel]) / kPixelStd[channel];
                    }
                }
            }
        }
        TensorMap inputs;
        inputs.emplace("pixel_tiles",
                       Tensor{pixels.data(), {batch, 3, 1, kVaeTile, kVaeTile}, DType::kFloat32});
        const auto outputs = module.forward(inputs);
        const auto values = copy_float_output(find_output(outputs, "posterior_parameter_tiles"),
                                              static_cast<std::size_t>(batch) * one_output_tile,
                                              "posterior_parameter_tiles");
        std::copy(values.begin(), values.end(),
                  posterior_tiles.begin() +
                      static_cast<std::ptrdiff_t>(first_tile * one_output_tile));
    }
    const auto posterior = stitch_ref_posterior_tiles(posterior_tiles, image.height, image.width);
    const int32_t latent_height = image.height / 16;
    const int32_t latent_width = image.width / 16;
    const auto latent = sample_ref_posterior(posterior, 1, latent_height, latent_width);
    Ref2vaEncodedCondition result;
    result.geometry = {VideoReferenceKind::kImage, 1, latent_height, latent_width, 0};
    result.video_hidden_states = patchify_ref_video(latent, 1, latent_height, latent_width);
    return result;
}

Ref2vaEncodedCondition run_ref2va_video_vae_encoder(ITrtModule& module,
                                                    const VideoClipInput& video) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kVideoVaeEncoder);
    if (video.num_frames <= 0 || video.height < kVaeTile || video.width < kVaeTile ||
        video.channels != 3 || video.height % 32 || video.width % 32 || video.fps_numerator != 24 ||
        video.fps_denominator != 1 ||
        video.pixels.size() != checked_product({static_cast<std::size_t>(video.num_frames),
                                                static_cast<std::size_t>(video.height),
                                                static_cast<std::size_t>(video.width), 3U},
                                               "normalized reference video"))
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA video VAE requires normalized RGB 24fps input");
    const auto schedule = make_ref2va_video_encode_schedule(video.num_frames);
    const auto layout = make_ref_tile_layout(video.height, video.width);
    const int32_t latent_height = video.height / 16;
    const int32_t latent_width = video.width / 16;
    const std::size_t latent_plane = static_cast<std::size_t>(latent_height) * latent_width;
    const std::size_t one_tile_frame =
        static_cast<std::size_t>(kPosteriorChannels) * kLatentTile * kLatentTile;
    const std::size_t one_output_tile = one_tile_frame * 5U;
    std::vector<float> raw_posterior(static_cast<std::size_t>(kPosteriorChannels) *
                                     schedule.raw_posterior_frames * latent_plane);
    for (int32_t clip = 0; clip < schedule.clip_count; ++clip) {
        std::vector<float> clip_tiles(static_cast<std::size_t>(layout.count) * one_output_tile);
        for (int32_t tile = 0; tile < layout.count; ++tile) {
            const int32_t tile_y = tile / layout.columns;
            const int32_t tile_x = tile % layout.columns;
            const int32_t y_start = layout.y.starts[static_cast<std::size_t>(tile_y)];
            const int32_t x_start = layout.x.starts[static_cast<std::size_t>(tile_x)];
            std::vector<float> pixels(3U * 17U * kVaeTile * kVaeTile);
            for (int32_t channel = 0; channel < 3; ++channel) {
                for (int32_t local_frame = 0; local_frame < 17; ++local_frame) {
                    const int32_t frame =
                        std::min(clip * 17 + local_frame, schedule.snapped_frames - 1);
                    for (int32_t y = 0; y < kVaeTile; ++y) {
                        for (int32_t x = 0; x < kVaeTile; ++x) {
                            const std::size_t source =
                                (((static_cast<std::size_t>(frame) * video.height + y_start + y) *
                                      video.width +
                                  x_start + x) *
                                     3U +
                                 channel);
                            const std::size_t target =
                                (((static_cast<std::size_t>(channel) * 17U + local_frame) *
                                      kVaeTile +
                                  y) *
                                     kVaeTile +
                                 x);
                            pixels[target] =
                                (video.pixels[source] - kPixelMean[channel]) / kPixelStd[channel];
                        }
                    }
                }
            }
            TensorMap inputs;
            inputs.emplace("pixel_tile_clip",
                           Tensor{pixels.data(), {1, 3, 17, kVaeTile, kVaeTile}, DType::kFloat32});
            const auto outputs = module.forward(inputs);
            const auto values =
                copy_float_output(find_output(outputs, "posterior_parameter_tile_clip"),
                                  one_output_tile, "posterior_parameter_tile_clip");
            std::copy(values.begin(), values.end(),
                      clip_tiles.begin() + static_cast<std::ptrdiff_t>(tile * one_output_tile));
        }
        for (int32_t local_frame = 0; local_frame < 5; ++local_frame) {
            std::vector<float> frame_tiles(static_cast<std::size_t>(layout.count) * one_tile_frame);
            for (int32_t tile = 0; tile < layout.count; ++tile) {
                for (int32_t channel = 0; channel < kPosteriorChannels; ++channel) {
                    const std::size_t source =
                        (static_cast<std::size_t>(tile) * kPosteriorChannels * 5U +
                         static_cast<std::size_t>(channel) * 5U + local_frame) *
                        kLatentTile * kLatentTile;
                    const std::size_t target =
                        (static_cast<std::size_t>(tile) * kPosteriorChannels + channel) *
                        kLatentTile * kLatentTile;
                    std::copy_n(clip_tiles.begin() + static_cast<std::ptrdiff_t>(source),
                                kLatentTile * kLatentTile,
                                frame_tiles.begin() + static_cast<std::ptrdiff_t>(target));
                }
            }
            const auto stitched =
                stitch_ref_posterior_tiles(frame_tiles, video.height, video.width);
            const int32_t global_frame = clip * 5 + local_frame;
            for (int32_t channel = 0; channel < kPosteriorChannels; ++channel) {
                const std::size_t source = static_cast<std::size_t>(channel) * latent_plane;
                const std::size_t target =
                    (static_cast<std::size_t>(channel) * schedule.raw_posterior_frames +
                     global_frame) *
                    latent_plane;
                std::copy_n(stitched.begin() + static_cast<std::ptrdiff_t>(source), latent_plane,
                            raw_posterior.begin() + static_cast<std::ptrdiff_t>(target));
            }
        }
    }
    std::vector<float> posterior(static_cast<std::size_t>(kPosteriorChannels) *
                                 schedule.output_latent_frames * latent_plane);
    for (int32_t channel = 0; channel < kPosteriorChannels; ++channel) {
        const std::size_t source =
            static_cast<std::size_t>(channel) * schedule.raw_posterior_frames * latent_plane;
        const std::size_t target =
            static_cast<std::size_t>(channel) * schedule.output_latent_frames * latent_plane;
        std::copy_n(raw_posterior.begin() + static_cast<std::ptrdiff_t>(source),
                    static_cast<std::size_t>(schedule.output_latent_frames) * latent_plane,
                    posterior.begin() + static_cast<std::ptrdiff_t>(target));
    }
    const auto latent =
        sample_ref_posterior(posterior, schedule.output_latent_frames, latent_height, latent_width);
    Ref2vaEncodedCondition result;
    result.geometry = {VideoReferenceKind::kVideo, schedule.output_latent_frames, latent_height,
                       latent_width, 0};
    result.video_hidden_states =
        patchify_ref_video(latent, schedule.output_latent_frames, latent_height, latent_width);
    return result;
}

Ref2vaEncodedCondition run_ref2va_audio_vae_encoder(ITrtModule& module, const AudioResult& audio,
                                                    const std::array<float, 32>& latent_mean,
                                                    const std::array<float, 32>& latent_std) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kAudioVaeEncoder);
    if (audio.sample_rate != kAudioRate || audio.channels != 2 || audio.samples.empty() ||
        audio.samples.size() % 2U ||
        (audio.num_samples != 0 && audio.num_samples != static_cast<int32_t>(audio.samples.size())))
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA audio VAE requires normalized 32kHz stereo input");
    const int32_t source_frames = checked_i32(audio.samples.size() / 2U, "audio frames");
    const int32_t latent_frames = ref2va_audio_latent_frames(source_frames);
    const int32_t padded_frames = latent_frames * kAudioHop;
    if (padded_frames < 64000 || padded_frames > 480000)
        throw std::invalid_argument("MiniMax-H3 Ref2VA audio exceeds encoder capacity");
    std::vector<float> separated(static_cast<std::size_t>(2) * padded_frames, 0.0F);
    for (int32_t frame = 0; frame < source_frames; ++frame) {
        separated[static_cast<std::size_t>(frame)] = audio.samples[frame * 2U];
        separated[static_cast<std::size_t>(padded_frames) + frame] = audio.samples[frame * 2U + 1U];
    }
    TensorMap inputs;
    inputs.emplace("audio_samples",
                   Tensor{separated.data(), {2, 1, padded_frames}, DType::kFloat32});
    const auto outputs = module.forward(inputs);
    const auto posterior =
        copy_float_output(find_output(outputs, "posterior_mean"),
                          static_cast<std::size_t>(2) * 32U * latent_frames, "posterior_mean");
    Ref2vaEncodedCondition result;
    result.geometry = {VideoReferenceKind::kAudio, 0, 0, 0, latent_frames};
    result.audio_hidden_states.resize(static_cast<std::size_t>(2) * latent_frames * 32U);
    for (int32_t channel = 0; channel < 32; ++channel) {
        if (!std::isfinite(latent_mean[channel]) || !std::isfinite(latent_std[channel]) ||
            latent_std[channel] <= 0.0F)
            throw std::invalid_argument("MiniMax-H3 Ref2VA audio latent normalization is invalid");
    }
    for (int32_t side = 0; side < 2; ++side) {
        for (int32_t frame = 0; frame < latent_frames; ++frame) {
            for (int32_t channel = 0; channel < 32; ++channel) {
                const std::size_t source =
                    (static_cast<std::size_t>(side) * 32U + channel) * latent_frames + frame;
                const std::size_t target =
                    (static_cast<std::size_t>(side) * latent_frames + frame) * 32U + channel;
                result.audio_hidden_states[target] =
                    (posterior[source] - latent_mean[channel]) / latent_std[channel];
            }
        }
    }
    return result;
}

int32_t Ref2vaEncodedReferenceGeometry::video_rows() const {
    if (kind == VideoReferenceKind::kAudio)
        return 0;
    return checked_i32(static_cast<int64_t>(latent_frames) * (latent_height / kVideoPatch) *
                           (latent_width / kVideoPatch),
                       "encoded reference video rows");
}

int32_t Ref2vaEncodedReferenceGeometry::audio_rows() const {
    return checked_i32(static_cast<int64_t>(audio_latents) * kAudioChannels,
                       "encoded reference audio rows");
}

int32_t Ref2vaPackedLayout::sequence_length() const {
    return checked_i32(static_cast<int64_t>(token_tags.size()), "packed sequence");
}

Ref2vaPackedLayout
make_ref2va_packed_layout(const std::vector<int32_t>& text_token_tags,
                          const std::vector<Ref2vaEncodedReferenceGeometry>& references,
                          int32_t target_latent_frames, int32_t target_latent_height,
                          int32_t target_latent_width, int32_t target_audio_latents) {
    if (text_token_tags.empty() || text_token_tags.size() > kRef2vaMaxTextRows ||
        std::any_of(text_token_tags.begin(), text_token_tags.end(),
                    [](int32_t tag) { return tag != kVideoTag && tag != kTextTag; }))
        throw std::invalid_argument("MiniMax-H3 Ref2VA text token tags are invalid");
    for (const auto& reference : references)
        validate_geometry(reference);
    if (target_latent_frames <= 0 || target_latent_height <= 0 || target_latent_width <= 0 ||
        target_audio_latents <= 0 || target_latent_height % kVideoPatch ||
        target_latent_width % kVideoPatch)
        throw std::invalid_argument("MiniMax-H3 Ref2VA target latent geometry is invalid");

    const int32_t text_rows = checked_i32(text_token_tags.size(), "text rows");
    int64_t condition_video_rows = 0;
    int64_t condition_audio_rows = 0;
    for (const auto& reference : references) {
        condition_video_rows += reference.video_rows();
        condition_audio_rows += reference.audio_rows();
    }
    const int32_t target_video_rows =
        checked_i32(static_cast<int64_t>(target_latent_frames) *
                        (target_latent_height / kVideoPatch) * (target_latent_width / kVideoPatch),
                    "target video rows");
    const int32_t target_audio_rows = checked_i32(
        static_cast<int64_t>(target_audio_latents) * kAudioChannels, "target audio rows");
    const int32_t sequence_rows =
        checked_i32(static_cast<int64_t>(text_rows) + condition_video_rows + condition_audio_rows +
                        target_video_rows + target_audio_rows,
                    "packed sequence rows");
    if (condition_video_rows + target_video_rows > kRef2vaMaxVideoRows ||
        condition_audio_rows + target_audio_rows > kRef2vaMaxAudioRows ||
        sequence_rows > kRef2vaMaxPackedRows)
        throw std::invalid_argument("MiniMax-H3 Ref2VA request exceeds the plan capacity");

    Ref2vaPackedLayout result;
    result.condition_video_rows = checked_i32(condition_video_rows, "condition video rows");
    result.condition_audio_rows = checked_i32(condition_audio_rows, "condition audio rows");
    result.position_ids.assign(static_cast<std::size_t>(sequence_rows) * 3U, 0.0F);
    result.token_tags.resize(static_cast<std::size_t>(sequence_rows));
    result.text_indices.resize(static_cast<std::size_t>(text_rows));
    std::iota(result.text_indices.begin(), result.text_indices.end(), 0);
    std::copy(text_token_tags.begin(), text_token_tags.end(), result.token_tags.begin());
    for (int32_t row = 0; row < text_rows; ++row)
        result.position_ids[static_cast<std::size_t>(row) * 3] = static_cast<float>(row);

    const auto target_grid = make_frame_grid(target_latent_height, target_latent_width);
    int32_t cursor = text_rows;
    double rotary_time = text_rows;
    for (const auto& reference : references) {
        if (reference.kind == VideoReferenceKind::kImage) {
            const int32_t begin = cursor;
            const int32_t end = begin + reference.video_rows();
            for (int32_t row = begin; row < end; ++row)
                result.video_indices.push_back(row);
            const auto grid = make_frame_grid(reference.latent_height, reference.latent_width);
            fill_video_positions(result.position_ids, begin, 1, grid, rotary_time);
            cursor = end;
            rotary_time += 1.0;
        } else if (reference.kind == VideoReferenceKind::kAudio) {
            const int32_t begin = cursor;
            const int32_t end = begin + reference.audio_rows();
            for (int32_t row = begin; row < end; ++row)
                result.audio_indices.push_back(row);
            fill_audio_positions(result.position_ids, begin, reference.audio_latents, rotary_time,
                                 target_grid.widths);
            cursor = end;
            rotary_time += reference.audio_latents;
        } else {
            const int32_t audio_begin = cursor;
            const int32_t video_begin = audio_begin + reference.audio_rows();
            const int32_t end = video_begin + reference.video_rows();
            for (int32_t row = audio_begin; row < video_begin; ++row)
                result.audio_indices.push_back(row);
            for (int32_t row = video_begin; row < end; ++row)
                result.video_indices.push_back(row);
            const auto grid = make_frame_grid(reference.latent_height, reference.latent_width);
            fill_audio_positions(result.position_ids, audio_begin, reference.audio_latents,
                                 rotary_time, grid.widths);
            fill_video_positions(result.position_ids, video_begin, reference.latent_frames, grid,
                                 rotary_time);
            cursor = end;
            rotary_time += std::max(static_cast<double>(reference.audio_latents),
                                    temporal_span(reference.latent_frames));
        }
    }

    const int32_t target_audio_begin = cursor;
    const int32_t target_video_begin = target_audio_begin + target_audio_rows;
    fill_audio_positions(result.position_ids, target_audio_begin, target_audio_latents, rotary_time,
                         target_grid.widths);
    fill_video_positions(result.position_ids, target_video_begin, target_latent_frames, target_grid,
                         rotary_time);
    for (int32_t row = target_audio_begin; row < target_video_begin; ++row)
        result.audio_indices.push_back(row);
    for (int32_t row = target_video_begin; row < sequence_rows; ++row)
        result.video_indices.push_back(row);
    std::fill(result.token_tags.begin() + target_audio_begin,
              result.token_tags.begin() + target_video_begin, kAudioTag);
    std::fill(result.token_tags.begin() + target_video_begin, result.token_tags.end(), kVideoTag);
    for (int32_t index : result.audio_indices)
        result.token_tags[static_cast<std::size_t>(index)] = kAudioTag;
    for (int32_t index : result.video_indices)
        result.token_tags[static_cast<std::size_t>(index)] = kVideoTag;
    if (cursor != target_audio_begin ||
        static_cast<int32_t>(result.video_indices.size()) !=
            result.condition_video_rows + target_video_rows ||
        static_cast<int32_t>(result.audio_indices.size()) !=
            result.condition_audio_rows + target_audio_rows)
        throw std::logic_error("MiniMax-H3 Ref2VA packed row accounting failed");
    return result;
}

Ref2vaRowTimesteps make_ref2va_row_timesteps(const Ref2vaPackedLayout& layout, float video_timestep,
                                             float audio_timestep) {
    const int32_t rows = layout.sequence_length();
    if (rows <= 0 || layout.position_ids.size() != static_cast<std::size_t>(rows) * 3U ||
        !std::isfinite(video_timestep) || !std::isfinite(audio_timestep))
        throw std::invalid_argument("MiniMax-H3 Ref2VA row-timestep input is invalid");
    std::vector<float> row_values(static_cast<std::size_t>(rows), video_timestep);
    if (layout.condition_video_rows < 0 || layout.condition_audio_rows < 0 ||
        layout.condition_video_rows > static_cast<int32_t>(layout.video_indices.size()) ||
        layout.condition_audio_rows > static_cast<int32_t>(layout.audio_indices.size()))
        throw std::invalid_argument("MiniMax-H3 Ref2VA condition row counts are invalid");
    for (int32_t index = 0; index < layout.condition_video_rows; ++index)
        row_values[static_cast<std::size_t>(layout.video_indices[index])] =
            std::max(video_timestep, kConditionVideoTimestep);
    for (std::size_t index = static_cast<std::size_t>(layout.condition_audio_rows);
         index < layout.audio_indices.size(); ++index)
        row_values[static_cast<std::size_t>(layout.audio_indices[index])] = audio_timestep;
    for (int32_t index = 0; index < layout.condition_audio_rows; ++index)
        row_values[static_cast<std::size_t>(layout.audio_indices[index])] = kConditionAudioTimestep;

    Ref2vaRowTimesteps result;
    result.unique_timesteps = row_values;
    std::sort(result.unique_timesteps.begin(), result.unique_timesteps.end());
    result.unique_timesteps.erase(
        std::unique(result.unique_timesteps.begin(), result.unique_timesteps.end()),
        result.unique_timesteps.end());
    if (result.unique_timesteps.empty() || result.unique_timesteps.size() > 4)
        throw std::invalid_argument("MiniMax-H3 Ref2VA needs 1..4 unique row timesteps");
    result.timestep_indices.resize(static_cast<std::size_t>(rows));
    result.adaln_indices.resize(static_cast<std::size_t>(rows));
    for (int32_t row = 0; row < rows; ++row) {
        const auto iterator =
            std::lower_bound(result.unique_timesteps.begin(), result.unique_timesteps.end(),
                             row_values[static_cast<std::size_t>(row)]);
        if (iterator == result.unique_timesteps.end() ||
            *iterator != row_values[static_cast<std::size_t>(row)])
            throw std::logic_error("MiniMax-H3 Ref2VA timestep inverse failed");
        const int32_t timestep =
            checked_i32(iterator - result.unique_timesteps.begin(), "timestep index");
        const int32_t tag = layout.token_tags[static_cast<std::size_t>(row)];
        if (tag < kVideoTag || tag > kAudioTag)
            throw std::invalid_argument("MiniMax-H3 Ref2VA row tag is invalid");
        result.timestep_indices[static_cast<std::size_t>(row)] = timestep;
        result.adaln_indices[static_cast<std::size_t>(row)] = timestep * 3 + tag;
    }
    return result;
}

Ref2vaTimestepTable pad_ref2va_timesteps(const std::vector<float>& timesteps) {
    if (timesteps.empty() || timesteps.size() > 4)
        throw std::invalid_argument("MiniMax-H3 Ref2VA timestep table needs 1..4 rows");
    for (std::size_t index = 0; index < timesteps.size(); ++index) {
        if (!std::isfinite(timesteps[index]) ||
            (index > 0 && timesteps[index] <= timesteps[index - 1]))
            throw std::invalid_argument(
                "MiniMax-H3 Ref2VA timesteps must be finite and strictly sorted");
    }
    Ref2vaTimestepTable result;
    result.values.fill(timesteps.back());
    std::copy(timesteps.begin(), timesteps.end(), result.values.begin());
    result.live_count = checked_i32(timesteps.size(), "live timesteps");
    return result;
}

void validate_ref2va_plan(ITrtModule& module, Ref2vaPlanKind kind) {
    switch (kind) {
    case Ref2vaPlanKind::kVisionEncoder:
        require_counts(module, 4, 4, "vision encoder");
        require_dynamic_input(module, "pixel_values", DType::kFloat32, {2040, 1536}, {4032, 1536},
                              {65536, 1536});
        require_dynamic_input(module, "interp_indices", DType::kInt32, {2040, 4}, {4032, 4},
                              {65536, 4});
        require_dynamic_input(module, "interp_weights", DType::kFloat32, {2040, 4}, {4032, 4},
                              {65536, 4});
        require_dynamic_input(module, "vision_position_ids", DType::kInt32, {2040, 2}, {4032, 2},
                              {65536, 2});
        for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
            require_output(module, name, DType::kFloat32, {16384, 5120});
        return;
    case Ref2vaPlanKind::kTextEncoder:
        require_counts(module, 9, 1, "text encoder");
        require_dynamic_input(module, "input_ids", DType::kInt32, {1}, {1144}, {262144});
        require_dynamic_input(module, "mrope_position_ids", DType::kInt32, {3, 1}, {3, 1144},
                              {3, 262144});
        require_dynamic_input(module, "vision_mask", DType::kFloat32, {1, 1}, {1144, 1},
                              {262144, 1});
        require_static_input(module, "vision_count", DType::kInt32, {1});
        require_dynamic_input(module, "vision_row_indices", DType::kInt32, {1}, {1008}, {262144});
        for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
            require_dynamic_input(module, name, DType::kFloat32, {1, 5120}, {1008, 5120},
                                  {262144, 5120});
        require_output(module, "encoder_hidden_states", DType::kFloat32, {262144, 5120});
        return;
    case Ref2vaPlanKind::kKeyframeVaeEncoder:
        require_counts(module, 1, 1, "keyframe VAE encoder");
        require_dynamic_input(module, "pixel_tiles", DType::kFloat32, {1, 3, 1, 256, 256},
                              {28, 3, 1, 256, 256}, {33, 3, 1, 256, 256});
        require_output(module, "posterior_parameter_tiles", DType::kFloat32, {33, 48, 1, 16, 16});
        return;
    case Ref2vaPlanKind::kVideoVaeEncoder:
        require_counts(module, 1, 1, "video VAE encoder");
        require_static_input(module, "pixel_tile_clip", DType::kFloat32, {1, 3, 17, 256, 256});
        require_output(module, "posterior_parameter_tile_clip", DType::kFloat32,
                       {1, 48, 5, 16, 16});
        return;
    case Ref2vaPlanKind::kAudioVaeEncoder:
        require_counts(module, 1, 1, "audio VAE encoder");
        require_dynamic_input(module, "audio_samples", DType::kFloat32, {2, 1, 64000},
                              {2, 1, 165600}, {2, 1, 480000});
        require_output(module, "posterior_mean", DType::kFloat32, {2, 32, 600});
        return;
    case Ref2vaPlanKind::kAdalnPrecompute:
        require_counts(module, 1, 51, "AdaLN precompute");
        require_static_input(module, "timestep_features", DType::kFloat32, {4, 256});
        for (int32_t layer = 0; layer < 50; ++layer)
            require_output(module, "block_modulation_" + std::to_string(layer), DType::kBFloat16,
                           {12, 6, 5376});
        require_output(module, "final_modulation", DType::kBFloat16, {4, 2, 5376});
        return;
    case Ref2vaPlanKind::kDenoiser:
        require_counts(module, 60, 2, "denoiser");
        require_dynamic_input(module, "video_hidden_states", DType::kFloat32, {kMinVideoRows, 96},
                              {kOptVideoRows, 96}, {kRef2vaMaxVideoRows, 96});
        require_dynamic_input(module, "audio_hidden_states", DType::kFloat32, {kMinAudioRows, 32},
                              {kOptAudioRows, 32}, {kRef2vaMaxAudioRows, 32});
        require_dynamic_input(module, "encoder_hidden_states", DType::kFloat32,
                              {kMinTextRows, 5120}, {kOptTextRows, 5120},
                              {kRef2vaMaxTextRows, 5120});
        require_dynamic_input(module, "position_ids", DType::kFloat32, {kMinPackedRows, 3},
                              {kOptPackedRows, 3}, {kRef2vaMaxPackedRows, 3});
        require_dynamic_input(module, "video_indices", DType::kInt32, {kMinVideoRows},
                              {kOptVideoRows}, {kRef2vaMaxVideoRows});
        require_dynamic_input(module, "audio_indices", DType::kInt32, {kMinAudioRows},
                              {kOptAudioRows}, {kRef2vaMaxAudioRows});
        require_dynamic_input(module, "text_indices", DType::kInt32, {kMinTextRows}, {kOptTextRows},
                              {kRef2vaMaxTextRows});
        for (const char* name : {"adaln_indices", "timestep_indices"})
            require_dynamic_input(module, name, DType::kInt32, {kMinPackedRows}, {kOptPackedRows},
                                  {kRef2vaMaxPackedRows});
        for (int32_t layer = 0; layer < 50; ++layer)
            require_static_input(module, "block_modulation_" + std::to_string(layer),
                                 DType::kBFloat16, {12, 6, 5376});
        require_static_input(module, "final_modulation", DType::kBFloat16, {4, 2, 5376});
        require_output(module, "video_velocity", DType::kFloat32, {kRef2vaMaxVideoRows, 96});
        require_output(module, "audio_velocity", DType::kFloat32, {kRef2vaMaxAudioRows, 32});
        return;
    }
    throw std::invalid_argument("MiniMax-H3 Ref2VA plan kind is invalid");
}

Ref2vaModulations run_ref2va_adaln_precompute(ITrtModule& module,
                                              const Ref2vaTimestepTable& timesteps) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kAdalnPrecompute);
    if (timesteps.live_count < 1 || timesteps.live_count > 4)
        throw std::invalid_argument("MiniMax-H3 Ref2VA AdaLN live timestep count is invalid");
    std::vector<float> features(4U * 256U);
    for (int32_t row = 0; row < 4; ++row) {
        for (int32_t index = 0; index < 128; ++index) {
            const double frequency = std::exp(-std::log(10000.0) * index / 128.0);
            const double phase = static_cast<double>(timesteps.values[row]) * frequency;
            features[static_cast<std::size_t>(row) * 256U + index] =
                static_cast<float>(std::cos(phase));
            features[static_cast<std::size_t>(row) * 256U + 128U + index] =
                static_cast<float>(std::sin(phase));
        }
    }
    TensorMap inputs;
    inputs.emplace("timestep_features", Tensor{features.data(), {4, 256}, DType::kFloat32});
    const TensorMap outputs = module.forward(inputs);
    Ref2vaModulations result;
    constexpr std::size_t block_values = 12U * 6U * 5376U;
    for (int32_t layer = 0; layer < 50; ++layer) {
        const std::string name = "block_modulation_" + std::to_string(layer);
        result.blocks[static_cast<std::size_t>(layer)] =
            copy_owned(find_output(outputs, name), DType::kBFloat16, block_values, name);
    }
    result.final = copy_owned(find_output(outputs, "final_modulation"), DType::kBFloat16,
                              4U * 2U * 5376U, "final_modulation");
    return result;
}

Ref2vaVelocities run_ref2va_denoiser(ITrtModule& module, Ref2vaDenoiserInputs& inputs,
                                     Ref2vaModulations& modulations) {
    validate_ref2va_plan(module, Ref2vaPlanKind::kDenoiser);
    const int32_t sequence_rows = inputs.layout.sequence_length();
    const int32_t video_rows = checked_i32(inputs.layout.video_indices.size(), "video rows");
    const int32_t audio_rows = checked_i32(inputs.layout.audio_indices.size(), "audio rows");
    const int32_t text_rows = checked_i32(inputs.layout.text_indices.size(), "text rows");
    if (video_rows < kMinVideoRows || video_rows > kRef2vaMaxVideoRows ||
        audio_rows < kMinAudioRows || audio_rows > kRef2vaMaxAudioRows ||
        text_rows < kMinTextRows || text_rows > kRef2vaMaxTextRows ||
        sequence_rows != video_rows + audio_rows + text_rows ||
        inputs.video_hidden_states.size() != static_cast<std::size_t>(video_rows) * 96U ||
        inputs.audio_hidden_states.size() != static_cast<std::size_t>(audio_rows) * 32U ||
        inputs.encoder_hidden_states.size() != static_cast<std::size_t>(text_rows) * 5120U ||
        inputs.layout.position_ids.size() != static_cast<std::size_t>(sequence_rows) * 3U ||
        inputs.timestep_indices.size() != static_cast<std::size_t>(sequence_rows) ||
        inputs.adaln_indices.size() != static_cast<std::size_t>(sequence_rows))
        throw std::invalid_argument("MiniMax-H3 Ref2VA denoiser input shape is invalid");

    std::vector<uint8_t> coverage(static_cast<std::size_t>(sequence_rows), 0);
    for (const auto* indices : {&inputs.layout.text_indices, &inputs.layout.video_indices,
                                &inputs.layout.audio_indices}) {
        for (int32_t index : *indices) {
            if (index < 0 || index >= sequence_rows || coverage[static_cast<std::size_t>(index)]++)
                throw std::invalid_argument(
                    "MiniMax-H3 Ref2VA scatter indices are not disjoint/in-range");
        }
    }
    if (std::find(coverage.begin(), coverage.end(), 0) != coverage.end())
        throw std::invalid_argument("MiniMax-H3 Ref2VA scatter indices are not exhaustive");

    TensorMap tensor_inputs;
    tensor_inputs.emplace(
        "video_hidden_states",
        Tensor{inputs.video_hidden_states.data(), {video_rows, 96}, DType::kFloat32});
    tensor_inputs.emplace(
        "audio_hidden_states",
        Tensor{inputs.audio_hidden_states.data(), {audio_rows, 32}, DType::kFloat32});
    tensor_inputs.emplace(
        "encoder_hidden_states",
        Tensor{inputs.encoder_hidden_states.data(), {text_rows, 5120}, DType::kFloat32});
    tensor_inputs.emplace(
        "position_ids",
        Tensor{inputs.layout.position_ids.data(), {sequence_rows, 3}, DType::kFloat32});
    tensor_inputs.emplace("video_indices",
                          Tensor{inputs.layout.video_indices.data(), {video_rows}, DType::kInt32});
    tensor_inputs.emplace("audio_indices",
                          Tensor{inputs.layout.audio_indices.data(), {audio_rows}, DType::kInt32});
    tensor_inputs.emplace("text_indices",
                          Tensor{inputs.layout.text_indices.data(), {text_rows}, DType::kInt32});
    tensor_inputs.emplace("adaln_indices",
                          Tensor{inputs.adaln_indices.data(), {sequence_rows}, DType::kInt32});
    tensor_inputs.emplace("timestep_indices",
                          Tensor{inputs.timestep_indices.data(), {sequence_rows}, DType::kInt32});
    for (int32_t layer = 0; layer < 50; ++layer) {
        auto& value = modulations.blocks[static_cast<std::size_t>(layer)];
        if (value.dtype != DType::kBFloat16 || value.shape != std::vector<int64_t>({12, 6, 5376}) ||
            value.bytes.empty())
            throw std::invalid_argument("MiniMax-H3 Ref2VA block modulation is invalid");
        tensor_inputs.emplace("block_modulation_" + std::to_string(layer),
                              Tensor{value.bytes.data(), value.shape, value.dtype});
    }
    if (modulations.final.dtype != DType::kBFloat16 ||
        modulations.final.shape != std::vector<int64_t>({4, 2, 5376}) ||
        modulations.final.bytes.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA final modulation is invalid");
    tensor_inputs.emplace(
        "final_modulation",
        Tensor{modulations.final.bytes.data(), modulations.final.shape, modulations.final.dtype});
    const TensorMap outputs = module.forward(tensor_inputs);
    Ref2vaVelocities result;
    result.video = copy_float_output(find_output(outputs, "video_velocity"),
                                     static_cast<std::size_t>(video_rows) * 96U, "video_velocity");
    result.audio = copy_float_output(find_output(outputs, "audio_velocity"),
                                     static_cast<std::size_t>(audio_rows) * 32U, "audio_velocity");
    return result;
}

} // namespace trtmc::minimax_h3
