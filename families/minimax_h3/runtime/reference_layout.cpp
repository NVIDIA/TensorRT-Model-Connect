/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Copyright 2026 The MiniMax and HuggingFace Teams. */

#include "families/minimax_h3/runtime/reference_layout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {

int32_t video_rows(int32_t frames, int32_t height, int32_t width) {
    if (frames <= 0 || height <= 0 || width <= 0 || height % 2 || width % 2)
        throw std::invalid_argument("MiniMax-H3 latent geometry requires positive T and even H/W");
    int64_t rows = frames;
    for (const auto factor : {height / 2, width / 2}) {
        if (rows > std::numeric_limits<int32_t>::max() / factor)
            throw std::length_error("MiniMax-H3 video rows exceed the int32 engine index range");
        rows *= factor;
    }
    return static_cast<int32_t>(rows);
}

std::vector<double> spatial_axis(int32_t dimension, double sqrt_area) {
    const double ratio = dimension / sqrt_area;
    const double left = (1.0 - ratio) / 2.0;
    // Reproduce np.linspace(left,left+ratio,n,endpoint=False), including the
    // endpoint subtraction. Force the separate product/add rounding even on
    // hosts whose compiler otherwise contracts them into an FMA.
    const double right = left + ratio;
    const double step = (right - left) / (dimension / 2);
    std::vector<double> axis(static_cast<std::size_t>(dimension / 2));
    for (std::size_t index = 0; index < axis.size(); ++index) {
        volatile double product = static_cast<double>(index) * step;
        axis[index] = (left + product) * 32.0;
    }
    return axis;
}

void append_indices(std::vector<int32_t>& values, int32_t begin, int32_t count) {
    const auto previous = values.size();
    values.resize(previous + static_cast<std::size_t>(count));
    std::iota(values.begin() + static_cast<std::ptrdiff_t>(previous), values.end(), begin);
}

void fill_video_positions(std::vector<double>& positions, int32_t begin, int32_t frames,
                          int32_t height, int32_t width, double origin) {
    const double area = static_cast<double>(height) * width;
    const auto vertical = spatial_axis(height, std::sqrt(area));
    const auto horizontal = spatial_axis(width, std::sqrt(area));
    std::size_t row = static_cast<std::size_t>(begin);
    double offset = 0.0;
    for (int32_t frame = 0; frame < frames; ++frame) {
        // torch.cumsum(spans) is computed before adding the rotary origin.
        const double time = origin + offset;
        for (const auto y : vertical)
            for (const auto x : horizontal) {
                positions[row * 3] = time;
                positions[row * 3 + 1] = y;
                positions[row * 3 + 2] = x;
                ++row;
            }
        offset += (5.0 / 3.0) * (frame % 5 == 0 ? 1.0 : 4.0);
    }
}

} // namespace

ImageReferenceLayout
make_image_reference_layout(const std::vector<int32_t>& text_tags,
                            const std::vector<ImageReferenceLatentShape>& images,
                            ReferenceTargetLatentShape target, int32_t max_rows) {
    if (max_rows <= 0 || text_tags.empty() || images.empty() || images.size() > 9 ||
        target.audio_latents <= 0 ||
        !std::all_of(text_tags.begin(), text_tags.end(),
                     [](int32_t tag) { return tag == 0 || tag == 1; }))
        throw std::invalid_argument(
            "MiniMax-H3 image Ref2VA requires text, 1..9 images and stereo audio");
    if (text_tags.size() > static_cast<std::size_t>(max_rows))
        throw std::length_error("MiniMax-H3 text exceeds the loaded reference DiT profile");
    const int32_t target_video_rows = video_rows(target.frames, target.height, target.width);
    const int64_t target_audio_rows = static_cast<int64_t>(target.audio_latents) * 2;
    int64_t total = static_cast<int64_t>(text_tags.size()) + target_video_rows + target_audio_rows;
    int64_t conditions = 0;
    for (const auto image : images) {
        conditions += video_rows(1, image.height, image.width);
        if (total + conditions > max_rows)
            throw std::length_error("MiniMax-H3 images exceed the loaded reference DiT profile");
    }
    total += conditions;
    if (total > max_rows)
        throw std::length_error("MiniMax-H3 sequence exceeds the loaded reference DiT profile");

    ImageReferenceLayout result;
    result.condition_video_rows = static_cast<int32_t>(conditions);
    result.positions.resize(static_cast<std::size_t>(total) * 3, 0.0);
    result.token_tags.resize(static_cast<std::size_t>(total), 0);
    std::copy(text_tags.begin(), text_tags.end(), result.token_tags.begin());
    append_indices(result.text_indices, 0, static_cast<int32_t>(text_tags.size()));
    for (std::size_t index = 0; index < text_tags.size(); ++index)
        result.positions[index * 3] = static_cast<double>(index);

    int32_t cursor = static_cast<int32_t>(text_tags.size());
    double rotary_time = cursor;
    for (const auto image : images) {
        const int32_t rows = video_rows(1, image.height, image.width);
        append_indices(result.video_indices, cursor, rows);
        fill_video_positions(result.positions, cursor, 1, image.height, image.width, rotary_time);
        cursor += rows;
        rotary_time += 1.0; // An image occupies one integer slot, not 5/3.
    }
    const auto width_axis =
        spatial_axis(target.width, std::sqrt(static_cast<double>(target.height) * target.width));
    for (int32_t channel = 0; channel < 2; ++channel)
        for (int32_t index = 0; index < target.audio_latents; ++index) {
            const auto row =
                static_cast<std::size_t>(cursor + channel * target.audio_latents + index);
            result.positions[row * 3] = rotary_time + index;
            result.positions[row * 3 + 2] = channel == 0 ? width_axis.front() : width_axis.back();
            result.token_tags[row] = 2;
        }
    append_indices(result.audio_indices, cursor, static_cast<int32_t>(target_audio_rows));
    result.target_audio_indices = result.audio_indices;
    cursor += static_cast<int32_t>(target_audio_rows);
    append_indices(result.target_video_indices, cursor, target_video_rows);
    append_indices(result.video_indices, cursor, target_video_rows);
    fill_video_positions(result.positions, cursor, target.frames, target.height, target.width,
                         rotary_time);
    result.packed_indices.resize(static_cast<std::size_t>(total));
    int32_t concatenated = 0;
    for (const auto* indices : {&result.text_indices, &result.audio_indices, &result.video_indices})
        for (const auto packed : *indices)
            result.packed_indices[static_cast<std::size_t>(packed)] = concatenated++;
    return result;
}

ImageReferenceLayout make_continuation_reference_layout(
    const std::vector<int32_t>& text_tags,
    const std::vector<ImageReferenceLatentShape>& images, ReferenceTargetLatentShape target,
    int32_t max_rows, ReferenceContinuationGuideShape guide, double temporal_offset,
    std::optional<int32_t> media_time_origin) {
    if (guide.video_frames <= 0 || guide.video_frames > target.frames ||
        guide.audio_latents <= 0 || guide.audio_latents > target.audio_latents ||
        !std::isfinite(temporal_offset) || temporal_offset < 0 ||
        (media_time_origin && (*media_time_origin < 0 ||
                              static_cast<std::size_t>(*media_time_origin) < text_tags.size())))
        throw std::invalid_argument("MiniMax-H3 continuation guide has invalid geometry or clock");
    const int32_t video_conditions = video_rows(guide.video_frames, target.height, target.width);
    const int64_t audio_conditions = int64_t{2} * guide.audio_latents;
    const int64_t extra = audio_conditions + video_conditions;
    if (max_rows <= 0 || extra >= max_rows)
        throw std::length_error("MiniMax-H3 continuation guide exceeds the loaded profile");
    // Keep legacy validation and its exact image-only layout unchanged.
    const auto base = make_image_reference_layout(text_tags, images, target,
                                                 max_rows - static_cast<int32_t>(extra));
    const auto total = base.token_tags.size() + static_cast<std::size_t>(extra);
    ImageReferenceLayout result;
    result.condition_video_rows = base.condition_video_rows + video_conditions;
    result.condition_audio_rows = static_cast<int32_t>(audio_conditions);
    result.continuation_video_rows = video_conditions;
    result.positions.resize(total * 3, 0.0);
    result.token_tags.resize(total, 0);
    result.text_indices = base.text_indices;
    std::copy(text_tags.begin(), text_tags.end(), result.token_tags.begin());
    for (std::size_t i = 0; i < text_tags.size(); ++i)
        result.positions[i * 3] = static_cast<double>(i);
    int32_t cursor = static_cast<int32_t>(text_tags.size());
    double origin = media_time_origin ? static_cast<double>(*media_time_origin)
                                      : static_cast<double>(text_tags.size());
    for (const auto image : images) {
        const auto count = video_rows(1, image.height, image.width);
        append_indices(result.video_indices, cursor, count);
        fill_video_positions(result.positions, cursor, 1, image.height, image.width, origin);
        cursor += count;
        origin += 1.0;
    }
    const auto width_axis =
        spatial_axis(target.width, std::sqrt(static_cast<double>(target.height) * target.width));
    const auto append_audio = [&](int32_t steps, bool generated) {
        const int32_t count = steps * 2;
        append_indices(result.audio_indices, cursor, count);
        if (generated)
            append_indices(result.target_audio_indices, cursor, count);
        for (int32_t channel = 0; channel < 2; ++channel)
            for (int32_t t = 0; t < steps; ++t) {
                const auto row = static_cast<std::size_t>(cursor + channel * steps + t);
                // Apply the global shift after each local FP64 coordinate,
                // matching the reference tensor operation's rounding order.
                result.positions[row * 3] = (origin + t) + temporal_offset;
                result.positions[row * 3 + 2] = channel == 0 ? width_axis.front() : width_axis.back();
                result.token_tags[row] = 2;
            }
        cursor += count;
    };
    const auto append_video = [&](int32_t frames, bool generated) {
        const auto count = video_rows(frames, target.height, target.width);
        append_indices(result.video_indices, cursor, count);
        if (generated)
            append_indices(result.target_video_indices, cursor, count);
        fill_video_positions(result.positions, cursor, frames, target.height, target.width, origin);
        for (int32_t i = 0; i < count; ++i)
            result.positions[static_cast<std::size_t>(cursor + i) * 3] += temporal_offset;
        cursor += count;
    };
    append_audio(guide.audio_latents, false);
    append_video(guide.video_frames, false);
    append_audio(target.audio_latents, true);
    append_video(target.frames, true);
    result.packed_indices.resize(total);
    int32_t concatenated = 0;
    for (const auto* indices : {&result.text_indices, &result.audio_indices, &result.video_indices})
        for (const auto packed : *indices)
            result.packed_indices[static_cast<std::size_t>(packed)] = concatenated++;
    return result;
}

ReferenceRowTimesteps make_image_reference_timesteps(const ImageReferenceLayout& layout,
                                                     float video_timestep, float audio_timestep) {
    const auto valid_time = [](float value) {
        return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    };
    if (!valid_time(video_timestep) || !valid_time(audio_timestep) || layout.token_tags.empty() ||
        layout.condition_video_rows <= 0 ||
        static_cast<std::size_t>(layout.condition_video_rows) >= layout.video_indices.size() ||
        layout.condition_audio_rows < 0 ||
        static_cast<std::size_t>(layout.condition_audio_rows) >= layout.audio_indices.size())
        throw std::invalid_argument(
            "MiniMax-H3 reference timestep plan has invalid times or layout");
    std::vector<float> rows(layout.token_tags.size(), video_timestep);
    for (int32_t index = 0; index < layout.condition_video_rows; ++index)
        rows.at(static_cast<std::size_t>(layout.video_indices[static_cast<std::size_t>(index)])) =
            std::max(video_timestep, 0.999F);
    for (const auto index : layout.audio_indices)
        rows.at(static_cast<std::size_t>(index)) = audio_timestep;
    for (int32_t index = 0; index < layout.condition_audio_rows; ++index)
        rows.at(static_cast<std::size_t>(layout.audio_indices[static_cast<std::size_t>(index)])) = 1.0F;
    ReferenceRowTimesteps result;
    result.timesteps = rows;
    std::sort(result.timesteps.begin(), result.timesteps.end());
    result.timesteps.erase(std::unique(result.timesteps.begin(), result.timesteps.end()),
                           result.timesteps.end());
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (layout.token_tags[index] < 0 || layout.token_tags[index] > 2)
            throw std::invalid_argument(
                "MiniMax-H3 reference layout contains an invalid modality tag");
        const auto slot = static_cast<int32_t>(
            std::lower_bound(result.timesteps.begin(), result.timesteps.end(), rows[index]) -
            result.timesteps.begin());
        result.timestep_indices.push_back(slot);
        result.adaln_indices.push_back(slot * 3 + layout.token_tags[index]);
    }
    return result;
}

std::vector<float> pack_reference_video_rows(const std::vector<float>& latents, int32_t frames,
                                             int32_t height, int32_t width) {
    const auto rows = video_rows(frames, height, width);
    if (latents.size() != static_cast<std::size_t>(rows) * 96)
        throw std::invalid_argument(
            "MiniMax-H3 reference latent payload does not match [1,24,T,H,W]");
    std::vector<float> packed(latents.size());
    std::size_t output = 0;
    for (int32_t frame = 0; frame < frames; ++frame)
        for (int32_t y = 0; y < height; y += 2)
            for (int32_t x = 0; x < width; x += 2)
                for (int32_t channel = 0; channel < 24; ++channel)
                    for (int32_t dy = 0; dy < 2; ++dy)
                        for (int32_t dx = 0; dx < 2; ++dx) {
                            const auto input =
                                ((static_cast<std::size_t>(channel) * frames + frame) * height + y +
                                 dy) *
                                    width +
                                x + dx;
                            packed[output++] = latents[input];
                        }
    return packed;
}

} // namespace trtmc::minimax_h3
