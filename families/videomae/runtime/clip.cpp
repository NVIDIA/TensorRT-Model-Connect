/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/videomae/runtime/clip.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::videomae {
namespace {

// The checkpoint's pixel targets are (value - mean) / (std + 1e-6) per tube
// and channel (norm_pix_loss); reconstruction applies the inverse.
constexpr double kStdEpsilon = 1e-6;

std::size_t index(std::int64_t value) {
    return static_cast<std::size_t>(value);
}

// Offset of pixel (frame, row, column) in an FHW plane stack.
std::int64_t plane_offset(const ClipGeometry& geometry, std::int64_t frame, std::int64_t row,
                          std::int64_t column) {
    return (frame * geometry.size + row) * geometry.size + column;
}

struct TubeOrigin {
    std::int64_t frame, row, column;
};

TubeOrigin origin(const ClipGeometry& geometry, std::int64_t tube) {
    const auto grid = geometry.grid();
    return {tube / (grid * grid) * geometry.tubelet, tube / grid % grid * geometry.patch,
            tube % grid * geometry.patch};
}

std::string describe(const TubeOrigin& tube) {
    return "frame " + std::to_string(tube.frame) + ", row " + std::to_string(tube.row) +
           ", column " + std::to_string(tube.column);
}

// Calls `visit(pixel, offset)` for the tube's pixels in (frame, row, column)
// order; `offset` is the pixel's index in an FHW plane stack.
template <class Visit>
void for_each_pixel(const ClipGeometry& geometry, const TubeOrigin& tube, Visit&& visit) {
    std::int64_t pixel = 0;
    for (std::int64_t frame = 0; frame < geometry.tubelet; ++frame)
        for (std::int64_t row = 0; row < geometry.patch; ++row)
            for (std::int64_t column = 0; column < geometry.patch; ++column)
                visit(pixel++, plane_offset(geometry, tube.frame + frame, tube.row + row,
                                            tube.column + column));
}

void copy_frame(const internal::ImageView& frame, float* out, std::size_t count) {
    if (frame.format == internal::ImageFormat::UInt8) {
        const auto* values = static_cast<const std::uint8_t*>(frame.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = static_cast<float>(static_cast<double>(values[i]) * (1.0 / 255.0));
        return;
    }
    const auto* values = static_cast<const float*>(frame.data);
    for (std::size_t i = 0; i < count; ++i) {
        if (!(values[i] >= 0.0F && values[i] <= 1.0F))
            throw std::invalid_argument(
                "videomae float32 frames must hold finite values in [0, 1]");
        out[i] = values[i];
    }
}

} // namespace

void validate_geometry(const ClipGeometry& geometry) {
    const auto limit = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    if (geometry.frames <= 0 || geometry.size <= 0 || geometry.patch <= 0 ||
        geometry.tubelet <= 0 || geometry.frames > limit || geometry.size > limit ||
        geometry.size % geometry.patch != 0 || geometry.frames % geometry.tubelet != 0)
        throw std::runtime_error("videomae runtime.json does not describe whole tubes");
    // Bounds by division first, so that no product below can overflow.
    if (geometry.size > limit / ClipGeometry::kChannels / geometry.size ||
        geometry.frame_values() > limit / geometry.frames)
        throw std::runtime_error("videomae runtime.json describes an oversized clip");
    if (geometry.tubes() < 2 || geometry.tube_pixels() < 2)
        throw std::runtime_error("videomae needs at least two tubes of at least two pixels");
}

std::vector<float> source_clip(Span<const internal::ImageView> frames,
                               const ClipGeometry& geometry) {
    if (frames.size() != index(geometry.frames))
        throw std::invalid_argument("videomae reconstructs clips of exactly " +
                                    std::to_string(geometry.frames) + " frames; got " +
                                    std::to_string(frames.size()));
    const auto count = index(geometry.frame_values());
    std::vector<float> clip(index(geometry.frames) * count);
    for (std::size_t f = 0; f < frames.size(); ++f) {
        const auto& frame = frames[f];
        if (frame.height != geometry.size || frame.width != geometry.size ||
            frame.channels != ClipGeometry::kChannels)
            throw std::invalid_argument(
                "videomae frames must be " + std::to_string(geometry.size) + "x" +
                std::to_string(geometry.size) +
                " RGB (resize and center-crop them as the checkpoint's processor does); frame " +
                std::to_string(f) + " is " + std::to_string(frame.height) + "x" +
                std::to_string(frame.width) + " with " + std::to_string(frame.channels) +
                " channels");
        copy_frame(frame, clip.data() + f * count, count);
    }
    return clip;
}

std::vector<float> pixel_values(const std::vector<float>& clip, const ClipGeometry& geometry,
                                const PixelNormalization& normalization) {
    const auto plane = index(geometry.size * geometry.size);
    if (clip.size() != index(geometry.frames) * plane * ClipGeometry::kChannels)
        throw std::logic_error("videomae clip does not match its geometry");
    std::vector<float> values(clip.size());
    for (std::size_t frame = 0; frame < index(geometry.frames); ++frame)
        for (std::size_t pixel = 0; pixel < plane; ++pixel)
            for (std::size_t channel = 0; channel < ClipGeometry::kChannels; ++channel)
                values[(frame * ClipGeometry::kChannels + channel) * plane + pixel] =
                    (clip[(frame * plane + pixel) * ClipGeometry::kChannels + channel] -
                     normalization.mean[channel]) /
                    normalization.std[channel];
    return values;
}

TubeSelection select_tubes(const internal::VideoMaskView& mask, const ClipGeometry& geometry) {
    if (mask.frames != index(geometry.frames) || mask.height != index(geometry.size) ||
        mask.width != index(geometry.size) ||
        mask.values.size() != index(geometry.frames * geometry.size * geometry.size))
        throw std::invalid_argument("videomae mask must cover every pixel of the clip");
    TubeSelection selection;
    for (std::int64_t tube = 0; tube < geometry.tubes(); ++tube) {
        const auto start = origin(geometry, tube);
        const float kind =
            mask.values[index(plane_offset(geometry, start.frame, start.row, start.column))];
        if (kind != 0.0F && kind != 1.0F)
            throw std::invalid_argument("videomae mask values must be 0 (condition) or 1 "
                                        "(reconstruct); the tube at " +
                                        describe(start) + " starts with another value");
        for_each_pixel(geometry, start, [&](std::int64_t, std::int64_t offset) {
            if (mask.values[index(offset)] != kind)
                throw std::invalid_argument(
                    "videomae masks select whole tubes of " + std::to_string(geometry.tubelet) +
                    " frames x " + std::to_string(geometry.patch) + "x" +
                    std::to_string(geometry.patch) + " pixels, each all 0 or all 1; the tube at " +
                    describe(start) + " is mixed");
        });
        (kind == 1.0F ? selection.masked : selection.visible)
            .push_back(static_cast<std::int32_t>(tube));
    }
    if (selection.masked.empty())
        throw std::invalid_argument("the videomae mask selects no tube to reconstruct");
    if (selection.visible.empty())
        throw std::invalid_argument(
            "the videomae mask selects every tube; at least one tube must condition the "
            "reconstruction");
    return selection;
}

void reconstruct(std::vector<float>& clip, const float* predictions, const TubeSelection& selection,
                 const ClipGeometry& geometry) {
    constexpr auto channels = static_cast<std::size_t>(ClipGeometry::kChannels);
    if (clip.size() != index(geometry.frames * geometry.frame_values()))
        throw std::logic_error("videomae clip does not match its geometry");
    const double samples = static_cast<double>(geometry.tube_pixels());
    for (std::size_t row = 0; row < selection.masked.size(); ++row) {
        const auto start = origin(geometry, selection.masked[row]);
        std::array<double, channels> mean{}, spread{};
        for_each_pixel(geometry, start, [&](std::int64_t, std::int64_t offset) {
            for (std::size_t c = 0; c < channels; ++c)
                mean[c] += clip[index(offset) * channels + c];
        });
        for (auto& value : mean)
            value /= samples;
        for_each_pixel(geometry, start, [&](std::int64_t, std::int64_t offset) {
            for (std::size_t c = 0; c < channels; ++c) {
                const double deviation = clip[index(offset) * channels + c] - mean[c];
                spread[c] += deviation * deviation;
            }
        });
        for (auto& value : spread)
            value = std::sqrt(value / (samples - 1.0)) + kStdEpsilon;
        const float* prediction = predictions + row * index(geometry.tube_values());
        for_each_pixel(geometry, start, [&](std::int64_t pixel, std::int64_t offset) {
            for (std::size_t c = 0; c < channels; ++c) {
                const double value =
                    static_cast<double>(prediction[index(pixel) * channels + c]) * spread[c] +
                    mean[c];
                if (!std::isfinite(value))
                    throw std::runtime_error("videomae engine returned a non-finite prediction");
                clip[index(offset) * channels + c] =
                    static_cast<float>(std::clamp(value, 0.0, 1.0));
            }
        });
    }
}

} // namespace trtmc::videomae
