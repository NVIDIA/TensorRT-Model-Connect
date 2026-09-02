/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/conditioning.h"

#include "runtime/models/minimax_h3/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

constexpr int32_t kModelFps = 24;
constexpr int32_t kModelAudioRate = 32000;
constexpr int32_t kCanvasMultiple = 32;
constexpr int32_t kReferenceImageShortEdge = 2048;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::size_t checked_product(std::initializer_list<std::size_t> factors, const char* label) {
    std::size_t result = 1;
    for (const std::size_t factor : factors) {
        if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor)
            throw std::invalid_argument(std::string(label) + " size overflows address space");
        result *= factor;
    }
    return result;
}

void validate_image(const VideoImageInput& image, const char* label) {
    if (image.height <= 0 || image.width <= 0)
        throw std::invalid_argument(std::string(label) + " must have positive dimensions");
    if (image.channels != 3 && image.channels != 4)
        throw std::invalid_argument(std::string(label) + " must be RGB or RGBA");
    const std::size_t expected = checked_product({static_cast<std::size_t>(image.height),
                                                  static_cast<std::size_t>(image.width),
                                                  static_cast<std::size_t>(image.channels)},
                                                 label);
    if (image.pixels.size() != expected)
        throw std::invalid_argument(std::string(label) + " pixel buffer size is invalid");
}

void validate_clip(const VideoClipInput& clip) {
    if (clip.num_frames <= 0 || clip.height <= 0 || clip.width <= 0)
        throw std::invalid_argument("MiniMax-H3 reference video must have positive dimensions");
    if (clip.channels != 3 && clip.channels != 4)
        throw std::invalid_argument("MiniMax-H3 reference video must be RGB or RGBA");
    if (clip.fps_numerator <= 0 || clip.fps_denominator <= 0)
        throw std::invalid_argument("MiniMax-H3 reference video must have a positive frame rate");
    const std::size_t expected = checked_product(
        {static_cast<std::size_t>(clip.num_frames), static_cast<std::size_t>(clip.height),
         static_cast<std::size_t>(clip.width), static_cast<std::size_t>(clip.channels)},
        "MiniMax-H3 reference video");
    if (clip.pixels.size() != expected)
        throw std::invalid_argument("MiniMax-H3 reference video pixel buffer size is invalid");
}

double sinc(double value) {
    if (std::abs(value) < 1.0e-12)
        return 1.0;
    const double angle = kPi * value;
    return std::sin(angle) / angle;
}

double lanczos(double value) {
    value = std::abs(value);
    if (value >= 3.0)
        return 0.0;
    return sinc(value) * sinc(value / 3.0);
}

struct ResampleContribution {
    int32_t first{0};
    std::vector<float> weights;
};

std::vector<ResampleContribution> make_lanczos_contributions(int32_t source_size,
                                                             int32_t target_size) {
    if (source_size <= 0 || target_size <= 0)
        throw std::invalid_argument("MiniMax-H3 resize dimensions must be positive");
    const double scale = static_cast<double>(source_size) / target_size;
    const double filter_scale = std::max(1.0, scale);
    const double support = 3.0 * filter_scale;
    std::vector<ResampleContribution> result(static_cast<std::size_t>(target_size));
    for (int32_t dst = 0; dst < target_size; ++dst) {
        const double center = (static_cast<double>(dst) + 0.5) * scale - 0.5;
        const int32_t first =
            std::max<int32_t>(0, static_cast<int32_t>(std::ceil(center - support)));
        const int32_t last =
            std::min<int32_t>(source_size - 1, static_cast<int32_t>(std::floor(center + support)));
        auto& contribution = result[static_cast<std::size_t>(dst)];
        contribution.first = first;
        contribution.weights.reserve(static_cast<std::size_t>(last - first + 1));
        double total = 0.0;
        for (int32_t src = first; src <= last; ++src) {
            const double weight = lanczos((static_cast<double>(src) - center) / filter_scale);
            contribution.weights.push_back(static_cast<float>(weight));
            total += weight;
        }
        if (std::abs(total) < 1.0e-15)
            throw std::runtime_error("MiniMax-H3 Lanczos kernel has zero weight");
        for (float& weight : contribution.weights)
            weight = static_cast<float>(static_cast<double>(weight) / total);
    }
    return result;
}

void resize_rgb_lanczos(const float* source, int32_t source_height, int32_t source_width,
                        int32_t source_channels, float* target, int32_t target_height,
                        int32_t target_width) {
    const auto horizontal = make_lanczos_contributions(source_width, target_width);
    const auto vertical = make_lanczos_contributions(source_height, target_height);
    const std::size_t intermediate_size = checked_product(
        {static_cast<std::size_t>(source_height), static_cast<std::size_t>(target_width), 3U},
        "MiniMax-H3 resize intermediate");
    std::vector<float> intermediate(intermediate_size);

    for (int32_t y = 0; y < source_height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            const auto& contribution = horizontal[static_cast<std::size_t>(x)];
            for (int32_t channel = 0; channel < 3; ++channel) {
                double sum = 0.0;
                for (std::size_t tap = 0; tap < contribution.weights.size(); ++tap) {
                    const int32_t source_x = contribution.first + static_cast<int32_t>(tap);
                    const std::size_t index =
                        (static_cast<std::size_t>(y) * source_width + source_x) * source_channels +
                        channel;
                    sum += static_cast<double>(source[index]) * contribution.weights[tap];
                }
                const std::size_t output_index =
                    (static_cast<std::size_t>(y) * target_width + x) * 3 + channel;
                intermediate[output_index] = static_cast<float>(sum);
            }
        }
    }

    for (int32_t y = 0; y < target_height; ++y) {
        const auto& contribution = vertical[static_cast<std::size_t>(y)];
        for (int32_t x = 0; x < target_width; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                double sum = 0.0;
                for (std::size_t tap = 0; tap < contribution.weights.size(); ++tap) {
                    const int32_t source_y = contribution.first + static_cast<int32_t>(tap);
                    const std::size_t index =
                        (static_cast<std::size_t>(source_y) * target_width + x) * 3 + channel;
                    sum += static_cast<double>(intermediate[index]) * contribution.weights[tap];
                }
                const std::size_t output_index =
                    (static_cast<std::size_t>(y) * target_width + x) * 3 + channel;
                target[output_index] = std::max(0.0F, std::min(1.0F, static_cast<float>(sum)));
            }
        }
    }
}

int64_t round_ties_to_even(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("MiniMax-H3 geometry must be finite");
    const double floor_value = std::floor(value);
    const double fraction = value - floor_value;
    if (fraction < 0.5)
        return static_cast<int64_t>(floor_value);
    if (fraction > 0.5)
        return static_cast<int64_t>(floor_value + 1.0);
    const int64_t integer = static_cast<int64_t>(floor_value);
    return (integer & 1LL) == 0 ? integer : integer + 1;
}

VideoImageInput crop_rgb(const VideoImageInput& source, int32_t top, int32_t left, int32_t height,
                         int32_t width) {
    validate_image(source, "MiniMax-H3 crop source");
    if (top < 0 || left < 0 || height <= 0 || width <= 0 || top + height > source.height ||
        left + width > source.width)
        throw std::invalid_argument("MiniMax-H3 crop is outside its source image");
    VideoImageInput result;
    result.height = height;
    result.width = width;
    result.channels = 3;
    result.pixels.resize(
        checked_product({static_cast<std::size_t>(height), static_cast<std::size_t>(width), 3U},
                        "MiniMax-H3 crop"));
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                const std::size_t source_index =
                    (static_cast<std::size_t>(top + y) * source.width + left + x) *
                        source.channels +
                    channel;
                const std::size_t target_index =
                    (static_cast<std::size_t>(y) * width + x) * 3 + channel;
                result.pixels[target_index] = source.pixels[source_index];
            }
        }
    }
    return result;
}

int64_t rounded_frame_slot(int64_t frame, int32_t fps_numerator, int32_t fps_denominator) {
    // floor(frame * 24 / fps + 0.5), evaluated as integer arithmetic.
    const int64_t numerator = frame * static_cast<int64_t>(kModelFps) * fps_denominator;
    const int64_t denominator = fps_numerator;
    return (2 * numerator + denominator) / (2 * denominator);
}

AudioResult resample_audio_sinc(const std::vector<float>& stereo, int32_t source_rate,
                                int32_t target_rate) {
    if (stereo.size() % 2 != 0)
        throw std::invalid_argument("MiniMax-H3 stereo audio has an odd sample count");
    const std::size_t source_frames = stereo.size() / 2;
    const std::size_t scaled_frames = checked_product(
        {source_frames, static_cast<std::size_t>(target_rate)}, "MiniMax-H3 audio resample ratio");
    const std::size_t target_frames =
        source_frames == 0 ? 0
                           : (scaled_frames + static_cast<std::size_t>(source_rate) - 1) /
                                 static_cast<std::size_t>(source_rate);
    AudioResult result;
    result.sample_rate = target_rate;
    result.channels = 2;
    result.samples.resize(checked_product({target_frames, 2U}, "MiniMax-H3 resampled audio"));
    result.num_samples = static_cast<int32_t>(result.samples.size());
    if (source_frames == 0)
        return result;

    constexpr int32_t half_width = 12;
    const double cutoff = std::min(1.0, static_cast<double>(target_rate) / source_rate) * 0.99;
    for (std::size_t dst = 0; dst < target_frames; ++dst) {
        const double source_position =
            static_cast<double>(dst) * static_cast<double>(source_rate) / target_rate;
        const int64_t center = static_cast<int64_t>(std::floor(source_position));
        for (int32_t channel = 0; channel < 2; ++channel) {
            double sum = 0.0;
            double total = 0.0;
            for (int32_t tap = -half_width + 1; tap <= half_width; ++tap) {
                const int64_t source_index = center + tap;
                if (source_index < 0 || source_index >= static_cast<int64_t>(source_frames))
                    continue;
                const double distance = source_position - static_cast<double>(source_index);
                const double window_position = distance / half_width;
                if (std::abs(window_position) >= 1.0)
                    continue;
                const double window = 0.5 * (1.0 + std::cos(kPi * window_position));
                const double weight = cutoff * sinc(cutoff * distance) * window;
                sum += static_cast<double>(stereo[static_cast<std::size_t>(source_index) * 2 +
                                                  static_cast<std::size_t>(channel)]) *
                       weight;
                total += weight;
            }
            const float value = total == 0.0 ? 0.0F : static_cast<float>(sum / total);
            result.samples[dst * 2 + static_cast<std::size_t>(channel)] =
                std::max(-1.0F, std::min(1.0F, value));
        }
    }
    return result;
}

} // namespace

VideoImageInput resize_minimax_h3_image_lanczos(const VideoImageInput& source,
                                                int32_t target_height, int32_t target_width) {
    validate_image(source, "MiniMax-H3 image");
    if (target_height <= 0 || target_width <= 0)
        throw std::invalid_argument("MiniMax-H3 resize target must have positive dimensions");
    VideoImageInput result;
    result.height = target_height;
    result.width = target_width;
    result.channels = 3;
    result.pixels.resize(checked_product(
        {static_cast<std::size_t>(target_height), static_cast<std::size_t>(target_width), 3U},
        "MiniMax-H3 resized image"));
    if (source.height == target_height && source.width == target_width) {
        for (int32_t y = 0; y < source.height; ++y) {
            for (int32_t x = 0; x < source.width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    result.pixels[(static_cast<std::size_t>(y) * target_width + x) * 3 + channel] =
                        source.pixels[(static_cast<std::size_t>(y) * source.width + x) *
                                          source.channels +
                                      channel];
                }
            }
        }
        return result;
    }
    resize_rgb_lanczos(source.pixels.data(), source.height, source.width, source.channels,
                       result.pixels.data(), target_height, target_width);
    return result;
}

MiniMaxH3PreparedKeyframes
prepare_minimax_h3_keyframes(const std::optional<VideoImageInput>& first_frame,
                             const std::optional<VideoImageInput>& last_frame,
                             int32_t target_height, int32_t target_width, int32_t output_frames) {
    if (!first_frame && !last_frame)
        throw std::invalid_argument(
            "MiniMax-H3 FL2VA requires a first frame, a last frame, or both");
    if (target_height <= 0 || target_width <= 0 || target_height % kCanvasMultiple != 0 ||
        target_width % kCanvasMultiple != 0)
        throw std::invalid_argument(
            "MiniMax-H3 FL2VA canvas axes must be positive multiples of 32");
    if (output_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 FL2VA output frame count must be positive");

    MiniMaxH3PreparedKeyframes result;
    std::vector<std::pair<const VideoImageInput*, int32_t>> supplied;
    if (first_frame)
        supplied.emplace_back(&*first_frame, 0);
    if (last_frame)
        supplied.emplace_back(&*last_frame, output_frames - 1);
    result.images.reserve(supplied.size());
    result.anchors.reserve(supplied.size());
    for (std::size_t index = 0; index < supplied.size(); ++index) {
        const auto& source = *supplied[index].first;
        validate_image(source, "MiniMax-H3 FL2VA keyframe");
        if (source.height == target_height && source.width == target_width) {
            result.images.push_back(
                resize_minimax_h3_image_lanczos(source, target_height, target_width));
        } else if (index == 0) {
            result.images.push_back(
                resize_minimax_h3_image_lanczos(source, target_height, target_width));
        } else {
            const double scale = std::max(static_cast<double>(target_width) / source.width,
                                          static_cast<double>(target_height) / source.height);
            const int32_t resized_width = static_cast<int32_t>(std::max<int64_t>(
                target_width, round_ties_to_even(static_cast<double>(source.width) * scale)));
            const int32_t resized_height = static_cast<int32_t>(std::max<int64_t>(
                target_height, round_ties_to_even(static_cast<double>(source.height) * scale)));
            const int32_t left = std::max(0, (resized_width - target_width) / 2);
            const int32_t top = std::max(0, (resized_height - target_height) / 2);
            auto resized = resize_minimax_h3_image_lanczos(source, resized_height, resized_width);
            result.images.push_back(crop_rgb(resized, top, left, target_height, target_width));
        }
        result.anchors.push_back(supplied[index].second);
    }
    return result;
}

VideoImageInput normalize_minimax_h3_reference_image(const VideoImageInput& source) {
    validate_image(source, "MiniMax-H3 reference image");
    if (static_cast<int64_t>(source.width) > 4LL * source.height ||
        static_cast<int64_t>(source.height) > 4LL * source.width)
        throw std::invalid_argument(
            "MiniMax-H3 reference image aspect must be within 1:4 through 4:1");
    const double scale =
        static_cast<double>(kReferenceImageShortEdge) / std::min(source.width, source.height);
    const int32_t target_height = static_cast<int32_t>(std::max<int64_t>(
        kCanvasMultiple,
        round_ties_to_even(static_cast<double>(source.height) * scale / kCanvasMultiple) *
            kCanvasMultiple));
    const int32_t target_width = static_cast<int32_t>(std::max<int64_t>(
        kCanvasMultiple,
        round_ties_to_even(static_cast<double>(source.width) * scale / kCanvasMultiple) *
            kCanvasMultiple));
    return resize_minimax_h3_image_lanczos(source, target_height, target_width);
}

std::vector<int32_t> make_minimax_h3_reference_frame_map(int32_t source_frames,
                                                         int32_t fps_numerator,
                                                         int32_t fps_denominator,
                                                         int32_t output_frames) {
    if (source_frames <= 0 || fps_numerator <= 0 || fps_denominator <= 0 || output_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 reference frame map requires positive inputs");
    std::vector<int32_t> source_indices;
    source_indices.reserve(static_cast<std::size_t>(output_frames));
    for (int32_t index = 0;
         index < source_frames && source_indices.size() < static_cast<std::size_t>(output_frames);
         ++index) {
        const int64_t begin = rounded_frame_slot(index, fps_numerator, fps_denominator);
        const int64_t end =
            rounded_frame_slot(static_cast<int64_t>(index) + 1, fps_numerator, fps_denominator);
        for (int64_t repeat = begin;
             repeat < end && source_indices.size() < static_cast<std::size_t>(output_frames);
             ++repeat)
            source_indices.push_back(index);
    }
    if (source_indices.empty())
        throw std::invalid_argument("MiniMax-H3 reference video has no frame on the 24 fps clock");
    return source_indices;
}

VideoClipInput normalize_minimax_h3_reference_video(const VideoClipInput& source,
                                                    int32_t output_frames) {
    validate_clip(source);
    if (output_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 output frame count must be positive");
    const MiniMaxH3Canvas canvas = resolve_minimax_h3_canvas(source.width, source.height);
    const std::size_t source_frame_stride = checked_product(
        {static_cast<std::size_t>(source.height), static_cast<std::size_t>(source.width),
         static_cast<std::size_t>(source.channels)},
        "MiniMax-H3 reference video frame");
    const std::size_t target_frame_stride = checked_product(
        {static_cast<std::size_t>(canvas.height), static_cast<std::size_t>(canvas.width), 3U},
        "MiniMax-H3 normalized reference frame");

    const auto source_indices = make_minimax_h3_reference_frame_map(
        source.num_frames, source.fps_numerator, source.fps_denominator, output_frames);

    VideoClipInput result;
    result.num_frames = static_cast<int32_t>(source_indices.size());
    result.height = canvas.height;
    result.width = canvas.width;
    result.channels = 3;
    result.fps_numerator = kModelFps;
    result.fps_denominator = 1;
    result.pixels.resize(checked_product({source_indices.size(), target_frame_stride},
                                         "MiniMax-H3 normalized reference video"));
    for (std::size_t target_index = 0; target_index < source_indices.size(); ++target_index) {
        const float* source_frame =
            source.pixels.data() +
            static_cast<std::size_t>(source_indices[target_index]) * source_frame_stride;
        float* target_frame = result.pixels.data() + target_index * target_frame_stride;
        if (source.height == canvas.height && source.width == canvas.width &&
            source.channels == 3) {
            std::copy_n(source_frame, target_frame_stride, target_frame);
        } else {
            resize_rgb_lanczos(source_frame, source.height, source.width, source.channels,
                               target_frame, canvas.height, canvas.width);
        }
    }
    if (!source.soundtrack.samples.empty())
        result.soundtrack = normalize_minimax_h3_reference_audio(source.soundtrack, output_frames);
    return result;
}

AudioResult normalize_minimax_h3_reference_audio(const AudioResult& source, int32_t output_frames) {
    if (output_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 output frame count must be positive");
    if (source.sample_rate <= 0 || (source.channels != 1 && source.channels != 2))
        throw std::invalid_argument(
            "MiniMax-H3 reference audio must be mono or stereo at a positive rate");
    if (source.samples.size() % static_cast<std::size_t>(source.channels) != 0)
        throw std::invalid_argument("MiniMax-H3 reference audio buffer size is invalid");
    if (source.samples.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("MiniMax-H3 reference audio is too large");
    if (source.num_samples != 0 &&
        source.num_samples != static_cast<int32_t>(source.samples.size()))
        throw std::invalid_argument("MiniMax-H3 reference audio sample metadata is inconsistent");

    const int64_t maximum_source_frames =
        (static_cast<int64_t>(output_frames) * source.sample_rate) / kModelFps;
    const std::size_t available_frames =
        source.samples.size() / static_cast<std::size_t>(source.channels);
    const std::size_t retained_frames =
        std::min<std::size_t>(available_frames, static_cast<std::size_t>(maximum_source_frames));
    std::vector<float> stereo(
        checked_product({retained_frames, 2U}, "MiniMax-H3 normalized stereo audio"));
    for (std::size_t index = 0; index < retained_frames; ++index) {
        const float left = source.samples[index * static_cast<std::size_t>(source.channels)];
        const float right = source.channels == 1 ? left : source.samples[index * 2 + 1];
        stereo[index * 2] = left;
        stereo[index * 2 + 1] = right;
    }
    if (source.sample_rate == kModelAudioRate) {
        AudioResult result;
        result.samples = std::move(stereo);
        result.num_samples = static_cast<int32_t>(result.samples.size());
        result.sample_rate = kModelAudioRate;
        result.channels = 2;
        return result;
    }
    return resample_audio_sinc(stereo, source.sample_rate, kModelAudioRate);
}

std::vector<VideoReferenceInput>
normalize_minimax_h3_references(const std::vector<VideoReferenceInput>& references,
                                int32_t output_frames) {
    if (references.empty())
        throw std::invalid_argument("MiniMax-H3 Ref2VA requires at least one reference");
    if (references.size() > 12)
        throw std::invalid_argument("MiniMax-H3 Ref2VA accepts at most 12 references");
    int32_t image_count = 0;
    int32_t video_count = 0;
    int32_t audio_count = 0;
    std::vector<VideoReferenceInput> result;
    result.reserve(references.size());
    for (const auto& reference : references) {
        VideoReferenceInput normalized;
        normalized.kind = reference.kind;
        switch (reference.kind) {
        case VideoReferenceKind::kImage:
            ++image_count;
            normalized.image = normalize_minimax_h3_reference_image(reference.image);
            break;
        case VideoReferenceKind::kVideo:
            ++video_count;
            normalized.video = normalize_minimax_h3_reference_video(reference.video, output_frames);
            break;
        case VideoReferenceKind::kAudio:
            ++audio_count;
            normalized.audio = normalize_minimax_h3_reference_audio(reference.audio, output_frames);
            break;
        }
        result.push_back(std::move(normalized));
    }
    if (image_count > 9 || video_count > 3 || audio_count > 3)
        throw std::invalid_argument("MiniMax-H3 Ref2VA reference count exceeds a modality limit");
    return result;
}

} // namespace trtmc
