/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/media_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

constexpr int32_t kRgbChannels = 3;
constexpr int32_t kStereoChannels = 2;
constexpr int32_t kPillowResizePrecisionBits = 22;
constexpr int32_t kReferenceImageShortEdge = 2048;
constexpr int32_t kReferenceVideoShortEdge = 768;
constexpr int32_t kReferenceVideoMaxPixels = 768 * 1344;
constexpr int32_t kReferenceCanvasMultiple = 32;

std::size_t checked_element_count(std::initializer_list<int32_t> dimensions, const char* label) {
    std::size_t count = 1;
    for (const int32_t dimension : dimensions) {
        if (dimension <= 0 ||
            count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension))
            throw std::invalid_argument(std::string(label) + " has invalid dimensions");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

void validate_values(const std::vector<float>& values, std::size_t expected, float minimum,
                     float maximum, const char* label) {
    if (values.size() != expected)
        throw std::invalid_argument(std::string(label) +
                                    " buffer size does not match its dimensions");
    const auto invalid =
        std::find_if(values.begin(), values.end(), [minimum, maximum](float value) {
            return !std::isfinite(value) || value < minimum || value > maximum;
        });
    if (invalid != values.end())
        throw std::invalid_argument(std::string(label) + " contains invalid values");
}

void validate_image(const MediaImageInput& image) {
    const auto expected =
        checked_element_count({image.height, image.width, kRgbChannels}, "MiniMax-H3 keyframe");
    validate_values(image.pixels, expected, 0.0F, 1.0F, "MiniMax-H3 keyframe");
}

void validate_audio(const MultiChannelAudioResult& audio);

std::size_t validate_video(const MediaVideoInput& video) {
    if (!std::isfinite(video.fps) || video.fps <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 reference video fps must be positive and finite");
    const auto frame_elements = checked_element_count({video.height, video.width, kRgbChannels},
                                                      "MiniMax-H3 reference video frame");
    const auto expected = checked_element_count(
        {video.num_frames, video.height, video.width, kRgbChannels}, "MiniMax-H3 reference video");
    validate_values(video.pixels, expected, 0.0F, 1.0F, "MiniMax-H3 reference video");
    if (video.soundtrack.has_value())
        validate_audio(*video.soundtrack);
    return frame_elements;
}

void validate_audio(const MultiChannelAudioResult& audio) {
    if (audio.num_channels != 1 && audio.num_channels != kStereoChannels)
        throw std::invalid_argument("MiniMax-H3 reference audio must be mono or stereo");
    if (audio.num_samples <= 0 || audio.sample_rate <= 0)
        throw std::invalid_argument(
            "MiniMax-H3 reference audio needs positive samples and sample rate");
    const auto expected = checked_element_count({audio.num_channels, audio.num_samples},
                                                "MiniMax-H3 reference audio");
    validate_values(audio.samples, expected, -1.0F, 1.0F, "MiniMax-H3 reference audio");
}

double pillow_sinc(double value) {
    if (value == 0.0)
        return 1.0;
    constexpr double kPi = 3.141592653589793238462643383279502884;
    value *= kPi;
    return std::sin(value) / value;
}

double pillow_lanczos3(double value) {
    if (-3.0 <= value && value < 3.0)
        return pillow_sinc(value) * pillow_sinc(value / 3.0);
    return 0.0;
}

struct ResizeContribution {
    std::vector<std::pair<int32_t, int32_t>> weights;
};

std::vector<ResizeContribution> make_pillow_lanczos_contributions(int32_t source_size,
                                                                  int32_t target_size) {
    const double scale = static_cast<double>(source_size) / target_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 3.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<ResizeContribution> result(static_cast<std::size_t>(target_size));
    for (int32_t target = 0; target < target_size; ++target) {
        const double center = (static_cast<double>(target) + 0.5) * scale;
        const int32_t first = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t last = std::min(static_cast<int32_t>(center + support + 0.5), source_size);
        std::vector<double> normalized(static_cast<std::size_t>(std::max(0, last - first)));
        double sum = 0.0;
        for (int32_t index = first; index < last; ++index) {
            const double weight =
                pillow_lanczos3((static_cast<double>(index) - center + 0.5) * inverse_filter_scale);
            normalized[static_cast<std::size_t>(index - first)] = weight;
            sum += weight;
        }
        auto& weights = result[static_cast<std::size_t>(target)].weights;
        weights.reserve(normalized.size());
        for (std::size_t index = 0; index < normalized.size(); ++index) {
            const double weight = sum == 0.0 ? normalized[index] : normalized[index] / sum;
            const double scaled = weight * static_cast<double>(1 << kPillowResizePrecisionBits);
            const auto fixed = static_cast<int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
            weights.emplace_back(first + static_cast<int32_t>(index), fixed);
        }
    }
    return result;
}

uint8_t float_to_uint8_pixel(float value) {
    const float scaled = std::clamp(value, 0.0F, 1.0F) * 255.0F;
    const float lower = std::floor(scaled);
    const float fraction = scaled - lower;
    int32_t rounded = static_cast<int32_t>(lower);
    if (fraction > 0.5F || (fraction == 0.5F && rounded % 2 != 0))
        ++rounded;
    return static_cast<uint8_t>(std::clamp(rounded, 0, 255));
}

uint8_t pillow_clip8(int32_t value) {
    const int32_t shifted = value >> kPillowResizePrecisionBits;
    return static_cast<uint8_t>(std::clamp(shifted, 0, 255));
}

std::vector<uint8_t> make_uint8_hwc(const std::vector<float>& pixels) {
    std::vector<uint8_t> result(pixels.size());
    std::transform(pixels.begin(), pixels.end(), result.begin(), float_to_uint8_pixel);
    return result;
}

std::vector<uint8_t>
resize_lanczos_horizontal(const std::vector<uint8_t>& source, int32_t source_width,
                          int32_t source_height, int32_t target_width,
                          const std::vector<ResizeContribution>& contributions) {
    std::vector<uint8_t> result(static_cast<std::size_t>(source_height) * target_width *
                                kRgbChannels);
    for (int32_t y = 0; y < source_height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            for (int32_t channel = 0; channel < kRgbChannels; ++channel) {
                int32_t accumulator = 1 << (kPillowResizePrecisionBits - 1);
                for (const auto& [source_x, weight] :
                     contributions[static_cast<std::size_t>(x)].weights) {
                    const auto source_index =
                        (static_cast<std::size_t>(y) * source_width + source_x) * kRgbChannels +
                        channel;
                    accumulator += static_cast<int32_t>(source[source_index]) * weight;
                }
                const auto target_index =
                    (static_cast<std::size_t>(y) * target_width + x) * kRgbChannels + channel;
                result[target_index] = pillow_clip8(accumulator);
            }
        }
    }
    return result;
}

std::vector<float> resize_lanczos_vertical(const std::vector<uint8_t>& source, int32_t target_width,
                                           int32_t target_height,
                                           const std::vector<ResizeContribution>& contributions) {
    std::vector<float> result(static_cast<std::size_t>(target_height) * target_width *
                              kRgbChannels);
    for (int32_t y = 0; y < target_height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            for (int32_t channel = 0; channel < kRgbChannels; ++channel) {
                int32_t accumulator = 1 << (kPillowResizePrecisionBits - 1);
                for (const auto& [source_y, weight] :
                     contributions[static_cast<std::size_t>(y)].weights) {
                    const auto source_index =
                        (static_cast<std::size_t>(source_y) * target_width + x) * kRgbChannels +
                        channel;
                    accumulator += static_cast<int32_t>(source[source_index]) * weight;
                }
                const auto target_index =
                    (static_cast<std::size_t>(y) * target_width + x) * kRgbChannels + channel;
                result[target_index] = static_cast<float>(pillow_clip8(accumulator)) / 255.0F;
            }
        }
    }
    return result;
}

std::vector<float> resize_pillow_lanczos(const std::vector<float>& source, int32_t source_height,
                                         int32_t source_width, int32_t target_height,
                                         int32_t target_width) {
    if (source_height == target_height && source_width == target_width)
        return source;
    const auto horizontal_weights = make_pillow_lanczos_contributions(source_width, target_width);
    const auto vertical_weights = make_pillow_lanczos_contributions(source_height, target_height);
    const auto source_u8 = make_uint8_hwc(source);
    const auto horizontal = resize_lanczos_horizontal(source_u8, source_width, source_height,
                                                      target_width, horizontal_weights);
    return resize_lanczos_vertical(horizontal, target_width, target_height, vertical_weights);
}

int32_t python_round_positive(double value, const char* label) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0))
        rounded += 1.0;
    if (rounded > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument(std::string(label) + " is too large");
    return static_cast<int32_t>(rounded);
}

MiniMaxH3ReferenceCanvas resolve_reference_canvas(int32_t source_height, int32_t source_width,
                                                  int32_t short_edge, int32_t max_pixels) {
    if (source_height <= 0 || source_width <= 0 || source_width > 4 * source_height ||
        source_height > 4 * source_width)
        throw std::invalid_argument("MiniMax-H3 reference aspect ratio must be within 1:4 and 4:1");
    const double ratio = static_cast<double>(source_width) / source_height;
    double width = ratio >= 1.0 ? short_edge * ratio : short_edge;
    double height = ratio >= 1.0 ? short_edge : short_edge / ratio;
    const double area = width * height;
    if (max_pixels > 0 && area > max_pixels) {
        const double scale = std::sqrt(static_cast<double>(max_pixels) / area);
        width *= scale;
        height *= scale;
    }
    const auto round_axis = [](double axis, const char* label) {
        return std::max(kReferenceCanvasMultiple,
                        python_round_positive(axis / kReferenceCanvasMultiple, label) *
                            kReferenceCanvasMultiple);
    };
    MiniMaxH3ReferenceCanvas result;
    result.height = round_axis(height, "MiniMax-H3 reference height");
    result.width = round_axis(width, "MiniMax-H3 reference width");
    (void)checked_element_count({result.height, result.width, kRgbChannels},
                                "MiniMax-H3 reference canvas");
    return result;
}

std::vector<float> resize_reference_video_frames(const MediaVideoInput& video, int32_t frame_count,
                                                 const MiniMaxH3ReferenceCanvas& canvas) {
    const auto target_frame_size = checked_element_count(
        {canvas.height, canvas.width, kRgbChannels}, "MiniMax-H3 prepared reference frame");
    std::vector<float> result(static_cast<std::size_t>(frame_count) * target_frame_size);
    const std::size_t source_frame_size =
        static_cast<std::size_t>(video.height) * video.width * kRgbChannels;
    for (int32_t frame = 0; frame < frame_count; ++frame) {
        const auto begin =
            video.pixels.begin() + static_cast<std::ptrdiff_t>(frame) * source_frame_size;
        std::vector<float> source(begin, begin + static_cast<std::ptrdiff_t>(source_frame_size));
        auto resized =
            resize_pillow_lanczos(source, video.height, video.width, canvas.height, canvas.width);
        std::copy(resized.begin(), resized.end(),
                  result.begin() + static_cast<std::ptrdiff_t>(frame) * target_frame_size);
    }
    return result;
}

struct CoverGeometry {
    int32_t resized_height{0};
    int32_t resized_width{0};
    int32_t crop_top{0};
    int32_t crop_left{0};
};

CoverGeometry make_cover_geometry(int32_t source_height, int32_t source_width,
                                  int32_t target_height, int32_t target_width) {
    const double scale = std::max(static_cast<double>(target_width) / source_width,
                                  static_cast<double>(target_height) / source_height);
    CoverGeometry geometry;
    geometry.resized_width =
        std::max(target_width, python_round_positive(static_cast<double>(source_width) * scale,
                                                     "MiniMax-H3 resized keyframe width"));
    geometry.resized_height =
        std::max(target_height, python_round_positive(static_cast<double>(source_height) * scale,
                                                      "MiniMax-H3 resized keyframe height"));
    geometry.crop_left = std::max(0, (geometry.resized_width - target_width) / 2);
    geometry.crop_top = std::max(0, (geometry.resized_height - target_height) / 2);
    (void)checked_element_count({geometry.resized_height, geometry.resized_width, kRgbChannels},
                                "MiniMax-H3 resized keyframe");
    return geometry;
}

std::vector<float> crop_hwc(const std::vector<float>& source, int32_t source_width,
                            int32_t target_height, int32_t target_width, int32_t crop_top,
                            int32_t crop_left) {
    std::vector<float> result(checked_element_count({target_height, target_width, kRgbChannels},
                                                    "MiniMax-H3 cropped keyframe"));
    for (int32_t y = 0; y < target_height; ++y) {
        const auto source_offset =
            (static_cast<std::size_t>(y + crop_top) * source_width + crop_left) * kRgbChannels;
        const auto target_offset = static_cast<std::size_t>(y) * target_width * kRgbChannels;
        std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(source_offset),
                    static_cast<std::size_t>(target_width) * kRgbChannels,
                    result.begin() + static_cast<std::ptrdiff_t>(target_offset));
    }
    return result;
}

std::size_t rounded_reference_slot(std::size_t frame_index, double scale) {
    return static_cast<std::size_t>(std::floor(static_cast<double>(frame_index) * scale + 0.5));
}

std::size_t normalized_frame_count(int32_t source_frames, double scale) {
    const double rounded = std::floor(static_cast<double>(source_frames) * scale + 0.5);
    if (rounded > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("MiniMax-H3 normalized reference video is too long");
    return static_cast<std::size_t>(rounded);
}

void copy_normalized_frames(const MediaVideoInput& video, std::size_t frame_elements, double scale,
                            std::vector<float>& output) {
    std::size_t current_slot = 0;
    std::size_t output_offset = 0;
    for (int32_t source_frame = 0; source_frame < video.num_frames; ++source_frame) {
        const auto next_slot =
            rounded_reference_slot(static_cast<std::size_t>(source_frame) + 1U, scale);
        const auto source_offset = static_cast<std::size_t>(source_frame) * frame_elements;
        for (std::size_t slot = current_slot; slot < next_slot; ++slot) {
            std::copy_n(video.pixels.begin() + static_cast<std::ptrdiff_t>(source_offset),
                        frame_elements,
                        output.begin() + static_cast<std::ptrdiff_t>(output_offset));
            output_offset += frame_elements;
        }
        current_slot = next_slot;
    }
}

std::size_t truncated_audio_samples(const MultiChannelAudioResult& audio,
                                    double max_duration_seconds) {
    const double source_duration = static_cast<double>(audio.num_samples) / audio.sample_rate;
    if (max_duration_seconds >= source_duration)
        return static_cast<std::size_t>(audio.num_samples);
    return static_cast<std::size_t>(max_duration_seconds * audio.sample_rate);
}

std::vector<float> make_stereo_audio(const MultiChannelAudioResult& audio,
                                     std::size_t num_samples) {
    std::vector<float> result(num_samples * kStereoChannels);
    std::copy_n(audio.samples.begin(), num_samples, result.begin());
    const auto right_source =
        audio.num_channels == 1 ? audio.samples.begin() : audio.samples.begin() + audio.num_samples;
    std::copy_n(right_source, num_samples,
                result.begin() + static_cast<std::ptrdiff_t>(num_samples));
    return result;
}

struct SincResampleKernel {
    std::vector<float> values;
    int32_t phases{0};
    int32_t width{0};
    int32_t size{0};
};

SincResampleKernel make_torchaudio_sinc_hann_kernel(int32_t source_rate, int32_t target_rate) {
    const int32_t divisor = std::gcd(source_rate, target_rate);
    const int32_t source = source_rate / divisor;
    const int32_t target = target_rate / divisor;
    constexpr double lowpass_width = 6.0;
    constexpr double rolloff = 0.99;
    const double base_frequency = std::min(source, target) * rolloff;
    const int32_t width = static_cast<int32_t>(std::ceil(lowpass_width * source / base_frequency));
    const int32_t kernel_size = 2 * width + source;
    SincResampleKernel kernel;
    kernel.values.resize(static_cast<std::size_t>(target) * kernel_size);
    kernel.phases = target;
    kernel.width = width;
    kernel.size = kernel_size;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double scale = base_frequency / source;
    for (int32_t phase = 0; phase < target; ++phase) {
        for (int32_t tap = 0; tap < kernel_size; ++tap) {
            const int32_t index = tap - width;
            double time =
                (-static_cast<double>(phase) / target + static_cast<double>(index) / source) *
                base_frequency;
            time = std::clamp(time, -lowpass_width, lowpass_width);
            const double window = std::pow(std::cos(time * pi / lowpass_width / 2.0), 2.0);
            time *= pi;
            const double sinc = time == 0.0 ? 1.0 : std::sin(time) / time;
            kernel.values[static_cast<std::size_t>(phase) * kernel_size + tap] =
                static_cast<float>(sinc * window * scale);
        }
    }
    return kernel;
}

std::vector<float> resample_torchaudio_sinc_hann(const std::vector<float>& channel_major,
                                                 int32_t channels, int32_t source_samples,
                                                 int32_t source_rate, int32_t target_rate) {
    const int32_t divisor = std::gcd(source_rate, target_rate);
    const int32_t source = source_rate / divisor;
    const int32_t target = target_rate / divisor;
    const auto kernel = make_torchaudio_sinc_hann_kernel(source_rate, target_rate);
    const std::size_t target_samples =
        static_cast<std::size_t>(std::ceil(static_cast<double>(target) * source_samples / source));
    const std::size_t convolution_steps = static_cast<std::size_t>(source_samples / source + 1);
    std::vector<float> result(static_cast<std::size_t>(channels) * target_samples);
    for (int32_t channel = 0; channel < channels; ++channel) {
        std::size_t output = 0;
        for (std::size_t step = 0; step < convolution_steps && output < target_samples; ++step) {
            for (int32_t phase = 0; phase < kernel.phases && output < target_samples; ++phase) {
                float sum = 0.0F;
                for (int32_t tap = 0; tap < kernel.size; ++tap) {
                    const int64_t source_index =
                        static_cast<int64_t>(step * source + tap) - kernel.width;
                    if (source_index < 0 || source_index >= source_samples)
                        continue;
                    const auto input_offset = static_cast<std::size_t>(channel) * source_samples +
                                              static_cast<std::size_t>(source_index);
                    const auto kernel_offset = static_cast<std::size_t>(phase) * kernel.size + tap;
                    sum += channel_major[input_offset] * kernel.values[kernel_offset];
                }
                result[static_cast<std::size_t>(channel) * target_samples + output++] = sum;
            }
        }
    }
    return result;
}

} // namespace

MediaImageInput minimax_h3_prepare_keyframe_image(const MediaImageInput& image,
                                                  int32_t target_height, int32_t target_width,
                                                  bool stretch) {
    validate_image(image);
    (void)checked_element_count({target_height, target_width, kRgbChannels},
                                "MiniMax-H3 target canvas");
    if (image.height == target_height && image.width == target_width)
        return image;

    MediaImageInput result;
    result.height = target_height;
    result.width = target_width;
    if (stretch) {
        result.pixels = resize_pillow_lanczos(image.pixels, image.height, image.width,
                                              target_height, target_width);
        return result;
    }

    const auto geometry =
        make_cover_geometry(image.height, image.width, target_height, target_width);
    const auto resized = resize_pillow_lanczos(image.pixels, image.height, image.width,
                                               geometry.resized_height, geometry.resized_width);
    result.pixels = crop_hwc(resized, geometry.resized_width, target_height, target_width,
                             geometry.crop_top, geometry.crop_left);
    return result;
}

MediaVideoInput minimax_h3_normalize_reference_video_fps(const MediaVideoInput& video) {
    const auto frame_elements = validate_video(video);
    if (video.fps == static_cast<float>(kMiniMaxH3ReferenceFps))
        return video;

    const double scale = static_cast<double>(kMiniMaxH3ReferenceFps) / video.fps;
    const auto output_frames = normalized_frame_count(video.num_frames, scale);
    if (output_frames > std::numeric_limits<std::size_t>::max() / frame_elements)
        throw std::invalid_argument("MiniMax-H3 normalized reference video is too large");

    MediaVideoInput result = video;
    result.pixels.assign(output_frames * frame_elements, 0.0F);
    copy_normalized_frames(video, frame_elements, scale, result.pixels);
    result.num_frames = static_cast<int32_t>(output_frames);
    result.fps = static_cast<float>(kMiniMaxH3ReferenceFps);
    return result;
}

MiniMaxH3ReferenceCanvas minimax_h3_resolve_reference_image_canvas(int32_t source_height,
                                                                   int32_t source_width) {
    return resolve_reference_canvas(source_height, source_width, kReferenceImageShortEdge, 0);
}

MiniMaxH3ReferenceCanvas minimax_h3_resolve_reference_video_canvas(int32_t source_height,
                                                                   int32_t source_width) {
    return resolve_reference_canvas(source_height, source_width, kReferenceVideoShortEdge,
                                    kReferenceVideoMaxPixels);
}

std::vector<float> minimax_h3_quantize_reference_pixels(const std::vector<float>& pixels) {
    std::vector<float> result(pixels.size());
    std::transform(pixels.begin(), pixels.end(), result.begin(), [](float value) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("MiniMax-H3 reference contains invalid RGB values");
        return static_cast<float>(float_to_uint8_pixel(value)) / 255.0F;
    });
    return result;
}

MediaImageInput minimax_h3_prepare_reference_image(const MediaImageInput& image) {
    validate_image(image);
    const auto canvas = minimax_h3_resolve_reference_image_canvas(image.height, image.width);
    MediaImageInput result = image;
    result.pixels = minimax_h3_quantize_reference_pixels(image.pixels);
    result.height = canvas.height;
    result.width = canvas.width;
    if (image.height != canvas.height || image.width != canvas.width)
        result.pixels = resize_pillow_lanczos(result.pixels, image.height, image.width,
                                              result.height, result.width);
    return result;
}

MediaVideoInput minimax_h3_prepare_reference_video(const MediaVideoInput& video,
                                                   int32_t max_frames) {
    if (max_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 reference video frame cap must be positive");
    auto normalized = minimax_h3_normalize_reference_video_fps(video);
    normalized.pixels = minimax_h3_quantize_reference_pixels(normalized.pixels);
    const int32_t frame_count = std::min(normalized.num_frames, max_frames);
    const auto canvas =
        minimax_h3_resolve_reference_video_canvas(normalized.height, normalized.width);
    const std::size_t source_frame_size =
        static_cast<std::size_t>(normalized.height) * normalized.width * kRgbChannels;
    MediaVideoInput result = normalized;
    result.num_frames = frame_count;
    result.height = canvas.height;
    result.width = canvas.width;
    if (normalized.height == canvas.height && normalized.width == canvas.width) {
        result.pixels.assign(normalized.pixels.begin(),
                             normalized.pixels.begin() +
                                 static_cast<std::ptrdiff_t>(frame_count) * source_frame_size);
    } else {
        result.pixels = resize_reference_video_frames(normalized, frame_count, canvas);
    }
    return result;
}

int32_t minimax_h3_trim_reference_num_frames(int32_t num_frames) {
    if (num_frames < 1)
        throw std::invalid_argument("MiniMax-H3 reference video must contain a frame");
    constexpr int32_t frames_per_chunk = 17;
    constexpr int32_t trailing_frames = 5;
    return std::max(1, (num_frames - trailing_frames) / frames_per_chunk) * frames_per_chunk +
           trailing_frames;
}

MultiChannelAudioResult minimax_h3_prepare_reference_audio(const MultiChannelAudioResult& audio,
                                                           double max_duration_seconds) {
    validate_audio(audio);
    if (!std::isfinite(max_duration_seconds) || max_duration_seconds <= 0.0)
        throw std::invalid_argument(
            "MiniMax-H3 reference audio maximum duration must be positive and finite");
    const auto num_samples = truncated_audio_samples(audio, max_duration_seconds);
    MultiChannelAudioResult result;
    result.samples = make_stereo_audio(audio, num_samples);
    if (audio.sample_rate != kMiniMaxH3ReferenceAudioSampleRate) {
        result.samples = resample_torchaudio_sinc_hann(
            result.samples, kStereoChannels, static_cast<int32_t>(num_samples), audio.sample_rate,
            kMiniMaxH3ReferenceAudioSampleRate);
    }
    result.num_samples = static_cast<int32_t>(result.samples.size() / kStereoChannels);
    result.sample_rate = kMiniMaxH3ReferenceAudioSampleRate;
    result.num_channels = kStereoChannels;
    return result;
}

MultiChannelAudioResult
minimax_h3_align_reference_audio_for_vae(const MultiChannelAudioResult& audio) {
    validate_audio(audio);
    if (audio.num_channels != kStereoChannels ||
        audio.sample_rate != kMiniMaxH3ReferenceAudioSampleRate)
        throw std::invalid_argument(
            "MiniMax-H3 AudioVAE input must be prepared 32 kHz stereo audio");
    constexpr int32_t hop_length = 800;
    const int64_t aligned =
        ((static_cast<int64_t>(audio.num_samples) + hop_length - 1) / hop_length) * hop_length;
    if (aligned > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("MiniMax-H3 AudioVAE-aligned reference is too long");
    const int32_t aligned_samples = static_cast<int32_t>(aligned);
    if (aligned_samples == audio.num_samples)
        return audio;

    MultiChannelAudioResult result;
    result.samples.assign(static_cast<std::size_t>(kStereoChannels) * aligned_samples, 0.0F);
    for (int32_t channel = 0; channel < kStereoChannels; ++channel) {
        const auto source =
            audio.samples.begin() + static_cast<std::ptrdiff_t>(channel) * audio.num_samples;
        const auto target =
            result.samples.begin() + static_cast<std::ptrdiff_t>(channel) * aligned_samples;
        std::copy_n(source, audio.num_samples, target);
    }
    result.num_samples = aligned_samples;
    result.sample_rate = audio.sample_rate;
    result.num_channels = audio.num_channels;
    return result;
}

} // namespace trtmc
