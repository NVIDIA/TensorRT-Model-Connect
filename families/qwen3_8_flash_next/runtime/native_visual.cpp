/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8_flash_next/runtime/native_visual.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace trtmc::qwen3_8_flash_next {
namespace {
uint16_t half_bits(float value) {
    const __half half = __float2half_rn(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

void validate_image(const internal::ImageView& image) {
    if (!image.data || !image.height || !image.width || image.channels != 3 ||
        image.height > INT32_MAX || image.width > INT32_MAX ||
        (image.format != internal::ImageFormat::UInt8 &&
         image.format != internal::ImageFormat::Float32))
        throw std::invalid_argument(
            "Flash-Next native media requires nonempty RGB8 or RGB float images");
    const uint64_t pixels = static_cast<uint64_t>(image.height) * image.width;
    const std::size_t element = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (pixels > std::numeric_limits<std::size_t>::max() / 3 / element ||
        image.byte_size != pixels * 3 * element)
        throw std::invalid_argument(
            "Flash-Next native RGB storage does not match image dimensions");
}

double cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.5 * x - 2.5) * x) * x + 1;
    if (x < 2)
        return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}
struct Tap {
    int32_t first;
    std::vector<int16_t> coefficients;
};
struct Axis {
    std::vector<Tap> taps;
    int shift;
};
Axis filters(int32_t source, int32_t target) {
    const double scale = double(source) / target, width = std::max(1.0, scale);
    std::vector<std::vector<double>> rows(target);
    Axis axis{{}, 0};
    double largest = 0;
    for (int32_t i = 0; i < target; ++i) {
        const double center = scale * (i + 0.5);
        const int32_t first = static_cast<int32_t>(std::max(0.0, center - 2 * width + 0.5));
        const int32_t end =
            static_cast<int32_t>(std::min(double(source), center + 2 * width + 0.5));
        auto& w = rows[i];
        double sum = 0;
        for (int32_t k = first; k < end; ++k) {
            const double v = cubic((k - center + 0.5) / width);
            w.push_back(v);
            sum += v;
        }
        for (auto& v : w) {
            v /= sum;
            largest = std::max(largest, v);
        }
        axis.taps.push_back({first, {}});
    }
    while (axis.shift < 22 && std::lround(std::ldexp(largest, axis.shift + 1)) < 32768)
        ++axis.shift;
    for (int32_t i = 0; i < target; ++i)
        for (double w : rows[i])
            axis.taps[i].coefficients.push_back(
                static_cast<int16_t>(std::lround(std::ldexp(w, axis.shift))));
    return axis;
}
template <class Pixel>
uint8_t resample(const Axis& axis, int32_t i, Pixel pixel) {
    const auto& tap = axis.taps[i];
    int64_t value = int64_t{1} << (axis.shift - 1);
    for (std::size_t j = 0; j < tap.coefficients.size(); ++j)
        value += tap.coefficients[j] * int64_t(pixel(tap.first + static_cast<int32_t>(j)));
    return static_cast<uint8_t>(
        std::clamp<int64_t>(std::max<int64_t>(0, value) >> axis.shift, 0, 255));
}

std::vector<uint16_t> normalized(const NativeVisionConfig& c, const internal::ImageView& image,
                                 int32_t height, int32_t width) {
    validate_image(image);
    const auto count = static_cast<std::size_t>(image.height) * image.width * 3;
    std::vector<uint8_t> converted;
    const auto* pixels = static_cast<const uint8_t*>(image.data);
    if (image.format == internal::ImageFormat::Float32) {
        converted.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            float pixel;
            std::memcpy(&pixel, pixels + i * sizeof(float), sizeof(float));
            if (!std::isfinite(pixel) || pixel < 0 || pixel > 1)
                throw std::invalid_argument(
                    "Flash-Next native RGB float pixels must be finite in [0,1]");
            converted[i] = static_cast<uint8_t>(std::lround(pixel * 255));
        }
        pixels = converted.data();
    }
    // The provider resizes byte RGB with a separable antialiased cubic. It
    // truncates and renormalizes edge filters and rounds to bytes after each
    // dimension, before converting and normalizing the image for the model.
    const auto horizontal = filters(image.width, width), vertical = filters(image.height, height);
    std::vector<uint8_t> rows(static_cast<size_t>(image.height) * width * 3);
    for (uint32_t y = 0; y < image.height; ++y)
        for (int32_t x = 0; x < width; ++x)
            for (int c0 = 0; c0 < 3; ++c0)
                rows[(static_cast<size_t>(y) * width + x) * 3 + c0] =
                    resample(horizontal, x, [&](int sx) {
                        return pixels[(static_cast<size_t>(y) * image.width + sx) * 3 + c0];
                    });
    std::vector<uint16_t> result(static_cast<size_t>(height) * width * 3);
    for (int32_t y = 0; y < height; ++y)
        for (int32_t x = 0; x < width; ++x)
            for (int c0 = 0; c0 < 3; ++c0) {
                const auto pixel = resample(vertical, y, [&](int sy) {
                    return rows[(static_cast<size_t>(sy) * width + x) * 3 + c0];
                });
                result[(static_cast<size_t>(y) * width + x) * 3 + c0] =
                    half_bits((float(pixel) / 255 - c.mean[c0]) / c.stddev[c0]);
            }
    return result;
}
} // namespace

void NativeVisionConfig::validate() const {
    if (patch != 16 || temporal != 2 || merge != 2 || positions != 2304 || head_dim != 72 ||
        hidden != 2560 || theta != 10000 || max_tokens <= 0 || image_min_pixels <= 0 ||
        image_max_pixels < image_min_pixels || video_min_pixels <= 0 ||
        video_max_pixels < video_min_pixels)
        throw std::invalid_argument("Invalid Flash-Next native visual geometry");
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] <= 0)
            throw std::invalid_argument("Invalid Flash-Next visual normalization");
}
std::array<int32_t, 2> native_visual_size(const NativeVisionConfig& c, int32_t frames, bool video,
                                          int32_t height, int32_t width) {
    c.validate();
    if (height <= 0 || width <= 0 || frames < (video ? c.temporal : 1) || (!video && frames != 1))
        throw std::invalid_argument("Invalid Flash-Next source image or video dimensions");
    const int64_t factor = c.patch * c.merge;
    int64_t raw_h = height, raw_w = width;
    if (video && (raw_h < factor || raw_w < factor)) {
        const double scale = std::max(double(factor) / raw_h, double(factor) / raw_w);
        raw_h = static_cast<int64_t>(raw_h * scale);
        raw_w = static_cast<int64_t>(raw_w * scale);
    }
    if (std::max(raw_h, raw_w) > 200 * std::min(raw_h, raw_w))
        throw std::invalid_argument("Flash-Next source aspect ratio exceeds the provider limit");
    const auto rounded = [](int64_t value, int64_t unit) {
        const auto q = value / unit, r = value % unit;
        return (q + (2 * r > unit || (2 * r == unit && (q & 1)))) * unit;
    };
    int64_t h = rounded(raw_h, factor), w = rounded(raw_w, factor);
    const double extent = video ? rounded(frames, c.temporal) : 1;
    const double min = video ? c.video_min_pixels : c.image_min_pixels,
                 max = video ? c.video_max_pixels : c.image_max_pixels;
    const double area = extent * h * w, original = double(frames) * raw_h * raw_w;
    if (area > max) {
        const double beta = std::sqrt(original / max);
        h = std::max(factor, static_cast<int64_t>(std::floor(raw_h / beta / factor)) * factor);
        w = std::max(factor, static_cast<int64_t>(std::floor(raw_w / beta / factor)) * factor);
    } else if (area < min) {
        const double beta = std::sqrt(min / original);
        h = static_cast<int64_t>(std::ceil(raw_h * beta / factor)) * factor;
        w = static_cast<int64_t>(std::ceil(raw_w * beta / factor)) * factor;
    }
    if (h <= 0 || w <= 0 || h > INT32_MAX || w > INT32_MAX)
        throw std::invalid_argument("Invalid provider-sized Flash-Next visual dimensions");
    const auto tokens = (h / factor) * (w / factor);
    if (tokens > c.max_tokens)
        throw std::invalid_argument(
            "Provider-sized Flash-Next image needs " + std::to_string(tokens) +
            " visual tokens; the native limit is " + std::to_string(c.max_tokens));
    return {static_cast<int32_t>(h), static_cast<int32_t>(w)};
}

TensorMap NativeVisualInputs::tensors() {
    const int64_t count = static_cast<int64_t>(height) * width;
    return {{"patches", {patches.data(), {count, patch_width}, DType::kFloat16}},
            {"rotary_angles", {rotary_angles.data(), {count, rotary_width}, DType::kFloat32}},
            {"position_indices", {position_indices.data(), {4, count}, DType::kInt32}},
            {"position_weights", {position_weights.data(), {4, count}, DType::kFloat32}}};
}

NativeVisualInputs native_visual_inputs(const NativeVisionConfig& c,
                                        Span<const internal::ImageView> frames, int32_t height,
                                        int32_t width) {
    c.validate();
    const int32_t factor = c.patch * c.merge;
    if (frames.empty() || frames.size() > static_cast<std::size_t>(c.temporal) || height <= 0 ||
        width <= 0 || height % factor || width % factor ||
        static_cast<int64_t>(height / factor) * (width / factor) > c.max_tokens)
        throw std::invalid_argument(
            "Invalid Flash-Next native temporal group or resized visual grid");
    std::vector<std::vector<uint16_t>> pixels;
    for (const auto& frame : frames) {
        if (frame.height != frames[0].height || frame.width != frames[0].width)
            throw std::invalid_argument(
                "Flash-Next native video frames must have matching dimensions");
        pixels.push_back(normalized(c, frame, height, width));
    }
    NativeVisualInputs result;
    result.height = height / c.patch;
    result.width = width / c.patch;
    result.patch_width = 3 * c.temporal * c.patch * c.patch;
    result.rotary_width = c.head_dim / 2;
    const std::size_t count = static_cast<std::size_t>(result.height) * result.width;
    result.patches.resize(count * result.patch_width);
    result.rotary_angles.resize(count * result.rotary_width);
    result.position_indices.resize(4 * count);
    result.position_weights.resize(4 * count);
    const auto side = static_cast<int32_t>(std::sqrt(static_cast<double>(c.positions)));
    std::size_t token = 0;
    for (int32_t block_y = 0; block_y < result.height; block_y += c.merge)
        for (int32_t block_x = 0; block_x < result.width; block_x += c.merge)
            for (int32_t dy = 0; dy < c.merge; ++dy)
                for (int32_t dx = 0; dx < c.merge; ++dx, ++token) {
                    const int32_t y = block_y + dy, x = block_x + dx;
                    std::size_t out = token * result.patch_width;
                    for (int32_t channel = 0; channel < 3; ++channel)
                        for (int32_t t = 0; t < c.temporal; ++t)
                            for (int32_t py = 0; py < c.patch; ++py)
                                for (int32_t px = 0; px < c.patch; ++px) {
                                    const auto& frame =
                                        pixels[std::min<std::size_t>(t, pixels.size() - 1)];
                                    result.patches[out++] =
                                        frame[(static_cast<std::size_t>(y * c.patch + py) * width +
                                               x * c.patch + px) *
                                                  3 +
                                              channel];
                                }
                    for (int32_t i = 0; i < c.head_dim / 4; ++i) {
                        const float inverse =
                            std::pow(c.theta, -static_cast<float>(2 * i) / (c.head_dim / 2));
                        result.rotary_angles[token * result.rotary_width + i] = y * inverse;
                        result.rotary_angles[token * result.rotary_width + c.head_dim / 4 + i] =
                            x * inverse;
                    }
                    const float fy = result.height == 1
                                         ? 0
                                         : static_cast<float>(static_cast<double>(side - 1) * y /
                                                              (result.height - 1));
                    const float fx = result.width == 1
                                         ? 0
                                         : static_cast<float>(static_cast<double>(side - 1) * x /
                                                              (result.width - 1));
                    const int32_t iy = static_cast<int32_t>(fy), ix = static_cast<int32_t>(fx);
                    const float ry = fy - iy, rx = fx - ix;
                    for (int32_t k = 0; k < 4; ++k) {
                        result.position_indices[k * count + token] =
                            std::min(iy + k / 2, side - 1) * side + std::min(ix + k % 2, side - 1);
                        result.position_weights[k * count + token] =
                            (k / 2 ? ry : 1 - ry) * (k % 2 ? rx : 1 - rx);
                    }
                }
    return result;
}

std::string native_visual_text(const std::vector<NativeVisualGroup>& groups) {
    if (groups.empty())
        return {};
    const bool video = groups.front().video;
    std::string text = video ? "<|vision_start|>" : "";
    for (const auto& group : groups) {
        if (group.video != video || group.height <= 0 || group.width <= 0 ||
            (video && !std::isfinite(group.timestamp)))
            throw std::invalid_argument("Flash-Next native visual part has inconsistent groups");
        if (video) {
            std::ostringstream timestamp;
            timestamp.imbue(std::locale::classic());
            timestamp << '<' << std::fixed << std::setprecision(1) << group.timestamp
                      << " seconds>";
            text += timestamp.str();
        }
        text += "<|vision_start|>";
        for (int64_t i = 0; i < static_cast<int64_t>(group.height) * group.width; ++i)
            text += video ? "<|video_pad|>" : "<|image_pad|>";
        text += "<|vision_end|>";
    }
    if (video)
        text += "<|vision_end|>";
    return text;
}

NativeVisualEncoder::NativeVisualEncoder(std::unique_ptr<ITrtModule> module,
                                         NativeVisionConfig config)
    : module_(std::move(module)), config_(config) {
    config_.validate();
    if (!module_ || !module_->ok() || !module_->has_output("visual_features") ||
        module_->tensor_dtype("visual_features") != DType::kFloat16)
        throw std::runtime_error("Invalid Flash-Next native visual engine");
    const auto maximum =
        module_->input_profile_shape("patches", module_->profile_idx(), ProfileShapeSelector::kMax);
    if (maximum != std::vector<int64_t>{
                       static_cast<int64_t>(config.max_tokens) * config.merge * config.merge,
                       static_cast<int64_t>(3) * config.temporal * config.patch * config.patch})
        throw std::runtime_error("Flash-Next native visual engine profile disagrees with metadata");
}

std::vector<NativeVisualGroup> NativeVisualEncoder::encode(const internal::ImageView& image,
                                                           int32_t remaining_tokens) {
    return encode_frames({&image, 1}, {}, false, remaining_tokens);
}
std::vector<NativeVisualGroup> NativeVisualEncoder::encode(const internal::VideoView& video,
                                                           int32_t remaining_tokens) {
    return encode_frames(video.frames, video.timestamps_seconds, true, remaining_tokens);
}
std::vector<NativeVisualGroup>
NativeVisualEncoder::encode_frames(Span<const internal::ImageView> frames,
                                   Span<const double> timestamps, bool video,
                                   int32_t remaining_tokens) {
    if (frames.empty() || frames.size() > INT32_MAX ||
        (!timestamps.empty() && timestamps.size() != frames.size()))
        throw std::invalid_argument("Flash-Next native video needs frames and matching timestamps");
    for (std::size_t i = 0; i < frames.size(); ++i) {
        validate_image(frames[i]);
        if (frames[i].height != frames[0].height || frames[i].width != frames[0].width)
            throw std::invalid_argument(
                "Flash-Next native video frames must have matching dimensions");
        if (!timestamps.empty() &&
            (!std::isfinite(timestamps[i]) || (i && timestamps[i] < timestamps[i - 1])))
            throw std::invalid_argument(
                "Flash-Next native video timestamps must be finite and ordered");
    }
    const auto size = native_visual_size(config_, static_cast<int32_t>(frames.size()), video,
                                         frames[0].height, frames[0].width);
    const int64_t factor = config_.patch * config_.merge;
    const int64_t per_group = (size[0] / factor) * (size[1] / factor);
    const int64_t groups = 1 + (static_cast<int64_t>(frames.size()) - 1) / config_.temporal;
    if (remaining_tokens < 0 || per_group > remaining_tokens / groups)
        throw std::invalid_argument(
            "Flash-Next media exceeds the remaining native visual budget of " +
            std::to_string(remaining_tokens) + " tokens");
    std::vector<NativeVisualGroup> result;
    for (std::size_t start = 0; start < frames.size(); start += config_.temporal) {
        const auto count = std::min<std::size_t>(config_.temporal, frames.size() - start);
        auto prepared =
            native_visual_inputs(config_, {frames.data() + start, count}, size[0], size[1]);
        TensorMap output;
        try {
            output = module_->forward(prepared.tensors());
        } catch (...) {
            try {
                module_->sync();
            } catch (...) {
            }
            throw;
        }
        const auto& features = output.at("visual_features");
        const int32_t h = prepared.height / config_.merge, w = prepared.width / config_.merge;
        if (features.dtype != DType::kFloat16 ||
            features.shape != std::vector<int64_t>{static_cast<int64_t>(h) * w, config_.hidden})
            throw std::runtime_error("Flash-Next native visual output has invalid geometry/dtype");
        const auto last = start + count - 1;
        const double timestamp = timestamps.empty() ? (static_cast<double>(start) + last) / 2
                                                    : (timestamps[start] + timestamps[last]) / 2;
        NativeVisualGroup group{h, w, video, timestamp, std::vector<uint16_t>(features.numel())};
        std::memcpy(group.features.data(), features.data, features.nbytes());
        result.push_back(std::move(group));
    }
    return result;
}
} // namespace trtmc::qwen3_8_flash_next
