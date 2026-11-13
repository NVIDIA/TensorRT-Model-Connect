/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_5_moe/runtime/native_visual.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace trtmc::qwen3_5_moe {
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
        throw std::invalid_argument("Qwen native media requires nonempty RGB8 or RGB float images");
    const uint64_t pixels = static_cast<uint64_t>(image.height) * image.width;
    const std::size_t element = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (pixels > std::numeric_limits<std::size_t>::max() / 3 / element ||
        image.byte_size != pixels * 3 * element)
        throw std::invalid_argument("Qwen native RGB storage does not match image dimensions");
}

float cubic(float x) {
    x = std::fabs(x);
    if (x < 1)
        return ((1.5F * x - 2.5F) * x) * x + 1;
    if (x < 2)
        return ((-0.5F * x + 2.5F) * x - 4) * x + 2;
    return 0;
}

struct Filter {
    int32_t first;
    std::vector<float> weights;
    float sum;
};
std::vector<Filter> filters(int32_t source, int32_t target) {
    std::vector<Filter> result;
    result.reserve(target);
    const float scale = static_cast<float>(source) / target;
    const float widen = std::max(1.0F, scale);
    for (int32_t i = 0; i < target; ++i) {
        const float center = (i + 0.5F) * scale;
        Filter filter{static_cast<int32_t>(std::ceil(center - 2 * widen - 0.5F)), {}, 0};
        const int32_t last = static_cast<int32_t>(std::floor(center + 2 * widen - 0.5F));
        for (int32_t k = filter.first; k <= last; ++k) {
            const float weight = cubic((k + 0.5F - center) / widen);
            filter.weights.push_back(weight);
            filter.sum += weight;
        }
        result.push_back(std::move(filter));
    }
    return result;
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
                throw std::invalid_argument("Qwen native RGB float pixels must be finite in [0,1]");
            converted[i] = static_cast<uint8_t>(std::lround(pixel * 255));
        }
        pixels = converted.data();
    }
    // Separable antialiased Catmull-Rom. Clamp source indices at edges, retain
    // floating-point horizontal intermediates, round to RGB8 only after both
    // axes, then normalize to FP16, matching the pinned provider RGB path.
    const auto horizontal = filters(image.width, width);
    const auto vertical = filters(image.height, height);
    std::vector<float> rows(static_cast<std::size_t>(image.height) * width * 3);
    for (uint32_t y = 0; y < image.height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const auto& f = horizontal[x];
            for (int32_t channel = 0; channel < 3; ++channel) {
                float value = 0;
                for (std::size_t k = 0; k < f.weights.size(); ++k) {
                    const int32_t sx = std::clamp(f.first + static_cast<int32_t>(k), 0,
                                                  static_cast<int32_t>(image.width) - 1);
                    value += f.weights[k] *
                             pixels[(static_cast<std::size_t>(y) * image.width + sx) * 3 + channel];
                }
                rows[(static_cast<std::size_t>(y) * width + x) * 3 + channel] = value * (1 / f.sum);
            }
        }
    }
    std::vector<uint16_t> result(static_cast<std::size_t>(height) * width * 3);
    for (int32_t y = 0; y < height; ++y) {
        const auto& f = vertical[y];
        for (int32_t x = 0; x < width; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                float value = 0;
                for (std::size_t k = 0; k < f.weights.size(); ++k) {
                    const int32_t sy = std::clamp(f.first + static_cast<int32_t>(k), 0,
                                                  static_cast<int32_t>(image.height) - 1);
                    value += f.weights[k] *
                             rows[(static_cast<std::size_t>(sy) * width + x) * 3 + channel];
                }
                const auto pixel =
                    static_cast<uint8_t>(std::clamp(value * (1 / f.sum), 0.0F, 255.0F) + 0.5F);
                result[(static_cast<std::size_t>(y) * width + x) * 3 + channel] = half_bits(
                    (static_cast<float>(pixel) / 255 - c.mean[channel]) / c.stddev[channel]);
            }
        }
    }
    return result;
}
} // namespace

void NativeVisionConfig::validate() const {
    const auto side = static_cast<int32_t>(std::sqrt(static_cast<double>(positions)));
    if (patch <= 0 || temporal <= 0 || merge <= 0 || positions <= 0 || side * side != positions ||
        head_dim <= 0 || head_dim % 4 || hidden <= 0 || min_tokens <= 0 ||
        max_tokens < min_tokens || !std::isfinite(theta) || theta <= 1 ||
        static_cast<int64_t>(patch) * merge > INT32_MAX ||
        static_cast<int64_t>(3) * temporal * patch * patch > INT32_MAX ||
        static_cast<int64_t>(max_tokens) * merge * merge > INT32_MAX)
        throw std::invalid_argument("Invalid Qwen native visual geometry or token profile");
    for (int32_t i = 0; i < 3; ++i)
        if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] <= 0)
            throw std::invalid_argument("Invalid Qwen native visual normalization");
}

std::array<int32_t, 2> native_visual_size(const NativeVisionConfig& c, int32_t frames, bool video,
                                          int32_t height, int32_t width) {
    c.validate();
    if (frames <= 0 || height <= 0 || width <= 0 || (!video && frames != 1) ||
        static_cast<double>(std::max(height, width)) / std::min(height, width) > 200)
        throw std::invalid_argument(
            "Qwen native visual dimensions/frame count/aspect ratio are invalid");
    const int64_t factor = static_cast<int64_t>(c.patch) * c.merge;
    const int64_t groups = video ? (static_cast<int64_t>(frames) + c.temporal - 1) / c.temporal : 1;
    if (groups > c.max_tokens)
        throw std::invalid_argument(
            "No resized visual shape fits the token profile; reduce frame count");
    // Pixel budgets below are divided by temporal extent, avoiding unnecessary
    // large intermediate frame*height*width products.
    const double min_area = static_cast<double>(c.min_tokens) * factor * factor;
    const double max_area = static_cast<double>(c.max_tokens) * factor * factor;
    const double raw_groups = video ? static_cast<double>(frames) / c.temporal : 1;
    const auto rounded = [factor](int64_t x) {
        int64_t q = x / factor, r = x % factor;
        if (2 * r > factor || (2 * r == factor && (q & 1)))
            ++q;
        return std::max(factor, q * factor);
    };
    int64_t h = rounded(height), w = rounded(width);
    auto shrink = [&](double temporal_groups) {
        const double beta = std::sqrt(temporal_groups * height * width / max_area);
        h = std::max(factor, static_cast<int64_t>(height / beta) / factor * factor);
        w = std::max(factor, static_cast<int64_t>(width / beta) / factor * factor);
    };
    if (static_cast<double>(groups) * h * w > max_area)
        shrink(raw_groups);
    else if (static_cast<double>(groups) * h * w < min_area) {
        const double beta = std::sqrt(min_area / (raw_groups * height * width));
        h = static_cast<int64_t>(std::ceil(height * beta / factor)) * factor;
        w = static_cast<int64_t>(std::ceil(width * beta / factor)) * factor;
    }
    bool constrained = false;
    if (static_cast<double>(groups) * h * w > max_area) {
        shrink(groups);
        constrained = true;
    }
    if (static_cast<double>(groups) * h * w < min_area ||
        static_cast<double>(groups) * h * w > max_area) {
        constrained = true;
        const int64_t low = (c.min_tokens + groups - 1) / groups, high = c.max_tokens / groups;
        if (low > high)
            throw std::invalid_argument("No resized visual shape fits the token profile");
        const int64_t target = static_cast<double>(groups) * h * w < min_area ? low : high;
        const double ratio = static_cast<double>(height) / width;
        double best = std::numeric_limits<double>::infinity();
        int64_t gap = INT64_MAX;
        for (int64_t p = 1; p <= high; ++p) {
            const int64_t lo = std::max<int64_t>(1, (low + p - 1) / p), hi = high / p;
            if (lo > hi)
                continue;
            const int64_t q = std::clamp<int64_t>(std::llround(p / ratio), lo, hi);
            const double distance = std::fabs(std::log(static_cast<double>(p) / q / ratio));
            const int64_t delta = std::abs(p * q - target);
            if (distance < best - 1e-12 || (std::fabs(distance - best) <= 1e-12 && delta < gap)) {
                best = distance;
                gap = delta;
                h = p * factor;
                w = q * factor;
            }
        }
    }
    if (h > INT32_MAX || w > INT32_MAX)
        throw std::invalid_argument("Qwen native resized dimensions overflow");
    if (constrained)
        std::cerr << "Qwen native visual token profile constrained resize to " << h << "x" << w
                  << "; preprocessing may differ from an unconstrained provider profile\n";
    return {static_cast<int32_t>(h), static_cast<int32_t>(w)};
}

TensorMap NativeVisualInputs::tensors() {
    const int64_t count = static_cast<int64_t>(height) * width;
    return {{"patches", {patches.data(), {count, patch_width}, DType::kFloat16}},
            {"rotary_angles", {rotary_angles.data(), {count, rotary_width}, DType::kFloat32}},
            {"position_indices", {position_indices.data(), {4, count}, DType::kInt32}},
            {"position_weights", {position_weights.data(), {4, count}, DType::kFloat16}}};
}

NativeVisualInputs native_visual_inputs(const NativeVisionConfig& c,
                                        Span<const internal::ImageView> frames, int32_t height,
                                        int32_t width) {
    c.validate();
    const int32_t factor = c.patch * c.merge;
    if (frames.empty() || frames.size() > static_cast<std::size_t>(c.temporal) || height <= 0 ||
        width <= 0 || height % factor || width % factor ||
        static_cast<int64_t>(height / factor) * (width / factor) > c.max_tokens)
        throw std::invalid_argument("Invalid Qwen native temporal group or resized visual grid");
    std::vector<std::vector<uint16_t>> pixels;
    for (const auto& frame : frames) {
        if (frame.height != frames[0].height || frame.width != frames[0].width)
            throw std::invalid_argument("Qwen native video frames must have matching dimensions");
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
                            half_bits((k / 2 ? ry : 1 - ry) * (k % 2 ? rx : 1 - rx));
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
            throw std::invalid_argument("Qwen native visual part has inconsistent groups");
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

Qwen35MoeMediaPrompt native_media_prompt(const std::vector<int32_t>& ids,
                                         const std::vector<NativeVisualGroup>& groups,
                                         int32_t hidden, DType dtype, int32_t image_token,
                                         int32_t video_token, int32_t vision_start,
                                         int32_t vision_end) {
    if (ids.empty() || ids.size() > INT32_MAX || hidden <= 0 ||
        (dtype != DType::kFloat16 && dtype != DType::kBFloat16 && dtype != DType::kFloat32))
        throw std::invalid_argument("Invalid Qwen native media prompt geometry or precision");
    Qwen35MoeMediaPrompt result;
    result.hidden_size = hidden;
    result.dtype = dtype;
    result.feature_rows.assign(ids.size(), -1);
    result.positions.resize(ids.size());
    int64_t logical = 0, feature_row = 0;
    std::size_t group_index = 0;
    for (std::size_t i = 0; i < ids.size();) {
        if (ids[i] != image_token && ids[i] != video_token) {
            if (logical >= INT32_MAX)
                throw std::invalid_argument("Qwen native media rotary positions overflow");
            result.positions[i++].fill(static_cast<int32_t>(logical++));
            continue;
        }
        if (group_index >= groups.size() || i == 0 || ids[i - 1] != vision_start)
            throw std::invalid_argument("Qwen native media tokens lack matching visual input");
        const auto& group = groups[group_index++];
        const int64_t count = static_cast<int64_t>(group.height) * group.width;
        if (group.height <= 0 || group.width <= 0 || count >= INT32_MAX ||
            static_cast<uint64_t>(count) >= ids.size() - i ||
            logical + std::max(group.height, group.width) >= INT32_MAX ||
            feature_row + count >= INT32_MAX ||
            group.features.size() / static_cast<std::size_t>(hidden) !=
                static_cast<std::size_t>(count) ||
            group.features.size() % hidden || ids[i + count] != vision_end)
            throw std::invalid_argument("Qwen native visual span and feature geometry disagree");
        const int32_t marker = ids[i];
        // Source video groups use video-pad; retain support for older image-pad
        // expansion, but never reinterpret an image as a video group.
        if (marker == video_token && !group.video)
            throw std::invalid_argument("Qwen native video token was supplied a still-image group");
        for (int32_t row = 0; row < group.height; ++row)
            for (int32_t col = 0; col < group.width; ++col, ++i, ++feature_row) {
                if (ids[i] != marker)
                    throw std::invalid_argument("Qwen native visual token span is not contiguous");
                result.positions[i] = {static_cast<int32_t>(logical),
                                       static_cast<int32_t>(logical + row),
                                       static_cast<int32_t>(logical + col)};
                result.feature_rows[i] = static_cast<int32_t>(feature_row);
            }
        logical += std::max(group.height, group.width);
        const auto old_size = result.features.size();
        const auto item_size = dtype_size(dtype);
        if (group.features.size() > (result.features.max_size() - old_size) / item_size)
            throw std::invalid_argument("Qwen native media feature storage overflows");
        result.features.resize(old_size + group.features.size() * item_size);
        if (dtype == DType::kFloat16) {
            std::memcpy(result.features.data() + old_size, group.features.data(),
                        group.features.size() * sizeof(uint16_t));
        } else {
            for (std::size_t k = 0; k < group.features.size(); ++k) {
                __half_raw raw;
                raw.x = group.features[k];
                const float value = __half2float(__half(raw));
                auto* target = result.features.data() + old_size + k * item_size;
                if (dtype == DType::kFloat32)
                    std::memcpy(target, &value, sizeof(value));
                else {
                    const auto rounded = __float2bfloat16_rn(value);
                    std::memcpy(target, &rounded, sizeof(rounded));
                }
            }
        }
    }
    if (group_index != groups.size())
        throw std::invalid_argument("Qwen native visual inputs lack matching media tokens");
    result.decode_position = static_cast<int32_t>(logical);
    return result;
}

NativeVisualEncoder::NativeVisualEncoder(std::unique_ptr<ITrtModule> module,
                                         NativeVisionConfig config)
    : module_(std::move(module)), config_(config) {
    config_.validate();
    if (!module_ || !module_->ok() || !module_->has_output("visual_features") ||
        module_->tensor_dtype("visual_features") != DType::kFloat16)
        throw std::runtime_error("Invalid Qwen native visual engine");
    const auto maximum =
        module_->input_profile_shape("patches", module_->profile_idx(), ProfileShapeSelector::kMax);
    if (maximum != std::vector<int64_t>{
                       static_cast<int64_t>(config.max_tokens) * config.merge * config.merge,
                       static_cast<int64_t>(3) * config.temporal * config.patch * config.patch})
        throw std::runtime_error("Qwen native visual engine profile disagrees with metadata");
}

std::vector<NativeVisualGroup> NativeVisualEncoder::encode(const internal::ImageView& image) {
    return encode_frames({&image, 1}, {}, false);
}
std::vector<NativeVisualGroup> NativeVisualEncoder::encode(const internal::VideoView& video) {
    return encode_frames(video.frames, video.timestamps_seconds, true);
}
std::vector<NativeVisualGroup>
NativeVisualEncoder::encode_frames(Span<const internal::ImageView> frames,
                                   Span<const double> timestamps, bool video) {
    if (frames.empty() || frames.size() > INT32_MAX ||
        (!timestamps.empty() && timestamps.size() != frames.size()))
        throw std::invalid_argument("Qwen native video needs frames and matching timestamps");
    for (std::size_t i = 0; i < frames.size(); ++i) {
        validate_image(frames[i]);
        if (frames[i].height != frames[0].height || frames[i].width != frames[0].width)
            throw std::invalid_argument("Qwen native video frames must have matching dimensions");
        if (!timestamps.empty() &&
            (!std::isfinite(timestamps[i]) || (i && timestamps[i] < timestamps[i - 1])))
            throw std::invalid_argument("Qwen native video timestamps must be finite and ordered");
    }
    const auto size = native_visual_size(config_, static_cast<int32_t>(frames.size()), video,
                                         frames[0].height, frames[0].width);
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
            throw std::runtime_error("Qwen native visual output has invalid geometry/dtype");
        const auto last = start + count - 1;
        const double timestamp = timestamps.empty() ? (static_cast<double>(start) + last) / 2
                                                    : (timestamps[start] + timestamps[last]) / 2;
        NativeVisualGroup group{h, w, video, timestamp, std::vector<uint16_t>(features.numel())};
        std::memcpy(group.features.data(), features.data, features.nbytes());
        result.push_back(std::move(group));
    }
    return result;
}
} // namespace trtmc::qwen3_5_moe
