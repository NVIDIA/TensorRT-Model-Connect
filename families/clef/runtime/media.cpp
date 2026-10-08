/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clef/runtime/media.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace trtmc::clef {
namespace {
static int32_t preprocess_worker_count() {
    static const int32_t workers = []() -> int32_t {
        constexpr unsigned worker_cap = 8;
        const unsigned hardware = std::thread::hardware_concurrency();
        return static_cast<int32_t>(hardware == 0 ? worker_cap : std::min(hardware, worker_cap));
    }();
    return workers;
}

template <typename Function>
static void parallel_for_ranges(int32_t count, int32_t min_grain, Function&& function) {
    if (count <= 0)
        return;
    const int32_t by_grain = (count + min_grain - 1) / min_grain;
    const int32_t workers = std::min({count, by_grain, preprocess_worker_count()});
    if (workers <= 1) {
        function(0, count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int32_t worker = 0; worker < workers - 1; ++worker) {
        const int32_t begin = count * worker / workers;
        const int32_t end = count * (worker + 1) / workers;
        threads.emplace_back([&, begin, end] { function(begin, end); });
    }
    function(count * (workers - 1) / workers, count);
    for (auto& thread : threads)
        thread.join();
}

// Qwen2-VL's fast Hugging Face processor resizes uint8 tensors with
// torchvision bicubic interpolation and antialiasing enabled. In particular,
// downsampling widens the cubic support; a plain Catmull-Rom resize does not.
// The fixed-point coefficient conversion and per-axis uint8 rounding below
// mirror PyTorch's CPU uint8 path so preprocessing does not drift before the
// vision transformer.
struct AntialiasWeights {
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int16_t> values;
    int32_t stride{0};
    uint32_t precision{0};
};

static double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

static int32_t aligned_coefficient_stride(int32_t kernel_size) {
    int32_t stride = kernel_size;
    while (stride % static_cast<int32_t>(sizeof(int32_t)) != 0)
        ++stride;
    return stride;
}

static double build_floating_antialias_weights(int32_t input_size, int32_t output_size,
                                               double scale, double support, int32_t kernel_size,
                                               AntialiasWeights& result,
                                               std::vector<double>& floating) {
    double maximum_weight = 0.0;
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = scale * (output_index + 0.5);
        const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
        const int32_t start = std::max(static_cast<int32_t>(center - support + 0.5), 0);
        const int32_t size =
            std::clamp(std::min(static_cast<int32_t>(center + support + 0.5), input_size) - start,
                       0, kernel_size);
        result.starts[static_cast<std::size_t>(output_index)] = start;
        result.sizes[static_cast<std::size_t>(output_index)] = size;

        double total = 0.0;
        double* row = floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        for (int32_t index = 0; index < size; ++index) {
            row[index] = keys_cubic((index + start - center + 0.5) * inverse_scale);
            total += row[index];
        }
        if (total != 0.0) {
            for (int32_t index = 0; index < size; ++index) {
                row[index] /= total;
                maximum_weight = std::max(maximum_weight, row[index]);
            }
        }
    }
    return maximum_weight;
}

static void quantize_antialias_weights(const std::vector<double>& floating, int32_t output_size,
                                       int32_t kernel_size, double maximum_weight,
                                       AntialiasWeights& result) {
    // Select the greatest fixed-point precision whose largest coefficient fits
    // in int16, exactly as PyTorch's uint8 antialias implementation does.
    for (; result.precision < 22; ++result.precision) {
        const int32_t next = static_cast<int32_t>(
            0.5 + maximum_weight * static_cast<double>(uint32_t{1} << (result.precision + 1)));
        if (next >= (1 << 15))
            break;
    }

    const double multiplier = static_cast<double>(uint32_t{1} << result.precision);
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double* source =
            floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        int16_t* destination =
            result.values.data() + static_cast<std::size_t>(output_index) * result.stride;
        for (int32_t index = 0; index < kernel_size; ++index) {
            const double value = source[index] * multiplier;
            destination[index] =
                static_cast<int16_t>(value < 0.0 ? static_cast<int32_t>(value - 0.5)
                                                 : static_cast<int32_t>(value + 0.5));
        }
    }
}

static AntialiasWeights make_bicubic_antialias_weights(int32_t input_size, int32_t output_size) {
    constexpr int32_t interp_size = 4;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    const int32_t kernel_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;

    // PyTorch pads each int16 coefficient row to a 32-bit-aligned size in its
    // optimized uint8 path. The padding does not participate in convolution.
    AntialiasWeights result;
    result.stride = aligned_coefficient_stride(kernel_size);
    result.starts.resize(static_cast<std::size_t>(output_size));
    result.sizes.resize(static_cast<std::size_t>(output_size));
    result.values.assign(static_cast<std::size_t>(output_size) * result.stride, 0);

    std::vector<double> floating(static_cast<std::size_t>(output_size) * kernel_size, 0.0);
    const double maximum_weight = build_floating_antialias_weights(
        input_size, output_size, scale, support, kernel_size, result, floating);
    quantize_antialias_weights(floating, output_size, kernel_size, maximum_weight, result);
    return result;
}

static uint8_t fixed_point_pixel(const uint8_t* source, int32_t source_stride,
                                 const int16_t* weights, int32_t count, uint32_t precision) {
    int32_t value = int32_t{1} << (precision - 1);
    for (int32_t index = 0; index < count; ++index)
        value += static_cast<int32_t>(source[static_cast<std::size_t>(index) * source_stride]) *
                 weights[index];
    return static_cast<uint8_t>(std::clamp(value >> precision, 0, 255));
}

static bool valid_resize_dimensions(const unsigned char* raw, int32_t width, int32_t height,
                                    int32_t target_width, int32_t target_height) {
    return raw != nullptr && width > 0 && height > 0 && target_width > 0 && target_height > 0;
}

static std::vector<unsigned char> resize_bicubic_horizontal(const unsigned char* raw, int32_t width,
                                                            int32_t height, int32_t target_width) {
    const auto weights = make_bicubic_antialias_weights(width, target_width);
    std::vector<unsigned char> resized(static_cast<std::size_t>(height) * target_width * 3);
    parallel_for_ranges(height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            for (int32_t x = 0; x < target_width; ++x) {
                const int32_t start = weights.starts[static_cast<std::size_t>(x)];
                const int32_t count = weights.sizes[static_cast<std::size_t>(x)];
                const int16_t* coefficients =
                    weights.values.data() + static_cast<std::size_t>(x) * weights.stride;
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        (static_cast<std::size_t>(y) * width + start) * 3 + channel;
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * target_width + x) * 3 + channel;
                    resized[destination_offset] = fixed_point_pixel(
                        raw + source_offset, 3, coefficients, count, weights.precision);
                }
            }
        }
    });
    return resized;
}

static std::vector<unsigned char> resize_bicubic_vertical(const unsigned char* raw, int32_t width,
                                                          int32_t height, int32_t target_height) {
    const auto weights = make_bicubic_antialias_weights(height, target_height);
    std::vector<unsigned char> resized(static_cast<std::size_t>(target_height) * width * 3);
    const int32_t row_stride = width * 3;
    parallel_for_ranges(target_height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            const int32_t start = weights.starts[static_cast<std::size_t>(y)];
            const int32_t count = weights.sizes[static_cast<std::size_t>(y)];
            const int16_t* coefficients =
                weights.values.data() + static_cast<std::size_t>(y) * weights.stride;
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        (static_cast<std::size_t>(start) * width + x) * 3 + channel;
                    const auto destination_offset =
                        (static_cast<std::size_t>(y) * width + x) * 3 + channel;
                    resized[destination_offset] = fixed_point_pixel(
                        raw + source_offset, row_stride, coefficients, count, weights.precision);
                }
            }
        }
    });
    return resized;
}

std::vector<unsigned char> resize_rgb(const unsigned char* raw, int32_t width, int32_t height,
                                      int32_t target_width, int32_t target_height) {
    if (!valid_resize_dimensions(raw, width, height, target_width, target_height))
        return {};
    if (width == target_width && height == target_height) {
        return {raw, raw + static_cast<std::size_t>(width) * height * 3};
    }

    std::vector<unsigned char> horizontal;
    const unsigned char* vertical_source = raw;
    if (width != target_width) {
        horizontal = resize_bicubic_horizontal(raw, width, height, target_width);
        vertical_source = horizontal.data();
    }

    if (height == target_height)
        return horizontal;

    return resize_bicubic_vertical(vertical_source, target_width, height, target_height);
}
} // namespace

MediaRecord preprocess_media(const ITokenizer& tokenizer, const StructuredDecisionRequest& request,
                             const Json& document, const Json& processor) {
    MediaRecord result;
    if (request.images.empty() && request.videos.empty())
        return result;
    const auto overrides = document.value("media_kwargs", Json::object());
    if (!overrides.is_object())
        throw std::invalid_argument("media_kwargs must be an object");
    for (const auto& item : overrides.items()) {
        if (item.key() != "min_pixels" && item.key() != "max_pixels" && item.key() != "do_resize" &&
            item.key() != "do_sample_frames" && item.key() != "fps" && item.key() != "num_frames" &&
            item.key() != "video_metadata")
            throw std::invalid_argument("unsupported Clef media option: " + item.key());
    }
    std::string text;
    auto process = [&](const ImageResult& input, bool video, int video_index) {
        const auto& settings = processor.at(video ? "video_processor" : "image_processor");
        const int h = input.height, w = input.width, frames = video ? input.num_frames : 1;
        if (h <= 0 || w <= 0 || frames <= 0 || input.channels != 3 ||
            input.pixels.size() != static_cast<std::size_t>(h) * w * frames * 3)
            throw std::invalid_argument(
                "media requires contiguous RGB pixels and valid dimensions");
        if (std::max(h, w) / static_cast<double>(std::min(h, w)) > 200)
            throw std::invalid_argument("image aspect ratio exceeds 200");
        std::vector<std::uint8_t> pixels(input.pixels.size());
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const float value = input.pixels[i];
            if (!std::isfinite(value) || value < 0 || value > 255 || std::floor(value) != value)
                throw std::invalid_argument(
                    "Clef media pixels must be integer RGB values from 0 through 255");
            pixels[i] = static_cast<std::uint8_t>(value);
        }
        double source_fps = 24;
        if (video && overrides.contains("video_metadata")) {
            const auto& metadata = overrides["video_metadata"].at(video_index);
            if (metadata.contains("fps") && !metadata["fps"].is_null())
                source_fps = metadata["fps"].get<double>();
        }
        if (!std::isfinite(source_fps) || source_fps <= 0)
            throw std::invalid_argument("video fps must be positive");
        int sample_count = frames;
        if (video && overrides.value("do_sample_frames", true)) {
            if (overrides.contains("num_frames") && overrides.contains("fps"))
                throw std::invalid_argument("num_frames and fps are mutually exclusive");
            if (overrides.contains("num_frames"))
                sample_count = overrides["num_frames"].get<int>();
            else {
                const double target_fps = overrides.value("fps", settings.at("fps").get<double>());
                if (!std::isfinite(target_fps) || target_fps <= 0)
                    throw std::invalid_argument("sampling fps must be positive");
                sample_count =
                    std::min({std::max(static_cast<int>(frames / source_fps * target_fps),
                                       settings.at("min_frames").get<int>()),
                              settings.at("max_frames").get<int>(), frames});
            }
        }
        if (sample_count < 1)
            throw std::invalid_argument("video must have at least one sampled frame");
        std::vector<int> indices(sample_count);
        for (int i = 0; i < sample_count; ++i)
            indices[i] = sample_count == 1
                             ? 0
                             : static_cast<int>(std::nearbyint(static_cast<double>(i) *
                                                               (frames - 1) / (sample_count - 1)));
        const int min_pixels =
            overrides.value("min_pixels", settings.at("size").at("shortest_edge").get<int>());
        const int max_pixels =
            overrides.value("max_pixels", settings.at("size").at("longest_edge").get<int>());
        if (min_pixels < 1 || max_pixels < min_pixels)
            throw std::invalid_argument("invalid media size limits");
        const int temporal = video ? ((sample_count + 1) / 2) * 2 : 1;
        int target_h = h, target_w = w;
        if (overrides.value("do_resize", true)) {
            if (video && (h < 32 || w < 32))
                throw std::invalid_argument("video dimensions must be at least 32");
            target_h = std::max(32, static_cast<int>(std::nearbyint(h / 32.0)) * 32);
            target_w = std::max(32, static_cast<int>(std::nearbyint(w / 32.0)) * 32);
            const double aligned = static_cast<double>(temporal) * target_h * target_w;
            const double original = static_cast<double>(video ? sample_count : 1) * h * w;
            if (aligned > max_pixels) {
                const double beta = std::sqrt(original / max_pixels);
                target_h = std::max(32, static_cast<int>(std::floor(h / beta / 32)) * 32);
                target_w = std::max(32, static_cast<int>(std::floor(w / beta / 32)) * 32);
            } else if (aligned < min_pixels) {
                const double beta = std::sqrt(min_pixels / original);
                target_h = static_cast<int>(std::ceil(h * beta / 32)) * 32;
                target_w = static_cast<int>(std::ceil(w * beta / 32)) * 32;
            }
        }
        if (target_h % 32 || target_w % 32)
            throw std::invalid_argument("media dimensions must align to the 32-pixel merge grid");
        std::vector<std::vector<std::uint8_t>> resized;
        for (const int index : indices)
            resized.push_back(
                resize_rgb(pixels.data() + static_cast<std::size_t>(index) * h * w * 3, w, h,
                           target_w, target_h));
        if (!video || resized.size() % 2) {
            resized.push_back(resized.back());
            indices.push_back(indices.back());
        }
        // The release's record encoder wraps the video placeholder once; the
        // processor replaces its inner token with timestamped frame wrappers.
        if (video)
            text += "<|vision_start|>";
        for (std::size_t ti = 0; ti < resized.size(); ti += 2) {
            VisionFrame frame;
            frame.grid_height = target_h / 16;
            frame.grid_width = target_w / 16;
            frame.video = video;
            frame.timestamp = (indices[ti] / source_fps + indices[ti + 1] / source_fps) / 2;
            frame.patches.reserve(static_cast<std::size_t>(frame.grid_height) * frame.grid_width *
                                  1536);
            for (int by = 0; by < frame.grid_height; by += 2)
                for (int bx = 0; bx < frame.grid_width; bx += 2)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            for (int c = 0; c < 3; ++c)
                                for (int t = 0; t < 2; ++t)
                                    for (int py = 0; py < 16; ++py)
                                        for (int px = 0; px < 16; ++px) {
                                            const auto pixel =
                                                resized[ti + t]
                                                       [((by + dy) * 16 + py) * target_w * 3 +
                                                        ((bx + dx) * 16 + px) * 3 + c];
                                            frame.patches.push_back(
                                                (static_cast<float>(pixel) - 127.5F) / 127.5F);
                                        }
            if (video) {
                std::ostringstream timestamp;
                timestamp << '<' << std::fixed << std::setprecision(1) << frame.timestamp
                          << " seconds>";
                text += timestamp.str();
            }
            text += "<|vision_start|>";
            for (int token = 0; token < frame.grid_height * frame.grid_width / 4; ++token)
                text += video ? "<|video_pad|>" : "<|image_pad|>";
            text += "<|vision_end|>";
            result.frames.push_back(std::move(frame));
        }
        if (video)
            text += "<|vision_end|>";
    };
    for (const auto& image : request.images)
        process(image, false, 0);
    for (std::size_t i = 0; i < request.videos.size(); ++i)
        process(request.videos[i], true, i);
    result.tokens = tokenizer.encode(text + "\n");
    return result;
}

std::vector<std::array<int, 3>> media_positions(const Record& record, const MediaRecord& media,
                                                int image_token, int video_token) {
    std::vector<std::array<int, 3>> positions;
    int current = 0;
    std::size_t frame = 0;
    for (std::size_t i = 0; i < record.input_ids.size();) {
        if (record.input_ids[i] != image_token && record.input_ids[i] != video_token) {
            positions.push_back({current, current, current});
            ++current;
            ++i;
            continue;
        }
        if (frame >= media.frames.size())
            throw std::invalid_argument("media tokens lack matching image or video data");
        const auto& data = media.frames[frame++];
        const int height = data.grid_height / 2, width = data.grid_width / 2;
        const int token = data.video ? video_token : image_token;
        for (int h = 0; h < height; ++h)
            for (int w = 0; w < width; ++w) {
                if (i >= record.input_ids.size() || record.input_ids[i++] != token)
                    throw std::invalid_argument("media token count does not match the vision grid");
                positions.push_back({current, current + h, current + w});
            }
        current += std::max(height, width);
    }
    if (frame != media.frames.size())
        throw std::invalid_argument("media data lacks matching input tokens");
    return positions;
}

void vision_positions(const VisionFrame& frame, const std::vector<char>& embedding, int width,
                      int heads, int side, std::vector<float>& positions, std::vector<float>& cos,
                      std::vector<float>& sin) {
    const int h = frame.grid_height, w = frame.grid_width, dim = width / heads;
    if (embedding.size() != static_cast<std::size_t>(side) * side * width * 2 || dim % 4)
        throw std::invalid_argument("invalid visual position embedding geometry");
    auto scalar = [&](int position, int column) {
        std::uint16_t bf16;
        std::memcpy(
            &bf16, embedding.data() + (static_cast<std::size_t>(position) * width + column) * 2, 2);
        std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    };
    positions.resize(static_cast<std::size_t>(h) * w * width);
    cos.resize(static_cast<std::size_t>(h) * w * dim);
    sin.resize(cos.size());
    auto linspace = [side](int i, int count) {
        const float step = static_cast<float>(side - 1) / (count - 1);
        return i < count / 2 ? i * step : (side - 1) - (count - i - 1) * step;
    };
    int token = 0;
    for (int by = 0; by < h; by += 2)
        for (int bx = 0; bx < w; bx += 2)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx, ++token) {
                    const int row = by + dy, col = bx + dx;
                    const float y = linspace(row, h), x = linspace(col, w);
                    const int y0 = y, x0 = x, y1 = std::min(y0 + 1, side - 1),
                              x1 = std::min(x0 + 1, side - 1);
                    const float fy = y - y0, fx = x - x0;
                    const std::array<int, 4> indices = {y0 * side + x0, y0 * side + x1,
                                                        y1 * side + x0, y1 * side + x1};
                    const std::array<float, 4> weights = {(1 - fy) * (1 - fx), (1 - fy) * fx,
                                                          fy * (1 - fx), fy * fx};
                    for (int c = 0; c < width; ++c) {
                        float sum = 0;
                        for (int corner = 0; corner < 4; ++corner)
                            sum += scalar(indices[corner], c) * weights[corner];
                        positions[static_cast<std::size_t>(token) * width + c] = sum;
                    }
                    for (int i = 0; i < dim / 2; ++i) {
                        const int coordinate = i < dim / 4 ? row : col;
                        const float frequency =
                            1.0F /
                            std::pow(10000.0F, static_cast<float>(2 * (i % (dim / 4))) / (dim / 2));
                        const float angle = coordinate * frequency;
                        cos[token * dim + i] = cos[token * dim + i + dim / 2] = std::cos(angle);
                        sin[token * dim + i] = sin[token * dim + i + dim / 2] = std::sin(angle);
                    }
                }
}
} // namespace trtmc::clef
