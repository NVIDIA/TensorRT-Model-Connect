/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/sam2_hoi_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace trtmc::sam2_hoi {
namespace {

constexpr int32_t kPillowPrecisionBits = 22;
constexpr int64_t kPillowRounding = int64_t{1} << (kPillowPrecisionBits - 1);
constexpr int32_t kPillowCoefficientScale = int32_t{1} << kPillowPrecisionBits;
constexpr std::array<float, 3> kImageNetMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kImageNetStd{0.229F, 0.224F, 0.225F};

struct CubicSpan {
    int32_t first{0};
    std::vector<int32_t> weights;
};

struct BilinearSpan {
    int32_t lower{0};
    int32_t upper{0};
    float upper_weight{0.0F};
};

template <typename Function>
void parallel_for_ranges(int32_t count, int32_t minimum_grain, Function&& function) {
    if (count <= 0) {
        return;
    }
    constexpr int32_t kMaximumWorkers = 8;
    const auto hardware_workers = static_cast<int32_t>(std::thread::hardware_concurrency());
    const int32_t worker_limit =
        hardware_workers > 0 ? std::min(kMaximumWorkers, hardware_workers) : kMaximumWorkers;
    const int32_t grain_workers = (count - 1) / minimum_grain + 1;
    const int32_t workers = std::max(1, std::min({count, grain_workers, worker_limit}));
    if (workers == 1) {
        function(0, count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int32_t worker = 0; worker < workers - 1; ++worker) {
        const int32_t begin = static_cast<int32_t>(static_cast<int64_t>(count) * worker / workers);
        const int32_t end =
            static_cast<int32_t>(static_cast<int64_t>(count) * (worker + 1) / workers);
        threads.emplace_back([&, begin, end] { function(begin, end); });
    }
    function(static_cast<int32_t>(static_cast<int64_t>(count) * (workers - 1) / workers), count);
    for (auto& thread : threads) {
        thread.join();
    }
}

std::size_t checked_plane_size(int32_t height, int32_t width, const char* contract) {
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument(std::string(contract) + " dimensions must be positive");
    }
    const auto height_size = static_cast<std::size_t>(height);
    const auto width_size = static_cast<std::size_t>(width);
    if (height_size > std::numeric_limits<std::size_t>::max() / width_size) {
        throw std::overflow_error(std::string(contract) + " dimensions overflow size_t");
    }
    return height_size * width_size;
}

std::size_t checked_tensor_size(int32_t count, std::size_t plane_size, const char* contract) {
    if (count < 0) {
        throw std::invalid_argument(std::string(contract) + " count must be non-negative");
    }
    const auto count_size = static_cast<std::size_t>(count);
    if (count_size > std::numeric_limits<std::size_t>::max() / plane_size) {
        throw std::overflow_error(std::string(contract) + " shape overflows size_t");
    }
    return count_size * plane_size;
}

double pillow_cubic_filter(double value) {
    constexpr double coefficient = -0.5;
    value = std::abs(value);
    if (value < 1.0) {
        return ((coefficient + 2.0) * value - (coefficient + 3.0)) * value * value + 1.0;
    }
    if (value < 2.0) {
        return (((coefficient * value - 5.0 * coefficient) * value + 8.0 * coefficient) * value -
                4.0 * coefficient);
    }
    return 0.0;
}

std::vector<CubicSpan> make_pillow_cubic_spans(int32_t source_size, int32_t target_size) {
    const double scale = static_cast<double>(source_size) / target_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<CubicSpan> spans(static_cast<std::size_t>(target_size));

    for (int32_t target_index = 0; target_index < target_size; ++target_index) {
        const double center = (static_cast<double>(target_index) + 0.5) * scale;
        const int32_t first = std::max(0, static_cast<int32_t>(center - support + 0.5));
        const int32_t end = std::min(source_size, static_cast<int32_t>(center + support + 0.5));
        auto& span = spans[static_cast<std::size_t>(target_index)];
        span.first = first;
        span.weights.resize(static_cast<std::size_t>(end - first));

        std::vector<double> floating_weights(span.weights.size());
        double weight_sum = 0.0;
        for (int32_t source_index = first; source_index < end; ++source_index) {
            const double distance =
                (static_cast<double>(source_index) - center + 0.5) * inverse_filter_scale;
            const double weight = pillow_cubic_filter(distance);
            floating_weights[static_cast<std::size_t>(source_index - first)] = weight;
            weight_sum += weight;
        }
        if (weight_sum == 0.0) {
            throw std::runtime_error("Pillow bicubic resize produced an empty filter span");
        }
        for (std::size_t index = 0; index < span.weights.size(); ++index) {
            // Pillow's 8-bit resampler rounds both signs away from zero before
            // the integer cast. The distinction matters for cubic's negative
            // lobes and can change the final uint8 result by one.
            const double normalized = floating_weights[index] / weight_sum;
            span.weights[index] = static_cast<int32_t>((normalized < 0.0 ? -0.5 : 0.5) +
                                                       normalized * kPillowCoefficientScale);
        }
    }
    return spans;
}

uint8_t apply_pillow_cubic_span(const uint8_t* input, std::size_t stride, const CubicSpan& span) {
    int64_t sum = kPillowRounding;
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        const auto source_index = span.first + static_cast<int32_t>(index);
        sum += static_cast<int64_t>(input[static_cast<std::size_t>(source_index) * stride]) *
               span.weights[index];
    }
    return static_cast<uint8_t>(std::clamp<int64_t>(sum >> kPillowPrecisionBits, 0, 255));
}

std::vector<uint8_t> resize_pillow_cubic_horizontal(const uint8_t* rgb_hwc, int32_t source_height,
                                                    int32_t source_width, int32_t target_width) {
    const auto spans = make_pillow_cubic_spans(source_width, target_width);
    const auto target_plane = checked_plane_size(source_height, target_width, "resize target");
    std::vector<uint8_t> resized(checked_tensor_size(3, target_plane, "resize target"));
    parallel_for_ranges(source_height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            for (int32_t x = 0; x < target_width; ++x) {
                const auto& span = spans[static_cast<std::size_t>(x)];
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset =
                        static_cast<std::size_t>(y) * source_width * 3U + channel;
                    const auto target_offset =
                        (static_cast<std::size_t>(y) * target_width + x) * 3U + channel;
                    resized[target_offset] =
                        apply_pillow_cubic_span(rgb_hwc + source_offset, 3, span);
                }
            }
        }
    });
    return resized;
}

std::vector<uint8_t> resize_pillow_cubic_vertical(const uint8_t* rgb_hwc, int32_t source_height,
                                                  int32_t width, int32_t target_height) {
    const auto spans = make_pillow_cubic_spans(source_height, target_height);
    const auto target_plane = checked_plane_size(target_height, width, "resize target");
    std::vector<uint8_t> resized(checked_tensor_size(3, target_plane, "resize target"));
    const auto source_stride = static_cast<std::size_t>(width) * 3U;
    parallel_for_ranges(target_height, 16, [&](int32_t begin_y, int32_t end_y) {
        for (int32_t y = begin_y; y < end_y; ++y) {
            const auto& span = spans[static_cast<std::size_t>(y)];
            for (int32_t x = 0; x < width; ++x) {
                for (int32_t channel = 0; channel < 3; ++channel) {
                    const auto source_offset = static_cast<std::size_t>(x) * 3U + channel;
                    const auto target_offset =
                        (static_cast<std::size_t>(y) * width + x) * 3U + channel;
                    resized[target_offset] =
                        apply_pillow_cubic_span(rgb_hwc + source_offset, source_stride, span);
                }
            }
        }
    });
    return resized;
}

std::vector<BilinearSpan> make_align_corners_false_spans(int32_t source_size, int32_t target_size) {
    std::vector<BilinearSpan> spans(static_cast<std::size_t>(target_size));
    const float scale = static_cast<float>(source_size) / static_cast<float>(target_size);
    for (int32_t target_index = 0; target_index < target_size; ++target_index) {
        const float source_index = (static_cast<float>(target_index) + 0.5F) * scale - 0.5F;
        const int32_t lower_unclamped = static_cast<int32_t>(std::floor(source_index));
        spans[static_cast<std::size_t>(target_index)] = {
            std::clamp(lower_unclamped, 0, source_size - 1),
            std::clamp(lower_unclamped + 1, 0, source_size - 1),
            source_index - static_cast<float>(lower_unclamped),
        };
    }
    return spans;
}

void append_background_neighbors(const float* mask, int32_t height, int32_t width,
                                 std::size_t current, std::vector<uint8_t>& visited,
                                 std::vector<std::size_t>& component) {
    const int32_t current_y = static_cast<int32_t>(current / static_cast<std::size_t>(width));
    const int32_t current_x = static_cast<int32_t>(current % static_cast<std::size_t>(width));
    const int32_t minimum_y = std::max(0, current_y - 1);
    const int32_t maximum_y = std::min(height - 1, current_y + 1);
    const int32_t minimum_x = std::max(0, current_x - 1);
    const int32_t maximum_x = std::min(width - 1, current_x + 1);
    for (int32_t y = minimum_y; y <= maximum_y; ++y) {
        for (int32_t x = minimum_x; x <= maximum_x; ++x) {
            const auto neighbor = static_cast<std::size_t>(y) * width + x;
            if (visited[neighbor] != 0U || !(mask[neighbor] <= 0.0F)) {
                continue;
            }
            visited[neighbor] = 1U;
            component.push_back(neighbor);
        }
    }
}

void fill_small_holes_in_mask(float* mask, std::size_t plane_size, int32_t height, int32_t width,
                              std::size_t max_area, float fill_value, std::vector<uint8_t>& visited,
                              std::vector<std::size_t>& component) {
    std::fill(visited.begin(), visited.end(), uint8_t{0});
    for (std::size_t pixel = 0; pixel < plane_size; ++pixel) {
        if (visited[pixel] != 0U || !(mask[pixel] <= 0.0F))
            continue;

        component.clear();
        component.push_back(pixel);
        visited[pixel] = 1U;
        for (std::size_t head = 0; head < component.size(); ++head) {
            append_background_neighbors(mask, height, width, component[head], visited, component);
        }
        if (component.size() > max_area)
            continue;
        for (const auto component_pixel : component)
            mask[component_pixel] = fill_value;
    }
}

bool set_error(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

} // namespace

std::vector<uint8_t> resize_pillow_bicubic_rgb_u8(const uint8_t* rgb_hwc, int32_t source_height,
                                                  int32_t source_width, int32_t target_height,
                                                  int32_t target_width) {
    const auto source_plane = checked_plane_size(source_height, source_width, "resize source");
    (void)checked_plane_size(target_height, target_width, "resize target");
    if (rgb_hwc == nullptr) {
        throw std::invalid_argument("Pillow bicubic resize source must be non-null");
    }
    const auto source_size = checked_tensor_size(3, source_plane, "resize source");
    if (source_height == target_height && source_width == target_width) {
        return {rgb_hwc, rgb_hwc + source_size};
    }

    std::vector<uint8_t> horizontal;
    const uint8_t* vertical_source = rgb_hwc;
    if (source_width != target_width) {
        horizontal =
            resize_pillow_cubic_horizontal(rgb_hwc, source_height, source_width, target_width);
        vertical_source = horizontal.data();
    }
    if (source_height == target_height) {
        return horizontal;
    }
    return resize_pillow_cubic_vertical(vertical_source, source_height, target_width,
                                        target_height);
}

std::vector<float> preprocess_image(const float* rgb_hwc, int32_t source_height,
                                    int32_t source_width) {
    const auto source_plane = checked_plane_size(source_height, source_width, "image source");
    if (rgb_hwc == nullptr) {
        throw std::invalid_argument("SAM2 HOI image source must be non-null");
    }
    const auto source_size = checked_tensor_size(3, source_plane, "image source");
    std::vector<uint8_t> source_u8(source_size);
    for (std::size_t index = 0; index < source_size; ++index) {
        const float value = rgb_hwc[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::invalid_argument("SAM2 HOI RGB source values must be finite and in [0, 1]");
        }
        source_u8[index] = detail::round_unit_float_to_u8(value);
    }

    const auto resized = resize_pillow_bicubic_rgb_u8(source_u8.data(), source_height, source_width,
                                                      kImageSize, kImageSize);
    const auto target_plane = static_cast<std::size_t>(kImageSize) * kImageSize;
    std::vector<float> pixel_values(3U * target_plane);
    parallel_for_ranges(3 * kImageSize, 32, [&](int32_t begin_row, int32_t end_row) {
        for (int32_t row = begin_row; row < end_row; ++row) {
            const auto channel = static_cast<std::size_t>(row / kImageSize);
            const int32_t y = row % kImageSize;
            for (int32_t x = 0; x < kImageSize; ++x) {
                const auto source_index =
                    (static_cast<std::size_t>(y) * kImageSize + x) * 3U + channel;
                const auto target_index = static_cast<std::size_t>(y) * kImageSize + x;
                const float rescaled = static_cast<float>(resized[source_index]) / 255.0F;
                pixel_values[channel * target_plane + target_index] =
                    (rescaled - kImageNetMean[channel]) / kImageNetStd[channel];
            }
        }
    });
    return pixel_values;
}

void fill_small_mask_holes(std::vector<float>& mask_logits, int32_t mask_count, int32_t height,
                           int32_t width, int32_t max_area, float fill_value) {
    const auto plane_size = checked_plane_size(height, width, "mask");
    const auto expected_size = checked_tensor_size(mask_count, plane_size, "mask");
    if (mask_logits.size() != expected_size) {
        throw std::invalid_argument("SAM2 HOI mask logits do not match the declared shape");
    }
    if (mask_count == 0 || max_area <= 0) {
        return;
    }
    if (!std::isfinite(fill_value) || !(fill_value > 0.0F)) {
        throw std::invalid_argument("SAM2 HOI hole fill value must be finite and positive");
    }

    std::vector<uint8_t> visited(plane_size);
    std::vector<std::size_t> component;
    for (int32_t mask_index = 0; mask_index < mask_count; ++mask_index) {
        float* mask = mask_logits.data() + static_cast<std::size_t>(mask_index) * plane_size;
        fill_small_holes_in_mask(mask, plane_size, height, width,
                                 static_cast<std::size_t>(max_area), fill_value, visited,
                                 component);
    }
}

std::vector<uint8_t> resize_and_threshold_masks(const float* low_res_logits, int32_t mask_count,
                                                int32_t source_height, int32_t source_width,
                                                int32_t target_height, int32_t target_width,
                                                float threshold) {
    const auto source_plane = checked_plane_size(source_height, source_width, "mask source");
    const auto target_plane = checked_plane_size(target_height, target_width, "mask target");
    (void)checked_tensor_size(mask_count, source_plane, "mask source");
    const auto target_size = checked_tensor_size(mask_count, target_plane, "mask target");
    if (mask_count == 0) {
        return {};
    }
    if (low_res_logits == nullptr) {
        throw std::invalid_argument("SAM2 HOI low-resolution masks must be non-null");
    }
    if (!std::isfinite(threshold)) {
        throw std::invalid_argument("SAM2 HOI mask threshold must be finite");
    }

    const auto x_spans = make_align_corners_false_spans(source_width, target_width);
    const auto y_spans = make_align_corners_false_spans(source_height, target_height);
    std::vector<float> horizontal(static_cast<std::size_t>(source_height) * target_width);
    std::vector<uint8_t> binary_masks(target_size);
    for (int32_t mask_index = 0; mask_index < mask_count; ++mask_index) {
        const float* source = low_res_logits + static_cast<std::size_t>(mask_index) * source_plane;
        parallel_for_ranges(source_height, 32, [&](int32_t begin_y, int32_t end_y) {
            for (int32_t y = begin_y; y < end_y; ++y) {
                const auto source_row = static_cast<std::size_t>(y) * source_width;
                const auto target_row = static_cast<std::size_t>(y) * target_width;
                for (int32_t x = 0; x < target_width; ++x) {
                    const auto& span = x_spans[static_cast<std::size_t>(x)];
                    horizontal[target_row + x] =
                        source[source_row + static_cast<std::size_t>(span.lower)] *
                            (1.0F - span.upper_weight) +
                        source[source_row + static_cast<std::size_t>(span.upper)] *
                            span.upper_weight;
                }
            }
        });

        uint8_t* target = binary_masks.data() + static_cast<std::size_t>(mask_index) * target_plane;
        parallel_for_ranges(target_height, 32, [&](int32_t begin_y, int32_t end_y) {
            for (int32_t y = begin_y; y < end_y; ++y) {
                const auto& span = y_spans[static_cast<std::size_t>(y)];
                const auto lower_row = static_cast<std::size_t>(span.lower) * target_width;
                const auto upper_row = static_cast<std::size_t>(span.upper) * target_width;
                const auto target_row = static_cast<std::size_t>(y) * target_width;
                for (int32_t x = 0; x < target_width; ++x) {
                    const float value = horizontal[lower_row + x] * (1.0F - span.upper_weight) +
                                        horizontal[upper_row + x] * span.upper_weight;
                    target[target_row + x] = static_cast<uint8_t>(value > threshold);
                }
            }
        });
    }
    return binary_masks;
}

bool write_uint8_npy(const std::string& path, const std::vector<uint8_t>& masks, int32_t mask_count,
                     int32_t height, int32_t width, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    std::size_t expected_size = 0;
    try {
        const auto plane_size = checked_plane_size(height, width, "NumPy mask");
        expected_size = checked_tensor_size(mask_count, plane_size, "NumPy mask");
    } catch (const std::exception& exception) {
        return set_error(error, exception.what());
    }
    if (masks.size() != expected_size) {
        return set_error(error, "NumPy mask payload does not match shape");
    }
    if (path.empty()) {
        return set_error(error, "NumPy output path must be non-empty");
    }

    std::ostringstream dictionary;
    dictionary << "{'descr': '|u1', 'fortran_order': False, 'shape': (" << mask_count << ", 1, "
               << height << ", " << width << "), }";
    std::string header = dictionary.str();
    constexpr std::size_t kPreambleSize = 10;
    constexpr std::size_t kHeaderAlignment = 64;
    const auto unpadded_size = kPreambleSize + header.size() + 1U;
    const auto padding = (kHeaderAlignment - unpadded_size % kHeaderAlignment) % kHeaderAlignment;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        return set_error(error, "NumPy v1.0 header is too large");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return set_error(error, "could not open NumPy output: " + path);
    }
    constexpr std::array<char, 8> kMagic{static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    const auto header_size = static_cast<uint16_t>(header.size());
    const std::array<char, 2> little_endian_header_size{
        static_cast<char>(header_size & 0xFFU),
        static_cast<char>((header_size >> 8U) & 0xFFU),
    };
    output.write(little_endian_header_size.data(),
                 static_cast<std::streamsize>(little_endian_header_size.size()));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!masks.empty()) {
        output.write(reinterpret_cast<const char*>(masks.data()),
                     static_cast<std::streamsize>(masks.size()));
    }
    if (!output) {
        return set_error(error, "failed while writing NumPy output: " + path);
    }
    return true;
}

} // namespace trtmc::sam2_hoi
