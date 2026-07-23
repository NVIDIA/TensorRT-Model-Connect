/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/openpi_data_plane.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::openpi {
namespace {

constexpr float kTransformEpsilon = 1.0e-6F;
// Pinned OpenPI JAX/XLA lowers `uint8 / 255 * 2 - 1` to this reciprocal
// scale (bits 0x3c008080), one ULP below the correctly rounded 2/255 value.
// Keeping the lowered constant explicit avoids compiler-dependent reassociation
// at the native preprocessing boundary.
constexpr float kUint8ToOpenPiScale = 0x1.0101p-7F;

std::size_t checked_element_count(int32_t height, int32_t width, int32_t channels) {
    if (height <= 0 || width <= 0 || channels <= 0) {
        throw std::invalid_argument("OpenPI image dimensions and channel count must be positive");
    }
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    const auto c = static_cast<std::size_t>(channels);
    if (h > std::numeric_limits<std::size_t>::max() / w ||
        h * w > std::numeric_limits<std::size_t>::max() / c) {
        throw std::overflow_error("OpenPI image element count overflow");
    }
    return h * w * c;
}

std::size_t checked_flow_count(int32_t batch_size, int32_t action_horizon,
                               int32_t action_dimension) {
    if (batch_size <= 0 || action_horizon <= 0 || action_dimension <= 0) {
        throw std::invalid_argument("OpenPI flow dimensions must be positive");
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    const auto horizon = static_cast<std::size_t>(action_horizon);
    const auto dimension = static_cast<std::size_t>(action_dimension);
    if (batch > std::numeric_limits<std::size_t>::max() / horizon ||
        batch * horizon > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::overflow_error("OpenPI flow element count overflow");
    }
    return batch * horizon * dimension;
}

void validate_quantile_stats(const QuantileStats& stats) {
    if (stats.q01.empty() || stats.q01.size() != stats.q99.size()) {
        throw std::invalid_argument("OpenPI q01/q99 statistics must have equal non-zero size");
    }
    for (std::size_t i = 0; i < stats.q01.size(); ++i) {
        if (!std::isfinite(stats.q01[i]) || !std::isfinite(stats.q99[i])) {
            throw std::invalid_argument("OpenPI quantile statistics must be finite");
        }
    }
}

struct Utf8Codepoint {
    char32_t value{0};
    std::size_t offset{0};
    std::size_t length{0};
};

struct Utf8Prefix {
    std::size_t length{0};
    char32_t value{0};
    char32_t minimum{0};
};

Utf8Prefix decode_utf8_prefix(uint8_t first) {
    if (first <= 0x7FU) {
        return Utf8Prefix{1, first, 0};
    }

    if ((first & 0xE0U) == 0xC0U) {
        return Utf8Prefix{2, first & 0x1FU, 0x80};
    }
    if ((first & 0xF0U) == 0xE0U) {
        return Utf8Prefix{3, first & 0x0FU, 0x800};
    }
    if ((first & 0xF8U) == 0xF0U) {
        return Utf8Prefix{4, first & 0x07U, 0x10000};
    }
    throw std::invalid_argument("OpenPI prompt contains invalid UTF-8");
}

char32_t decode_utf8_continuations(std::string_view text, std::size_t offset,
                                   const Utf8Prefix& prefix) {
    char32_t value = prefix.value;
    for (std::size_t i = 1; i < prefix.length; ++i) {
        const auto byte = static_cast<uint8_t>(text[offset + i]);
        if ((byte & 0xC0U) != 0x80U) {
            throw std::invalid_argument("OpenPI prompt contains invalid UTF-8 continuation byte");
        }
        value = static_cast<char32_t>((value << 6U) | (byte & 0x3FU));
    }
    return value;
}

bool is_invalid_utf8_scalar(char32_t value, char32_t minimum) {
    if (value < minimum || value > 0x10FFFFU) {
        return true;
    }
    return value >= 0xD800U && value <= 0xDFFFU;
}

Utf8Codepoint decode_utf8(std::string_view text, std::size_t offset) {
    if (offset >= text.size()) {
        throw std::invalid_argument("OpenPI prompt UTF-8 offset is out of range");
    }

    const Utf8Prefix prefix = decode_utf8_prefix(static_cast<uint8_t>(text[offset]));
    if (offset + prefix.length > text.size()) {
        throw std::invalid_argument("OpenPI prompt contains truncated UTF-8");
    }
    const char32_t value = decode_utf8_continuations(text, offset, prefix);
    if (is_invalid_utf8_scalar(value, prefix.minimum)) {
        throw std::invalid_argument("OpenPI prompt contains a non-canonical UTF-8 codepoint");
    }
    return Utf8Codepoint{value, offset, prefix.length};
}

struct CodepointRange {
    char32_t first;
    char32_t last;
};

bool is_python_whitespace(char32_t value) {
    // CPython's Unicode whitespace set (Py_UNICODE_ISSPACE), used by str.strip().
    constexpr std::array<CodepointRange, 10> kWhitespaceRanges = {
        CodepointRange{0x0009U, 0x000DU}, CodepointRange{0x001CU, 0x0020U},
        CodepointRange{0x0085U, 0x0085U}, CodepointRange{0x00A0U, 0x00A0U},
        CodepointRange{0x1680U, 0x1680U}, CodepointRange{0x2000U, 0x200AU},
        CodepointRange{0x2028U, 0x2029U}, CodepointRange{0x202FU, 0x202FU},
        CodepointRange{0x205FU, 0x205FU}, CodepointRange{0x3000U, 0x3000U},
    };
    for (const auto& range : kWhitespaceRanges) {
        if (value >= range.first && value <= range.last) {
            return true;
        }
    }
    return false;
}

void validate_uint8_camera(const CameraFrameView& camera) {
    if (camera.uint8_data == nullptr || camera.float_data != nullptr) {
        throw std::invalid_argument("OpenPI uint8 camera must provide only uint8 pixel data");
    }
    if (camera.valid) {
        return;
    }
    if (std::any_of(camera.uint8_data, camera.uint8_data + camera.element_count,
                    [](uint8_t value) { return value != 0U; })) {
        throw std::invalid_argument("OpenPI masked uint8 camera slot must be black (zero)");
    }
}

void validate_float_camera_pixel(float value, bool valid) {
    if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
        throw std::invalid_argument("OpenPI float camera pixels must be finite and in [-1, 1]");
    }
    if (!valid && value != -1.0F) {
        throw std::invalid_argument("OpenPI masked float camera slot must be black (-1)");
    }
}

void validate_float_camera(const CameraFrameView& camera) {
    if (camera.float_data == nullptr || camera.uint8_data != nullptr) {
        throw std::invalid_argument("OpenPI float camera must provide only float32 pixel data");
    }
    for (std::size_t i = 0; i < camera.element_count; ++i) {
        validate_float_camera_pixel(camera.float_data[i], camera.valid);
    }
}

std::vector<Utf8Codepoint> decode_prompt(std::string_view prompt) {
    std::vector<Utf8Codepoint> result;
    for (std::size_t offset = 0; offset < prompt.size();) {
        auto codepoint = decode_utf8(prompt, offset);
        offset += codepoint.length;
        result.push_back(codepoint);
    }
    return result;
}

std::size_t camera_index(std::string_view name) {
    for (std::size_t i = 0; i < kCameraNames.size(); ++i) {
        if (name == kCameraNames[i]) {
            return i;
        }
    }
    throw std::invalid_argument("Unknown OpenPI camera slot: " + std::string(name));
}

void validate_camera(const CameraFrameView& camera) {
    const auto expected_count = checked_element_count(camera.height, camera.width, camera.channels);
    if (camera.channels != 3) {
        throw std::invalid_argument("OpenPI camera slots must contain HWC RGB pixels");
    }
    if (camera.element_count != expected_count) {
        throw std::invalid_argument("OpenPI camera pixel count does not match its geometry");
    }

    if (camera.pixel_type == CameraPixelType::kUint8) {
        validate_uint8_camera(camera);
        return;
    }

    if (camera.pixel_type != CameraPixelType::kFloat32) {
        throw std::invalid_argument("OpenPI float camera must provide only float32 pixel data");
    }
    validate_float_camera(camera);
}

struct AxisWeight {
    int32_t source_index{0};
    float weight{0.0F};
};

using AxisWeights = std::vector<std::vector<AxisWeight>>;

AxisWeights make_identity_axis_weights(int32_t size) {
    AxisWeights result(static_cast<std::size_t>(size));
    for (int32_t i = 0; i < size; ++i) {
        result[static_cast<std::size_t>(i)].push_back({i, 1.0F});
    }
    return result;
}

bool jax_sample_has_no_support(float total, float sample, int32_t input_size) {
    constexpr float minimum_total = 1000.0F * std::numeric_limits<float>::epsilon();
    return std::fabs(total) <= minimum_total || sample < -0.5F ||
           sample > static_cast<float>(input_size) - 0.5F;
}

std::vector<AxisWeight> make_jax_sample_weights(int32_t input_size, float sample,
                                                float kernel_scale) {
    std::vector<AxisWeight> weights;
    float total = 0.0F;
    for (int32_t input = 0; input < input_size; ++input) {
        const float distance = std::fabs(sample - static_cast<float>(input)) / kernel_scale;
        const float weight = std::max(0.0F, 1.0F - std::fabs(distance));
        if (weight != 0.0F) {
            weights.push_back({input, weight});
            total += weight;
        }
    }
    if (jax_sample_has_no_support(total, sample, input_size)) {
        return {};
    }
    for (auto& weight : weights) {
        weight.weight /= total;
    }
    return weights;
}

AxisWeights make_jax_linear_weights(int32_t input_size, int32_t output_size) {
    if (input_size <= 0 || output_size < 0) {
        throw std::invalid_argument("OpenPI resize axis dimensions are invalid");
    }
    AxisWeights result(static_cast<std::size_t>(output_size));
    if (output_size == 0) {
        return result;
    }
    if (input_size == output_size) {
        return make_identity_axis_weights(output_size);
    }

    // This follows jax._src.image.scale.compute_weight_mat in JAX 0.5.3.
    const float scale = static_cast<float>(output_size) / static_cast<float>(input_size);
    const float inverse_scale = 1.0F / scale;
    const float kernel_scale = std::max(inverse_scale, 1.0F);

    for (int32_t output = 0; output < output_size; ++output) {
        const float sample = (static_cast<float>(output) + 0.5F) * inverse_scale - 0.5F;
        result[static_cast<std::size_t>(output)] =
            make_jax_sample_weights(input_size, sample, kernel_scale);
    }
    return result;
}

std::vector<float> resize_jax_horizontal(const float* pixels, int32_t source_height,
                                         int32_t source_width, int32_t channels,
                                         int32_t target_width,
                                         const AxisWeights& horizontal_weights) {
    const auto temporary_count = checked_element_count(source_height, target_width, channels);
    std::vector<float> temporary(temporary_count, 0.0F);
    for (int32_t y = 0; y < source_height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            for (int32_t channel = 0; channel < channels; ++channel) {
                float value = 0.0F;
                for (const auto& weight : horizontal_weights[static_cast<std::size_t>(x)]) {
                    const auto source_index =
                        (static_cast<std::size_t>(y) * source_width + weight.source_index) *
                            static_cast<std::size_t>(channels) +
                        static_cast<std::size_t>(channel);
                    value += pixels[source_index] * weight.weight;
                }
                const auto destination_index = (static_cast<std::size_t>(y) * target_width + x) *
                                                   static_cast<std::size_t>(channels) +
                                               static_cast<std::size_t>(channel);
                temporary[destination_index] = value;
            }
        }
    }
    return temporary;
}

std::vector<float> resize_jax_vertical(const std::vector<float>& temporary, int32_t target_height,
                                       int32_t target_width, int32_t channels,
                                       const AxisWeights& vertical_weights) {
    const auto output_count = checked_element_count(target_height, target_width, channels);
    std::vector<float> output(output_count, 0.0F);
    for (int32_t y = 0; y < target_height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            for (int32_t channel = 0; channel < channels; ++channel) {
                float value = 0.0F;
                for (const auto& weight : vertical_weights[static_cast<std::size_t>(y)]) {
                    const auto source_index =
                        (static_cast<std::size_t>(weight.source_index) * target_width + x) *
                            static_cast<std::size_t>(channels) +
                        static_cast<std::size_t>(channel);
                    value += temporary[source_index] * weight.weight;
                }
                const auto destination_index = (static_cast<std::size_t>(y) * target_width + x) *
                                                   static_cast<std::size_t>(channels) +
                                               static_cast<std::size_t>(channel);
                output[destination_index] = value;
            }
        }
    }
    return output;
}

std::vector<float> resize_jax_linear(const float* pixels, int32_t source_height,
                                     int32_t source_width, int32_t channels, int32_t target_height,
                                     int32_t target_width) {
    if (target_height == 0 || target_width == 0) {
        return {};
    }
    const auto horizontal_weights = make_jax_linear_weights(source_width, target_width);
    const auto vertical_weights = make_jax_linear_weights(source_height, target_height);
    const auto temporary = resize_jax_horizontal(pixels, source_height, source_width, channels,
                                                 target_width, horizontal_weights);
    return resize_jax_vertical(temporary, target_height, target_width, channels, vertical_weights);
}

float round_to_nearest_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction < 0.5F) {
        return lower;
    }
    if (fraction > 0.5F) {
        return lower + 1.0F;
    }
    const auto integer = static_cast<int64_t>(lower);
    return integer % 2 == 0 ? lower : lower + 1.0F;
}

template <typename T>
void copy_resized_into_padded(const std::vector<T>& resized, int32_t resized_height,
                              int32_t resized_width, int32_t channels,
                              const ResizeWithPadGeometry& geometry, std::vector<T>& padded) {
    for (int32_t y = 0; y < resized_height; ++y) {
        for (int32_t x = 0; x < resized_width; ++x) {
            for (int32_t channel = 0; channel < channels; ++channel) {
                const auto source_index = (static_cast<std::size_t>(y) * resized_width + x) *
                                              static_cast<std::size_t>(channels) +
                                          static_cast<std::size_t>(channel);
                const auto destination_index =
                    (static_cast<std::size_t>(y + geometry.pad_top) * geometry.target_width + x +
                     geometry.pad_left) *
                        static_cast<std::size_t>(channels) +
                    static_cast<std::size_t>(channel);
                padded[destination_index] = resized[source_index];
            }
        }
    }
}

} // namespace

std::vector<float> quantile_normalize(const std::vector<float>& values, std::size_t last_dimension,
                                      const QuantileStats& stats) {
    validate_quantile_stats(stats);
    if (last_dimension == 0 || values.size() % last_dimension != 0) {
        throw std::invalid_argument("OpenPI quantile input has an invalid last dimension");
    }
    if (stats.q01.size() < last_dimension) {
        throw std::invalid_argument("OpenPI input statistics do not cover the last dimension");
    }

    std::vector<float> output(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t dimension = i % last_dimension;
        output[i] = (values[i] - stats.q01[dimension]) /
                        (stats.q99[dimension] - stats.q01[dimension] + kTransformEpsilon) * 2.0F -
                    1.0F;
    }
    return output;
}

std::vector<float> quantile_unnormalize(const std::vector<float>& values,
                                        std::size_t last_dimension, const QuantileStats& stats) {
    validate_quantile_stats(stats);
    if (last_dimension == 0 || values.size() % last_dimension != 0) {
        throw std::invalid_argument("OpenPI quantile output has an invalid last dimension");
    }
    if (stats.q01.size() > last_dimension) {
        throw std::invalid_argument("OpenPI output statistics exceed the last dimension");
    }

    std::vector<float> output(values);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::size_t dimension = i % last_dimension;
        if (dimension < stats.q01.size()) {
            output[i] = (values[i] + 1.0F) / 2.0F *
                            (stats.q99[dimension] - stats.q01[dimension] + kTransformEpsilon) +
                        stats.q01[dimension];
        }
    }
    return output;
}

int32_t discretize_state_value(float value) {
    std::array<double, 256> boundaries{};
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        boundaries[i] = -1.0 + static_cast<double>(i) / 128.0;
    }
    const auto iterator =
        std::upper_bound(boundaries.begin(), boundaries.end(), static_cast<double>(value));
    return static_cast<int32_t>(std::distance(boundaries.begin(), iterator)) - 1;
}

std::vector<int32_t> discretize_state(const std::vector<float>& state) {
    std::vector<int32_t> output;
    output.reserve(state.size());
    std::transform(state.begin(), state.end(), std::back_inserter(output),
                   [](float value) { return discretize_state_value(value); });
    return output;
}

std::string clean_prompt(std::string_view prompt) {
    const auto codepoints = decode_prompt(prompt);
    std::size_t first = 0;
    while (first < codepoints.size() && is_python_whitespace(codepoints[first].value)) {
        ++first;
    }
    std::size_t last = codepoints.size();
    while (last > first && is_python_whitespace(codepoints[last - 1].value)) {
        --last;
    }
    if (first == last) {
        return {};
    }

    const std::size_t begin_offset = codepoints[first].offset;
    const std::size_t end_offset = codepoints[last - 1].offset + codepoints[last - 1].length;
    std::string output(prompt.substr(begin_offset, end_offset - begin_offset));
    std::replace(output.begin(), output.end(), '_', ' ');
    std::replace(output.begin(), output.end(), '\n', ' ');
    return output;
}

std::string format_pi05_prompt(std::string_view prompt, const std::vector<float>& state) {
    const auto bins = discretize_state(state);
    std::string output = "Task: " + clean_prompt(prompt) + ", State: ";
    for (std::size_t i = 0; i < bins.size(); ++i) {
        if (i != 0) {
            output.push_back(' ');
        }
        output.append(std::to_string(bins[i]));
    }
    output.append(";\nAction: ");
    return output;
}

OrderedCameraSlots validate_and_order_camera_slots(const std::vector<CameraFrameView>& cameras) {
    if (cameras.size() != kCameraNames.size()) {
        throw std::invalid_argument("OpenPI requires exactly three named camera slots");
    }

    OrderedCameraSlots ordered{};
    std::array<bool, 3> found{};
    for (const auto& camera : cameras) {
        const std::size_t index = camera_index(camera.name);
        if (found[index]) {
            throw std::invalid_argument("Duplicate OpenPI camera slot: " +
                                        std::string(camera.name));
        }
        validate_camera(camera);
        ordered[index] = camera;
        found[index] = true;
    }
    if (std::find(found.begin(), found.end(), false) != found.end()) {
        throw std::invalid_argument("OpenPI camera set is incomplete");
    }
    return ordered;
}

void validate_pi05_two_camera_masks(const OrderedCameraSlots& cameras) {
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (cameras[i].name != kCameraNames[i]) {
            throw std::invalid_argument("OpenPI camera slots are not in canonical order");
        }
    }
    if (!cameras[0].valid || !cameras[1].valid || cameras[2].valid) {
        throw std::invalid_argument(
            "OpenPI pi0.5 two-camera profile requires masks [true, true, false]");
    }
}

ResizeWithPadGeometry compute_resize_with_pad_geometry(int32_t source_height, int32_t source_width,
                                                       int32_t target_height,
                                                       int32_t target_width) {
    if (source_height <= 0 || source_width <= 0 || target_height <= 0 || target_width <= 0) {
        throw std::invalid_argument("OpenPI resize dimensions must be positive");
    }
    const double ratio =
        std::max(static_cast<double>(source_width) / static_cast<double>(target_width),
                 static_cast<double>(source_height) / static_cast<double>(target_height));
    const int32_t resized_height = static_cast<int32_t>(static_cast<double>(source_height) / ratio);
    const int32_t resized_width = static_cast<int32_t>(static_cast<double>(source_width) / ratio);
    if (resized_height < 0 || resized_height > target_height || resized_width < 0 ||
        resized_width > target_width) {
        throw std::runtime_error("OpenPI resize geometry is inconsistent");
    }

    ResizeWithPadGeometry geometry;
    geometry.source_height = source_height;
    geometry.source_width = source_width;
    geometry.target_height = target_height;
    geometry.target_width = target_width;
    geometry.resized_height = resized_height;
    geometry.resized_width = resized_width;
    geometry.pad_top = (target_height - resized_height) / 2;
    geometry.pad_bottom = target_height - resized_height - geometry.pad_top;
    geometry.pad_left = (target_width - resized_width) / 2;
    geometry.pad_right = target_width - resized_width - geometry.pad_left;
    return geometry;
}

std::vector<uint8_t> resize_with_pad_uint8(const uint8_t* pixels, int32_t source_height,
                                           int32_t source_width, int32_t channels,
                                           int32_t target_height, int32_t target_width) {
    const auto source_count = checked_element_count(source_height, source_width, channels);
    if (pixels == nullptr) {
        throw std::invalid_argument("OpenPI uint8 resize source must not be null");
    }
    const auto geometry =
        compute_resize_with_pad_geometry(source_height, source_width, target_height, target_width);
    const auto target_count = checked_element_count(target_height, target_width, channels);
    std::vector<uint8_t> padded(target_count, 0U);
    if (geometry.resized_height == 0 || geometry.resized_width == 0) {
        return padded;
    }

    std::vector<float> input(source_count);
    std::transform(pixels, pixels + source_count, input.begin(),
                   [](uint8_t value) { return static_cast<float>(value); });
    const auto resized_float =
        resize_jax_linear(input.data(), source_height, source_width, channels,
                          geometry.resized_height, geometry.resized_width);
    std::vector<uint8_t> resized(resized_float.size());
    std::transform(resized_float.begin(), resized_float.end(), resized.begin(), [](float value) {
        const float clipped = std::max(0.0F, std::min(255.0F, value));
        return static_cast<uint8_t>(round_to_nearest_even(clipped));
    });
    copy_resized_into_padded(resized, geometry.resized_height, geometry.resized_width, channels,
                             geometry, padded);
    return padded;
}

std::vector<float> resize_with_pad_float(const float* pixels, int32_t source_height,
                                         int32_t source_width, int32_t channels,
                                         int32_t target_height, int32_t target_width) {
    const auto source_count = checked_element_count(source_height, source_width, channels);
    if (pixels == nullptr) {
        throw std::invalid_argument("OpenPI float resize source must not be null");
    }
    for (std::size_t i = 0; i < source_count; ++i) {
        if (!std::isfinite(pixels[i]) || pixels[i] < -1.0F || pixels[i] > 1.0F) {
            throw std::invalid_argument("OpenPI float resize pixels must be finite and in [-1, 1]");
        }
    }
    const auto geometry =
        compute_resize_with_pad_geometry(source_height, source_width, target_height, target_width);
    const auto target_count = checked_element_count(target_height, target_width, channels);
    std::vector<float> padded(target_count, -1.0F);
    if (geometry.resized_height == 0 || geometry.resized_width == 0) {
        return padded;
    }

    auto resized = resize_jax_linear(pixels, source_height, source_width, channels,
                                     geometry.resized_height, geometry.resized_width);
    std::transform(resized.begin(), resized.end(), resized.begin(),
                   [](float value) { return std::max(-1.0F, std::min(1.0F, value)); });
    copy_resized_into_padded(resized, geometry.resized_height, geometry.resized_width, channels,
                             geometry, padded);
    return padded;
}

std::vector<float> uint8_to_openpi_float(const std::vector<uint8_t>& pixels) {
    std::vector<float> result(pixels.size());
    std::transform(pixels.begin(), pixels.end(), result.begin(), [](uint8_t value) {
        return static_cast<float>(value) * kUint8ToOpenPiScale - 1.0F;
    });
    return result;
}

EulerSchedule make_euler_schedule(int32_t num_steps) {
    if (num_steps <= 0) {
        throw std::invalid_argument("OpenPI Euler step count must be positive");
    }
    EulerSchedule schedule;
    schedule.dt = -1.0F / static_cast<float>(num_steps);
    schedule.timesteps.reserve(static_cast<std::size_t>(num_steps));
    float time = 1.0F;
    while (time >= -schedule.dt / 2.0F) {
        schedule.timesteps.push_back(time);
        if (schedule.timesteps.size() > static_cast<std::size_t>(num_steps) + 1U) {
            throw std::runtime_error("OpenPI Euler schedule failed to converge");
        }
        time += schedule.dt;
    }
    if (schedule.timesteps.size() != static_cast<std::size_t>(num_steps)) {
        throw std::runtime_error("OpenPI Euler schedule produced an unexpected step count");
    }
    return schedule;
}

std::vector<float> make_fixed_noise(const std::vector<float>& noise, int32_t batch_size,
                                    int32_t action_horizon, int32_t action_dimension) {
    if (noise.size() != checked_flow_count(batch_size, action_horizon, action_dimension)) {
        throw std::invalid_argument("OpenPI fixed noise size does not match [B, H, D]");
    }
    if (std::any_of(noise.begin(), noise.end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("OpenPI fixed noise must contain finite values");
    }
    return noise;
}

void euler_step_in_place(std::vector<float>& sample, const std::vector<float>& velocity, float dt) {
    if (sample.size() != velocity.size()) {
        throw std::invalid_argument("OpenPI Euler sample and velocity sizes do not match");
    }
    if (!std::isfinite(dt)) {
        throw std::invalid_argument("OpenPI Euler dt must be finite");
    }
    for (std::size_t i = 0; i < sample.size(); ++i) {
        sample[i] = sample[i] + dt * velocity[i];
    }
}

} // namespace trtmc::openpi
