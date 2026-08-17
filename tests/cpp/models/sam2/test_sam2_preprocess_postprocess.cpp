/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_mask_postprocess.h"
#include "runtime/models/sam2/sam2_preprocess.h"
#include "utils/sha256.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Function>
void expectThrows(Function&& function, const char* context) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string("expected exception: ") + context);
}

std::vector<std::uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("failed to open exact SAM2 RGB fixture: " + path.string());
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("failed to size exact SAM2 RGB fixture");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input)
        throw std::runtime_error("failed to read exact SAM2 RGB fixture");
    return bytes;
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
    trtmc::internal::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.hex_digest();
}

struct OracleSpan {
    std::int32_t first{0};
    std::vector<std::int32_t> weights;
};

double oracleBicubic(double value) {
    constexpr double kA = -0.5;
    value = std::abs(value);
    if (value < 1.0)
        return ((kA + 2.0) * value - (kA + 3.0)) * value * value + 1.0;
    if (value < 2.0)
        return ((kA * value - 5.0 * kA) * value + 8.0 * kA) * value - 4.0 * kA;
    return 0.0;
}

std::vector<OracleSpan> makeOracleSpans(std::int32_t input_size, std::int32_t output_size) {
    constexpr std::int64_t kScale = std::int64_t{1} << trtmc::sam2::kPillowResizePrecisionBits;
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double filter_scale = std::max(scale, 1.0);
    const double support = 2.0 * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    std::vector<OracleSpan> result(static_cast<std::size_t>(output_size));
    for (std::int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = (static_cast<double>(output_index) + 0.5) * scale;
        const auto first =
            std::max<std::int32_t>(0, static_cast<std::int32_t>(center - support + 0.5));
        const auto end =
            std::min<std::int32_t>(input_size, static_cast<std::int32_t>(center + support + 0.5));
        auto& span = result[static_cast<std::size_t>(output_index)];
        span.first = first;
        std::vector<double> floating(static_cast<std::size_t>(end - first));
        double sum = 0.0;
        for (std::int32_t input_index = first; input_index < end; ++input_index) {
            const double distance =
                (static_cast<double>(input_index) - center + 0.5) * inverse_filter_scale;
            const double weight = oracleBicubic(distance);
            floating[static_cast<std::size_t>(input_index - first)] = weight;
            sum += weight;
        }
        for (const double weight : floating) {
            const double scaled = weight / sum * static_cast<double>(kScale);
            span.weights.push_back(
                static_cast<std::int32_t>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5));
        }
    }
    return result;
}

std::uint8_t applyOracleSpan(const std::uint8_t* source, std::int32_t stride,
                             const OracleSpan& span) {
    std::int64_t sum = std::int64_t{1} << (trtmc::sam2::kPillowResizePrecisionBits - 1);
    for (std::size_t index = 0; index < span.weights.size(); ++index) {
        sum += static_cast<std::int64_t>(
                   source[static_cast<std::size_t>(span.first) * static_cast<std::size_t>(stride) +
                          index * static_cast<std::size_t>(stride)]) *
               span.weights[index];
    }
    return static_cast<std::uint8_t>(
        std::clamp<std::int64_t>(sum >> trtmc::sam2::kPillowResizePrecisionBits, 0, 255));
}

std::vector<std::uint8_t> oracleResize(const std::vector<std::uint8_t>& input,
                                       std::int32_t input_height, std::int32_t input_width,
                                       std::int32_t output_height, std::int32_t output_width) {
    std::vector<std::uint8_t> horizontal(static_cast<std::size_t>(input_height) *
                                         static_cast<std::size_t>(output_width) * 3U);
    const auto horizontal_spans = makeOracleSpans(input_width, output_width);
    for (std::int32_t y = 0; y < input_height; ++y) {
        for (std::int32_t x = 0; x < output_width; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset =
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(input_width) * 3U +
                    static_cast<std::size_t>(channel);
                const auto destination_offset =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(output_width) +
                     static_cast<std::size_t>(x)) *
                        3U +
                    static_cast<std::size_t>(channel);
                horizontal[destination_offset] = applyOracleSpan(
                    input.data() + source_offset, 3, horizontal_spans[static_cast<std::size_t>(x)]);
            }
        }
    }

    std::vector<std::uint8_t> output(static_cast<std::size_t>(output_height) *
                                     static_cast<std::size_t>(output_width) * 3U);
    const auto vertical_spans = makeOracleSpans(input_height, output_height);
    for (std::int32_t y = 0; y < output_height; ++y) {
        for (std::int32_t x = 0; x < output_width; ++x) {
            for (std::int32_t channel = 0; channel < 3; ++channel) {
                const auto offset =
                    static_cast<std::size_t>(x) * 3U + static_cast<std::size_t>(channel);
                const auto destination_offset =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(output_width) +
                     static_cast<std::size_t>(x)) *
                        3U +
                    static_cast<std::size_t>(channel);
                output[destination_offset] =
                    applyOracleSpan(horizontal.data() + offset, output_width * 3,
                                    vertical_spans[static_cast<std::size_t>(y)]);
            }
        }
    }
    return output;
}

void testSharedPlansAgainstIndependentOracle() {
    // Exhaust every bounded axis pair, including upsampling, downsampling,
    // identity, clipped border support, negative lobes, and unequal ratios.
    for (std::int32_t input_size = 1; input_size <= 96; ++input_size) {
        for (std::int32_t output_size = 1; output_size <= 96; ++output_size) {
            const auto production = trtmc::sam2::makePillowBicubicAxisPlan(input_size, output_size);
            const auto oracle = makeOracleSpans(input_size, output_size);
            if (production.input_size != input_size || production.output_size != output_size ||
                production.spans.size() != static_cast<std::size_t>(output_size) ||
                production.spans.size() != oracle.size()) {
                throw std::runtime_error("SAM2 shared Pillow axis-plan shape drifted");
            }
            std::size_t expected_offset = 0U;
            for (std::size_t index = 0; index < production.spans.size(); ++index) {
                const auto& actual = production.spans[index];
                const auto& expected = oracle[index];
                if (actual.first != expected.first || actual.weight_count <= 0 ||
                    actual.weight_offset < 0 ||
                    static_cast<std::size_t>(actual.weight_offset) != expected_offset ||
                    actual.first < 0 || actual.first + actual.weight_count > input_size ||
                    static_cast<std::size_t>(actual.weight_count) != expected.weights.size()) {
                    throw std::runtime_error("SAM2 shared Pillow axis-plan span drifted");
                }
                for (std::int32_t weight = 0; weight < actual.weight_count; ++weight) {
                    if (production.weights[expected_offset + static_cast<std::size_t>(weight)] !=
                        expected.weights[static_cast<std::size_t>(weight)]) {
                        throw std::runtime_error("SAM2 shared Pillow coefficient drifted");
                    }
                }
                expected_offset += static_cast<std::size_t>(actual.weight_count);
            }
            if (expected_offset != production.weights.size())
                throw std::runtime_error("SAM2 shared Pillow weight packing drifted");
        }
    }

    // Exhaust every small 2-D geometry pair through both separable passes.
    for (std::int32_t input_height = 1; input_height <= 10; ++input_height) {
        for (std::int32_t input_width = 1; input_width <= 10; ++input_width) {
            std::vector<std::uint8_t> input(static_cast<std::size_t>(input_height) *
                                            static_cast<std::size_t>(input_width) * 3U);
            for (std::size_t index = 0; index < input.size(); ++index) {
                input[index] = static_cast<std::uint8_t>(
                    (index * 73U + static_cast<std::size_t>(input_height) * 29U +
                     static_cast<std::size_t>(input_width) * 151U) &
                    0xFFU);
            }
            for (std::int32_t output_height = 1; output_height <= 10; ++output_height) {
                for (std::int32_t output_width = 1; output_width <= 10; ++output_width) {
                    const auto expected =
                        oracleResize(input, input_height, input_width, output_height, output_width);
                    const auto actual = trtmc::sam2::resizePillowBicubicRgb(
                        input.data(), input_height, input_width, output_height, output_width);
                    if (actual != expected)
                        throw std::runtime_error("SAM2 packed Pillow plan diverged from oracle");
                }
            }
        }
    }
}

std::uint32_t floatBits(float value) {
    std::uint32_t result = 0U;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void validateFixedPlan(const trtmc::sam2::PillowResizeAxisPlan& plan, std::int32_t expected_input,
                       std::size_t expected_weights,
                       const std::vector<std::pair<std::int32_t, std::size_t>>& expected_taps,
                       std::int64_t expected_prefix_min, std::int64_t expected_prefix_max) {
    if (plan.input_size != expected_input || plan.output_size != 1024 ||
        plan.spans.size() != 1024U || plan.weights.size() != expected_weights) {
        throw std::runtime_error("SAM2 exact-v3 resize-plan shape drifted");
    }
    std::vector<std::pair<std::int32_t, std::size_t>> taps;
    std::size_t packed_offset = 0U;
    std::int64_t observed_min = 0;
    std::int64_t observed_max = 0;
    for (const auto& span : plan.spans) {
        const auto found = std::find_if(taps.begin(), taps.end(), [&](const auto& item) {
            return item.first == span.weight_count;
        });
        if (found == taps.end())
            taps.emplace_back(span.weight_count, 1U);
        else
            ++found->second;
        if (span.weight_offset < 0 ||
            static_cast<std::size_t>(span.weight_offset) != packed_offset ||
            span.weight_count <= 0 || span.first < 0 ||
            span.first + span.weight_count > expected_input) {
            throw std::runtime_error("SAM2 exact-v3 resize-plan packing drifted");
        }

        std::int64_t worst_negative = std::int64_t{1} << 21;
        std::int64_t worst_positive = std::int64_t{1} << 21;
        for (std::int32_t index = 0; index < span.weight_count; ++index) {
            const auto weight = plan.weights[packed_offset + static_cast<std::size_t>(index)];
            if (weight < 0)
                worst_negative += static_cast<std::int64_t>(weight) * 255;
            else
                worst_positive += static_cast<std::int64_t>(weight) * 255;
            observed_min = std::min(observed_min, worst_negative);
            observed_max = std::max(observed_max, worst_positive);
            if (worst_negative < std::numeric_limits<std::int32_t>::min() ||
                worst_positive > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error("SAM2 exact-v3 resize prefix exceeds int32 bounds");
            }
        }
        packed_offset += static_cast<std::size_t>(span.weight_count);
    }
    std::sort(taps.begin(), taps.end());
    if (taps != expected_taps || packed_offset != plan.weights.size() ||
        observed_min != expected_prefix_min || observed_max != expected_prefix_max) {
        throw std::runtime_error("SAM2 exact-v3 tap or prefix-bound evidence drifted");
    }
}

void testExactV3PlansAndNormalizationTable() {
    validateFixedPlan(trtmc::sam2::makePillowBicubicAxisPlan(1088, 1024), 1088, 4346U,
                      {{3, 2U}, {4, 770U}, {5, 252U}}, -140608243, 1214350067);
    validateFixedPlan(trtmc::sam2::makePillowBicubicAxisPlan(1280, 1024), 1280, 5114U,
                      {{3, 2U}, {4, 2U}, {5, 1020U}}, -98531203, 1172272772);

    constexpr std::array<float, 3> kMean = {0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> kStd = {0.229F, 0.224F, 0.225F};
    const auto& table = trtmc::sam2::sam2Rgb8NormalizationTable();
    for (std::size_t channel = 0; channel < 3U; ++channel) {
        for (std::size_t value = 0; value < 256U; ++value) {
            const float unit = static_cast<float>(static_cast<double>(value) / 255.0);
            const float expected = (unit - kMean[channel]) / kStd[channel];
            const float actual = table[channel * 256U + value];
            if (floatBits(actual) != floatBits(expected)) {
                throw std::runtime_error("SAM2 RGB8 normalization lookup bit pattern drifted");
            }
        }
    }
}

void testExactFiveFrameResizeIfAvailable() {
    const char* directory_text = std::getenv("TRTMC_SAM2_DECODED_RGB_DIR");
    if (directory_text == nullptr || *directory_text == '\0')
        return;
    constexpr std::array<const char*, 5> kExpected = {
        "bfc4b87e211b8437ede1b2244fe4c4bd0565afa7b05514d42305c7a2c2b1c275",
        "a1ac93ad5fe41f2109245b25d7068097f7d09f26e0592a4375d12bbfa81e44bb",
        "4c3da2fcc6a9c154036fad6742cd8bef9dc89a6abf7dd37cd6d9026ebb3b2676",
        "663996af0514e10932c035aedf81191c4e2420c726be930bd0c9809241a3f238",
        "bc35d5119f83ac8e169c42f68fd00117ae3bdcbfcf3ed33c45da677b60aa8bfc",
    };
    constexpr std::size_t kInputBytes = 1280U * 1088U * 3U;
    for (std::size_t frame = 0; frame < kExpected.size(); ++frame) {
        std::string name = "00000" + std::to_string(frame) + ".rgb";
        const auto input = readBytes(std::filesystem::path(directory_text) / name);
        if (input.size() != kInputBytes)
            throw std::runtime_error("exact SAM2 RGB fixture has the wrong geometry");
        const auto resized =
            trtmc::sam2::resizePillowBicubicRgb(input.data(), 1280, 1088, 1024, 1024);
        if (sha256(resized) != kExpected[frame])
            throw std::runtime_error("SAM2 Pillow resize hash drifted on " + name);
    }
}

} // namespace

int main() {
    testSharedPlansAgainstIndependentOracle();
    testExactV3PlansAndNormalizationTable();

    // Identity resize must preserve exact bytes.
    const std::array<std::uint8_t, 12> two_by_two = {0,   10,  20,  30,  40,  50,
                                                     100, 110, 120, 240, 250, 255};
    const auto identity = trtmc::sam2::resizePillowBicubicRgb(two_by_two.data(), 2, 2, 2, 2);
    if (!std::equal(identity.begin(), identity.end(), two_by_two.begin(), two_by_two.end()))
        throw std::runtime_error("SAM2 identity resize drifted");

    // Constant colors remain constant under the separable Pillow filter.
    std::vector<std::uint8_t> constant(7U * 5U * 3U);
    for (std::size_t index = 0; index < constant.size(); index += 3U) {
        constant[index] = 17;
        constant[index + 1U] = 91;
        constant[index + 2U] = 233;
    }
    const auto resized = trtmc::sam2::resizePillowBicubicRgb(constant.data(), 5, 7, 11, 13);
    for (std::size_t index = 0; index < resized.size(); index += 3U) {
        if (resized[index] != 17 || resized[index + 1U] != 91 || resized[index + 2U] != 233)
            throw std::runtime_error("SAM2 constant Pillow resize drifted");
    }

    const std::array<float, 3> one_pixel = {0.0F, 0.5F, 1.0F};
    const auto preprocessed = trtmc::sam2::preprocessFrame(one_pixel.data(), 1, 1);
    const std::array<std::uint8_t, 3> one_pixel_rgb8 = {0, 128, 255};
    const auto preprocessed_rgb8 = trtmc::sam2::preprocessRgb8Frame(one_pixel_rgb8.data(), 1, 1);
    if (preprocessed.resized_rgb_hwc.size() != 1024U * 1024U * 3U ||
        preprocessed.pixel_values.size() != 1024U * 1024U * 3U)
        throw std::runtime_error("SAM2 preprocessed shape is wrong");
    if (preprocessed.resized_rgb_hwc[0] != 0 || preprocessed.resized_rgb_hwc[1] != 128 ||
        preprocessed.resized_rgb_hwc[2] != 255)
        throw std::runtime_error("SAM2 float-to-byte conversion drifted");
    if (preprocessed.resized_rgb_hwc != preprocessed_rgb8.resized_rgb_hwc ||
        preprocessed.pixel_values != preprocessed_rgb8.pixel_values) {
        throw std::runtime_error("SAM2 float and RGB8 preprocessing semantics diverged");
    }
    const float expected_green = (static_cast<float>(128.0 / 255.0) - 0.456F) / 0.224F;
    if (std::abs(preprocessed.pixel_values[1024U * 1024U] - expected_green) > 1.0e-7F)
        throw std::runtime_error("SAM2 normalization drifted");

    // 2x2 -> 4x4 align_corners=false has clamped border samples and strict
    // zero thresholding.
    const std::array<float, 4> logits = {-1.0F, 1.0F, 1.0F, -1.0F};
    const auto mask = trtmc::sam2::resizeAndThresholdMask(logits.data(), 2, 2, 4, 4);
    const std::array<std::uint8_t, 16> expected = {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0};
    if (!std::equal(mask.begin(), mask.end(), expected.begin(), expected.end()))
        throw std::runtime_error("SAM2 mask resize/threshold drifted");

    std::vector<float> contraction_witness(256U * 256U, 0x1.2b6cfep+0F);
    for (std::size_t row = 0; row < 256U; ++row)
        contraction_witness[row * 256U + 1U] = -0x1.82c21cp+3F;
    const auto contraction_mask =
        trtmc::sam2::resizeAndThresholdMask(contraction_witness.data(), 256, 256, 1280, 1088);
    if (contraction_mask[2U] != 0U)
        throw std::runtime_error("SAM2 mask interpolation unexpectedly contracted arithmetic");

    float invalid = std::numeric_limits<float>::quiet_NaN();
    expectThrows([&] { (void)trtmc::sam2::preprocessFrame(&invalid, 1, 1); }, "nonfinite image");
    expectThrows([&] { (void)trtmc::sam2::resizeAndThresholdMask(&invalid, 1, 1, 1, 1); },
                 "nonfinite mask");

    testExactFiveFrameResizeIfAvailable();

    std::cout << "SAM2 pure C++ preprocess/postprocess tests passed\n";
    return 0;
}
