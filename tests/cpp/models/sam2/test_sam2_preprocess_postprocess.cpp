/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_preprocess.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void checkPlan(const trtmc::sam2::PillowResizeAxisPlan& plan, std::int32_t input,
               std::size_t weights, const std::array<std::size_t, 3>& taps) {
    check(plan.spans.size() == 1024U && plan.weights.size() == weights,
          "fixed Pillow plan shape drifted");
    std::array<std::size_t, 3> observed{};
    std::size_t offset = 0;
    for (const auto& span : plan.spans) {
        check(span.weight_count >= 3 && span.weight_count <= 5, "fixed Pillow tap count drifted");
        check(span.weight_offset >= 0 && static_cast<std::size_t>(span.weight_offset) == offset &&
                  span.first >= 0 && span.first + span.weight_count <= input,
              "fixed Pillow plan packing drifted");
        ++observed[static_cast<std::size_t>(span.weight_count - 3)];
        offset += static_cast<std::size_t>(span.weight_count);
    }
    check(observed == taps && offset == plan.weights.size(),
          "fixed Pillow plan fingerprint drifted");
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

} // namespace

int main() {
    checkPlan(trtmc::sam2::makePillowBicubicAxisPlan(1088, 1024), 1088, 4346U, {2U, 770U, 252U});
    checkPlan(trtmc::sam2::makePillowBicubicAxisPlan(1280, 1024), 1280, 5114U, {2U, 2U, 1020U});

    constexpr std::array<float, 3> kMean{0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> kStd{0.229F, 0.224F, 0.225F};
    const auto& table = trtmc::sam2::sam2Rgb8NormalizationTable();
    for (std::size_t channel = 0; channel < 3U; ++channel) {
        for (std::size_t value = 0; value < 256U; ++value) {
            const float unit = static_cast<float>(static_cast<double>(value) / 255.0);
            check(bits(table[channel * 256U + value]) ==
                      bits((unit - kMean[channel]) / kStd[channel]),
                  "RGB8 normalization lookup drifted");
        }
    }
    return 0;
}
