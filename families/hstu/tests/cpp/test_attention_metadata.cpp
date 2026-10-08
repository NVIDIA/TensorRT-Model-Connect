/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/attention_metadata.h"

#include <array>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<std::int32_t> input(std::size_t batch, std::size_t start) {
    constexpr std::array<std::int32_t, 8> histories{0, 1, 127, 128, 129, 400, 402, 1024};
    const auto stride = 8 * (batch + 1);
    std::vector<std::int32_t> value(5 * stride, 0x13579BDF);
    value[stride] = 0;
    for (std::size_t sample = 0; sample < batch; ++sample) {
        const auto history = histories[(start + sample) % histories.size()];
        const auto candidates = static_cast<std::int32_t>(sample + 7);
        value[stride + sample + 1] = value[stride + sample] + history + candidates;
        value[2 * stride + sample] = candidates;
    }
    return value;
}

void valid_layouts() {
    // Explicit expected boundary values, including empty and exact-page history.
    constexpr std::array<std::int32_t, 8> tails{128, 1, 127, 128, 1, 16, 18, 128};
    for (const std::size_t batch : {1U, 2U, 4U, 8U}) {
        for (std::size_t start = 0; start < tails.size(); ++start) {
            auto values = input(batch, start);
            const auto before = values;
            trtmc::hstu::fill_attention_page_lengths(values.data(), batch, values.size());
            const auto first = 17 * (batch + 1);
            for (std::size_t index = 0; index < values.size(); ++index) {
                if (index >= first && index < first + batch)
                    check(values[index] == tails[(start + index - first) % tails.size()],
                          "empty/page-boundary/append history has the correct encoded tail");
                else
                    check(values[index] == before[index],
                          "all original fields and other reserved metadata remain unchanged");
            }
        }
    }
}

void invalid(std::vector<std::int32_t> values, std::size_t batch, std::size_t extent,
             bool null_pointer = false) {
    const auto before = values;
    bool rejected = false;
    try {
        trtmc::hstu::fill_attention_page_lengths(null_pointer ? nullptr : values.data(), batch,
                                                 extent);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "invalid metadata is rejected");
    check(values == before, "rejection leaves the complete metadata unchanged");
}

void invalid_layouts() {
    const auto values = input(2, 0);
    invalid(values, 2, values.size(), true);
    invalid(values, 0, values.size());
    invalid(values, std::numeric_limits<std::size_t>::max(), values.size());
    invalid(values, 2, values.size() - 1);
    invalid(values, 2, values.size() + 1);
    auto negative_targets = values;
    negative_targets[2 * 24 + 1] = -1;
    invalid(negative_targets, 2, negative_targets.size());
    auto too_many_targets = values;
    too_many_targets[2 * 24 + 1] = 100;
    invalid(too_many_targets, 2, too_many_targets.size());
    auto reversed_offsets = values;
    reversed_offsets[24 + 2] = -1;
    invalid(reversed_offsets, 2, reversed_offsets.size());
}

} // namespace

int main() {
    valid_layouts();
    invalid_layouts();
    return failures == 0 ? 0 : 1;
}
