/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_matmul.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (condition)
        return;
    std::cerr << "FAIL: " << name << '\n';
    ++g_failures;
}

void test_gpu_matmul_matches_fp32_contract() {
    constexpr int32_t rows = 17;
    constexpr int32_t inner = 13;
    constexpr int32_t columns = 19;
    std::vector<float> lhs(static_cast<std::size_t>(rows * inner));
    std::vector<float> rhs(static_cast<std::size_t>(inner * columns));
    std::vector<float> bias(static_cast<std::size_t>(columns));
    for (std::size_t i = 0; i < lhs.size(); ++i)
        lhs[i] = static_cast<float>(static_cast<int32_t>(i % 11) - 5) * 0.125F;
    for (std::size_t i = 0; i < rhs.size(); ++i)
        rhs[i] = static_cast<float>(static_cast<int32_t>(i % 7) - 3) * 0.0625F;
    for (int32_t column = 0; column < columns; ++column)
        bias[static_cast<std::size_t>(column)] = static_cast<float>(column - 9) * 0.03125F;

    std::vector<float> expected(static_cast<std::size_t>(rows * columns));
    for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < columns; ++column) {
            double value = bias[static_cast<std::size_t>(column)];
            for (int32_t k = 0; k < inner; ++k) {
                value += static_cast<double>(lhs[static_cast<std::size_t>(row * inner + k)]) *
                         static_cast<double>(rhs[static_cast<std::size_t>(k * columns + column)]);
            }
            expected[static_cast<std::size_t>(row * columns + column)] = static_cast<float>(value);
        }
    }

    trtmc::PixArtGpuMatmul matmul;
    std::vector<float> actual(expected.size());
    check(matmul.run(lhs.data(), rhs.data(), bias.data(), actual.data(), rows, inner, columns),
          "PixArt GPU matmul executes");
    float max_error = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i)
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    check(max_error <= 1.0e-5F, "PixArt GPU matmul matches FP32 CPU accumulation");
    check(!matmul.run(lhs.data(), rhs.data(), bias.data(), actual.data(), 0, inner, columns),
          "PixArt GPU matmul rejects invalid dimensions");
}

} // namespace

int main() {
    test_gpu_matmul_matches_fp32_contract();
    return g_failures == 0 ? 0 : 1;
}
