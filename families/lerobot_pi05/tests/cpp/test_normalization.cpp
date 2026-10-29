/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Expected values come from LeRobot's NormalizerProcessorStep formulas evaluated in float32 with
// numpy: MEAN_STD (x - mean) / (std + eps), QUANTILES 2 (x - q01) / (q99 - q01) - 1.

#include "families/lerobot_pi05/runtime/normalization.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void near(float actual, float expected, const std::string& message) {
    if (std::fabs(actual - expected) > 2.0e-6F * (1.0F + std::fabs(expected))) {
        std::cerr << "FAIL: " << message << ": " << actual << " != " << expected << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using namespace trtmc::lerobot_pi05;

    Normalization identity;
    near(normalize_value(identity, 0, 3.5F), 3.5F, "identity normalize");
    near(unnormalize_value(identity, 0, -2.0F), -2.0F, "identity unnormalize");

    // Statistics with a mode that says IDENTITY, or a mode with no statistics, also pass through.
    Normalization no_stats;
    no_stats.mode = NormalizationMode::kRange;
    near(normalize_value(no_stats, 0, 0.25F), 0.25F, "no statistics passes through");

    Normalization mean_std;
    mean_std.mode = NormalizationMode::kMeanStd;
    mean_std.offset = {1.0F, -2.0F};
    mean_std.scale = {4.0F, 0.5F};
    near(normalize_value(mean_std, 0, 3.0F), 0.5F, "mean/std normalize");
    near(normalize_value(mean_std, 1, -1.0F), 2.0F, "mean/std normalize, second dimension");
    near(unnormalize_value(mean_std, 0, 0.5F), 3.0F, "mean/std unnormalize");
    near(unnormalize_value(mean_std, 1, 2.0F), -1.0F, "mean/std unnormalize, second dimension");

    Normalization quantiles;
    quantiles.mode = NormalizationMode::kRange;
    quantiles.offset = {-1.0F, 10.0F, 5.0F};  // q01
    quantiles.scale = {4.0F, 20.0F, 1.0e-8F}; // q99 - q01; the last was zero, so eps
    near(normalize_value(quantiles, 0, 1.0F), 0.0F, "quantile midpoint");
    near(normalize_value(quantiles, 0, -1.0F), -1.0F, "quantile lower edge");
    near(normalize_value(quantiles, 0, 3.0F), 1.0F, "quantile upper edge");
    near(normalize_value(quantiles, 1, 15.0F), -0.5F, "quantile interior point");
    near(unnormalize_value(quantiles, 1, -0.5F), 15.0F, "quantile unnormalize");
    near(unnormalize_value(quantiles, 0, 1.0F), 3.0F, "quantile unnormalize upper edge");
    // Outside the quantile range LeRobot does not clip.
    near(normalize_value(quantiles, 0, 7.0F), 3.0F, "quantile extrapolates");
    // A degenerate (zero-spread) dimension unnormalizes to its quantile for any input in range.
    near(unnormalize_value(quantiles, 2, 0.0F), 5.0F, "zero-spread dimension");

    // MIN_MAX uses the same arithmetic with min and max - min.
    Normalization min_max;
    min_max.mode = NormalizationMode::kRange;
    min_max.offset = {-0.7F};
    min_max.scale = {2.6F}; // max 1.9
    near(normalize_value(min_max, 0, 0.3F), -0.23076922F, "min/max normalize");
    near(unnormalize_value(min_max, 0, -0.23076922F), 0.3F, "min/max round trip");

    if (failures == 0)
        std::cout << "lerobot_pi05 normalization tests passed\n";
    return failures == 0 ? 0 : 1;
}
