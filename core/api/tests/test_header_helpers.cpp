/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/features.hpp"
#include "trtmc/numeric.hpp"

#include <array>
#include <cstring>
#include <iostream>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
template <class Function>
void rejects(Function call, const char* message) {
    bool rejected = false;
    try {
        call();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == TRTMC_INVALID_ARGUMENT;
    }
    check(rejected, message);
}
void ranking() {
    const float inf = std::numeric_limits<float>::infinity();
    float values[]{3, -0.0F, 0, 3, -inf, inf, -4, 2, 2, 1, -0.0F, 0};
    trtmc_feature_token_v1 positions[]{{42, 0, 4, 1, 3, 9}, {99, 1, 7, 0, 0, 0}};
    trtmc_vocabulary_scores_view_v1 input{{values, 12, 2, 6}, positions, 2, {"fixture", 7}};
    auto ranked = trtmc::rank_masked_tokens(input, 100);
    check(ranked.size() == 2 && ranked[0].position.token_index == 4 &&
              ranked[0].position.byte_begin == 3 && ranked[1].position.input_index == 1,
          "ranking copies positions in original selected-row order");
    const int64_t expected[]{5, 0, 3, 1, 2, 4};
    for (size_t i = 0; i < 6; ++i) {
        check(ranked[0].candidates[i].token_id == expected[i],
              "ties/inf have deterministic token-ID ordering");
        check(std::memcmp(&ranked[0].candidates[i].logit, &values[expected[i]], sizeof(float)) == 0,
              "every returned score is the exact raw float, not a probability");
    }
    check(ranked[1].candidates[0].token_id == 1 && ranked[1].candidates[1].token_id == 2,
          "second row ranking is independent and tie order is stable");
    check(trtmc::rank_masked_tokens(input).front().candidates.size() == 5,
          "default limit is five candidates");
    auto zero = trtmc::rank_masked_tokens(input, 0);
    check(zero.size() == 2 && zero[0].candidates.empty() && zero[1].position.token_id == 99,
          "zero limit retains rows with empty candidate lists");
    check(trtmc::rank_masked_tokens(input, std::numeric_limits<int64_t>::max())[0]
                  .candidates.size() == 6,
          "oversized candidate bound clamps to real vocabulary without oversized allocation");
    rejects([&] { (void)trtmc::rank_masked_tokens(input, -1); }, "negative candidate bound fails");
    auto malformed = input;
    malformed.logits.count = 11;
    rejects([&] { (void)trtmc::rank_masked_tokens(malformed); }, "ranking matrix count mismatch");
    malformed = input;
    malformed.positions = nullptr;
    rejects([&] { (void)trtmc::rank_masked_tokens(malformed); }, "ranking missing position array");
    malformed = input;
    malformed.logits.data = nullptr;
    rejects([&] { (void)trtmc::rank_masked_tokens(malformed); }, "ranking missing score array");
    malformed = input;
    malformed.position_count = 1;
    rejects([&] { (void)trtmc::rank_masked_tokens(malformed); }, "ranking mapping rows disagree");
    malformed = input;
    malformed.logits.columns = std::numeric_limits<uint64_t>::max();
    rejects([&] { (void)trtmc::rank_masked_tokens(malformed); },
            "ranking vocabulary/shape overflow");
    malformed = input;
    malformed.logits = {nullptr, 0, 2, 0};
    rejects([&] { (void)trtmc::rank_masked_tokens(malformed); },
            "nonempty mask rows need a vocabulary");
    const trtmc_vocabulary_scores_view_v1 empty{{nullptr, 0, 0, 6}, nullptr, 0, {}};
    check(trtmc::rank_masked_tokens(empty).empty(), "no selected masks means no ranking rows");
    values[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)trtmc::rank_masked_tokens(input); },
            "NaN ranking is not comparator-dependent");
    rejects([&] { (void)trtmc::rank_masked_tokens(input, 0); },
            "NaN rule also applies to zero limit");
    positions[0].token_index = 100;
    check(ranked[0].position.token_index == 4 && ranked[0].candidates[1].logit == 3,
          "ranked values and source metadata own their storage");
    const float small_values[]{1, 3, 2};
    const trtmc_vocabulary_scores_view_v1 small{{small_values, 3, 1, 3}, positions, 1, {}};
    check(trtmc::rank_masked_tokens(small)[0].candidates.size() == 3,
          "default is useful even for a vocabulary smaller than five");
    std::vector<float> wide_values(2 * 1024);
    for (size_t i = 0; i < wide_values.size(); ++i)
        wide_values[i] = static_cast<float>(i);
    const trtmc_vocabulary_scores_view_v1 wide{
        {wide_values.data(), wide_values.size(), 2, 1024}, positions, 2, {}};
    auto compact = trtmc::rank_masked_tokens(wide, 5);
    check(compact.size() == 2 && compact[0].candidates.size() == 5 &&
              compact[1].candidates.size() == 5 && compact[0].candidates.capacity() < 1024 &&
              compact[1].candidates.capacity() < 1024 &&
              compact[0].candidates[0].token_id == 1023 && compact[0].candidates[0].logit == 1023 &&
              compact[1].candidates[0].token_id == 1023 && compact[1].candidates[0].logit == 2047,
          "ranking reuses one vocabulary scratch buffer and owns only compact per-row candidates");
}

void quantiles() {
    const double grid[]{0.1, 0.4, 0.9};
    float values[]{-0.0F, 2, 4, 6, 10, 12, 14, 16, 30, 32, 34, 36};
    int64_t steps[]{1, 4};
    char first_name[]{'x', '\0', 'p', 'o', 's'};
    trtmc_string_view names[]{{first_name, 5}, {"speed", 5}}, units[]{{"m", 1}, {"m/s", 3}};
    trtmc_quantile_forecast_view_v1 input{values, 12, {grid, 3},
                                          2,      2,  {{steps, 2}, {names, 2}, {units, 2}}};
    auto exact = trtmc::interpolate_quantile(input, grid[0]);
    check(exact.level == 0.1 && exact.horizon == 2 && exact.channels == 2 &&
              std::memcmp(exact.values.data(), values, 4 * sizeof(float)) == 0,
          "exact quantile copies its Q slice without changing float bits");
    auto middle = trtmc::interpolate_quantile(input, 0.65);
    check(middle.values == std::vector<float>({20, 22, 24, 26}) &&
              middle.horizon_steps == std::vector<int64_t>({1, 4}) &&
              middle.channel_names[0] == std::string(first_name, 5) &&
              middle.channel_units[1] == "m/s",
          "irregular-grid interpolation preserves H/C axes, actual steps and length-delimited "
          "metadata");
    auto median = trtmc::median_forecast(input);
    check(median.level == 0.5 && median.values == std::vector<float>({14, 16, 18, 20}),
          "median interpolates by declared probabilities, not row ordinal or arithmetic mean");
    auto unnamed = input;
    unnamed.axes.channel_names = {};
    unnamed.axes.channel_units = {};
    check(trtmc::median_forecast(unnamed).channel_names.empty() &&
              trtmc::median_forecast(unnamed).channel_units.empty(),
          "unspecified channel metadata stays empty");
    for (double level : {-0.1, 0.05, 0.95, 1.1, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
        rejects([&] { (void)trtmc::interpolate_quantile(input, level); },
                "invalid/out-of-grid level cannot clamp silently");
    for (const auto& bad_grid :
         std::vector<std::array<double, 3>>{{0.1, 0.1, 0.9},
                                            {0.4, 0.1, 0.9},
                                            {-0.1, 0.4, 0.9},
                                            {0.1, 0.4, 1.1},
                                            {0.1, std::numeric_limits<double>::quiet_NaN(), 0.9},
                                            {0.1, 0.4, std::numeric_limits<double>::infinity()}}) {
        auto bad = input;
        bad.quantile_levels.data = bad_grid.data();
        rejects([&] { (void)trtmc::median_forecast(bad); },
                "quantile grid must be finite, ordered and unique");
    }
    auto bad = input;
    bad.value_count = 11;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "quantile Q/H/C count mismatch");
    bad = input;
    bad.values = nullptr;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "quantile values cannot be null");
    bad = input;
    bad.quantile_levels.data = nullptr;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "quantile grid cannot be null");
    bad = input;
    bad.horizon = std::numeric_limits<uint64_t>::max();
    rejects([&] { (void)trtmc::median_forecast(bad); },
            "quantile shape overflow fails before access");
    bad = input;
    bad.channels = 0;
    rejects([&] { (void)trtmc::median_forecast(bad); },
            "zero channel forecast is not invented empty output");
    rejects([&] { (void)trtmc::median_forecast(trtmc_quantile_forecast_view_v1{}); },
            "empty/moved-from forecast rejected");
    bad = input;
    bad.axes.horizon_steps.size = 1;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "horizon metadata must match its axis");
    bad = input;
    bad.axes.horizon_steps.data = nullptr;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "horizon metadata cannot be null");
    bad = input;
    bad.axes.channel_names.size = 1;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "channel metadata count mismatch");
    bad = input;
    bad.axes.channel_units.data = nullptr;
    rejects([&] { (void)trtmc::median_forecast(bad); }, "channel metadata cannot be null");
    names[0] = {nullptr, 5};
    rejects([&] { (void)trtmc::median_forecast(input); }, "channel string has missing bytes");
    names[0] = {first_name, 5};
    steps[0] = 0;
    rejects([&] { (void)trtmc::median_forecast(input); }, "horizon steps must be positive");
    steps[0] = 5;
    rejects([&] { (void)trtmc::median_forecast(input); }, "horizon steps must increase");
    steps[0] = 1;
    for (float invalid :
         {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        values[11] = invalid;
        rejects([&] { (void)trtmc::interpolate_quantile(input, grid[0]); },
                "nonfinite forecast data is rejected, not repaired");
    }
    first_name[0] = 'z';
    steps[1] = 9;
    values[0] = -123;
    check(exact.values[0] == 0 && std::signbit(exact.values[0]) && exact.horizon_steps[1] == 4 &&
              exact.channel_names[0][0] == 'x',
          "summary owns its values and axes after input mutation/release");

    const double crossing_grid[]{0.2, 0.8};
    const float crossing_values[]{10, 0};
    const int64_t step = 1;
    const trtmc_quantile_forecast_view_v1 crossed{crossing_values,     2, {crossing_grid, 2}, 1, 1,
                                                  {{&step, 1}, {}, {}}};
    check(trtmc::interpolate_quantile(crossed, 0.35).values[0] == 7.5F,
          "helper does not sort or repair crossed quantile predictions");
    const double one_level = 0.5;
    const float one_values[]{7, 8}, point_values[]{1000, 2000};
    const trtmc_quantile_forecast_view_v1 one{one_values, 2, {&one_level, 1},
                                              1,          2, {{&step, 1}, {}, {}}};
    const trtmc_point_and_quantile_forecast_view_v1 joint{
        {{point_values, 2, 1, 2}, {{&step, 1}, {}, {}}}, one};
    check(trtmc::median_forecast(joint.quantiles).values[0] == 7 &&
              joint.point.values.data[0] == 1000,
          "quantile-derived median is not an independent model point/mean prediction");
    rejects([&] { (void)trtmc::interpolate_quantile(one, 0.4); },
            "one-level grid permits exact lookup only");
    const double above_median = 0.8;
    auto absent_median = one;
    absent_median.quantile_levels.data = &above_median;
    rejects([&] { (void)trtmc::median_forecast(absent_median); },
            "missing median cannot borrow an out-of-grid endpoint");
}
} // namespace
int main() {
    ranking();
    quantiles();
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
