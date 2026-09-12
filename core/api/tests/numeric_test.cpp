/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/numeric.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    if (!ok) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string json = "{\"format\":1,\"family\":\"numeric_fixture\",\"task\":\"" + mode +
                             "\",\"backend\":\"fake\",\"sections\":{}}";
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(magic), 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((static_cast<std::uint64_t>(json.size()) >> shift) & 255));
    file.write(json.data(), static_cast<std::streamsize>(json.size()));
}
template <class F>
void rejects(F invoke, trtmc_status expected, const char* label) {
    bool ok = false;
    try {
        invoke();
    } catch (const trtmc::Error& error) {
        ok = error.code() == expected;
    }
    check(ok, label);
}
bool name(trtmc_string_view value, std::string_view expected) {
    return trtmc::detail::string_view(value) == expected;
}
void regression_values(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    const auto load = [&](const std::string& mode) {
        const auto path = root / (mode + ".bundle");
        bundle(path, mode);
        return trtmc::Model::load(path.string(), options);
    };
    const float values[]{1, 2, 3, 4};
    const std::uint8_t mask[]{1, 0, 1, 1};
    const trtmc::SeriesHistory history{{{values}, 2, 2}, {mask}};
    auto model = load("regression_values");
    check(model.tasks().size() == 1 && model.supports<trtmc::SeriesToRegressionValues>() &&
              !model.supports<trtmc::SeriesToRegressionDistribution>() &&
              !model.supports<trtmc::SeriesToPointForecast>(),
          "deterministic targets are a distinct loaded Task, not a distribution or horizon");
    auto task = model.task<trtmc::SeriesToRegressionValues>();
    check(task.config_fields().size() == 1, "target regression declares its own configuration");
    rejects([&] { task.run({history}, {{"unknown", 1}}); }, TRTMC_INVALID_CONFIG,
            "unknown config cannot execute target regression");
    rejects([&] { task.run({history}, {{"scale", std::int64_t{1}}}); }, TRTMC_INVALID_CONFIG,
            "wrong config type cannot execute target regression");
    auto first = task.run({history}, {{"scale", 2.0}});
    check(first.view().values.size == 2 && first.view().values.data[0] == 16 &&
              first.view().values.data[1] == 1 && first.view().target_names.size == 0 &&
              first.view().target_units.size == 0,
          "target order, complete values, observed mask and absence of metadata survive");
    auto zero = task.run({history}, {{"scale", 0.0}});
    check(zero.view().values.data[0] == 0 && zero.view().values.data[1] == 2,
          "explicit zero is preserved and invalid configurations caused no side effects");
    auto retained = [&] {
        auto named = load("regression_values_named");
        return named.task<trtmc::SeriesToRegressionValues>().run(
            {trtmc::SeriesHistory::from_flat({values})});
    }();
    check(retained.view().values.data[0] == 10 &&
              name(retained.view().target_names.data[0], "total") &&
              name(retained.view().target_units.data[1], "count"),
          "owned target values and optional names/units survive the caller model scope");
    auto moved = std::move(retained);
    check(retained.view().values.size == 0 && moved.view().values.data[0] == 10,
          "moving target regression results retains ownership");
    for (const auto mode : {"regression_values_empty", "regression_values_nonfinite",
                            "regression_values_bad_names", "regression_values_bad_units"})
        rejects([&] { load(mode).task<trtmc::SeriesToRegressionValues>().run({history}); },
                TRTMC_INTERNAL_ERROR, "malformed target values or metadata are rejected");
}
void forecasts(const trtmc::Model& model, const trtmc::SeriesHistory& history) {
    auto joint =
        model.task<trtmc::SeriesToPointAndQuantileForecast>().run({history}, {{"frequency", 2}});
    const auto both = joint.view();
    check(both.point.values.rows == 2 && both.point.values.columns == 3 &&
              both.quantiles.channels == 3 && both.quantiles.horizon == 2 &&
              both.quantiles.quantile_levels.size == 3 &&
              both.quantiles.quantile_levels.data[1] == 0.5 && both.point.values.data[0] == 1190 &&
              both.quantiles.values[6] == 1195,
          "one joint evaluation preserves distinct point and median quantile outputs");
    check(both.point.axes.horizon_steps.data[1] == 3 &&
              both.quantiles.axes.horizon_steps.data[1] == 3,
          "joint outputs retain matching future-time axes");
    auto derived = trtmc::median_forecast(both.quantiles);
    check(derived.level == 0.5 && derived.values[0] == 1195 && both.point.values.data[0] == 1190 &&
              derived.horizon_steps == std::vector<int64_t>({1, 3}),
          "header-only median uses actual quantiles, never the independently returned point head");
    auto point = model.task<trtmc::SeriesToPointForecast>();
    check(point.config_fields().size() == 1, "family forecast config discoverable");
    auto p = point.run({history}, {{"frequency", 2}});
    const auto pv = p.view();
    check(pv.values.rows == 2 && pv.values.columns == 3 && pv.values.count == 6 &&
              pv.values.data[0] == 190,
          "point forecast preserves horizon/channel axes and observed mask");
    check(pv.axes.horizon_steps.size == 2 && pv.axes.horizon_steps.data[0] == 1 &&
              pv.axes.horizon_steps.data[1] == 3,
          "horizon offsets are explicit rather than guessed");
    auto q = model.task<trtmc::SeriesToQuantileForecast>().run({history});
    const auto qv = q.view();
    check(qv.quantile_levels.size == 2 && qv.quantile_levels.data[0] == 0.1 &&
              qv.quantile_levels.data[1] == 0.9 && qv.horizon == 2 && qv.channels == 3 &&
              qv.value_count == 12 && qv.values[6] == 7,
          "quantile-major Q,H,C output retains levels and layout");
    auto median = trtmc::median_forecast(q);
    check(median.values == std::vector<float>({4, 5, 6, 7, 8, 9}) && median.horizon == 2 &&
              median.channels == 3 && median.channel_names.empty(),
          "result-owner median interpolation preserves Q/H/C interpretation and omitted channel "
          "names");
    auto retained_summary = [&] {
        auto temporary = model.task<trtmc::SeriesToQuantileForecast>().run({history});
        return trtmc::interpolate_quantile(temporary, 0.9);
    }();
    check(retained_summary.values[0] == 7 && retained_summary.horizon_steps[1] == 3,
          "quantile summary owns values/axes after native result destruction");
    auto r = model.task<trtmc::SeriesToRegressionDistribution>().run({history});
    const auto rv = r.view();
    check(rv.distribution == TRTMC_DISTRIBUTION_NORMAL && rv.target_count == 2 &&
              rv.parameter_count == 2 && name(rv.parameters[0].name, "scale") &&
              rv.parameters[0].values[0] == 0.5F && name(rv.parameters[1].name, "location") &&
              rv.parameters[1].values[1] == 20,
          "regression has named per-target parameters rather than forecast axes");
    check(rv.target_names.size == 0 && rv.target_units.size == 0,
          "unknown physical targets and units stay unspecified");
    auto student = model.task<trtmc::SeriesToRegressionDistribution>().run(
        {history}, {{"distribution", "student_t"}});
    check(student.view().parameter_count == 3 &&
              name(student.view().parameters[0].name, "degrees_of_freedom"),
          "Student-t parameters are named without renormalization");
    auto count = model.task<trtmc::SeriesToRegressionDistribution>().run(
        {history}, {{"distribution", "negative_binomial"}});
    check(count.view().distribution == TRTMC_DISTRIBUTION_NEGATIVE_BINOMIAL &&
              name(count.view().parameters[0].name, "total_count"),
          "count distribution stays distinct from location-scale distributions");
    rejects(
        [&] {
            (void)model.task<trtmc::SeriesToRegressionDistribution>().run({history},
                                                                          {{"frequency", 1}});
        },
        TRTMC_INVALID_CONFIG, "forecast-only frequency does not silently alter regression");
    auto moved = std::move(r);
    check(r.view().parameter_count == 0 && moved.view().parameters[1].values[1] == 20,
          "numeric result move retains owned arrays");
}
void latents(const trtmc::Model& model) {
    const float values[] = {1, 2, 3, 4}, initial[] = {9, 8, 7, 6}, mask[] = {1, 0.5F},
                noise[] = {10, 11, 12, 13, 14, 15, 16, 17};
    const trtmc::FloatMatrixView latent{{values}, 2, 2}, start{{initial}, 2, 2};
    const trtmc::Span<const float> noises{noise};
    const trtmc::Config schedule{{"sampling_steps", std::vector<double>{0, 0.25, 0.75, 1}}};
    auto conditioned = model.task<trtmc::LatentConditionedTextGeneration>().run(
        {{values}, {mask}, {initial}, noises, "preserved"}, schedule);
    check(conditioned.text() == "conditioned" && conditioned.token_ids().size() == 6 &&
              conditioned.token_ids()[0] == 1 && conditioned.token_ids()[1] == 50 &&
              conditioned.token_ids()[2] == 9 && conditioned.token_ids()[3] == 17 &&
              conditioned.token_ids()[4] == 25 && conditioned.token_ids()[5] == 9,
          "all latent/float-mask/noise/schedule replay inputs reach the family");
    auto replay =
        model.task<trtmc::LatentReplayToText>().run({{initial}, "source prompt", noises}, schedule);
    check(replay.text() == "source prompt|replayed" && replay.token_ids()[0] == 9 &&
              replay.token_ids()[1] == 17,
          "initial-latent-only replay retains the actual text prompt");
    check(model.task<trtmc::LatentReplayToText>().run({{initial}, "", {}}).text() == "|replayed",
          "replay without text conditioning is independently valid");
    auto noise_only = model.task<trtmc::LatentReplayToText>().run({{}, "", noises}, schedule);
    check(noise_only.text() == "|replayed" && noise_only.token_ids()[0] == -1 &&
              noise_only.token_ids()[1] == 17,
          "SDE-only replay preserves the family initial-state policy without shape flags");
    auto denoised = model.task<trtmc::LatentDenoisingStep>().run({latent, start, 0.25});
    check(denoised.view().latents.rows == 2 && denoised.view().latents.columns == 2 &&
              denoised.view().latents.data[0] == 10.25,
          "denoising receives explicit timestep and self-conditioning");
    auto scores = model.task<trtmc::LatentToTokenLogits>().run({latent, {}, 0.25});
    check(scores.view().logits.rows == 2 && scores.view().logits.columns == 3 &&
              scores.view().logits.data[0] == 1.25 &&
              name(scores.view().vocabulary_id, "fixture-vocabulary"),
          "decoder result has position/vocabulary axes and identity");
    const float packed[] = {1, 2, 9, 8, 3, 4, 7, 6};
    const trtmc::Config guidance{{"self_cond_cfg_scale", 3.0}};
    const auto native_request = trtmc::LatentDenoisingStepRequest::native_packed({packed}, 0.25);
    auto native_denoised = model.task<trtmc::LatentDenoisingStep>().run(native_request, guidance);
    auto logical_denoised =
        model.task<trtmc::LatentDenoisingStep>().run({latent, start, 0.25}, guidance);
    check(native_denoised.view().latents.rows == 2 && native_denoised.view().latents.columns == 2 &&
              native_denoised.view().latents.data[0] == 12.25F &&
              native_denoised.view().latents.data[2] == 3 &&
              logical_denoised.view().latents.data[0] == native_denoised.view().latents.data[0],
          "family-native packed 2*D input is not packed twice and matches the logical "
          "representation");
    auto native_logits = model.task<trtmc::LatentToTokenLogits>().run(
        trtmc::LatentToTokenLogitsRequest::native_packed({packed}, 0.25), guidance);
    check(native_logits.view().logits.columns == 3 && native_logits.view().logits.data[0] == 12.25F,
          "native packed decoder retains timestep/guidance and its different vocabulary width");
    auto double_self = native_request;
    double_self.self_condition = start;
    rejects([&] { model.task<trtmc::LatentDenoisingStep>().run(double_self); },
            TRTMC_INVALID_ARGUMENT,
            "native packed input forbids a second self-conditioning operand");
    rejects(
        [&] {
            model.task<trtmc::LatentToTokenLogits>().run(
                trtmc::LatentToTokenLogitsRequest::native_packed({packed, 7}, 0.25));
        },
        TRTMC_INVALID_ARGUMENT, "family checks native packed length against its own engine layout");
    rejects(
        [&] {
            (void)model.task<trtmc::LatentConditionedTextGeneration>().run(
                {{values}, {}, {initial}, {}});
        },
        TRTMC_INVALID_ARGUMENT, "latent conditioning cannot omit its required mask");
    rejects([&] { (void)model.task<trtmc::LatentReplayToText>().run({{}, "prompt", {}}); },
            TRTMC_INVALID_ARGUMENT, "latent replay requires initial state or SDE noise");
    rejects([&] { (void)model.task<trtmc::LatentReplayToText>().run({{initial}, "", noises}); },
            TRTMC_INVALID_CONFIG, "family checks noise count against its actual sampling schedule");
    for (const bool with_initial : {false, true}) {
        for (const bool with_noise : {false, true}) {
            for (const std::string& prompt : {std::string{}, std::string{"preserved"}}) {
                const trtmc::Span<const float> start_values =
                    with_initial ? trtmc::Span<const float>{initial} : trtmc::Span<const float>{};
                const trtmc::Span<const float> noise_values =
                    with_noise ? noises : trtmc::Span<const float>{};
                const auto config = with_noise ? schedule : trtmc::Config{};
                auto result = model.task<trtmc::LatentConditionedTextGeneration>().run(
                    {{values}, {mask}, start_values, noise_values, prompt}, config);
                check(
                    result.text() == "conditioned" && result.token_ids()[0] == 1 &&
                        result.token_ids()[1] == 50 &&
                        result.token_ids()[2] == (with_initial ? 9 : -1) &&
                        result.token_ids()[3] == (with_noise ? 17 : -1) &&
                        result.token_ids()[5] == static_cast<std::int32_t>(prompt.size()),
                    "all raw-condition replay combinations preserve prompt and family precedence");
                if (with_initial || with_noise) {
                    auto plain = model.task<trtmc::LatentReplayToText>().run(
                        {start_values, prompt, noise_values}, config);
                    check(plain.text() == prompt + "|replayed" &&
                              plain.token_ids()[0] == (with_initial ? 9 : -1) &&
                              plain.token_ids()[1] == (with_noise ? 17 : -1),
                          "initial/noise/both work with either absent or actual text condition");
                }
            }
        }
    }
    const float zero_latents[] = {0, 0, 0, 0}, zero_mask[] = {0, 0};
    auto zero = model.task<trtmc::LatentConditionedTextGeneration>().run(
        {{zero_latents}, {zero_mask}, {zero_latents}, {}});
    check(zero.token_ids()[0] == 0 && zero.token_ids()[1] == 0 && zero.token_ids()[2] == 0,
          "nonempty zero-valued latent and mask buffers are real supplied data");
    rejects(
        [&] {
            (void)model.task<trtmc::LatentConditionedTextGeneration>().run({{}, {mask}, {}, {}});
        },
        TRTMC_INVALID_ARGUMENT, "mask without raw condition cannot execute");
    rejects(
        [&] {
            (void)model.task<trtmc::LatentConditionedTextGeneration>().run(
                {{values, 3}, {mask}, {}, {}});
        },
        TRTMC_INVALID_ARGUMENT, "family validates raw condition layout without shared inference");
    rejects(
        [&] {
            (void)model.task<trtmc::LatentConditionedTextGeneration>().run(
                {{values}, {mask, 1}, {}, {}});
        },
        TRTMC_INVALID_ARGUMENT, "family validates required mask length");
    rejects([&] { (void)model.task<trtmc::LatentReplayToText>().run({{initial, 3}, "", {}}); },
            TRTMC_INVALID_ARGUMENT, "family validates unshaped initial replay length");
}
template <class Function>
void rejects_item(Function function, trtmc_status status, size_t index, const char* label) {
    bool matched = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        matched = error.code() == status &&
                  std::string(error.what()).find("batch item[" + std::to_string(index) + "]") !=
                      std::string::npos;
    }
    check(matched, label);
}
void forecast_batches(const trtmc::Model& model, const std::filesystem::path& root) {
    using namespace trtmc;
    float first_values[] = {3, 4}, second_values[] = {8, 9, 10, 11};
    const uint8_t observed[] = {0, 1, 1, 1};
    const SeriesHistory first{{{first_values}, 2, 1}, {}},
        second{{{second_values}, 4, 1}, {observed}};
    const Config frequency{{"frequency", 2}};
    const BatchSeriesToPointForecastRequest input{{{{first}, {}}, {{second}, frequency}}};
    const auto point = model.task<BatchSeriesToPointForecast>();
    auto points = point.run(input);
    check(points.size() == 2 && points[0].values.rows == 2 && points[0].values.columns == 1 &&
              points[0].values.data[0] == 2003 && points[0].values.data[1] == 2015 &&
              points[1].values.data[0] == 2190 && points[1].values.data[1] == 2204,
          "batch point preserves ragged histories, per-item observed mask and I64 frequency");
    auto quantiles =
        model.task<BatchSeriesToQuantileForecast>().run({{{{first}, {}}, {{second}, frequency}}});
    check(quantiles.size() == 2 && quantiles[0].channels == 1 && quantiles[0].horizon == 2 &&
              quantiles[0].value_count == 6 && quantiles[0].quantile_levels.data[0] == 0.1 &&
              quantiles[0].quantile_levels.data[2] == 0.9 && quantiles[0].values[0] == 3001 &&
              quantiles[1].values[2] == 3195,
          "batch quantile preserves actual levels and per-item Q,H,C axes");
    auto joint = model.task<BatchSeriesToPointAndQuantileForecast>().run(
        {{{{first}, {}}, {{second}, frequency}}});
    check(joint.size() == 2 && joint[0].point.values.data[0] == 4003 &&
              joint[1].point.values.data[0] == 4190 && joint[1].quantiles.values[2] == 4195 &&
              joint[1].quantiles.values[3] == 4209 &&
              joint[1].point.axes.horizon_steps.data[1] ==
                  joint[1].quantiles.axes.horizon_steps.data[1],
          "joint batch returns matched outputs from one evaluation without median substitution");
    auto batch_median = trtmc::median_forecast(joint[1].quantiles);
    check(batch_median.values[0] == 4195 && joint[1].point.values.data[0] == 4190,
          "existing batch item view feeds the same helper with no extra runtime call");
    const float two_channel_values[] = {1, 2, 3, 4};
    const SeriesHistory two_channels{{{two_channel_values}, 2, 2}, {}};
    auto multivariate = point.run({{{{two_channels}, {}}}});
    check(multivariate.size() == 1 && multivariate[0].values.columns == 2,
          "two channels of one series are not silently converted into two batch items");
    auto invalid = input;
    invalid.items[1].config = {{"frequency", 1.5}};
    rejects_item([&] { point.run(invalid); }, TRTMC_INVALID_CONFIG, 1,
                 "family rejects a wrongly typed later frequency before execution");
    invalid.items[1].config = {{"frequency", 0}, {"frequency", 2}};
    rejects_item([&] { point.run(invalid); }, TRTMC_INVALID_CONFIG, 1,
                 "family rejects duplicate later-item config without dropping entries");
    invalid.items[1].config = {{"frequency", 3}};
    rejects_item([&] { point.run(invalid); }, TRTMC_INVALID_CONFIG, 1,
                 "category range remains provider validation");
    auto next = point.run(input);
    check(next[0].values.data[0] == 6003, "failed config preflight did not execute a family batch");
    invalid = input;
    invalid.items[1].input.history.observed = {observed, 1};
    rejects_item([&] { point.run(invalid); }, TRTMC_INVALID_ARGUMENT, 1,
                 "transport errors identify the malformed history item");
    rejects([&] { point.run({}); }, TRTMC_INVALID_ARGUMENT,
            "empty native forecast batch fails explicitly");
    rejects([&] { (void)joint[2]; }, TRTMC_INVALID_ARGUMENT,
            "forecast batch item view validates its index");
    second_values[0] = 999;
    check(joint[1].point.values.data[0] == 4190,
          "batch forecast results do not borrow input values");
    auto moved = std::move(joint);
    check(joint.size() == 0 && moved.size() == 2,
          "batch result move transfers ownership and clears source count");
    const float nan_values[] = {std::numeric_limits<float>::quiet_NaN(), 2};
    const SeriesHistory missing{{{nan_values}, 2, 1}, {}};
    auto nan = model.task<BatchSeriesToQuantileForecast>().run({{{{missing}, {}}}});
    check(nan[0].values[0] == 6978,
          "NaN missingness reaches family rather than being sanitized by shared transport");

    LoadOptions options;
    options.runtime_root = root.string();
    auto load = [&](const std::string& mode) {
        const auto path = root / ("numeric-forecast-" + mode + ".bundle");
        bundle(path, mode);
        return Model::load(path.string(), options);
    };
    auto custom = load("frequency_default_one").task<BatchSeriesToPointForecast>();
    auto custom_values = custom.run({{{{first}, {}}, {{first}, {{"frequency", 0}}}}});
    check(custom.config_fields()[0].default_value->get<std::int64_t>() == 1 &&
              custom_values[0].values.data[0] == 1103 && custom_values[1].values.data[0] == 1003,
          "per-item missing frequency uses family default while explicit zero passes through");
    auto neutral = load("neutral_frequency").task<BatchSeriesToPointForecast>();
    check(neutral.run({{{{first}, {}}}})[0].values.data[0] == 1003,
          "provider accepting only neutral frequency accepts omission");
    rejects_item([&] { neutral.run({{{{first}, {}}, {{first}, {{"frequency", 1}}}}}); },
                 TRTMC_INVALID_CONFIG, 1,
                 "another provider can reject nonzero frequency without shared category policy");
    auto bad_count = load("broken_batch_count").task<BatchSeriesToPointForecast>();
    rejects([&] { bad_count.run(input); }, TRTMC_INTERNAL_ERROR,
            "batch must return one forecast per input");
    for (const std::string mode : {"broken_joint_axes", "broken_joint_shape"}) {
        auto bad = load(mode).task<SeriesToPointAndQuantileForecast>();
        rejects([&] { bad.run({first}); }, TRTMC_INTERNAL_ERROR,
                "joint component shape/axis disagreement fails explicitly");
    }
    auto bad_axes = load("broken_batch_axes");
    rejects_item(
        [&] {
            bad_axes.task<BatchSeriesToPointAndQuantileForecast>().run(
                {{{{first}, {}}, {{first}, {}}}});
        },
        TRTMC_INTERNAL_ERROR, 1,
        "later-item packing error returns no partial batch and identifies its item");
    auto after_pack = bad_axes.task<BatchSeriesToPointForecast>().run({{{{first}, {}}}});
    check(after_pack[0].values.data[0] == 2003,
          "failed result packing does not pretend to roll back family evaluation effects");
    auto retained = [&] {
        auto owner = load("all_numeric");
        return owner.task<BatchSeriesToPointAndQuantileForecast>().run({{{{first}, {}}}});
    }();
    check(retained[0].point.values.data[0] == 1003 && retained[0].quantiles.values[2] == 1008,
          "owned joint batch views survive local model scope");
    auto flat_model = load("flat_channels_two");
    const auto flat = SeriesHistory::from_flat({two_channel_values});
    auto flat_point = flat_model.task<SeriesToPointForecast>().run({flat});
    check(flat_point.view().values.columns == 2 && flat_point.view().values.count == 4,
          "family resolves a nonempty flat history using its own bundle channel count");
    const float longer_values[] = {1, 2, 3, 4, 5, 6};
    const auto longer_flat = SeriesHistory::from_flat({longer_values});
    auto flat_batch =
        flat_model.task<BatchSeriesToPointForecast>().run({{{{flat}, {}}, {{longer_flat}, {}}}});
    check(flat_batch.size() == 2 && flat_batch[0].values.columns == 2 &&
              flat_batch[1].values.columns == 2 && flat_batch[0].values.data[3] == 1014 &&
              flat_batch[1].values.data[3] == 1015,
          "family-owned flat shape resolution preserves ragged lengths without turning channels "
          "into batch");
    const auto indivisible = SeriesHistory::from_flat({longer_values, 3});
    rejects([&] { flat_model.task<SeriesToPointForecast>().run({indivisible}); },
            TRTMC_INVALID_ARGUMENT, "family rejects flat data not divisible by its channel count");
    rejects_item(
        [&] {
            flat_model.task<BatchSeriesToPointForecast>().run(
                {{{{flat}, {}}, {{indivisible}, {}}}});
        },
        TRTMC_INVALID_ARGUMENT, 1, "later flat shape error is indexed during family preflight");
    auto after_shape_error = flat_model.task<BatchSeriesToPointForecast>().run({{{{flat}, {}}}});
    check(after_shape_error[0].values.data[0] == 2001,
          "flat shape preflight fails before family batch execution");
    const SeriesHistory conflicting{{{two_channel_values}, 1, 4}, {}};
    rejects([&] { flat_model.task<SeriesToPointForecast>().run({conflicting}); },
            TRTMC_INVALID_ARGUMENT,
            "explicit shape is preserved and rejected by family instead of being reinterpreted");
    const SeriesHistory partial{{{two_channel_values}, 0, 2}, {}};
    rejects([&] { flat_model.task<SeriesToPointForecast>().run({partial}); },
            TRTMC_INVALID_ARGUMENT,
            "partially specified history shape is not treated as a flat default");
    const uint8_t short_mask[] = {1};
    rejects(
        [&] {
            flat_model.task<SeriesToPointForecast>().run(
                {SeriesHistory::from_flat({two_channel_values}, {short_mask})});
        },
        TRTMC_INVALID_ARGUMENT, "flat history mask still matches the supplied element count");
    auto explicit_only = load("flat_shape_unsupported").task<SeriesToPointForecast>();
    rejects([&] { explicit_only.run({flat}); }, TRTMC_INVALID_ARGUMENT,
            "a provider may reject flat input when it has no shape default");
    rejects([&] { flat_model.task<SeriesToPointForecast>().run({SeriesHistory::from_flat({})}); },
            TRTMC_INVALID_ARGUMENT, "empty flat history is not a meaningful default");
    rejects([&] { model.task<LatentDenoisingStep>().run({{{two_channel_values}, 0, 2}, {}, 0.0}); },
            TRTMC_INVALID_ARGUMENT, "partially specified latent matrix dimensions remain invalid");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        const auto full = root / "numeric-cpp-all.bundle",
                   restricted = root / "numeric-cpp-regression.bundle",
                   broken_q = root / "numeric-cpp-bad-q.bundle",
                   broken_r = root / "numeric-cpp-bad-r.bundle";
        bundle(full, "all_numeric");
        bundle(restricted, "regression_only");
        bundle(broken_q, "broken_quantile");
        bundle(broken_r, "broken_regression");
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto model = trtmc::Model::load(full.string(), options);
        regression_values(root, options);
        const float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        const std::uint8_t mask[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        const trtmc::SeriesHistory history{{{values}, 4, 3}, {mask}};
        check(model.tasks().size() == 11, "all eleven numeric contracts are discoverable");
        forecasts(model, history);
        forecast_batches(model, root);
        latents(model);
        auto regression = trtmc::Model::load(restricted.string(), options);
        check(regression.supports<trtmc::SeriesToRegressionDistribution>() &&
                  !regression.supports<trtmc::SeriesToPointForecast>(),
              "model availability is narrower than its C++ inheritance");
        rejects(
            [&] {
                (void)trtmc::Model::load(broken_q.string(), options)
                    .task<trtmc::SeriesToQuantileForecast>()
                    .run({history});
            },
            TRTMC_INTERNAL_ERROR, "incorrect family quantile metadata is rejected");
        rejects(
            [&] {
                (void)trtmc::Model::load(broken_r.string(), options)
                    .task<trtmc::SeriesToRegressionDistribution>()
                    .run({history});
            },
            TRTMC_INTERNAL_ERROR, "regression parameters cannot be mislabeled as horizon");
        auto retained = [&] {
            auto temporary = trtmc::Model::load(full.string(), options);
            return temporary.task<trtmc::SeriesToQuantileForecast>().run({history});
        }();
        check(retained.view().quantile_levels.data[1] == 0.9 && retained.view().values[11] == 12,
              "numeric outputs survive caller model scope");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
