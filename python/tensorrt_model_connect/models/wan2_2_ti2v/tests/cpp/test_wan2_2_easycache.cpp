/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config_schema.h"
#include "easycache.h"
#include "runtime_config.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/schema_registry.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::wan2_2_ti2v::EasyCacheConfig;
using trtmc::wan2_2_ti2v::EasyCacheController;
using trtmc::wan2_2_ti2v::LateCfgAction;
using trtmc::wan2_2_ti2v::LateCfgController;

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

bool close(double left, double right, double tolerance = 1.0e-6) {
    return std::abs(left - right) <= tolerance;
}

template <typename Callable>
void check_throws(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::invalid_argument&) {
    }
}

EasyCacheConfig qualified_config() {
    EasyCacheConfig config;
    config.enabled = true;
    config.threshold = trtmc::wan2_2_ti2v::kQualifiedEasyCacheThreshold;
    config.first_exact_steps = trtmc::wan2_2_ti2v::kQualifiedEasyCacheFirstExactSteps;
    config.last_exact_steps = trtmc::wan2_2_ti2v::kQualifiedEasyCacheLastExactSteps;
    config.max_consecutive_reuse = trtmc::wan2_2_ti2v::kQualifiedEasyCacheMaxConsecutiveReuse;
    config.total_steps = trtmc::wan2_2_ti2v::kQualifiedEasyCacheTotalSteps;
    return config;
}

EasyCacheConfig thor_performance_config() {
    EasyCacheConfig config;
    config.enabled = true;
    config.threshold = trtmc::wan2_2_ti2v::kThorPerformanceEasyCacheThreshold;
    config.first_exact_steps = trtmc::wan2_2_ti2v::kThorPerformanceEasyCacheFirstExactSteps;
    config.last_exact_steps = trtmc::wan2_2_ti2v::kThorPerformanceEasyCacheLastExactSteps;
    config.max_consecutive_reuse = trtmc::wan2_2_ti2v::kThorPerformanceEasyCacheMaxConsecutiveReuse;
    config.total_steps = trtmc::wan2_2_ti2v::kThorPerformanceEasyCacheTotalSteps;
    return config;
}

trtmc::wan2_2_ti2v::EasyCacheRuntimeProfile qualified_thor_runtime() {
    trtmc::wan2_2_ti2v::EasyCacheRuntimeProfile runtime;
    runtime.video_height = 704;
    runtime.video_width = 1280;
    runtime.video_frames = 121;
    runtime.guidance_scale = 5.0F;
    runtime.integrated_gpu = true;
    runtime.compute_capability_major = 11;
    runtime.compute_capability_minor = 0;
    return runtime;
}

void register_runtime_schema() {
    auto& registry = trtmc::config::SchemaRegistry::instance();
    registry.clear_for_testing();
    registry.register_schema(trtmc::config::schemas::make_wan2_2_ti2v_schema());
}

trtmc::wan2_2_ti2v::RuntimeConfig
resolve_settings(const std::vector<std::string>& set_tokens = {}) {
    const auto bundle = trtmc::config::resolve_cli_config("", set_tokens);
    return trtmc::wan2_2_ti2v::resolve_runtime_config(&bundle);
}

void test_declarative_defaults_are_inert() {
    register_runtime_schema();
    const auto settings = resolve_settings();
    check(!settings.easycache.enabled, "EasyCache is disabled by default");
    check(close(settings.easycache.threshold, 0.02), "EasyCache threshold default is preserved");
    check(settings.easycache.first_exact_steps == 7 && settings.easycache.last_exact_steps == 2 &&
              settings.easycache.max_consecutive_reuse == 1,
          "EasyCache integer defaults are preserved");
    check(!trtmc::wan2_2_ti2v::validate_late_cfg_request(settings.late_cfg_enabled,
                                                         settings.easycache),
          "late-CFG is disabled by default");
}

void test_declarative_values_reach_runtime_api() {
    register_runtime_schema();
    const auto settings = resolve_settings({
        "wan2_2_ti2v.easycache_enabled=true",
        "wan2_2_ti2v.easycache_threshold=0.08",
        "wan2_2_ti2v.easycache_first_exact_steps=7",
        "wan2_2_ti2v.easycache_last_exact_steps=2",
        "wan2_2_ti2v.easycache_max_consecutive_reuse=4",
        "wan2_2_ti2v.late_cfg_enabled=true",
    });
    auto parsed = settings.easycache;
    parsed.total_steps = 50;
    check(settings.late_cfg_enabled, "late-CFG declarative value reaches the runtime API");
    check(parsed.enabled && close(parsed.threshold, 0.08) && parsed.first_exact_steps == 7 &&
              parsed.last_exact_steps == 2 && parsed.max_consecutive_reuse == 4,
          "EasyCache declarative values reach the runtime API");
    check(trtmc::wan2_2_ti2v::validate_late_cfg_request(settings.late_cfg_enabled, parsed),
          "late-CFG accepts the qualified EasyCache profile");

    auto invalid = qualified_config();
    invalid.enabled = false;
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, invalid); },
                 "late-CFG rejects disabled EasyCache");
    invalid = qualified_config();
    invalid.threshold = 0.05;
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, invalid); },
                 "late-CFG rejects an unqualified threshold");
    invalid = qualified_config();
    invalid.first_exact_steps = 8;
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, invalid); },
                 "late-CFG rejects an unqualified prefix");
    invalid = qualified_config();
    invalid.last_exact_steps = 1;
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, invalid); },
                 "late-CFG rejects an unqualified suffix");
    invalid = qualified_config();
    invalid.max_consecutive_reuse = 3;
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, invalid); },
                 "late-CFG rejects an unqualified reuse cap");
    invalid = qualified_config();
    invalid.total_steps = 49;
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, invalid); },
                 "late-CFG rejects an unqualified step count");
}

void test_thor_performance_profile_scope() {
    const auto config = thor_performance_config();
    const auto runtime = qualified_thor_runtime();

    check(trtmc::wan2_2_ti2v::is_thor_performance_easycache_config(config),
          "Thor performance EasyCache config is exact");
    check(trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config, runtime),
          "Thor performance profile accepts official integrated SM110");
    check(trtmc::wan2_2_ti2v::validate_late_cfg_request(true, config, true),
          "late-CFG accepts runtime-qualified Thor performance profile");
    check_throws([&] { (void)trtmc::wan2_2_ti2v::validate_late_cfg_request(true, config, false); },
                 "late-CFG rejects an unqualified Thor performance profile");

    auto invalid_runtime = runtime;
    invalid_runtime.video_height = 384;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects L0 height");
    invalid_runtime = runtime;
    invalid_runtime.video_width = 672;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects L0 width");
    invalid_runtime = runtime;
    invalid_runtime.video_frames = 5;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects L0 frame count");
    invalid_runtime = runtime;
    invalid_runtime.guidance_scale = 4.0F;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects non-qualified CFG");
    invalid_runtime = runtime;
    invalid_runtime.integrated_gpu = false;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects discrete SM110");
    invalid_runtime = runtime;
    invalid_runtime.compute_capability_major = 10;
    invalid_runtime.compute_capability_minor = 3;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects SM103");
    invalid_runtime = runtime;
    invalid_runtime.compute_capability_minor = 1;
    check(!trtmc::wan2_2_ti2v::is_qualified_thor_performance_easycache_profile(config,
                                                                               invalid_runtime),
          "Thor performance profile rejects non-Thor SM111");

    auto invalid_config = config;
    invalid_config.max_consecutive_reuse = 5;
    check(!trtmc::wan2_2_ti2v::is_thor_performance_easycache_config(invalid_config),
          "Thor performance profile rejects config drift");
    invalid_config = config;
    invalid_config.total_steps = 49;
    check(!trtmc::wan2_2_ti2v::is_thor_performance_easycache_config(invalid_config),
          "Thor performance profile rejects non-qualified step count");
}

void test_schema_rejects_invalid_values() {
    register_runtime_schema();
    check_throws([] { (void)resolve_settings({"wan2_2_ti2v.easycache_threshold=0"}); },
                 "schema rejects a non-positive threshold");
    check_throws([] { (void)resolve_settings({"wan2_2_ti2v.easycache_first_exact_steps=-1"}); },
                 "schema rejects a negative exact-step count");
    check_throws([] { (void)resolve_settings({"wan2_2_ti2v.easycache_max_consecutive_reuse=0"}); },
                 "schema rejects a zero reuse cap");
}

void test_easycache_exact_and_reuse_paths() {
    EasyCacheConfig config;
    config.enabled = true;
    config.threshold = 1.0;
    config.first_exact_steps = 1;
    config.last_exact_steps = 1;
    config.max_consecutive_reuse = 2;
    config.total_steps = 6;
    EasyCacheController controller(config);

    check(controller.decide(0, {0.0F}), "EasyCache computes the exact prefix");
    controller.update_conditional({0.0F}, {1.0F});
    controller.update_unconditional({0.0F}, {-1.0F});
    check(controller.decide(1, {1.0F}), "EasyCache initializes its change factor");
    controller.update_conditional({1.0F}, {2.0F});
    controller.update_unconditional({1.0F}, {0.0F});

    check(!controller.decide(2, {1.01F}), "EasyCache reuses below threshold");
    const auto conditional = controller.reuse_conditional({1.01F});
    const auto unconditional = controller.reuse_unconditional({1.01F});
    check(close(conditional[0], 2.01) && close(unconditional[0], 0.01),
          "EasyCache keeps branch-local residuals");
    check(!controller.decide(3, {1.02F}), "EasyCache permits the qualified reuse sequence");
    check(controller.decide(4, {1.03F}), "EasyCache refreshes at the reuse cap");
    controller.update_conditional({1.03F}, {2.03F});
    controller.update_unconditional({1.03F}, {0.03F});
    check(controller.decide(5, {1.04F}), "EasyCache computes the exact suffix");
    controller.update_conditional({1.04F}, {2.04F});
    controller.update_unconditional({1.04F}, {0.04F});

    check(controller.stats().compute_steps == 4 && controller.stats().reuse_steps == 2,
          "EasyCache reports compute and reuse accounting");
}

void record_simple_actual(LateCfgController& controller, int32_t step) {
    const float conditional = static_cast<float>(step + 1);
    controller.record_actual({conditional}, {conditional - 1.0F}, 5.0);
}

void test_late_cfg_exact_windows_cadence_and_prediction() {
    LateCfgController controller;
    for (int32_t step = 0; step < 50; ++step) {
        const bool compute = step <= 21 || step == 47 || step >= 48;
        const auto action = controller.decide(step, 1000 - 10 * step, compute);
        if (!compute) {
            check(action == LateCfgAction::kEasyCacheReuse,
                  "late-CFG ignores EasyCache reuse events");
            continue;
        }
        if (step == 21) {
            check(action == LateCfgAction::kPredictUnconditional,
                  "late-CFG predicts every second late compute event");
            const auto prediction = controller.try_predict({30.0F, 12.0F}, 5.0);
            check(prediction.has_value(), "late-CFG produces a bounded prediction");
            check(close(prediction->guided[0], 46.0) && close(prediction->guided[1], 24.0) &&
                      close(prediction->synthetic_unconditional[0], 26.0) &&
                      close(prediction->synthetic_unconditional[1], 9.0),
                  "late-CFG preserves guided and synthetic-unconditional algebra");
            continue;
        }

        check(action == LateCfgAction::kActualUnconditional,
              "late-CFG refreshes exact windows and cadence anchors");
        if (step == 19)
            controller.record_actual({10.0F, 4.0F}, {8.0F, 3.0F}, 5.0);
        else if (step == 20)
            controller.record_actual({20.0F, 8.0F}, {17.0F, 6.0F}, 5.0);
        else
            record_simple_actual(controller, step);
    }

    const auto& stats = controller.stats();
    check(stats.processed_steps == 50 && stats.easycache_compute_events == 25 &&
              stats.easycache_reuse_events == 25 && stats.actual_unconditional_calls == 24 &&
              stats.predicted_unconditional_reuses == 1 && stats.prediction_fallbacks == 0,
          "late-CFG reports exact interval-2 accounting");
}

void test_prediction_falls_back_to_actual() {
    LateCfgController controller;
    for (int32_t step = 0; step <= 22; ++step) {
        const bool compute = step == 0 || step == 19 || step == 20 || step == 21;
        int64_t timestep = 1000 - step;
        if (step == 19)
            timestep = 100;
        if (step == 20)
            timestep = 99;
        if (step == 21)
            timestep = 0;
        const auto action = controller.decide(step, timestep, compute);
        if (!compute) {
            check(action == LateCfgAction::kEasyCacheReuse,
                  "late-CFG continues after prediction fallback");
            continue;
        }
        if (step == 21) {
            check(action == LateCfgAction::kPredictUnconditional,
                  "late-CFG requests its interval-2 prediction");
            check(!controller.try_predict({4.0F}, 5.0).has_value(),
                  "late-CFG rejects an unbounded timestep extrapolation");
            controller.record_actual({4.0F}, {3.0F}, 5.0);
        } else {
            record_simple_actual(controller, step);
        }
    }
    const auto& stats = controller.stats();
    check(stats.actual_unconditional_calls == 4 && stats.predicted_unconditional_reuses == 0 &&
              stats.prediction_fallbacks == 1,
          "late-CFG accounts for an actual-unconditional fallback");
}

void test_synthetic_unconditional_updates_easycache() {
    LateCfgController late_cfg;
    std::optional<trtmc::wan2_2_ti2v::LateCfgPrediction> prediction;
    for (int32_t step = 0; step <= 21; ++step) {
        const bool compute = step == 0 || step == 19 || step == 20 || step == 21;
        const auto action = late_cfg.decide(step, 1000 - 10 * step, compute);
        if (!compute)
            continue;
        if (step == 19)
            late_cfg.record_actual({1.0F}, {0.0F}, 5.0);
        else if (step == 20)
            late_cfg.record_actual({1.2F}, {0.1F}, 5.0);
        else if (action == LateCfgAction::kPredictUnconditional)
            prediction = late_cfg.try_predict({1.3F}, 5.0);
        else
            record_simple_actual(late_cfg, step);
    }
    check(prediction.has_value(), "late-CFG produces the stacked EasyCache prediction");
    if (!prediction.has_value())
        return;

    EasyCacheConfig config;
    config.enabled = true;
    config.threshold = 0.01;
    config.first_exact_steps = 2;
    config.last_exact_steps = 0;
    config.max_consecutive_reuse = 4;
    config.total_steps = 4;
    EasyCacheController easycache(config);

    (void)easycache.decide(0, {0.0F});
    easycache.update_conditional({0.0F}, {1.0F});
    easycache.update_unconditional({0.0F}, {0.0F});
    (void)easycache.decide(1, {0.1F});
    easycache.update_conditional({0.1F}, {1.2F});
    easycache.update_unconditional({0.1F}, {0.1F});
    check(easycache.decide(2, {0.11F}), "EasyCache reaches a stacked compute event");
    easycache.update_conditional({0.11F}, {1.3F});
    easycache.update_unconditional({0.11F}, prediction->synthetic_unconditional);

    check(!easycache.decide(3, {0.111F}), "EasyCache reuses after the stacked prediction");
    const auto conditional = easycache.reuse_conditional({0.111F});
    const auto unconditional = easycache.reuse_unconditional({0.111F});
    const double guided = unconditional[0] + 5.0 * (conditional[0] - unconditional[0]);
    const double predicted_residual = prediction->guided[0] - 1.3;
    check(close(guided, conditional[0] + predicted_residual, 1.0e-5),
          "synthetic unconditional preserves the predicted guidance residual");
}

} // namespace

int main() {
    test_declarative_defaults_are_inert();
    test_declarative_values_reach_runtime_api();
    test_thor_performance_profile_scope();
    test_schema_rejects_invalid_values();
    test_easycache_exact_and_reuse_paths();
    test_late_cfg_exact_windows_cadence_and_prediction();
    test_prediction_falls_back_to_actual();
    test_synthetic_unconditional_updates_easycache();
    return failures == 0 ? 0 : 1;
}
