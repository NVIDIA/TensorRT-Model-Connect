/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/easycache.h"

#include "runtime/models/wan2_2_ti2v/options.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc::wan2_2_ti2v {
namespace {

constexpr const char* kEnableEnvironment = "TRTMC_WAN22_EASYCACHE";
constexpr const char* kThresholdEnvironment = "TRTMC_WAN22_EASYCACHE_THRESHOLD";
constexpr const char* kFirstExactEnvironment = "TRTMC_WAN22_EASYCACHE_FIRST_EXACT_STEPS";
constexpr const char* kLastExactEnvironment = "TRTMC_WAN22_EASYCACHE_LAST_EXACT_STEPS";
constexpr const char* kMaxConsecutiveReuseEnvironment =
    "TRTMC_WAN22_EASYCACHE_MAX_CONSECUTIVE_REUSE";
constexpr const char* kLateCfgEnableEnvironment = "TRTMC_WAN22_EASYCACHE_LATE_CFG";
constexpr double kMinimumNorm = 1.0e-8;
constexpr double kMaximumTimestepExtrapolation = 2.0;

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool parse_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return false;
    const auto normalized = lowercase(value);
    constexpr std::array<std::string_view, 4> enabled = {"1", "true", "yes", "on"};
    constexpr std::array<std::string_view, 4> disabled = {"0", "false", "no", "off"};
    if (std::find(enabled.begin(), enabled.end(), normalized) != enabled.end())
        return true;
    if (std::find(disabled.begin(), disabled.end(), normalized) != disabled.end())
        return false;
    throw std::invalid_argument(std::string(name) +
                                " must be one of 0/1, false/true, no/yes, or off/on");
}

double parse_positive_double(const char* name, double default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be a finite positive number");
    return parsed;
}

int32_t parse_nonnegative_int(const char* name, int32_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return default_value;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    return static_cast<int32_t>(parsed);
}

bool exact_windows_are_valid(const EasyCacheConfig& config) noexcept {
    return config.first_exact_steps >= 0 && config.last_exact_steps >= 0 &&
           config.first_exact_steps <= config.total_steps &&
           config.last_exact_steps <= config.total_steps - config.first_exact_steps;
}

void validate_config(const EasyCacheConfig& config) {
    if (!config.enabled)
        throw std::invalid_argument("EasyCacheController requires an enabled configuration");
    if (!std::isfinite(config.threshold) || config.threshold <= 0.0)
        throw std::invalid_argument("Wan2.2 EasyCache threshold must be finite and positive");
    if (config.total_steps <= 0)
        throw std::invalid_argument("Wan2.2 EasyCache total_steps must be positive");
    if (!exact_windows_are_valid(config))
        throw std::invalid_argument("Wan2.2 EasyCache exact-step windows must not overlap");
    if (config.max_consecutive_reuse <= 0) {
        throw std::invalid_argument(
            "Wan2.2 EasyCache max_consecutive_reuse must be a positive integer");
    }
}

bool is_qualified_conservative_late_cfg_profile(const EasyCacheConfig& config) noexcept {
    return config.enabled && config.threshold == kQualifiedEasyCacheThreshold &&
           config.first_exact_steps == kQualifiedEasyCacheFirstExactSteps &&
           config.last_exact_steps == kQualifiedEasyCacheLastExactSteps &&
           config.max_consecutive_reuse == kQualifiedEasyCacheMaxConsecutiveReuse &&
           config.total_steps == kQualifiedEasyCacheTotalSteps;
}

} // namespace

EasyCacheConfig easycache_config_from_environment(int32_t total_steps) {
    EasyCacheConfig config;
    config.total_steps = total_steps;
    config.enabled = parse_enabled(kEnableEnvironment);
    if (!config.enabled)
        return config;

    config.threshold = parse_positive_double(kThresholdEnvironment, config.threshold);
    config.first_exact_steps =
        parse_nonnegative_int(kFirstExactEnvironment, config.first_exact_steps);
    config.last_exact_steps = parse_nonnegative_int(kLastExactEnvironment, config.last_exact_steps);
    config.max_consecutive_reuse =
        parse_nonnegative_int(kMaxConsecutiveReuseEnvironment, config.max_consecutive_reuse);
    validate_config(config);
    return config;
}

bool is_thor_performance_easycache_config(const EasyCacheConfig& easycache) noexcept {
    return easycache.enabled && easycache.threshold == kThorPerformanceEasyCacheThreshold &&
           easycache.first_exact_steps == kThorPerformanceEasyCacheFirstExactSteps &&
           easycache.last_exact_steps == kThorPerformanceEasyCacheLastExactSteps &&
           easycache.max_consecutive_reuse == kThorPerformanceEasyCacheMaxConsecutiveReuse &&
           easycache.total_steps == kThorPerformanceEasyCacheTotalSteps;
}

bool is_qualified_thor_performance_easycache_profile(
    const EasyCacheConfig& easycache, const EasyCacheRuntimeProfile& runtime) noexcept {
    return is_thor_performance_easycache_config(easycache) &&
           runtime.video_height == trtmc::kWan22OfficialVideoHeight &&
           runtime.video_width == trtmc::kWan22OfficialVideoWidth &&
           runtime.video_frames == trtmc::kWan22OfficialVideoFrames &&
           runtime.guidance_scale == trtmc::kWan22OfficialGuidanceScale && runtime.integrated_gpu &&
           runtime.compute_capability_major == 11 && runtime.compute_capability_minor == 0;
}

bool late_cfg_enabled_from_environment(const EasyCacheConfig& easycache,
                                       bool thor_performance_profile_qualified) {
    if (!parse_enabled(kLateCfgEnableEnvironment))
        return false;
    if (!is_qualified_conservative_late_cfg_profile(easycache) &&
        !(thor_performance_profile_qualified && is_thor_performance_easycache_config(easycache))) {
        throw std::invalid_argument(
            "TRTMC_WAN22_EASYCACHE_LATE_CFG requires either the conservative 50-step "
            "EasyCache profile (threshold=0.08, first_exact_steps=7, last_exact_steps=2, "
            "max_consecutive_reuse=4) or its runtime-qualified Thor performance profile");
    }
    return true;
}

EasyCacheController::EasyCacheController(EasyCacheConfig config) : config_(config) {
    validate_config(config_);
    stats_.total_steps = config_.total_steps;
}

bool EasyCacheController::decide(int32_t step, const std::vector<float>& raw_latents) {
    validate_decision_input(step, raw_latents);
    const bool compute = choose_compute(step, raw_latents);
    finish_decision(compute, raw_latents);
    return compute;
}

void EasyCacheController::validate_decision_input(int32_t step,
                                                  const std::vector<float>& raw_latents) const {
    if (step != stats_.compute_steps + stats_.reuse_steps || step < 0 ||
        step >= config_.total_steps) {
        throw std::invalid_argument("Wan2.2 EasyCache steps must be consecutive and in range");
    }
    if (raw_latents.empty())
        throw std::invalid_argument("Wan2.2 EasyCache raw latent input must not be empty");
    if (!previous_step_input_.empty() && previous_step_input_.size() != raw_latents.size())
        throw std::invalid_argument("Wan2.2 EasyCache latent shape changed during generation");
}

bool EasyCacheController::choose_compute(int32_t step, const std::vector<float>& raw_latents) {
    if (in_exact_window(step) || !has_reusable_state() ||
        consecutive_reuse_ >= config_.max_consecutive_reuse) {
        accumulator_ = 0.0;
        return true;
    }
    const double input_change = mean_abs_delta(raw_latents, previous_step_input_);
    const double output_norm = std::max(mean_abs(last_full_output_), kMinimumNorm);
    accumulator_ += *change_factor_ * input_change / output_norm;
    if (accumulator_ < config_.threshold)
        return false;
    accumulator_ = 0.0;
    return true;
}

bool EasyCacheController::in_exact_window(int32_t step) const noexcept {
    return step < config_.first_exact_steps ||
           step >= config_.total_steps - config_.last_exact_steps;
}

bool EasyCacheController::has_reusable_state() const noexcept {
    return !previous_step_input_.empty() && !conditional_residual_.empty() &&
           !unconditional_residual_.empty() && change_factor_.has_value();
}

void EasyCacheController::finish_decision(bool compute, const std::vector<float>& raw_latents) {
    previous_step_input_ = raw_latents;
    last_decision_compute_ = compute;
    if (compute) {
        consecutive_reuse_ = 0;
        ++stats_.compute_steps;
        return;
    }
    ++consecutive_reuse_;
    ++stats_.reuse_steps;
}

void EasyCacheController::update_conditional(const std::vector<float>& raw_latents,
                                             const std::vector<float>& denoiser_output) {
    if (!last_decision_compute_)
        throw std::logic_error("Wan2.2 EasyCache cannot refresh after a reuse decision");
    require_same_nonempty_size(raw_latents, denoiser_output, "conditional refresh");
    if (!last_full_input_.empty()) {
        const double input_change =
            std::max(mean_abs_delta(raw_latents, last_full_input_), kMinimumNorm);
        const double output_change = mean_abs_delta(denoiser_output, last_full_output_);
        change_factor_ = output_change / input_change;
    }
    last_full_input_ = raw_latents;
    last_full_output_ = denoiser_output;
    conditional_residual_ = make_residual(denoiser_output, raw_latents);
}

void EasyCacheController::update_unconditional(const std::vector<float>& raw_latents,
                                               const std::vector<float>& denoiser_output) {
    if (!last_decision_compute_)
        throw std::logic_error("Wan2.2 EasyCache cannot refresh after a reuse decision");
    require_same_nonempty_size(raw_latents, denoiser_output, "unconditional refresh");
    unconditional_residual_ = make_residual(denoiser_output, raw_latents);
}

std::vector<float>
EasyCacheController::reuse_conditional(const std::vector<float>& raw_latents) const {
    if (last_decision_compute_ || conditional_residual_.empty())
        throw std::logic_error("Wan2.2 EasyCache conditional residual is not reusable");
    return add_residual(raw_latents, conditional_residual_);
}

std::vector<float>
EasyCacheController::reuse_unconditional(const std::vector<float>& raw_latents) const {
    if (last_decision_compute_ || unconditional_residual_.empty())
        throw std::logic_error("Wan2.2 EasyCache unconditional residual is not reusable");
    return add_residual(raw_latents, unconditional_residual_);
}

double EasyCacheController::mean_abs(const std::vector<float>& values) {
    if (values.empty())
        throw std::invalid_argument("Wan2.2 EasyCache reduction requires a non-empty tensor");
    double sum = 0.0;
    for (const float value : values)
        sum += std::abs(static_cast<double>(value));
    return sum / static_cast<double>(values.size());
}

double EasyCacheController::mean_abs_delta(const std::vector<float>& left,
                                           const std::vector<float>& right) {
    require_same_nonempty_size(left, right, "mean absolute delta");
    double sum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index)
        sum += std::abs(static_cast<double>(left[index]) - static_cast<double>(right[index]));
    return sum / static_cast<double>(left.size());
}

std::vector<float> EasyCacheController::make_residual(const std::vector<float>& output,
                                                      const std::vector<float>& input) {
    require_same_nonempty_size(output, input, "residual update");
    std::vector<float> residual(output.size());
    for (std::size_t index = 0; index < output.size(); ++index)
        residual[index] = output[index] - input[index];
    return residual;
}

std::vector<float> EasyCacheController::add_residual(const std::vector<float>& input,
                                                     const std::vector<float>& residual) {
    require_same_nonempty_size(input, residual, "residual reuse");
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
        output[index] = input[index] + residual[index];
    return output;
}

void EasyCacheController::require_same_nonempty_size(const std::vector<float>& left,
                                                     const std::vector<float>& right,
                                                     const char* operation) {
    if (left.empty() || left.size() != right.size()) {
        throw std::invalid_argument(std::string("Wan2.2 EasyCache ") + operation +
                                    " requires equal non-empty tensors");
    }
}

LateCfgController::LateCfgController() {
    stats_.total_steps = kQualifiedEasyCacheTotalSteps;
}

LateCfgAction LateCfgController::decide(int32_t step, int64_t timestep, bool easycache_compute) {
    if (actual_pending_ || prediction_pending_)
        throw std::logic_error("Wan2.2 late-CFG result was not consumed");
    if (step != stats_.processed_steps || step < 0 || step >= stats_.total_steps)
        throw std::invalid_argument("Wan2.2 late-CFG steps must be consecutive and in range");

    ++stats_.processed_steps;
    if (!easycache_compute) {
        ++stats_.easycache_reuse_events;
        return LateCfgAction::kEasyCacheReuse;
    }

    ++stats_.easycache_compute_events;
    pending_timestep_ = timestep;
    if (in_exact_window(step))
        return request_actual();

    const int32_t event = late_compute_event_++;
    if (!has_predictable_history() || event % 2 == 0)
        return request_actual();

    prediction_pending_ = true;
    return LateCfgAction::kPredictUnconditional;
}

bool LateCfgController::in_exact_window(int32_t step) const noexcept {
    constexpr int32_t first_exact_steps = 20;
    constexpr int32_t last_exact_steps = 2;
    return step < first_exact_steps || step >= stats_.total_steps - last_exact_steps;
}

bool LateCfgController::has_predictable_history() const noexcept {
    return actual_history_size_ == 2 && latest_actual_timestep_ != older_actual_timestep_;
}

LateCfgAction LateCfgController::request_actual() {
    actual_pending_ = true;
    return LateCfgAction::kActualUnconditional;
}

void LateCfgController::record_actual(const std::vector<float>& conditional,
                                      const std::vector<float>& unconditional,
                                      double guidance_scale) {
    if (!actual_pending_ && !prediction_pending_)
        throw std::logic_error("Wan2.2 late-CFG has no actual unconditional result to record");
    require_same_nonempty_size(conditional, unconditional, "actual refresh");
    if (!std::isfinite(guidance_scale))
        throw std::invalid_argument("Wan2.2 late-CFG guidance scale must be finite");

    const bool prediction_fallback = prediction_pending_;
    const double residual_scale = guidance_scale - 1.0;
    std::vector<float> residual(conditional.size());
    for (std::size_t index = 0; index < conditional.size(); ++index) {
        const double value = residual_scale * (static_cast<double>(conditional[index]) -
                                               static_cast<double>(unconditional[index]));
        if (!std::isfinite(value) ||
            std::abs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
            throw std::invalid_argument("Wan2.2 late-CFG actual guidance residual is not finite");
        }
        residual[index] = static_cast<float>(value);
    }

    if (actual_history_size_ != 0) {
        older_actual_residual_ = std::move(latest_actual_residual_);
        older_actual_timestep_ = latest_actual_timestep_;
    }
    latest_actual_residual_ = std::move(residual);
    latest_actual_timestep_ = pending_timestep_;
    actual_history_size_ = std::min(actual_history_size_ + 1, 2);
    ++stats_.actual_unconditional_calls;
    if (prediction_fallback)
        ++stats_.prediction_fallbacks;
    actual_pending_ = false;
    prediction_pending_ = false;
}

std::optional<double> LateCfgController::prediction_factor() const noexcept {
    const double denominator =
        static_cast<double>(latest_actual_timestep_ - older_actual_timestep_);
    if (denominator == 0.0)
        return std::nullopt;
    const double factor =
        static_cast<double>(pending_timestep_ - latest_actual_timestep_) / denominator;
    if (!std::isfinite(factor) || std::abs(factor) > kMaximumTimestepExtrapolation)
        return std::nullopt;
    return factor;
}

std::optional<std::pair<float, float>>
LateCfgController::synthesize(float conditional, float older_residual, float latest_residual,
                              double factor, double residual_scale) noexcept {
    const double predicted =
        latest_residual + factor * (static_cast<double>(latest_residual) - older_residual);
    const double guided = static_cast<double>(conditional) + predicted;
    const double synthetic = residual_scale == 0.0
                                 ? conditional
                                 : static_cast<double>(conditional) - predicted / residual_scale;
    if (!std::isfinite(predicted) || !std::isfinite(guided) || !std::isfinite(synthetic))
        return std::nullopt;
    const float guided_value = static_cast<float>(guided);
    const float synthetic_value = static_cast<float>(synthetic);
    if (!std::isfinite(guided_value) || !std::isfinite(synthetic_value))
        return std::nullopt;
    return std::pair<float, float>{guided_value, synthetic_value};
}

std::optional<LateCfgPrediction>
LateCfgController::try_predict(const std::vector<float>& conditional, double guidance_scale) {
    if (!prediction_pending_)
        throw std::logic_error("Wan2.2 late-CFG prediction was not requested");
    if (!std::isfinite(guidance_scale) || conditional.empty())
        return std::nullopt;
    if (conditional.size() != latest_actual_residual_.size() ||
        conditional.size() != older_actual_residual_.size()) {
        return std::nullopt;
    }
    const auto factor = prediction_factor();
    if (!factor.has_value())
        return std::nullopt;

    LateCfgPrediction result;
    result.guided.resize(conditional.size());
    result.synthetic_unconditional.resize(conditional.size());
    const double residual_scale = guidance_scale - 1.0;
    for (std::size_t index = 0; index < conditional.size(); ++index) {
        const auto values = synthesize(conditional[index], older_actual_residual_[index],
                                       latest_actual_residual_[index], *factor, residual_scale);
        if (!values.has_value())
            return std::nullopt;
        result.guided[index] = values->first;
        result.synthetic_unconditional[index] = values->second;
    }
    prediction_pending_ = false;
    ++stats_.predicted_unconditional_reuses;
    return result;
}

void LateCfgController::require_same_nonempty_size(const std::vector<float>& left,
                                                   const std::vector<float>& right,
                                                   const char* operation) {
    if (left.empty() || left.size() != right.size()) {
        throw std::invalid_argument(std::string("Wan2.2 late-CFG ") + operation +
                                    " requires equal non-empty tensors");
    }
}

} // namespace trtmc::wan2_2_ti2v
