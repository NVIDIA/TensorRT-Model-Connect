/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace trtmc::wan2_2_ti2v {

constexpr double kQualifiedEasyCacheThreshold = 0.08;
constexpr int32_t kQualifiedEasyCacheFirstExactSteps = 7;
constexpr int32_t kQualifiedEasyCacheLastExactSteps = 2;
constexpr int32_t kQualifiedEasyCacheMaxConsecutiveReuse = 4;
constexpr int32_t kQualifiedEasyCacheTotalSteps = 50;

struct EasyCacheConfig {
    bool enabled{false};
    double threshold{0.02};
    int32_t first_exact_steps{7};
    int32_t last_exact_steps{2};
    int32_t max_consecutive_reuse{1};
    int32_t total_steps{50};
};

struct EasyCacheStats {
    int32_t total_steps{0};
    int32_t compute_steps{0};
    int32_t reuse_steps{0};
};

EasyCacheConfig easycache_config_from_environment(int32_t total_steps);
bool late_cfg_enabled_from_environment(const EasyCacheConfig& easycache);

class EasyCacheController {
  public:
    explicit EasyCacheController(EasyCacheConfig config);

    bool decide(int32_t step, const std::vector<float>& raw_latents);
    void update_conditional(const std::vector<float>& raw_latents,
                            const std::vector<float>& denoiser_output);
    void update_unconditional(const std::vector<float>& raw_latents,
                              const std::vector<float>& denoiser_output);
    std::vector<float> reuse_conditional(const std::vector<float>& raw_latents) const;
    std::vector<float> reuse_unconditional(const std::vector<float>& raw_latents) const;

    const EasyCacheConfig& config() const noexcept { return config_; }
    const EasyCacheStats& stats() const noexcept { return stats_; }

  private:
    void validate_decision_input(int32_t step, const std::vector<float>& raw_latents) const;
    bool choose_compute(int32_t step, const std::vector<float>& raw_latents);
    bool in_exact_window(int32_t step) const noexcept;
    bool has_reusable_state() const noexcept;
    void finish_decision(bool compute, const std::vector<float>& raw_latents);

    static double mean_abs(const std::vector<float>& values);
    static double mean_abs_delta(const std::vector<float>& left, const std::vector<float>& right);
    static std::vector<float> make_residual(const std::vector<float>& output,
                                            const std::vector<float>& input);
    static std::vector<float> add_residual(const std::vector<float>& input,
                                           const std::vector<float>& residual);
    static void require_same_nonempty_size(const std::vector<float>& left,
                                           const std::vector<float>& right, const char* operation);

    EasyCacheConfig config_;
    EasyCacheStats stats_;
    std::vector<float> previous_step_input_;
    std::vector<float> last_full_input_;
    std::vector<float> last_full_output_;
    std::vector<float> conditional_residual_;
    std::vector<float> unconditional_residual_;
    std::optional<double> change_factor_;
    double accumulator_{0.0};
    int32_t consecutive_reuse_{0};
    bool last_decision_compute_{true};
};

enum class LateCfgAction {
    kEasyCacheReuse,
    kActualUnconditional,
    kPredictUnconditional,
};

struct LateCfgPrediction {
    std::vector<float> guided;
    std::vector<float> synthetic_unconditional;
};

struct LateCfgStats {
    int32_t total_steps{0};
    int32_t processed_steps{0};
    int32_t easycache_compute_events{0};
    int32_t easycache_reuse_events{0};
    int32_t actual_unconditional_calls{0};
    int32_t predicted_unconditional_reuses{0};
    int32_t prediction_fallbacks{0};
};

class LateCfgController {
  public:
    LateCfgController();

    LateCfgAction decide(int32_t step, int64_t timestep, bool easycache_compute);
    void record_actual(const std::vector<float>& conditional,
                       const std::vector<float>& unconditional, double guidance_scale);
    std::optional<LateCfgPrediction> try_predict(const std::vector<float>& conditional,
                                                 double guidance_scale);

    const LateCfgStats& stats() const noexcept { return stats_; }

  private:
    bool in_exact_window(int32_t step) const noexcept;
    bool has_predictable_history() const noexcept;
    LateCfgAction request_actual();
    std::optional<double> prediction_factor() const noexcept;
    static std::optional<std::pair<float, float>> synthesize(float conditional,
                                                             float older_residual,
                                                             float latest_residual, double factor,
                                                             double residual_scale) noexcept;
    static void require_same_nonempty_size(const std::vector<float>& left,
                                           const std::vector<float>& right, const char* operation);

    LateCfgStats stats_;
    std::vector<float> older_actual_residual_;
    std::vector<float> latest_actual_residual_;
    int64_t older_actual_timestep_{0};
    int64_t latest_actual_timestep_{0};
    int64_t pending_timestep_{0};
    int32_t actual_history_size_{0};
    int32_t late_compute_event_{0};
    bool actual_pending_{false};
    bool prediction_pending_{false};
};

} // namespace trtmc::wan2_2_ti2v
