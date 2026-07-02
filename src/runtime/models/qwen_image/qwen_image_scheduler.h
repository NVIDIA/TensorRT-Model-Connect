/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// IScheduler: diffusion noise scheduler interface.
// HF equivalent: SchedulerMixin / FlowMatchEulerDiscreteScheduler.
//
// Pipelines compose IScheduler as an interchangeable component — swap
// FlowMatchEuler for DDPM without changing the pipeline code.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class IScheduler {
  public:
    virtual ~IScheduler() = default;

    // Configure the timestep schedule for the given number of steps.
    virtual void set_timesteps(int32_t num_steps) = 0;

    // Access the timestep schedule (descending from ~1000 to ~0).
    virtual const std::vector<float>& timesteps() const = 0;

    // Access the sigma schedule (descending from ~1.0 to ~0.0).
    virtual const std::vector<float>& sigmas() const = 0;

    // Single scheduler step: update latents in-place.
    // latents += (sigma_next - sigma) * velocity
    virtual void step(float* latents, const float* velocity, int32_t num_elements,
                      int32_t step_index) = 0;
};

// Full FlowMatchEulerDiscreteScheduler configuration. Mirrors the diffusers
// constructor surface used by dynamic-shift schedulers. The defaults match
// the legacy static-shift behavior so the existing single-arg constructor
// remains a no-op equivalent.
struct FlowMatchEulerConfig {
    int32_t num_train_timesteps = 1000;
    float shift = 1.0f; // static shift; used when use_dynamic_shifting=false
    bool use_dynamic_shifting = false;
    float base_shift = 0.5f;
    float max_shift = 1.15f;
    int32_t base_image_seq_len = 256;
    int32_t max_image_seq_len = 4096;
    float shift_terminal = 0.0f; // 0 = disabled (matches diffusers None)
    // "exponential" or empty/"linear". Empty defaults to "exponential" when
    // use_dynamic_shifting=true (diffusers' default).
    std::string time_shift_type;
};

// Flow Matching Euler Discrete Scheduler.
//
// Two construction modes:
//   - Static-shift: pass `shift` directly. Equivalent to
//     diffusers' use_dynamic_shifting=False.
//   - Full config: pass FlowMatchEulerConfig. Supports dynamic-mu shifting,
//     exponential time_shift_type, and shift_terminal.
class FlowMatchEulerScheduler final : public IScheduler {
  public:
    explicit FlowMatchEulerScheduler(float shift = 1.0f, int32_t num_train_timesteps = 1000);
    explicit FlowMatchEulerScheduler(const FlowMatchEulerConfig& cfg);

    // Static-shift path that preserves the original scheduler behavior.
    void set_timesteps(int32_t num_steps) override;

    // Dynamic-shifting path. Mirrors diffusers'
    //   sigmas = np.linspace(1.0, 1.0/N, N)
    //   set_timesteps(sigmas=sigmas, mu=calculate_shift(image_seq_len, ...))
    // when use_dynamic_shifting=true. Falls back to the static-shift path
    // when use_dynamic_shifting=false (image_seq_len ignored).
    void set_timesteps(int32_t num_steps, int32_t image_seq_len);

    const std::vector<float>& timesteps() const override { return timesteps_; }
    const std::vector<float>& sigmas() const override { return sigmas_; }

    void step(float* latents, const float* velocity, int32_t num_elements,
              int32_t step_index) override;

  private:
    FlowMatchEulerConfig cfg_;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};

// Factory: create scheduler by name.
inline std::unique_ptr<IScheduler> create_scheduler(const std::string& name, float shift = 1.0f) {
    if (name == "flow_match_euler") {
        return std::make_unique<FlowMatchEulerScheduler>(shift);
    }
    return nullptr;
}

} // namespace trtmc
