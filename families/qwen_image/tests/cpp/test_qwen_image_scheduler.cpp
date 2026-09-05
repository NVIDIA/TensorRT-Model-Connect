/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image/runtime/qwen_image_scheduler.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

bool close(float left, float right, float tolerance = 1e-3F) {
    return std::fabs(left - right) < tolerance;
}

void test_set_timesteps_no_shift() {
    trtmc::FlowMatchEulerConfig config;
    config.shift = 1.0F;
    config.num_train_timesteps = 1000;
    trtmc::FlowMatchEulerScheduler scheduler(config);
    scheduler.set_timesteps(4);
    const auto& timesteps = scheduler.timesteps();
    const auto& sigmas = scheduler.sigmas();
    check(timesteps.size() == 4, "four timesteps");
    check(timesteps[0] > 900.0F && timesteps[3] > 0.0F && timesteps[0] > timesteps[1],
          "timesteps are positive and descending");
    check(sigmas.size() == 5 && close(sigmas[0], 1.0F) && sigmas[4] == 0.0F,
          "sigma schedule includes terminal zero");
}

void test_set_timesteps_with_shift() {
    trtmc::FlowMatchEulerConfig config;
    config.shift = 3.0F;
    config.num_train_timesteps = 1000;
    trtmc::FlowMatchEulerScheduler scheduler(config);
    scheduler.set_timesteps(28);
    check(scheduler.timesteps().size() == 28, "28 timesteps");
    check(scheduler.sigmas().size() == 29, "29 sigmas");
    check(close(scheduler.sigmas()[0], 1.0F, 0.01F), "first sigma is one");
    check(scheduler.sigmas()[28] == 0.0F, "terminal sigma is zero");
    check(scheduler.sigmas()[14] > 0.5F, "shift raises the middle sigma");
}

void test_euler_step() {
    trtmc::FlowMatchEulerConfig config;
    config.shift = 1.0F;
    config.num_train_timesteps = 1000;
    trtmc::FlowMatchEulerScheduler scheduler(config);
    scheduler.set_timesteps(4);
    const float delta = scheduler.sigmas()[1] - scheduler.sigmas()[0];
    check(delta < 0.0F, "sigma decreases");
    float latents[3] = {1.0F, 2.0F, 3.0F};
    float velocity[3] = {0.1F, 0.2F, 0.3F};
    const float original[3] = {1.0F, 2.0F, 3.0F};
    scheduler.step(latents, velocity, 3, 0);
    for (int index = 0; index < 3; ++index) {
        const float expected = original[index] + delta * velocity[index];
        check(close(latents[index], expected, 1e-5F),
              (std::string("euler element ") + std::to_string(index)).c_str());
    }
}

void test_scheduler_interface() {
    trtmc::FlowMatchEulerConfig config;
    config.shift = 3.0F;
    std::unique_ptr<trtmc::IScheduler> scheduler =
        std::make_unique<trtmc::FlowMatchEulerScheduler>(config);
    scheduler->set_timesteps(2);
    check(scheduler->timesteps().size() == 2,
          "scheduler is usable through the current abstract interface");
}

void test_single_step() {
    trtmc::FlowMatchEulerConfig config;
    config.shift = 1.0F;
    config.num_train_timesteps = 1000;
    trtmc::FlowMatchEulerScheduler scheduler(config);
    scheduler.set_timesteps(1);
    check(scheduler.timesteps().size() == 1, "single timestep");
    check(scheduler.sigmas().size() == 2, "single step has terminal sigma");
}

} // namespace

int main() {
    test_set_timesteps_no_shift();
    test_set_timesteps_with_shift();
    test_euler_step();
    test_scheduler_interface();
    test_single_step();
    return failures == 0 ? 0 : 1;
}
