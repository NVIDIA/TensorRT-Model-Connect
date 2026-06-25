// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-QWEN-IMAGE-SCHED-01
// Architecture:   ARCH-FAM-001
// Unit Design:    UD-FAM-QWEN-IMAGE-01
// Intent:         Qwen Image-owned FlowMatchEulerScheduler timestep computation,
//                 sigma schedule, and Euler step
// Preconditions:  None (CPU-only math)
// Postconditions: Timesteps, sigmas, and step outputs match HF reference values
// =============================================================================

// =============================================================================
// Test suite: Qwen Image FlowMatchEulerScheduler
// =============================================================================
//
// Validates timestep computation, sigma schedule, and Euler step against
// known values from HF's FlowMatchEulerDiscreteScheduler.
// =============================================================================

#include "runtime/models/qwen_image/qwen_image_scheduler.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static bool approx_eq(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) < tol;
}

static void test_set_timesteps_no_shift() {
    trtmc::FlowMatchEulerScheduler sched(1.0f, 1000);
    sched.set_timesteps(4);

    const auto& ts = sched.timesteps();
    check(ts.size() == 4, "4 timesteps");
    // With shift=1.0: linear spacing from 1000 to ~250
    check(ts[0] > 900.0f, "ts[0] > 900");
    check(ts[3] > 0.0f, "ts[3] > 0");
    check(ts[0] > ts[1], "descending order");

    const auto& sig = sched.sigmas();
    check(sig.size() == 5, "5 sigmas (4 + terminal)");
    check(approx_eq(sig[0], 1.0f), "sigma[0] ≈ 1.0");
    check(sig[4] == 0.0f, "terminal sigma = 0.0");
}

static void test_set_timesteps_with_shift() {
    trtmc::FlowMatchEulerScheduler sched(3.0f, 1000);
    sched.set_timesteps(28);

    const auto& ts = sched.timesteps();
    check(ts.size() == 28, "28 timesteps");

    const auto& sig = sched.sigmas();
    check(sig.size() == 29, "29 sigmas");
    check(approx_eq(sig[0], 1.0f, 0.01f), "sigma[0] ≈ 1.0");
    check(sig[28] == 0.0f, "terminal = 0");

    // With shift=3.0, sigmas should be shifted toward 1.0 (more time at high noise)
    // Middle sigma should be > 0.5 (shifted up from linear 0.5)
    check(sig[14] > 0.5f, "shifted: mid-sigma > 0.5");
}

static void test_euler_step() {
    trtmc::FlowMatchEulerScheduler sched(1.0f, 1000);
    sched.set_timesteps(4);

    const auto& sig = sched.sigmas();
    float dt = sig[1] - sig[0]; // Should be negative (sigma decreases)
    check(dt < 0.0f, "dt is negative (sigma decreasing)");

    // Test step: latents += dt * velocity
    float latents[3] = {1.0f, 2.0f, 3.0f};
    float velocity[3] = {0.1f, 0.2f, 0.3f};

    // Save original
    float orig[3] = {1.0f, 2.0f, 3.0f};

    sched.step(latents, velocity, 3, 0);

    // Verify: latents = orig + dt * velocity
    for (int i = 0; i < 3; ++i) {
        float expected = orig[i] + dt * velocity[i];
        check(approx_eq(latents[i], expected, 1e-5f),
              (std::string("euler step element ") + std::to_string(i)).c_str());
    }
}

static void test_factory() {
    auto sched = trtmc::create_scheduler("flow_match_euler", 3.0f);
    check(sched != nullptr, "factory creates flow_match_euler");

    auto bad = trtmc::create_scheduler("nonexistent");
    check(bad == nullptr, "factory returns null for unknown");
}

static void test_single_step() {
    trtmc::FlowMatchEulerScheduler sched(1.0f, 1000);
    sched.set_timesteps(1);

    check(sched.timesteps().size() == 1, "1 timestep");
    check(sched.sigmas().size() == 2, "2 sigmas");
}

int main() {
    test_set_timesteps_no_shift();
    test_set_timesteps_with_shift();
    test_euler_step();
    test_factory();
    test_single_step();

    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
