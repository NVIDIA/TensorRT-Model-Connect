// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-LTX-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-DIFF-LTX-01
// Intent:         LTX dynamic flow-match scheduler fields
// Preconditions:  Flow-match scheduler configured with LTX sequence lengths
// Postconditions: Dynamic shift and terminal sigma fields match expected values
// =============================================================================

#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"

#include <cmath>
#include <iostream>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

void test_ltx_dynamic_flow_match_scheduler_fields() {
    trtmc::diffusion::FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = 1000;
    scheduler.use_dynamic_shifting = true;
    scheduler.base_shift = 0.95F;
    scheduler.max_shift = 2.05F;
    scheduler.base_image_seq_len = 1024;
    scheduler.max_image_seq_len = 4096;
    scheduler.shift_terminal = 0.1F;
    scheduler.image_seq_len = 21 * 15 * 22;

    scheduler.set_timesteps(50);

    check(scheduler.last_used_dynamic_shifting, "ltx scheduler uses dynamic shifting");
    check(scheduler.sigmas.size() == 51, "ltx scheduler appends terminal sigma");
    check_close(static_cast<float>(scheduler.sigmas[49]), 0.1F, 1e-5F,
                "ltx scheduler stretches final pre-terminal sigma");
    check_close(static_cast<float>(scheduler.sigmas[50]), 0.0F, 1e-6F,
                "ltx scheduler terminal sigma remains zero");
    check_close(static_cast<float>(scheduler.last_dynamic_mu), 3.06478F, 1e-4F,
                "ltx scheduler uses LTX base/max sequence shift formula");
}

} // namespace

int main() {
    test_ltx_dynamic_flow_match_scheduler_fields();

    if (g_failures != 0) {
        std::cerr << g_failures << " ltx scheduler test(s) failed\n";
        return 1;
    }
    return 0;
}
