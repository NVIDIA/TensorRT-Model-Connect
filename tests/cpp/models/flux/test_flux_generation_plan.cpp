// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-FLUX-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-DIFF-FLUX-01
// Intent:         Flux generation plan layout and scheduler derivation
// Preconditions:  Flux diffusion config with valid latent dimensions
// Postconditions: Layout dimensions and scheduler parameters match expected values
// =============================================================================

#include "runtime/models/flux/flux_generation_plan.h"

#include <cmath>
#include <cstddef>
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

void test_flux_generation_plan_derives_layout_and_scheduler() {
    trtmc::FluxDiffusionConfig config;
    config.num_inference_steps = 28;
    config.guidance_scale = 4.5F;
    config.z_dim = 16;
    config.dit_dim = 3072;
    config.text_seq_len = 256;
    config.patch_size = {1, 2, 2};
    config.flow_shift = 1.2F;
    config.use_dynamic_shifting = true;
    config.base_shift = 0.7F;
    config.max_shift = 1.3F;
    config.base_image_seq_len = 256;
    config.max_image_seq_len = 4096;

    trtmc::FluxPreprocessorWeights weights;
    weights.vae_bn_mean = {0.0F};

    const auto plan =
        trtmc::diffusion::make_flux_generation_plan(config, weights, 0, -1.0F, 128, 128, 4096);

    check(plan.num_inference_steps == 28, "flux plan uses fallback steps when request is zero");
    check_close(plan.guidance_scale, 4.5F, 1e-6F, "flux plan uses fallback guidance");
    check(plan.layout.ph == 2 && plan.layout.pw == 2, "flux plan derives patch layout");
    check(plan.layout.packed_channels == 64, "flux plan derives packed channels");
    check(plan.layout.h_packed == 64 && plan.layout.w_packed == 64,
          "flux plan derives packed spatial size");
    check(plan.is_flux2, "flux plan detects flux2 from bn weights");
    check(plan.latent_size == static_cast<std::size_t>(64 * 64 * 64),
          "flux plan computes flux2 latent size");
    check(plan.scheduler_config.use_dynamic_shifting, "flux plan forwards dynamic shift flag");
    check(plan.scheduler_config.use_empirical_mu, "flux plan uses empirical mu for flux2");
    check(plan.scheduler_config.image_seq_len == 4096, "flux plan forwards image token count");

    const auto scheduler = trtmc::diffusion::make_flux_scheduler_state(plan);
    check(scheduler.timesteps.size() == 28, "flux scheduler size matches plan");
    check(scheduler.last_used_dynamic_shifting, "flux scheduler records dynamic shifting");
}

} // namespace

int main() {
    test_flux_generation_plan_derives_layout_and_scheduler();

    if (g_failures != 0) {
        std::cerr << g_failures << " flux generation plan test(s) failed\n";
        return 1;
    }
    return 0;
}
