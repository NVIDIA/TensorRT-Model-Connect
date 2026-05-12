// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         FluxPipeline, WanPipeline, and ZImagePipeline construction
//                 with null modules; verifies trivial constructors execute
//                 and pipeline_type() returns correct values
// Preconditions:  TRT headers available for type and compile check
// Postconditions: Diffusion pipeline types construct correctly with null
//                 modules and report accurate pipeline type strings
// =============================================================================

// =============================================================================
// Test suite: Diffusion pipeline construction tests
//
// FluxPipeline, WanPipeline, and ZImagePipeline all have trivial constructors
// (no module validation), so they can be constructed with null modules for
// testing the constructor body and pipeline_type() accessor.
//
// FluxPipeline constructor also computes h_latent_, w_latent_, and
// num_img_tokens_ from DiffusionConfig defaults (480x832 / scale_factor=8).
//
// For full E2E validation with real models, see tests/test_e2e.py.
// =============================================================================

#include "runtime/models/flux/pipeline.h"
#include "runtime/models/wan/pipeline.h"
#include "runtime/models/z_image/pipeline.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}

static void test_flux_construction() {
    // FluxPipeline constructor computes latent layout from DiffusionConfig defaults:
    //   h_latent = video_height(480) / scale_factor_spatial(8) = 60
    //   w_latent = video_width(832)  / scale_factor_spatial(8) = 104
    //   num_img_tokens = (60/2) * (104/2) = 30 * 52 = 1560
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;

    trtmc::FluxPipeline pipeline(
        /*text_encoders=*/{},
        /*denoiser=*/nullptr,
        /*vae=*/nullptr, cfg, weights,
        /*tokenizer=*/nullptr,
        /*clip_tokenizer=*/nullptr,
        /*model_id_str=*/"test-flux");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline", "FluxPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-flux", "FluxPipeline model_id");
}

static void test_wan_construction() {
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;

    trtmc::WanPipeline pipeline(
        /*text_encoder=*/nullptr,
        /*denoiser=*/nullptr,
        /*vae=*/nullptr, cfg, weights,
        /*tokenizer=*/nullptr,
        /*model_id_str=*/"test-wan");

    check(std::string(pipeline.pipeline_type()) == "WanPipeline", "WanPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-wan", "WanPipeline model_id");
}

static void test_zimage_construction() {
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;
    trtmc::ZImagePreprocessorWeights z_weights;

    trtmc::ZImagePipeline pipeline(
        /*text_encoder=*/nullptr,
        /*denoiser=*/nullptr,
        /*vae=*/nullptr, cfg, weights, z_weights,
        /*tokenizer=*/nullptr,
        /*model_id_str=*/"test-zimage",
        /*bundle_path=*/"/tmp/test.trtfb");

    check(std::string(pipeline.pipeline_type()) == "ZImagePipeline",
          "ZImagePipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-zimage", "ZImagePipeline model_id");
}

static void test_flux_with_custom_config() {
    // Test FluxPipeline with non-default config to exercise the latent layout
    // computation path with patch_size override.
    trtmc::DiffusionConfig cfg;
    cfg.video_height = 256;
    cfg.video_width = 256;
    cfg.scale_factor_spatial = 8;
    cfg.patch_size = {1, 2, 2}; // ph=2, pw=2

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, trtmc::PreprocessorWeights{}, nullptr,
                                 nullptr, "test-flux-custom");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline",
          "FluxPipeline custom config pipeline_type");
}

int main() {
    test_flux_construction();
    test_wan_construction();
    test_zimage_construction();
    test_flux_with_custom_config();
    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
