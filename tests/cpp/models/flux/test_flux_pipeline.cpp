#include "runtime/models/flux/pipeline.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_flux_construction() {
    trtmc::FluxDiffusionConfig cfg;
    trtmc::FluxPreprocessorWeights weights;

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, weights, nullptr, nullptr, "test-flux");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline", "FluxPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-flux", "FluxPipeline model_id");
}

void test_flux_with_custom_config() {
    trtmc::FluxDiffusionConfig cfg;
    cfg.video_height = 256;
    cfg.video_width = 256;
    cfg.scale_factor_spatial = 8;
    cfg.patch_size = {1, 2, 2};

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, trtmc::FluxPreprocessorWeights{},
                                 nullptr, nullptr, "test-flux-custom");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline",
          "FluxPipeline custom config pipeline_type");
}

} // namespace

int main() {
    test_flux_construction();
    test_flux_with_custom_config();
    if (failures > 0) {
        std::cerr << failures << " flux pipeline test(s) FAILED\n";
    }
    return failures;
}
