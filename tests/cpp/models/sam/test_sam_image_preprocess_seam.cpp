// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-03-SAM
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SAM image preprocessing seam: resize and padding plan
// Preconditions:  Decoded image data and SAM config are available
// Postconditions: SAM preprocessing tracks original/rescaled dimensions and
//                 preserves right padding
// =============================================================================

#include "runtime/models/sam/sam_image_preprocess_seam.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

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

void test_sam_image_plan_tracks_resize_and_padding() {
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = 1;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0, 0, 255, 0, 0,
    };

    trtmc::SamConfig config;
    config.image_size = 4;
    config.image_mean = {0.0F, 0.0F, 0.0F};
    config.image_std = {1.0F, 1.0F, 1.0F};

    const auto plan = trtmc::build_sam_image_encode_plan(image, config);
    check(plan.ok(), "sam image plan ok");
    check(plan.original_width == 1 && plan.original_height == 2, "sam image plan original dims");
    check(plan.rescaled_width == 2 && plan.rescaled_height == 4, "sam image plan rescaled dims");
    check(plan.pixel_values.size() == 48, "sam image plan output size");
    if (plan.pixel_values.size() != 48) {
        return;
    }

    const std::size_t red_plane_offset = 0;
    const std::size_t green_plane_offset = 16;
    check_close(plan.pixel_values[red_plane_offset], 1.0F, 1e-6F,
                "sam image plan red data preserved");
    check_close(plan.pixel_values[green_plane_offset], 0.0F, 1e-6F,
                "sam image plan green data preserved");
    check_close(plan.pixel_values[3], 0.0F, 1e-6F, "sam image plan right padding stays zero");
}

} // namespace

int main() {
    test_sam_image_plan_tracks_resize_and_padding();

    if (g_failures != 0) {
        std::cerr << g_failures << " SAM image preprocess seam test(s) failed\n";
        return 1;
    }
    return 0;
}
