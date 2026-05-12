#include "runtime/models/ltx_video/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"

#if TRTMC_HAS_TRT

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void test_parse_ltx_video_options() {
    const std::string json = R"JSON({
      "negative_prompt": "bad frames",
      "frame_rate": 24,
      "guidance_rescale": 0.35
    })JSON";

    const auto opts = trtmc::parse_ltx_video_options(json);
    check(opts.negative_prompt == "bad frames", "negative prompt parsed");
    check(opts.frame_rate == 24, "frame rate parsed");
    check(std::fabs(opts.guidance_rescale - 0.35F) < 1e-6F, "guidance rescale parsed");
}

void test_ltx_latent_stats_parse_full_channel_count() {
    std::ostringstream mean;
    std::ostringstream stddev;
    for (int i = 0; i < 128; ++i) {
        if (i != 0) {
            mean << ',';
            stddev << ',';
        }
        mean << static_cast<float>(i) * 0.01F;
        stddev << 1.0F + static_cast<float>(i) * 0.02F;
    }

    const std::string json = std::string(R"JSON({
      "z_dim": 128,
      "latents_mean": [)JSON") +
                             mean.str() + R"JSON(],
      "latents_std": [)JSON" +
                             stddev.str() + R"JSON(]
    })JSON";

    const auto config = trtmc::make_diffusion_config(json);
    check(config.latents_mean.size() == 128, "ltx parses all latent mean channels");
    check(config.latents_std.size() == 128, "ltx parses all latent std channels");
    check(std::fabs(config.latents_mean.back() - 1.27F) < 1e-5F, "ltx parses last latent mean");
    check(std::fabs(config.latents_std.back() - 3.54F) < 1e-5F, "ltx parses last latent std");
}

} // namespace

int main() {
    test_parse_ltx_video_options();
    test_ltx_latent_stats_parse_full_channel_count();
    std::cerr << "test_ltx_video_pipeline PASS\n";
    return 0;
}

#else
int main() {
    return 0;
}
#endif
