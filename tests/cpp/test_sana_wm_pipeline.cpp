// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SANAWM-CPP-01
// Architecture:   ARCH-RUNTIME-001
// Unit Design:    UD-SANAWM-01
// Intent:         SANA-WM C++ runtime forwards official action-control contract
// Preconditions:  Bundle config requests strict official-script execution
// Postconditions: Bridge argv includes action/speed/frame flags and strict mode
// =============================================================================

#include "../../src/runtime/models/sana_wm/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

bool contains_arg(const std::vector<std::string>& argv, const std::string& arg) {
    return std::find(argv.begin(), argv.end(), arg) != argv.end();
}

bool near(float actual, float expected, float eps = 1.0e-4F) {
    return std::fabs(actual - expected) <= eps;
}

std::string value_after(const std::vector<std::string>& argv, const std::string& flag) {
    auto it = std::find(argv.begin(), argv.end(), flag);
    if (it == argv.end() || ++it == argv.end())
        return "";
    return *it;
}

class FakeSubprocessRunner final : public trtmc::ISubprocessRunner {
  public:
    std::vector<std::string> last_argv;
    int call_count{0};

    int run(const std::vector<std::string>& argv, const void*, std::size_t,
            std::vector<char>& out_stdout, std::string& out_stderr) override {
        ++call_count;
        last_argv = argv;
        out_stdout.clear();
        out_stderr.clear();

        if (value_after(argv, "--frames-dir").empty()) {
            out_stderr = "missing --frames-dir";
            return 2;
        }
        return 0;
    }
};

void test_action_rollout_matches_model_card_frame_count_and_translation() {
    const auto poses = trtmc::sana_wm_action_to_c2w("w-80,jw-40,w-40,lw-60,w-100", 0.055F, 1.2F);

    check(poses.size() == 321, "sana wm action: model-card action rolls out to 321 poses");
    check(near(poses.front().c2w[0], 1.0F) && near(poses.front().c2w[5], 1.0F) &&
              near(poses.front().c2w[10], 1.0F) && near(poses.front().c2w[15], 1.0F),
          "sana wm action: first pose is identity");
    check(near(poses[1].c2w[11], 0.055F), "sana wm action: w moves forward on +Z");
    check(near(poses[80].c2w[11], 4.399996F, 1.0e-3F),
          "sana wm action: first w-80 segment accumulates translation");
    check(poses[81].c2w[2] < 0.0F, "sana wm action: j yaw turns left");
    check(poses[121].c2w[2] < poses[81].c2w[2], "sana wm action: repeated j yaw accumulates");
    check(std::fabs(poses.back().c2w[3]) > 0.01F, "sana wm action: yaw changes final x motion");
}

void test_action_rollout_rejects_invalid_segments() {
    bool rejected_empty = false;
    try {
        (void)trtmc::sana_wm_action_to_c2w("", 0.055F, 1.2F);
    } catch (const std::invalid_argument&) {
        rejected_empty = true;
    }
    check(rejected_empty, "sana wm action: empty string rejected");

    bool rejected_bad_key = false;
    try {
        (void)trtmc::sana_wm_action_to_c2w("wx-1", 0.055F, 1.2F);
    } catch (const std::invalid_argument&) {
        rejected_bad_key = true;
    }
    check(rejected_bad_key, "sana wm action: unknown key rejected");
}

void test_resize_crop_plan_matches_upstream_geometry() {
    const auto plan = trtmc::sana_wm_make_resize_crop_plan(640, 480, 704, 1280);

    check(plan.resized_width == 1280, "sana wm crop: width scales to target");
    check(plan.resized_height == 960, "sana wm crop: height preserves aspect");
    check(plan.crop_left == 0, "sana wm crop: no horizontal crop for 4:3 source");
    check(plan.crop_top == 128, "sana wm crop: centered vertical crop");

    const auto intr = trtmc::sana_wm_transform_intrinsics_for_crop(
        trtmc::SanaWmIntrinsics{100.0F, 120.0F, 320.0F, 240.0F}, plan);
    check(near(intr.fx, 200.0F), "sana wm crop: fx scaled");
    check(near(intr.fy, 240.0F), "sana wm crop: fy scaled");
    check(near(intr.cx, 640.0F), "sana wm crop: cx scaled");
    check(near(intr.cy, 352.0F), "sana wm crop: cy scaled and crop-adjusted");
}

void test_resize_center_crop_crops_hwc_pixels() {
    const std::vector<float> src = {
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F,
        0.5F, 0.5F, 0.0F, 0.0F, 0.5F, 0.5F, 0.5F, 0.0F, 0.5F,
    };
    const auto image = trtmc::sana_wm_resize_and_center_crop(src, 3, 2, 2, 2);

    check(image.ok, "sana wm crop: resize helper succeeds");
    check(image.plan.resized_width == 3 && image.plan.resized_height == 2,
          "sana wm crop: no-op resize shape preserved");
    check(image.plan.crop_left == 0 && image.plan.crop_top == 0,
          "sana wm crop: integer center crop starts at origin");
    check(image.pixels_hwc.size() == 12, "sana wm crop: output is target HWC RGB");
    check(near(image.pixels_hwc[0], 1.0F) && near(image.pixels_hwc[1], 0.0F) &&
              near(image.pixels_hwc[2], 0.0F),
          "sana wm crop: first RGB pixel preserved");
    check(near(image.pixels_hwc[9], 0.0F) && near(image.pixels_hwc[10], 0.5F) &&
              near(image.pixels_hwc[11], 0.5F),
          "sana wm crop: last cropped pixel preserved");
}

void test_bridge_command_forwards_strict_sana_wm_contract() {
    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";
    cfg.action = "bundle-action";
    cfg.translation_speed = 0.01F;
    cfg.rotation_speed_deg = 0.5F;
    cfg.num_frames = 99;
    cfg.require_official_script = true;

    auto runner = std::make_shared<FakeSubprocessRunner>();
    trtmc::SanaWmPipeline pipeline(cfg, "/usr/bin/python3", runner);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = "asset/sana_wm/demo_0.png";
    gen_cfg.camera_action = "w-80,jw-40,w-40,lw-60,w-100";
    gen_cfg.translation_speed = 0.055F;
    gen_cfg.rotation_speed_deg = 1.2F;
    gen_cfg.num_frames = 321;

    bool missing_frames_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        missing_frames_reported =
            std::string(exc.what()).find("produced no frame_*.png") != std::string::npos;
    }

    check(runner->call_count == 1, "sana wm: subprocess invoked once");
    check(missing_frames_reported, "sana wm: requires bridge to materialize frames");
    check(contains_arg(runner->last_argv, "-m"), "sana wm: python module mode");
    check(contains_arg(runner->last_argv, "tensorrt_model_connect.sana_wm_bridge"),
          "sana wm: bridge module");
    check(value_after(runner->last_argv, "--hf-id") ==
              "Efficient-Large-Model/SANA-WM_bidirectional",
          "sana wm: hf id forwarded");
    check(value_after(runner->last_argv, "--image") == "asset/sana_wm/demo_0.png",
          "sana wm: image forwarded");
    check(value_after(runner->last_argv, "--prompt-text") == "drive forward",
          "sana wm: prompt forwarded");
    check(value_after(runner->last_argv, "--action") == "w-80,jw-40,w-40,lw-60,w-100",
          "sana wm: action forwarded");
    check(value_after(runner->last_argv, "--translation-speed").rfind("0.055", 0) == 0,
          "sana wm: translation speed forwarded");
    check(value_after(runner->last_argv, "--rotation-speed-deg").rfind("1.200", 0) == 0,
          "sana wm: rotation speed forwarded");
    check(value_after(runner->last_argv, "--num-frames") == "321",
          "sana wm: frame count forwarded");
    check(contains_arg(runner->last_argv, "--no-diffusers-fallback"),
          "sana wm: strict official runtime required");
}

} // namespace

int main() {
    test_action_rollout_matches_model_card_frame_count_and_translation();
    test_action_rollout_rejects_invalid_segments();
    test_resize_crop_plan_matches_upstream_geometry();
    test_resize_center_crop_crops_hwc_pixels();
    test_bridge_command_forwards_strict_sana_wm_contract();
    return failures == 0 ? 0 : 1;
}
