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

#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

std::size_t chunk_plucker_offset(int32_t channel, int32_t chunk, int32_t y, int32_t x,
                                 int32_t chunk_count, int32_t h, int32_t w) {
    return (((static_cast<std::size_t>(channel) * static_cast<std::size_t>(chunk_count) +
              static_cast<std::size_t>(chunk)) *
                 static_cast<std::size_t>(h) +
             static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(w) +
            static_cast<std::size_t>(x));
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

class FakeTrtModule final : public trtmc::ITrtModule {
  public:
    FakeTrtModule() = default;
    FakeTrtModule(std::vector<float> output, std::vector<int64_t> shape)
        : output_(std::move(output)), output_shape_(std::move(shape)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        if (output_.empty())
            return {};
        return {{"latent", trtmc::Tensor{output_.data(), output_shape_, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        if (output_.empty())
            return {};
        return {{"sample", {1, 3, 1, 4, 4}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return !output_.empty() && name == "sample";
    }
    bool has_output(const std::string& name) const override {
        return !output_.empty() && name == "latent";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<float> output_;
    std::vector<int64_t> output_shape_;
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

void test_camera_pose_vector_parses_row_major_matrices() {
    const auto poses = trtmc::sana_wm_row_major_c2w_to_poses({
        1.0F, 0.0F, 0.0F, 0.1F,
        0.0F, 1.0F, 0.0F, 0.2F,
        0.0F, 0.0F, 1.0F, 0.3F,
        0.0F, 0.0F, 0.0F, 1.0F,
        0.0F, -1.0F, 0.0F, 1.1F,
        1.0F, 0.0F, 0.0F, 1.2F,
        0.0F, 0.0F, 1.0F, 1.3F,
        0.0F, 0.0F, 0.0F, 1.0F,
    });

    check(poses.size() == 2, "sana wm camera: two row-major poses parsed");
    check(near(poses[0].c2w[3], 0.1F) && near(poses[0].c2w[7], 0.2F) &&
              near(poses[0].c2w[11], 0.3F),
          "sana wm camera: first pose translation preserved");
    check(near(poses[1].c2w[1], -1.0F) && near(poses[1].c2w[3], 1.1F),
          "sana wm camera: second pose rotation and translation preserved");

    bool rejected = false;
    try {
        (void)trtmc::sana_wm_row_major_c2w_to_poses({1.0F, 2.0F});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "sana wm camera: malformed pose vector rejected");
}

void test_intrinsics_expand_model_card_shapes() {
    const auto four = trtmc::sana_wm_expand_intrinsics({10.0F, 11.0F, 6.0F, 7.0F}, 2);
    check(four.size() == 2 && near(four[1].fx, 10.0F) && near(four[1].cy, 7.0F),
          "sana wm intrinsics: fx/fy/cx/cy expands to all frames");

    const auto matrix = trtmc::sana_wm_expand_intrinsics({
        20.0F, 0.0F, 8.0F,
        0.0F, 21.0F, 9.0F,
        0.0F, 0.0F, 1.0F,
    }, 3);
    check(matrix.size() == 3 && near(matrix[2].fx, 20.0F) && near(matrix[2].fy, 21.0F) &&
              near(matrix[2].cx, 8.0F) && near(matrix[2].cy, 9.0F),
          "sana wm intrinsics: single 3x3 matrix expands to all frames");

    const auto per_frame = trtmc::sana_wm_expand_intrinsics({
        30.0F, 0.0F, 12.0F,
        0.0F, 31.0F, 13.0F,
        0.0F, 0.0F, 1.0F,
        40.0F, 0.0F, 14.0F,
        0.0F, 41.0F, 15.0F,
        0.0F, 0.0F, 1.0F,
    }, 2);
    check(per_frame.size() == 2 && near(per_frame[0].fx, 30.0F) &&
              near(per_frame[1].fx, 40.0F) && near(per_frame[1].cy, 15.0F),
          "sana wm intrinsics: per-frame 3x3 matrices parsed");

    bool rejected = false;
    try {
        (void)trtmc::sana_wm_expand_intrinsics({1.0F, 2.0F, 3.0F}, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "sana wm intrinsics: malformed vector rejected");
}

void test_runtime_config_parses_native_sana_wm_fields() {
    const auto cfg = trtmc::parse_sana_wm_config(
        R"json({
          "sana_wm_hf_id": "Efficient-Large-Model/SANA-WM_bidirectional",
          "video_height": 704,
          "video_width": 1280,
          "video_num_frames": 321,
          "fps": 16,
          "num_inference_steps": 60,
          "guidance_scale": 5.0,
          "flow_shift": 9.8,
          "seed": 42,
          "vae_latent_dim": 128,
          "vae_time_stride": 8,
          "vae_spatial_stride": 32,
          "text_encoder_max_length": 300,
          "sana_wm_chi_prompt": "Generate an \"Enhanced prompt\".\nUser Prompt: "
        })json");

    check(cfg.height == 704 && cfg.width == 1280 && cfg.num_frames == 321,
          "sana wm config: model-card video shape parsed");
    check(cfg.fps == 16 && cfg.num_steps == 60 && near(cfg.cfg_scale, 5.0F),
          "sana wm config: generation defaults parsed");
    check(near(cfg.flow_shift, 9.8F) && cfg.seed == 42,
          "sana wm config: scheduler seed defaults parsed");
    check(cfg.vae_latent_dim == 128 && cfg.vae_time_stride == 8 && cfg.vae_spatial_stride == 32,
          "sana wm config: vae shape contract parsed");
    check(cfg.text_encoder_max_length == 300, "sana wm config: text encoder length parsed");
    check(cfg.chi_prompt == "Generate an \"Enhanced prompt\".\nUser Prompt: ",
          "sana wm config: chi prompt parsed");
}

void test_conditioning_prompt_matches_upstream_chi_prefix() {
    const std::string chi = "Generate an \"Enhanced prompt\".\nUser Prompt: ";
    check(trtmc::sana_wm_make_conditioning_prompt("drive forward", chi) ==
              "Generate an \"Enhanced prompt\".\nUser Prompt: drive forward",
          "sana wm prompt: chi prompt prepended without extra separator");
    check(trtmc::sana_wm_make_conditioning_prompt("drive forward", "") == "drive forward",
          "sana wm prompt: empty chi prompt leaves prompt unchanged");
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

void test_prepare_vae_input_image_matches_upstream_tensor_layout() {
    const std::vector<float> src = {
        1.0F, 0.5F, 0.0F, 0.25F, 0.75F, 1.0F,
    };

    const auto image = trtmc::sana_wm_prepare_vae_input_image(src, 2, 1, 1, 2);

    check(image.ok, "sana wm vae input: conversion succeeds");
    check(image.height == 1 && image.width == 2 && image.channels == 3,
          "sana wm vae input: shape metadata propagated");
    check(image.pixels_chw.size() == 6, "sana wm vae input: output is CHW RGB");
    check(near(image.pixels_chw[0], 1.0F) && near(image.pixels_chw[1], -0.5F),
          "sana wm vae input: red channel is normalized CHW");
    check(near(image.pixels_chw[2], 0.0F) && near(image.pixels_chw[3], 0.5F),
          "sana wm vae input: green channel is normalized CHW");
    check(near(image.pixels_chw[4], -1.0F) && near(image.pixels_chw[5], 1.0F),
          "sana wm vae input: blue channel is normalized CHW");
}

void test_camera_conditions_match_upstream_shapes_and_raymap() {
    const auto poses = trtmc::sana_wm_action_to_c2w("w-2", 1.0F, 0.0F);
    const std::vector<trtmc::SanaWmIntrinsics> intrinsics(poses.size(), {2.0F, 2.0F, 0.0F, 0.0F});

    const auto conditions = trtmc::sana_wm_prepare_camera_conditions(poses, intrinsics, 4, 4, 2, 2);

    check(conditions.num_frames == 3, "sana wm camera: frame count propagated");
    check(conditions.latent_frames == 2, "sana wm camera: latent frame count");
    check(conditions.latent_height == 2 && conditions.latent_width == 2,
          "sana wm camera: latent spatial shape");
    check(conditions.time_indices == std::vector<int32_t>({0, 2}),
          "sana wm camera: latent time indices match upstream arange stride");
    check(conditions.raymap.size() == 40, "sana wm camera: raymap is chunks x 20");
    check(near(conditions.raymap[0], 1.0F) && near(conditions.raymap[5], 1.0F) &&
              near(conditions.raymap[10], 1.0F) && near(conditions.raymap[15], 1.0F),
          "sana wm camera: first raymap pose row is identity");
    check(near(conditions.raymap[16], 1.0F) && near(conditions.raymap[17], 1.0F) &&
              near(conditions.raymap[18], 0.0F) && near(conditions.raymap[19], 0.0F),
          "sana wm camera: intrinsics scaled to latent grid");
    check(conditions.chunk_plucker_channels == 12,
          "sana wm camera: chunk plucker channel count is stride times six");
    check(conditions.chunk_plucker.size() == 96, "sana wm camera: chunk plucker shape");

    const auto z_dir = chunk_plucker_offset(2, 0, 0, 0, 2, 2, 2);
    check(near(conditions.chunk_plucker[z_dir], 1.0F), "sana wm camera: first pixel looks down +Z");

    const auto x_dir = chunk_plucker_offset(0, 0, 0, 1, 2, 2, 2);
    check(near(conditions.chunk_plucker[x_dir], 1.0F / std::sqrt(2.0F)),
          "sana wm camera: x ray direction is normalized");
}

void test_camera_conditions_relativize_to_first_pose() {
    trtmc::SanaWmPose first;
    first.c2w = {1.0F, 0.0F, 0.0F, 10.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                 0.0F, 0.0F, 1.0F, 5.0F,  0.0F, 0.0F, 0.0F, 1.0F};
    trtmc::SanaWmPose second;
    second.c2w = {1.0F, 0.0F, 0.0F, 10.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                  0.0F, 0.0F, 1.0F, 7.0F,  0.0F, 0.0F, 0.0F, 1.0F};

    const std::vector<trtmc::SanaWmPose> poses{first, second};
    const std::vector<trtmc::SanaWmIntrinsics> intrinsics{{1.0F, 1.0F, 0.0F, 0.0F}};
    const auto conditions = trtmc::sana_wm_prepare_camera_conditions(poses, intrinsics, 2, 2, 1, 1);

    check(conditions.time_indices == std::vector<int32_t>({0, 1}),
          "sana wm camera: one latent frame per source frame");
    check(near(conditions.raymap[0], 1.0F) && near(conditions.raymap[5], 1.0F) &&
              near(conditions.raymap[10], 1.0F) && near(conditions.raymap[15], 1.0F),
          "sana wm camera: first relative pose is identity");
    check(near(conditions.raymap[20 + 11], 2.0F),
          "sana wm camera: second pose is relative to first pose");
}

void test_stage1_latents_anchor_first_frame() {
    const std::vector<float> first_frame{10.0F, 11.0F, 20.0F, 21.0F};
    const std::vector<float> initial{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};

    const auto latents =
        trtmc::sana_wm_prepare_stage1_latents(first_frame, initial, 2, 2, 1, 2, 42);

    check(latents.channels == 2 && latents.frames == 2 && latents.height == 1 && latents.width == 2,
          "sana wm latents: shape metadata propagated");
    check(latents.values.size() == 8, "sana wm latents: output size is CTHW");
    check(near(latents.values[0], 10.0F) && near(latents.values[1], 11.0F),
          "sana wm latents: channel zero first frame anchored");
    check(near(latents.values[4], 20.0F) && near(latents.values[5], 21.0F),
          "sana wm latents: channel one first frame anchored");
    check(near(latents.values[2], 2.0F) && near(latents.values[3], 3.0F) &&
              near(latents.values[6], 6.0F) && near(latents.values[7], 7.0F),
          "sana wm latents: later frames preserve caller noise");
}

void test_stage1_latents_seeded_noise_is_deterministic() {
    const std::vector<float> first_frame{1.0F, 2.0F};

    const auto a = trtmc::sana_wm_prepare_stage1_latents(first_frame, {}, 1, 3, 1, 2, 1234);
    const auto b = trtmc::sana_wm_prepare_stage1_latents(first_frame, {}, 1, 3, 1, 2, 1234);

    check(a.values == b.values, "sana wm latents: seeded noise is deterministic");
    check(near(a.values[0], 1.0F) && near(a.values[1], 2.0F),
          "sana wm latents: seeded path still anchors first frame");
    check(!(near(a.values[2], 0.0F) && near(a.values[3], 0.0F)),
          "sana wm latents: seeded path fills later frames");
}

void test_stage1_latents_reject_mismatched_buffers() {
    bool rejected_first = false;
    try {
        (void)trtmc::sana_wm_prepare_stage1_latents({1.0F}, {}, 2, 1, 1, 1, 0);
    } catch (const std::invalid_argument&) {
        rejected_first = true;
    }
    check(rejected_first, "sana wm latents: mismatched first latent rejected");

    bool rejected_initial = false;
    try {
        (void)trtmc::sana_wm_prepare_stage1_latents({1.0F}, {0.0F, 1.0F}, 1, 1, 1, 1, 0);
    } catch (const std::invalid_argument&) {
        rejected_initial = true;
    }
    check(rejected_initial, "sana wm latents: mismatched initial latents rejected");
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

void test_native_module_sections_do_not_fall_back_to_bridge() {
    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";

    trtmc::SanaWmNativeModules modules;
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>();
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F, 0.2F, 0.3F, 0.4F,
            0.5F, 0.6F, 0.7F, 0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});

    auto runner = std::make_shared<FakeSubprocessRunner>();
    trtmc::SanaWmPipeline pipeline(cfg, "/usr/bin/python3", runner, std::move(modules));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = "/tmp/trtmc_sana_wm_missing_input.png";

    bool native_input_error_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        native_input_error_reported =
            std::string(exc.what()).find("failed to load image") != std::string::npos;
    }

    check(pipeline.has_native_modules(), "sana wm native: modules recorded");
    check(!pipeline.has_native_stage1(), "sana wm native: partial stage1 is not complete");
    check(native_input_error_reported, "sana wm native: native input errors reported");
    check(runner->call_count == 0, "sana wm native: bridge not used when native sections exist");
}

void test_native_input_preparation_reaches_solver_boundary() {
    const auto image_path =
        std::filesystem::temp_directory_path() / "trtmc_sana_wm_native_input_test.png";
    trtmc::io::save_png(image_path.string(),
                        {
                            1.0F, 0.0F, 0.0F,
                            0.0F, 1.0F, 0.0F,
                            0.0F, 0.0F, 1.0F,
                            1.0F, 1.0F, 1.0F,
                        },
                        2, 2);

    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";
    cfg.height = 4;
    cfg.width = 4;
    cfg.num_frames = 2;
    cfg.vae_latent_dim = 2;
    cfg.vae_time_stride = 1;
    cfg.vae_spatial_stride = 2;

    trtmc::SanaWmNativeModules modules;
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>();
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F, 0.2F, 0.3F, 0.4F,
            0.5F, 0.6F, 0.7F, 0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});

    auto runner = std::make_shared<FakeSubprocessRunner>();
    trtmc::SanaWmPipeline pipeline(cfg, "/usr/bin/python3", runner, std::move(modules));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = image_path.string();
    gen_cfg.camera_action = "w-1";
    gen_cfg.camera_intrinsics = {4.0F, 4.0F, 2.0F, 2.0F};
    gen_cfg.num_frames = 2;

    bool solver_boundary_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        solver_boundary_reported = std::string(exc.what()).find(
                                       "text encoding/solver/refiner execution is not implemented") !=
                                   std::string::npos;
    }

    check(solver_boundary_reported, "sana wm native: input prep reaches solver boundary");
    check(runner->call_count == 0, "sana wm native: prepared inputs without bridge");

    std::error_code ec;
    std::filesystem::remove(image_path, ec);
}

} // namespace

int main() {
    test_action_rollout_matches_model_card_frame_count_and_translation();
    test_action_rollout_rejects_invalid_segments();
    test_camera_pose_vector_parses_row_major_matrices();
    test_intrinsics_expand_model_card_shapes();
    test_runtime_config_parses_native_sana_wm_fields();
    test_conditioning_prompt_matches_upstream_chi_prefix();
    test_resize_crop_plan_matches_upstream_geometry();
    test_resize_center_crop_crops_hwc_pixels();
    test_prepare_vae_input_image_matches_upstream_tensor_layout();
    test_camera_conditions_match_upstream_shapes_and_raymap();
    test_camera_conditions_relativize_to_first_pose();
    test_stage1_latents_anchor_first_frame();
    test_stage1_latents_seeded_noise_is_deterministic();
    test_stage1_latents_reject_mismatched_buffers();
    test_bridge_command_forwards_strict_sana_wm_contract();
    test_native_module_sections_do_not_fall_back_to_bridge();
    test_native_input_preparation_reaches_solver_boundary();
    return failures == 0 ? 0 : 1;
}
