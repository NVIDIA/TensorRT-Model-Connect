// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SANAWM-CPP-01
// Architecture:   ARCH-RUNTIME-001
// Unit Design:    UD-SANAWM-01
// Intent:         SANA-WM C++ runtime enforces native TensorRT execution
// Preconditions:  Bundle has complete native component sections
// Postconditions: Runtime decodes without Python subprocess fallback
// =============================================================================

#include "../../src/runtime/models/sana_wm/pipeline.h"
#include "trtmc/trtmc_io.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

std::vector<float> copy_float_tensor(const trtmc::Tensor& tensor) {
    if (tensor.dtype != trtmc::DType::kFloat32 || tensor.data == nullptr)
        return {};
    const auto* data = static_cast<const float*>(tensor.data);
    return std::vector<float>(data, data + tensor.numel());
}

std::vector<int32_t> copy_int_tensor(const trtmc::Tensor& tensor) {
    if (tensor.dtype != trtmc::DType::kInt32 || tensor.data == nullptr)
        return {};
    const auto* data = static_cast<const int32_t*>(tensor.data);
    return std::vector<int32_t>(data, data + tensor.numel());
}

std::size_t stage1_bcthw_index(int32_t batch, int32_t channel, int32_t frame, int32_t y, int32_t x,
                               int32_t channels, int32_t frames, int32_t height, int32_t width) {
    return ((((static_cast<std::size_t>(batch) * static_cast<std::size_t>(channels) +
               static_cast<std::size_t>(channel)) *
                  static_cast<std::size_t>(frames) +
              static_cast<std::size_t>(frame)) *
                 static_cast<std::size_t>(height) +
             static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x));
}

class FakeTrtModule final : public trtmc::ITrtModule {
  public:
    FakeTrtModule() = default;
    FakeTrtModule(std::vector<float> output, std::vector<int64_t> shape)
        : output_(std::move(output)), output_shape_(std::move(shape)), input_names_({"sample"}) {}
    FakeTrtModule(std::vector<float> output, std::vector<int64_t> shape,
                  std::vector<std::string> input_names)
        : output_(std::move(output)), output_shape_(std::move(shape)),
          input_names_(std::move(input_names)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++call_count;
        last_input_shapes.clear();
        last_input_dtypes.clear();
        for (const auto& [name, tensor] : inputs) {
            last_input_shapes[name] = tensor.shape;
            last_input_dtypes[name] = tensor.dtype;
        }
        input_value_calls.push_back({});
        auto& values = input_value_calls.back();
        for (const auto& [name, tensor] : inputs)
            values[name] = copy_float_tensor(tensor);
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
        std::vector<trtmc::TensorInfo> out;
        out.reserve(input_names_.size());
        for (const auto& name : input_names_)
            out.push_back(
                {name, {}, name == "mask" ? trtmc::DType::kInt32 : trtmc::DType::kFloat32, true});
        return out;
    }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return !output_.empty() &&
               std::find(input_names_.begin(), input_names_.end(), name) != input_names_.end();
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

    int call_count{0};
    std::unordered_map<std::string, std::vector<int64_t>> last_input_shapes;
    std::unordered_map<std::string, trtmc::DType> last_input_dtypes;
    std::vector<std::unordered_map<std::string, std::vector<float>>> input_value_calls;

  private:
    std::vector<float> output_;
    std::vector<int64_t> output_shape_;
    std::vector<std::string> input_names_;
};

class FakeDecoderTextModule final : public trtmc::ITrtModule {
  public:
    explicit FakeDecoderTextModule(int64_t max_tokens = 2) : max_tokens_(max_tokens) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++call_count;
        last_input_shapes.clear();
        last_input_dtypes.clear();
        input_value_calls.push_back({});
        auto& values = input_value_calls.back();
        for (const auto& [name, tensor] : inputs) {
            last_input_shapes[name] = tensor.shape;
            last_input_dtypes[name] = tensor.dtype;
            values[name] = copy_float_tensor(tensor);
        }

        const auto token_values = copy_int_tensor(inputs.at("token_id"));
        const auto position_values = copy_int_tensor(inputs.at("position_id"));
        token_ids.push_back(token_values.empty() ? -1 : token_values.front());
        position_ids.push_back(position_values.empty() ? -1 : position_values.front());

        hidden_state_ = {static_cast<float>(token_ids.back() * 10 + position_ids.back()),
                         static_cast<float>(token_ids.back() * 10 + position_ids.back() + 1)};
        present_k_ = {100.0F + static_cast<float>(position_ids.back()),
                      101.0F + static_cast<float>(position_ids.back())};
        present_v_ = {200.0F + static_cast<float>(position_ids.back()),
                      201.0F + static_cast<float>(position_ids.back())};
        return {
            {"hidden_state", trtmc::Tensor{hidden_state_.data(), {1, 2}, trtmc::DType::kFloat32}},
            {"present_k_0", trtmc::Tensor{present_k_.data(), {1, 2}, trtmc::DType::kFloat32}},
            {"present_v_0", trtmc::Tensor{present_v_.data(), {1, 2}, trtmc::DType::kFloat32}}};
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
        return {
            {"token_id", {1}, trtmc::DType::kInt32, true},
            {"position_id", {1}, trtmc::DType::kInt32, true},
            {"attention_mask", {1, max_tokens_ + 1}, trtmc::DType::kFloat32, true},
            {"cache_k_0", {max_tokens_, 2}, trtmc::DType::kFloat32, true},
            {"cache_v_0", {max_tokens_, 2}, trtmc::DType::kFloat32, true},
        };
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"hidden_state", {1, 2}, trtmc::DType::kFloat32, false},
                {"present_k_0", {1, 2}, trtmc::DType::kFloat32, false},
                {"present_v_0", {1, 2}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        for (const auto& info : input_info()) {
            if (info.name == name)
                return true;
        }
        return false;
    }
    bool has_output(const std::string& name) const override {
        for (const auto& info : output_info()) {
            if (info.name == name)
                return true;
        }
        return false;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        for (const auto& info : input_info()) {
            if (info.name == name)
                return info.dtype;
        }
        for (const auto& info : output_info()) {
            if (info.name == name)
                return info.dtype;
        }
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        for (const auto& info : input_info()) {
            if (info.name == name)
                return info.shape;
        }
        for (const auto& info : output_info()) {
            if (info.name == name)
                return info.shape;
        }
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    int call_count{0};
    std::vector<int32_t> token_ids;
    std::vector<int32_t> position_ids;
    std::unordered_map<std::string, std::vector<int64_t>> last_input_shapes;
    std::unordered_map<std::string, trtmc::DType> last_input_dtypes;
    std::vector<std::unordered_map<std::string, std::vector<float>>> input_value_calls;

  private:
    int64_t max_tokens_{2};
    std::vector<float> hidden_state_;
    std::vector<float> present_k_;
    std::vector<float> present_v_;
};

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        if (text.empty())
            return {};
        return {1, 2};
    }
    std::string decode(const std::vector<int32_t>&) const override { return ""; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
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
        1.0F, 0.0F, 0.0F, 0.1F, 0.0F, 1.0F, 0.0F,  0.2F, 0.0F, 0.0F, 1.0F,
        0.3F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, -1.0F, 0.0F, 1.1F, 1.0F, 0.0F,
        0.0F, 1.2F, 0.0F, 0.0F, 1.0F, 1.3F, 0.0F,  0.0F, 0.0F, 1.0F,
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

    const auto matrix = trtmc::sana_wm_expand_intrinsics(
        {
            20.0F,
            0.0F,
            8.0F,
            0.0F,
            21.0F,
            9.0F,
            0.0F,
            0.0F,
            1.0F,
        },
        3);
    check(matrix.size() == 3 && near(matrix[2].fx, 20.0F) && near(matrix[2].fy, 21.0F) &&
              near(matrix[2].cx, 8.0F) && near(matrix[2].cy, 9.0F),
          "sana wm intrinsics: single 3x3 matrix expands to all frames");

    const auto per_frame = trtmc::sana_wm_expand_intrinsics(
        {
            30.0F,
            0.0F,
            12.0F,
            0.0F,
            31.0F,
            13.0F,
            0.0F,
            0.0F,
            1.0F,
            40.0F,
            0.0F,
            14.0F,
            0.0F,
            41.0F,
            15.0F,
            0.0F,
            0.0F,
            1.0F,
        },
        2);
    check(per_frame.size() == 2 && near(per_frame[0].fx, 30.0F) && near(per_frame[1].fx, 40.0F) &&
              near(per_frame[1].cy, 15.0F),
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
          "sana_wm_dit_text_embed_dim": 2304,
          "sana_wm_chi_prompt": "Generate an \"Enhanced prompt\".\nUser Prompt: ",
          "sana_wm_default_intrinsics": [797.87866, 830.0503, 844.2675, 463.7225]
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
    check(cfg.text_encoder_dim == 2304, "sana wm config: text encoder dim parsed");
    check(cfg.chi_prompt == "Generate an \"Enhanced prompt\".\nUser Prompt: ",
          "sana wm config: chi prompt parsed");
    check(cfg.default_intrinsics.size() == 4 && near(cfg.default_intrinsics[0], 797.87866F) &&
              near(cfg.default_intrinsics[3], 463.7225F),
          "sana wm config: default demo intrinsics parsed");
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

void test_pipeline_requires_native_tensor_rt_modules() {
    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";

    trtmc::SanaWmPipeline pipeline(cfg);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = "asset/sana_wm/demo_0.png";

    bool native_required_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        native_required_reported =
            std::string(exc.what()).find("pure C++ execution requires native TensorRT plan") !=
            std::string::npos;
    }

    check(native_required_reported, "sana wm native: missing native plans rejected");
}

void test_native_module_sections_require_complete_native_set() {
    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";

    trtmc::SanaWmNativeModules modules;
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>();
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = "/tmp/trtmc_sana_wm_missing_input.png";

    bool incomplete_stage1_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        incomplete_stage1_reported =
            std::string(exc.what()).find("complete stage1 module set") != std::string::npos;
    }

    check(pipeline.has_native_modules(), "sana wm native: modules recorded");
    check(!pipeline.has_native_stage1(), "sana wm native: partial stage1 is not complete");
    check(incomplete_stage1_reported, "sana wm native: incomplete module set reported");
}

void test_native_default_requires_refiner_plan_set() {
    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";

    trtmc::SanaWmNativeModules modules;
    modules.text_encoder = std::make_unique<FakeTrtModule>();
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>();
    modules.vae_encoder = std::make_unique<FakeTrtModule>();
    modules.vae_decoder = std::make_unique<FakeTrtModule>();

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules), std::make_shared<FakeTokenizer>());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = "/tmp/trtmc_sana_wm_default_requires_refiner.png";

    bool refiner_required_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        const std::string message = exc.what();
        refiner_required_reported =
            message.find("native refiner execution requires") != std::string::npos &&
            message.find("--no_refiner") != std::string::npos;
    }

    check(refiner_required_reported,
          "sana wm native: model-card default requires refiner plan set");
}

void test_native_stage1_solver_decodes_with_native_modules() {
    const auto image_path =
        std::filesystem::temp_directory_path() / "trtmc_sana_wm_native_input_test.png";
    trtmc::io::save_png(image_path.string(),
                        {
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            1.0F,
                            1.0F,
                            1.0F,
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
    cfg.text_encoder_max_length = 2;
    cfg.text_encoder_dim = 2;
    cfg.num_steps = 2;
    cfg.default_intrinsics = {2.0F, 2.0F, 1.0F, 1.0F};

    trtmc::SanaWmNativeModules modules;
    modules.text_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, std::vector<int64_t>{1, 2, 2},
        std::vector<std::string>{"input_ids", "attention_mask"});
    auto denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(32, 0.25F), std::vector<int64_t>{2, 2, 2, 2, 2},
        std::vector<std::string>{"x", "timestep", "y", "mask", "camera_conditions",
                                 "chunk_plucker"});
    auto* denoiser_ptr = denoiser.get();
    modules.stage1_denoiser = std::move(denoiser);
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});
    auto decoder = std::make_unique<FakeTrtModule>(std::vector<float>(96, 0.0F),
                                                   std::vector<int64_t>{1, 3, 2, 4, 4},
                                                   std::vector<std::string>{"latents"});
    auto* decoder_ptr = decoder.get();
    modules.vae_decoder = std::move(decoder);

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules), std::make_shared<FakeTokenizer>());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = image_path.string();
    gen_cfg.camera_action = "w-1";
    gen_cfg.num_frames = 2;
    gen_cfg.no_refiner = true;

    const auto result = pipeline.generate_image("drive forward", gen_cfg);

    check(denoiser_ptr->call_count == 2, "sana wm native: denoiser invoked per solver step");
    check(decoder_ptr->call_count == 1, "sana wm native: VAE decoder invoked once");
    check(result.num_frames == 2 && result.height == 4 && result.width == 4,
          "sana wm native: decoded video dimensions");
    check(result.pixels.size() == 96 && near(result.pixels[0], 0.5F),
          "sana wm native: VAE decoder output converted to pixels");
    check(denoiser_ptr->last_input_shapes["x"] == std::vector<int64_t>({2, 2, 2, 2, 2}),
          "sana wm native: denoiser latent shape");
    check(denoiser_ptr->last_input_shapes["timestep"] == std::vector<int64_t>({2, 1, 2}),
          "sana wm native: denoiser timestep shape");
    check(denoiser_ptr->last_input_shapes["y"] == std::vector<int64_t>({2, 1, 2, 2}),
          "sana wm native: denoiser text shape");
    check(denoiser_ptr->last_input_shapes["mask"] == std::vector<int64_t>({2, 2}),
          "sana wm native: denoiser mask shape");
    check(denoiser_ptr->last_input_shapes["camera_conditions"] == std::vector<int64_t>({2, 2, 20}),
          "sana wm native: denoiser camera shape");
    check(denoiser_ptr->last_input_shapes["chunk_plucker"] == std::vector<int64_t>({2, 6, 2, 2, 2}),
          "sana wm native: denoiser chunk plucker shape");
    check(denoiser_ptr->last_input_dtypes["mask"] == trtmc::DType::kInt32,
          "sana wm native: denoiser mask dtype");
    check(decoder_ptr->last_input_shapes["latents"] == std::vector<int64_t>({1, 2, 2, 2, 2}),
          "sana wm native: VAE decoder latent shape");
    check(denoiser_ptr->input_value_calls.size() == 2,
          "sana wm native: denoiser input values recorded per step");
    if (denoiser_ptr->input_value_calls.size() == 2) {
        const auto& first_x = denoiser_ptr->input_value_calls[0]["x"];
        const auto& second_x = denoiser_ptr->input_value_calls[1]["x"];
        const auto& first_t = denoiser_ptr->input_value_calls[0]["timestep"];
        check(first_t.size() == 4 && near(first_t[0], 0.0F) && near(first_t[2], 0.0F),
              "sana wm native: anchor frame timestep is zero for CFG batches");
        check(first_x.size() == 32 && second_x.size() == 32,
              "sana wm native: denoiser latent values captured");
        if (first_x.size() == 32 && second_x.size() == 32) {
            const auto anchor_idx = stage1_bcthw_index(0, 0, 0, 0, 0, 2, 2, 2, 2);
            const auto moving_idx = stage1_bcthw_index(0, 0, 1, 0, 0, 2, 2, 2, 2);
            check(near(first_x[anchor_idx], second_x[anchor_idx]),
                  "sana wm native: solver keeps first latent frame anchored");
            check(!near(first_x[moving_idx], second_x[moving_idx]),
                  "sana wm native: solver updates non-anchor latent frames");
        }
    }

    std::error_code ec;
    std::filesystem::remove(image_path, ec);
}

void test_native_stage1_accepts_decoder_style_text_encoder() {
    const auto image_path =
        std::filesystem::temp_directory_path() / "trtmc_sana_wm_decoder_text_input_test.png";
    trtmc::io::save_png(image_path.string(),
                        {
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            1.0F,
                            1.0F,
                            1.0F,
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
    cfg.text_encoder_max_length = 2;
    cfg.text_encoder_dim = 2;
    cfg.num_steps = 1;
    cfg.default_intrinsics = {2.0F, 2.0F, 1.0F, 1.0F};

    trtmc::SanaWmNativeModules modules;
    auto text_encoder = std::make_unique<FakeDecoderTextModule>();
    auto* text_encoder_ptr = text_encoder.get();
    modules.text_encoder = std::move(text_encoder);
    auto denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(32, 0.25F), std::vector<int64_t>{2, 2, 2, 2, 2},
        std::vector<std::string>{"x", "timestep", "y", "mask", "camera_conditions",
                                 "chunk_plucker"});
    auto* denoiser_ptr = denoiser.get();
    modules.stage1_denoiser = std::move(denoiser);
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});
    modules.vae_decoder = std::make_unique<FakeTrtModule>(std::vector<float>(96, 0.0F),
                                                          std::vector<int64_t>{1, 3, 2, 4, 4},
                                                          std::vector<std::string>{"latents"});

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules), std::make_shared<FakeTokenizer>());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = image_path.string();
    gen_cfg.camera_action = "w-1";
    gen_cfg.num_frames = 2;
    gen_cfg.no_refiner = true;

    const auto result = pipeline.generate_image("drive forward", gen_cfg);

    check(result.num_frames == 2, "sana wm decoder text: generated native video");
    check(text_encoder_ptr->call_count == 4,
          "sana wm decoder text: encoder invoked once per cond/negative token");
    check(text_encoder_ptr->token_ids == std::vector<int32_t>({1, 2, 0, 0}),
          "sana wm decoder text: token ids are fed sequentially");
    check(text_encoder_ptr->position_ids == std::vector<int32_t>({0, 1, 0, 1}),
          "sana wm decoder text: position ids reset for each prompt");
    check(text_encoder_ptr->last_input_shapes["token_id"] == std::vector<int64_t>({1}),
          "sana wm decoder text: token input shape");
    check(text_encoder_ptr->last_input_shapes["position_id"] == std::vector<int64_t>({1}),
          "sana wm decoder text: position input shape");
    check(text_encoder_ptr->last_input_shapes["attention_mask"] == std::vector<int64_t>({1, 3}),
          "sana wm decoder text: attention mask shape");
    check(text_encoder_ptr->last_input_shapes["cache_k_0"] == std::vector<int64_t>({2, 2}),
          "sana wm decoder text: cache K input shape");
    if (text_encoder_ptr->input_value_calls.size() >= 2) {
        const auto& first_mask = text_encoder_ptr->input_value_calls[0]["attention_mask"];
        const auto& second_mask = text_encoder_ptr->input_value_calls[1]["attention_mask"];
        check(first_mask.size() == 3 && near(first_mask[0], -10000.0F) &&
                  near(first_mask[1], -10000.0F) && near(first_mask[2], 0.0F),
              "sana wm decoder text: first token sees only current slot");
        check(second_mask.size() == 3 && near(second_mask[0], 0.0F) &&
                  near(second_mask[1], -10000.0F) && near(second_mask[2], 0.0F),
              "sana wm decoder text: second token sees first cache row and current slot");
        const auto& second_cache = text_encoder_ptr->input_value_calls[1]["cache_k_0"];
        check(second_cache.size() == 4 && near(second_cache[0], 100.0F) &&
                  near(second_cache[1], 101.0F) && near(second_cache[2], 0.0F) &&
                  near(second_cache[3], 0.0F),
              "sana wm decoder text: present K updates the next cache input");
    }
    if (!denoiser_ptr->input_value_calls.empty()) {
        const auto& text = denoiser_ptr->input_value_calls.front()["y"];
        check(text.size() == 8 && near(text[0], 0.0F) && near(text[1], 1.0F) &&
                  near(text[2], 1.0F) && near(text[3], 2.0F) && near(text[4], 10.0F) &&
                  near(text[5], 11.0F) && near(text[6], 21.0F) && near(text[7], 22.0F),
              "sana wm decoder text: hidden states feed stage1 text conditioning");
    }

    std::error_code ec;
    std::filesystem::remove(image_path, ec);
}

void test_native_refiner_decodes_and_drops_sink_frame() {
    const auto image_path =
        std::filesystem::temp_directory_path() / "trtmc_sana_wm_native_refiner_test.png";
    trtmc::io::save_png(image_path.string(),
                        {
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            1.0F,
                            1.0F,
                            1.0F,
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
    cfg.text_encoder_max_length = 2;
    cfg.text_encoder_dim = 2;
    cfg.num_steps = 1;
    cfg.default_intrinsics = {2.0F, 2.0F, 1.0F, 1.0F};

    trtmc::SanaWmNativeModules modules;
    modules.text_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, std::vector<int64_t>{1, 2, 2},
        std::vector<std::string>{"input_ids", "attention_mask"});
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(32, 0.25F), std::vector<int64_t>{2, 2, 2, 2, 2},
        std::vector<std::string>{"x", "timestep", "y", "mask", "camera_conditions",
                                 "chunk_plucker"});
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});
    auto refiner_text = std::make_unique<FakeTrtModule>(
        std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F}, std::vector<int64_t>{1, 2, 2},
        std::vector<std::string>{"input_ids", "attention_mask"});
    auto* refiner_text_ptr = refiner_text.get();
    modules.refiner_text_encoder = std::move(refiner_text);
    auto refiner_denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(8, 0.0F), std::vector<int64_t>{1, 4, 2},
        std::vector<std::string>{"latent", "clean_latent", "denoise_mask", "positions", "v_context",
                                 "sigma"});
    auto* refiner_denoiser_ptr = refiner_denoiser.get();
    modules.refiner_denoiser = std::move(refiner_denoiser);
    auto refiner_video = std::vector<float>(96, -1.0F);
    const auto refiner_cthw_index = [](int32_t channel, int32_t frame, int32_t y, int32_t x) {
        return (((static_cast<std::size_t>(channel) * 2U + static_cast<std::size_t>(frame)) * 4U +
                 static_cast<std::size_t>(y)) *
                    4U +
                static_cast<std::size_t>(x));
    };
    refiner_video[refiner_cthw_index(0, 1, 0, 0)] = -1.0F;
    refiner_video[refiner_cthw_index(1, 1, 0, 0)] = 0.0F;
    refiner_video[refiner_cthw_index(2, 1, 0, 0)] = 1.0F;
    auto refiner_decoder = std::make_unique<FakeTrtModule>(
        refiner_video, std::vector<int64_t>{1, 3, 2, 4, 4}, std::vector<std::string>{"latents"});
    auto* refiner_decoder_ptr = refiner_decoder.get();
    modules.refiner_vae_decoder = std::move(refiner_decoder);

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules), std::make_shared<FakeTokenizer>());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = image_path.string();
    gen_cfg.camera_action = "w-1";
    gen_cfg.num_frames = 2;

    const auto result = pipeline.generate_image("drive forward", gen_cfg);

    check(pipeline.has_native_stage1(),
          "sana wm native refiner: stage1 core is complete without stage1 VAE decoder");
    check(refiner_text_ptr->call_count == 1, "sana wm native refiner: text encoded once");
    check(refiner_denoiser_ptr->call_count == 3,
          "sana wm native refiner: distilled denoiser steps");
    check(refiner_decoder_ptr->call_count == 1, "sana wm native refiner: VAE decoded once");
    check(result.num_frames == 1 && result.pixels.size() == 48 && near(result.pixels[0], 0.0F) &&
              near(result.pixels[1], 0.5F) && near(result.pixels[2], 1.0F),
          "sana wm native refiner: drops sink CTHW frame and normalizes pixels");
    check(refiner_denoiser_ptr->last_input_shapes["latent"] == std::vector<int64_t>({1, 8, 2}),
          "sana wm native refiner: combined latent shape");
    check(refiner_denoiser_ptr->last_input_shapes["clean_latent"] ==
              std::vector<int64_t>({1, 8, 2}),
          "sana wm native refiner: clean latent shape");
    check(refiner_denoiser_ptr->last_input_shapes["denoise_mask"] ==
              std::vector<int64_t>({1, 8, 1}),
          "sana wm native refiner: denoise mask shape");
    check(refiner_denoiser_ptr->last_input_shapes["positions"] ==
              std::vector<int64_t>({1, 3, 8, 2}),
          "sana wm native refiner: positions shape");
    check(refiner_denoiser_ptr->last_input_shapes["v_context"] == std::vector<int64_t>({1, 2, 2}),
          "sana wm native refiner: text context shape");
    if (!refiner_denoiser_ptr->input_value_calls.empty()) {
        const auto& mask = refiner_denoiser_ptr->input_value_calls.front()["denoise_mask"];
        check(mask.size() == 8 && near(mask.front(), 0.0F) && near(mask.back(), 1.0F),
              "sana wm native refiner: mask freezes context and updates current tokens");
    }

    std::error_code ec;
    std::filesystem::remove(image_path, ec);
}

void test_native_no_refiner_uses_stage1_decoder_when_refiner_is_bundled() {
    const auto image_path =
        std::filesystem::temp_directory_path() / "trtmc_sana_wm_no_refiner_test.png";
    trtmc::io::save_png(image_path.string(),
                        {
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            1.0F,
                            1.0F,
                            1.0F,
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
    cfg.text_encoder_max_length = 2;
    cfg.text_encoder_dim = 2;
    cfg.num_steps = 1;
    cfg.default_intrinsics = {2.0F, 2.0F, 1.0F, 1.0F};

    trtmc::SanaWmNativeModules modules;
    modules.text_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, std::vector<int64_t>{1, 2, 2},
        std::vector<std::string>{"input_ids", "attention_mask"});
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(32, 0.25F), std::vector<int64_t>{2, 2, 2, 2, 2},
        std::vector<std::string>{"x", "timestep", "y", "mask", "camera_conditions",
                                 "chunk_plucker"});
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});
    auto stage1_decoder = std::make_unique<FakeTrtModule>(std::vector<float>(96, 0.0F),
                                                          std::vector<int64_t>{1, 3, 2, 4, 4},
                                                          std::vector<std::string>{"latents"});
    auto* stage1_decoder_ptr = stage1_decoder.get();
    modules.vae_decoder = std::move(stage1_decoder);

    auto refiner_text = std::make_unique<FakeTrtModule>(
        std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F}, std::vector<int64_t>{1, 2, 2},
        std::vector<std::string>{"input_ids", "attention_mask"});
    auto* refiner_text_ptr = refiner_text.get();
    modules.refiner_text_encoder = std::move(refiner_text);
    auto refiner_denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(8, 0.0F), std::vector<int64_t>{1, 4, 2},
        std::vector<std::string>{"latent", "clean_latent", "denoise_mask", "positions", "v_context",
                                 "sigma"});
    auto* refiner_denoiser_ptr = refiner_denoiser.get();
    modules.refiner_denoiser = std::move(refiner_denoiser);
    auto refiner_decoder = std::make_unique<FakeTrtModule>(std::vector<float>(96, 255.0F),
                                                           std::vector<int64_t>{2, 4, 4, 3},
                                                           std::vector<std::string>{"latents"});
    auto* refiner_decoder_ptr = refiner_decoder.get();
    modules.refiner_vae_decoder = std::move(refiner_decoder);

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules), std::make_shared<FakeTokenizer>());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = image_path.string();
    gen_cfg.camera_action = "w-1";
    gen_cfg.num_frames = 2;
    gen_cfg.no_refiner = true;

    const auto result = pipeline.generate_image("drive forward", gen_cfg);

    check(result.num_frames == 2 && result.pixels.size() == 96 && near(result.pixels[0], 0.5F),
          "sana wm no_refiner: decodes full stage1 video");
    check(stage1_decoder_ptr->call_count == 1, "sana wm no_refiner: stage1 decoder used");
    check(refiner_text_ptr->call_count == 0, "sana wm no_refiner: refiner text skipped");
    check(refiner_denoiser_ptr->call_count == 0, "sana wm no_refiner: refiner denoiser skipped");
    check(refiner_decoder_ptr->call_count == 0, "sana wm no_refiner: refiner VAE skipped");

    std::error_code ec;
    std::filesystem::remove(image_path, ec);
}

void test_native_refiner_accepts_decoder_style_text_encoder() {
    const auto image_path =
        std::filesystem::temp_directory_path() / "trtmc_sana_wm_refiner_decoder_text_test.png";
    trtmc::io::save_png(image_path.string(),
                        {
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            0.0F,
                            0.0F,
                            0.0F,
                            1.0F,
                            1.0F,
                            1.0F,
                            1.0F,
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
    cfg.text_encoder_max_length = 2;
    cfg.text_encoder_dim = 2;
    cfg.num_steps = 1;
    cfg.default_intrinsics = {2.0F, 2.0F, 1.0F, 1.0F};

    trtmc::SanaWmNativeModules modules;
    modules.text_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, std::vector<int64_t>{1, 2, 2},
        std::vector<std::string>{"input_ids", "attention_mask"});
    modules.stage1_denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(32, 0.25F), std::vector<int64_t>{2, 2, 2, 2, 2},
        std::vector<std::string>{"x", "timestep", "y", "mask", "camera_conditions",
                                 "chunk_plucker"});
    modules.vae_encoder = std::make_unique<FakeTrtModule>(
        std::vector<float>{
            0.1F,
            0.2F,
            0.3F,
            0.4F,
            0.5F,
            0.6F,
            0.7F,
            0.8F,
        },
        std::vector<int64_t>{1, 2, 1, 2, 2});
    auto refiner_text = std::make_unique<FakeDecoderTextModule>(256);
    auto* refiner_text_ptr = refiner_text.get();
    modules.refiner_text_encoder = std::move(refiner_text);
    auto refiner_denoiser = std::make_unique<FakeTrtModule>(
        std::vector<float>(8, 0.0F), std::vector<int64_t>{1, 4, 2},
        std::vector<std::string>{"latent", "clean_latent", "denoise_mask", "positions", "v_context",
                                 "sigma"});
    auto* refiner_denoiser_ptr = refiner_denoiser.get();
    modules.refiner_denoiser = std::move(refiner_denoiser);
    modules.refiner_vae_decoder = std::make_unique<FakeTrtModule>(
        std::vector<float>(96, 255.0F), std::vector<int64_t>{2, 4, 4, 3},
        std::vector<std::string>{"latents"});

    trtmc::SanaWmPipeline pipeline(cfg, std::move(modules), std::make_shared<FakeTokenizer>());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = image_path.string();
    gen_cfg.camera_action = "w-1";
    gen_cfg.num_frames = 2;

    const auto result = pipeline.generate_image("drive forward", gen_cfg);

    check(result.num_frames == 1, "sana wm refiner decoder text: generated refined video");
    check(refiner_text_ptr->call_count == 256,
          "sana wm refiner decoder text: encoder invoked once per refiner token");
    check(refiner_text_ptr->position_ids.front() == 0 &&
              refiner_text_ptr->position_ids.back() == 255,
          "sana wm refiner decoder text: positions span full refiner context");
    check(refiner_text_ptr->last_input_shapes["attention_mask"] == std::vector<int64_t>({1, 257}),
          "sana wm refiner decoder text: attention mask covers cache and current token");
    check(refiner_text_ptr->last_input_shapes["cache_k_0"] == std::vector<int64_t>({256, 2}),
          "sana wm refiner decoder text: cache input covers refiner context");
    check(refiner_denoiser_ptr->last_input_shapes["v_context"] == std::vector<int64_t>({1, 256, 2}),
          "sana wm refiner decoder text: inferred text context shape");
    if (!refiner_denoiser_ptr->input_value_calls.empty()) {
        const auto& text = refiner_denoiser_ptr->input_value_calls.front()["v_context"];
        check(text.size() == 512 && near(text[0], 10.0F) && near(text[1], 11.0F) &&
                  near(text[2], 21.0F) && near(text[3], 22.0F),
              "sana wm refiner decoder text: hidden states feed refiner conditioning");
    }

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
    test_pipeline_requires_native_tensor_rt_modules();
    test_native_module_sections_require_complete_native_set();
    test_native_default_requires_refiner_plan_set();
    test_native_stage1_solver_decodes_with_native_modules();
    test_native_stage1_accepts_decoder_style_text_encoder();
    test_native_refiner_decodes_and_drops_sink_frame();
    test_native_no_refiner_uses_stage1_decoder_when_refiner_is_bundled();
    test_native_refiner_accepts_decoder_style_text_encoder();
    return failures == 0 ? 0 : 1;
}
