#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/domains/audio/subprocess_runner.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct SanaWmRuntimeConfig {
    std::string hf_id{"Efficient-Large-Model/SANA-WM_bidirectional"};
    std::string script_path;
    std::string default_image;
    std::string action{"w-80,jw-40,w-40,lw-60,w-100"};
    float translation_speed{0.055F};
    float rotation_speed_deg{1.2F};
    int32_t num_frames{321};
    int32_t height{704};
    int32_t width{1280};
    int32_t fps{16};
    int32_t num_steps{60};
    float cfg_scale{5.0F};
    float flow_shift{9.8F};
    int32_t seed{42};
    int32_t vae_latent_dim{128};
    int32_t vae_time_stride{8};
    int32_t vae_spatial_stride{32};
    int32_t text_encoder_max_length{300};
    int32_t text_encoder_dim{2304};
    std::string chi_prompt;
    bool require_official_script{false};
};

SanaWmRuntimeConfig parse_sana_wm_config(const std::string& config_json);

struct SanaWmPose {
    // Row-major 4x4 camera-to-world matrix, matching the upstream .npy layout.
    std::array<float, 16> c2w{};
};

struct SanaWmResizeCropPlan {
    int32_t src_width{0};
    int32_t src_height{0};
    int32_t resized_width{0};
    int32_t resized_height{0};
    int32_t crop_left{0};
    int32_t crop_top{0};
    int32_t target_width{0};
    int32_t target_height{0};
};

struct SanaWmIntrinsics {
    float fx{0.0F};
    float fy{0.0F};
    float cx{0.0F};
    float cy{0.0F};
};

struct SanaWmPreprocessedImage {
    std::vector<float> pixels_hwc;
    SanaWmResizeCropPlan plan;
    bool ok{false};
};

struct SanaWmVaeInputImage {
    // Flat [3, H, W] float tensor in [-1, 1], matching upstream ToTensor()*2-1
    // after aspect-preserving resize and center crop.
    std::vector<float> pixels_chw;
    SanaWmResizeCropPlan plan;
    int32_t height{0};
    int32_t width{0};
    int32_t channels{3};
    bool ok{false};
};

struct SanaWmCameraConditions {
    std::vector<float> raymap;
    std::vector<float> chunk_plucker;
    std::vector<int32_t> time_indices;
    int32_t num_frames{0};
    int32_t latent_frames{0};
    int32_t latent_height{0};
    int32_t latent_width{0};
    int32_t vae_time_stride{0};
    int32_t vae_spatial_stride{0};
    int32_t raymap_width{20};
    int32_t chunk_plucker_channels{0};
};

struct SanaWmStage1Latents {
    // Flat [C, T, H, W] row-major tensor matching upstream Stage-1 latent shape
    // after dropping the implicit batch dimension.
    std::vector<float> values;
    int32_t channels{0};
    int32_t frames{0};
    int32_t height{0};
    int32_t width{0};
};

struct SanaWmNativeModules {
    std::unique_ptr<ITrtModule> text_encoder;
    std::unique_ptr<ITrtModule> stage1_denoiser;
    std::unique_ptr<ITrtModule> vae_encoder;
    std::unique_ptr<ITrtModule> vae_decoder;
    std::unique_ptr<ITrtModule> refiner_text_encoder;
    std::unique_ptr<ITrtModule> refiner_denoiser;
    std::unique_ptr<ITrtModule> refiner_vae_decoder;

    bool has_any() const;
    bool has_stage1() const;
    bool has_refiner() const;
};

std::vector<SanaWmPose> sana_wm_action_to_c2w(const std::string& action, float translation_speed,
                                              float rotation_speed_deg);
std::string sana_wm_make_conditioning_prompt(const std::string& prompt,
                                             const std::string& chi_prompt);
std::vector<SanaWmPose> sana_wm_row_major_c2w_to_poses(const std::vector<float>& c2w_values);
std::vector<SanaWmIntrinsics> sana_wm_expand_intrinsics(const std::vector<float>& values,
                                                        int32_t num_frames);

SanaWmResizeCropPlan sana_wm_make_resize_crop_plan(int32_t src_width, int32_t src_height,
                                                   int32_t target_height, int32_t target_width);
SanaWmIntrinsics sana_wm_transform_intrinsics_for_crop(const SanaWmIntrinsics& intrinsics,
                                                       const SanaWmResizeCropPlan& plan);
SanaWmPreprocessedImage sana_wm_resize_and_center_crop(const std::vector<float>& src_hwc,
                                                       int32_t src_width, int32_t src_height,
                                                       int32_t target_height, int32_t target_width);
SanaWmVaeInputImage sana_wm_prepare_vae_input_image(const std::vector<float>& src_hwc,
                                                    int32_t src_width, int32_t src_height,
                                                    int32_t target_height, int32_t target_width);
SanaWmCameraConditions
sana_wm_prepare_camera_conditions(const std::vector<SanaWmPose>& c2w,
                                  const std::vector<SanaWmIntrinsics>& intrinsics,
                                  int32_t target_height, int32_t target_width,
                                  int32_t vae_time_stride, int32_t vae_spatial_stride);
SanaWmStage1Latents sana_wm_prepare_stage1_latents(const std::vector<float>& first_frame_chw,
                                                   const std::vector<float>& initial_latents_cthw,
                                                   int32_t channels, int32_t latent_frames,
                                                   int32_t latent_height, int32_t latent_width,
                                                   uint64_t seed);

class SanaWmPipeline final : public IPipeline {
  public:
    SanaWmPipeline(SanaWmRuntimeConfig config, std::string hf_python,
                   std::shared_ptr<ISubprocessRunner> subprocess_runner = nullptr,
                   SanaWmNativeModules native_modules = {},
                   std::shared_ptr<ITokenizer> tokenizer = nullptr);

    const char* model_id() const override { return config_.hf_id.c_str(); }
    const char* pipeline_type() const override { return "SanaWmPipeline"; }

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    bool has_native_modules() const { return native_modules_.has_any(); }
    bool has_native_stage1() const { return native_modules_.has_stage1(); }
    bool has_native_refiner() const { return native_modules_.has_refiner(); }

  private:
    SanaWmRuntimeConfig config_;
    std::string hf_python_;
    std::shared_ptr<ISubprocessRunner> subprocess_runner_;
    SanaWmNativeModules native_modules_;
    std::shared_ptr<ITokenizer> tokenizer_;
};

} // namespace trtmc
