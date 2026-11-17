/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/stable_video_diffusion/runtime/runtime_config.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::stable_video_diffusion {

// Creates the engine of one bundle section; the pipeline keeps it only for the
// stage that runs it, so the stages' device memory is not held together.
using ModuleLoader = std::function<std::unique_ptr<ITrtModule>(const std::string& section)>;

inline constexpr const char* kImageEncoderPlan = "image_encoder.plan";
inline constexpr const char* kVaeEncoderPlan = "vae_encoder.plan";
inline constexpr const char* kUnetPlan = "unet.plan";
inline constexpr const char* kVaeDecoderPlan = "vae_decoder.plan";

// Resolved per-request settings of StableVideoDiffusionPipeline.__call__.
struct GenerationSettings {
    std::uint32_t seed{0};
    std::int32_t num_steps{0};
    float min_guidance_scale{0.0F};
    float max_guidance_scale{0.0F};
    // Classifier-free guidance runs when max_guidance_scale exceeds one.
    bool guided{true};
    std::int32_t fps{0};
    std::int32_t motion_bucket_id{0};
    float noise_aug_strength{0.0F};
};

// Image-to-video generation: the initial image conditions every frame through
// its CLIP embedding and its VAE latent; the prompt must be empty because the
// model has no text conditioning. Every returned frame is generated, so no
// frame is reported as conditioning context.
class Pipeline final : public internal::IModel, public internal::IInitialImageTextToVideo {
  public:
    Pipeline(ModuleLoader loader, RuntimeConfig config);

    const char* task() const noexcept override {
        return internal::IInitialImageTextToVideo::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::VideoResult run(const internal::InitialImageTextToVideoRequest& request,
                              internal::ConfigView config) override;

    GenerationSettings resolve(internal::ConfigView config) const;

  private:
    std::unique_ptr<ITrtModule> load(const char* section) const;
    std::vector<float> encode_image(const std::vector<float>& pixels);
    std::vector<float> encode_latent(const std::vector<float>& noisy_image);
    void denoise(std::vector<float>& latents, const std::vector<float>& image_embedding,
                 const std::vector<float>& image_latent, const GenerationSettings& settings);
    ImageResult decode(const std::vector<float>& latents);

    ModuleLoader loader_;
    RuntimeConfig config_;
    std::vector<internal::ConfigField> fields_;
    std::mutex generation_mutex_;
};

} // namespace trtmc::stable_video_diffusion
