/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/wan2_2_ti2v/runtime/options.h"
#include "families/wan2_2_ti2v/runtime/runtime_config.h"
#include "families/wan2_2_ti2v/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/task.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using Wan22ModuleLoader = std::function<std::unique_ptr<ITrtModule>(
    const std::string&, cudaStream_t, const std::vector<ModuleExternalBinding>&)>;

struct Wan22TI2VRuntimeShape {
    int32_t latent_frames{0};
    int32_t latent_height{0};
    int32_t latent_width{0};
    int32_t video_frames{0};
    int32_t video_height{0};
    int32_t video_width{0};
    std::size_t latent_count{0};
};

Wan22TI2VRuntimeShape make_wan22_runtime_shape(const Wan22TI2VRequest& request);

std::vector<ModuleExternalBinding>
make_wan22_vae_cache_bindings(const std::vector<void*>& input_addresses,
                              const std::vector<void*>& output_addresses,
                              const Wan22TI2VRuntimeShape& shape);

// One bundle, two tasks: text-to-video, and image-to-video when the bundle
// carries the first-frame VAE encoder. Bundles built for image_generation keep
// the legacy interface.
class Wan22TI2VPipeline final : public internal::IModel,
                                public IImageGeneration,
                                public internal::ITextToVideo,
                                public internal::IInitialImageTextToVideo {
  public:
    Wan22TI2VPipeline(Wan22ModuleLoader module_loader, std::unique_ptr<ITokenizer> tokenizer,
                      Wan22TI2VOptions options, wan2_2_ti2v::RuntimeConfig runtime_config,
                      std::string model_id, std::string primary_task = IImageGeneration::kTask,
                      bool image_encoder = false);
    ~Wan22TI2VPipeline() override;

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    ImageResult generate_image(const std::string& prompt,
                               const ImageGenerationConfig& cfg = {}) override;
    internal::VideoResult run(const internal::TextToVideoRequest& request,
                              internal::ConfigView config) override;
    internal::VideoResult run(const internal::InitialImageTextToVideoRequest& request,
                              internal::ConfigView config) override;

  private:
    // Callers hold generation_mutex_. A first frame selects image-to-video.
    ImageResult generate(const std::string& prompt, const Wan22TI2VRequest& request,
                         const std::vector<float>* first_frame_pixels);
    Wan22TI2VRequest resolve_config(internal::ConfigView config) const;
    std::vector<int32_t> tokenize(const std::string& text) const;
    std::vector<float> encode_text(const std::vector<int32_t>& ids, ITrtModule& text_encoder);
    std::vector<float> encode_first_frame(const std::vector<float>& pixels,
                                          const Wan22TI2VRuntimeShape& shape);
    std::vector<float> run_denoiser(const std::vector<float>& latents,
                                    const std::vector<float>& context,
                                    const std::vector<float>& time,
                                    const Wan22TI2VRuntimeShape& shape, ITrtModule& denoiser);
    void run_denoising(std::vector<float>& latents, const std::vector<float>& prompt_context,
                       const std::vector<float>& negative_context,
                       const std::vector<float>* first_frame_latent,
                       const Wan22TI2VRequest& request, const Wan22TI2VRuntimeShape& shape,
                       double& denoiser_ms, double& scheduler_ms);
    ImageResult decode_video(const std::vector<float>& latents, const Wan22TI2VRuntimeShape& shape);
    std::unique_ptr<ITrtModule>
    load_module(const std::string& section_name,
                const std::vector<ModuleExternalBinding>& external_bindings = {}) const;
    void synchronize_stream(const char* transition) const;
    void synchronize_stream_noexcept() const noexcept;

    Wan22ModuleLoader module_loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    Wan22TI2VOptions options_;
    wan2_2_ti2v::RuntimeConfig runtime_config_;
    std::string model_id_;
    std::string primary_task_;
    bool image_encoder_{false};
    std::vector<internal::ConfigField> config_fields_; // Borrows options_.negative_prompt.
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
};

} // namespace trtmc
