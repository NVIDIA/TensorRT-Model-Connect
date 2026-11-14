/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/chroma/runtime/sampling.h"
#include "families/chroma/runtime/t5_tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::chroma {

// runtime.json written by families/chroma/model.py.
struct RuntimeConfig {
    bool bfloat16{true};
    std::int32_t height{0};
    std::int32_t width{0};
    std::int32_t text_length{0};
    std::int32_t latent_channels{0};
    std::int32_t vae_scale_factor{0};
    float vae_scaling{1.0F};
    float vae_shift{0.0F};
    float scheduler_shift{1.0F};
    std::int32_t num_train_timesteps{1000};
    std::int64_t default_steps{1};
    double default_guidance{1.0};

    LatentShape latent_shape() const {
        return {latent_channels, height / vae_scale_factor, width / vae_scale_factor};
    }
};

RuntimeConfig parse_runtime_config(std::string_view json_text);

using ModuleLoader = std::function<std::unique_ptr<ITrtModule>(const std::string& section)>;

// Text-to-image: T5 prompt encoding, a flow-matching Euler loop over the
// MMDiT with classifier-free guidance, and the VAE decoder.
class ChromaPipeline final : public internal::IModel, public internal::ITextToImage {
  public:
    ChromaPipeline(const ModuleLoader& loader, T5Tokenizer tokenizer, RuntimeConfig config);

    const char* task() const noexcept override { return internal::ITextToImage::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::ImageResult run(const internal::TextToImageRequest& request,
                              internal::ConfigView config) override;

  private:
    struct Settings {
        std::string negative_prompt;
        std::int32_t steps{1};
        float guidance{1.0F};
        std::uint32_t seed{0};
    };
    struct EncodedText {
        std::vector<float> embeddings;
        std::vector<float> mask;
    };

    Settings resolve(internal::ConfigView config, bool replay) const;
    EncodedText encode(std::string_view text);
    std::vector<float> denoise(const std::vector<float>& latents, const EncodedText& text,
                               const std::array<float, 16>& features);
    std::vector<float> decode(std::vector<float> packed);

    RuntimeConfig config_;
    T5Tokenizer tokenizer_;
    std::unique_ptr<ITrtModule> text_encoder_;
    std::unique_ptr<ITrtModule> transformer_;
    std::unique_ptr<ITrtModule> vae_decoder_;
    std::vector<internal::ConfigField> fields_;
    std::mutex mutex_;
};

} // namespace trtmc::chroma
