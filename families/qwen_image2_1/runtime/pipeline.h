/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen_image2_1/runtime/runtime_config.h"
#include "families/qwen_image2_1/runtime/tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc::qwen_image2_1 {

// One generation request after Config resolution.
struct GenerationSettings {
    std::uint64_t seed{0};
    std::int32_t num_steps{0};
    double cfg_scale{1.0};
    // Set only when classifier-free guidance is active.
    bool use_negative_prompt{false};
    std::string negative_prompt;
};

// Text-to-image with Qwen-Image 2.1: Qwen3-VL text encoder, block-causal single-stream
// transformer, flow-match Euler sampling, RGBA VAE decoder. All three engines are static.
class Pipeline final : public internal::IModel, public internal::ITextToImage {
  public:
    struct Engines {
        std::unique_ptr<ITrtModule> text_encoder;
        std::unique_ptr<ITrtModule> denoiser;
        std::unique_ptr<ITrtModule> vae;
    };

    Pipeline(Engines engines, std::unique_ptr<ITokenizer> tokenizer, RuntimeConfig config);

    const char* task() const noexcept override { return internal::ITextToImage::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    ImageResult run(const internal::TextToImageRequest& request,
                    internal::ConfigView config) override;

  private:
    struct EncodedPrompt {
        // [text_tokens, text_hidden_size]; rows at or beyond `length` are zero.
        std::vector<float> hidden;
        std::int32_t length{0};
    };

    GenerationSettings resolve(internal::ConfigView config) const;
    EncodedPrompt encode_prompt(const std::string& prompt);
    std::vector<float> predict_velocity(const std::vector<float>& latents, float timestep,
                                        const EncodedPrompt& prompt);
    ImageResult decode(const std::vector<float>& latents);

    Engines engines_;
    std::unique_ptr<ITokenizer> tokenizer_;
    RuntimeConfig config_;
    std::vector<internal::ConfigField> fields_;
    std::mutex mutex_;
};

} // namespace trtmc::qwen_image2_1
