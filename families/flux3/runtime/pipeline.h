/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/flux3/runtime/engine_contract.h"
#include "families/flux3/runtime/runtime_config.h"
#include "families/flux3/runtime/tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/span.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace trtmc::flux3 {

// FLUX 3 Image Klein text-to-image: Qwen3-VL text encoder, flow-map denoiser
// and VAE decoder engines, with the sampling math on the host (sampling.h).
// The factory has already validated the engines (engine_contract.h) and the
// tokenizer against `config`.
class Flux3Pipeline final : public internal::IModel, public internal::ITextToImage {
  public:
    Flux3Pipeline(RuntimeConfig config, std::unique_ptr<ITokenizer> tokenizer, Engines engines);

    const char* task() const noexcept override { return internal::ITextToImage::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::ImageResult run(const internal::TextToImageRequest& request,
                              internal::ConfigView config) override;

  private:
    struct Options {
        std::uint32_t seed{0};
        std::int32_t num_steps{0};
    };

    Options resolve_options(internal::ConfigView config) const;
    std::vector<float> initial_latents(Span<const float> supplied, std::uint32_t seed) const;
    void encode_prompt(std::string_view prompt);
    Span<const float> predict_velocity(std::vector<float>& latents, float timestep, float target);
    void denoise(std::vector<float>& latents, std::int32_t num_steps);
    internal::ImageResult decode(std::vector<float>& latents);

    RuntimeConfig config_;
    std::unique_ptr<ITokenizer> tokenizer_;
    Engines engines_;
    std::vector<internal::ConfigField> config_fields_;
};

} // namespace trtmc::flux3
