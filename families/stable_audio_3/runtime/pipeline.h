/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Stable Audio 3 text-to-audio: T5Gemma conditioning, pingpong sampling of the
// rf_denoiser DiT, SAME decoding. Implements IModel + ITextToAudio.

#include "families/stable_audio_3/runtime/sampling.h"
#include "families/stable_audio_3/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::stable_audio_3 {

struct RuntimeConfig {
    std::int32_t sample_rate{44100};
    std::int32_t audio_channels{2};
    std::int32_t latent_dim{256};
    std::int32_t prompt_tokens{256};
    std::int32_t pad_token_id{0};
    std::int32_t context_tokens{257};
    std::int32_t cond_dim{768};
    // "fp16" rounds the schedule to half precision, as the reference does in half mode.
    std::string time_precision{"fp16"};
    RenderLimits limits{};
    LogSnrShift shift{};
};

class Pipeline final : public trtmc::internal::IModel, public trtmc::internal::ITextToAudio {
  public:
    Pipeline(std::unique_ptr<ITrtModule> conditioner, std::unique_ptr<ITrtModule> dit,
             std::unique_ptr<ITrtModule> decoder, std::unique_ptr<ITokenizer> tokenizer,
             RuntimeConfig config);

    const char* task() const noexcept override;
    std::vector<trtmc::internal::TaskInstance> task_bindings() override;
    trtmc::internal::AudioResult run(const trtmc::internal::TextToAudioRequest& request,
                                     trtmc::internal::ConfigView config) override;

  private:
    struct Conditioning {
        std::vector<float> context;
        std::vector<float> global;
    };
    Conditioning condition(const std::string& prompt, double seconds);
    std::vector<float> velocity(std::vector<float>& x, float t, const Conditioning& conditioning,
                                const RenderPlan& plan);
    std::vector<float> sample(const Conditioning& conditioning, const RenderPlan& plan,
                              std::int32_t steps, std::uint64_t seed);
    std::vector<float> decode(std::vector<float>& latents, const RenderPlan& plan);

    std::unique_ptr<ITrtModule> conditioner_;
    std::unique_ptr<ITrtModule> dit_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<ITokenizer> tokenizer_;
    RuntimeConfig config_;
};

} // namespace trtmc::stable_audio_3
