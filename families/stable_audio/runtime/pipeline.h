/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Stable Audio Open text-to-audio (diffusers StableAudioPipeline), Task text_to_audio.
//
// Per request: the conditioning engine runs once per prompt (and once for a
// negative prompt), the DiT runs once per denoising step on the stacked
// unconditional and conditional branches, and the Oobleck decoder turns the final
// latents into stereo audio, which is cropped to [audio_start_in_s, audio_end_in_s).

#include "families/stable_audio/runtime/request.h"
#include "families/stable_audio/runtime/scheduler.h"
#include "families/stable_audio/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::stable_audio {

struct StableAudioEngines {
    std::unique_ptr<ITrtModule> conditioner;
    std::unique_ptr<ITrtModule> transformer;
    std::unique_ptr<ITrtModule> decoder;
};

class StableAudioPipeline final : public trtmc::internal::IModel,
                                  public trtmc::internal::ITextToAudio {
  public:
    StableAudioPipeline(StableAudioEngines engines, std::unique_ptr<ITokenizer> tokenizer,
                        StableAudioConfig config);

    const char* task() const noexcept override;
    std::vector<trtmc::internal::TaskInstance> task_bindings() override;
    trtmc::internal::AudioResult run(const trtmc::internal::TextToAudioRequest& request,
                                     trtmc::internal::ConfigView config) override;

  private:
    struct Conditioning {
        std::vector<float> encoder_states; // [encoder_tokens, conditioning_dim]
        std::vector<float> global_states;  // [global_dim]
    };
    Conditioning condition(const std::string& text, double start, double end);
    std::vector<float> denoise(const GenerationPlan& plan, const Conditioning& positive,
                               const std::optional<Conditioning>& negative);
    std::vector<float> decode(const std::vector<float>& latents);

    StableAudioEngines engines_;
    std::unique_ptr<ITokenizer> tokenizer_;
    StableAudioConfig config_;
    CosineDpmSolver scheduler_;
};

} // namespace trtmc::stable_audio
