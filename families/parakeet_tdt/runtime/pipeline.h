/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "families/parakeet_tdt/runtime/tdt_config.h"
#include "families/parakeet_tdt/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <mutex>

namespace trtmc::parakeet_tdt {
struct MelFilterbank {
    std::vector<float> data;
    int32_t n_freq_bins{0};
    int32_t n_mel_bins{0};
};

class TdtPipeline final : public internal::IModel, public internal::ISpeechTranscription {
  public:
    TdtPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> predictor,
                std::unique_ptr<ITrtModule> joint, TdtConfig config, MelFilterbank mel,
                std::shared_ptr<ITokenizer> tokenizer);
    const char* task() const noexcept override { return "speech_transcription"; }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::SpeechTranscriptionRequest&, internal::ConfigView) override;

  private:
    std::vector<float> extract_padded_mel(const float*, int32_t, int32_t, int32_t&) const;
    std::vector<float> run_encoder(const std::vector<float>&, int32_t);
    std::vector<float> run_predictor(int32_t, std::vector<float>&, std::vector<float>&);
    std::vector<float> run_joint(const float*, const float*);
    void decode_encoder_frames(const std::vector<float>&, int32_t, int32_t, std::vector<float>&,
                               std::vector<float>&, std::vector<float>&, std::vector<int32_t>&);
    std::unique_ptr<ITrtModule> encoder_, predictor_, joint_;
    TdtConfig config_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::mutex mutex_;
};
} // namespace trtmc::parakeet_tdt
