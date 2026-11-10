/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_denoiser.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"

namespace trtmc::minimax_h3 {

enum class ReferenceMemoryPreset { Minimal, Original };

struct ReferencePipelineConfig {
    ReferenceDenoiserConfig denoiser;
    ReferenceLatentNormalization normalization;
    int32_t max_text_rows{0};
    int32_t max_condition_video_rows{16384};
    ReferenceMemoryPreset memory{ReferenceMemoryPreset::Minimal};
    int32_t output_frames{124}; // Explicit bundle profile: one window or two (226 frames).
    int32_t max_condition_audio_rows{0};

    void validate_forward_count(int32_t forwards) const;
};

// Native image Ref2VA composition with an explicit original or student recipe.
// This class uses the existing
// semantic contract. The family factory accepts its explicit reference bundle,
// but support.py does not advertise it before complete trained gates pass.
// The stream and loader's borrowed backend must outlive this model.
class ImageReferencePipeline final : public internal::IModel,
                                     public internal::IReferencesTextToAudioVideo {
  public:
    ImageReferencePipeline(ReferenceModuleLoader loader, std::unique_ptr<ITokenizer> tokenizer,
                           ReferencePipelineConfig config, cudaStream_t stream = nullptr);
    ~ImageReferencePipeline() override;

    const char* task() const noexcept override { return "references_text_to_audio_video"; }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::AudioVideoResult run(const internal::ReferencesTextToAudioVideoRequest& request,
                                   internal::ConfigView config) override;

  private:
    struct State;
    ReferenceModuleLoader loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    ReferencePipelineConfig config_;
    cudaStream_t stream_;
    std::unique_ptr<State> state_;
    std::mutex mutex_;
};

} // namespace trtmc::minimax_h3
