/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/nemotron3_diarization/runtime/frontend.h"
#include "families/nemotron3_diarization/runtime/speaker_cache.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::nemotron3_diarization {

struct DiarizationConfig {
    FrontEndConfig front_end;
    SpeakerCachePolicy cache; // with the offline FIFO sizes
    std::int32_t chunk_length{340};
    std::int32_t chunk_left_context{0};
    std::int32_t chunk_right_context{40};
    std::int32_t max_step_rows{684};
    std::vector<std::string> labels;
    std::vector<float> silence_embedding;
};

// Speaker activity of one 10 ms frame and speaker: the logistic of its logit
// above the threshold, both in float32 as the reference compares them.
bool speaker_active(float logit, float threshold);

// Labelled intervals from row-major [frames, speakers] logits, frame f covering
// [f * hop, (f + 1) * hop) samples: one interval per run of active frames,
// ordered by start, then speaker.
std::vector<internal::TimedLabel> active_intervals(const std::vector<float>& logits,
                                                   std::int64_t frames, std::int64_t speakers,
                                                   float threshold, std::int32_t hop_length,
                                                   std::int32_t sample_rate);

// Whole-recording diarization, the reference's offline mode: the recording is
// cut into chunks of encoder frames, each run with its look-ahead behind the
// speaker cache and FIFO queue that the earlier chunks left.
class DiarizationPipeline final : public internal::IModel, public internal::IAudioToTimedLabels {
  public:
    DiarizationPipeline(std::unique_ptr<ITrtModule> embedder, std::unique_ptr<ITrtModule> encoder,
                        DiarizationConfig config);

    const char* task() const noexcept override {
        return internal::IAudioToTimedLabels::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::TimedLabelsResult run(const internal::AudioToTimedLabelsRequest& request,
                                    internal::ConfigView config) override;

  private:
    std::vector<float> embed(const LogMel& features, std::int64_t& rows);
    std::vector<float> diarize(const std::vector<float>& embeddings, std::int64_t rows,
                               const std::vector<std::uint8_t>& valid_rows, std::int64_t frames);

    std::unique_ptr<ITrtModule> embedder_;
    std::unique_ptr<ITrtModule> encoder_;
    DiarizationConfig config_;
    LogMelFrontEnd front_end_;
};

} // namespace trtmc::nemotron3_diarization
