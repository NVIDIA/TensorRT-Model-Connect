/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SpeechPipeline: speech-to-speech pipeline with temporal + depth engines.
// Uses ITrtModule(mimi_encoder) + ITrtModule(temporal) + PersonaplexKvCache +
// ITrtModule(depth)[] + ITrtModule(mimi_decoder).

#include "families/personaplex/runtime/inference_state.h"
#include "families/personaplex/runtime/kv_cache.h"
#include "families/personaplex/runtime/speech_config.h"
#include "families/personaplex/runtime/speech_delay_cache.h"
#include "families/personaplex/runtime/speech_generation_policy.h"
#include "families/personaplex/runtime/speech_performance.h"
#include "families/personaplex/runtime/speech_runtime_plan.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct SpeechDeviceWorkspace;

struct SpeechTemporalDeviceOutput {
    const void* hidden{nullptr};
    DType hidden_dtype{DType::kFloat32};
};

class SpeechPipeline final : public ISpeechToSpeech {
  public:
    SpeechPipeline(std::unique_ptr<ITrtModule> mimi_encoder, std::unique_ptr<ITrtModule> temporal,
                   std::unique_ptr<PersonaplexInferenceState> temporal_state,
                   std::vector<std::unique_ptr<ITrtModule>> depth_engines,
                   std::unique_ptr<PersonaplexInferenceState> depth_state,
                   std::unique_ptr<ITrtModule> mimi_decoder, SpeechConfig config,
                   cudaStream_t stream, std::string model_id_str = "");

    ~SpeechPipeline() override;

    AudioResult speak(const float* audio_in, int32_t num_samples,
                      const SpeechToSpeechConfig& cfg = {}, int32_t input_sample_rate = 0) override;

  private:
    std::vector<int32_t> run_mimi_encode(const float* samples, int32_t num_samples);

    SpeechTemporalDeviceOutput run_temporal_embed_step(const float* embed_ptr, int32_t embed_size);

    void run_depth(const SpeechTemporalDeviceOutput& temporal_output, int32_t text_token,
                   bool text_token_is_forced, const int32_t* forced_audio_tokens = nullptr,
                   const uint8_t* forced_audio_provided = nullptr);
    ITrtModule& depth_engine_for_codebook(int32_t codebook);
    void prepare_depth_input(const SpeechTemporalDeviceOutput& temporal_output, int32_t codebook,
                             int32_t text_token, bool text_token_is_forced,
                             const int32_t* forced_audio_tokens,
                             const uint8_t* forced_audio_provided, int32_t* selected_tokens);
    void enqueue_depth_step(ITrtModule& engine, int32_t codebook, int32_t* selected_tokens);
    void download_selected_frame_tokens(std::vector<int32_t>& selected_tokens);

    std::vector<float> run_mimi_decode(const std::vector<int32_t>& codec_tokens,
                                       int32_t num_frames);

    void run_text_prompt();

    bool speak_validate_dual_stream() const;
    void speak_run_generation_loop(const SpeechGenerationSettings& settings,
                                   const SpeechOutputPlan& plan, DelayCacheState& delay_state,
                                   const std::vector<int32_t>& codec_tokens,
                                   std::vector<int32_t>& output_codes, int32_t& frames_collected,
                                   SpeechPerformanceTimings& timings);
    void speak_postprocess_waveform(std::vector<float>& waveform, int32_t generated_frames) const;

    std::unique_ptr<ITrtModule> temporal_;
    std::unique_ptr<ITrtModule> mimi_encoder_;
    std::unique_ptr<PersonaplexInferenceState> temporal_state_;
    std::vector<std::unique_ptr<ITrtModule>> depth_engines_;
    std::unique_ptr<PersonaplexInferenceState> depth_state_;
    std::unique_ptr<ITrtModule> mimi_decoder_;
    std::unique_ptr<SpeechDeviceWorkspace> device_workspace_;

    cudaStream_t stream_;
    SpeechConfig config_;
    std::string model_id_;

    int32_t last_encode_frames_{0};
    int32_t last_encode_codebooks_{0};
};

} // namespace trtmc
