/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// RnntPipeline: native C++ greedy RNN-T speech-to-text pipeline.
// Uses separate TRT modules for the acoustic encoder, prediction network,
// and joint network. Build-time Python may parse NeMo checkpoints, but
// runtime inference does not call Python.

#include "runtime/models/nemotron_speech_streaming/rnnt_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct MelFilterbank;

class RnntPipeline final : public IPipeline {
  public:
    RnntPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> predictor,
                 std::unique_ptr<TrtModule> joint,
                 std::map<int32_t, std::string> streaming_encoder_sections, IBackend* backend,
                 ModuleCreateOptions module_options,
                 std::map<int32_t, std::string> streaming_first_encoder_sections,
                 std::string bundle_path, RnntConfig config, MelFilterbank mel_fb,
                 cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer = nullptr,
                 std::string model_id_str = "");

    ~RnntPipeline() override;

    TextResult transcribe(const float* audio_data, int32_t num_samples, int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;

    std::unique_ptr<ITranscriptionStream>
    create_transcription_stream(const TranscriptionStreamConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "RnntPipeline"; }

  private:
    friend class RnntTranscriptionStream;

    std::vector<float> extract_padded_mel(const float* audio_data, int32_t num_samples,
                                          int32_t input_sample_rate, int32_t& actual_frames) const;
    std::vector<float> run_encoder(const std::vector<float>& mel, int32_t actual_frames);
    TrtModule& streaming_encoder_for(int32_t right_context, bool first_step);
    std::vector<float> run_streaming_encoder(int32_t right_context, const std::vector<float>& mel,
                                             const std::vector<float>& cache_last_channel,
                                             const std::vector<float>& cache_last_time,
                                             int32_t cache_last_channel_len,
                                             int32_t valid_query_frames, bool first_step,
                                             std::vector<float>& next_channel,
                                             std::vector<float>& next_time);
    std::vector<float> run_predictor(int32_t token_id, std::vector<float>& state_h,
                                     std::vector<float>& state_c);
    std::vector<float> run_joint(const float* encoder_frame, const float* pred_output);
    void decode_encoder_frames(const std::vector<float>& encoder_output, int32_t frame_count,
                               int32_t token_limit, std::vector<float>& pred_output,
                               std::vector<float>& state_h, std::vector<float>& state_c,
                               std::vector<int32_t>& emitted);

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> predictor_;
    std::unique_ptr<TrtModule> joint_;
    std::map<int32_t, std::string> streaming_encoder_sections_;
    std::map<int32_t, std::string> streaming_first_encoder_sections_;
    std::map<int32_t, std::unique_ptr<TrtModule>> streaming_encoders_;
    std::map<int32_t, std::unique_ptr<TrtModule>> streaming_first_encoders_;
    IBackend* backend_{nullptr};
    ModuleCreateOptions module_options_;
    std::string bundle_path_;
    RnntConfig config_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtmc
