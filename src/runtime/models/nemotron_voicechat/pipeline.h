/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Model-owned runtime for NVIDIA-NemotronLabs-VoiceChat-11B.
//
// Python is used only while building the bundle. Every object below is a
// native C++/TensorRT runtime object: cache-aware FastConformer perception,
// RNN-T transcription, the hybrid Nemotron-H thinker, EAR-TTS, and the RVQ
// codec/ISTFT reconstruction path.

#include "runtime/models/nemotron_voicechat/voicechat_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/speech_session.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

namespace nemotron_voicechat {

struct StreamingMelStep {
    int32_t history_frames{0};
    int32_t requested_new_frames{0};
    int32_t valid_new_frames{0};
    int32_t engine_frames{0};
};

// Right-context-zero FastConformer schedule used by the runtime and its
// boundary tests. The final step may contain fewer than eight new mel frames;
// the engine input remains fixed and the missing tail columns stay zero.
StreamingMelStep make_streaming_mel_step(bool first_step, int32_t next_mel_frame,
                                         int32_t available_mel_frames, bool final);

bool should_barge_in(const SpeechSessionConfig& config, bool agent_turn_active, double rms);
int32_t streaming_frontend_capacity_seconds(const Config& config);

} // namespace nemotron_voicechat

struct VoiceChatTtsPrompt {
    std::vector<float> aria_embeddings; // [warmup_steps, tts_hidden_size]
    std::vector<int32_t> subword_ids;
    std::vector<float> subword_mask;
    std::vector<float> audio_prompt_mode;
    std::vector<float> bos_flags;
    std::vector<int32_t> position_ids;
    std::vector<int32_t> first_codes;
    std::vector<int32_t> silence_codes;
    std::vector<int32_t> control_codes;
    int32_t warmup_steps{0};
    int32_t first_generation_position{0};
};

struct VoiceChatAssets {
    std::vector<float> mel_filterbank; // [mel_freq_bins, mel_bins]
    int32_t mel_freq_bins{0};
    int32_t mel_bins{0};
    std::vector<float> mel_window; // Exact checkpoint window before centering in the FFT.
    std::vector<std::string> rnnt_vocabulary;
    VoiceChatTtsPrompt tts_prompt;
};

class NemotronVoiceChatRuntime;

class NemotronVoiceChatPipeline final : public IPipeline, public ISpeechSessionProvider {
  public:
    NemotronVoiceChatPipeline(std::unique_ptr<TrtModule> thinker,
                              std::unique_ptr<TrtModule> perception_stream_first,
                              std::unique_ptr<TrtModule> perception_stream,
                              std::unique_ptr<TrtModule> rnnt_predictor,
                              std::unique_ptr<TrtModule> rnnt_joint, std::unique_ptr<TrtModule> tts,
                              std::unique_ptr<TrtModule> codec, nemotron_voicechat::Config config,
                              VoiceChatAssets assets, std::shared_ptr<ITokenizer> tokenizer,
                              std::string model_id);
    ~NemotronVoiceChatPipeline() override;

    std::unique_ptr<ISpeechSession>
    create_speech_session(const SpeechSessionConfig& cfg = {}) override;

    AudioResult speak(const float* audio_in, int32_t num_samples, const GenerateConfig& cfg = {},
                      int32_t input_sample_rate = 0) override;

    TextResult transcribe(const float* audio_samples, int32_t num_samples, int32_t max_tokens = 224,
                          int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "NemotronVoiceChatPipeline"; }

  private:
    std::shared_ptr<NemotronVoiceChatRuntime> runtime_;
    std::string model_id_;
};

} // namespace trtmc
