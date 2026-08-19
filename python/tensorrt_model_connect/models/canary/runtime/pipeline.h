/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// CanaryPipeline: encoder-decoder speech-to-text pipeline.
// Uses TrtModule(encoder) + TrtModule(decoder) + CanaryInferenceState.

#include "canary_config.h"
#include "inference_state.h"
#include "kv_cache.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct MelFilterbank;
struct CanaryBatchSegment;
struct CanaryBatchWorkGroup;

class CanaryPipeline final : public IPipeline {
  public:
    CanaryPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
                   std::unique_ptr<CanaryInferenceState> state, CanaryConfig canary_config,
                   int32_t hidden_size, int32_t num_decoder_layers, MelFilterbank mel_fb,
                   int32_t mel_n_fft, int32_t mel_win_length, int32_t mel_hop_length,
                   int32_t mel_chunk_length, int32_t mel_sampling_rate, float mel_preemph,
                   bool mel_normalize_per_feature, cudaStream_t stream,
                   std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~CanaryPipeline() override;

    TextResult transcribe(const float* audio_data, int32_t num_samples, int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;
    TextResult transcribe(const float* audio_data, int32_t num_samples,
                          const TranscriptionConfig& cfg) override;
    std::vector<TextResult>
    transcribe_batch(const std::vector<TranscriptionRequest>& requests) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "CanaryPipeline"; }

  private:
    TextResult transcribe_segment(const float* audio_data, int32_t num_samples,
                                  int32_t input_sample_rate,
                                  const std::vector<int32_t>& initial_tokens,
                                  int32_t max_output_tokens, int32_t beam_size,
                                  float length_penalty, int32_t beam_fallback_max_size);
    void transcribe_batch_group(const CanaryBatchWorkGroup& group,
                                const std::vector<CanaryBatchSegment>& work,
                                const std::vector<TranscriptionRequest>& requests,
                                std::vector<TextResult>& segment_results);
    void retry_batch_fallback_segments(const std::vector<CanaryBatchSegment>& work,
                                       const std::vector<TranscriptionRequest>& requests,
                                       std::vector<TextResult>& segment_results);
    void run_encoder(const float* mel_data, int32_t mel_bins, int32_t mel_length,
                     int32_t valid_mel_frames);
    void run_encoder_batch(const std::vector<std::vector<float>>& mel_data, int32_t mel_bins,
                           int32_t mel_length, const std::vector<int32_t>& valid_mel_frames);
    void setup_cross_attention(int32_t actual_enc_seq_len);
    void setup_cross_attention(const std::vector<int32_t>& actual_enc_seq_lens,
                               const std::vector<int32_t>& lane_to_sample);
    std::vector<int32_t> run_decoder(const std::vector<int32_t>& initial_tokens,
                                     int32_t max_new_tokens, int32_t beam_size,
                                     float length_penalty);
    std::vector<int32_t> run_decoder_with_fallback(const std::vector<int32_t>& initial_tokens,
                                                   int32_t max_new_tokens, int32_t beam_size,
                                                   float length_penalty,
                                                   int32_t beam_fallback_max_size);
    std::vector<std::vector<int32_t>>
    run_decoder_batch(const std::vector<std::vector<int32_t>>& initial_tokens,
                      const std::vector<int32_t>& max_new_tokens, int32_t beam_size,
                      float length_penalty, const std::vector<int32_t>& actual_enc_seq_lens);
    std::vector<std::vector<int32_t>>
    run_greedy_decoder_batch(const std::vector<std::vector<int32_t>>& initial_tokens,
                             const std::vector<int32_t>& max_new_tokens);
    std::vector<std::vector<int32_t>>
    run_beam_decoder_batch(const std::vector<std::vector<int32_t>>& initial_tokens,
                           const std::vector<int32_t>& max_new_tokens, int32_t beam_size,
                           float length_penalty, const std::vector<int32_t>& actual_enc_seq_lens);
    std::vector<int32_t> run_beam_decoder(const std::vector<int32_t>& initial_tokens,
                                          int32_t max_new_tokens, int32_t beam_size,
                                          float length_penalty);
    void run_decoder_step(int32_t token_id, std::vector<float>& logits);
    void run_decoder_step_batch(const std::vector<int32_t>& token_ids, std::vector<float>& logits);
    void ensure_beam_state_capacity(int32_t beam_size);
    CanaryKvCache& batch_cache();
    const CanaryKvCache& batch_cache() const;
    void ensure_batch_beam_state();

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<CanaryInferenceState> state_;
    std::vector<std::unique_ptr<CanaryInferenceState>> beam_states_a_;
    std::vector<std::unique_ptr<CanaryInferenceState>> beam_states_b_;
    std::unique_ptr<CanaryInferenceState> batch_beam_state_;
    CanaryConfig canary_config_;
    int32_t hidden_size_;
    int32_t num_decoder_layers_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    int32_t mel_n_fft_;
    int32_t mel_win_length_;
    int32_t mel_hop_length_;
    int32_t mel_chunk_length_;
    int32_t mel_sampling_rate_;
    float mel_preemph_;
    bool mel_normalize_per_feature_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    void* cross_kv_ptr_{nullptr};
    std::size_t cross_kv_sample_bytes_{0};
    int32_t encoder_batch_capacity_{1};
    int32_t decoder_lane_capacity_{1};
    std::vector<float> cross_attention_mask_;
};

} // namespace trtmc
