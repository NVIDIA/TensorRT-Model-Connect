/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// CanaryPipeline: encoder-decoder speech-to-text pipeline.
// Uses TrtModule(encoder) + TrtModule(decoder) + CanaryInferenceState.

#include "runtime/models/canary/canary_config.h"
#include "runtime/models/canary/inference_state.h"
#include "runtime/models/canary/kv_cache.h"
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

class CanaryPipeline final : public IPipeline {
  public:
    CanaryPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
                   std::unique_ptr<CanaryInferenceState> state, CanaryConfig canary_config,
                   int32_t hidden_size, int32_t num_decoder_layers, MelFilterbank mel_fb,
                   int32_t mel_n_fft, int32_t mel_hop_length, int32_t mel_chunk_length,
                   int32_t mel_sampling_rate, cudaStream_t stream,
                   std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~CanaryPipeline() override;

    TextResult transcribe(const float* audio_data, int32_t num_samples, int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "CanaryPipeline"; }

  private:
    void run_encoder(const float* mel_data, int32_t mel_bins, int32_t mel_length,
                     int32_t valid_mel_frames);
    void setup_cross_attention(int32_t actual_enc_seq_len);
    std::vector<int32_t> run_decoder(const std::vector<int32_t>& initial_tokens,
                                     int32_t max_new_tokens);
    void run_decoder_step(int32_t token_id, std::vector<float>& logits);

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<CanaryInferenceState> state_;
    CanaryConfig canary_config_;
    int32_t hidden_size_;
    int32_t num_decoder_layers_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    int32_t mel_n_fft_;
    int32_t mel_hop_length_;
    int32_t mel_chunk_length_;
    int32_t mel_sampling_rate_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    std::vector<std::vector<uint8_t>> cross_k_host_;
    std::vector<void*> cross_k_ptrs_;
    std::vector<void*> cross_v_ptrs_;
    std::size_t cross_kv_bytes_{0};
};

} // namespace trtmc
