/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// GlmAsrPipeline: speech-to-text over an audio encoder plus a Llama decoder.
//
// The encoder engine turns one padded mel chunk into a fixed number of
// projected embeddings. Only the rows the real audio covers are spliced into
// the decoder prompt, replacing the audio placeholder tokens through the
// decoder's embed-input contract. There is no cross-attention.
//
// The prompt is replayed one token at a time so each placeholder position can
// carry its own projected frame. A batched prefill profile would be faster,
// but this family has no bundle configuration that emits one, so that path is
// deliberately absent rather than present and unexercised.

#include "runtime/models/glmasr/glmasr_config.h"
#include "runtime/models/glmasr/inference_state.h"
#include "runtime/models/glmasr/kv_cache.h"
#include "runtime/models/glmasr/plugin_helpers.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct GlmAsrRunStats {
    int32_t prompt_tokens{0};
    int32_t audio_embeddings{0};
    int32_t decode_launches{0};
    int32_t encoder_launches{0};
};

class GlmAsrPipeline final : public IPipeline {
  public:
    GlmAsrPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
                   std::unique_ptr<GlmAsrInferenceState> state, GlmAsrConfig config,
                   MelFilterbank mel_filterbank, cudaStream_t stream,
                   std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~GlmAsrPipeline() override;

    TextResult transcribe(const float* audio_samples, int32_t num_samples, int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "GlmAsrPipeline"; }

    const GlmAsrRunStats& run_stats() const { return stats_; }

  private:
    // Run the encoder over one padded mel chunk and return [max_audio_embeddings, size].
    std::vector<float> run_audio_encoder(const std::vector<float>& mel);

    // Prompt token ids the decoder consumes, with placeholders for the audio.
    std::vector<int32_t> tokenize_instruction() const;

    // audio_embed is one projected frame for an audio placeholder position,
    // or nullptr for an ordinary text token.
    int32_t run_decode_step(int32_t token_id, const float* audio_embed = nullptr);
    std::vector<int32_t> run_decoder(const std::vector<int32_t>& input_ids,
                                     const std::vector<float>& audio_embeds, int32_t audio_offset,
                                     int32_t num_audio_embeddings, int32_t max_new_tokens);

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<GlmAsrInferenceState> state_;
    GlmAsrConfig config_;
    MelFilterbank mel_filterbank_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    GlmAsrRunStats stats_;
};

} // namespace trtmc
