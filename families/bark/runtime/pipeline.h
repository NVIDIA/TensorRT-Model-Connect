/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// BarkPipeline: text-to-audio pipeline with semantic, coarse, fine, and codec stages.
// Uses ITrtModule(semantic) + ITrtModule(coarse) + ITrtModule(codec) + ITrtModule(fine) +
// KvCaches + embeddings.

#include "families/bark/runtime/bark_config.h"
#include "families/bark/runtime/inference_state.h"
#include "families/bark/runtime/kv_cache.h"
#include "families/bark/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

class BarkSampler;

class BarkPipeline final : public IAudioGeneration {
  public:
    BarkPipeline(std::unique_ptr<ITrtModule> semantic, std::unique_ptr<ITrtModule> coarse,
                 std::unique_ptr<BarkInferenceState> semantic_state,
                 std::unique_ptr<BarkInferenceState> coarse_state,
                 std::vector<float> semantic_embed, std::vector<float> coarse_embed,
                 BarkConfig config, cudaStream_t stream,
                 std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~BarkPipeline() override;

    AudioResult generate_audio(const std::string& prompt,
                               const AudioGenerationConfig& cfg = {}) override;

    void set_codec_module(std::unique_ptr<ITrtModule> codec);
    void set_fine_module(std::unique_ptr<ITrtModule> fine);
    void set_fine_embeddings(std::vector<float> embed, std::vector<float> pos_embed);
    void set_prefill_modules(std::unique_ptr<ITrtModule> semantic_prefill,
                             std::unique_ptr<ITrtModule> coarse_prefill);

  private:
    std::vector<int32_t> run_semantic(const std::vector<int32_t>& text_ids, int32_t max_tokens);
    std::vector<int32_t> run_coarse(const std::vector<int32_t>& semantic_tokens);
    std::vector<int32_t> run_fine(const std::vector<int32_t>& coarse_tokens);
    std::vector<float> run_codec(const std::vector<int32_t>& coarse_tokens);
    std::vector<float> run_codec(const std::vector<int32_t>& codes_flat, int32_t n_frames);

    void run_step_with_embed(ITrtModule& module, BarkInferenceState& state, const float* embed,
                             int32_t embed_dim, std::vector<float>& logits);
    void run_step_with_token(ITrtModule& module, BarkInferenceState& state, int32_t token_id,
                             std::vector<float>& logits);
    bool run_batched_prefill(ITrtModule* module, BarkInferenceState& state,
                             const std::vector<float>& embeddings, int32_t hidden_size,
                             std::vector<float>& logits, const char* stage);
    int32_t sample_top_k(const float* logits, int32_t vocab_size, float temperature, int32_t top_k);

    std::unique_ptr<ITrtModule> semantic_;
    std::unique_ptr<ITrtModule> coarse_;
    std::unique_ptr<ITrtModule> codec_;
    std::unique_ptr<ITrtModule> fine_;
    std::unique_ptr<ITrtModule> semantic_prefill_;
    std::unique_ptr<ITrtModule> coarse_prefill_;
    std::unique_ptr<BarkInferenceState> semantic_state_;
    std::unique_ptr<BarkInferenceState> coarse_state_;
    std::vector<float> semantic_embed_;
    std::vector<float> coarse_embed_;
    std::vector<float> fine_embed_;
    std::vector<float> fine_position_embed_;
    BarkConfig config_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<BarkSampler> sampler_;
};

} // namespace trtmc
