/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// DeepseekOcrPipeline: vision-language generation (Qwen2.5-VL, Qwen3-VL, InternVL3, Phi4).
// Composes: vision_encoder ITrtModule + text_decoder ITrtModule + DeepseekOcrKvCache.

#include "families/deepseek_ocr/runtime/image_preprocessor.h"
#include "families/deepseek_ocr/runtime/inference_state.h"
#include "families/deepseek_ocr/runtime/kv_cache.h"
#include "families/deepseek_ocr/runtime/runtime_config.h"
#include "families/deepseek_ocr/runtime/sampler.h"
#include "families/deepseek_ocr/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

class DeepseekOcrPipeline final : public ITextGeneration, public IVisionLanguageGeneration {
  public:
    const char* task() const noexcept override { return IVisionLanguageGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return config_.prefill_max_length; }

    DeepseekOcrPipeline(std::unique_ptr<ITrtModule> text_decoder,
                        std::unique_ptr<ITrtModule> vision_encoder,
                        std::unique_ptr<DeepseekOcrInferenceState> state, DeepseekOcrConfig config,
                        DeepseekOcrPreprocessConfig vl_preprocess, cudaStream_t stream,
                        std::shared_ptr<ITokenizer> tokenizer = nullptr,
                        std::string model_id_str = "",
                        std::unique_ptr<ITrtModule> prefill = nullptr);

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;

    TextResult generate(const std::string& prompt, const float* image_pixels, int32_t image_height,
                        int32_t image_width, const TextGenerationConfig& cfg = {}) override;

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const TextGenerationConfig& cfg);

    // VL preprocessing config (public for testing).
    const DeepseekOcrPreprocessConfig& vl_preprocess_config() const { return vl_preprocess_; }
    bool has_vision_encoder() const { return vision_encoder_ != nullptr; }

  private:
    std::unique_ptr<ITrtModule> text_decoder_;
    std::unique_ptr<ITrtModule> prefill_;
    std::unique_ptr<ITrtModule> vision_encoder_;
    std::unique_ptr<DeepseekOcrInferenceState> state_;
    double last_setup_ms_{0.0};
    DeepseekOcrConfig config_;
    DeepseekOcrPreprocessConfig vl_preprocess_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const DeepseekOcrSamplingParams& params);

    // VL generation with vision features injected at image token positions.
    std::vector<int32_t> generate_vl_from_ids(
        const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
        const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
        int32_t feature_dim, int32_t max_new_tokens, const DeepseekOcrSamplingParams& params);

    std::pair<int32_t, int32_t> resolve_gen_limits(const TextGenerationConfig& cfg) const;

    void reset_generation_context(int32_t prompt_length);

    void run_vl_prefill_token(int32_t token_id, const std::vector<float>& image_features,
                              const std::vector<std::vector<float>>& deepstack_features,
                              int32_t num_features, int32_t feature_dim, int32_t& feature_index,
                              std::vector<float>& logits);

    bool run_vl_prefill_batched(const std::vector<int32_t>& input_ids,
                                const std::vector<float>& image_features,
                                const std::vector<std::vector<float>>& deepstack_features,
                                int32_t num_features, int32_t feature_dim,
                                std::vector<float>& logits);

    void run_vl_decode_loop(DeepseekOcrISampler* sampler, const DeepseekOcrSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens);

    void run_text_step(int32_t token_id, std::vector<float>& logits);

    // Run a text step with optional vision embedding override.
    void run_text_step_with_embed(int32_t token_id, const float* input_embed, float use_input_embed,
                                  const std::vector<const float*>& deepstack_embeds,
                                  float deepstack_active, std::vector<float>& logits);

    // Run vision encoder on preprocessed image inputs.
    bool run_vision_encoder(const DeepseekOcrPreprocessedImage& preprocessed,
                            std::vector<float>& image_features,
                            std::vector<std::vector<float>>* deepstack_features = nullptr);
};

} // namespace trtmc
