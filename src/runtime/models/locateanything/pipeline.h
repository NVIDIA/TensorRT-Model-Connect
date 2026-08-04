/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// LocateAnythingPipeline: vision-language generation (Qwen2.5-VL, Qwen3-VL, InternVL3, Phi4).
// Composes: vision_encoder TrtModule + text_decoder TrtModule + LocateanythingKvCache.

#include "runtime/models/locateanything/image_preprocessor.h"
#include "runtime/models/locateanything/inference_state.h"
#include "runtime/models/locateanything/kv_cache.h"
#include "runtime/models/locateanything/sampler.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

// Decode generated LocateAnything tokens while preserving the model's semantic
// localization tokens. The generic BPE decoder intentionally removes tokens
// marked as special, but LocateAnything uses <ref>, <box>, and <0>..<1000> as
// part of its user-visible answer.
std::string locateanything_decode_generated_text(const ITokenizer& tokenizer,
                                                 const std::vector<int32_t>& token_ids);

struct LocateAnythingConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    int32_t image_token_id{-1};
    int32_t vision_output_dim{0};
    bool has_position_input{true};
    int32_t num_layers{0};
    int32_t prefill_max_length{0};
    std::string present_k_pattern{"present_k_{i}"};
    std::string present_v_pattern{"present_v_{i}"};
};

class LocateAnythingPipeline final : public IPipeline {
  public:
    LocateAnythingPipeline(std::unique_ptr<TrtModule> text_decoder,
                           std::unique_ptr<TrtModule> vision_encoder,
                           std::unique_ptr<LocateanythingInferenceState> state,
                           LocateAnythingConfig config,
                           LocateAnythingPreprocessConfig vl_preprocess, cudaStream_t stream,
                           std::shared_ptr<ITokenizer> tokenizer = nullptr,
                           std::string model_id_str = "",
                           std::unique_ptr<LocateAnythingISampler> sampler = nullptr,
                           std::unique_ptr<TrtModule> prefill = nullptr);

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    TextResult generate(const std::string& prompt, const float* image_pixels, int32_t image_height,
                        int32_t image_width, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "LocateAnythingPipeline"; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg);

    // VL preprocessing config (public for testing).
    const LocateAnythingPreprocessConfig& vl_preprocess_config() const { return vl_preprocess_; }
    bool has_vision_encoder() const { return vision_encoder_ != nullptr; }

    static int32_t argmax(const std::vector<float>& logits);

  private:
    std::unique_ptr<TrtModule> text_decoder_;
    std::unique_ptr<TrtModule> prefill_;
    std::unique_ptr<TrtModule> vision_encoder_;
    std::unique_ptr<LocateanythingInferenceState> state_;
    double last_setup_ms_{0.0};
    LocateAnythingConfig config_;
    LocateAnythingPreprocessConfig vl_preprocess_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::unique_ptr<LocateAnythingISampler> sampler_;

    std::vector<int32_t> generate_from_ids(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const LocateAnythingSamplingParams& params);

    // VL generation with vision features injected at image token positions.
    std::vector<int32_t> generate_vl_from_ids(
        const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
        const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
        int32_t feature_dim, int32_t max_new_tokens, const LocateAnythingSamplingParams& params);

    std::pair<int32_t, int32_t> resolve_gen_limits(const GenerateConfig& cfg) const;

    void reset_generation_context(int32_t prompt_length);

    void run_vl_prefill_batched(const std::vector<int32_t>& input_ids,
                                const std::vector<float>& image_features,
                                const std::vector<std::vector<float>>& deepstack_features,
                                int32_t num_features, int32_t feature_dim,
                                std::vector<float>& logits);

    void run_vl_decode_loop(LocateAnythingISampler* sampler,
                            const LocateAnythingSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens);

    void run_text_step(int32_t token_id, std::vector<float>& logits);

    // Run a text step with optional vision embedding override.
    void run_text_step_with_embed(int32_t token_id, const float* input_embed, float use_input_embed,
                                  const std::vector<const float*>& deepstack_embeds,
                                  float deepstack_active, std::vector<float>& logits);

    // Run vision encoder on preprocessed image inputs.
    bool run_vision_encoder(const LocateAnythingPreprocessedImage& preprocessed,
                            std::vector<float>& image_features,
                            std::vector<std::vector<float>>* deepstack_features = nullptr);
};

} // namespace trtmc
